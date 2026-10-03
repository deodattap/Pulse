# ============================================================
# FastAPI server that exposes the existing ML pipeline and
# research evaluation results to the React frontend.
#
# Run from the Backend folder:
#   python -m uvicorn api_server:app --host 127.0.0.1 --port 8000
# ============================================================

from __future__ import annotations

import os
import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BACKEND_DIR, 'src')
for p in (BACKEND_DIR, SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from services.live_prediction import run_live_prediction, download_history
from services.research import (
    STOCK_META,
    POPULAR_STOCKS,
    feature_analysis,
    historical_prediction,
    load_streaming_results,
    model_stats,
    recent_predictions,
    regime_analysis,
    research_payload,
    research_stock_list,
)

app = FastAPI(title='PULSE API', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:8443',
        'http://127.0.0.1:8443',
        'http://localhost:5173',
        'http://127.0.0.1:5173',
        'http://localhost:4173',
        'http://127.0.0.1:4173',
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

INDEX_TICKERS = [
    ('^NSEI', 'NIFTY 50'),
    ('^BSESN', 'SENSEX'),
    ('^NSEBANK', 'NIFTY BANK'),
    ('^CNXIT', 'NIFTY IT'),
    ('^CNXAUTO', 'NIFTY AUTO'),
    ('^CNXPHARMA', 'NIFTY PHARMA'),
    ('^CNXFMCG', 'NIFTY FMCG'),
    ('^CNXMETAL', 'NIFTY METAL'),
    ('^NSEMDCP50', 'NIFTY MIDCAP'),
    ('^INDIAVIX', 'INDIA VIX'),
    ('INR=X', 'USD/INR'),
    ('GC=F', 'GOLD'),
]

_quote_cache = {'ts': 0.0, 'data': None}
_index_cache = {'ts': 0.0, 'data': None}
_pred_cache = {}
QUOTE_TTL = 300
PRED_TTL = 1800


class PredictRequest(BaseModel):
    ticker: str
    date: Optional[str] = None
    mode: Optional[str] = None  # 'live' | 'historical' | auto


def _fmt_volume(v: float) -> str:
    if v >= 10_000_000:
        return f'{v / 10_000_000:.1f}Cr'
    if v >= 100_000:
        return f'{v / 100_000:.1f}L'
    if v >= 1_000_000:
        return f'{v / 1_000_000:.1f}M'
    if v >= 1000:
        return f'{v / 1000:.1f}K'
    return f'{int(v)}'


def _stock_meta(symbol: str):
    name, sector = STOCK_META.get(symbol.upper(), (symbol.upper(), 'Equity'))
    return name, sector


def _ohlcv_frame(raw, ticker: str):
    import pandas as pd
    if raw is None or getattr(raw, 'empty', True):
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        level1 = set(raw.columns.get_level_values(1))
        if ticker in level0:
            df = raw[ticker]
        elif ticker in level1:
            df = raw.xs(ticker, axis=1, level=1)
        else:
            return None
    else:
        df = raw
    if 'Close' not in df.columns:
        return None
    return df.dropna(subset=['Close'])


def fetch_quotes(symbols: list[str]) -> list[dict]:
    now = time.time()
    if _quote_cache['data'] is not None and now - _quote_cache['ts'] < QUOTE_TTL:
        cached = {s['symbol']: s for s in _quote_cache['data']}
        hit = [cached[s] for s in symbols if s in cached]
        if len(hit) == len(symbols):
            return hit
        return _quote_cache['data']

    import yfinance as yf
    import pandas as pd

    tickers = [f'{s}.NS' for s in symbols]
    
    try:
        raw = yf.download(
            tickers,
            period='6mo',
            group_by='ticker',
            auto_adjust=True,
            threads=True,
            progress=False,
        )
    except Exception as e:
        # If yfinance fails completely, return empty list instead of fallback data
        return []

    out = []
    for sym in symbols:
        t = f'{sym}.NS'
        try:
            df = _ohlcv_frame(raw, t)
            if df is None or len(df) < 2:
                continue
            last = df.iloc[-1]
            prev = df.iloc[-2]
            close = float(last['Close'])
            prev_c = float(prev['Close'])
            change = close - prev_c
            pct = change / prev_c * 100 if prev_c else 0
            vol = float(last['Volume']) if 'Volume' in df.columns and pd.notna(last.get('Volume')) else 0
            high52 = float(df['High'].max()) if 'High' in df.columns else close
            low52 = float(df['Low'].min()) if 'Low' in df.columns else close
            name, sector = _stock_meta(sym)
            out.append({
                'symbol': sym,
                'name': name,
                'sector': sector,
                'price': round(close, 2),
                'change': round(change, 2),
                'pct': round(pct, 2),
                'up': change >= 0,
                'volume': _fmt_volume(vol),
                'volume_raw': int(vol),
                'open': round(float(last['Open']), 2) if 'Open' in df.columns else round(close, 2),
                'high': round(float(last['High']), 2) if 'High' in df.columns else round(close, 2),
                'low': round(float(last['Low']), 2) if 'Low' in df.columns else round(close, 2),
                'high52': round(high52, 2),
                'low52': round(low52, 2),
                'date': str(df.index[-1].date()) if hasattr(df.index[-1], 'date') else str(df.index[-1])[:10],
                'research': sym in research_stock_list(),
            })
        except Exception:
            continue

    if out:
        _quote_cache['ts'] = now
        _quote_cache['data'] = out
    return out


def fetch_indices() -> list[dict]:
    now = time.time()
    if _index_cache['data'] is not None and now - _index_cache['ts'] < QUOTE_TTL:
        return _index_cache['data']

    import yfinance as yf

    out = []
    tickers = [t for t, _ in INDEX_TICKERS]
    names = {t: n for t, n in INDEX_TICKERS}
    try:
        raw = yf.download(
            tickers, period='5d', group_by='ticker',
            auto_adjust=True, threads=True, progress=False,
        )
    except Exception:
        return []  # Return empty list instead of cached fallback data

    for t, name in INDEX_TICKERS:
        try:
            df = _ohlcv_frame(raw, t)
            if df is None or len(df) < 2:
                continue
            last = float(df['Close'].iloc[-1])
            prev = float(df['Close'].iloc[-2])
            change = last - prev
            pct = change / prev * 100 if prev else 0
            decimals = 2 if last < 1000 else 2
            out.append({
                'name': name,
                'ticker': t,
                'value': f'{last:,.{decimals}f}',
                'value_raw': round(last, 2),
                'change': f'{change:+.{decimals}f}',
                'pct': f'{pct:+.2f}',
                'up': change >= 0,
            })
        except Exception:
            continue

    if out:
        _index_cache['ts'] = now
        _index_cache['data'] = out
    return out


def nifty_history(days: int = 180):
    import yfinance as yf
    hist = yf.Ticker('^NSEI').history(period='1y', auto_adjust=True)
    if hist is None or hist.empty:
        return []
    hist = hist.tail(days).reset_index()
    rows = []
    for _, row in hist.iterrows():
        rows.append({
            'date': str(row['Date'].date()) if hasattr(row['Date'], 'date') else str(row['Date'])[:10],
            'value': round(float(row['Close']), 2),
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
            'volume': int(row['Volume']) if row['Volume'] == row['Volume'] else 0,
        })
    return rows


def price_history(symbol: str, period: str = '1y'):
    hist, _ = download_history(symbol, period=period)
    rows = []
    for _, row in hist.iterrows():
        rows.append({
            'date': str(row['Date'].date()) if hasattr(row['Date'], 'date') else str(row['Date'])[:10],
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
            'volume': int(row['Volume']),
        })
    return rows


def sector_performance(stocks: list[dict]) -> list[dict]:
    buckets = {}
    for s in stocks:
        sector = s.get('sector') or 'Other'
        buckets.setdefault(sector, []).append(s['pct'])
    out = []
    for name, pcts in buckets.items():
        out.append({
            'name': name,
            'change': round(sum(pcts) / len(pcts), 2),
            'stocks': len(pcts),
        })
    out.sort(key=lambda x: -abs(x['change']))
    return out


@app.get('/api/health')
def health():
    df = load_streaming_results()
    return {
        'ok': True,
        'research_rows': 0 if df is None else int(len(df)),
        'research_stocks': research_stock_list(),
    }


@app.get('/api/overview')
def overview():
    try:
        quotes = fetch_quotes(POPULAR_STOCKS)
    except Exception as e:
        quotes = []
        quote_error = str(e)
    else:
        quote_error = None

    try:
        indices = fetch_indices()
    except Exception:
        indices = []

    try:
        chart = nifty_history(180)
    except Exception:
        chart = []

    nifty = next((i for i in indices if i['name'] == 'NIFTY 50'), None)
    return {
        'indices': indices,
        'stocks': quotes,
        'sectors': sector_performance(quotes),
        'model_stats': model_stats(),
        'recent_predictions': recent_predictions(8),
        'nifty_history': chart,
        'nifty': nifty,
        'quote_error': quote_error,
        'research_stocks': research_stock_list(),
    }


@app.get('/api/stocks')
def stocks():
    quotes = fetch_quotes(POPULAR_STOCKS)
    return {
        'stocks': quotes,
        'research_stocks': research_stock_list(),
        'popular': POPULAR_STOCKS,
        'all_available': POPULAR_STOCKS,  # Add this for reference
    }


@app.get('/api/stocks/{symbol}/history')
def stock_history(symbol: str, range: str = Query('1y')):
    period_map = {'1M': '1mo', '3M': '3mo', '6M': '6mo', '1Y': '1y', '1y': '1y', '2y': '2y'}
    period = period_map.get(range, '1y')
    try:
        rows = price_history(symbol, period=period)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Live data unavailable: {str(e)}")
    
    if not rows or len(rows) == 0:
        raise HTTPException(status_code=404, detail="Live data unavailable for this stock")
    
    quotes = fetch_quotes(POPULAR_STOCKS)
    quote = next((q for q in quotes if q['symbol'].upper() == symbol.upper()), None)
    
    if quote is None:
        name, sector = _stock_meta(symbol)
        last = rows[-1] if rows else None
        if last is None:
            raise HTTPException(status_code=404, detail="Live data unavailable for this stock")
        
        quote = {
            'symbol': symbol.upper(),
            'name': name,
            'sector': sector,
            'price': last['close'] if last else 0,
            'change': 0,
            'pct': 0,
            'up': True,
            'volume': _fmt_volume(last['volume']) if last else '0',
            'volume_raw': last['volume'] if last else 0,
            'open': last['open'] if last else 0,
            'high': last['high'] if last else 0,
            'low': last['low'] if last else 0,
            'high52': max((r['high'] for r in rows), default=0),
            'low52': min((r['low'] for r in rows), default=0),
            'date': last['date'] if last else None,
            'research': symbol.upper() in research_stock_list(),
        }
    return {'quote': quote, 'history': rows}


@app.get('/api/indices')
def indices():
    return {'indices': fetch_indices()}


@app.get('/api/research')
def research():
    return research_payload()


@app.get('/api/features')
def features(stock: Optional[str] = None):
    data = feature_analysis(stock)
    if 'error' in data:
        raise HTTPException(status_code=404, detail=data['error'])
    return data


@app.get('/api/regime')
def regime():
    data = regime_analysis()
    if 'error' in data:
        raise HTTPException(status_code=404, detail=data['error'])
    return data


@app.get('/api/historical/{symbol}')
def historical(symbol: str, date: Optional[str] = None):
    try:
        return historical_prediction(symbol, date)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post('/api/predict')
def predict(body: PredictRequest):
    ticker = body.ticker.upper().strip().replace('.NS', '')
    research_set = set(research_stock_list())
    mode = (body.mode or 'auto').lower()

    use_historical = False
    if mode == 'historical':
        use_historical = True
    elif mode == 'live':
        use_historical = False
    elif body.date and ticker in research_set:
        use_historical = True

    if use_historical:
        try:
            hist = historical_prediction(ticker, body.date)
            return hist
        except ValueError as e:
            if mode == 'historical':
                raise HTTPException(status_code=404, detail=str(e))
            # fall through to live

    cache_key = ticker
    now = time.time()
    cached = _pred_cache.get(cache_key)
    if cached and now - cached['ts'] < PRED_TTL:
        return cached['data']

    try:
        result = run_live_prediction(ticker)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Prediction failed: {e}')

    result['mode'] = 'live'
    name, sector = _stock_meta(ticker)
    result['name'] = name
    result['sector'] = sector
    _pred_cache[cache_key] = {'ts': now, 'data': result}
    return result


if __name__ == '__main__':
    import uvicorn
    uvicorn.run('api_server:app', host='127.0.0.1', port=8000, reload=False)
