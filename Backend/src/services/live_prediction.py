# ============================================================
# Live prediction engine extracted from dashboard/app.py
# Streamlit UI removed so the same ML pipeline can be served
# over HTTP. Feature computation, XGBoost DFS, and SRPClassifier
# training match the existing dashboard logic.
# ============================================================

from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

CAT_MAP = {
    'SMA_10': 'Trend', 'SMA_20': 'Trend',
    'SMA_50': 'Trend', 'EMA_12': 'Trend',
    'EMA_26': 'Trend',
    'RSI_14': 'Momentum', 'MACD': 'Momentum',
    'MACD_Signal': 'Momentum',
    'MACD_Hist': 'Momentum', 'ROC': 'Momentum',
    'Williams_R': 'Momentum', 'Stochastic': 'Momentum',
    'ATR_14': 'Volatility', 'BB_Upper': 'Volatility',
    'BB_Lower': 'Volatility', 'BB_Width': 'Volatility',
    'OBV': 'Volume', 'Volume_MA': 'Volume',
    'Volume_Ratio': 'Volume',
    'CCI': 'Statistical', 'ADX': 'Statistical',
    'MFI': 'Statistical',
    'Daily_Return': 'Returns',
    'Log_Return': 'Returns',
    'Return_5d': 'Returns',
    'Return_10d': 'Returns',
    'Rolling_Std_10': 'Returns',
    'Rolling_Volatility': 'Returns',
    'High_Low_Pct': 'Price Pattern',
    'Open_Close_Pct': 'Price Pattern',
    'Price_Change': 'Price Pattern',
    'Rolling_Mean_Return': 'Returns',
    'Rolling_Median_Return': 'Returns',
}

FEAT_DESC = {
    'RSI_14': 'RSI - measures momentum strength and overbought/oversold conditions',
    'MACD': 'MACD - captures trend direction and momentum crossovers',
    'MACD_Hist': 'MACD Histogram - shows momentum shift strength',
    'MACD_Signal': 'MACD Signal - smoothed MACD for crossover detection',
    'ATR_14': 'ATR - measures market volatility and price range',
    'BB_Width': 'Bollinger Width - indicates volatility expansion or contraction',
    'BB_Upper': 'Bollinger Upper Band - price resistance level',
    'BB_Lower': 'Bollinger Lower Band - price support level',
    'OBV': 'OBV - tracks volume flow relative to price direction',
    'Volume_Ratio': 'Volume Ratio - compares current to average trading volume',
    'Volume_MA': 'Volume MA - smoothed volume trend',
    'SMA_10': 'SMA-10 - short-term price trend average',
    'SMA_20': 'SMA-20 - medium-short price trend average',
    'SMA_50': 'SMA-50 - medium-term price trend average',
    'EMA_12': 'EMA-12 - fast exponential trend indicator',
    'EMA_26': 'EMA-26 - slow exponential trend indicator',
    'ROC': 'Rate of Change - measures price momentum over 10 days',
    'CCI': 'CCI - identifies cyclical turns in price',
    'ADX': 'ADX - measures trend strength regardless of direction',
    'MFI': 'MFI - money flow pressure using price and volume',
    'Williams_R': 'Williams %R - overbought and oversold oscillator',
    'Stochastic': 'Stochastic - compares closing price to recent range',
    'Rolling_Volatility': 'Rolling Volatility - recent price instability measure',
    'Daily_Return': 'Daily Return - most recent single-day price change',
    'Return_5d': '5-Day Return - short-term cumulative return',
    'Return_10d': '10-Day Return - medium-short cumulative return',
    'Rolling_Std_10': 'Rolling Std-10 - price variability over 10 days',
    'Rolling_Mean_Return': 'Rolling Mean Return - average return persistence',
    'Rolling_Median_Return': 'Rolling Median Return - robust return trend estimate',
    'High_Low_Pct': 'High-Low % - daily price range relative to close',
    'Open_Close_Pct': 'Open-Close % - intraday directional pressure',
    'Price_Change': 'Price Change - absolute daily price movement',
    'Log_Return': 'Log Return - logarithmic daily return for stability',
}

REGIME_MSGS = {
    'Trend': 'Trend indicators are dominant, suggesting a clear directional price movement is underway.',
    'Momentum': 'Momentum indicators are showing strong directional pressure in the market.',
    'Volatility': 'Volatility indicators are the primary signal, suggesting elevated market uncertainty.',
    'Volume': 'Volume-based indicators dominate, indicating strong market participation confirming the move.',
    'Returns': 'Recent return patterns are the strongest predictors, suggesting momentum continuation.',
    'Statistical': 'Statistical indicators show a measurable pattern in current market conditions.',
    'Price Pattern': 'Price pattern features are dominant, indicating significant intraday movement patterns.',
}

FEAT_COLS = [
    'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
    'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Hist',
    'ROC', 'Williams_R', 'Stochastic',
    'ATR_14', 'BB_Upper', 'BB_Lower', 'BB_Width',
    'OBV', 'Volume_MA', 'Volume_Ratio',
    'CCI', 'ADX', 'Daily_Return', 'Log_Return',
    'Return_5d', 'Return_10d', 'Rolling_Std_10',
    'Rolling_Volatility', 'High_Low_Pct',
    'Open_Close_Pct', 'Price_Change', 'MFI',
    'Rolling_Mean_Return', 'Rolling_Median_Return',
]


def generate_explanation(selected_features, prediction, confidence,
                         stock_name, indicators):
    cats = Counter(CAT_MAP.get(f, 'Other') for f in selected_features)
    dominant_cat, cat_count = cats.most_common(1)[0]
    top3 = selected_features[:3]
    top3_lines = [
        {'feature': f, 'description': FEAT_DESC.get(f, f)}
        for f in top3
    ]

    insights = []
    if 'RSI_14' in selected_features and 'RSI_14' in indicators:
        rsi_val = indicators['RSI_14']
        if rsi_val > 70:
            insights.append(f'RSI is at {rsi_val:.1f} - overbought conditions.')
        elif rsi_val < 30:
            insights.append(f'RSI is at {rsi_val:.1f} - oversold conditions.')
        else:
            insights.append(f'RSI is at {rsi_val:.1f} - in neutral territory.')

    if 'MACD' in selected_features and 'MACD' in indicators:
        macd_val = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        if macd_val > macd_signal:
            insights.append(
                f'MACD ({macd_val:.3f}) is above Signal ({macd_signal:.3f}) - bullish crossover.'
            )
        else:
            insights.append(
                f'MACD ({macd_val:.3f}) is below Signal ({macd_signal:.3f}) - bearish crossover.'
            )

    if 'ATR_14' in selected_features and 'ATR_14' in indicators:
        insights.append(
            f'ATR at {indicators["ATR_14"]:.2f} reflects current market volatility.'
        )

    if 'BB_Width' in selected_features and 'BB_Width' in indicators:
        bb_val = indicators['BB_Width']
        if bb_val > 0.1:
            insights.append(
                f'Bollinger Width at {bb_val:.3f} indicates high volatility and potential breakout.'
            )
        else:
            insights.append(
                f'Bollinger Width at {bb_val:.3f} indicates low volatility and consolidation.'
            )

    if 'Volume_Ratio' in selected_features and 'Volume_Ratio' in indicators:
        vr = indicators['Volume_Ratio']
        if vr > 1.5:
            insights.append(
                f'Volume Ratio at {vr:.2f}x shows above-average buying activity.'
            )
        elif vr < 0.7:
            insights.append(
                f'Volume Ratio at {vr:.2f}x shows below-average activity, low conviction.'
            )

    signal_word = 'positive' if prediction == 1 else 'negative'
    action = 'BUY' if prediction == 1 else 'SELL'
    if not insights:
        insights.append(
            f'The top features - {", ".join(top3)} - collectively indicate a {signal_word} directional bias.'
        )

    text = (
        f'The Dynamic Feature Selection system analysed the last 252 trading days of {stock_name}. '
        f'From 33 available technical indicators, XGBoost importance ranking identified '
        f'{len(selected_features)} features as most relevant. Dominant category: {dominant_cat} '
        f'({cat_count} of top 10). {REGIME_MSGS.get(dominant_cat, "")} '
        f'The SRPClassifier predicted a {signal_word} 20-day forward return direction '
        f'with {confidence * 100:.1f}% confidence, generating a {action} signal.'
    )

    return {
        'action': action,
        'dominant_category': dominant_cat,
        'category_count': int(cat_count),
        'top_features': top3_lines,
        'insights': insights,
        'text': text,
        'regime_description': REGIME_MSGS.get(dominant_cat, ''),
    }


def _compute_indicators(hist: pd.DataFrame) -> pd.DataFrame:
    import pandas_ta as ta

    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    volume = hist['Volume']

    hist['SMA_10'] = ta.sma(close, length=10)
    hist['SMA_20'] = ta.sma(close, length=20)
    hist['SMA_50'] = ta.sma(close, length=50)
    hist['EMA_12'] = ta.ema(close, length=12)
    hist['EMA_26'] = ta.ema(close, length=26)
    hist['RSI_14'] = ta.rsi(close, length=14)

    macd_df = ta.macd(close)
    if macd_df is not None:
        hist['MACD'] = macd_df.iloc[:, 0]
        hist['MACD_Signal'] = macd_df.iloc[:, 1]
        hist['MACD_Hist'] = macd_df.iloc[:, 2]
    else:
        hist['MACD'] = hist['MACD_Signal'] = hist['MACD_Hist'] = 0

    hist['ROC'] = ta.roc(close, length=10)
    hist['Williams_R'] = ta.willr(high, low, close, length=14)

    stoch = ta.stoch(high, low, close)
    hist['Stochastic'] = stoch.iloc[:, 0] if stoch is not None else 50

    hist['ATR_14'] = ta.atr(high, low, close, length=14)

    bb = ta.bbands(close, length=20)
    if bb is not None:
        hist['BB_Upper'] = bb.iloc[:, 0]
        hist['BB_Lower'] = bb.iloc[:, 1]
        hist['BB_Width'] = bb.iloc[:, 3]
    else:
        hist['BB_Upper'] = hist['BB_Lower'] = close
        hist['BB_Width'] = 0

    hist['OBV'] = ta.obv(close, volume)
    hist['Volume_MA'] = volume.rolling(20).mean()
    hist['Volume_Ratio'] = volume / hist['Volume_MA'].replace(0, 1)
    hist['CCI'] = ta.cci(high, low, close, length=14)

    adx_df = ta.adx(high, low, close, length=14)
    hist['ADX'] = adx_df.iloc[:, 0] if adx_df is not None else 25

    hist['Daily_Return'] = close.pct_change()
    hist['Log_Return'] = np.log(close / close.shift(1))
    hist['Return_5d'] = close.pct_change(5)
    hist['Return_10d'] = close.pct_change(10)
    hist['Rolling_Std_10'] = hist['Daily_Return'].rolling(10).std()
    hist['Rolling_Volatility'] = hist['Daily_Return'].rolling(20).std()
    hist['High_Low_Pct'] = (high - low) / close
    hist['Open_Close_Pct'] = (hist['Open'] - close) / close
    hist['Price_Change'] = close.diff()

    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0)
    neg = mf.where(tp <= tp.shift(1), 0)
    hist['MFI'] = 100 - 100 / (
        1 + pos.rolling(14).sum() / neg.rolling(14).sum().replace(0, 1)
    )

    hist['Rolling_Mean_Return'] = hist['Daily_Return'].rolling(10).mean()
    hist['Rolling_Median_Return'] = hist['Daily_Return'].rolling(10).median()
    hist['Binary_Target'] = (close.shift(-20) > close).astype(int)
    return hist.dropna().reset_index(drop=True)


def _select_features(hist: pd.DataFrame, feat_cols):
    from xgboost import XGBClassifier

    window_data = hist.tail(252)
    X_mi = window_data[feat_cols].values
    y_mi = window_data['Binary_Target'].values.astype(int)

    selected_features = feat_cols[:10]
    feature_importances = {f: 0.0 for f in feat_cols}

    try:
        counts = Counter(y_mi)
        total = len(y_mi)
        n = len(counts)
        wmap = {c: total / (n * cnt) for c, cnt in counts.items()}
        sw = np.array([wmap[i] for i in y_mi])

        xgb_fs = XGBClassifier(
            n_estimators=100,
            random_state=42,
            verbosity=0,
            eval_metric='logloss',
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
        )
        xgb_fs.fit(X_mi, y_mi, sample_weight=sw)
        importances = xgb_fs.feature_importances_
        scored = sorted(zip(feat_cols, importances), key=lambda x: x[1], reverse=True)
        selected_features = [f for f, _ in scored[:10]]
        feature_importances = {f: float(s) for f, s in scored}
    except Exception:
        pass

    return selected_features, feature_importances


def _train_srp(hist: pd.DataFrame, selected_features):
    from river import tree, ensemble, preprocessing

    srp_model = (
        preprocessing.StandardScaler()
        | ensemble.SRPClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=10,
            seed=42,
        )
    )

    train_data = hist.iloc[:-1]
    for _, row in train_data.iterrows():
        x = {f: float(row[f]) for f in selected_features}
        y = int(row['Binary_Target'])
        srp_model.learn_one(x, y)

    latest_row = hist.iloc[-1]
    x_pred = {f: float(latest_row[f]) for f in selected_features}
    pred = srp_model.predict_one(x_pred)
    proba = srp_model.predict_proba_one(x_pred)

    if proba and len(proba) > 0:
        confidence = float(max(proba.values()))
        pred_class = int(max(proba, key=proba.get))
    else:
        confidence = 0.5
        pred_class = int(pred) if pred is not None else 1

    return pred_class, confidence


def download_history(ticker: str, period: str = '2y') -> pd.DataFrame:
    import yfinance as yf

    ticker_sym = ticker.upper().strip()
    if not ticker_sym.endswith('.NS'):
        ticker_sym += '.NS'

    tkr = yf.Ticker(ticker_sym)
    hist = tkr.history(period=period, auto_adjust=True)
    if hist is None or len(hist) < 30:
        raise ValueError(
            f'Not enough data for {ticker}. Check the NSE ticker (e.g. RELIANCE, TCS, INFY).'
        )

    hist = hist.reset_index()
    hist['Date'] = pd.to_datetime(hist['Date']).dt.tz_localize(None)
    return hist.sort_values('Date').reset_index(drop=True), ticker_sym


def run_live_prediction(ticker: str) -> dict:
    hist, ticker_sym = download_history(ticker, period='2y')
    if len(hist) < 100:
        raise ValueError(f'Not enough data for {ticker}.')

    hist = _compute_indicators(hist)
    if len(hist) < 60:
        raise ValueError('Not enough clean data after indicator computation.')

    feat_cols = [f for f in FEAT_COLS if f in hist.columns]
    selected_features, feature_importances = _select_features(hist, feat_cols)
    pred_class, confidence = _train_srp(hist, selected_features)

    latest = hist.iloc[-1]
    prev_close = float(hist['Close'].iloc[-2])
    close = float(latest['Close'])
    change = close - prev_close
    pct = (change / prev_close) * 100 if prev_close else 0.0

    high52 = float(hist['High'].max())
    low52 = float(hist['Low'].min())

    price_history = hist[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(250)
    history_rows = []
    for _, row in price_history.iterrows():
        history_rows.append({
            'date': pd.Timestamp(row['Date']).strftime('%Y-%m-%d'),
            'open': round(float(row['Open']), 2),
            'high': round(float(row['High']), 2),
            'low': round(float(row['Low']), 2),
            'close': round(float(row['Close']), 2),
            'volume': int(row['Volume']),
        })

    indicators = {
        f: float(hist[f].iloc[-1])
        for f in feat_cols
        if f in hist.columns and pd.notna(hist[f].iloc[-1])
    }

    explanation = generate_explanation(
        selected_features, pred_class, confidence,
        ticker.upper().strip(), indicators,
    )

    cats = Counter(CAT_MAP.get(f, 'Other') for f in selected_features)
    technicals = []
    for f in feat_cols:
        if f not in indicators:
            continue
        val = indicators[f]
        selected = f in selected_features
        signal = _indicator_signal(f, val, indicators)
        technicals.append({
            'name': f,
            'value': round(val, 4),
            'category': CAT_MAP.get(f, 'Other'),
            'description': FEAT_DESC.get(f, f),
            'selected': selected,
            'signal': signal,
        })

    selected_detailed = []
    max_imp = max((feature_importances.get(f, 0) for f in selected_features), default=1) or 1
    for i, f in enumerate(selected_features, 1):
        imp = float(feature_importances.get(f, 0))
        selected_detailed.append({
            'rank': i,
            'feature': f,
            'importance': imp,
            'category': CAT_MAP.get(f, 'Other'),
            'description': FEAT_DESC.get(f, f),
        })

    return {
        'ticker': ticker_sym,
        'stock_name': ticker.upper().strip(),
        'latest_date': pd.Timestamp(latest['Date']).strftime('%Y-%m-%d'),
        'open': round(float(latest['Open']), 2),
        'high': round(float(latest['High']), 2),
        'low': round(float(latest['Low']), 2),
        'close': round(close, 2),
        'volume': int(latest['Volume']),
        'prev_close': round(prev_close, 2),
        'change': round(change, 2),
        'pct': round(pct, 2),
        'up': change >= 0,
        'high52': round(high52, 2),
        'low52': round(low52, 2),
        'prediction': int(pred_class),
        'signal': 'BUY' if pred_class == 1 else 'SELL',
        'confidence': round(confidence, 4),
        'selected_features': selected_features,
        'selected_detailed': selected_detailed,
        'feature_importances': {k: float(v) for k, v in feature_importances.items()},
        'price_history': history_rows,
        'indicators': indicators,
        'technicals': technicals,
        'all_features': feat_cols,
        'n_rows_used': int(len(hist)),
        'explanation': explanation,
        'dominant_category': explanation['dominant_category'],
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'max_importance': float(max_imp),
        'category_distribution': [
            {'name': k, 'count': int(v)} for k, v in cats.most_common()
        ],
    }


def _indicator_signal(name: str, val: float, indicators: dict) -> str:
    if name == 'RSI_14':
        if val > 70:
            return 'Overbought'
        if val < 30:
            return 'Oversold'
        return 'Neutral'
    if name == 'MACD':
        sig = indicators.get('MACD_Signal', 0)
        return 'Bullish' if val > sig else 'Bearish'
    if name == 'ADX':
        return 'Trending' if val >= 25 else 'Weak trend'
    if name == 'Volume_Ratio':
        if val > 1.5:
            return 'High volume'
        if val < 0.7:
            return 'Low volume'
        return 'Average'
    if name in ('SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26'):
        return 'Level'
    if name == 'Williams_R':
        if val > -20:
            return 'Overbought'
        if val < -80:
            return 'Oversold'
        return 'Neutral'
    if name == 'Stochastic':
        if val > 80:
            return 'Overbought'
        if val < 20:
            return 'Oversold'
        return 'Neutral'
    return '—'
