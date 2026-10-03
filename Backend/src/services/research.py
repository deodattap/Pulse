# ============================================================
# Load research evaluation CSVs and aggregate them for the UI.
# ============================================================

from collections import Counter
from functools import lru_cache
from typing import Optional
import os
import sys

import pandas as pd

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
try:
    from config.config import LOGS_DIR, METRICS_DIR
except ImportError:
    from config import LOGS_DIR, METRICS_DIR

from services.live_prediction import CAT_MAP, FEAT_DESC

STOCK_META = {
    'RELIANCE': ('Reliance Industries Ltd', 'Energy'),
    'TCS': ('Tata Consultancy Services', 'IT'),
    'HDFCBANK': ('HDFC Bank Ltd', 'Banking'),
    'INFY': ('Infosys Ltd', 'IT'),
    'ICICIBANK': ('ICICI Bank Ltd', 'Banking'),
    'WIPRO': ('Wipro Ltd', 'IT'),
    'HINDUNILVR': ('Hindustan Unilever Ltd', 'FMCG'),
    'ITC': ('ITC Ltd', 'FMCG'),
    'BAJFINANCE': ('Bajaj Finance Ltd', 'NBFC'),
    'KOTAKBANK': ('Kotak Mahindra Bank', 'Banking'),
    'AXISBANK': ('Axis Bank Ltd', 'Banking'),
    'MARUTI': ('Maruti Suzuki India', 'Auto'),
    'SUNPHARMA': ('Sun Pharmaceutical', 'Pharma'),
    'TITAN': ('Titan Company Ltd', 'Consumer'),
    'NESTLEIND': ('Nestle India Ltd', 'FMCG'),
    'POWERGRID': ('Power Grid Corporation', 'Energy'),
    'SBIN': ('State Bank of India', 'Banking'),
    'HCLTECH': ('HCL Technologies', 'IT'),
    'NTPC': ('NTPC Ltd', 'Energy'),
    'ONGC': ('Oil and Natural Gas Corp', 'Energy'),
}

POPULAR_STOCKS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK',
    'ICICIBANK', 'KOTAKBANK', 'HINDUNILVR',
    'POWERGRID', 'BAJFINANCE', 'WIPRO',
    'HCLTECH', 'AXISBANK', 'SBIN', 'MARUTI',
    'TITAN', 'NESTLEIND', 'SUNPHARMA', 'ITC',
]


@lru_cache(maxsize=1)
def load_streaming_results():
    path = os.path.join(LOGS_DIR, 'streaming_ALL.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['predict_date'])
    return df


@lru_cache(maxsize=1)
def load_streaming_metrics():
    path = os.path.join(METRICS_DIR, 'streaming_results.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def load_financial_metrics():
    path = os.path.join(METRICS_DIR, 'financial_metrics.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def research_stock_list():
    df = load_streaming_results()
    if df is None:
        return []
    return sorted(df['stock'].unique().tolist())


def model_stats():
    df = load_streaming_results()
    metrics = load_streaming_metrics()
    if df is None:
        return {
            'totalStocks': 0,
            'avgAccuracy': 0,
            'totalPredictions': 0,
            'activeDays': 0,
            'f1Score': 0,
            'buySignals': 0,
            'sellSignals': 0,
            'lastUpdated': None,
        }

    buy = int((df['dynamic_signal'] == 'BUY').sum())
    sell = int((df['dynamic_signal'] == 'SELL').sum())
    no_trade = int((df['dynamic_signal'] == 'NO TRADE').sum())
    avg_acc = float(df['dynamic_correct'].mean() * 100)
    f1 = 0.0
    if metrics is not None and len(metrics):
        f1 = float(metrics['dynamic_f1'].mean() / 100.0)

    last = pd.to_datetime(df['predict_date']).max()
    return {
        'totalStocks': int(df['stock'].nunique()),
        'avgAccuracy': round(avg_acc, 1),
        'totalPredictions': int(len(df)),
        'activeDays': int(df['predict_date'].nunique()),
        'f1Score': round(f1, 3),
        'buySignals': buy,
        'sellSignals': sell,
        'noTradeSignals': no_trade,
        'lastUpdated': last.strftime('%d %b %Y') if pd.notna(last) else None,
    }


def research_payload():
    metrics = load_streaming_metrics()
    df = load_streaming_results()
    fin = load_financial_metrics()

    per_stock = []
    if metrics is not None:
        for _, row in metrics.iterrows():
            per_stock.append({
                'symbol': row['stock'],
                'n_predictions': int(row['n_predictions']),
                'srp_acc': float(row['static_accuracy']),
                'pulse_acc': float(row['dynamic_accuracy']),
                'diff': float(row['accuracy_improvement']),
                'srp_f1': float(row['static_f1']) / 100.0,
                'pulse_f1': float(row['dynamic_f1']) / 100.0,
                'srp_precision': float(row['static_precision']),
                'pulse_precision': float(row['dynamic_precision']),
                'srp_recall': float(row['static_recall']),
                'pulse_recall': float(row['dynamic_recall']),
            })

    baseline = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
    pulse = {'accuracy': 0, 'f1': 0, 'precision': 0, 'recall': 0}
    if metrics is not None and len(metrics):
        baseline = {
            'accuracy': round(float(metrics['static_accuracy'].mean()), 2),
            'f1': round(float(metrics['static_f1'].mean()) / 100.0, 4),
            'precision': round(float(metrics['static_precision'].mean()) / 100.0, 4),
            'recall': round(float(metrics['static_recall'].mean()) / 100.0, 4),
        }
        pulse = {
            'accuracy': round(float(metrics['dynamic_accuracy'].mean()), 2),
            'f1': round(float(metrics['dynamic_f1'].mean()) / 100.0, 4),
            'precision': round(float(metrics['dynamic_precision'].mean()) / 100.0, 4),
            'recall': round(float(metrics['dynamic_recall'].mean()) / 100.0, 4),
        }

    improvement = {
        'accuracy': round(pulse['accuracy'] - baseline['accuracy'], 2),
        'f1': round(pulse['f1'] - baseline['f1'], 4),
        'precision': round(pulse['precision'] - baseline['precision'], 4),
        'recall': round(pulse['recall'] - baseline['recall'], 4),
    }

    yearly = []
    if df is not None:
        grouped = df.groupby('year').agg(
            static_acc=('static_correct', 'mean'),
            dynamic_acc=('dynamic_correct', 'mean'),
        ).reset_index()
        for _, row in grouped.iterrows():
            s = float(row['static_acc'] * 100)
            d = float(row['dynamic_acc'] * 100)
            yearly.append({
                'year': str(int(row['year'])),
                'srp_acc': round(s, 1),
                'pulse_acc': round(d, 1),
                'diff': round(d - s, 1),
            })

    financial = []
    if fin is not None:
        for _, row in fin.iterrows():
            financial.append({
                'metric': row.iloc[0],
                'static': str(row.iloc[1]),
                'dynamic': str(row.iloc[2]) if len(row) > 2 else '',
            })

    journey = [
        {'algorithm': 'XGBoost (Batch)', 'accuracy': 57.42,
         'note': 'Batch learning — cannot adapt to streams'},
        {'algorithm': 'Hoeffding Tree (Streaming)', 'accuracy': 62.99,
         'note': 'Single tree — misses complex patterns'},
        {'algorithm': 'KNN (Streaming)', 'accuracy': 70.95,
         'note': 'No explicit drift handling — not scalable'},
        {'algorithm': 'SRP Static (Streaming)', 'accuracy': baseline['accuracy'],
         'note': 'Best streaming algorithm — no dynamic FS'},
        {'algorithm': 'SRP + Dynamic FS (Ours)', 'accuracy': pulse['accuracy'],
         'note': 'FINAL: Best algorithm + dynamic features'},
    ]

    return {
        'baseline_srp': baseline,
        'pulse_dynamic': pulse,
        'improvement': improvement,
        'per_stock': per_stock,
        'yearly': yearly,
        'financial': financial,
        'journey': journey,
        'stats': model_stats(),
    }


def feature_analysis(stock: Optional[str] = None):
    df = load_streaming_results()
    if df is None:
        return {'error': 'Historical results not found'}

    source = df
    if stock:
        source = df[df['stock'].str.upper() == stock.upper()]
        if len(source) == 0:
            return {'error': f'No feature logs for {stock}'}

    all_feats = []
    for fs in source['dynamic_features'].dropna():
        all_feats.extend([f.strip() for f in str(fs).split(',') if f.strip()])

    freq = Counter(all_feats)
    n_windows = max(len(source), 1)
    ranking = []
    for i, (feat, count) in enumerate(freq.most_common(20), 1):
        ranking.append({
            'rank': i,
            'feature': feat,
            'frequency': round(count / n_windows * 100, 1),
            'count': int(count),
            'category': CAT_MAP.get(feat, 'Other'),
            'description': FEAT_DESC.get(feat, feat),
            'importance': round(count / n_windows, 4),
        })

    cat_counts = Counter()
    for feat, count in freq.items():
        cat_counts[CAT_MAP.get(feat, 'Other')] += count
    categories = [{'name': k, 'count': int(v)} for k, v in cat_counts.most_common()]

    monthly = []
    if 'top_mi_feature' in source.columns:
        tmp = source.copy()
        tmp['month'] = pd.to_datetime(tmp['predict_date']).dt.to_period('M').astype(str)
        top_monthly = tmp.groupby('month')['top_mi_feature'].agg(
            lambda x: x.value_counts().index[0] if len(x) else 'N/A'
        )
        monthly = [{'month': m, 'top_feature': f} for m, f in top_monthly.items()]

        # selection rate over time for the top 5 features
        top5 = [r['feature'] for r in ranking[:5]]
        time_series = []
        for month, group in tmp.groupby('month'):
            row = {'month': month}
            month_feats = []
            for fs in group['dynamic_features'].dropna():
                month_feats.extend([f.strip() for f in str(fs).split(',')])
            mc = Counter(month_feats)
            n = max(len(group), 1)
            for f in top5:
                row[f] = round(mc.get(f, 0) / n, 3)
            time_series.append(row)
    else:
        time_series = []
        top5 = []

    latest_row = source.sort_values('predict_date').iloc[-1]
    latest_feats = []
    if pd.notna(latest_row.get('dynamic_features')):
        latest_feats = [f.strip() for f in str(latest_row['dynamic_features']).split(',') if f.strip()]

    return {
        'stock': stock.upper() if stock else 'ALL',
        'total_predictions': int(len(source)),
        'unique_features': int(len(freq)),
        'avg_features_per_window': round(float(source['n_dynamic_features'].mean()), 1)
        if 'n_dynamic_features' in source.columns else 10,
        'ranking': ranking,
        'categories': categories,
        'monthly_top': monthly,
        'time_series': time_series,
        'time_series_keys': top5,
        'latest_selected': latest_feats,
        'latest_date': pd.Timestamp(latest_row['predict_date']).strftime('%Y-%m-%d'),
        'latest_top_feature': latest_row.get('top_mi_feature', None),
        'research_stocks': research_stock_list(),
    }


def regime_analysis():
    df = load_streaming_results()
    if df is None:
        return {'error': 'Historical results not found'}

    tmp = df.copy()
    tmp['period'] = pd.to_datetime(tmp['predict_date']).dt.to_period('Q').astype(str)

    rows = []
    for (period, regime), group in tmp.groupby(['period', 'market_regime']):
        rows.append({
            'period': str(period),
            'regime': str(regime),
            'accuracy': round(float(group['dynamic_correct'].mean() * 100), 1),
            'trades': int(len(group)),
        })
    rows.sort(key=lambda r: r['period'])

    # one dominant regime per quarter for the table
    quarterly = []
    for period, group in tmp.groupby('period'):
        regime = group['market_regime'].value_counts().index[0]
        quarterly.append({
            'period': str(period),
            'regime': str(regime),
            'accuracy': round(float(group['dynamic_correct'].mean() * 100), 1),
            'trades': int(len(group)),
        })
    quarterly.sort(key=lambda r: r['period'])

    summary = []
    for regime, group in tmp.groupby('market_regime'):
        summary.append({
            'regime': str(regime),
            'count': int(len(group)),
            'accuracy': round(float(group['dynamic_correct'].mean() * 100), 1),
        })
    summary.sort(key=lambda r: -r['count'])

    latest = tmp.sort_values('predict_date').iloc[-1]
    current = str(latest['market_regime'])

    return {
        'quarters': quarterly,
        'breakdown': rows,
        'summary': summary,
        'current': current,
        'latest_date': pd.Timestamp(latest['predict_date']).strftime('%Y-%m-%d'),
    }


def historical_prediction(stock: str, date: Optional[str] = None):
    df = load_streaming_results()
    if df is None:
        raise ValueError('Historical results not found. Run streaming_model.py first.')

    stock_data = df[df['stock'].str.upper() == stock.upper()].sort_values('predict_date')
    if len(stock_data) == 0:
        raise ValueError(f'{stock} is not in the research set. Use live prediction instead.')

    dates = pd.to_datetime(stock_data['predict_date']).dt.strftime('%Y-%m-%d').tolist()
    if date:
        match = stock_data[pd.to_datetime(stock_data['predict_date']).dt.strftime('%Y-%m-%d') == date]
        if len(match) == 0:
            # nearest previous date
            before = stock_data[pd.to_datetime(stock_data['predict_date']) <= pd.Timestamp(date)]
            if len(before) == 0:
                match = stock_data.iloc[[-1]]
            else:
                match = before.iloc[[-1]]
        row = match.iloc[0]
    else:
        row = stock_data.iloc[-1]

    pred_val = int(row['dynamic_pred'])
    conf_val = float(row['dynamic_confidence'])
    actual_val = int(row['actual_target'])
    correct = int(row['dynamic_correct'])
    signal = str(row['dynamic_signal'])
    feats = []
    if pd.notna(row.get('dynamic_features')):
        feats = [f.strip() for f in str(row['dynamic_features']).split(',') if f.strip()]

    recent = []
    sel_date = pd.to_datetime(row['predict_date'])
    hist_view = stock_data[pd.to_datetime(stock_data['predict_date']) <= sel_date].tail(10)
    for _, r in hist_view.iloc[::-1].iterrows():
        recent.append({
            'date': pd.Timestamp(r['predict_date']).strftime('%Y-%m-%d'),
            'signal': str(r['dynamic_signal']),
            'confidence': round(float(r['dynamic_confidence']), 4),
            'actual': 'Up' if int(r['actual_target']) == 1 else 'Down',
            'correct': bool(int(r['dynamic_correct'])),
            'regime': str(r['market_regime']),
        })

    selected_detailed = [
        {
            'rank': i,
            'feature': f,
            'importance': round(1.0 / i, 4),
            'category': CAT_MAP.get(f, 'Other'),
            'description': FEAT_DESC.get(f, f),
        }
        for i, f in enumerate(feats, 1)
    ]

    return {
        'mode': 'historical',
        'stock': stock.upper(),
        'date': pd.Timestamp(row['predict_date']).strftime('%Y-%m-%d'),
        'available_dates': dates,
        'prediction': pred_val,
        'signal': signal,
        'confidence': round(conf_val, 4),
        'actual_target': actual_val,
        'actual_label': 'Up' if actual_val == 1 else 'Down',
        'correct': bool(correct),
        'regime': str(row['market_regime']),
        'selected_features': feats,
        'selected_detailed': selected_detailed,
        'top_feature': row.get('top_mi_feature', None),
        'recent': recent,
        'static_signal': str(row.get('static_signal', '')),
        'static_correct': bool(int(row['static_correct'])) if pd.notna(row.get('static_correct')) else None,
    }


def recent_predictions(limit: int = 8):
    df = load_streaming_results()
    if df is None:
        return []
    latest = (
        df.sort_values('predict_date')
        .groupby('stock', as_index=False)
        .tail(1)
        .sort_values('predict_date', ascending=False)
    )
    out = []
    for _, row in latest.head(limit).iterrows():
        out.append({
            'symbol': row['stock'],
            'date': pd.Timestamp(row['predict_date']).strftime('%Y-%m-%d'),
            'signal': str(row['dynamic_signal']),
            'confidence': round(float(row['dynamic_confidence']), 4),
            'regime': str(row['market_regime']),
        })
    return out
