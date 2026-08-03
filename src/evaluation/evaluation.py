# ============================================================
# src/evaluation/evaluation.py
#
# Loads walk-forward results and generates:
# 1. Overall comparison table
# 2. Year-by-year accuracy graph
# 3. Regime-wise accuracy graph
# 4. Feature frequency graph
# 5. Prediction distribution graph
# 6. Portfolio simulation + financial metrics
#
# How to run:
#   python src/evaluation/evaluation.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *

# ── Style ──────────────────────────────────────────────────
plt.rcParams['figure.dpi']      = 150
plt.rcParams['font.size']       = 11
plt.rcParams['axes.titlesize']  = 13
plt.rcParams['axes.labelsize']  = 11
COLORS = {
    'static' : '#2196F3',
    'dynamic': '#4CAF50',
    'neutral': '#9E9E9E',
    'buy'    : '#4CAF50',
    'sell'   : '#F44336',
    'notrade': '#9E9E9E',
}


# ============================================================
# LOAD DATA
# ============================================================

def load_results():
    print("="*55)
    print("  LOADING RESULTS")
    print("="*55)

    results_path = os.path.join(
        LOGS_DIR, 'streaming_ALL.csv'
    )
    metrics_path = os.path.join(
        METRICS_DIR, 'streaming_results.csv'
    )

    if not os.path.exists(results_path):
        print(f"  ❌ nse_walkforward_ALL.csv not found")
        print(f"  Run nse_walkforward.py --stock ALL first")
        sys.exit(1)

    df = pd.read_csv(
        results_path, parse_dates=['predict_date']
    )
    metrics_df = pd.read_csv(metrics_path)

    print(f"  ✅ Results loaded: {len(df):,} predictions")
    print(f"  ✅ Stocks: {df['stock'].nunique()}")
    print(f"  ✅ Date range: "
          f"{df['predict_date'].min().date()} → "
          f"{df['predict_date'].max().date()}")

    return df, metrics_df


# ============================================================
# TABLE 1 — Overall Comparison
# ============================================================

def table_overall(df, metrics_df):
    print("\n" + "="*55)
    print("  TABLE 1: OVERALL COMPARISON")
    print("="*55)

    from sklearn.metrics import (
        accuracy_score, f1_score,
        precision_score, recall_score
    )

    y_true   = df['actual_target'].values
    y_static = df['static_pred'].values
    y_dyn    = df['dynamic_pred'].values

    sm = y_static != -1
    dm = y_dyn    != -1
    
    def met(yt, yp, mask):
        yt2 = yt[mask]; yp2 = yp[mask]
        return {
            'Accuracy' : accuracy_score(yt2,yp2)*100,
            'F1'       : f1_score(yt2,yp2,
                         average='weighted',
                         zero_division=0)*100,
            'Precision': precision_score(yt2,yp2,
                         average='weighted',
                         zero_division=0)*100,
            'Recall'   : recall_score(yt2,yp2,
                         average='weighted',
                         zero_division=0)*100,
            'Coverage' : mask.sum()/len(mask)*100
        }

    s_met = met(y_true, y_static, sm)
    d_met = met(y_true, y_dyn,    dm)

    print(f"\n  {'Metric':<12} {'Static XGB':>12} "
          f"{'Dynamic XGB':>13} {'Diff':>8}")
    print(f"  {'─'*12} {'─'*12} {'─'*13} {'─'*8}")

    summary_rows = []
    for key in ['Accuracy','F1','Precision',
                'Recall','Coverage']:
        s_val = s_met[key]
        d_val = d_met[key]
        diff  = d_val - s_val
        sign  = '+' if diff >= 0 else ''
        print(f"  {key:<12} {s_val:>11.2f}% "
              f"{d_val:>12.2f}% "
              f"{sign}{diff:>6.2f}%")
        summary_rows.append({
            'Metric'     : key,
            'Static_XGB' : round(s_val, 2),
            'Dynamic_XGB': round(d_val, 2),
            'Difference' : round(diff,  2)
        })

    summary_df = pd.DataFrame(summary_rows)
    out = os.path.join(
        METRICS_DIR, 'evaluation_summary.csv'
    )
    summary_df.to_csv(out, index=False)
    print(f"\n  ✅ Saved: evaluation_summary.csv")

    return s_met, d_met


# ============================================================
# GRAPH 1 — Overall Accuracy Bar Chart
# ============================================================

def graph_overall(metrics_df):
    fig, ax = plt.subplots(figsize=(10, 6))

    stocks   = metrics_df['stock'].tolist()
    static_a = metrics_df['static_accuracy'].tolist()
    dynamic_a= metrics_df['dynamic_accuracy'].tolist()

    x    = np.arange(len(stocks))
    w    = 0.35

    b1 = ax.bar(x - w/2, static_a,  w,
                label='Static XGB',
                color=COLORS['static'],
                alpha=0.85, edgecolor='white')
    b2 = ax.bar(x + w/2, dynamic_a, w,
                label='Dynamic XGB',
                color=COLORS['dynamic'],
                alpha=0.85, edgecolor='white')

    # Value labels
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom',
                fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{bar.get_height():.1f}%',
                ha='center', va='bottom',
                fontsize=9)

    ax.set_xlabel('Stock')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(
        'Static vs Dynamic XGBoost Accuracy\n'
        'NSE Stocks — Walk-Forward Validation'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(stocks, rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(45, 75)
    ax.axhline(y=50, color='red', linestyle='--',
               alpha=0.3, label='Random baseline')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, 'accuracy_comparison.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Graph 1 saved: accuracy_comparison.png")


# ============================================================
# GRAPH 2 — Year-by-Year Accuracy
# ============================================================

def graph_yearly(df):
    df['year'] = pd.to_datetime(
        df['predict_date']
    ).dt.year

    yearly_s = df.groupby('year')[
        'static_correct'
    ].mean() * 100
    yearly_d = df.groupby('year')[
        'dynamic_correct'
    ].mean() * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    years = sorted(df['year'].unique())
    x     = np.arange(len(years))
    w     = 0.35

    ax.bar(x - w/2,
           [yearly_s.get(y, 0) for y in years],
           w, label='Static XGB',
           color=COLORS['static'],
           alpha=0.85, edgecolor='white')
    ax.bar(x + w/2,
           [yearly_d.get(y, 0) for y in years],
           w, label='Dynamic XGB',
           color=COLORS['dynamic'],
           alpha=0.85, edgecolor='white')

    ax.set_xlabel('Year')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(
        'Year-by-Year Accuracy Comparison\n'
        'Static vs Dynamic Feature Selection'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.legend()
    ax.set_ylim(45, 75)
    ax.axhline(y=50, color='red',
               linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, year in enumerate(years):
        sv = yearly_s.get(year, 0)
        dv = yearly_d.get(year, 0)
        ax.text(i - w/2, sv + 0.3,
                f'{sv:.1f}%', ha='center',
                fontsize=9)
        ax.text(i + w/2, dv + 0.3,
                f'{dv:.1f}%', ha='center',
                fontsize=9)

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, 'yearly_accuracy.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Graph 2 saved: yearly_accuracy.png")


# ============================================================
# GRAPH 3 — Regime-wise Accuracy
# ============================================================

def graph_regime(df):
    regimes = ['Bull', 'Bear', 'Sideways']
    s_accs  = []
    d_accs  = []
    counts  = []

    for regime in regimes:
        rdf = df[df['market_regime'] == regime]
        if len(rdf) == 0:
            s_accs.append(0)
            d_accs.append(0)
            counts.append(0)
            continue
        s_accs.append(
            rdf['static_correct'].mean() * 100
        )
        d_accs.append(
            rdf['dynamic_correct'].mean() * 100
        )
        counts.append(len(rdf))

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(regimes))
    w = 0.35

    ax.bar(x - w/2, s_accs, w,
           label='Static XGB',
           color=COLORS['static'],
           alpha=0.85, edgecolor='white')
    ax.bar(x + w/2, d_accs, w,
           label='Dynamic XGB',
           color=COLORS['dynamic'],
           alpha=0.85, edgecolor='white')

    ax.set_xlabel('Market Regime')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(
        'Accuracy by Market Regime\n'
        'Static vs Dynamic Feature Selection'
    )
    ax.set_xticks(x)
    ax.set_xticklabels([
        f'{r}\n(n={c:,})' for r, c in
        zip(regimes, counts)
    ])
    ax.legend()
    ax.set_ylim(45, 75)
    ax.axhline(y=50, color='red',
               linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)

    for i in range(len(regimes)):
        ax.text(i - w/2, s_accs[i] + 0.3,
                f'{s_accs[i]:.1f}%',
                ha='center', fontsize=9)
        ax.text(i + w/2, d_accs[i] + 0.3,
                f'{d_accs[i]:.1f}%',
                ha='center', fontsize=9)

    plt.tight_layout()
    out = os.path.join(GRAPHS_DIR, 'regime_accuracy.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Graph 3 saved: regime_accuracy.png")


# ============================================================
# GRAPH 4 — Feature Selection Frequency
# ============================================================

def graph_features(df):
    all_features = []
    for feat_str in df['dynamic_features'].dropna():
        all_features.extend(feat_str.split(','))

    from collections import Counter
    freq  = Counter(all_features)
    top15 = freq.most_common(15)
    names = [f[0] for f in top15]
    vals  = [f[1] / len(df) * 100 for f in top15]

    fig, ax = plt.subplots(figsize=(10, 7))

    colors = [COLORS['dynamic']] * len(names)
    bars   = ax.barh(names[::-1], vals[::-1],
                     color=colors, alpha=0.85,
                     edgecolor='white')

    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%',
                va='center', fontsize=9)

    ax.set_xlabel('Selection Frequency (%)')
    ax.set_title(
        'Dynamic Feature Selection Frequency\n'
        'How Often Each Feature Was Selected '
        'Across All Windows'
    )
    ax.set_xlim(0, max(vals) + 10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(
        GRAPHS_DIR, 'feature_frequency.png'
    )
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Graph 4 saved: feature_frequency.png")


# ============================================================
# GRAPH 5 — Prediction Distribution
# ============================================================

def graph_predictions(df):
    total = len(df)

    categories = ['BUY (1)', 'SELL (0)', 'NO TRADE']
    actual  = [
        (df['actual_target']==1).sum()/total*100,
        (df['actual_target']==0).sum()/total*100,
        0
    ]
    static  = [
        (df['static_pred']==1).sum()/total*100,
        (df['static_pred']==0).sum()/total*100,
        (df['static_pred']==-1).sum()/total*100
    ]
    dynamic = [
        (df['dynamic_pred']==1).sum()/total*100,
        (df['dynamic_pred']==0).sum()/total*100,
        (df['dynamic_pred']==-1).sum()/total*100
    ]

    x = np.arange(len(categories))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.bar(x - w,   actual,  w, label='Actual',
           color='#FF9800', alpha=0.85,
           edgecolor='white')
    ax.bar(x,       static,  w, label='Static XGB',
           color=COLORS['static'], alpha=0.85,
           edgecolor='white')
    ax.bar(x + w,   dynamic, w, label='Dynamic XGB',
           color=COLORS['dynamic'], alpha=0.85,
           edgecolor='white')

    ax.set_ylabel('Percentage of Predictions (%)')
    ax.set_title(
        'Prediction Distribution Comparison\n'
        'Actual vs Static vs Dynamic'
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(
        GRAPHS_DIR, 'prediction_distribution.png'
    )
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Graph 5 saved: prediction_distribution.png")


# ============================================================
# GRAPH 6 + FINANCIAL METRICS — Portfolio Simulation
# ============================================================

def financial_metrics(df):
    print("\n" + "="*55)
    print("  FINANCIAL METRICS")
    print("="*55)

    df = df.copy()
    df['predict_date'] = pd.to_datetime(
        df['predict_date']
    )
    df = df.sort_values(
        ['stock', 'predict_date']
    )

    def simulate_one_stock(stock_df, pred_col):
        """
        Simulate portfolio for one stock.
        Start: 10000
        BUY  signal → gain if actual=Up
        SELL signal → gain if actual=Down
        NO TRADE    → no change
        Use realistic +/-0.5% per trade return.
        """
        portfolio = 10000.0
        values    = [portfolio]
        wins      = 0
        trades    = 0

        for _, row in stock_df.iterrows():
            pred   = row[pred_col]
            actual = row['actual_target']

            if pred == 1:    # BUY
                ret = 0.005 if actual == 1 else -0.005
                portfolio *= (1 + ret)
                trades    += 1
                if actual == 1:
                    wins += 1
            elif pred == 0:  # SELL
                ret = 0.005 if actual == 0 else -0.005
                portfolio *= (1 + ret)
                trades    += 1
                if actual == 0:
                    wins += 1

            values.append(portfolio)

        win_rate = wins/trades*100 if trades > 0 else 0
        return values, win_rate, trades

    def sharpe(values):
        rets = np.diff(values) / np.array(values[:-1])
        if rets.std() == 0:
            return 0.0
        return (rets.mean() / rets.std()) * np.sqrt(252)

    def mdd(values):
        v    = np.array(values)
        peak = np.maximum.accumulate(v)
        dd   = (v - peak) / peak
        return dd.min() * 100

    def roi(values):
        return (values[-1] - values[0]) / values[0] * 100

    # Per stock simulation then average
    stocks         = df['stock'].unique()
    s_rois, d_rois = [], []
    s_srs,  d_srs  = [], []
    s_mdds, d_mdds = [], []
    s_wrs,  d_wrs  = [], []
    s_trds, d_trds = [], []

    print(f"\n  {'Stock':<12} "
          f"{'S-ROI':>8} {'D-ROI':>8} "
          f"{'S-SR':>7} {'D-SR':>7}")
    print(f"  {'─'*12} "
          f"{'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7}")

    for stock in stocks:
        sdf = df[df['stock'] == stock]

        sv, s_wr, s_tr = simulate_one_stock(
            sdf, 'static_pred'
        )
        dv, d_wr, d_tr = simulate_one_stock(
            sdf, 'dynamic_pred'
        )

        s_rois.append(roi(sv))
        d_rois.append(roi(dv))
        s_srs.append(sharpe(sv))
        d_srs.append(sharpe(dv))
        s_mdds.append(mdd(sv))
        d_mdds.append(mdd(dv))
        s_wrs.append(s_wr)
        d_wrs.append(d_wr)
        s_trds.append(s_tr)
        d_trds.append(d_tr)

        print(f"  {stock:<12} "
              f"{roi(sv):>7.1f}% "
              f"{roi(dv):>7.1f}% "
              f"{sharpe(sv):>6.3f} "
              f"{sharpe(dv):>6.3f}")

    # Averages
    print(f"  {'─'*12} "
          f"{'─'*8} {'─'*8} "
          f"{'─'*7} {'─'*7}")
    print(f"  {'AVERAGE':<12} "
          f"{np.mean(s_rois):>7.1f}% "
          f"{np.mean(d_rois):>7.1f}% "
          f"{np.mean(s_srs):>6.3f} "
          f"{np.mean(d_srs):>6.3f}")

    # Final summary table
    metrics = {
        'Metric'      : [
            'Avg ROI (%)',
            'Avg Sharpe Ratio',
            'Avg Max Drawdown (%)',
            'Avg Win Rate (%)',
            'Avg Trades per Stock'
        ],
        'Static XGB'  : [
            f"{np.mean(s_rois):.2f}%",
            f"{np.mean(s_srs):.3f}",
            f"{np.mean(s_mdds):.2f}%",
            f"{np.mean(s_wrs):.2f}%",
            f"{int(np.mean(s_trds))}"
        ],
        'Dynamic XGB' : [
            f"{np.mean(d_rois):.2f}%",
            f"{np.mean(d_srs):.3f}",
            f"{np.mean(d_mdds):.2f}%",
            f"{np.mean(d_wrs):.2f}%",
            f"{int(np.mean(d_trds))}"
        ]
    }

    print(f"\n  {'Metric':<24} "
          f"{'Static XGB':>12} "
          f"{'Dynamic XGB':>13}")
    print(f"  {'─'*24} {'─'*12} {'─'*13}")

    mdf = pd.DataFrame(metrics)
    for _, row in mdf.iterrows():
        print(f"  {row['Metric']:<24} "
              f"{row['Static XGB']:>12} "
              f"{row['Dynamic XGB']:>13}")

    # Save
    mdf.to_csv(
        os.path.join(METRICS_DIR,
                     'financial_metrics.csv'),
        index=False
    )
    print(f"\n  ✅ Saved: financial_metrics.csv")

    # Portfolio graph — use one representative stock
    rep_stock = df[
        df['stock'] == stocks[0]
    ].sort_values('predict_date')

    sv, _, _ = simulate_one_stock(
        rep_stock, 'static_pred'
    )
    dv, _, _ = simulate_one_stock(
        rep_stock, 'dynamic_pred'
    )
    bh = [10000.0]
    for _, row in rep_stock.iterrows():
        ret = 0.005 if row['actual_target'] == 1 \
              else -0.005
        bh.append(bh[-1] * (1 + ret))

    dates = (
        [rep_stock['predict_date'].iloc[0]] +
        rep_stock['predict_date'].tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, sv, label='Static XGB',
            color=COLORS['static'],
            linewidth=1.5)
    ax.plot(dates, dv, label='Dynamic XGB',
            color=COLORS['dynamic'],
            linewidth=1.5)
    ax.plot(dates, bh, label='Buy and Hold',
            color=COLORS['neutral'],
            linewidth=1.5, linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (₹)')
    ax.set_title(
        f'Portfolio Growth — {stocks[0]}\n'
        f'Starting Capital: ₹10,000'
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, p: f'₹{x:,.0f}'
        )
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(GRAPHS_DIR,
                     'portfolio_growth.png'),
        bbox_inches='tight'
    )
    plt.close()
    print(f"  ✅ Graph 6 saved: portfolio_growth.png")

    return mdf

    def simulate_portfolio(
        predictions, daily_returns,
        start=10000, threshold=0.55
    ):
        portfolio = [start]
        wins = 0
        trades = 0

        for pred, ret in zip(
            predictions, daily_returns
        ):
            current = portfolio[-1]
            if pred == 1:   # BUY signal
                new_val = current * (1 + ret)
                trades += 1
                if ret > 0:
                    wins += 1
            elif pred == 0: # SELL signal
                new_val = current * (1 - ret)
                trades += 1
                if ret < 0:
                    wins += 1
            else:           # NO TRADE
                new_val = current
            portfolio.append(new_val)

        win_rate = wins/trades*100 if trades > 0 else 0
        return portfolio, win_rate, trades

    def sharpe_ratio(portfolio):
        returns = np.diff(portfolio) / portfolio[:-1]
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)

    def max_drawdown(portfolio):
        portfolio = np.array(portfolio)
        peak = np.maximum.accumulate(portfolio)
        dd   = (portfolio - peak) / peak
        return dd.min() * 100

    def roi(portfolio):
        return (portfolio[-1] - portfolio[0]) / portfolio[0] * 100

    # Run simulations
    s_port, s_wr, s_trades = simulate_portfolio(
        df['static_pred'].values,
        df['daily_return'].values
    )
    d_port, d_wr, d_trades = simulate_portfolio(
        df['dynamic_pred'].values,
        df['daily_return'].values
    )

    # Buy and Hold
    bh_port = [10000]
    for ret in df['daily_return'].values:
        bh_port.append(bh_port[-1] * (1 + ret))

    # Calculate metrics
    metrics = {
        'Metric'            : [
            'Final Portfolio (₹)',
            'ROI (%)',
            'Sharpe Ratio',
            'Max Drawdown (%)',
            'Win Rate (%)',
            'Total Trades'
        ],
        'Static XGB'        : [
            f"₹{s_port[-1]:,.0f}",
            f"{roi(s_port):.2f}%",
            f"{sharpe_ratio(s_port):.3f}",
            f"{max_drawdown(s_port):.2f}%",
            f"{s_wr:.2f}%",
            f"{s_trades}"
        ],
        'Dynamic XGB'       : [
            f"₹{d_port[-1]:,.0f}",
            f"{roi(d_port):.2f}%",
            f"{sharpe_ratio(d_port):.3f}",
            f"{max_drawdown(d_port):.2f}%",
            f"{d_wr:.2f}%",
            f"{d_trades}"
        ],
        'Buy and Hold'      : [
            f"₹{bh_port[-1]:,.0f}",
            f"{roi(bh_port):.2f}%",
            f"{sharpe_ratio(bh_port):.3f}",
            f"{max_drawdown(bh_port):.2f}%",
            "N/A",
            "N/A"
        ]
    }

    metrics_df = pd.DataFrame(metrics)

    print(f"\n  {'Metric':<22} "
          f"{'Static XGB':>14} "
          f"{'Dynamic XGB':>14} "
          f"{'Buy & Hold':>12}")
    print(f"  {'─'*22} {'─'*14} {'─'*14} {'─'*12}")

    for _, row in metrics_df.iterrows():
        print(f"  {row['Metric']:<22} "
              f"{row['Static XGB']:>14} "
              f"{row['Dynamic XGB']:>14} "
              f"{row['Buy and Hold']:>12}")

    # Save
    out = os.path.join(
        METRICS_DIR, 'financial_metrics.csv'
    )
    metrics_df.to_csv(out, index=False)
    print(f"\n  ✅ Saved: financial_metrics.csv")

    # Portfolio growth graph
    fig, ax = plt.subplots(figsize=(12, 6))

    dates = [df['predict_date'].iloc[0]] + \
            df['predict_date'].tolist()

    ax.plot(dates, s_port,
            label='Static XGB',
            color=COLORS['static'],
            linewidth=1.5, alpha=0.85)
    ax.plot(dates, d_port,
            label='Dynamic XGB',
            color=COLORS['dynamic'],
            linewidth=1.5, alpha=0.85)
    ax.plot(dates, bh_port,
            label='Buy and Hold',
            color=COLORS['neutral'],
            linewidth=1.5, alpha=0.7,
            linestyle='--')

    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value (₹)')
    ax.set_title(
        'Portfolio Growth Comparison\n'
        'Starting Capital: ₹10,000'
    )
    ax.legend()
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda x, p: f'₹{x:,.0f}'
        )
    )

    plt.tight_layout()
    out = os.path.join(
        GRAPHS_DIR, 'portfolio_growth.png'
    )
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Graph 6 saved: portfolio_growth.png")

    return metrics_df


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*55)
    print("  EVALUATION MODULE")
    print("  Dynamic Feature Selection Project")
    print("="*55)

    os.makedirs(GRAPHS_DIR,  exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # Load results
    df, metrics_df = load_results()

    # Table 1
    s_met, d_met = table_overall(df, metrics_df)

    # Graphs
    print("\n  Generating graphs...")
    graph_overall(metrics_df)
    graph_yearly(df)
    graph_regime(df)
    graph_features(df)
    graph_predictions(df)

    # Financial metrics + Graph 6
    financial_metrics(df)

    print("\n" + "="*55)
    print("  ✅ EVALUATION COMPLETE")
    print(f"  Graphs saved to: reports/graphs/")
    print(f"  Metrics saved to: reports/metrics/")
    print("\n  Files generated:")
    for f in [
        'accuracy_comparison.png',
        'yearly_accuracy.png',
        'regime_accuracy.png',
        'feature_frequency.png',
        'prediction_distribution.png',
        'portfolio_growth.png',
        'evaluation_summary.csv',
        'financial_metrics.csv'
    ]:
        print(f"    ✅ {f}")
    print("="*55 + "\n")


if __name__ == '__main__':
    main()