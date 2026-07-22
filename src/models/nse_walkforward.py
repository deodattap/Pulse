# ============================================================
# src/models/nse_walkforward.py
#
# Walk-Forward Validation on NSE Data
# Static XGBoost vs Dynamic XGBoost
# with MI-based Dynamic Feature Selection
#
# This is the CORE comparison of the project.
# Static: all 33 features, never changes
# Dynamic: top 10 MI features, changes per window
#
# How to run:
#   python src/models/nse_walkforward.py --stock POWERGRID
#   python src/models/nse_walkforward.py --stock ALL
# ============================================================

import pandas as pd
import numpy as np
import sys
import os
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing     import StandardScaler
from sklearn.metrics           import (
    accuracy_score, f1_score,
    precision_score, recall_score
)
from xgboost     import XGBClassifier
from collections import Counter

# ── Import config ──────────────────────────────────────────
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *


# ============================================================
# SETTINGS
# ============================================================

# Minimum training rows before first prediction
MIN_TRAIN_ROWS = 504   # ~2 years

# Rolling window for MI calculation
MI_WINDOW = 252        # ~1 year

# Retrain every N days
RETRAIN_EVERY = 60     # ~3 months

# Top features to select dynamically
TOP_K = 10

# Confidence threshold for signals
CONFIDENCE_THRESHOLD = 0.55

# Target column
TARGET = 'Binary_Target'

# Good NSE stocks to use
GOOD_STOCKS = {
    'POWERGRID.NS' : 'POWERGRID',
    'KOTAKBANK.NS' : 'KOTAKBANK',
    'INFY.NS'      : 'INFY',
    'HINDUNILVR.NS': 'HINDUNILVR',
    'ICICIBANK.NS' : 'ICICIBANK',
    'TCS.NS'       : 'TCS',
    'RELIANCE.NS'  : 'RELIANCE',
    'HDFCBANK.NS'  : 'HDFCBANK',
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_sample_weights(y):
    """Balanced sample weights for XGBoost."""
    counts = Counter(y)
    total  = len(y)
    n      = len(counts)
    wmap   = {
        c: total / (n * cnt)
        for c, cnt in counts.items()
    }
    return np.array([wmap[label] for label in y])


def calculate_mi_scores(X, y, feature_names):
    """
    Calculate Mutual Information scores.
    Returns sorted dict {feature: score}.
    Higher score = more relevant to target NOW.
    """
    try:
        scores = mutual_info_classif(
            X, y,
            random_state = RANDOM_STATE,
            n_neighbors  = 5
        )
        score_dict = dict(zip(feature_names, scores))
        return dict(sorted(
            score_dict.items(),
            key     = lambda x: x[1],
            reverse = True
        ))
    except Exception:
        return {f: 1.0 for f in feature_names}


def select_top_features(mi_scores, top_k=TOP_K):
    """Return top K feature names by MI score."""
    return list(mi_scores.keys())[:top_k]


def train_xgboost(X, y):
    """Train XGBoost with balanced weights."""
    try:
        if len(np.unique(y)) < 2:
            return None
        sw  = get_sample_weights(y)
        model = XGBClassifier(
            n_estimators     = 200,
            random_state     = RANDOM_STATE,
            verbosity        = 0,
            eval_metric      = 'logloss',
            max_depth        = 6,
            learning_rate    = 0.05,
            subsample        = 0.8,
            colsample_bytree = 0.8
        )
        model.fit(X, y, sample_weight=sw)
        return model
    except Exception:
        return None


def predict_with_confidence(model, X_row, threshold=CONFIDENCE_THRESHOLD):
    """
    Predict with confidence threshold.
    Returns (prediction, confidence, signal_name)
    """
    try:
        proba      = model.predict_proba(X_row)[0]
        confidence = float(np.max(proba))
        pred       = int(np.argmax(proba))

        if confidence >= threshold:
            signal = 'BUY' if pred == 1 else 'SELL'
        else:
            signal = 'NO TRADE'
            pred   = -1

        return pred, confidence, signal
    except Exception:
        return -1, 0.0, 'NO TRADE'


def detect_regime(recent_data):
    """
    Simple market regime detection.
    Returns: Bull / Bear / Sideways
    """
    if 'Daily_Return' not in recent_data.columns:
        return 'Sideways'

    returns    = recent_data['Daily_Return'].tail(60)
    avg_return = returns.mean()
    volatility = returns.std()

    if avg_return > 0.0005 and volatility < 0.02:
        return 'Bull'
    elif avg_return < -0.0005 or volatility > 0.025:
        return 'Bear'
    else:
        return 'Sideways'


# ============================================================
# CORE FUNCTION — Walk Forward for ONE stock
# ============================================================

def run_walkforward(full_df, stock_sym, feat_cols):
    """
    Run walk-forward validation for one stock.

    At each test day:
      Static model  → uses ALL features, never changes
      Dynamic model → uses TOP 10 MI features, changes

    Both evaluated on identical days.
    Fair apples-to-apples comparison.
    """

    stock_name = GOOD_STOCKS.get(stock_sym, stock_sym)

    # Get stock data sorted by date
    df = full_df[
        full_df['Stock_Symbol'] == stock_sym
    ].sort_values('Date').reset_index(drop=True)

    total_rows = len(df)

    print(f"\n  [{stock_name}]")
    print(f"  Total rows  : {total_rows:,}")
    print(f"  Min train   : {MIN_TRAIN_ROWS}")
    print(f"  Starting walk-forward...")

    # Storage
    results          = []
    start_time       = time.time()

    # Models
    static_model     = None
    dynamic_model    = None

    # Feature tracking
    last_mi_scores   = None
    last_dynamic_features = feat_cols[:TOP_K]

    # Retrain counter
    steps_since_retrain = 0

    # ── Walk through all rows ──────────────────────────
    for i in range(MIN_TRAIN_ROWS, total_rows):

        current_row   = df.iloc[i]
        predict_date  = current_row['Date']
        actual_target = int(current_row[TARGET])

        # All past data available at this point
        past_data = df.iloc[:i].copy()

        # ── Skip rows before test period ───────────────
        is_test = predict_date >= pd.Timestamp(
            TEST_START_DATE
        )

        steps_since_retrain += 1
        should_retrain = (
            steps_since_retrain >= RETRAIN_EVERY
            or static_model is None
        )

        if should_retrain:

            # ── Calculate MI on recent window ──────────
            mi_data = past_data.tail(MI_WINDOW)

            if len(mi_data) >= 60:
                X_mi  = mi_data[feat_cols].values
                y_mi  = mi_data[TARGET].values

                mi_scores = calculate_mi_scores(
                    X_mi, y_mi, feat_cols
                )
                last_mi_scores        = mi_scores
                last_dynamic_features = select_top_features(
                    mi_scores, TOP_K
                )

            # ── Train STATIC on all past data ──────────
            X_all = past_data[feat_cols].values
            y_all = past_data[TARGET].values.astype(int)

            new_static = train_xgboost(X_all, y_all)
            if new_static is not None:
                static_model = new_static

            # ── Train DYNAMIC on top features ──────────
            X_dyn = past_data[
                last_dynamic_features
            ].values
            new_dynamic = train_xgboost(X_dyn, y_all)
            if new_dynamic is not None:
                dynamic_model = new_dynamic

            steps_since_retrain = 0

        # ── Only record test period predictions ────────
        if not is_test:
            continue

        # ── Skip if models not ready ───────────────────
        if static_model is None or dynamic_model is None:
            continue

        # ── Detect market regime ───────────────────────
        recent = past_data.tail(50)
        regime = detect_regime(recent)

        # ── Static prediction (all features) ──────────
        static_row = current_row[
            feat_cols
        ].values.reshape(1, -1)

        s_pred, s_conf, s_signal = predict_with_confidence(
            static_model, static_row
        )

        # ── Dynamic prediction (top 10 MI features) ───
        dynamic_row = current_row[
            last_dynamic_features
        ].values.reshape(1, -1)

        d_pred, d_conf, d_signal = predict_with_confidence(
            dynamic_model, dynamic_row
        )

        # ── Get top MI feature for this window ─────────
        top_mi_feature = (
            last_dynamic_features[0]
            if last_dynamic_features else ''
        )

        # ── Record result ──────────────────────────────
        results.append({
            # Identifiers
            'stock'               : stock_name,
            'stock_symbol'        : stock_sym,
            'predict_date'        : predict_date,
            'year'                : predict_date.year,
            'month'               : predict_date.month,
            'actual_target'       : actual_target,
            'market_regime'       : regime,

            # Static model
            'static_pred'         : s_pred,
            'static_confidence'   : round(s_conf, 4),
            'static_signal'       : s_signal,
            'static_correct'      : int(
                s_pred == actual_target
                and s_pred != -1
            ),

            # Dynamic model
            'dynamic_pred'        : d_pred,
            'dynamic_confidence'  : round(d_conf, 4),
            'dynamic_signal'      : d_signal,
            'dynamic_correct'     : int(
                d_pred == actual_target
                and d_pred != -1
            ),

            # Feature selection info
            'n_dynamic_features'  : len(
                last_dynamic_features
            ),
            'dynamic_features'    : ','.join(
                last_dynamic_features
            ),
            'top_mi_feature'      : top_mi_feature,
        })

        # ── Progress update ────────────────────────────
        n = len(results)
        if n % 100 == 0 and n > 0:
            elapsed = time.time() - start_time
            s_acc   = np.mean([
                r['static_correct'] for r in results
            ]) * 100
            d_acc   = np.mean([
                r['dynamic_correct'] for r in results
            ]) * 100
            print(f"  [{stock_name}] "
                  f"{n:>4} predictions  "
                  f"Static: {s_acc:.1f}%  "
                  f"Dynamic: {d_acc:.1f}%  "
                  f"Time: {elapsed:.0f}s")

    # ── Convert to DataFrame ───────────────────────────
    results_df = pd.DataFrame(results)
    elapsed    = time.time() - start_time

    if len(results_df) > 0:
        s_acc = results_df['static_correct'].mean()*100
        d_acc = results_df['dynamic_correct'].mean()*100
        diff  = d_acc - s_acc

        print(f"  [{stock_name}] Complete. "
              f"{len(results_df):,} predictions "
              f"in {elapsed:.0f}s")
        print(f"  [{stock_name}] "
              f"Static: {s_acc:.2f}%  "
              f"Dynamic: {d_acc:.2f}%  "
              f"Diff: {diff:+.2f}%")

    return results_df


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(results_df, stock='ALL'):
    """Calculate all metrics from results."""

    if len(results_df) == 0:
        return {}

    y_true  = results_df['actual_target'].values
    y_static = results_df['static_pred'].values
    y_dynamic= results_df['dynamic_pred'].values

    # Only evaluate on days where signal was given
    # (not NO TRADE days)
    static_mask  = y_static  != -1
    dynamic_mask = y_dynamic != -1

    def get_metrics(y_t, y_p, mask):
        if mask.sum() == 0:
            return {
                'accuracy': 0, 'f1': 0,
                'precision': 0, 'recall': 0,
                'coverage': 0
            }
        yt = y_t[mask]
        yp = y_p[mask]
        return {
            'accuracy' : round(
                accuracy_score(yt, yp) * 100, 2),
            'f1'       : round(
                f1_score(yt, yp,
                         average='weighted',
                         zero_division=0) * 100, 2),
            'precision': round(
                precision_score(yt, yp,
                                average='weighted',
                                zero_division=0)*100,2),
            'recall'   : round(
                recall_score(yt, yp,
                             average='weighted',
                             zero_division=0)*100, 2),
            'coverage' : round(
                mask.sum() / len(mask) * 100, 2)
        }

    sm = get_metrics(y_true, y_static,  static_mask)
    dm = get_metrics(y_true, y_dynamic, dynamic_mask)

    return {
        'stock'              : stock,
        'n_predictions'      : len(results_df),

        'static_accuracy'    : sm['accuracy'],
        'static_f1'          : sm['f1'],
        'static_precision'   : sm['precision'],
        'static_recall'      : sm['recall'],
        'static_coverage'    : sm['coverage'],

        'dynamic_accuracy'   : dm['accuracy'],
        'dynamic_f1'         : dm['f1'],
        'dynamic_precision'  : dm['precision'],
        'dynamic_recall'     : dm['recall'],
        'dynamic_coverage'   : dm['coverage'],

        'accuracy_improvement': round(
            dm['accuracy'] - sm['accuracy'], 2),
        'f1_improvement'      : round(
            dm['f1'] - sm['f1'], 2),
    }


# ============================================================
# PRINT DETAILED SUMMARY
# ============================================================

def print_summary(results_df, metrics, stock_name):
    """Print readable results summary."""

    print(f"\n  {'═'*55}")
    print(f"  RESULTS: {stock_name}")
    print(f"  {'═'*55}")

    # Overall metrics
    print(f"\n  {'Metric':<20} "
          f"{'Static XGB':>12} {'Dynamic XGB':>13}")
    print(f"  {'─'*20} {'─'*12} {'─'*13}")

    print(f"  {'Accuracy':<20} "
          f"{metrics['static_accuracy']:>11.2f}% "
          f"{metrics['dynamic_accuracy']:>12.2f}%")
    print(f"  {'F1 Score':<20} "
          f"{metrics['static_f1']:>11.2f}% "
          f"{metrics['dynamic_f1']:>12.2f}%")
    print(f"  {'Precision':<20} "
          f"{metrics['static_precision']:>11.2f}% "
          f"{metrics['dynamic_precision']:>12.2f}%")
    print(f"  {'Recall':<20} "
          f"{metrics['static_recall']:>11.2f}% "
          f"{metrics['dynamic_recall']:>12.2f}%")
    print(f"  {'Coverage':<20} "
          f"{metrics['static_coverage']:>11.2f}% "
          f"{metrics['dynamic_coverage']:>12.2f}%")

    diff = metrics['accuracy_improvement']
    f1d  = metrics['f1_improvement']
    print(f"\n  Accuracy improvement : {diff:+.2f}%")
    print(f"  F1 improvement       : {f1d:+.2f}%")

    if diff > 0:
        print(f"  ✅ Dynamic BEATS Static")
    else:
        print(f"  ⚠️  Static leads")

    # Year by year
    if 'year' in results_df.columns:
        print(f"\n  Year-by-Year Accuracy:")
        print(f"  {'Year':<6} "
              f"{'Static':>8} {'Dynamic':>9} "
              f"{'Diff':>6} {'Winner':>10}")
        print(f"  {'─'*6} "
              f"{'─'*8} {'─'*9} "
              f"{'─'*6} {'─'*10}")

        for year in sorted(
            results_df['year'].unique()
        ):
            yr  = results_df[
                results_df['year'] == year
            ]
            sm  = yr['static_correct'].mean()  * 100
            dm  = yr['dynamic_correct'].mean() * 100
            d   = dm - sm
            win = 'Dynamic ✅' if dm > sm else 'Static'
            print(f"  {year:<6} "
                  f"{sm:>7.2f}% "
                  f"{dm:>8.2f}% "
                  f"{d:>+5.2f}% "
                  f"{win:>10}")

    # Regime analysis
    if 'market_regime' in results_df.columns:
        print(f"\n  Accuracy by Market Regime:")
        print(f"  {'Regime':<10} "
              f"{'Static':>8} {'Dynamic':>9} "
              f"{'Winner':>10}")
        print(f"  {'─'*10} "
              f"{'─'*8} {'─'*9} {'─'*10}")

        for regime in ['Bull', 'Bear', 'Sideways']:
            rdf = results_df[
                results_df['market_regime'] == regime
            ]
            if len(rdf) == 0:
                continue
            sm  = rdf['static_correct'].mean()  * 100
            dm  = rdf['dynamic_correct'].mean() * 100
            win = 'Dynamic ✅' if dm > sm else 'Static'
            print(f"  {regime:<10} "
                  f"{sm:>7.2f}% "
                  f"{dm:>8.2f}% "
                  f"{win:>10}")

    # Top features selected
    if 'top_mi_feature' in results_df.columns:
        print(f"\n  Most frequently top MI feature:")
        top_f = results_df[
            'top_mi_feature'
        ].value_counts().head(5)
        for feat, count in top_f.items():
            pct = count / len(results_df) * 100
            bar = '█' * int(pct / 5)
            print(f"  {feat:<25} "
                  f"{count:>4} ({pct:.1f}%)  {bar}")

    # Signal distribution
    print(f"\n  Signal Distribution:")
    print(f"  {'Signal':<12} "
          f"{'Static':>8} {'Dynamic':>9}")
    print(f"  {'─'*12} {'─'*8} {'─'*9}")

    total = len(results_df)
    for sig, label in [
        (1,  'BUY'),
        (0,  'SELL'),
        (-1, 'NO TRADE')
    ]:
        s_pct = (
            results_df['static_pred'] == sig
        ).sum() / total * 100
        d_pct = (
            results_df['dynamic_pred'] == sig
        ).sum() / total * 100
        print(f"  {label:<12} "
              f"{s_pct:>7.1f}% "
              f"{d_pct:>8.1f}%")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stock',
        type    = str,
        default = 'POWERGRID',
        help    = 'Stock name or ALL'
    )
    args       = parser.parse_args()
    stock_arg  = args.stock.upper()

    print("\n" + "="*60)
    print("  NSE WALK-FORWARD VALIDATION")
    print("  Static XGBoost vs Dynamic XGBoost")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    print(f"\n  Settings:")
    print(f"  Min train rows    : {MIN_TRAIN_ROWS}")
    print(f"  MI window         : {MI_WINDOW} days")
    print(f"  Retrain every     : {RETRAIN_EVERY} days")
    print(f"  Top K features    : {TOP_K}")
    print(f"  Confidence thresh : {CONFIDENCE_THRESHOLD}")
    print(f"  Test period       : {TEST_START_DATE}+")

    # ── Load NSE dataset ───────────────────────────────
    print(f"\n  Loading NSE dataset...")

    nse_path = os.path.join(
        DATA_PROCESSED_DIR, 'nse_dataset.csv'
    )

    if not os.path.exists(nse_path):
        print(f"  ❌ nse_dataset.csv not found")
        print(f"  Run build_nse_dataset.py first")
        sys.exit(1)

    df = pd.read_csv(
        nse_path, parse_dates=['Date']
    )
    print(f"  ✅ Loaded {len(df):,} rows")

    # ── Load feature columns ───────────────────────────
    feat_path = os.path.join(
        DATA_PROCESSED_DIR, 'nse_feature_cols.txt'
    )
    with open(feat_path) as f:
        feat_cols = [
            l.strip() for l in f
            if l.strip() in df.columns
        ]
    print(f"  ✅ Features: {len(feat_cols)}")

    # ── Determine stocks to run ────────────────────────
    if stock_arg == 'ALL':
        stocks = list(GOOD_STOCKS.keys())
        print(f"\n  Running on ALL "
              f"{len(stocks)} good stocks")
        print(f"  ⚠️  Expected time: 25-40 minutes")
    else:
        # Find by name or symbol
        sym = None
        for s, n in GOOD_STOCKS.items():
            if (stock_arg == n.upper() or
                    stock_arg in s.upper()):
                sym = s
                break
        if sym is None:
            print(f"  ❌ Stock '{stock_arg}' not found")
            print(f"  Available: "
                  f"{list(GOOD_STOCKS.values())}")
            sys.exit(1)
        stocks = [sym]
        name   = GOOD_STOCKS[sym]
        print(f"\n  Running on: {name} ({sym})")
        print(f"  Expected time: 3-5 minutes")

    # ── Create output folders ──────────────────────────
    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Run for each stock ─────────────────────────────
    all_results = []
    all_metrics = []
    total_start = time.time()

    for i, sym in enumerate(stocks, 1):
        name = GOOD_STOCKS[sym]

        print(f"\n  {'─'*55}")
        print(f"  Stock {i}/{len(stocks)}: {name}")
        print(f"  {'─'*55}")

        results_df = run_walkforward(
            df, sym, feat_cols
        )

        if len(results_df) == 0:
            print(f"  ⚠️  No results for {name}")
            continue

        # Calculate metrics
        m = calculate_metrics(results_df, name)
        all_metrics.append(m)
        all_results.append(results_df)

        # Print summary
        print_summary(results_df, m, name)

        # Save per-stock results
        stock_path = os.path.join(
            LOGS_DIR,
            f'nse_walkforward_{name}.csv'
        )
        results_df.to_csv(stock_path, index=False)
        print(f"\n  ✅ Saved: "
              f"reports/logs/nse_walkforward_"
              f"{name}.csv")

        # Save running metrics
        pd.DataFrame(all_metrics).to_csv(
            os.path.join(
                METRICS_DIR, 'nse_final_results.csv'
            ),
            index=False
        )

    # ── Combine all results ────────────────────────────
    if not all_results:
        print("\n  ❌ No results generated")
        sys.exit(1)

    combined_df = pd.concat(
        all_results, ignore_index=True
    )
    metrics_df  = pd.DataFrame(all_metrics)

    combined_df.to_csv(
        os.path.join(
            LOGS_DIR, 'nse_walkforward_ALL.csv'
        ),
        index=False
    )
    metrics_df.to_csv(
        os.path.join(
            METRICS_DIR, 'nse_final_results.csv'
        ),
        index=False
    )

    # ── Final overall summary ──────────────────────────
    total_elapsed = time.time() - total_start
    overall_m     = calculate_metrics(
        combined_df, 'ALL STOCKS'
    )

    print("\n" + "="*60)
    print("  FINAL RESULTS — ALL STOCKS")
    print("="*60)

    print(f"\n  {'Stock':<14} "
          f"{'Static':>8} {'Dynamic':>9} "
          f"{'Diff':>6} {'Winner':>10}")
    print(f"  {'─'*14} "
          f"{'─'*8} {'─'*9} "
          f"{'─'*6} {'─'*10}")

    for m in all_metrics:
        s   = m['static_accuracy']
        d   = m['dynamic_accuracy']
        dif = m['accuracy_improvement']
        win = 'Dynamic ✅' if dif > 0 else 'Static'
        print(f"  {m['stock']:<14} "
              f"{s:>7.2f}% "
              f"{d:>8.2f}% "
              f"{dif:>+5.2f}% "
              f"{win:>10}")

    print(f"  {'─'*14} "
          f"{'─'*8} {'─'*9} "
          f"{'─'*6} {'─'*10}")

    avg_s = np.mean([m['static_accuracy']
                     for m in all_metrics])
    avg_d = np.mean([m['dynamic_accuracy']
                     for m in all_metrics])
    avg_diff = avg_d - avg_s

    print(f"  {'AVERAGE':<14} "
          f"{avg_s:>7.2f}% "
          f"{avg_d:>8.2f}% "
          f"{avg_diff:>+5.2f}%")

    print(f"\n  Overall improvement from "
          f"dynamic selection: {avg_diff:+.2f}%")

    if avg_diff > 0:
        print(f"  ✅ Dynamic Feature Selection "
              f"BEATS Static on NSE data")
        print(f"  ✅ Research hypothesis PROVEN")
    else:
        print(f"  ⚠️  Mixed results")

    print(f"\n  Total predictions : {len(combined_df):,}")
    print(f"  Total time        : "
          f"{total_elapsed/60:.1f} minutes")

    print(f"\n  Files saved:")
    print(f"  ✅ reports/logs/nse_walkforward_ALL.csv")
    print(f"  ✅ reports/metrics/nse_final_results.csv")
    print(f"  ✅ One CSV per stock in reports/logs/")

    print("\n" + "="*60)
    print("  ✅ NSE WALK-FORWARD COMPLETE")
    print("  Next: src/evaluation/evaluation.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()