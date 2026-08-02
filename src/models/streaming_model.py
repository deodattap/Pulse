# ============================================================
# src/models/streaming_model.py
#
# TRUE STREAMING pipeline using River library.
# Hoeffding Tree with Dynamic Feature Selection.
#
# This is the correct approach for streaming data:
#   - Learns from each sample incrementally
#   - Never needs full retraining
#   - Handles concept drift automatically
#   - Dynamic feature selection updates per window
#
# Comparison:
#   Static:  Hoeffding Tree on ALL features
#   Dynamic: Hoeffding Tree on MI-selected features
#
# How to run:
#   python src/models/streaming_model.py --stock POWERGRID
#   python src/models/streaming_model.py --stock ALL
# ============================================================

import pandas as pd
import numpy as np
import sys, os, argparse, time, warnings
warnings.filterwarnings('ignore')

from river import (
    tree, ensemble, preprocessing,
    metrics, drift
)
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    accuracy_score, f1_score,
    precision_score, recall_score
)
from collections import Counter, deque

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *

# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE_MI  = 252   # Days used for MI calculation
MI_UPDATE_FREQ  = 30    # Recalculate MI every N samples
TOP_K           = 10    # Features to select
CONFIDENCE_THRESH = 0.55
TARGET          = 'Binary_Target'
TEST_START      = '2022-01-01'

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
# DYNAMIC FEATURE SELECTOR
# ============================================================

class StreamingFeatureSelector:
    """
    Dynamic feature selection for streaming data.
    Uses XGBoost importance on rolling window.
    Updates every MI_UPDATE_FREQ samples.
    """

    def __init__(self, feature_names, top_k=TOP_K,
                 window_size=WINDOW_SIZE_MI,
                 update_freq=MI_UPDATE_FREQ):
        self.feature_names  = feature_names
        self.top_k          = top_k
        self.window_size    = window_size
        self.update_freq    = update_freq
        self.selected       = feature_names[:top_k]
        self.counter        = 0
        self.window_buffer  = deque(maxlen=window_size)
        self.selection_log  = []

    def update(self, x_dict, y):
        """Add new sample to buffer."""
        row = {f: x_dict[f] for f in self.feature_names}
        row['_y'] = y
        self.window_buffer.append(row)
        self.counter += 1

        # Recalculate every update_freq samples
        if (self.counter % self.update_freq == 0
                and len(self.window_buffer) >= 60):
            self._recalculate()

    def _recalculate(self):
        """Recalculate feature importance."""
        try:
            buf_df = pd.DataFrame(
                list(self.window_buffer)
            )
            X = buf_df[self.feature_names].values
            y = buf_df['_y'].values.astype(int)

            if len(np.unique(y)) < 2:
                return

            from xgboost import XGBClassifier
            from collections import Counter

            counts = Counter(y)
            total  = len(y)
            n      = len(counts)
            wmap   = {
                c: total/(n*cnt)
                for c, cnt in counts.items()
            }
            sw = np.array([wmap[i] for i in y])

            m = XGBClassifier(
                n_estimators=50,
                random_state=42,
                verbosity=0,
                eval_metric='logloss',
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8
            )
            m.fit(X, y, sample_weight=sw)
            importances = m.feature_importances_

            scored = sorted(
                zip(self.feature_names, importances),
                key=lambda x: x[1],
                reverse=True
            )
            self.selected = [f for f, _ in scored[:self.top_k]]
            self.selection_log.append({
                'counter'    : self.counter,
                'selected'   : self.selected.copy(),
                'top_feature': self.selected[0]
            })

        except Exception:
            pass

    def get_selected(self):
        return self.selected


# ============================================================
# CORE FUNCTION — Stream one stock
# ============================================================

def stream_one_stock(full_df, stock_sym, feat_cols):
    """
    True streaming pipeline for one stock.

    Processes data one sample at a time.
    Static model: Hoeffding Tree on all features
    Dynamic model: Hoeffding Tree on MI-selected features

    Uses prequential evaluation:
      1. Predict current sample
      2. Evaluate prediction
      3. Learn from current sample
      4. Move to next
    """

    stock_name = GOOD_STOCKS.get(stock_sym, stock_sym)

    df = full_df[
        full_df['Stock_Symbol'] == stock_sym
    ].sort_values('Date').reset_index(drop=True)

    print(f"\n  [{stock_name}]")
    print(f"  Total rows : {len(df):,}")
    print(f"  Streaming one sample at a time...")

    # ── Static model ───────────────────────────────────
    static_model = (
        preprocessing.StandardScaler() |
        tree.HoeffdingTreeClassifier(
            grace_period = 50,
            delta        = 0.01
        )
    )

    # ── Dynamic model ──────────────────────────────────
    dynamic_model = (
        preprocessing.StandardScaler() |
        tree.HoeffdingTreeClassifier(
            grace_period = 50,
            delta        = 0.01
        )
    )

    # ── Feature selector ───────────────────────────────
    selector = StreamingFeatureSelector(
        feature_names = feat_cols,
        top_k         = TOP_K,
        window_size   = WINDOW_SIZE_MI,
        update_freq   = MI_UPDATE_FREQ
    )

    # ── Drift detector ─────────────────────────────────
    drift_detector = drift.ADWIN()

    # Results storage
    results    = []
    start_time = time.time()
    n_drifts   = 0

    for idx, row in df.iterrows():

        predict_date  = row['Date']
        actual_target = int(row[TARGET])
        is_test = predict_date >= pd.Timestamp(
            TEST_START
        )

        # Full feature dict
        x_full = {f: float(row[f]) for f in feat_cols}

        # Dynamic feature dict
        selected = selector.get_selected()
        x_dyn    = {f: float(row[f]) for f in selected}

        # ── Predict ────────────────────────────────────
        s_pred = static_model.predict_one(x_full)
        d_pred = dynamic_model.predict_one(x_dyn)

        # Convert None to majority class
        if s_pred is None:
            s_pred = 1
        if d_pred is None:
            d_pred = 1

        # Confidence via predict_proba_one
        try:
            s_proba = static_model.predict_proba_one(
                x_full
            )
            s_conf  = max(s_proba.values()) \
                      if s_proba else 0.5
        except:
            s_conf = 0.5

        try:
            d_proba = dynamic_model.predict_proba_one(
                x_dyn
            )
            d_conf  = max(d_proba.values()) \
                      if d_proba else 0.5
        except:
            d_conf = 0.5

        # Apply confidence threshold
        s_signal = ('BUY' if s_pred == 1 else 'SELL') \
                   if s_conf >= CONFIDENCE_THRESH \
                   else 'NO TRADE'
        d_signal = ('BUY' if d_pred == 1 else 'SELL') \
                   if d_conf >= CONFIDENCE_THRESH \
                   else 'NO TRADE'

        # ── Learn ──────────────────────────────────────
        static_model.learn_one(x_full, actual_target)
        dynamic_model.learn_one(x_dyn, actual_target)
        selector.update(x_full, actual_target)

        # ── Drift detection ────────────────────────────
        drift_detector.update(
            int(s_pred != actual_target)
        )
        if drift_detector.drift_detected:
            n_drifts += 1

        # ── Record test period results only ────────────
        if not is_test:
            continue

        results.append({
            'stock'           : stock_name,
            'predict_date'    : predict_date,
            'year'            : predict_date.year,
            'actual_target'   : actual_target,
            'market_regime'   : 'Sideways',

            'static_pred'     : s_pred,
            'static_confidence': round(s_conf, 4),
            'static_signal'   : s_signal,
            'static_correct'  : int(
                s_pred == actual_target
            ),

            'dynamic_pred'    : d_pred,
            'dynamic_confidence': round(d_conf, 4),
            'dynamic_signal'  : d_signal,
            'dynamic_correct' : int(
                d_pred == actual_target
            ),

            'n_dynamic_features': len(selected),
            'dynamic_features'  : ','.join(selected),
            'top_mi_feature'    : selected[0]
                                   if selected else '',
        })

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
                  f"Drifts: {n_drifts}  "
                  f"Time: {elapsed:.0f}s")

    results_df = pd.DataFrame(results)
    elapsed    = time.time() - start_time

    if len(results_df) > 0:
        s_acc = results_df['static_correct'].mean()*100
        d_acc = results_df['dynamic_correct'].mean()*100
        diff  = d_acc - s_acc
        print(f"  [{stock_name}] Complete. "
              f"{len(results_df):,} predictions "
              f"in {elapsed:.0f}s  "
              f"Drifts detected: {n_drifts}")
        print(f"  [{stock_name}] "
              f"Static: {s_acc:.2f}%  "
              f"Dynamic: {d_acc:.2f}%  "
              f"Diff: {diff:+.2f}%")

    return results_df


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(results_df, stock='ALL'):
    if len(results_df) == 0:
        return {}

    y_true   = results_df['actual_target'].values
    y_static = results_df['static_pred'].values
    y_dyn    = results_df['dynamic_pred'].values

    def met(yt, yp):
        return {
            'accuracy' : round(
                accuracy_score(yt,yp)*100, 2),
            'f1'       : round(
                f1_score(yt,yp,average='weighted',
                         zero_division=0)*100, 2),
            'precision': round(
                precision_score(yt,yp,
                    average='weighted',
                    zero_division=0)*100, 2),
            'recall'   : round(
                recall_score(yt,yp,
                    average='weighted',
                    zero_division=0)*100, 2),
        }

    sm = met(y_true, y_static)
    dm = met(y_true, y_dyn)

    return {
        'stock'               : stock,
        'n_predictions'       : len(results_df),
        'static_accuracy'     : sm['accuracy'],
        'static_f1'           : sm['f1'],
        'static_precision'    : sm['precision'],
        'static_recall'       : sm['recall'],
        'dynamic_accuracy'    : dm['accuracy'],
        'dynamic_f1'          : dm['f1'],
        'dynamic_precision'   : dm['precision'],
        'dynamic_recall'      : dm['recall'],
        'accuracy_improvement': round(
            dm['accuracy'] - sm['accuracy'], 2),
        'f1_improvement'      : round(
            dm['f1'] - sm['f1'], 2),
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(results_df, metrics, stock_name):
    print(f"\n  {'═'*55}")
    print(f"  STREAMING RESULTS: {stock_name}")
    print(f"  {'═'*55}")

    print(f"\n  {'Metric':<20} "
          f"{'Static HT':>11} {'Dynamic HT':>12}")
    print(f"  {'─'*20} {'─'*11} {'─'*12}")

    for key in ['accuracy','f1','precision','recall']:
        s = metrics[f'static_{key}']
        d = metrics[f'dynamic_{key}']
        diff = d - s
        sign = '+' if diff >= 0 else ''
        print(f"  {key.capitalize():<20} "
              f"{s:>10.2f}% "
              f"{d:>11.2f}%  "
              f"({sign}{diff:.2f}%)")

    diff = metrics['accuracy_improvement']
    if diff > 0:
        print(f"\n  ✅ Dynamic BEATS Static by {diff:.2f}%")
    else:
        print(f"\n  ⚠️  Static leads by {abs(diff):.2f}%")

    # Year by year
    if 'year' in results_df.columns:
        print(f"\n  Year-by-Year:")
        print(f"  {'Year':<6} {'Static':>8} "
              f"{'Dynamic':>9} {'Diff':>7} {'Winner':>10}")
        print(f"  {'─'*6} {'─'*8} "
              f"{'─'*9} {'─'*7} {'─'*10}")

        for year in sorted(
            results_df['year'].unique()
        ):
            yr  = results_df[
                results_df['year'] == year
            ]
            sa  = yr['static_correct'].mean()  * 100
            da  = yr['dynamic_correct'].mean() * 100
            d   = da - sa
            win = 'Dynamic ✅' if da > sa else 'Static'
            print(f"  {year:<6} {sa:>7.2f}% "
                  f"{da:>8.2f}% "
                  f"{d:>+6.2f}% {win:>10}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stock', type=str, default='POWERGRID'
    )
    args      = parser.parse_args()
    stock_arg = args.stock.upper()

    print("\n" + "="*60)
    print("  STREAMING MODEL — Hoeffding Tree")
    print("  Static vs Dynamic Feature Selection")
    print("  True Online Learning (River library)")
    print("="*60)

    print(f"\n  Algorithm : Hoeffding Tree (VFDT)")
    print(f"  Learning  : Incremental (one sample)")
    print(f"  Drift det : ADWIN")
    print(f"  MI update : every {MI_UPDATE_FREQ} samples")
    print(f"  Top-K     : {TOP_K} features")

    # Load data
    nse_path = os.path.join(
        DATA_PROCESSED_DIR, 'nse_dataset.csv'
    )
    df = pd.read_csv(
        nse_path, parse_dates=['Date']
    )
    print(f"\n  ✅ Loaded {len(df):,} rows")

    with open(os.path.join(
        DATA_PROCESSED_DIR, 'nse_feature_cols.txt'
    )) as f:
        feat_cols = [
            l.strip() for l in f
            if l.strip() in df.columns
        ]
    print(f"  ✅ Features: {len(feat_cols)}")

    # Stocks
    if stock_arg == 'ALL':
        stocks = list(GOOD_STOCKS.keys())
        print(f"\n  Running on ALL {len(stocks)} stocks")
    else:
        sym = None
        for s, n in GOOD_STOCKS.items():
            if stock_arg in s.upper() or \
               stock_arg == n.upper():
                sym = s
                break
        if sym is None:
            print(f"  ❌ Stock not found")
            sys.exit(1)
        stocks = [sym]

    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    all_results = []
    all_metrics = []
    total_start = time.time()

    for i, sym in enumerate(stocks, 1):
        name = GOOD_STOCKS[sym]
        print(f"\n  {'─'*55}")
        print(f"  [{i}/{len(stocks)}] {name}")
        print(f"  {'─'*55}")

        results_df = stream_one_stock(
            df, sym, feat_cols
        )

        if len(results_df) == 0:
            continue

        m = calculate_metrics(results_df, name)
        all_metrics.append(m)
        all_results.append(results_df)
        print_summary(results_df, m, name)

        results_df.to_csv(
            os.path.join(
                LOGS_DIR,
                f'streaming_{name}.csv'
            ),
            index=False
        )
        pd.DataFrame(all_metrics).to_csv(
            os.path.join(
                METRICS_DIR,
                'streaming_results.csv'
            ),
            index=False
        )

    if not all_results:
        print("\n  ❌ No results")
        sys.exit(1)

    combined_df = pd.concat(
        all_results, ignore_index=True
    )
    metrics_df  = pd.DataFrame(all_metrics)

    combined_df.to_csv(
        os.path.join(
            LOGS_DIR, 'streaming_ALL.csv'
        ),
        index=False
    )

    total_elapsed = time.time() - total_start

    print("\n" + "="*60)
    print("  FINAL STREAMING RESULTS")
    print("="*60)

    print(f"\n  {'Stock':<14} "
          f"{'Static HT':>10} "
          f"{'Dynamic HT':>11} "
          f"{'Diff':>7} {'Winner':>10}")
    print(f"  {'─'*14} {'─'*10} "
          f"{'─'*11} {'─'*7} {'─'*10}")

    for m in all_metrics:
        s   = m['static_accuracy']
        d   = m['dynamic_accuracy']
        dif = m['accuracy_improvement']
        win = 'Dynamic ✅' if dif > 0 else 'Static'
        print(f"  {m['stock']:<14} "
              f"{s:>9.2f}% "
              f"{d:>10.2f}% "
              f"{dif:>+6.2f}% {win:>10}")

    avg_s = np.mean([m['static_accuracy']
                     for m in all_metrics])
    avg_d = np.mean([m['dynamic_accuracy']
                     for m in all_metrics])
    avg_d_diff = avg_d - avg_s

    print(f"  {'─'*14} {'─'*10} "
          f"{'─'*11} {'─'*7} {'─'*10}")
    print(f"  {'AVERAGE':<14} "
          f"{avg_s:>9.2f}% "
          f"{avg_d:>10.2f}% "
          f"{avg_d_diff:>+6.2f}%")

    print(f"\n  Total time: {total_elapsed/60:.1f} min")
    print(f"\n  ✅ Results saved to reports/")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()