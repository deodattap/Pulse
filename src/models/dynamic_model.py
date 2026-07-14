# ============================================================
# src/models/dynamic_model.py
#
# Purpose:
# Run the complete Dynamic Feature Selection pipeline.
#
# For each sliding window:
#   1. Select features dynamically using MI
#   2. Train RF and XGBoost on selected features
#   3. Predict next day Buy/Sell/Hold
#   4. Log all results
#
# Compare against static baseline to prove
# dynamic beats static.
#
# How to run (test on AAPL first):
# python src/models/dynamic_model.py --stock AAPL
#
# How to run on all stocks:
# python src/models/dynamic_model.py --stock ALL
# ============================================================

import pandas as pd
import numpy as np
import sys
import os
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics   import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from xgboost import XGBClassifier
from collections import Counter

# ── Import config ──────────────────────────────────────────
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *

# ── Import sliding window and feature selector ─────────────
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'features')
)
from sliding_window              import (
    sliding_window_generator,
    split_windows_by_date,
    get_stock_windows
)
from dynamic_feature_selection   import DynamicFeatureSelector


# ============================================================
# HELPER — Create XGBoost sample weights
# ============================================================

def get_sample_weights(y):
    """
    XGBoost needs manual sample weights
    because it does not support class_weight='balanced'.
    This calculates weights to balance the classes.
    """
    counts    = Counter(y)
    total     = len(y)
    n_classes = len(counts)

    weights_map = {
        cls: total / (n_classes * cnt)
        for cls, cnt in counts.items()
    }

    return np.array([weights_map[label] for label in y])


# ============================================================
# CORE FUNCTION — Run dynamic pipeline for ONE stock
# ============================================================

def run_dynamic_pipeline_for_stock(stock_df, stock_symbol):
    """
    Runs the full dynamic pipeline for one stock.

    For each test window:
      - Selects features dynamically
      - Trains RF and XGB on selected features
      - Predicts next day signal
      - Logs results

    Parameters:
        stock_df     : DataFrame for this stock only
        stock_symbol : e.g. 'AAPL'

    Returns:
        results_df   : DataFrame with all predictions
        selector     : DynamicFeatureSelector with history
    """

    # Sort by date
    stock_df = stock_df.sort_values(DATE_COL)
    stock_df = stock_df.reset_index(drop=True)

    # Get all windows and split into train/test
    all_windows = []
    for w_data, target, date, num in sliding_window_generator(
        stock_df
    ):
        all_windows.append((w_data, target, date, num))

    train_windows, test_windows = split_windows_by_date(
        all_windows
    )

    total_test = len(test_windows)

    if total_test == 0:
        print(f"  ⚠️  No test windows for {stock_symbol}")
        return pd.DataFrame(), None

    # Create feature selector
    selector = DynamicFeatureSelector()

    # Storage for results
    results = []

    # Track previous model for warm starting
    prev_rf_model  = None
    prev_xgb_model = None

    print(f"\n  [{stock_symbol}] "
          f"Test windows: {total_test:,} "
          f"| Starting...")

    start_time = time.time()

    # ── Process each test window ───────────────────────────
    for idx, (w_data, actual_target,
               predict_date, window_num) in enumerate(
        test_windows
    ):

        # ── Step 1: Dynamic feature selection ─────────────
        selected_features = selector.select(w_data, window_num)

        # ── Step 2: Prepare training data from window ─────
        X_window = w_data[selected_features].values
        y_window = w_data[TARGET_COL].values

        # Need at least 2 classes to train
        unique_classes = np.unique(y_window)
        if len(unique_classes) < 2:
            # Skip this window
            continue

        # ── Step 3: Train Random Forest ───────────────────
        try:
            rf_model = RandomForestClassifier(
                n_estimators = 50,       # Fewer trees = faster
                class_weight = CLASS_WEIGHT,
                random_state = RANDOM_STATE,
                n_jobs       = -1,
                max_depth    = 5         # Shallow = faster
            )
            rf_model.fit(X_window, y_window)
        except Exception as e:
            rf_model = prev_rf_model
            if rf_model is None:
                continue

        # ── Step 4: Train XGBoost ─────────────────────────
        try:
            sample_w = get_sample_weights(y_window)

            xgb_model = XGBClassifier(
                n_estimators = 50,
                max_depth    = 3,
                random_state = RANDOM_STATE,
                verbosity    = 0,
                use_label_encoder = False,
                eval_metric  = 'mlogloss'
            )
            xgb_model.fit(
                X_window, y_window,
                sample_weight = sample_w
            )
        except Exception as e:
            xgb_model = prev_xgb_model
            if xgb_model is None:
                continue

        # ── Step 5: Predict next day ───────────────────────
        # Get the current day's features for prediction
        current_features = w_data[selected_features].iloc[-1].values.reshape(1, -1)

        try:
            rf_pred   = rf_model.predict(current_features)[0]
            rf_proba  = rf_model.predict_proba(
                current_features
            )[0]
            rf_conf   = float(np.max(rf_proba))
        except:
            rf_pred  = 1
            rf_conf  = 0.0

        try:
            xgb_pred  = xgb_model.predict(current_features)[0]
            xgb_proba = xgb_model.predict_proba(
                current_features
            )[0]
            xgb_conf  = float(np.max(xgb_proba))
        except:
            xgb_pred = 1
            xgb_conf = 0.0

        # ── Step 6: Log results ────────────────────────────
        results.append({
            'stock'            : stock_symbol,
            'predict_date'     : predict_date,
            'window_num'       : window_num,
            'actual_target'    : actual_target,
            'rf_prediction'    : rf_pred,
            'xgb_prediction'   : xgb_pred,
            'rf_correct'       : int(rf_pred == actual_target),
            'xgb_correct'      : int(xgb_pred == actual_target),
            'rf_confidence'    : round(rf_conf, 4),
            'xgb_confidence'   : round(xgb_conf, 4),
            'n_features_used'  : len(selected_features),
            'features_used'    : ','.join(selected_features),
            'top_feature'      : selected_features[0]
                                  if selected_features else ''
        })

        # Store models for next iteration
        prev_rf_model  = rf_model
        prev_xgb_model = xgb_model

        # ── Print progress every 100 windows ──────────────
        if (idx + 1) % 100 == 0:
            elapsed  = time.time() - start_time
            done_pct = (idx + 1) / total_test * 100
            rf_acc_so_far = np.mean([
                r['rf_correct'] for r in results
            ]) * 100

            print(f"  [{stock_symbol}] "
                  f"Window {idx+1:>4}/{total_test}  "
                  f"({done_pct:.0f}%)  "
                  f"RF Acc so far: {rf_acc_so_far:.1f}%  "
                  f"Time: {elapsed:.0f}s")

    # ── Convert results to DataFrame ───────────────────────
    results_df = pd.DataFrame(results)

    elapsed = time.time() - start_time
    print(f"  [{stock_symbol}] "
          f"Done. {len(results_df):,} predictions "
          f"in {elapsed:.0f}s")

    return results_df, selector


# ============================================================
# EVALUATE — Calculate metrics from results
# ============================================================

def evaluate_results(results_df, stock_symbol='ALL'):
    """
    Calculate accuracy and other metrics
    from the prediction results DataFrame.
    """

    if len(results_df) == 0:
        return {}

    y_true    = results_df['actual_target'].values
    y_rf      = results_df['rf_prediction'].values
    y_xgb     = results_df['xgb_prediction'].values

    metrics = {
        'stock'        : stock_symbol,
        'n_predictions': len(results_df),

        'rf_accuracy'  : round(
            accuracy_score(y_true, y_rf) * 100, 2),
        'rf_f1'        : round(
            f1_score(y_true, y_rf,
                     average='weighted',
                     zero_division=0) * 100, 2),
        'rf_precision' : round(
            precision_score(y_true, y_rf,
                            average='weighted',
                            zero_division=0) * 100, 2),
        'rf_recall'    : round(
            recall_score(y_true, y_rf,
                         average='weighted',
                         zero_division=0) * 100, 2),

        'xgb_accuracy' : round(
            accuracy_score(y_true, y_xgb) * 100, 2),
        'xgb_f1'       : round(
            f1_score(y_true, y_xgb,
                     average='weighted',
                     zero_division=0) * 100, 2),
        'xgb_precision': round(
            precision_score(y_true, y_xgb,
                            average='weighted',
                            zero_division=0) * 100, 2),
        'xgb_recall'   : round(
            recall_score(y_true, y_xgb,
                         average='weighted',
                         zero_division=0) * 100, 2),
    }

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    # ── Parse arguments ────────────────────────────────────
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stock',
        type    = str,
        default = 'AAPL',
        help    = 'Stock symbol or ALL'
    )
    args = parser.parse_args()

    stock_arg = args.stock.upper()

    print("\n" + "="*60)
    print("  DYNAMIC MODEL TRAINING")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    # ── Load data ──────────────────────────────────────────
    print("\n  Loading cleaned dataset...")

    if not os.path.exists(CLEAN_DATA_PATH):
        print("  ❌ cleaned_dataset.csv not found.")
        print("  Run preprocessing.py first.")
        sys.exit(1)

    df = pd.read_csv(
        CLEAN_DATA_PATH,
        parse_dates=[DATE_COL]
    )
    print(f"  ✅ Loaded {len(df):,} rows")

    # ── Decide which stocks to run ─────────────────────────
    if stock_arg == 'ALL':
        stocks_to_run = ALL_STOCKS
        print(f"  Running on ALL {len(stocks_to_run)} stocks")
        print(f"  ⚠️  This will take 30-60 minutes")
    else:
        if stock_arg not in ALL_STOCKS:
            print(f"  ❌ Stock {stock_arg} not found.")
            print(f"  Available: {ALL_STOCKS}")
            sys.exit(1)
        stocks_to_run = [stock_arg]
        print(f"  Running on: {stock_arg}")

    # ── Create output folders ──────────────────────────────
    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Run pipeline for each stock ────────────────────────
    all_results   = []
    all_metrics   = []
    total_start   = time.time()

    for i, stock in enumerate(stocks_to_run, 1):

        print(f"\n  {'─'*55}")
        print(f"  Stock {i}/{len(stocks_to_run)}: {stock}")
        print(f"  {'─'*55}")

        # Get this stock's data
        stock_df = df[df[STOCK_COL] == stock].copy()

        # Run dynamic pipeline
        results_df, selector = run_dynamic_pipeline_for_stock(
            stock_df, stock
        )

        if len(results_df) == 0:
            continue

        # Calculate metrics
        metrics = evaluate_results(results_df, stock)
        all_metrics.append(metrics)
        all_results.append(results_df)

        # Print stock summary
        print(f"\n  {stock} Results:")
        print(f"  RF  Accuracy : {metrics['rf_accuracy']}%  "
              f"F1: {metrics['rf_f1']}%")
        print(f"  XGB Accuracy : {metrics['xgb_accuracy']}%  "
              f"F1: {metrics['xgb_f1']}%")

        # Save results for this stock
        stock_log_path = os.path.join(
            LOGS_DIR, f'dynamic_{stock}.csv'
        )
        results_df.to_csv(stock_log_path, index=False)

    # ── Combine all results ────────────────────────────────
    if not all_results:
        print("\n  ❌ No results generated.")
        sys.exit(1)

    combined_df = pd.concat(all_results, ignore_index=True)
    metrics_df  = pd.DataFrame(all_metrics)

    # ── Save combined results ──────────────────────────────
    combined_path = os.path.join(
        LOGS_DIR, 'dynamic_all_results.csv'
    )
    combined_df.to_csv(combined_path, index=False)

    metrics_path = os.path.join(
        METRICS_DIR, 'dynamic_results.csv'
    )
    metrics_df.to_csv(metrics_path, index=False)

    # ── Overall summary ────────────────────────────────────
    total_elapsed = time.time() - total_start

    print("\n" + "="*60)
    print("  DYNAMIC MODEL RESULTS SUMMARY")
    print("="*60)

    # Overall metrics across all stocks
    overall = evaluate_results(combined_df, 'ALL STOCKS')

    print(f"\n  {'Metric':<20} {'Static RF':>10} "
          f"{'Dynamic RF':>11} {'Dynamic XGB':>12}")
    print(f"  {'─'*20} {'─'*10} "
          f"{'─'*11} {'─'*12}")

    # Load baseline for comparison
    baseline_path = os.path.join(
        METRICS_DIR, 'baseline_results.csv'
    )

    static_rf_acc = 'N/A'
    static_rf_f1  = 'N/A'

    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        rf_baseline = baseline_df[
            baseline_df['Model'] == 'Random Forest'
        ]
        if len(rf_baseline) > 0:
            static_rf_acc = f"{rf_baseline['Accuracy'].values[0]:.2f}%"
            static_rf_f1  = f"{rf_baseline['F1_Score'].values[0]:.2f}%"

    print(f"  {'Accuracy':<20} "
          f"{static_rf_acc:>10} "
          f"{overall['rf_accuracy']:>10.2f}% "
          f"{overall['xgb_accuracy']:>11.2f}%")

    print(f"  {'F1 Score':<20} "
          f"{static_rf_f1:>10} "
          f"{overall['rf_f1']:>10.2f}% "
          f"{overall['xgb_f1']:>11.2f}%")

    print(f"  {'Precision':<20} "
          f"{'N/A':>10} "
          f"{overall['rf_precision']:>10.2f}% "
          f"{overall['xgb_precision']:>11.2f}%")

    print(f"  {'Recall':<20} "
          f"{'N/A':>10} "
          f"{overall['rf_recall']:>10.2f}% "
          f"{overall['xgb_recall']:>11.2f}%")

    # Per stock table
    if len(metrics_df) > 1:
        print(f"\n  Per-stock results:")
        print(f"  {'Stock':<8} {'RF Acc':>8} "
              f"{'XGB Acc':>9} {'RF F1':>7} {'XGB F1':>8}")
        print(f"  {'─'*8} {'─'*8} "
              f"{'─'*9} {'─'*7} {'─'*8}")

        for _, row in metrics_df.iterrows():
            print(f"  {row['stock']:<8} "
                  f"{row['rf_accuracy']:>7.2f}% "
                  f"{row['xgb_accuracy']:>8.2f}% "
                  f"{row['rf_f1']:>6.2f}% "
                  f"{row['xgb_f1']:>7.2f}%")

    print(f"\n  Total predictions : {len(combined_df):,}")
    print(f"  Total time        : {total_elapsed:.0f}s")
    print(f"\n  Files saved:")
    print(f"  ✅ reports/logs/dynamic_all_results.csv")
    print(f"  ✅ reports/metrics/dynamic_results.csv")
    print(f"  ✅ reports/logs/dynamic_{{stock}}.csv "
          f"(one per stock)")

    # ── Improvement check ──────────────────────────────────
    print(f"\n  {'─'*55}")
    print(f"  COMPARISON: Static vs Dynamic")
    print(f"  {'─'*55}")

    if static_rf_acc != 'N/A':
        static_val  = float(static_rf_acc.replace('%',''))
        dynamic_val = overall['rf_accuracy']
        diff        = dynamic_val - static_val

        if diff > 0:
            print(f"\n  ✅ Dynamic RF BEATS Static RF")
            print(f"  Static  : {static_val:.2f}%")
            print(f"  Dynamic : {dynamic_val:.2f}%")
            print(f"  Gain    : +{diff:.2f}%")
        else:
            print(f"\n  ⚠️  Dynamic RF vs Static RF")
            print(f"  Static  : {static_val:.2f}%")
            print(f"  Dynamic : {dynamic_val:.2f}%")
            print(f"  Diff    : {diff:.2f}%")

    print("\n" + "="*60)
    print("  ✅ DYNAMIC MODEL COMPLETE")
    print("  Next step: Evaluation")
    print("  File: src/evaluation/evaluation.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()