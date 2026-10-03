# ============================================================
# src/models/dynamic_model.py
#
# CORRECTED VERSION — Warm Start Incremental Learning
#
# Strategy:
# Phase 1: Pre-train on all training data (stable foundation)
# Phase 2: For each test window:
#           - Select features dynamically using MI
#           - Update model using recent 252 days of data
#           - Predict next day signal
#
# This is the correct approach for streaming ML.
# Reference: "Online Learning with Concept Drift" literature
#
# How to run:
# python src/models/dynamic_model.py --stock AAPL
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
from xgboost           import XGBClassifier
from collections       import Counter

# ── Import config ──────────────────────────────────────────
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *

# ── Import modules ─────────────────────────────────────────
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'features')
)
from sliding_window            import sliding_window_generator
from dynamic_feature_selection import DynamicFeatureSelector


# ============================================================
# SETTINGS
# ============================================================

# How many recent days to use for model update
# 252 = approximately one trading year
# More data = more stable model
RETRAIN_WINDOW = 252

# How often to fully retrain the model (in windows)
# Every 20 windows = full retrain
# In between = use existing model
RETRAIN_FREQUENCY = 20


# ============================================================
# HELPER
# ============================================================

def get_sample_weights(y):
    counts    = Counter(y)
    total     = len(y)
    n_classes = len(counts)
    weights_map = {
        cls: total / (n_classes * cnt)
        for cls, cnt in counts.items()
    }
    return np.array([weights_map[lbl] for lbl in y])


# ============================================================
# CORE — Run dynamic pipeline for ONE stock
# ============================================================

def run_dynamic_pipeline_for_stock(full_stock_df,
                                   stock_symbol):
    """
    Corrected dynamic pipeline using warm start strategy.

    Phase 1: Pre-train on all data before TEST_START_DATE
    Phase 2: For each test window — dynamic feature selection
             + incremental model update every RETRAIN_FREQUENCY
    """

    # Sort by date
    df = full_stock_df.sort_values(DATE_COL).reset_index(
        drop=True
    )

    # Split into train and test periods
    train_df = df[df[DATE_COL] <= TRAIN_END_DATE].copy()
    test_df  = df[df[DATE_COL] >= TEST_START_DATE].copy()
    test_df  = test_df.reset_index(drop=True)

    if len(train_df) < WINDOW_SIZE:
        print(f"  ⚠️  Not enough training data for "
              f"{stock_symbol}")
        return pd.DataFrame(), None

    if len(test_df) < 2:
        print(f"  ⚠️  Not enough test data for "
              f"{stock_symbol}")
        return pd.DataFrame(), None

    # ── Phase 1: Pre-train on all training data ────────────
    # Use ALL features for pre-training
    # This gives the model a solid foundation
    X_pretrain = train_df[FEATURE_COLS].values
    y_pretrain = train_df[TARGET_COL].values

    # Pre-train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators = N_ESTIMATORS,
        class_weight = CLASS_WEIGHT,
        random_state = RANDOM_STATE,
        n_jobs       = -1,
        max_depth    = 10
    )
    rf_model.fit(X_pretrain, y_pretrain)

    # Pre-train XGBoost
    xgb_model = XGBClassifier(
        n_estimators = N_ESTIMATORS,
        random_state = RANDOM_STATE,
        verbosity    = 0,
        eval_metric  = 'mlogloss',
        max_depth    = 5
    )
    sw = get_sample_weights(y_pretrain)
    xgb_model.fit(X_pretrain, y_pretrain,
                  sample_weight=sw)

    # ── Create dynamic feature selector ───────────────────
    selector = DynamicFeatureSelector()

    # ── Phase 2: Test window loop ──────────────────────────
    results    = []
    total_rows = len(test_df)
    start_time = time.time()

    # We combine train + test for the rolling window
    # So we can look back RETRAIN_WINDOW days from test start
    full_df = df.reset_index(drop=True)

    print(f"  [{stock_symbol}] "
          f"Pre-training done. "
          f"Test rows: {total_rows}  "
          f"Starting dynamic phase...")

    for i in range(total_rows):

        current_row  = test_df.iloc[i]
        predict_date = current_row[DATE_COL]
        actual_target = current_row[TARGET_COL]
        window_num   = i + 1

        # ── Get recent data for this point ────────────────
        # Look back RETRAIN_WINDOW rows from current position
        current_idx = full_df.index[
            full_df[DATE_COL] == predict_date
        ]

        if len(current_idx) == 0:
            continue

        current_idx = current_idx[0]

        # Get last RETRAIN_WINDOW rows before current date
        start_idx = max(0, current_idx - RETRAIN_WINDOW)
        recent_data = full_df.iloc[start_idx:current_idx]

        if len(recent_data) < 30:
            # Not enough data yet
            continue

        # ── Dynamic feature selection ──────────────────────
        selected_features = selector.select(
            recent_data, window_num
        )

        # ── Retrain every RETRAIN_FREQUENCY windows ────────
        if window_num % RETRAIN_FREQUENCY == 0:

            X_recent = recent_data[selected_features].values
            y_recent = recent_data[TARGET_COL].values

            unique_classes = np.unique(y_recent)
            if len(unique_classes) >= 2:

                # Retrain RF
                try:
                    rf_model = RandomForestClassifier(
                        n_estimators = 50,
                        class_weight = CLASS_WEIGHT,
                        random_state = RANDOM_STATE,
                        n_jobs       = -1,
                        max_depth    = 8
                    )
                    rf_model.fit(X_recent, y_recent)
                except:
                    pass  # Keep existing model

                # Retrain XGB
                try:
                    sw = get_sample_weights(y_recent)
                    xgb_model = XGBClassifier(
                        n_estimators = 50,
                        random_state = RANDOM_STATE,
                        verbosity    = 0,
                        eval_metric  = 'mlogloss',
                        max_depth    = 4
                    )
                    xgb_model.fit(
                        X_recent, y_recent,
                        sample_weight=sw
                    )
                except:
                    pass  # Keep existing model

        # ── Predict current row ────────────────────────────
        current_features_vals = current_row[
            selected_features
        ].values.reshape(1, -1)

        try:
            rf_pred  = rf_model.predict(
                current_features_vals
            )[0]
            rf_proba = rf_model.predict_proba(
                current_features_vals
            )[0]
            rf_conf  = float(np.max(rf_proba))
        except:
            rf_pred = 1
            rf_conf = 0.0

        try:
            xgb_pred  = xgb_model.predict(
                current_features_vals
            )[0]
            xgb_proba = xgb_model.predict_proba(
                current_features_vals
            )[0]
            xgb_conf  = float(np.max(xgb_proba))
        except:
            xgb_pred = 1
            xgb_conf = 0.0

        # ── Log result ─────────────────────────────────────
        results.append({
            'stock'           : stock_symbol,
            'predict_date'    : predict_date,
            'window_num'      : window_num,
            'actual_target'   : actual_target,
            'rf_prediction'   : rf_pred,
            'xgb_prediction'  : xgb_pred,
            'rf_correct'      : int(rf_pred == actual_target),
            'xgb_correct'     : int(xgb_pred == actual_target),
            'rf_confidence'   : round(rf_conf, 4),
            'xgb_confidence'  : round(xgb_conf, 4),
            'n_features_used' : len(selected_features),
            'features_used'   : ','.join(selected_features),
            'top_feature'     : selected_features[0]
                                 if selected_features else ''
        })

        # ── Progress update ────────────────────────────────
        if window_num % 100 == 0:
            elapsed      = time.time() - start_time
            pct          = window_num / total_rows * 100
            rf_acc_so_far = np.mean([
                r['rf_correct'] for r in results
            ]) * 100

            print(f"  [{stock_symbol}] "
                  f"Window {window_num:>4}/{total_rows}  "
                  f"({pct:.0f}%)  "
                  f"RF Acc: {rf_acc_so_far:.1f}%  "
                  f"Time: {elapsed:.0f}s")

    results_df = pd.DataFrame(results)
    elapsed    = time.time() - start_time

    print(f"  [{stock_symbol}] "
          f"Done. {len(results_df):,} predictions "
          f"in {elapsed:.0f}s")

    return results_df, selector


# ============================================================
# EVALUATE
# ============================================================

def evaluate_results(results_df, stock_symbol='ALL'):

    if len(results_df) == 0:
        return {}

    y_true = results_df['actual_target'].values
    y_rf   = results_df['rf_prediction'].values
    y_xgb  = results_df['xgb_prediction'].values

    return {
        'stock'         : stock_symbol,
        'n_predictions' : len(results_df),
        'rf_accuracy'   : round(
            accuracy_score(y_true, y_rf) * 100, 2),
        'rf_f1'         : round(
            f1_score(y_true, y_rf,
                     average='weighted',
                     zero_division=0) * 100, 2),
        'rf_precision'  : round(
            precision_score(y_true, y_rf,
                            average='weighted',
                            zero_division=0) * 100, 2),
        'rf_recall'     : round(
            recall_score(y_true, y_rf,
                         average='weighted',
                         zero_division=0) * 100, 2),
        'xgb_accuracy'  : round(
            accuracy_score(y_true, y_xgb) * 100, 2),
        'xgb_f1'        : round(
            f1_score(y_true, y_xgb,
                     average='weighted',
                     zero_division=0) * 100, 2),
        'xgb_precision' : round(
            precision_score(y_true, y_xgb,
                            average='weighted',
                            zero_division=0) * 100, 2),
        'xgb_recall'    : round(
            recall_score(y_true, y_xgb,
                         average='weighted',
                         zero_division=0) * 100, 2),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stock', type=str, default='AAPL'
    )
    args      = parser.parse_args()
    stock_arg = args.stock.upper()

    print("\n" + "="*60)
    print("  DYNAMIC MODEL — WARM START VERSION")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    # ── Load data ──────────────────────────────────────────
    print("\n  Loading cleaned dataset...")

    if not os.path.exists(CLEAN_DATA_PATH):
        print("  ❌ cleaned_dataset.csv not found.")
        sys.exit(1)

    df = pd.read_csv(
        CLEAN_DATA_PATH,
        parse_dates=[DATE_COL]
    )
    print(f"  ✅ Loaded {len(df):,} rows")

    # ── Stocks to run ──────────────────────────────────────
    if stock_arg == 'ALL':
        stocks_to_run = ALL_STOCKS
        print(f"  Running on ALL {len(ALL_STOCKS)} stocks")
        print(f"  ⚠️  Expected time: 60-90 minutes")
    else:
        if stock_arg not in ALL_STOCKS:
            print(f"  ❌ {stock_arg} not found.")
            sys.exit(1)
        stocks_to_run = [stock_arg]
        print(f"  Running on: {stock_arg}")
        print(f"  Strategy: Pre-train on 2015-2021, "
              f"dynamic update on 2022-2025")

    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    all_results = []
    all_metrics = []
    total_start = time.time()

    for i, stock in enumerate(stocks_to_run, 1):

        print(f"\n  {'─'*55}")
        print(f"  [{i}/{len(stocks_to_run)}] {stock}")
        print(f"  {'─'*55}")

        stock_df = df[df[STOCK_COL] == stock].copy()

        results_df, selector = \
            run_dynamic_pipeline_for_stock(
                stock_df, stock
            )

        if len(results_df) == 0:
            continue

        metrics = evaluate_results(results_df, stock)
        all_metrics.append(metrics)
        all_results.append(results_df)

        print(f"\n  {stock} Summary:")
        print(f"  RF  → Acc: {metrics['rf_accuracy']}%  "
              f"F1: {metrics['rf_f1']}%")
        print(f"  XGB → Acc: {metrics['xgb_accuracy']}%  "
              f"F1: {metrics['xgb_f1']}%")

        # Save per-stock results
        stock_path = os.path.join(
            LOGS_DIR, f'dynamic_{stock}.csv'
        )
        results_df.to_csv(stock_path, index=False)

    if not all_results:
        print("\n  ❌ No results.")
        sys.exit(1)

    combined_df = pd.concat(all_results, ignore_index=True)
    metrics_df  = pd.DataFrame(all_metrics)

    combined_df.to_csv(
        os.path.join(LOGS_DIR, 'dynamic_all_results.csv'),
        index=False
    )
    metrics_df.to_csv(
        os.path.join(METRICS_DIR, 'dynamic_results.csv'),
        index=False
    )

    # ── Final summary ──────────────────────────────────────
    overall       = evaluate_results(combined_df, 'ALL')
    total_elapsed = time.time() - total_start

    print("\n" + "="*60)
    print("  FINAL RESULTS SUMMARY")
    print("="*60)

    # Load baseline for comparison
    baseline_path = os.path.join(
        METRICS_DIR, 'baseline_results.csv'
    )
    static_rf_acc = None

    if os.path.exists(baseline_path):
        bl_df = pd.read_csv(baseline_path)
        rf_bl = bl_df[bl_df['Model'] == 'Random Forest']
        if len(rf_bl) > 0:
            static_rf_acc = rf_bl['Accuracy'].values[0]

    print(f"\n  {'Metric':<20} {'Static RF':>10} "
          f"{'Dynamic RF':>11} {'Dynamic XGB':>12}")
    print(f"  {'─'*20} {'─'*10} {'─'*11} {'─'*12}")

    static_str = (f"{static_rf_acc:.2f}%"
                  if static_rf_acc else 'N/A')

    print(f"  {'Accuracy':<20} {static_str:>10} "
          f"{overall['rf_accuracy']:>10.2f}% "
          f"{overall['xgb_accuracy']:>11.2f}%")
    print(f"  {'F1 Score':<20} {'N/A':>10} "
          f"{overall['rf_f1']:>10.2f}% "
          f"{overall['xgb_f1']:>11.2f}%")
    print(f"  {'Precision':<20} {'N/A':>10} "
          f"{overall['rf_precision']:>10.2f}% "
          f"{overall['xgb_precision']:>11.2f}%")
    print(f"  {'Recall':<20} {'N/A':>10} "
          f"{overall['rf_recall']:>10.2f}% "
          f"{overall['xgb_recall']:>11.2f}%")

    print(f"\n  Total predictions : {len(combined_df):,}")
    print(f"  Total time        : "
          f"{total_elapsed/60:.1f} minutes")

    # Improvement check
    if static_rf_acc:
        diff = overall['rf_accuracy'] - static_rf_acc
        print(f"\n  {'─'*55}")
        if diff > 0:
            print(f"  ✅ Dynamic BEATS Static by +{diff:.2f}%")
        else:
            print(f"  ⚠️  Dynamic vs Static: {diff:.2f}%")
            print(f"  Check evaluation.py for detailed "
                  f"regime-wise analysis")
        print(f"  {'─'*55}")

    print(f"\n  ✅ Files saved to reports/")
    print("\n" + "="*60)
    print("  ✅ DYNAMIC MODEL COMPLETE")
    print("  Next: src/evaluation/evaluation.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()