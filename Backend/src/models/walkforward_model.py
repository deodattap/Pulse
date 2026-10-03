# ============================================================
# src/models/walkforward_model.py
#
# Walk-Forward Validation with Dynamic Feature Selection
#
# This is the CORRECT and FAIR way to compare
# static vs dynamic feature selection.
#
# How it works:
#   Both static and dynamic models predict ONE DAY AT A TIME
#   Both only use past data — no future leakage
#   Static: never changes which features it uses
#   Dynamic: recalculates feature importance each step
#   Both evaluated on identical test days
#   This is a fair apples-to-apples comparison
#
# This approach is called Walk-Forward Validation
# and is the gold standard for time series ML research
#
# How to run:
#   python src/models/walkforward_model.py --stock AAPL
#   python src/models/walkforward_model.py --stock ALL
# ============================================================

import pandas as pd
import numpy as np
import sys
import os
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble      import RandomForestClassifier
from sklearn.metrics       import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report
)
from sklearn.feature_selection import mutual_info_classif
from xgboost               import XGBClassifier
from collections           import Counter

# ── Import config ──────────────────────────────────────────
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *


# ============================================================
# SETTINGS
# ============================================================

# Minimum rows needed before we start predicting
# 3 years of data = ~756 trading days
# This ensures MI scores are stable
MIN_TRAIN_ROWS = 756

# How many recent days to use for MI calculation
# 252 = one trading year
# Enough data for stable MI scores
MI_WINDOW = 252

# How often to fully retrain the model
# 60 = retrain every 3 months
RETRAIN_EVERY = 60

# Top features to select
TOP_K = TOP_K_FEATURES  # 10 from config

# Confidence threshold
# If model is less than this confident → predict Hold
CONFIDENCE_THRESHOLD = 0.45

# Market regime settings
N_REGIMES = 3  # Bull, Bear, Sideways


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_sample_weights(y):
    """Calculate balanced sample weights for XGBoost."""
    counts    = Counter(y)
    total     = len(y)
    n_classes = len(counts)
    wmap = {
        c: total / (n_classes * cnt)
        for c, cnt in counts.items()
    }
    return np.array([wmap[label] for label in y])


def calculate_mi_scores(X, y, feature_names):
    """
    Calculate Mutual Information scores for all features.
    Uses last MI_WINDOW rows for stability.
    Returns dict of {feature: score} sorted by score.
    """
    try:
        scores = mutual_info_classif(
            X, y,
            random_state=RANDOM_STATE,
            n_neighbors=5
        )
        score_dict = dict(zip(feature_names, scores))
        return dict(sorted(
            score_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    except Exception:
        # If MI fails return equal scores
        return {f: 1.0 for f in feature_names}


def select_top_features(mi_scores, top_k=TOP_K):
    """Select top K features by MI score."""
    return list(mi_scores.keys())[:top_k]


def detect_regime(recent_data):
    """
    Simple regime detection based on recent returns.
    Returns: 'Bull', 'Bear', or 'Sideways'
    """
    if 'Daily_Return' not in recent_data.columns:
        return 'Sideways'

    returns = recent_data['Daily_Return'].tail(60)
    avg_return    = returns.mean()
    volatility    = returns.std()

    if avg_return > 0.0005 and volatility < 0.015:
        return 'Bull'
    elif avg_return < -0.0005 or volatility > 0.02:
        return 'Bear'
    else:
        return 'Sideways'


def train_model(X, y, model_type='rf'):
    """
    Train a model on given data.
    Returns trained model or None if training fails.
    """
    try:
        if len(np.unique(y)) < 2:
            return None

        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators = 100,
                class_weight = CLASS_WEIGHT,
                random_state = RANDOM_STATE,
                n_jobs       = -1,
                max_depth    = 10,
                min_samples_leaf = 5
            )
            model.fit(X, y)
            return model

        elif model_type == 'xgb':
            sw = get_sample_weights(y)
            model = XGBClassifier(
                n_estimators = 100,
                random_state = RANDOM_STATE,
                verbosity    = 0,
                eval_metric  = 'mlogloss',
                max_depth    = 6,
                learning_rate = 0.05
            )
            model.fit(X, y, sample_weight=sw)
            return model

    except Exception as e:
        return None


def predict_with_threshold(model, X_pred,
                           threshold=CONFIDENCE_THRESHOLD):
    """
    Predict with confidence threshold.
    If confidence below threshold → return Hold (2).
    This prevents over-confident wrong predictions.
    """
    try:
        proba = model.predict_proba(X_pred)[0]
        confidence = float(np.max(proba))

        if confidence >= threshold:
            return int(np.argmax(proba)), confidence
        else:
            return 2, confidence  # Hold when uncertain

    except Exception:
        return 2, 0.0  # Default to Hold on error


# ============================================================
# CORE FUNCTION — Walk Forward for ONE stock
# ============================================================

def walk_forward_one_stock(stock_df, stock_symbol):
    """
    Run walk-forward validation for one stock.

    For each day in the test period:
      1. Use all available past data for training
      2. Calculate MI on last 252 days
      3. Static model: predict using ALL features
      4. Dynamic model: predict using TOP 10 MI features
      5. Record predictions and actual values

    Parameters:
        stock_df     : DataFrame for this stock only
        stock_symbol : e.g. 'AAPL'

    Returns:
        results_df   : One row per test day
    """

    # Sort by date
    df = stock_df.sort_values(DATE_COL).reset_index(drop=True)
    total_rows = len(df)

    print(f"\n  [{stock_symbol}]")
    print(f"  Total rows     : {total_rows:,}")
    print(f"  Min train rows : {MIN_TRAIN_ROWS}")
    print(f"  Starting walk-forward validation...")

    # Storage
    results     = []
    start_time  = time.time()

    # Models — initialized as None
    # Will be trained when we have enough data
    static_rf  = None
    static_xgb = None
    dynamic_rf  = None
    dynamic_xgb = None

    # Track last MI scores and selected features
    last_mi_scores      = None
    last_static_features = FEATURE_COLS  # Never changes
    last_dynamic_features = FEATURE_COLS  # Changes each step

    # Track retrain counter
    steps_since_retrain = 0

    # ── Walk forward through ALL rows ─────────────────────
    for i in range(MIN_TRAIN_ROWS, total_rows):

        current_row   = df.iloc[i]
        predict_date  = current_row[DATE_COL]
        actual_target = int(current_row[TARGET_COL])

        # All data BEFORE current row = training data
        past_data = df.iloc[:i].copy()

        # ── Only evaluate test period ──────────────────────
        # Skip rows before TEST_START_DATE
        if predict_date < pd.Timestamp(TEST_START_DATE):
            # But still retrain periodically on train data
            steps_since_retrain += 1

            if (steps_since_retrain >= RETRAIN_EVERY
                    or static_rf is None):

                # Get MI window data for feature selection
                mi_data = past_data.tail(MI_WINDOW)
                X_mi    = mi_data[FEATURE_COLS].values
                y_mi    = mi_data[TARGET_COL].values

                # Calculate MI scores
                if len(mi_data) >= 60:
                    mi_scores = calculate_mi_scores(
                        X_mi, y_mi, FEATURE_COLS
                    )
                    last_mi_scores = mi_scores
                    last_dynamic_features = select_top_features(
                        mi_scores, TOP_K
                    )

                # Train on all past data
                X_all = past_data[FEATURE_COLS].values
                y_all = past_data[TARGET_COL].values

                # Static models use ALL features
                static_rf  = train_model(X_all, y_all, 'rf')
                static_xgb = train_model(X_all, y_all, 'xgb')

                # Dynamic models use selected features
                X_dyn = past_data[
                    last_dynamic_features
                ].values
                dynamic_rf  = train_model(X_dyn, y_all, 'rf')
                dynamic_xgb = train_model(X_dyn, y_all, 'xgb')

                steps_since_retrain = 0

            continue  # Do not record train period predictions

        # ── We are now in test period ──────────────────────
        steps_since_retrain += 1

        # ── Recalculate MI every RETRAIN_EVERY steps ──────
        should_retrain = (
            steps_since_retrain >= RETRAIN_EVERY
            or static_rf is None
        )

        if should_retrain:

            # Use last MI_WINDOW rows for stable MI
            mi_data = past_data.tail(MI_WINDOW)
            X_mi    = mi_data[FEATURE_COLS].values
            y_mi    = mi_data[TARGET_COL].values

            if len(mi_data) >= 60:
                # Recalculate MI scores
                mi_scores = calculate_mi_scores(
                    X_mi, y_mi, FEATURE_COLS
                )
                last_mi_scores        = mi_scores
                last_dynamic_features = select_top_features(
                    mi_scores, TOP_K
                )

            # Retrain on ALL available past data
            X_all = past_data[FEATURE_COLS].values
            y_all = past_data[TARGET_COL].values

            # Static: always ALL features
            new_static_rf  = train_model(X_all, y_all, 'rf')
            new_static_xgb = train_model(X_all, y_all, 'xgb')

            if new_static_rf  is not None:
                static_rf  = new_static_rf
            if new_static_xgb is not None:
                static_xgb = new_static_xgb

            # Dynamic: only selected features
            X_dyn = past_data[last_dynamic_features].values
            new_dynamic_rf  = train_model(X_dyn, y_all, 'rf')
            new_dynamic_xgb = train_model(X_dyn, y_all, 'xgb')

            if new_dynamic_rf  is not None:
                dynamic_rf  = new_dynamic_rf
            if new_dynamic_xgb is not None:
                dynamic_xgb = new_dynamic_xgb

            steps_since_retrain = 0

        # ── Skip if models not trained yet ─────────────────
        if static_rf is None or dynamic_rf is None:
            continue

        # ── Detect current market regime ───────────────────
        recent_50 = past_data.tail(50)
        regime    = detect_regime(recent_50)

        # ── Get current row features for prediction ────────
        static_features_row = current_row[
            FEATURE_COLS
        ].values.reshape(1, -1)

        dynamic_features_row = current_row[
            last_dynamic_features
        ].values.reshape(1, -1)

        # ── Static model predictions ───────────────────────
        s_rf_pred,  s_rf_conf  = predict_with_threshold(
            static_rf,  static_features_row
        )
        s_xgb_pred, s_xgb_conf = predict_with_threshold(
            static_xgb, static_features_row
        )

        # ── Dynamic model predictions ──────────────────────
        d_rf_pred,  d_rf_conf  = predict_with_threshold(
            dynamic_rf,  dynamic_features_row
        )
        d_xgb_pred, d_xgb_conf = predict_with_threshold(
            dynamic_xgb, dynamic_features_row
        )

        # ── Record result ──────────────────────────────────
        results.append({
            # Identifiers
            'stock'              : stock_symbol,
            'predict_date'       : predict_date,
            'year'               : predict_date.year,
            'actual_target'      : actual_target,
            'market_regime'      : regime,

            # Static RF
            'static_rf_pred'     : s_rf_pred,
            'static_rf_conf'     : round(s_rf_conf,  4),
            'static_rf_correct'  : int(
                s_rf_pred == actual_target
            ),

            # Static XGB
            'static_xgb_pred'    : s_xgb_pred,
            'static_xgb_conf'    : round(s_xgb_conf, 4),
            'static_xgb_correct' : int(
                s_xgb_pred == actual_target
            ),

            # Dynamic RF
            'dynamic_rf_pred'    : d_rf_pred,
            'dynamic_rf_conf'    : round(d_rf_conf,  4),
            'dynamic_rf_correct' : int(
                d_rf_pred == actual_target
            ),

            # Dynamic XGB
            'dynamic_xgb_pred'   : d_xgb_pred,
            'dynamic_xgb_conf'   : round(d_xgb_conf, 4),
            'dynamic_xgb_correct': int(
                d_xgb_pred == actual_target
            ),

            # Feature info
            'n_dynamic_features' : len(last_dynamic_features),
            'dynamic_features'   : ','.join(
                last_dynamic_features
            ),
            'top_dynamic_feature': last_dynamic_features[0]
                                    if last_dynamic_features
                                    else ''
        })

        # ── Progress update ────────────────────────────────
        n_results = len(results)
        if n_results % 100 == 0 and n_results > 0:
            elapsed = time.time() - start_time
            d_acc   = np.mean([
                r['dynamic_rf_correct'] for r in results
            ]) * 100
            s_acc   = np.mean([
                r['static_rf_correct'] for r in results
            ]) * 100
            print(f"  [{stock_symbol}] "
                  f"{n_results:>4} predictions  "
                  f"Static RF: {s_acc:.1f}%  "
                  f"Dynamic RF: {d_acc:.1f}%  "
                  f"Time: {elapsed:.0f}s")

    # ── Convert to DataFrame ───────────────────────────────
    results_df = pd.DataFrame(results)
    elapsed    = time.time() - start_time

    if len(results_df) > 0:
        s_acc = results_df['static_rf_correct'].mean() * 100
        d_acc = results_df['dynamic_rf_correct'].mean() * 100
        print(f"  [{stock_symbol}] Complete. "
              f"{len(results_df):,} predictions in "
              f"{elapsed:.0f}s")
        print(f"  [{stock_symbol}] "
              f"Static RF: {s_acc:.1f}%  "
              f"Dynamic RF: {d_acc:.1f}%  "
              f"Diff: {d_acc-s_acc:+.1f}%")

    return results_df


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(results_df, stock='ALL'):
    """Calculate comprehensive metrics from results."""

    if len(results_df) == 0:
        return {}

    y_true    = results_df['actual_target'].values
    y_s_rf    = results_df['static_rf_pred'].values
    y_s_xgb   = results_df['static_xgb_pred'].values
    y_d_rf    = results_df['dynamic_rf_pred'].values
    y_d_xgb   = results_df['dynamic_xgb_pred'].values

    def metrics(y_true, y_pred):
        return {
            'accuracy' : round(
                accuracy_score(y_true, y_pred) * 100, 2),
            'f1'       : round(
                f1_score(y_true, y_pred,
                         average='weighted',
                         zero_division=0) * 100, 2),
            'precision': round(
                precision_score(y_true, y_pred,
                                average='weighted',
                                zero_division=0) * 100, 2),
            'recall'   : round(
                recall_score(y_true, y_pred,
                             average='weighted',
                             zero_division=0) * 100, 2),
        }

    s_rf_m  = metrics(y_true, y_s_rf)
    s_xgb_m = metrics(y_true, y_s_xgb)
    d_rf_m  = metrics(y_true, y_d_rf)
    d_xgb_m = metrics(y_true, y_d_xgb)

    return {
        'stock'              : stock,
        'n_predictions'      : len(results_df),

        'static_rf_acc'      : s_rf_m['accuracy'],
        'static_rf_f1'       : s_rf_m['f1'],
        'static_rf_prec'     : s_rf_m['precision'],
        'static_rf_rec'      : s_rf_m['recall'],

        'static_xgb_acc'     : s_xgb_m['accuracy'],
        'static_xgb_f1'      : s_xgb_m['f1'],
        'static_xgb_prec'    : s_xgb_m['precision'],
        'static_xgb_rec'     : s_xgb_m['recall'],

        'dynamic_rf_acc'     : d_rf_m['accuracy'],
        'dynamic_rf_f1'      : d_rf_m['f1'],
        'dynamic_rf_prec'    : d_rf_m['precision'],
        'dynamic_rf_rec'     : d_rf_m['recall'],

        'dynamic_xgb_acc'    : d_xgb_m['accuracy'],
        'dynamic_xgb_f1'     : d_xgb_m['f1'],
        'dynamic_xgb_prec'   : d_xgb_m['precision'],
        'dynamic_xgb_rec'    : d_xgb_m['recall'],

        'rf_acc_improvement' : round(
            d_rf_m['accuracy'] - s_rf_m['accuracy'], 2),
        'rf_f1_improvement'  : round(
            d_rf_m['f1'] - s_rf_m['f1'], 2),
    }


# ============================================================
# PRINT DETAILED SUMMARY
# ============================================================

def print_summary(results_df, metrics, stock_symbol):
    """Print a detailed readable summary."""

    print(f"\n  {'═'*55}")
    print(f"  RESULTS: {stock_symbol}")
    print(f"  {'═'*55}")

    print(f"\n  {'Metric':<20} {'Static RF':>10} "
          f"{'Static XGB':>11} {'Dynamic RF':>11} "
          f"{'Dynamic XGB':>12}")
    print(f"  {'─'*20} {'─'*10} "
          f"{'─'*11} {'─'*11} {'─'*12}")

    print(f"  {'Accuracy':<20} "
          f"{metrics['static_rf_acc']:>9.2f}% "
          f"{metrics['static_xgb_acc']:>10.2f}% "
          f"{metrics['dynamic_rf_acc']:>10.2f}% "
          f"{metrics['dynamic_xgb_acc']:>11.2f}%")

    print(f"  {'F1 Score':<20} "
          f"{metrics['static_rf_f1']:>9.2f}% "
          f"{metrics['static_xgb_f1']:>10.2f}% "
          f"{metrics['dynamic_rf_f1']:>10.2f}% "
          f"{metrics['dynamic_xgb_f1']:>11.2f}%")

    print(f"  {'Precision':<20} "
          f"{metrics['static_rf_prec']:>9.2f}% "
          f"{metrics['static_xgb_prec']:>10.2f}% "
          f"{metrics['dynamic_rf_prec']:>10.2f}% "
          f"{metrics['dynamic_xgb_prec']:>11.2f}%")

    print(f"  {'Recall':<20} "
          f"{metrics['static_rf_rec']:>9.2f}% "
          f"{metrics['static_xgb_rec']:>10.2f}% "
          f"{metrics['dynamic_rf_rec']:>10.2f}% "
          f"{metrics['dynamic_xgb_rec']:>11.2f}%")

    # Year by year
    if 'year' in results_df.columns:
        print(f"\n  Year-by-Year RF Accuracy:")
        print(f"  {'Year':<6} {'Static RF':>10} "
              f"{'Dynamic RF':>11} {'Winner':>8}")
        print(f"  {'─'*6} {'─'*10} {'─'*11} {'─'*8}")

        for year in sorted(results_df['year'].unique()):
            yr = results_df[results_df['year'] == year]
            s_acc = yr['static_rf_correct'].mean() * 100
            d_acc = yr['dynamic_rf_correct'].mean() * 100
            winner = 'Dynamic ✅' if d_acc > s_acc \
                     else 'Static'
            print(f"  {year:<6} {s_acc:>9.2f}% "
                  f"{d_acc:>10.2f}% {winner:>8}")

    # Regime analysis
    if 'market_regime' in results_df.columns:
        print(f"\n  Accuracy by Market Regime:")
        print(f"  {'Regime':<10} {'Static RF':>10} "
              f"{'Dynamic RF':>11} {'Winner':>8}")
        print(f"  {'─'*10} {'─'*10} {'─'*11} {'─'*8}")

        for regime in ['Bull', 'Bear', 'Sideways']:
            rdf = results_df[
                results_df['market_regime'] == regime
            ]
            if len(rdf) == 0:
                continue
            s_acc = rdf['static_rf_correct'].mean() * 100
            d_acc = rdf['dynamic_rf_correct'].mean() * 100
            winner = 'Dynamic ✅' if d_acc > s_acc \
                     else 'Static'
            print(f"  {regime:<10} {s_acc:>9.2f}% "
                  f"{d_acc:>10.2f}% {winner:>8}")

    # Prediction distribution
    print(f"\n  Prediction Distribution:")
    print(f"  {'Class':<8} {'Actual':>8} "
          f"{'Static RF':>10} {'Dynamic RF':>11}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*11}")

    total = len(results_df)
    for val, label in [(0,'Sell'),(1,'Buy'),(2,'Hold')]:
        actual_pct  = (
            results_df['actual_target'] == val
        ).sum() / total * 100
        static_pct  = (
            results_df['static_rf_pred'] == val
        ).sum() / total * 100
        dynamic_pct = (
            results_df['dynamic_rf_pred'] == val
        ).sum() / total * 100
        print(f"  {label:<8} {actual_pct:>7.1f}% "
              f"{static_pct:>9.1f}% "
              f"{dynamic_pct:>10.1f}%")

    # Improvement
    rf_diff = metrics['rf_acc_improvement']
    f1_diff = metrics['rf_f1_improvement']

    print(f"\n  RF Accuracy improvement : {rf_diff:+.2f}%")
    print(f"  RF F1 improvement       : {f1_diff:+.2f}%")

    if rf_diff > 0:
        print(f"  ✅ Dynamic RF BEATS Static RF")
    else:
        print(f"  ⚠️  Static RF leads by "
              f"{abs(rf_diff):.2f}%")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--stock',
        type    = str,
        default = 'AAPL',
        help    = 'Stock symbol or ALL'
    )
    args      = parser.parse_args()
    stock_arg = args.stock.upper()

    print("\n" + "="*60)
    print("  WALK-FORWARD VALIDATION")
    print("  Static vs Dynamic Feature Selection")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    print(f"\n  Settings:")
    print(f"  Min training rows : {MIN_TRAIN_ROWS} "
          f"(~3 years)")
    print(f"  MI window         : {MI_WINDOW} days "
          f"(~1 year)")
    print(f"  Retrain every     : {RETRAIN_EVERY} days "
          f"(~3 months)")
    print(f"  Top K features    : {TOP_K}")
    print(f"  Confidence thresh : {CONFIDENCE_THRESHOLD}")
    print(f"  Test period       : {TEST_START_DATE} onwards")

    # ── Load data ──────────────────────────────────────────
    print(f"\n  Loading dataset...")

    if not os.path.exists(CLEAN_DATA_PATH):
        print("  ❌ cleaned_dataset.csv not found.")
        print("  Run preprocessing.py first.")
        sys.exit(1)

    df = pd.read_csv(
        CLEAN_DATA_PATH,
        parse_dates=[DATE_COL]
    )
    print(f"  ✅ Loaded {len(df):,} rows")

    # ── Stocks to run ──────────────────────────────────────
    if stock_arg == 'ALL':
        stocks = ALL_STOCKS
        print(f"\n  Running on ALL {len(stocks)} stocks")
        print(f"  ⚠️  Expected time: 45-90 minutes")
        print(f"  Do not interrupt the run")
    else:
        if stock_arg not in ALL_STOCKS:
            print(f"  ❌ {stock_arg} not in dataset")
            sys.exit(1)
        stocks = [stock_arg]
        print(f"\n  Running on: {stock_arg}")
        print(f"  Expected time: 3-8 minutes")

    # ── Create output folders ──────────────────────────────
    os.makedirs(LOGS_DIR,    exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Run for each stock ─────────────────────────────────
    all_results = []
    all_metrics = []
    total_start = time.time()

    for i, stock in enumerate(stocks, 1):

        print(f"\n  {'─'*55}")
        print(f"  Stock {i}/{len(stocks)}: {stock}")
        print(f"  {'─'*55}")

        stock_df = df[df[STOCK_COL] == stock].copy()

        # Run walk-forward
        results_df = walk_forward_one_stock(
            stock_df, stock
        )

        if len(results_df) == 0:
            print(f"  ⚠️  No results for {stock}")
            continue

        # Calculate metrics
        m = calculate_metrics(results_df, stock)
        all_metrics.append(m)
        all_results.append(results_df)

        # Print summary
        print_summary(results_df, m, stock)

        # Save per-stock results
        stock_path = os.path.join(
            LOGS_DIR, f'walkforward_{stock}.csv'
        )
        results_df.to_csv(stock_path, index=False)
        print(f"\n  ✅ Saved: reports/logs/"
              f"walkforward_{stock}.csv")

        # Save running combined metrics
        # (so you have partial results if interrupted)
        pd.DataFrame(all_metrics).to_csv(
            os.path.join(
                METRICS_DIR, 'walkforward_results.csv'
            ),
            index=False
        )

    # ── Final combined results ─────────────────────────────
    if not all_results:
        print("\n  ❌ No results generated.")
        sys.exit(1)

    combined_df = pd.concat(all_results, ignore_index=True)
    metrics_df  = pd.DataFrame(all_metrics)

    combined_df.to_csv(
        os.path.join(
            LOGS_DIR, 'walkforward_ALL.csv'
        ),
        index=False
    )
    metrics_df.to_csv(
        os.path.join(
            METRICS_DIR, 'walkforward_results.csv'
        ),
        index=False
    )

    # ── Overall summary ────────────────────────────────────
    total_elapsed = time.time() - total_start
    overall_m     = calculate_metrics(combined_df, 'ALL')

    print("\n" + "="*60)
    print("  OVERALL SUMMARY — ALL STOCKS")
    print("="*60)

    print_summary(combined_df, overall_m, 'ALL STOCKS')

    print(f"\n  Total predictions : {len(combined_df):,}")
    print(f"  Total time        : "
          f"{total_elapsed/60:.1f} minutes")

    print(f"\n  Files saved:")
    print(f"  ✅ reports/logs/walkforward_ALL.csv")
    print(f"  ✅ reports/metrics/walkforward_results.csv")
    print(f"  ✅ reports/logs/walkforward_{{stock}}.csv "
          f"(one per stock)")

    print("\n" + "="*60)
    print("  ✅ WALK-FORWARD VALIDATION COMPLETE")
    print("  Next: src/evaluation/evaluation.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()