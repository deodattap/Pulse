# ============================================================
# src/models/baseline_model.py
#
# Purpose:
# Build the Static Baseline Model.
#
# This model:
# - Trains ONCE on all 35 features
# - Never updates as new data arrives
# - Never changes which features it uses
# - Represents the traditional approach
#
# The accuracy from this file = LEFT COLUMN
# in your Static vs Dynamic comparison table.
#
# How to run:
# python src/models/baseline_model.py
#
# Output:
# reports/metrics/baseline_results.csv
# reports/models/baseline_rf.pkl
# reports/models/baseline_xgb.pkl
# reports/models/baseline_lr.pkl
# ============================================================

import pandas as pd
import numpy as np
import sys
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble      import RandomForestClassifier
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# ── Import config ──────────────────────────────────────────
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *


# ============================================================
# STEP 1 — Load Train and Test Data
# ============================================================

def step1_load():
    print("\n" + "="*60)
    print("STEP 1: Loading train and test data")
    print("="*60)

    # Check files exist
    for path, name in [
        (TRAIN_DATA_PATH, 'train_dataset.csv'),
        (TEST_DATA_PATH,  'test_dataset.csv')
    ]:
        if not os.path.exists(path):
            print(f"  ❌ {name} not found.")
            print(f"  Run preprocessing.py first.")
            sys.exit(1)

    # Load
    train_df = pd.read_csv(
        TRAIN_DATA_PATH,
        parse_dates=[DATE_COL]
    )
    test_df  = pd.read_csv(
        TEST_DATA_PATH,
        parse_dates=[DATE_COL]
    )

    print(f"  ✅ train_dataset.csv loaded : {len(train_df):,} rows")
    print(f"  ✅ test_dataset.csv loaded  : {len(test_df):,} rows")

    return train_df, test_df


# ============================================================
# STEP 2 — Prepare Features and Target
# ============================================================

def step2_prepare(train_df, test_df):
    print("\n" + "="*60)
    print("STEP 2: Preparing features and target")
    print("="*60)

    # ── Check all feature columns exist ───────────────────
    missing = [c for c in FEATURE_COLS
               if c not in train_df.columns]
    if missing:
        print(f"  ❌ Missing feature columns: {missing}")
        sys.exit(1)

    # ── Extract X (features) and y (target) ───────────────
    X_train = train_df[FEATURE_COLS].values
    y_train = train_df[TARGET_COL].values

    X_test  = test_df[FEATURE_COLS].values
    y_test  = test_df[TARGET_COL].values

    print(f"  Features used  : {len(FEATURE_COLS)} (ALL features)")
    print(f"  X_train shape  : {X_train.shape}")
    print(f"  X_test shape   : {X_test.shape}")
    print(f"  y_train unique : {np.unique(y_train)}")
    print(f"  y_test unique  : {np.unique(y_test)}")

    # ── Show what ALL features means ───────────────────────
    print(f"\n  NOTE: Baseline uses ALL {len(FEATURE_COLS)} features.")
    print(f"  Features NEVER change for baseline.")
    print(f"  This is the static traditional approach.")
    print(f"  Dynamic model will select only TOP {TOP_K_FEATURES} per window.")

    # ── Scale features for Logistic Regression ─────────────
    # Random Forest and XGBoost don't need scaling.
    # Logistic Regression does need scaling.
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"\n  Scaling done for Logistic Regression ✅")
    print(f"  (RF and XGBoost use unscaled data)")

    return (X_train, y_train,
            X_test,  y_test,
            X_train_scaled, X_test_scaled,
            scaler)


# ============================================================
# STEP 3 — Train All 3 Models
# ============================================================

def step3_train(X_train, y_train,
                X_train_scaled):
    print("\n" + "="*60)
    print("STEP 3: Training baseline models")
    print("="*60)
    print(f"  Training on {len(X_train):,} rows")
    print(f"  Using ALL {len(FEATURE_COLS)} features (static)")
    print(f"  class_weight=balanced to handle imbalance\n")

    models = {}

    # ── Model 1: Logistic Regression ──────────────────────
    print(f"  [1/3] Logistic Regression...")
    lr = LogisticRegression(
        class_weight  = CLASS_WEIGHT,
        random_state  = RANDOM_STATE,
        max_iter      = 1000,
        solver        = 'lbfgs',
    )
    lr.fit(X_train_scaled, y_train)
    models['Logistic Regression'] = lr
    print(f"        ✅ Done")

    # ── Model 2: Random Forest ─────────────────────────────
    print(f"  [2/3] Random Forest...")
    rf = RandomForestClassifier(
        n_estimators  = N_ESTIMATORS,
        class_weight  = CLASS_WEIGHT,
        random_state  = RANDOM_STATE,
        n_jobs        = -1        # Use all CPU cores
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    print(f"        ✅ Done")

    # ── Model 3: XGBoost ──────────────────────────────────
    print(f"  [3/3] XGBoost...")

    # XGBoost needs numeric class weights
    # Calculate manually since it doesn't take 'balanced'
    from collections import Counter
    counts    = Counter(y_train)
    total     = len(y_train)
    n_classes = len(counts)

    # Weight = total / (n_classes * count_of_class)
    weights_map = {
        cls: total / (n_classes * cnt)
        for cls, cnt in counts.items()
    }

    # Create sample weights array
    sample_weights = np.array([
        weights_map[label] for label in y_train
    ])

    xgb = XGBClassifier(
        n_estimators  = N_ESTIMATORS,
        random_state  = RANDOM_STATE,
        use_label_encoder = False,
        eval_metric   = 'mlogloss',
        verbosity     = 0
    )
    xgb.fit(X_train, y_train,
            sample_weight=sample_weights)
    models['XGBoost'] = xgb
    print(f"        ✅ Done")

    print(f"\n  All 3 models trained ✅")
    return models


# ============================================================
# STEP 4 — Evaluate All Models
# ============================================================

def step4_evaluate(models, test_df,
                   X_test, y_test,
                   X_test_scaled):
    print("\n" + "="*60)
    print("STEP 4: Evaluating on test data (2022-2025)")
    print("="*60)
    print(f"  Evaluating on {len(X_test):,} rows\n")

    results = []

    for name, model in models.items():

        # Use scaled data for LR, unscaled for RF and XGB
        if name == 'Logistic Regression':
            X_eval = X_test_scaled
        else:
            X_eval = X_test

        # Predict
        y_pred = model.predict(X_eval)

        # Calculate metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(
            y_test, y_pred,
            average='weighted',
            zero_division=0
        )
        rec  = recall_score(
            y_test, y_pred,
            average='weighted',
            zero_division=0
        )
        f1   = f1_score(
            y_test, y_pred,
            average='weighted',
            zero_division=0
        )

        results.append({
            'Model'     : name,
            'Type'      : 'Static Baseline',
            'Accuracy'  : round(acc  * 100, 2),
            'Precision' : round(prec * 100, 2),
            'Recall'    : round(rec  * 100, 2),
            'F1_Score'  : round(f1   * 100, 2),
            'Features'  : len(FEATURE_COLS),
            'Train_Rows': 'All (static)',
            'Test_Rows' : len(y_test)
        })

        # Print results for this model
        bar_acc = '█' * int(acc * 20)
        print(f"  {'─'*50}")
        print(f"  Model     : {name}")
        print(f"  Accuracy  : {acc*100:.2f}%  {bar_acc}")
        print(f"  Precision : {prec*100:.2f}%")
        print(f"  Recall    : {rec*100:.2f}%")
        print(f"  F1 Score  : {f1*100:.2f}%")

    return results


# ============================================================
# STEP 5 — Per Class Performance
# ============================================================

def step5_per_class(models, X_test, y_test,
                    X_test_scaled):
    print("\n" + "="*60)
    print("STEP 5: Per-class performance")
    print("="*60)
    print("  (Shows performance for Sell/Buy/Hold separately)\n")

    for name, model in models.items():

        if name == 'Logistic Regression':
            X_eval = X_test_scaled
        else:
            X_eval = X_test

        y_pred = model.predict(X_eval)

        print(f"  {name}:")
        print(f"  {'─'*45}")

        report = classification_report(
            y_test, y_pred,
            target_names=['Sell', 'Buy', 'Hold'],
            zero_division=0
        )

        # Print each line with indentation
        for line in report.strip().split('\n'):
            print(f"  {line}")
        print()


# ============================================================
# STEP 6 — Per Stock Performance
# ============================================================

def step6_per_stock(models, test_df,
                    X_test_scaled):
    print("\n" + "="*60)
    print("STEP 6: Per-stock accuracy (Random Forest)")
    print("="*60)

    rf_model = models['Random Forest']

    print(f"  {'Stock':<8} {'Rows':>5}  "
          f"{'Accuracy':>9}  {'F1':>6}")
    print(f"  {'─'*8} {'─'*5}  {'─'*9}  {'─'*6}")

    stock_results = []

    for stock in sorted(test_df[STOCK_COL].unique()):
        s_df = test_df[test_df[STOCK_COL] == stock]
        X_s  = s_df[FEATURE_COLS].values
        y_s  = s_df[TARGET_COL].values

        y_pred = rf_model.predict(X_s)
        acc    = accuracy_score(y_s, y_pred) * 100
        f1     = f1_score(
            y_s, y_pred,
            average='weighted',
            zero_division=0
        ) * 100

        stock_results.append({
            'Stock'    : stock,
            'Rows'     : len(s_df),
            'Accuracy' : round(acc, 2),
            'F1'       : round(f1,  2)
        })

        print(f"  {stock:<8} {len(s_df):>5}  "
              f"{acc:>8.2f}%  {f1:>5.2f}%")

    avg_acc = np.mean([r['Accuracy'] for r in stock_results])
    avg_f1  = np.mean([r['F1']      for r in stock_results])
    print(f"  {'─'*8} {'─'*5}  {'─'*9}  {'─'*6}")
    print(f"  {'AVERAGE':<8} {'':>5}  "
          f"{avg_acc:>8.2f}%  {avg_f1:>5.2f}%")

    return stock_results


# ============================================================
# STEP 7 — Save Results and Models
# ============================================================

def step7_save(results, models, scaler, stock_results):
    print("\n" + "="*60)
    print("STEP 7: Saving results and models")
    print("="*60)

    # Create output folders
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR,  exist_ok=True)

    # ── Save overall metrics ───────────────────────────────
    results_df = pd.DataFrame(results)
    metrics_path = os.path.join(
        METRICS_DIR, 'baseline_results.csv'
    )
    results_df.to_csv(metrics_path, index=False)
    print(f"\n  ✅ baseline_results.csv saved")
    print(f"     Path: {metrics_path}")

    # ── Save per-stock results ─────────────────────────────
    stock_df = pd.DataFrame(stock_results)
    stock_path = os.path.join(
        METRICS_DIR, 'baseline_per_stock.csv'
    )
    stock_df.to_csv(stock_path, index=False)
    print(f"\n  ✅ baseline_per_stock.csv saved")

    # ── Save trained models ────────────────────────────────
    model_files = {
        'Logistic Regression' : 'baseline_lr.pkl',
        'Random Forest'       : 'baseline_rf.pkl',
        'XGBoost'             : 'baseline_xgb.pkl',
    }

    for name, filename in model_files.items():
        model_path = os.path.join(MODELS_DIR, filename)
        joblib.dump(models[name], model_path)
        print(f"  ✅ {filename} saved")

    # ── Save scaler ────────────────────────────────────────
    scaler_path = os.path.join(MODELS_DIR, 'baseline_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"  ✅ baseline_scaler.pkl saved")

    print(f"\n  All files saved to reports/")


# ============================================================
# STEP 8 — Final Summary
# ============================================================

def step8_summary(results):
    print("\n" + "="*60)
    print("STEP 8: BASELINE RESULTS SUMMARY")
    print("="*60)

    print(f"\n  These are your STATIC BASELINE numbers.")
    print(f"  Your dynamic model must BEAT these scores.")
    print(f"  Save these — they go in your research paper.\n")

    print(f"  {'Model':<22} {'Accuracy':>9} "
          f"{'Precision':>10} {'Recall':>7} {'F1':>7}")
    print(f"  {'─'*22} {'─'*9} "
          f"{'─'*10} {'─'*7} {'─'*7}")

    for r in results:
        print(f"  {r['Model']:<22} "
              f"{r['Accuracy']:>8.2f}% "
              f"{r['Precision']:>9.2f}% "
              f"{r['Recall']:>6.2f}% "
              f"{r['F1_Score']:>6.2f}%")

    print(f"\n" + "="*60)
    print(f"  ✅ BASELINE MODEL COMPLETE")
    print(f"  Next step: Sliding Window")
    print(f"  File: src/features/sliding_window.py")
    print("="*60 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "="*60)
    print("  BASELINE MODEL (STATIC FEATURE SELECTION)")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    # Step 1: Load data
    train_df, test_df = step1_load()

    # Step 2: Prepare features
    (X_train, y_train,
     X_test,  y_test,
     X_train_scaled,
     X_test_scaled,
     scaler) = step2_prepare(train_df, test_df)

    # Step 3: Train models
    models = step3_train(
        X_train, y_train,
        X_train_scaled
    )

    # Step 4: Evaluate
    results = step4_evaluate(
        models, test_df,
        X_test, y_test,
        X_test_scaled
    )

    # Step 5: Per class
    step5_per_class(
        models,
        X_test, y_test,
        X_test_scaled
    )

    # Step 6: Per stock
    stock_results = step6_per_stock(
        models, test_df,
        X_test_scaled
    )

    # Step 7: Save everything
    step7_save(results, models, scaler, stock_results)

    # Step 8: Summary
    step8_summary(results)


if __name__ == '__main__':
    main()