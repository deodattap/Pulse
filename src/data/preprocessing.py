# ============================================================
# src/data/preprocessing.py
#
# Purpose:
# Prepare the raw dataset for the entire ML pipeline.
# This runs once and saves cleaned files that
# every other module will load from.
#
# What it does:
# 1. Loads raw dataset
# 2. Fixes Date column
# 3. Sorts by Stock then Date
# 4. Validates everything
# 5. Creates Train / Test split by DATE
# 6. Saves cleaned files to data/processed/
#
# How to run:
# python src/data/preprocessing.py
#
# Output files created:
# data/processed/cleaned_dataset.csv
# data/processed/train_dataset.csv
# data/processed/test_dataset.csv
# data/processed/feature_cols.txt
# ============================================================

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ── Import config ──────────────────────────────────────────
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *


# ============================================================
# STEP 1 — Load Raw Dataset
# ============================================================

def step1_load():
    print("\n" + "="*60)
    print("STEP 1: Loading raw dataset")
    print("="*60)

    if not os.path.exists(RAW_DATA_PATH):
        print(f"  ❌ File not found: {RAW_DATA_PATH}")
        print(f"  Run data_check.py first.")
        sys.exit(1)

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"  ✅ Loaded successfully")
    print(f"  Rows    : {len(df):,}")
    print(f"  Columns : {len(df.columns)}")

    return df


# ============================================================
# STEP 2 — Fix Date Column
# ============================================================

def step2_fix_dates(df):
    print("\n" + "="*60)
    print("STEP 2: Fixing Date column")
    print("="*60)

    print(f"  Before : dtype = {df[DATE_COL].dtype}")

    # Convert string to proper datetime
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])

    print(f"  After  : dtype = {df[DATE_COL].dtype}  ✅")
    print(f"  Range  : {df[DATE_COL].min().date()} "
          f"→ {df[DATE_COL].max().date()}")

    return df


# ============================================================
# STEP 3 — Sort by Stock then Date
# ============================================================

def step3_sort(df):
    print("\n" + "="*60)
    print("STEP 3: Sorting by Stock Symbol then Date")
    print("="*60)

    # WHY THIS IS CRITICAL:
    # Stock market data is time series.
    # Day 1 must always come before Day 2 for each stock.
    # If not sorted, the sliding window will be wrong.
    # Wrong window = wrong training data = wrong results.

    df = df.sort_values(
        by=[STOCK_COL, DATE_COL],
        ascending=[True, True]
    ).reset_index(drop=True)

    print(f"  Sorted by: {STOCK_COL} → {DATE_COL}  ✅")
    print(f"  Index reset: Done  ✅")

    # Verify sort is correct
    print(f"\n  First 3 rows after sort:")
    print(f"  {'Stock':<8} {'Date'}")
    print(f"  {'-'*8} {'-'*12}")
    for _, row in df.head(3).iterrows():
        print(f"  {row[STOCK_COL]:<8} "
              f"{str(row[DATE_COL].date())}")

    return df


# ============================================================
# STEP 4 — Verify No Data Issues
# ============================================================

def step4_verify(df):
    print("\n" + "="*60)
    print("STEP 4: Final verification")
    print("="*60)

    issues = []

    # Check nulls
    nulls = df.isnull().sum().sum()
    if nulls == 0:
        print(f"  Missing values : 0  ✅")
    else:
        print(f"  Missing values : {nulls}  ⚠️")
        issues.append(f"{nulls} missing values")

    # Check duplicates
    dupes = df.duplicated().sum()
    if dupes == 0:
        print(f"  Duplicates     : 0  ✅")
    else:
        print(f"  Duplicates     : {dupes}  ⚠️")
        issues.append(f"{dupes} duplicate rows")

    # Check infinite values
    num_cols  = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[num_cols].values).sum()
    if inf_count == 0:
        print(f"  Infinite vals  : 0  ✅")
    else:
        print(f"  Infinite vals  : {inf_count}  ⚠️")
        issues.append(f"{inf_count} infinite values")

    # Check all 20 stocks still present
    found = df[STOCK_COL].nunique()
    if found == 20:
        print(f"  Stocks         : {found}  ✅")
    else:
        print(f"  Stocks         : {found}  ⚠️ Expected 20")
        issues.append(f"Only {found} stocks found")

    # Check feature columns all numeric
    bad_cols = []
    for col in FEATURE_COLS:
        if col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                bad_cols.append(col)

    if len(bad_cols) == 0:
        print(f"  Feature types  : All numeric  ✅")
    else:
        print(f"  Non-numeric    : {bad_cols}  ⚠️")
        issues.append(f"Non-numeric features: {bad_cols}")

    if len(issues) == 0:
        print(f"\n  ✅ All verification checks passed")
    else:
        print(f"\n  ⚠️  Issues found:")
        for issue in issues:
            print(f"    - {issue}")

    return df


# ============================================================
# STEP 5 — Create Train / Test Split
# ============================================================

def step5_split(df):
    print("\n" + "="*60)
    print("STEP 5: Creating Train / Test split")
    print("="*60)

    # ── WHY DATE-BASED SPLIT ───────────────────────────────
    # Time series data MUST be split by date.
    # NEVER use random split (train_test_split with shuffle).
    #
    # Random split example of what goes wrong:
    # Training sees: Jan 2023 data
    # Testing on:    Dec 2022 data
    # Model already "knows" the future → fake high accuracy
    # This is called data leakage.
    #
    # Date-based split:
    # Training: 2015 → 2021 (past)
    # Testing:  2022 → 2025 (future)
    # Model never sees future data → honest results
    # ──────────────────────────────────────────────────────

    train_df = df[df[DATE_COL] <= TRAIN_END_DATE].copy()
    test_df  = df[df[DATE_COL] >= TEST_START_DATE].copy()

    # ── Show training set details ──────────────────────────
    print(f"\n  TRAINING SET  (2015 → {TRAIN_END_DATE})")
    print(f"  {'─'*40}")
    print(f"  Rows         : {len(train_df):,}")
    print(f"  Stocks       : {train_df[STOCK_COL].nunique()}")
    print(f"  Date start   : {train_df[DATE_COL].min().date()}")
    print(f"  Date end     : {train_df[DATE_COL].max().date()}")
    print(f"  Sell (0)     : "
          f"{(train_df[TARGET_COL]==0).sum():,}")
    print(f"  Buy  (1)     : "
          f"{(train_df[TARGET_COL]==1).sum():,}")
    print(f"  Hold (2)     : "
          f"{(train_df[TARGET_COL]==2).sum():,}")

    # ── Show testing set details ───────────────────────────
    print(f"\n  TESTING SET   ({TEST_START_DATE} → 2025)")
    print(f"  {'─'*40}")
    print(f"  Rows         : {len(test_df):,}")
    print(f"  Stocks       : {test_df[STOCK_COL].nunique()}")
    print(f"  Date start   : {test_df[DATE_COL].min().date()}")
    print(f"  Date end     : {test_df[DATE_COL].max().date()}")
    print(f"  Sell (0)     : "
          f"{(test_df[TARGET_COL]==0).sum():,}")
    print(f"  Buy  (1)     : "
          f"{(test_df[TARGET_COL]==1).sum():,}")
    print(f"  Hold (2)     : "
          f"{(test_df[TARGET_COL]==2).sum():,}")

    # ── Check for data leakage ─────────────────────────────
    train_dates = set(train_df[DATE_COL].dt.date)
    test_dates  = set(test_df[DATE_COL].dt.date)
    overlap     = train_dates.intersection(test_dates)

    print(f"\n  Date overlap check:")
    if len(overlap) == 0:
        print(f"  ✅ Zero overlap — No data leakage")
    else:
        print(f"  ❌ {len(overlap)} overlapping dates found")
        print(f"  This means data leakage. Check split dates.")

    return train_df, test_df


# ============================================================
# STEP 6 — Save All Files
# ============================================================

def step6_save(df, train_df, test_df):
    print("\n" + "="*60)
    print("STEP 6: Saving files to data/processed/")
    print("="*60)

    # Create processed folder if it doesn't exist
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR,           exist_ok=True)
    os.makedirs(GRAPHS_DIR,         exist_ok=True)
    os.makedirs(METRICS_DIR,        exist_ok=True)
    os.makedirs(MODELS_DIR,         exist_ok=True)

    # ── Save full cleaned dataset ──────────────────────────
    df.to_csv(CLEAN_DATA_PATH, index=False)
    size = os.path.getsize(CLEAN_DATA_PATH) / 1024 / 1024
    print(f"\n  ✅ cleaned_dataset.csv")
    print(f"     Rows : {len(df):,}")
    print(f"     Size : {size:.1f} MB")

    # ── Save training set ──────────────────────────────────
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    size = os.path.getsize(TRAIN_DATA_PATH) / 1024 / 1024
    print(f"\n  ✅ train_dataset.csv")
    print(f"     Rows : {len(train_df):,}")
    print(f"     Size : {size:.1f} MB")

    # ── Save testing set ───────────────────────────────────
    test_df.to_csv(TEST_DATA_PATH, index=False)
    size = os.path.getsize(TEST_DATA_PATH) / 1024 / 1024
    print(f"\n  ✅ test_dataset.csv")
    print(f"     Rows : {len(test_df):,}")
    print(f"     Size : {size:.1f} MB")

    # ── Save feature column names ──────────────────────────
    # Every module reads this file to know
    # which columns are features.
    # This ensures all modules use exactly the same features.
    with open(FEATURE_LIST_PATH, 'w') as f:
        for col in FEATURE_COLS:
            f.write(col + '\n')

    print(f"\n  ✅ feature_cols.txt")
    print(f"     Features : {len(FEATURE_COLS)}")
    print(f"     Location : {FEATURE_LIST_PATH}")


# ============================================================
# STEP 7 — Print Final Summary
# ============================================================

def step7_summary(df, train_df, test_df):
    print("\n" + "="*60)
    print("STEP 7: Final Summary")
    print("="*60)

    print(f"\n  {'Item':<28} {'Value'}")
    print(f"  {'─'*28} {'─'*20}")
    print(f"  {'Full dataset rows':<28} {len(df):,}")
    print(f"  {'Training rows':<28} {len(train_df):,}")
    print(f"  {'Testing rows':<28} {len(test_df):,}")
    print(f"  {'Number of stocks':<28} "
          f"{df[STOCK_COL].nunique()}")
    print(f"  {'Number of features':<28} {len(FEATURE_COLS)}")
    print(f"  {'Date range':<28} "
          f"{df[DATE_COL].min().date()} → "
          f"{df[DATE_COL].max().date()}")
    print(f"  {'Train period':<28} "
          f"2015-03-16 → {TRAIN_END_DATE}")
    print(f"  {'Test period':<28} "
          f"{TEST_START_DATE} → 2025-12-30")
    print(f"  {'Missing values':<28} "
          f"{df.isnull().sum().sum()}")
    print(f"  {'Class weight setting':<28} {CLASS_WEIGHT}")

    print(f"\n  Files in data/processed/:")
    files = [
        'cleaned_dataset.csv',
        'train_dataset.csv',
        'test_dataset.csv',
        'feature_cols.txt'
    ]
    for f in files:
        fpath = os.path.join(DATA_PROCESSED_DIR, f)
        exists = '✅' if os.path.exists(fpath) else '❌'
        print(f"    {exists} {f}")

    print("\n" + "="*60)
    print("  ✅ PREPROCESSING COMPLETE")
    print("  All files saved to data/processed/")
    print("  Next step: Build the Baseline Model")
    print("  File: src/models/baseline_model.py")
    print("="*60 + "\n")


# ============================================================
# MAIN — Run all steps
# ============================================================

def main():
    print("\n" + "="*60)
    print("  DATA PREPROCESSING")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    # Step 1: Load
    df = step1_load()

    # Step 2: Fix dates
    df = step2_fix_dates(df)

    # Step 3: Sort
    df = step3_sort(df)

    # Step 4: Verify
    df = step4_verify(df)

    # Step 5: Split
    train_df, test_df = step5_split(df)

    # Step 6: Save
    step6_save(df, train_df, test_df)

    # Step 7: Summary
    step7_summary(df, train_df, test_df)


if __name__ == '__main__':
    main()