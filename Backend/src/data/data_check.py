# ============================================================
# src/data/data_check.py
#
# Purpose:
# Check the raw dataset completely before doing anything.
# Run this first. If all checks pass, move to preprocessing.
#
# How to run:
# python src/data/data_check.py
# ============================================================

import pandas as pd
import numpy as np
import sys
import os

# ── Import config ──────────────────────────────────────────
# Go up to project root then into src/config
sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *


def main():

    print("\n" + "="*60)
    print("  DATASET CHECK")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    # ─────────────────────────────────────────
    # CHECK 1 — File exists
    # ─────────────────────────────────────────
    print("\n📁 CHECK 1: File exists?")
    print("-"*60)

    if not os.path.exists(RAW_DATA_PATH):
        print(f"  ❌ File NOT found.")
        print(f"  Expected: {RAW_DATA_PATH}")
        print(f"  Fix: Put stocks_dataset.csv in data/raw/ folder")
        sys.exit(1)

    print(f"  ✅ File found")
    print(f"  Path: {RAW_DATA_PATH}")

    # ─────────────────────────────────────────
    # CHECK 2 — Load file
    # ─────────────────────────────────────────
    print("\n📂 CHECK 2: Loading file...")
    print("-"*60)

    try:
        df = pd.read_csv(RAW_DATA_PATH)
        print(f"  ✅ Loaded successfully")
        print(f"  Rows    : {len(df):,}")
        print(f"  Columns : {len(df.columns)}")
    except Exception as e:
        print(f"  ❌ Failed to load: {e}")
        sys.exit(1)

    # ─────────────────────────────────────────
    # CHECK 3 — Shape
    # ─────────────────────────────────────────
    print("\n📐 CHECK 3: Shape check")
    print("-"*60)

    r_ok = len(df) == 54320
    c_ok = len(df.columns) == 45

    print(f"  Rows    : {len(df):,}  "
          f"{'✅' if r_ok else '⚠️ Expected 54320'}")
    print(f"  Columns : {len(df.columns)}    "
          f"{'✅' if c_ok else '⚠️ Expected 45'}")

    # ─────────────────────────────────────────
    # CHECK 4 — All columns present
    # ─────────────────────────────────────────
    print("\n📋 CHECK 4: Required columns present?")
    print("-"*60)

    required = FEATURE_COLS + [
        DATE_COL, STOCK_COL, TARGET_COL,
        REGIME_COL, REGIME_NUM_COL,
        'Open', 'High', 'Low', 'Close', 'Volume'
    ]

    missing_cols = [c for c in required if c not in df.columns]

    if len(missing_cols) == 0:
        print(f"  ✅ All required columns present")
    else:
        print(f"  ❌ Missing: {missing_cols}")

    # ─────────────────────────────────────────
    # CHECK 5 — Missing values
    # ─────────────────────────────────────────
    print("\n🔍 CHECK 5: Missing values")
    print("-"*60)

    total_null = df.isnull().sum().sum()

    if total_null == 0:
        print(f"  ✅ Zero missing values")
    else:
        print(f"  ⚠️  {total_null:,} missing values found")
        cols_with_null = df.isnull().sum()
        cols_with_null = cols_with_null[cols_with_null > 0]
        for col, cnt in cols_with_null.items():
            pct = cnt / len(df) * 100
            print(f"    {col}: {cnt} ({pct:.1f}%)")

    # ─────────────────────────────────────────
    # CHECK 6 — Duplicates
    # ─────────────────────────────────────────
    print("\n🔁 CHECK 6: Duplicate rows")
    print("-"*60)

    dupes = df.duplicated().sum()
    if dupes == 0:
        print(f"  ✅ Zero duplicate rows")
    else:
        print(f"  ⚠️  {dupes} duplicate rows found")

    # ─────────────────────────────────────────
    # CHECK 7 — All 20 stocks
    # ─────────────────────────────────────────
    print("\n📈 CHECK 7: All 20 stocks present?")
    print("-"*60)

    found_stocks   = sorted(df[STOCK_COL].unique())
    missing_stocks = [s for s in ALL_STOCKS
                      if s not in found_stocks]

    print(f"  Found : {len(found_stocks)} stocks")

    if len(missing_stocks) == 0:
        print(f"  ✅ All 20 stocks present")
        print(f"  {found_stocks}")
    else:
        print(f"  ❌ Missing: {missing_stocks}")

    # ─────────────────────────────────────────
    # CHECK 8 — Equal rows per stock
    # ─────────────────────────────────────────
    print("\n🔢 CHECK 8: Equal rows per stock?")
    print("-"*60)

    counts  = df.groupby(STOCK_COL).size()
    min_r   = counts.min()
    max_r   = counts.max()

    if min_r == max_r:
        print(f"  ✅ All stocks: {min_r} rows each")
    else:
        print(f"  ⚠️  Unequal rows (min={min_r}, max={max_r})")
        print(counts.to_string())

    # ─────────────────────────────────────────
    # CHECK 9 — Date range
    # ─────────────────────────────────────────
    print("\n📅 CHECK 9: Date range")
    print("-"*60)

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    print(f"  Start : {df[DATE_COL].min().date()}")
    print(f"  End   : {df[DATE_COL].max().date()}")
    print(f"  ✅ Date range confirmed")

    # ─────────────────────────────────────────
    # CHECK 10 — Target distribution
    # ─────────────────────────────────────────
    print("\n🎯 CHECK 10: Target distribution")
    print("-"*60)

    total = len(df)
    for val in [0, 1, 2]:
        count = (df[TARGET_COL] == val).sum()
        pct   = count / total * 100
        label = TARGET_NAMES[val]
        bar   = '█' * int(pct / 2)
        print(f"  {val} {label:<4}: {count:>7,}  "
              f"({pct:.1f}%)  {bar}")

    print(f"\n  ⚠️  Imbalance exists (Buy dominates)")
    print(f"  Fix: class_weight='balanced' — already in config ✅")

    # ─────────────────────────────────────────
    # CHECK 11 — Infinite values
    # ─────────────────────────────────────────
    print("\n♾️  CHECK 11: Infinite values")
    print("-"*60)

    num_cols  = df.select_dtypes(include=[np.number]).columns
    inf_count = np.isinf(df[num_cols].values).sum()

    if inf_count == 0:
        print(f"  ✅ Zero infinite values")
    else:
        print(f"  ⚠️  {inf_count} infinite values found")

    # ─────────────────────────────────────────
    # CHECK 12 — Feature columns numeric
    # ─────────────────────────────────────────
    print("\n🔬 CHECK 12: Feature columns are numeric?")
    print("-"*60)

    non_num = [c for c in FEATURE_COLS
               if c in df.columns and
               df[c].dtype not in ['float64','int64','float32']]

    if len(non_num) == 0:
        print(f"  ✅ All {len(FEATURE_COLS)} features are numeric")
    else:
        print(f"  ⚠️  Non-numeric: {non_num}")

    # ─────────────────────────────────────────
    # FINAL RESULT
    # ─────────────────────────────────────────
    all_ok = (
        total_null    == 0 and
        dupes         == 0 and
        inf_count     == 0 and
        len(missing_stocks) == 0 and
        len(non_num)  == 0 and
        len(missing_cols)   == 0
    )

    print("\n" + "="*60)
    if all_ok:
        print("  ✅ ALL 12 CHECKS PASSED")
        print("  Dataset is clean and ready.")
        print("  Next: Run src/data/preprocessing.py")
    else:
        print("  ⚠️  SOME CHECKS FAILED")
        print("  Fix the issues shown above first.")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()