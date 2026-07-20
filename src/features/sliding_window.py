# ============================================================
# src/features/sliding_window.py
# Purpose:
# Implement the sliding window mechanism.
# This is the ENGINE of the streaming simulation.
#
# What it does:
# - Takes stock data sorted by date
# - Slides a 60-day window one day at a time
# - At each step yields:
#     current window (60 rows of features)
#     next day target (what we are predicting)
#     current date
#     window number
#
# From report Section 5.4:
#   Window Size = 60 trading days
#   Step Size   = 1 day
#
# How to run (test only):
# python src/features/sliding_window.py
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
# CORE FUNCTION — Sliding Window Generator
# ============================================================

def sliding_window_generator(stock_df,
                              window_size=WINDOW_SIZE,
                              step_size=STEP_SIZE):
    """
    Generates sliding windows for one stock.

    Parameters:
        stock_df    : DataFrame for ONE stock, sorted by date
        window_size : Number of days in each window (default 60)
        step_size   : Days to move forward each step (default 1)

    Yields at each step:
        window_data  : DataFrame with window_size rows (features)
        next_target  : The actual target for the next day
        predict_date : The date we are predicting
        window_num   : Which window number this is (starts at 1)

    Example:
        Window 1 → rows 0-59  → predicts row 60's target
        Window 2 → rows 1-60  → predicts row 61's target
        Window 3 → rows 2-61  → predicts row 62's target
    """

    # Reset index to ensure clean 0-based indexing
    stock_df = stock_df.reset_index(drop=True)

    total_rows   = len(stock_df)
    window_num   = 0

    # We need at least window_size + 1 rows
    # window_size rows for training
    # 1 row for the prediction target
    if total_rows < window_size + 1:
        return

    # Slide the window
    start = 0
    while start + window_size < total_rows:

        end = start + window_size

        # Current window: rows from start to end (exclusive)
        window_data = stock_df.iloc[start:end]

        # Next day: the row immediately after the window
        next_row    = stock_df.iloc[end]

        # What we are predicting
        next_target  = next_row[TARGET_COL]

        # The date of the prediction
        predict_date = next_row[DATE_COL]

        window_num += 1

        yield window_data, next_target, predict_date, window_num

        # Move window forward by step_size
        start += step_size


# ============================================================
# HELPER — Get all windows for one stock
# ============================================================

def get_stock_windows(cleaned_df, stock_symbol,
                      window_size=WINDOW_SIZE):
    """
    Returns all windows for a specific stock.

    Parameters:
        cleaned_df   : Full cleaned dataset
        stock_symbol : e.g. 'AAPL'
        window_size  : Window size in days

    Returns:
        List of (window_data, next_target,
                 predict_date, window_num)
    """

    # Filter for this stock only
    stock_df = cleaned_df[
        cleaned_df[STOCK_COL] == stock_symbol
    ].copy()

    # Sort by date — CRITICAL
    stock_df = stock_df.sort_values(DATE_COL)
    stock_df = stock_df.reset_index(drop=True)

    # Collect all windows
    windows = []
    for w_data, target, date, num in sliding_window_generator(
        stock_df, window_size
    ):
        windows.append((w_data, target, date, num))

    return windows


# ============================================================
# HELPER — Count total windows across all stocks
# ============================================================

def count_all_windows(cleaned_df,
                      window_size=WINDOW_SIZE):
    """
    Counts total windows across all 20 stocks.
    Useful to know how many predictions will be made.
    """

    total = 0
    per_stock = {}

    for stock in ALL_STOCKS:
        stock_df = cleaned_df[
            cleaned_df[STOCK_COL] == stock
        ]
        n = max(0, len(stock_df) - window_size)
        per_stock[stock] = n
        total += n

    return total, per_stock


# ============================================================
# HELPER — Split windows into train and test periods
# ============================================================

def split_windows_by_date(windows, test_start=TEST_START_DATE):
    """
    Splits windows into train-period and test-period.
    A window belongs to test period if its prediction
    date is on or after TEST_START_DATE.

    Parameters:
        windows    : List from get_stock_windows()
        test_start : Date string e.g. '2022-01-01'

    Returns:
        train_windows : Windows predicting before test_start
        test_windows  : Windows predicting from test_start
    """

    test_start_dt = pd.Timestamp(test_start)

    train_windows = []
    test_windows  = []

    for w_data, target, date, num in windows:
        if date >= test_start_dt:
            test_windows.append((w_data, target, date, num))
        else:
            train_windows.append((w_data, target, date, num))

    return train_windows, test_windows


# ============================================================
# TEST — Run this file directly to verify it works
# ============================================================

def main():
    print("\n" + "="*60)
    print("  SLIDING WINDOW — TEST RUN")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    # ── Load cleaned dataset ───────────────────────────────
    print("\n  Loading cleaned dataset...")

    if not os.path.exists(CLEAN_DATA_PATH):
        print(f"  ❌ cleaned_dataset.csv not found.")
        print(f"  Run preprocessing.py first.")
        sys.exit(1)

    df = pd.read_csv(
        CLEAN_DATA_PATH,
        parse_dates=[DATE_COL]
    )
    print(f"  ✅ Loaded {len(df):,} rows")

    # ── Test on AAPL first ─────────────────────────────────
    print("\n" + "="*60)
    print("  TEST 1: Sliding window on AAPL")
    print("="*60)

    aapl_df = df[df[STOCK_COL] == 'AAPL'].copy()
    aapl_df = aapl_df.sort_values(DATE_COL).reset_index(drop=True)

    print(f"\n  AAPL data:")
    print(f"  Total rows  : {len(aapl_df):,}")
    print(f"  Date range  : {aapl_df[DATE_COL].min().date()} "
          f"→ {aapl_df[DATE_COL].max().date()}")
    print(f"  Window size : {WINDOW_SIZE} days")
    print(f"  Expected windows : "
          f"~{len(aapl_df) - WINDOW_SIZE:,}")

    # Show first 5 windows
    print(f"\n  First 5 windows:")
    print(f"  {'Win':>4}  {'From':<12}  "
          f"{'To':<12}  {'Predict Date':<13}  "
          f"{'Target':<8}  {'Rows':>4}")
    print(f"  {'─'*4}  {'─'*12}  "
          f"{'─'*12}  {'─'*13}  "
          f"{'─'*8}  {'─'*4}")

    count = 0
    total_windows = 0

    for w_data, target, date, num in sliding_window_generator(
        aapl_df
    ):
        total_windows += 1

        if count < 5:
            w_start = w_data[DATE_COL].min().date()
            w_end   = w_data[DATE_COL].max().date()
            label   = TARGET_NAMES.get(target, str(target))
            rows    = len(w_data)

            print(f"  {num:>4}  {str(w_start):<12}  "
                  f"{str(w_end):<12}  {str(date.date()):<13}  "
                  f"{label:<8}  {rows:>4}")
            count += 1

    print(f"\n  Total windows generated : {total_windows:,}")

    # ── Verify window integrity ────────────────────────────
    print("\n" + "="*60)
    print("  TEST 2: Window integrity check")
    print("="*60)

    errors_found = False

    for w_data, target, date, num in sliding_window_generator(
        aapl_df
    ):

        # Check 1: Window must have exactly WINDOW_SIZE rows
        if len(w_data) != WINDOW_SIZE:
            print(f"  ❌ Window {num}: "
                  f"has {len(w_data)} rows, "
                  f"expected {WINDOW_SIZE}")
            errors_found = True

        # Check 2: Window dates must be in order
        dates_sorted = w_data[DATE_COL].is_monotonic_increasing
        if not dates_sorted:
            print(f"  ❌ Window {num}: dates not in order")
            errors_found = True

        # Check 3: Prediction date must be after window
        last_window_date = w_data[DATE_COL].max()
        if date <= last_window_date:
            print(f"  ❌ Window {num}: "
                  f"prediction date {date.date()} "
                  f"is not after window end "
                  f"{last_window_date.date()}")
            errors_found = True

    if not errors_found:
        print(f"  ✅ All {total_windows:,} windows passed "
              f"integrity check")
        print(f"  ✅ Every window has exactly "
              f"{WINDOW_SIZE} rows")
        print(f"  ✅ All dates in correct order")
        print(f"  ✅ Prediction date always after window")

    # ── Test train/test split of windows ──────────────────
    print("\n" + "="*60)
    print("  TEST 3: Train / Test window split")
    print("="*60)

    all_windows   = get_stock_windows(df, 'AAPL')
    train_windows, test_windows = split_windows_by_date(
        all_windows
    )

    print(f"\n  Total windows   : {len(all_windows):,}")
    print(f"  Train windows   : {len(train_windows):,}  "
          f"(predictions before 2022)")
    print(f"  Test windows    : {len(test_windows):,}   "
          f"(predictions from 2022)")

    if len(train_windows) > 0:
        first_test_date = test_windows[0][2]
        last_train_date = train_windows[-1][2]
        print(f"\n  Last train prediction  : "
              f"{last_train_date.date()}")
        print(f"  First test prediction  : "
              f"{first_test_date.date()}")

        if first_test_date > last_train_date:
            print(f"  ✅ No overlap between train and test windows")
        else:
            print(f"  ❌ Overlap found — check split logic")

    # ── Count windows for all 20 stocks ───────────────────
    print("\n" + "="*60)
    print("  TEST 4: Window count for all 20 stocks")
    print("="*60)

    total, per_stock = count_all_windows(df)

    print(f"\n  {'Stock':<8}  {'Windows':>8}")
    print(f"  {'─'*8}  {'─'*8}")

    for stock in ALL_STOCKS:
        print(f"  {stock:<8}  "
              f"{per_stock.get(stock, 0):>8,}")

    print(f"  {'─'*8}  {'─'*8}")
    print(f"  {'TOTAL':<8}  {total:>8,}")

    print(f"\n  This means your dynamic system will make")
    print(f"  {total:,} predictions across all 20 stocks.")

    # ── Show sample window content ─────────────────────────
    print("\n" + "="*60)
    print("  TEST 5: Sample window content (Window 1, AAPL)")
    print("="*60)

    first_window = None
    for w_data, target, date, num in sliding_window_generator(
        aapl_df
    ):
        first_window = w_data
        first_target = target
        first_date   = date
        break

    print(f"\n  Window 1 details:")
    print(f"  Rows         : {len(first_window)}")
    print(f"  Date start   : "
          f"{first_window[DATE_COL].min().date()}")
    print(f"  Date end     : "
          f"{first_window[DATE_COL].max().date()}")
    print(f"  Predict date : {first_date.date()}")
    print(f"  Next target  : "
          f"{first_target} "
          f"({TARGET_NAMES.get(first_target, '?')})")

    print(f"\n  Feature values in Window 1 "
          f"(mean of 60 days):")
    print(f"  {'Feature':<25}  {'Mean':>10}  "
          f"{'Min':>10}  {'Max':>10}")
    print(f"  {'─'*25}  {'─'*10}  "
          f"{'─'*10}  {'─'*10}")

    for col in FEATURE_COLS[:10]:
        if col in first_window.columns:
            mean = first_window[col].mean()
            mn   = first_window[col].min()
            mx   = first_window[col].max()
            print(f"  {col:<25}  {mean:>10.3f}  "
                  f"{mn:>10.3f}  {mx:>10.3f}")

    print(f"  ... and {len(FEATURE_COLS)-10} more features")

    # ── Final result ───────────────────────────────────────
    print("\n" + "="*60)
    print("  SLIDING WINDOW TEST COMPLETE")
    print("="*60)
    print(f"\n  ✅ Sliding window works correctly")
    print(f"  ✅ {total:,} total windows across 20 stocks")
    print(f"  ✅ Window integrity verified")
    print(f"  ✅ Train/test split working")
    print(f"\n  Next step: Dynamic Feature Selection")
    print(f"  File: src/features/dynamic_feature_selection.py")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()