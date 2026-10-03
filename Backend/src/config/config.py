# ============================================================
# src/config/config.py
#
# Central settings file for the entire project.
# Every other file imports from here.
# Change any setting here and it updates everywhere.
# ============================================================

import os

# ─────────────────────────────────────────────
# PROJECT PATHS
# Finds project root automatically
# Works on any laptop regardless of username
# ─────────────────────────────────────────────

# Go up two levels from src/config/ to reach project root
BASE_DIR = os.path.dirname(
               os.path.dirname(
                   os.path.dirname(
                       os.path.abspath(__file__)
                   )
               )
           )

# Data paths
DATA_RAW_DIR      = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR= os.path.join(BASE_DIR, 'data', 'processed')

RAW_DATA_PATH     = os.path.join(DATA_RAW_DIR, 'stocks_dataset.csv')
CLEAN_DATA_PATH   = os.path.join(DATA_PROCESSED_DIR, 'cleaned_dataset.csv')
TRAIN_DATA_PATH   = os.path.join(DATA_PROCESSED_DIR, 'train_dataset.csv')
TEST_DATA_PATH    = os.path.join(DATA_PROCESSED_DIR, 'test_dataset.csv')
FEATURE_LIST_PATH = os.path.join(DATA_PROCESSED_DIR, 'feature_cols.txt')

# Results paths
REPORTS_DIR   = os.path.join(BASE_DIR, 'reports')
LOGS_DIR      = os.path.join(REPORTS_DIR, 'logs')
GRAPHS_DIR    = os.path.join(REPORTS_DIR, 'graphs')
METRICS_DIR   = os.path.join(REPORTS_DIR, 'metrics')
MODELS_DIR    = os.path.join(REPORTS_DIR, 'models')

# ─────────────────────────────────────────────
# COLUMN NAMES
# ─────────────────────────────────────────────

DATE_COL       = 'Date'
STOCK_COL      = 'Stock Symbol'
TARGET_COL     = 'Target'
REGIME_COL     = 'Market_Regime'
REGIME_NUM_COL = 'Market_Regime_Numeric'

# ─────────────────────────────────────────────
# ALL 20 STOCKS
# ─────────────────────────────────────────────

ALL_STOCKS = [
    'AAPL',  'AMZN',  'BAC',   'BRK-B', 'CAT',
    'CVX',   'GOOGL', 'GS',    'JNJ',   'JPM',
    'KO',    'META',  'MSFT',  'NFLX',  'NVDA',
    'PG',    'SPY',   'TSLA',  'WMT',   'XOM'
]

# ─────────────────────────────────────────────
# TARGET LABELS
# ─────────────────────────────────────────────

# 0 = Sell
# 1 = Buy
# 2 = Hold
TARGET_NAMES  = {0: 'Sell', 1: 'Buy',   2: 'Hold'}
TARGET_COLORS = {0: 'red',  1: 'green', 2: 'grey'}

# ─────────────────────────────────────────────
# FEATURE COLUMNS
# 35 technical indicators
# Dynamic Feature Selection will choose from these
# ─────────────────────────────────────────────

FEATURE_COLS = [
    # Trend
    'SMA_10', 'SMA_20', 'SMA_50',
    'EMA_12', 'EMA_26',

    # Momentum
    'RSI_14', 'MACD', 'MACD_Signal',
    'MACD_Histogram', 'ROC',
    'Stochastic_Oscillator', 'Williams_R',

    # Volatility
    'ATR_14', 'Bollinger_Upper',
    'Bollinger_Lower', 'Bollinger_Width',

    # Return based
    'Daily_Return', 'Log_Return',
    '5_Day_Return', '10_Day_Return',
    'Rolling_Std_10', 'Daily_Volatility',

    # Volume based
    'Volume_MA', 'Volume_Ratio', 'OBV',

    # Price pattern
    'High_Low_Pct', 'Open_Close_Pct',
    'Price_Change', 'Gap_UpDown',
    'Rolling_Volatility_20',

    # Statistical
    'Rolling_Mean_Return', 'Rolling_Median_Return',
    'CCI', 'ADX', 'Money_Flow_Index',
]

# ─────────────────────────────────────────────
# DATE SPLIT
# ─────────────────────────────────────────────

TRAIN_END_DATE  = '2021-12-31'   # Train on data before this
TEST_START_DATE = '2022-01-01'   # Test on data from this

# ─────────────────────────────────────────────
# SLIDING WINDOW SETTINGS
# As defined in report Section 5.4
# ─────────────────────────────────────────────

WINDOW_SIZE   = 60   # 60 trading days per window
STEP_SIZE     = 1    # Move 1 day at a time

# ─────────────────────────────────────────────
# DYNAMIC FEATURE SELECTION SETTINGS
# As defined in report Section 5.4
# ─────────────────────────────────────────────

TOP_K_FEATURES = 10  # Select top 10 features per window
MI_FREQUENCY   = 5   # Recalculate MI every 5 windows
RFE_FREQUENCY  = 20  # Run RFE every 20 windows

# ─────────────────────────────────────────────
# MODEL SETTINGS
# ─────────────────────────────────────────────

RANDOM_STATE = 42            # Reproducible results
N_ESTIMATORS = 100           # Trees in Random Forest
CLASS_WEIGHT = 'balanced'    # Fix class imbalance automatically