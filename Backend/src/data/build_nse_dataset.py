# ============================================================
# src/data/build_nse_dataset.py
#
# Downloads NSE India stock data and computes
# all technical indicators.
# Creates the dataset that replaces US stocks.
#
# How to run:
# python src/data/build_nse_dataset.py
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(
    os.path.join(os.path.dirname(__file__), '..', 'config')
)
from config import *

# ─────────────────────────────────────────────
# NSE STOCKS TO DOWNLOAD
# 20 Nifty 50 stocks across sectors
# ─────────────────────────────────────────────

NSE_STOCKS = {
    # Technology
    'TCS.NS'        : 'TCS',
    'INFY.NS'       : 'Infosys',
    'WIPRO.NS'      : 'Wipro',
    'HCLTECH.NS'    : 'HCL Tech',
    'TECHM.NS'      : 'Tech Mahindra',

    # Finance
    'HDFCBANK.NS'   : 'HDFC Bank',
    'ICICIBANK.NS'  : 'ICICI Bank',
    'SBIN.NS'       : 'SBI',
    'KOTAKBANK.NS'  : 'Kotak Bank',
    'AXISBANK.NS'   : 'Axis Bank',

    # Energy and Industry
    'RELIANCE.NS'   : 'Reliance',
    'ONGC.NS'       : 'ONGC',
    'NTPC.NS'       : 'NTPC',
    'POWERGRID.NS'  : 'Power Grid',
    'TATASTEEL.NS'  : 'Tata Steel',

    # Consumer
    'HINDUNILVR.NS' : 'HUL',
    'ITC.NS'        : 'ITC',
    'NESTLEIND.NS'  : 'Nestle',
    'BRITANNIA.NS'  : 'Britannia',
    'TITAN.NS'      : 'Titan',
}

START_DATE    = '2015-01-01'
END_DATE      = '2025-01-01'
FORWARD_DAYS  = 20   # 20-day prediction horizon


def compute_features(df):
    """
    Compute all technical indicators
    from OHLCV data.
    """

    close  = df['Close']
    high   = df['High']
    low    = df['Low']
    volume = df['Volume']

    # Trend
    df['SMA_10']  = ta.sma(close, length=10)
    df['SMA_20']  = ta.sma(close, length=20)
    df['SMA_50']  = ta.sma(close, length=50)
    df['EMA_12']  = ta.ema(close, length=12)
    df['EMA_26']  = ta.ema(close, length=26)

    # Momentum
    df['RSI_14']  = ta.rsi(close, length=14)

    macd_df = ta.macd(close)
    if macd_df is not None and len(macd_df.columns) >= 2:
        df['MACD']        = macd_df.iloc[:, 0]
        df['MACD_Signal'] = macd_df.iloc[:, 1]
        df['MACD_Hist']   = macd_df.iloc[:, 2]
    else:
        df['MACD']        = 0
        df['MACD_Signal'] = 0
        df['MACD_Hist']   = 0

    df['ROC']     = ta.roc(close, length=10)
    df['Williams_R'] = ta.willr(high, low, close,
                                length=14)

    # Stochastic
    stoch = ta.stoch(high, low, close)
    if stoch is not None and len(stoch.columns) >= 1:
        df['Stochastic'] = stoch.iloc[:, 0]
    else:
        df['Stochastic'] = 50

    # Volatility
    df['ATR_14']  = ta.atr(high, low, close, length=14)

    bb = ta.bbands(close, length=20)
    if bb is not None and len(bb.columns) >= 4:
        df['BB_Upper'] = bb.iloc[:, 0]
        df['BB_Lower'] = bb.iloc[:, 1]
        df['BB_Width'] = bb.iloc[:, 3]
    else:
        df['BB_Upper'] = close
        df['BB_Lower'] = close
        df['BB_Width'] = 0

    # Volume
    df['OBV']          = ta.obv(close, volume)
    df['Volume_MA']    = volume.rolling(20).mean()
    df['Volume_Ratio'] = volume / df['Volume_MA']

    # Statistical
    df['CCI']    = ta.cci(high, low, close, length=14)

    adx_df = ta.adx(high, low, close, length=14)
    if adx_df is not None and len(adx_df.columns) >= 1:
        df['ADX'] = adx_df.iloc[:, 0]
    else:
        df['ADX'] = 25

    # Returns
    df['Daily_Return']       = close.pct_change()
    df['Log_Return']         = np.log(
        close / close.shift(1)
    )
    df['Return_5d']          = close.pct_change(5)
    df['Return_10d']         = close.pct_change(10)
    df['Rolling_Std_10']     = (
        df['Daily_Return'].rolling(10).std()
    )
    df['Rolling_Volatility'] = (
        df['Daily_Return'].rolling(20).std()
    )

    # Price patterns
    df['High_Low_Pct']   = (high - low) / close
    df['Open_Close_Pct'] = (
        df['Open'] - close
    ) / close
    df['Price_Change']   = close.diff()

    # Money Flow
    typical_price = (high + low + close) / 3
    mf = typical_price * volume
    pos_mf = mf.where(
        typical_price > typical_price.shift(1), 0
    )
    neg_mf = mf.where(
        typical_price < typical_price.shift(1), 0
    )
    mfr = (
        pos_mf.rolling(14).sum() /
        neg_mf.rolling(14).sum().replace(0, 1)
    )
    df['MFI'] = 100 - (100 / (1 + mfr))

    # Rolling mean and median returns
    df['Rolling_Mean_Return']   = (
        df['Daily_Return'].rolling(10).mean()
    )
    df['Rolling_Median_Return'] = (
        df['Daily_Return'].rolling(10).median()
    )

    return df


def create_target(df):
    """
    Create 20-day binary forward return target.
    1 = price higher 20 days from now (BUY)
    0 = price lower  20 days from now (SELL)
    """
    df['Forward_Return'] = (
        df['Close'].shift(-FORWARD_DAYS) /
        df['Close'] - 1
    )
    df['Binary_Target'] = (
        df['Forward_Return'] > 0
    ).astype(int)
    return df


def download_and_process(ticker_symbol, name):
    """
    Download one stock and compute all features.
    Returns processed DataFrame or None if failed.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(
            start=START_DATE,
            end=END_DATE,
            auto_adjust=True
        )

        if len(df) < 300:
            print(f"    ⚠️  Only {len(df)} rows — skipping")
            return None

        df = df.reset_index()
        df['Date'] = pd.to_datetime(
            df['Date']
        ).dt.tz_localize(None)
        df['Stock_Symbol'] = ticker_symbol
        df['Stock_Name']   = name

        # Compute features
        df = compute_features(df)

        # Create target
        df = create_target(df)

        # Keep only needed columns
        feature_cols = [
            'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_12', 'EMA_26',
            'RSI_14', 'MACD', 'MACD_Signal',
            'MACD_Hist', 'ROC', 'Williams_R',
            'Stochastic', 'ATR_14',
            'BB_Upper', 'BB_Lower', 'BB_Width',
            'OBV', 'Volume_MA', 'Volume_Ratio',
            'CCI', 'ADX',
            'Daily_Return', 'Log_Return',
            'Return_5d', 'Return_10d',
            'Rolling_Std_10', 'Rolling_Volatility',
            'High_Low_Pct', 'Open_Close_Pct',
            'Price_Change', 'MFI',
            'Rolling_Mean_Return',
            'Rolling_Median_Return',
        ]

        keep_cols = (
            ['Date', 'Stock_Symbol', 'Stock_Name',
             'Open', 'High', 'Low', 'Close', 'Volume']
            + feature_cols
            + ['Forward_Return', 'Binary_Target']
        )

        existing_cols = [
            c for c in keep_cols if c in df.columns
        ]
        df = df[existing_cols]

        # Drop NaN rows (from indicator computation
        # and last FORWARD_DAYS rows)
        df = df.dropna().reset_index(drop=True)

        return df, feature_cols

    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None, None


def main():

    print("\n" + "="*60)
    print("  NSE DATASET BUILDER")
    print("  Dynamic Feature Selection Project")
    print("="*60)

    print(f"\n  Stocks   : {len(NSE_STOCKS)}")
    print(f"  Period   : {START_DATE} to {END_DATE}")
    print(f"  Target   : {FORWARD_DAYS}-day binary")
    print(f"  Features : ~35 technical indicators")

    all_dfs      = []
    feature_cols = None
    success      = 0
    failed       = []

    for ticker_sym, name in NSE_STOCKS.items():

        print(f"\n  Downloading {name} ({ticker_sym})...")

        result, fcols = download_and_process(
            ticker_sym, name
        )

        if result is not None:
            all_dfs.append(result)
            feature_cols = fcols
            success += 1

            # Show quick stats
            up_pct = result['Binary_Target'].mean() * 100
            print(f"    ✅ {len(result):,} rows  "
                  f"Up={up_pct:.1f}%  "
                  f"Down={100-up_pct:.1f}%")
        else:
            failed.append(name)

    if not all_dfs:
        print("\n  ❌ No data downloaded successfully")
        sys.exit(1)

    # Combine all stocks
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(
        ['Stock_Symbol', 'Date']
    ).reset_index(drop=True)

    print(f"\n" + "="*60)
    print(f"  DATASET SUMMARY")
    print(f"="*60)
    print(f"\n  Stocks downloaded : {success}/20")
    if failed:
        print(f"  Failed            : {failed}")
    print(f"  Total rows        : {len(combined):,}")
    print(f"  Date range        : "
          f"{combined['Date'].min().date()} → "
          f"{combined['Date'].max().date()}")
    print(f"  Features computed : {len(feature_cols)}")

    # Class balance
    up_pct = combined['Binary_Target'].mean() * 100
    print(f"\n  Target distribution:")
    print(f"    Up   (BUY)  : "
          f"{(combined['Binary_Target']==1).sum():,} "
          f"({up_pct:.1f}%)")
    print(f"    Down (SELL) : "
          f"{(combined['Binary_Target']==0).sum():,} "
          f"({100-up_pct:.1f}%)")

    if 45 <= up_pct <= 55:
        print(f"    ✅ Well balanced")
    else:
        print(f"    ⚠️  Slight imbalance "
              f"(still manageable)")

    # Save dataset
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    # Save main dataset
    out_path = os.path.join(
        DATA_PROCESSED_DIR, 'nse_dataset.csv'
    )
    combined.to_csv(out_path, index=False)
    size = os.path.getsize(out_path) / 1024 / 1024
    print(f"\n  ✅ Saved: data/processed/nse_dataset.csv")
    print(f"     Size : {size:.1f} MB")

    # Save feature column names
    feat_path = os.path.join(
        DATA_PROCESSED_DIR, 'nse_feature_cols.txt'
    )
    with open(feat_path, 'w') as f:
        for col in feature_cols:
            f.write(col + '\n')
    print(f"  ✅ Saved: nse_feature_cols.txt "
          f"({len(feature_cols)} features)")

    # Save train/test splits
    train = combined[
        combined['Date'] <= '2021-12-31'
    ].copy()
    test  = combined[
        combined['Date'] >= '2022-01-01'
    ].copy()

    train.to_csv(
        os.path.join(
            DATA_PROCESSED_DIR, 'nse_train.csv'
        ),
        index=False
    )
    test.to_csv(
        os.path.join(
            DATA_PROCESSED_DIR, 'nse_test.csv'
        ),
        index=False
    )

    print(f"  ✅ Saved: nse_train.csv "
          f"({len(train):,} rows)")
    print(f"  ✅ Saved: nse_test.csv  "
          f"({len(test):,} rows)")

    print(f"\n" + "="*60)
    print(f"  ✅ NSE DATASET BUILD COMPLETE")
    print(f"  Next: Run walkforward with LSTM")
    print(f"  File: src/models/lstm_walkforward.py")
    print(f"="*60 + "\n")

    return feature_cols


if __name__ == '__main__':
    main()