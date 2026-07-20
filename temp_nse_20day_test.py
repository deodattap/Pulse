# temp_nse_20day_test.py
# Tests NSE stock with 20-day binary target
# This will tell us expected accuracy
# Delete after running

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import pandas_ta as ta
    HAS_TA = True
except:
    HAS_TA = False
    print("pandas_ta not installed")
    print("Run: pip install pandas-ta")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score, f1_score
from xgboost import XGBClassifier
from collections import Counter

def get_sample_weights(y):
    counts = Counter(y)
    total  = len(y)
    n      = len(counts)
    wmap   = {c: total/(n*cnt) for c,cnt in counts.items()}
    return np.array([wmap[l] for l in y])

print("="*55)
print("  NSE + 20-DAY TARGET ACCURACY TEST")
print("="*55)

# Test on 3 NSE stocks
test_stocks = [
    ('TCS.NS',       'TCS'),
    ('RELIANCE.NS',  'Reliance'),
    ('HDFCBANK.NS',  'HDFC Bank'),
]

results = []

for ticker_sym, name in test_stocks:
    print(f"\n  Downloading {name} ({ticker_sym})...")

    try:
        ticker = yf.Ticker(ticker_sym)
        df = ticker.history(
            start="2015-01-01",
            end="2025-01-01"
        )
        df = df.reset_index()

        if len(df) < 500:
            print(f"  Not enough data for {name}")
            continue

        print(f"  Rows: {len(df)}")

        # Compute features
        close  = df['Close']
        high   = df['High']
        low    = df['Low']
        volume = df['Volume']

        if HAS_TA:
            df['sma_10']  = ta.sma(close, length=10)
            df['sma_20']  = ta.sma(close, length=20)
            df['sma_50']  = ta.sma(close, length=50)
            df['ema_12']  = ta.ema(close, length=12)
            df['ema_26']  = ta.ema(close, length=26)
            df['rsi_14']  = ta.rsi(close, length=14)

            macd_df = ta.macd(close)
            if macd_df is not None:
                df['macd'] = macd_df.iloc[:, 0]
                df['macd_signal'] = macd_df.iloc[:, 1]
            else:
                df['macd'] = 0
                df['macd_signal'] = 0

            bb_df = ta.bbands(close)
            if bb_df is not None:
                df['bb_upper'] = bb_df.iloc[:, 0]
                df['bb_lower'] = bb_df.iloc[:, 1]
                df['bb_width'] = bb_df.iloc[:, 3]
            else:
                df['bb_upper'] = close
                df['bb_lower'] = close
                df['bb_width'] = 0

            df['atr_14'] = ta.atr(
                high, low, close, length=14
            )
            df['obv']    = ta.obv(close, volume)
            df['roc']    = ta.roc(close, length=10)
            df['cci']    = ta.cci(
                high, low, close, length=14
            )
            df['adx']    = ta.adx(
                high, low, close, length=14
            ).iloc[:, 0]

        else:
            # Manual computation if pandas_ta not available
            df['sma_10'] = close.rolling(10).mean()
            df['sma_20'] = close.rolling(20).mean()
            df['sma_50'] = close.rolling(50).mean()
            df['ema_12'] = close.ewm(span=12).mean()
            df['ema_26'] = close.ewm(span=26).mean()

            delta = close.diff()
            gain  = delta.where(delta > 0, 0)
            loss  = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            df['rsi_14'] = 100 - (100 / (1 + rs))

            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            df['macd']        = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(
                span=9
            ).mean()

            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            df['bb_upper'] = sma20 + 2 * std20
            df['bb_lower'] = sma20 - 2 * std20
            df['bb_width'] = (
                df['bb_upper'] - df['bb_lower']
            ) / sma20

            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs()
            ], axis=1).max(axis=1)
            df['atr_14'] = tr.rolling(14).mean()

            df['obv'] = (
                np.sign(close.diff()) * volume
            ).fillna(0).cumsum()

            df['roc'] = close.pct_change(10) * 100
            df['cci'] = (close - sma20) / (
                0.015 * close.rolling(20).std()
            )
            df['adx'] = tr.rolling(14).mean()

        # Daily and rolling returns
        df['daily_return']  = close.pct_change()
        df['return_5d']     = close.pct_change(5)
        df['rolling_std']   = (
            df['daily_return'].rolling(10).std()
        )
        df['volume_ma']     = volume.rolling(20).mean()
        df['volume_ratio']  = volume / df['volume_ma']

        # 20-DAY BINARY TARGET
        df['forward_20d'] = (
            close.shift(-20) / close - 1
        )
        df['target_20d']  = (
            df['forward_20d'] > 0
        ).astype(int)

        # Also test 5-day for comparison
        df['forward_5d']  = (
            close.shift(-5) / close - 1
        )
        df['target_5d']   = (
            df['forward_5d'] > 0
        ).astype(int)

        # Drop NaN
        feature_cols = [
            'sma_10', 'sma_20', 'sma_50',
            'ema_12', 'ema_26', 'rsi_14',
            'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'bb_width',
            'atr_14', 'obv', 'roc', 'cci', 'adx',
            'daily_return', 'return_5d',
            'rolling_std', 'volume_ratio'
        ]

        df_clean = df.dropna(
            subset=feature_cols + ['target_20d']
        ).copy()

        print(f"  Clean rows: {len(df_clean)}")

        # Train/test split
        df_clean['Date'] = pd.to_datetime(
            df_clean['Date']
        ).dt.tz_localize(None)

        train = df_clean[
            df_clean['Date'] <= '2021-12-31'
        ]
        test  = df_clean[
            df_clean['Date'] >= '2022-01-01'
        ]

        if len(train) < 100 or len(test) < 50:
            print(f"  Not enough split data")
            continue

        # Test both targets
        for target_col, label in [
            ('target_20d', '20-day'),
            ('target_5d',  '5-day')
        ]:
            X_train = train[feature_cols].values
            y_train = train[target_col].values
            X_test  = test[feature_cols].values
            y_test  = test[target_col].values

            # Random Forest
            rf = RandomForestClassifier(
                n_estimators = 200,
                class_weight = 'balanced',
                random_state = 42,
                n_jobs       = -1,
                max_depth    = 15,
                min_samples_leaf = 3
            )
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)
            rf_acc  = accuracy_score(
                y_test, rf_pred
            ) * 100
            rf_f1   = f1_score(
                y_test, rf_pred,
                average='weighted'
            ) * 100

            # XGBoost
            sw  = get_sample_weights(y_train)
            xgb = XGBClassifier(
                n_estimators  = 200,
                random_state  = 42,
                verbosity     = 0,
                eval_metric   = 'logloss',
                max_depth     = 6,
                learning_rate = 0.05,
                subsample     = 0.8
            )
            xgb.fit(
                X_train, y_train,
                sample_weight=sw
            )
            xgb_pred = xgb.predict(X_test)
            xgb_acc  = accuracy_score(
                y_test, xgb_pred
            ) * 100
            xgb_f1   = f1_score(
                y_test, xgb_pred,
                average='weighted'
            ) * 100

            results.append({
                'Stock'   : name,
                'Target'  : label,
                'RF Acc'  : round(rf_acc,  2),
                'RF F1'   : round(rf_f1,   2),
                'XGB Acc' : round(xgb_acc, 2),
                'XGB F1'  : round(xgb_f1,  2),
            })

    except Exception as e:
        print(f"  Error with {name}: {e}")
        continue

# Print results
print("\n" + "="*55)
print("  RESULTS SUMMARY")
print("="*55)

print(f"\n  {'Stock':<12} {'Target':<8} "
      f"{'RF Acc':>8} {'XGB Acc':>9} {'Best':>8}")
print(f"  {'─'*12} {'─'*8} "
      f"{'─'*8} {'─'*9} {'─'*8}")

for r in results:
    best = max(r['RF Acc'], r['XGB Acc'])
    print(f"  {r['Stock']:<12} {r['Target']:<8} "
          f"{r['RF Acc']:>7.2f}% "
          f"{r['XGB Acc']:>8.2f}% "
          f"{best:>7.2f}%")

if results:
    best_results = [r for r in results
                    if r['Target'] == '20-day']
    if best_results:
        avg_rf  = np.mean([r['RF Acc']
                           for r in best_results])
        avg_xgb = np.mean([r['XGB Acc']
                           for r in best_results])
        print(f"\n  Average 20-day RF  : {avg_rf:.2f}%")
        print(f"  Average 20-day XGB : {avg_xgb:.2f}%")

        print(f"\n  Your current accuracy : ~34-36%")
        print(f"  NSE 20-day accuracy   : ~{avg_xgb:.1f}%")
        print(f"  Improvement           : "
              f"+{avg_xgb-35:.1f}%")

        if avg_xgb >= 72:
            print(f"\n  ✅ NSE + 20-day reaches 72%+")
            print(f"  ✅ Proceed with this approach")
            print(f"  ✅ Add LSTM to push to 75-80%")
        elif avg_xgb >= 62:
            print(f"\n  ✅ Good improvement achieved")
            print(f"  Add LSTM ensemble to push higher")
        else:
            print(f"\n  ⚠️  Below 62%")
            print(f"  Try longer target or LSTM only")

print("\n  Delete this file after reading results:")
print("  del temp_nse_20day_test.py")
print("="*55)