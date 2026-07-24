# temp_nifty_test.py
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.append('src/config')
from config import *

from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from collections import Counter

print("=== NIFTY RELATIVE STRENGTH TEST ===\n")

# Download Nifty 50 index
print("Downloading Nifty 50...")
nifty = yf.Ticker("^NSEI")
nifty_df = nifty.history(
    start="2015-01-01", end="2025-01-01"
)
nifty_df = nifty_df.reset_index()
nifty_df['Date'] = pd.to_datetime(
    nifty_df['Date']
).dt.tz_localize(None)
nifty_df = nifty_df[['Date','Close']].rename(
    columns={'Close':'nifty_close'}
)
nifty_df['nifty_return'] = nifty_df[
    'nifty_close'
].pct_change()
nifty_df['nifty_5d_return'] = nifty_df[
    'nifty_close'
].pct_change(5)
nifty_df['nifty_rsi'] = ta.rsi(
    nifty_df['nifty_close'], length=14
)
print(f"Nifty rows: {len(nifty_df)}")

# Load NSE dataset
df = pd.read_csv(
    os.path.join(DATA_PROCESSED_DIR,'nse_dataset.csv'),
    parse_dates=['Date']
)

# Test on POWERGRID and HINDUNILVR
for sym in ['POWERGRID.NS', 'HINDUNILVR.NS',
            'RELIANCE.NS']:
    name = sym.replace('.NS','')
    stock = df[
        df['Stock_Symbol']==sym
    ].copy().sort_values('Date').reset_index(drop=True)

    # Merge with Nifty
    stock = stock.merge(nifty_df, on='Date', how='left')
    stock['nifty_return']   = stock['nifty_return'].fillna(0)
    stock['nifty_5d_return']= stock['nifty_5d_return'].fillna(0)
    stock['nifty_rsi']      = stock['nifty_rsi'].fillna(50)

    # Relative strength features
    stock['stock_vs_nifty']    = (
        stock['Daily_Return'] - stock['nifty_return']
    )
    stock['stock_vs_nifty_5d'] = (
        stock['Return_5d'] - stock['nifty_5d_return']
    )
    stock['nifty_momentum']    = (
        stock['nifty_rsi'] - 50
    ) / 50

    with open(os.path.join(
        DATA_PROCESSED_DIR,'nse_feature_cols.txt'
    )) as f:
        base_feats = [
            l.strip() for l in f
            if l.strip() in stock.columns
        ]

    new_feats = [
        'nifty_return','nifty_5d_return',
        'nifty_rsi','stock_vs_nifty',
        'stock_vs_nifty_5d','nifty_momentum'
    ]
    all_feats = base_feats + new_feats

    TARGET = 'Binary_Target'
    train = stock[stock['Date']<='2021-12-31']
    test  = stock[stock['Date']>='2022-01-01']

    def get_w(y):
        c=Counter(y); t=len(y); n=len(c)
        w={k:t/(n*v) for k,v in c.items()}
        return np.array([w[i] for i in y])

    # Without Nifty
    X_tr = train[base_feats].values
    y_tr = train[TARGET].values.astype(int)
    X_te = test[base_feats].values
    y_te = test[TARGET].values.astype(int)
    sw   = get_w(y_tr)
    m1   = XGBClassifier(
        n_estimators=200,random_state=42,
        verbosity=0,eval_metric='logloss',
        max_depth=6,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.8
    )
    m1.fit(X_tr,y_tr,sample_weight=sw)
    acc1 = accuracy_score(
        y_te, m1.predict(X_te)
    ) * 100

    # With Nifty features
    X_tr2 = train[all_feats].values
    X_te2 = test[all_feats].values
    sw2   = get_w(y_tr)
    m2    = XGBClassifier(
        n_estimators=200,random_state=42,
        verbosity=0,eval_metric='logloss',
        max_depth=6,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.8
    )
    m2.fit(X_tr2,y_tr,sample_weight=sw2)
    acc2 = accuracy_score(
        y_te, m2.predict(X_te2)
    ) * 100

    print(f"{name}:")
    print(f"  Without Nifty : {acc1:.2f}%")
    print(f"  With Nifty    : {acc2:.2f}%")
    print(f"  Improvement   : {acc2-acc1:+.2f}%\n")

print("del temp_nifty_test.py")