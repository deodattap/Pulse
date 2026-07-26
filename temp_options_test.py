# temp_options_test.py
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.append('src/config')
from config import *
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

df = pd.read_csv(
    os.path.join(DATA_PROCESSED_DIR,'nse_dataset.csv'),
    parse_dates=['Date']
)

with open(os.path.join(
    DATA_PROCESSED_DIR,'nse_feature_cols.txt'
)) as f:
    FEATS = [l.strip() for l in f
             if l.strip() in df.columns]

TARGET = 'Binary_Target'
TEST_STOCKS = [
    'POWERGRID.NS','HINDUNILVR.NS','RELIANCE.NS'
]

def get_w(y):
    c=Counter(y);t=len(y);n=len(c)
    w={k:t/(n*v) for k,v in c.items()}
    return np.array([w[i] for i in y])

def test_acc(tr, te, feats):
    X_tr=tr[feats].values
    y_tr=tr[TARGET].values.astype(int)
    X_te=te[feats].values
    y_te=te[TARGET].values.astype(int)
    sw=get_w(y_tr)
    m=XGBClassifier(
        n_estimators=200,random_state=42,
        verbosity=0,eval_metric='logloss',
        max_depth=6,learning_rate=0.05,
        subsample=0.8,colsample_bytree=0.8
    )
    m.fit(X_tr,y_tr,sample_weight=sw)
    return accuracy_score(y_te,m.predict(X_te))*100

# ── Option 3: India VIX ────────────────────────────
print("Downloading India VIX...")
try:
    vix = yf.Ticker("^INDIAVIX")
    vix_df = vix.history(
        start="2015-01-01",end="2025-01-01"
    ).reset_index()
    vix_df['Date'] = pd.to_datetime(
        vix_df['Date']
    ).dt.tz_localize(None)
    vix_df = vix_df[['Date','Close']].rename(
        columns={'Close':'india_vix'}
    )
    vix_df['vix_change'] = vix_df['india_vix'].pct_change()
    vix_df['vix_ma']     = vix_df['india_vix'].rolling(10).mean()
    vix_df['vix_ratio']  = vix_df['india_vix'] / vix_df['vix_ma']
    VIX_FEATS = ['india_vix','vix_change','vix_ratio']
    HAS_VIX = True
    print(f"VIX rows: {len(vix_df)}")
except:
    HAS_VIX = False
    print("VIX download failed")

# ── Option 5: Weekly features ──────────────────────
print("\nComputing weekly features...")

print("\n" + "="*55)
print(f"  {'Stock':<12} {'Base':>7} "
      f"{'+VIX':>7} {'+Weekly':>9} {'Best':>9}")
print("="*55)

for sym in TEST_STOCKS:
    name = sym.replace('.NS','')
    s = df[df['Stock_Symbol']==sym].copy()
    s = s.sort_values('Date').reset_index(drop=True)

    # Add VIX
    if HAS_VIX:
        s = s.merge(vix_df, on='Date', how='left')
        for c in VIX_FEATS:
            s[c] = s[c].ffill().fillna(20)

    # Add weekly features
    # Weekly = 5-day rolling of daily indicators
    s['weekly_rsi']    = s['RSI_14'].rolling(5).mean()
    s['weekly_macd']   = s['MACD'].rolling(5).mean()
    s['weekly_atr']    = s['ATR_14'].rolling(5).mean()
    s['weekly_volume'] = s['Volume_Ratio'].rolling(5).mean()
    s['weekly_bb']     = s['BB_Width'].rolling(5).mean()
    WEEKLY_FEATS = [
        'weekly_rsi','weekly_macd','weekly_atr',
        'weekly_volume','weekly_bb'
    ]

    s = s.dropna()
    tr = s[s['Date']<='2021-12-31']
    te = s[s['Date']>='2022-01-01']

    if len(tr) < 100 or len(te) < 50:
        continue

    # Base
    base = test_acc(tr, te, FEATS)

    # With VIX
    vix_feats = FEATS + VIX_FEATS if HAS_VIX else FEATS
    vix_acc   = test_acc(tr, te, vix_feats) if HAS_VIX else base

    # With Weekly
    weekly_feats = FEATS + WEEKLY_FEATS
    weekly_acc   = test_acc(tr, te, weekly_feats)

    best = max(base, vix_acc, weekly_acc)
    print(f"  {name:<12} {base:>6.2f}% "
          f"{vix_acc:>6.2f}% "
          f"{weekly_acc:>8.2f}% "
          f"{best:>8.2f}%")

print("\ndel temp_options_test.py")