# temp_feature_test.py
# Tests which feature groups are most predictive
# Delete after running

import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.append('src/config')
from config import *

from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

# Load data
df = pd.read_csv(
    os.path.join(DATA_PROCESSED_DIR, 'nse_dataset.csv'),
    parse_dates=['Date']
)

# POWERGRID only
pg = df[df['Stock_Symbol']=='POWERGRID.NS'].copy()
pg = pg.sort_values('Date').reset_index(drop=True)

train = pg[pg['Date'] <= '2021-12-31']
test  = pg[pg['Date'] >= '2022-01-01']

TARGET = 'Binary_Target'

with open(os.path.join(
    DATA_PROCESSED_DIR,'nse_feature_cols.txt'
)) as f:
    ALL_FEATS = [
        l.strip() for l in f
        if l.strip() in df.columns
    ]

# Define feature groups
TREND_FEATS = [
    'SMA_10','SMA_20','SMA_50','EMA_12','EMA_26'
]
MOMENTUM_FEATS = [
    'RSI_14','MACD','MACD_Signal','MACD_Hist',
    'ROC','Williams_R','Stochastic'
]
VOLATILITY_FEATS = [
    'ATR_14','BB_Upper','BB_Lower','BB_Width'
]
VOLUME_FEATS = [
    'OBV','Volume_MA','Volume_Ratio'
]
RETURN_FEATS = [
    'Daily_Return','Log_Return','Return_5d',
    'Return_10d','Rolling_Std_10',
    'Rolling_Volatility'
]
OTHER_FEATS = [
    'CCI','ADX','MFI','High_Low_Pct',
    'Open_Close_Pct','Price_Change',
    'Rolling_Mean_Return','Rolling_Median_Return'
]

def get_weights(y):
    c = Counter(y)
    t = len(y)
    n = len(c)
    w = {k: t/(n*v) for k,v in c.items()}
    return np.array([w[i] for i in y])

def test_feature_group(name, feats):
    feats = [f for f in feats if f in ALL_FEATS]
    if len(feats) == 0:
        return 0

    X_tr = train[feats].values
    y_tr = train[TARGET].values.astype(int)
    X_te = test[feats].values
    y_te = test[TARGET].values.astype(int)

    sw  = get_weights(y_tr)
    xgb = XGBClassifier(
        n_estimators=200, random_state=42,
        verbosity=0, eval_metric='logloss',
        max_depth=6, learning_rate=0.05
    )
    xgb.fit(X_tr, y_tr, sample_weight=sw)
    pred = xgb.predict(X_te)
    acc  = accuracy_score(y_te, pred) * 100
    return acc

print("="*50)
print("  FEATURE GROUP IMPORTANCE TEST")
print("  POWERGRID.NS")
print("="*50)

# Test each group
groups = {
    'ALL features (33)'    : ALL_FEATS,
    'Trend only (SMA/EMA)' : TREND_FEATS,
    'Momentum only'        : MOMENTUM_FEATS,
    'Volatility only'      : VOLATILITY_FEATS,
    'Volume only'          : VOLUME_FEATS,
    'Returns only'         : RETURN_FEATS,
    'Other (CCI/ADX/MFI)' : OTHER_FEATS,
    'No Trend (remove SMA/EMA)': [
        f for f in ALL_FEATS
        if f not in TREND_FEATS
    ],
    'Momentum+Volatility+Volume': (
        MOMENTUM_FEATS + VOLATILITY_FEATS +
        VOLUME_FEATS
    ),
}

print(f"\n  {'Feature Group':<30} {'Accuracy':>10}")
print(f"  {'─'*30} {'─'*10}")

results = {}
for name, feats in groups.items():
    feats = [f for f in feats if f in ALL_FEATS]
    if len(feats) == 0:
        continue
    acc = test_feature_group(name, feats)
    results[name] = acc
    print(f"  {name:<30} {acc:>9.2f}%")

print(f"\n  Best group: "
      f"{max(results, key=results.get)}")
print(f"  Best accuracy: "
      f"{max(results.values()):.2f}%")

# Also show MI scores for each feature
print(f"\n  MI Scores (top 15):")
X_mi = train[ALL_FEATS].values
y_mi = train[TARGET].values.astype(int)
mi   = mutual_info_classif(
    X_mi, y_mi, random_state=42
)
mi_dict = dict(zip(ALL_FEATS, mi))
mi_sorted = sorted(
    mi_dict.items(), key=lambda x: x[1],
    reverse=True
)
print(f"  {'Feature':<25} {'MI Score':>10}")
print(f"  {'─'*25} {'─'*10}")
for feat, score in mi_sorted[:15]:
    bar = '█' * int(score * 100)
    print(f"  {feat:<25} {score:>10.4f}  {bar}")