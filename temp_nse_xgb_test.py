# temp_nse_xgb_test.py
# Quick XGBoost test on NSE dataset
# Delete after running

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics  import accuracy_score, f1_score
from xgboost import XGBClassifier
from collections import Counter
import os, sys, warnings
warnings.filterwarnings('ignore')

sys.path.append('src/config')
from config import *

print("="*55)
print("  XGBoost BASELINE TEST ON NSE DATA")
print("="*55)

# Load NSE data
train_path = os.path.join(
    DATA_PROCESSED_DIR, 'nse_train.csv'
)
test_path  = os.path.join(
    DATA_PROCESSED_DIR, 'nse_test.csv'
)

train = pd.read_csv(train_path, parse_dates=['Date'])
test  = pd.read_csv(test_path,  parse_dates=['Date'])

print(f"\n  Train rows : {len(train):,}")
print(f"  Test rows  : {len(test):,}")

# Load feature columns
feat_path = os.path.join(
    DATA_PROCESSED_DIR, 'nse_feature_cols.txt'
)
with open(feat_path, 'r') as f:
    feature_cols = [l.strip() for l in f.readlines()]

# Filter to existing columns
feature_cols = [
    c for c in feature_cols if c in train.columns
]
print(f"  Features   : {len(feature_cols)}")

TARGET = 'Binary_Target'

def get_weights(y):
    counts = Counter(y)
    total  = len(y)
    n      = len(counts)
    wmap   = {
        c: total/(n*cnt) for c, cnt in counts.items()
    }
    return np.array([wmap[l] for l in y])

# ── Test on individual stocks ──────────────────────
print(f"\n  Per-stock results (Static XGB):")
print(f"  {'Stock':<15} {'Rows':>5} "
      f"{'Acc':>8} {'F1':>7}")
print(f"  {'─'*15} {'─'*5} "
      f"{'─'*8} {'─'*7}")

all_accs = []

for sym in train['Stock_Symbol'].unique():

    tr = train[train['Stock_Symbol'] == sym]
    te = test[test['Stock_Symbol']   == sym]

    if len(tr) < 100 or len(te) < 30:
        continue

    X_tr = tr[feature_cols].values
    y_tr = tr[TARGET].values
    X_te = te[feature_cols].values
    y_te = te[TARGET].values

    # XGBoost
    sw  = get_weights(y_tr)
    xgb = XGBClassifier(
        n_estimators  = 200,
        random_state  = 42,
        verbosity     = 0,
        eval_metric   = 'logloss',
        max_depth     = 6,
        learning_rate = 0.05,
        subsample     = 0.8,
        colsample_bytree = 0.8
    )
    xgb.fit(X_tr, y_tr, sample_weight=sw)
    pred = xgb.predict(X_te)

    acc = accuracy_score(y_te, pred) * 100
    f1  = f1_score(
        y_te, pred, average='weighted'
    ) * 100

    all_accs.append(acc)
    name = sym.replace('.NS', '')
    print(f"  {name:<15} {len(te):>5} "
          f"{acc:>7.2f}% {f1:>6.2f}%")

print(f"  {'─'*15} {'─'*5} "
      f"{'─'*8} {'─'*7}")
avg = np.mean(all_accs)
mx  = np.max(all_accs)
mn  = np.min(all_accs)
print(f"  {'AVERAGE':<15} {'':>5} "
      f"{avg:>7.2f}%")
print(f"  {'MAX':<15} {'':>5} "
      f"{mx:>7.2f}%")
print(f"  {'MIN':<15} {'':>5} "
      f"{mn:>7.2f}%")

print(f"\n  Previous US stock accuracy : ~34-36%")
print(f"  NSE XGBoost accuracy       : {avg:.2f}%")
print(f"  Improvement                : +{avg-35:.1f}%")

if avg >= 65:
    print(f"\n  ✅ Strong baseline achieved")
    print(f"  ✅ LSTM will push this to 75-80%")
elif avg >= 58:
    print(f"\n  ✅ Good baseline")
    print(f"  LSTM needed to reach 75-80%")
else:
    print(f"\n  ⚠️  Lower than expected")
    print(f"  Tell mentor and discuss next steps")