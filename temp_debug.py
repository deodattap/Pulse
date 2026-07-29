# temp_debug.py
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.append('src/config')
from config import *
from sklearn.feature_selection import mutual_info_classif
from collections import Counter

print("="*60)
print("  DEEP DIAGNOSTIC")
print("="*60)

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

# Focus on POWERGRID
s = df[df['Stock_Symbol']=='POWERGRID.NS'].copy()
s = s.sort_values('Date').reset_index(drop=True)
train = s[s['Date']<='2021-12-31']

print(f"\n  POWERGRID training data:")
print(f"  Rows          : {len(train)}")
print(f"  Features      : {len(FEATS)}")
print(f"  Ratio rows/feats: {len(train)/len(FEATS):.1f}x")
print(f"  (Should be >10x for reliable MI)")

# Check MI window sizes
print(f"\n  MI Window size tests:")
print(f"  {'Window':>8} {'Rows':>6} "
      f"{'Top Feature':>20} {'Top Score':>10}")
print(f"  {'─'*8} {'─'*6} {'─'*20} {'─'*10}")

for window in [60, 126, 252, 504, len(train)]:
    data = train.tail(window)
    X = data[FEATS].values
    y = data[TARGET].values.astype(int)

    if len(np.unique(y)) < 2:
        continue

    mi = mutual_info_classif(
        X, y, random_state=42, n_neighbors=3
    )
    mi_dict = dict(zip(FEATS, mi))
    top_feat = max(mi_dict, key=mi_dict.get)
    top_score = mi_dict[top_feat]

    # Show top 3 features for this window
    top3 = sorted(
        mi_dict.items(),
        key=lambda x: x[1], reverse=True
    )[:3]
    top3_str = ', '.join([f[0] for f in top3])

    print(f"  {window:>8} {len(data):>6} "
          f"{top_feat:>20} {top_score:>10.4f}")
    print(f"           Top 3: {top3_str}")

# Check if features have NaN or constant values
print(f"\n  Feature quality check:")
print(f"  {'Feature':<25} {'NaN':>5} "
      f"{'Unique':>7} {'Std':>10} {'Issue'}")
print(f"  {'─'*25} {'─'*5} "
      f"{'─'*7} {'─'*10} {'─'*10}")

issues = []
for feat in FEATS:
    nan_count = s[feat].isnull().sum()
    unique    = s[feat].nunique()
    std       = s[feat].std()
    issue     = ''
    if nan_count > 0:
        issue = 'HAS NaN'
    elif unique < 10:
        issue = 'LOW UNIQUE'
    elif std < 0.0001:
        issue = 'NEAR CONSTANT'

    if issue:
        issues.append(feat)
        print(f"  {feat:<25} {nan_count:>5} "
              f"{unique:>7} {std:>10.4f} {issue}")

if not issues:
    print(f"  All {len(FEATS)} features look clean ✅")

# Check target quality
print(f"\n  Target quality check:")
print(f"  Total rows  : {len(s)}")
print(f"  Up   (1)    : {(s[TARGET]==1).sum()}")
print(f"  Down (0)    : {(s[TARGET]==0).sum()}")
print(f"  NaN target  : {s[TARGET].isnull().sum()}")

# Check walk-forward window vs feature ratio
print(f"\n  Walk-forward settings check:")
print(f"  MIN_TRAIN_ROWS : {504}")
print(f"  MI_WINDOW      : {252}")
print(f"  FEATURE COUNT  : {len(FEATS)}")
print(f"  RETRAIN_EVERY  : {60}")
print(f"  TOP_K          : {10}")
print(f"\n  Ratio MI_WINDOW/FEATURES: "
      f"{252/len(FEATS):.1f}x")
print(f"  Recommended minimum: 10x")
if 252/len(FEATS) < 10:
    print(f"  ⚠️  MI window too small for feature count!")
    print(f"  Increase MI_WINDOW to "
          f"{len(FEATS)*10} days minimum")
else:
    print(f"  ✅ Ratio is acceptable")

# Check feature correlation with target
print(f"\n  Feature-Target correlation (top 10):")
corrs = {}
for feat in FEATS:
    corr = abs(s[feat].corr(s[TARGET]))
    corrs[feat] = corr

top_corr = sorted(
    corrs.items(), key=lambda x: x[1], reverse=True
)[:10]
print(f"  {'Feature':<25} {'|Correlation|':>14}")
print(f"  {'─'*25} {'─'*14}")
for feat, corr in top_corr:
    bar = '█' * int(corr * 20)
    print(f"  {feat:<25} {corr:>14.4f}  {bar}")

# Check if XGBoost is overfitting
print(f"\n  Overfitting check (train vs test):")
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def get_w(y):
    c=Counter(y);t=len(y);n=len(c)
    w={k:t/(n*v) for k,v in c.items()}
    return np.array([w[i] for i in y])

test_s = s[s['Date']>='2022-01-01']
X_tr = train[FEATS].values
y_tr = train[TARGET].values.astype(int)
X_te = test_s[FEATS].values
y_te = test_s[TARGET].values.astype(int)

for depth in [3, 6, 10, 15]:
    sw = get_w(y_tr)
    m  = XGBClassifier(
        n_estimators=200, max_depth=depth,
        random_state=42, verbosity=0,
        eval_metric='logloss',
        learning_rate=0.05, subsample=0.8
    )
    m.fit(X_tr, y_tr, sample_weight=sw)
    tr_acc = accuracy_score(y_tr, m.predict(X_tr))*100
    te_acc = accuracy_score(y_te, m.predict(X_te))*100
    gap    = tr_acc - te_acc
    flag   = '⚠️ OVERFIT' if gap > 15 else '✅'
    print(f"  depth={depth:>2}: "
          f"train={tr_acc:.1f}%  "
          f"test={te_acc:.1f}%  "
          f"gap={gap:.1f}%  {flag}")

print("\n  del temp_debug.py")
print("="*60)