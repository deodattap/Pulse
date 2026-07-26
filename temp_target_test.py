# temp_target_test.py
import pandas as pd
import numpy as np
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

def get_w(y):
    c=Counter(y);t=len(y);n=len(c)
    w={k:t/(n*v) for k,v in c.items()}
    return np.array([w[i] for i in y])

print("=== TARGET HORIZON TEST ===\n")
print(f"{'Stock':<12} {'5-day':>7} {'10-day':>8} "
      f"{'20-day':>8} {'Best':>8}")
print("-"*50)

for sym in ['POWERGRID.NS','HINDUNILVR.NS',
            'KOTAKBANK.NS','RELIANCE.NS']:
    name = sym.replace('.NS','')
    s = df[df['Stock_Symbol']==sym].copy()
    s = s.sort_values('Date').reset_index(drop=True)

    results = []
    for days in [5, 10, 20]:
        s['target'] = (
            s['Close'].shift(-days) > s['Close']
        ).astype(int)
        s2 = s.dropna(subset=['target'])

        tr = s2[s2['Date']<='2021-12-31']
        te = s2[s2['Date']>='2022-01-01']

        X_tr = tr[FEATS].values
        y_tr = tr['target'].values.astype(int)
        X_te = te[FEATS].values
        y_te = te['target'].values.astype(int)

        sw = get_w(y_tr)
        m  = XGBClassifier(
            n_estimators=200,random_state=42,
            verbosity=0,eval_metric='logloss',
            max_depth=6,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8
        )
        m.fit(X_tr,y_tr,sample_weight=sw)
        acc = accuracy_score(
            y_te,m.predict(X_te)
        )*100
        results.append(acc)

    best = max(results)
    best_label = ['5d','10d','20d'][
        results.index(best)
    ]
    print(f"  {name:<12} "
          f"{results[0]:>6.2f}% "
          f"{results[1]:>7.2f}% "
          f"{results[2]:>7.2f}% "
          f"{best:>6.2f}%({best_label})")