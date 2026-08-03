# temp_stream_compare.py
import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.append('src/config')
from config import *

from river import (
    tree, ensemble, neighbors,
    preprocessing, metrics, drift
)

print("="*60)
print("  STREAMING ALGORITHM COMPARISON")
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

TARGET   = 'Binary_Target'
TEST_START = '2022-01-01'

STOCKS = [
    'POWERGRID.NS','HDFCBANK.NS',
    'HINDUNILVR.NS','RELIANCE.NS',
    'INFY.NS','KOTAKBANK.NS',
    'TCS.NS','ICICIBANK.NS'
]

def make_models():
    return {
        'HoeffdingTree': (
            preprocessing.StandardScaler() |
            tree.HoeffdingTreeClassifier(
                grace_period=50, delta=0.01
            )
        ),
        'AdaptiveRF': (
            preprocessing.StandardScaler() |
            ensemble.ADWINBaggingClassifier(
                model=tree.HoeffdingTreeClassifier(
                    grace_period=50
                ),
                n_models=10, seed=42
            )
        ),
        'SAMKNN': (
            preprocessing.StandardScaler() |
            neighbors.KNNClassifier(
         n_neighbors=5
         )
        ),
        'SRPClassifier': (
            preprocessing.StandardScaler() |
            ensemble.SRPClassifier(
                model=tree.HoeffdingTreeClassifier(),
                n_models=10,
                seed=42
            )
        ),
    }

print(f"\n  {'Stock':<12}", end='')
model_names = [
    'HoeffdingTree','AdaptiveRF',
    'SAMKNN','SRPClassifier'
]
for n in model_names:
    print(f" {n[:12]:>13}", end='')
print()
print(f"  {'─'*12}", end='')
for _ in model_names:
    print(f" {'─'*13}", end='')
print()

all_accs = {n: [] for n in model_names}

for sym in STOCKS:
    name = sym.replace('.NS','')
    s = df[df['Stock_Symbol']==sym].copy()
    s = s.sort_values('Date').reset_index(drop=True)

    models  = make_models()
    correct = {n: 0 for n in model_names}
    total   = 0

    for _, row in s.iterrows():
        x = {f: float(row[f]) for f in FEATS}
        y = int(row[TARGET])
        is_test = row['Date'] >= pd.Timestamp(TEST_START)

        preds = {}
        for mname, model in models.items():
            pred = model.predict_one(x)
            preds[mname] = pred if pred is not None else 1

        for mname, model in models.items():
            model.learn_one(x, y)

        if is_test:
            total += 1
            for mname in model_names:
                if preds[mname] == y:
                    correct[mname] += 1

    print(f"  {name:<12}", end='')
    for mname in model_names:
        acc = correct[mname]/total*100 if total > 0 else 0
        all_accs[mname].append(acc)
        print(f" {acc:>12.2f}%", end='')
    print()

print(f"  {'─'*12}", end='')
for _ in model_names:
    print(f" {'─'*13}", end='')
print()
print(f"  {'AVERAGE':<12}", end='')
for mname in model_names:
    avg = np.mean(all_accs[mname])
    print(f" {avg:>12.2f}%", end='')
print()

best = max(model_names,
           key=lambda n: np.mean(all_accs[n]))
print(f"\n  Best algorithm: {best}")
print(f"  Best accuracy:  "
      f"{np.mean(all_accs[best]):.2f}%")
print(f"\n  del temp_stream_compare.py")
print("="*60)