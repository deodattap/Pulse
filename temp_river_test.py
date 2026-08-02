# temp_river_test.py
# Tests proper streaming ML algorithms
# from the River library
# Delete after running

import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.append('src/config')
from config import *

print("="*55)
print("  RIVER STREAMING ML TEST")
print("="*55)

try:
    from river import (
        tree, ensemble, naive_bayes,
        preprocessing, metrics, drift
    )
    print("  River library: OK ✅\n")
except ImportError:
    print("  Install river: pip install river")
    sys.exit(1)

# Load data
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
    'POWERGRID.NS',
    'HDFCBANK.NS',
    'KOTAKBANK.NS'
]

print(f"  {'Stock':<12} {'HoeffdingTree':>14} "
      f"{'AdaptiveRF':>11} {'NaiveBayes':>11}")
print(f"  {'─'*12} {'─'*14} "
      f"{'─'*11} {'─'*11}")

for sym in TEST_STOCKS:
    name = sym.replace('.NS','')
    s = df[df['Stock_Symbol']==sym].copy()
    s = s.sort_values('Date').reset_index(drop=True)

    # Models
    ht_model = (
        preprocessing.StandardScaler() |
        tree.HoeffdingTreeClassifier(
            grace_period=100,
            delta=0.01
        )
    )
    arf_model = (
        preprocessing.StandardScaler() |
        ensemble.ADWINBaggingClassifier(
            model=tree.HoeffdingTreeClassifier(),
            n_models=10,
            seed=42
        )
    )
    nb_model = (
        preprocessing.StandardScaler() |
        naive_bayes.GaussianNB()
    )

    # Metrics
    ht_metric  = metrics.Accuracy()
    arf_metric = metrics.Accuracy()
    nb_metric  = metrics.Accuracy()

    # Stream through ALL data
    # This is true streaming — one sample at a time
    for _, row in s.iterrows():
        x = {f: row[f] for f in FEATS}
        y = int(row[TARGET])

        # Predict then learn (prequential evaluation)
        ht_pred  = ht_model.predict_one(x)
        arf_pred = arf_model.predict_one(x)
        nb_pred  = nb_model.predict_one(x)

        if ht_pred  is not None:
            ht_metric.update(y, ht_pred)
        if arf_pred is not None:
            arf_metric.update(y, arf_pred)
        if nb_pred  is not None:
            nb_metric.update(y, nb_pred)

        # Learn from this sample
        ht_model.learn_one(x, y)
        arf_model.learn_one(x, y)
        nb_model.learn_one(x, y)

    ht_acc  = ht_metric.get()  * 100
    arf_acc = arf_metric.get() * 100
    nb_acc  = nb_metric.get()  * 100

    print(f"  {name:<12} "
          f"{ht_acc:>13.2f}% "
          f"{arf_acc:>10.2f}% "
          f"{nb_acc:>10.2f}%")

print(f"\n  del temp_river_test.py")
print("="*55)