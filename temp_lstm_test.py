# temp_lstm_test.py
# Tests LSTM on top NSE stocks
# Compares with XGBoost
# Delete after running

import pandas as pd
import numpy as np
import os, sys, warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append('src/config')
from config import *

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
from collections import Counter

print("="*55)
print("  LSTM vs XGBoost TEST — TOP NSE STOCKS")
print("="*55)

# Check TensorFlow
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping
    )
    HAS_TF = True
    print(f"\n  TensorFlow: {tf.__version__} ✅")
except ImportError:
    HAS_TF = False
    print("\n  ⚠️  TensorFlow not installed")
    print("  Run: pip install tensorflow")
    sys.exit(1)

# Load data
train_path = os.path.join(
    DATA_PROCESSED_DIR, 'nse_train.csv'
)
test_path  = os.path.join(
    DATA_PROCESSED_DIR, 'nse_test.csv'
)

train_df = pd.read_csv(
    train_path, parse_dates=['Date']
)
test_df  = pd.read_csv(
    test_path,  parse_dates=['Date']
)

feat_path = os.path.join(
    DATA_PROCESSED_DIR, 'nse_feature_cols.txt'
)
with open(feat_path) as f:
    FEAT_COLS = [
        l.strip() for l in f
        if l.strip() in train_df.columns
    ]

TARGET   = 'Binary_Target'
SEQ_LEN  = 60   # 60-day sequences

# Test on top 3 performing stocks
TEST_STOCKS = [
    'POWERGRID.NS',
    'KOTAKBANK.NS',
    'INFY.NS'
]

def get_weights(y):
    c = Counter(y)
    t = len(y)
    n = len(c)
    w = {k: t/(n*v) for k, v in c.items()}
    return np.array([w[i] for i in y])


def make_sequences(X, y, seq_len):
    """
    Convert flat features into sequences.
    Each sample = last seq_len days of features.
    """
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def build_lstm(seq_len, n_features):
    """Build simple 2-layer LSTM."""
    model = Sequential([
        LSTM(64, input_shape=(seq_len, n_features),
             return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1,  activation='sigmoid')
    ])
    model.compile(
        optimizer = 'adam',
        loss      = 'binary_crossentropy',
        metrics   = ['accuracy']
    )
    return model


results = []

for sym in TEST_STOCKS:
    name = sym.replace('.NS', '')
    print(f"\n  {'─'*50}")
    print(f"  Testing: {name}")
    print(f"  {'─'*50}")

    tr = train_df[
        train_df['Stock_Symbol'] == sym
    ].sort_values('Date')
    te = test_df[
        test_df['Stock_Symbol'] == sym
    ].sort_values('Date')

    if len(tr) < 200 or len(te) < 60:
        print(f"  Not enough data")
        continue

    X_tr = tr[FEAT_COLS].values.astype(np.float32)
    y_tr = tr[TARGET].values.astype(np.float32)
    X_te = te[FEAT_COLS].values.astype(np.float32)
    y_te = te[TARGET].values.astype(np.float32)

    # Scale features (important for LSTM)
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    # ── XGBoost baseline ──────────────────────────
    print(f"  Training XGBoost...")
    sw  = get_weights(y_tr.astype(int))
    xgb = XGBClassifier(
        n_estimators     = 200,
        random_state     = 42,
        verbosity        = 0,
        eval_metric      = 'logloss',
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8
    )
    xgb.fit(X_tr, y_tr.astype(int),
            sample_weight=sw)

    xgb_pred = xgb.predict(X_te)
    xgb_acc  = accuracy_score(
        y_te.astype(int), xgb_pred
    ) * 100
    xgb_f1   = f1_score(
        y_te.astype(int), xgb_pred,
        average='weighted'
    ) * 100
    print(f"  XGBoost → Acc: {xgb_acc:.2f}%  "
          f"F1: {xgb_f1:.2f}%")

    # ── LSTM ──────────────────────────────────────
    print(f"  Building LSTM sequences...")

    # Combine train+test for scaling then split
    X_all    = np.vstack([X_tr_sc, X_te_sc])
    y_all    = np.concatenate([y_tr, y_te])
    n_train  = len(X_tr_sc)

    X_seq, y_seq = make_sequences(
        X_all, y_all, SEQ_LEN
    )

    # Split back into train/test
    # Subtract SEQ_LEN because sequences start after
    split_idx   = n_train - SEQ_LEN
    X_lstm_tr   = X_seq[:split_idx]
    y_lstm_tr   = y_seq[:split_idx]
    X_lstm_te   = X_seq[split_idx:]
    y_lstm_te   = y_seq[split_idx:]

    if len(X_lstm_tr) < 100 or len(X_lstm_te) < 30:
        print(f"  Not enough sequence data")
        continue

    print(f"  LSTM train sequences : {len(X_lstm_tr)}")
    print(f"  LSTM test sequences  : {len(X_lstm_te)}")

    # Class weights for LSTM
    n_pos = y_lstm_tr.sum()
    n_neg = len(y_lstm_tr) - n_pos
    w_pos = len(y_lstm_tr) / (2 * n_pos) if n_pos > 0 else 1
    w_neg = len(y_lstm_tr) / (2 * n_neg) if n_neg > 0 else 1
    class_weight = {0: w_neg, 1: w_pos}

    print(f"  Training LSTM...")
    model = build_lstm(SEQ_LEN, len(FEAT_COLS))

    es = EarlyStopping(
        monitor   = 'val_loss',
        patience  = 5,
        restore_best_weights = True
    )

    model.fit(
        X_lstm_tr, y_lstm_tr,
        epochs          = 30,
        batch_size      = 32,
        validation_split= 0.15,
        class_weight    = class_weight,
        callbacks       = [es],
        verbose         = 0
    )

    # Predict
    lstm_prob  = model.predict(
        X_lstm_te, verbose=0
    ).flatten()
    lstm_pred  = (lstm_prob > 0.5).astype(int)

    lstm_acc   = accuracy_score(
        y_lstm_te.astype(int), lstm_pred
    ) * 100
    lstm_f1    = f1_score(
        y_lstm_te.astype(int), lstm_pred,
        average='weighted'
    ) * 100

    print(f"  LSTM    → Acc: {lstm_acc:.2f}%  "
          f"F1: {lstm_f1:.2f}%")

    # ── Ensemble ──────────────────────────────────
    # Align XGBoost predictions with LSTM test size
    xgb_prob_te = xgb.predict_proba(
        X_te[-len(lstm_prob):]
    )[:, 1]

    ens_prob = (lstm_prob + xgb_prob_te) / 2
    ens_pred = (ens_prob > 0.5).astype(int)

    ens_acc  = accuracy_score(
        y_lstm_te.astype(int), ens_pred
    ) * 100
    ens_f1   = f1_score(
        y_lstm_te.astype(int), ens_pred,
        average='weighted'
    ) * 100

    print(f"  Ensemble→ Acc: {ens_acc:.2f}%  "
          f"F1: {ens_f1:.2f}%")

    improvement = ens_acc - xgb_acc
    print(f"\n  XGBoost → LSTM Ensemble improvement: "
          f"+{improvement:.2f}%")

    results.append({
        'Stock'   : name,
        'XGB Acc' : round(xgb_acc,  2),
        'LSTM Acc': round(lstm_acc, 2),
        'Ens Acc' : round(ens_acc,  2),
        'XGB F1'  : round(xgb_f1,   2),
        'LSTM F1' : round(lstm_f1,  2),
        'Ens F1'  : round(ens_f1,   2),
    })

    # Clean up
    del model
    import gc
    gc.collect()

# Summary
print(f"\n" + "="*55)
print(f"  FINAL SUMMARY")
print(f"="*55)

if results:
    print(f"\n  {'Stock':<12} {'XGB':>8} "
          f"{'LSTM':>8} {'Ensemble':>10}")
    print(f"  {'─'*12} {'─'*8} "
          f"{'─'*8} {'─'*10}")

    for r in results:
        print(f"  {r['Stock']:<12} "
              f"{r['XGB Acc']:>7.2f}% "
              f"{r['LSTM Acc']:>7.2f}% "
              f"{r['Ens Acc']:>9.2f}%")

    avg_xgb = np.mean([r['XGB Acc'] for r in results])
    avg_lstm = np.mean([r['LSTM Acc'] for r in results])
    avg_ens  = np.mean([r['Ens Acc'] for r in results])

    print(f"\n  Average XGBoost  : {avg_xgb:.2f}%")
    print(f"  Average LSTM     : {avg_lstm:.2f}%")
    print(f"  Average Ensemble : {avg_ens:.2f}%")

    print(f"\n  LSTM improvement over XGBoost : "
          f"+{avg_lstm-avg_xgb:.2f}%")
    print(f"  Ensemble improvement          : "
          f"+{avg_ens-avg_xgb:.2f}%")

    if avg_ens >= 72:
        print(f"\n  ✅ Target 75-80% achievable")
        print(f"  ✅ Proceed with LSTM + Ensemble")
        print(f"  ✅ Focus on top stocks")
    elif avg_ens >= 65:
        print(f"\n  ✅ Good improvement")
        print(f"  Add dynamic feature selection")
        print(f"  Should push to 70-75%")
    else:
        print(f"\n  ⚠️  More work needed")
        print(f"  Will discuss next steps")

print(f"\n  Delete: del temp_lstm_test.py")
print(f"="*55)