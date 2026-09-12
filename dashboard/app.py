# ============================================================
# dashboard/app.py
# PULSE — Adaptive Streaming Stock Market Prediction
#
# Option C Dashboard:
#   Tab 1: Live prediction for any NSE stock
#   Tab 2: Historical analysis for 8 research stocks
#   Page 3: Feature Analysis
#   Page 4: Research Results
#   Page 5: About and Methodology
#
# How to run:
#   streamlit run dashboard/app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import time
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

sys.path.append(
    os.path.join(
        os.path.dirname(__file__), '..', 'src', 'config'
    )
)
from config import *

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title = "PULSE — Stock Prediction",
    page_icon  = "📈",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
    .pulse-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(
            90deg, #00C9FF 0%, #92FE9D 100%
        );
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .pulse-sub {
        text-align: center;
        color: #888;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .pred-card-buy {
        background: linear-gradient(
            135deg, #1a472a, #2d6a4f
        );
        border: 2px solid #4CAF50;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .pred-card-sell {
        background: linear-gradient(
            135deg, #4a1515, #7b2929
        );
        border: 2px solid #F44336;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .pred-card-notrade {
        background: linear-gradient(
            135deg, #2a2a2a, #3a3a3a
        );
        border: 2px solid #888;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
    }
    .signal-text {
        font-size: 3rem;
        font-weight: 900;
        margin: 0.5rem 0;
    }
    .info-box {
        background: rgba(33, 150, 243, 0.1);
        border-left: 4px solid #2196F3;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .warning-box {
        background: rgba(255, 152, 0, 0.1);
        border-left: 4px solid #FF9800;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .success-box {
        background: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }
    .feature-chip {
        display: inline-block;
        background: rgba(76, 175, 80, 0.2);
        border: 1px solid #4CAF50;
        border-radius: 20px;
        padding: 4px 12px;
        margin: 4px;
        font-size: 0.85rem;
        color: #4CAF50;
    }
    .step-box {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .ticker-help {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 0.8rem;
        font-size: 0.85rem;
        color: #aaa;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================

CAT_MAP = {
    'SMA_10':'Trend','SMA_20':'Trend',
    'SMA_50':'Trend','EMA_12':'Trend',
    'EMA_26':'Trend',
    'RSI_14':'Momentum','MACD':'Momentum',
    'MACD_Signal':'Momentum',
    'MACD_Hist':'Momentum','ROC':'Momentum',
    'Williams_R':'Momentum','Stochastic':'Momentum',
    'ATR_14':'Volatility','BB_Upper':'Volatility',
    'BB_Lower':'Volatility','BB_Width':'Volatility',
    'OBV':'Volume','Volume_MA':'Volume',
    'Volume_Ratio':'Volume',
    'CCI':'Statistical','ADX':'Statistical',
    'MFI':'Statistical',
    'Daily_Return':'Returns',
    'Log_Return':'Returns',
    'Return_5d':'Returns',
    'Return_10d':'Returns',
    'Rolling_Std_10':'Returns',
    'Rolling_Volatility':'Returns',
    'High_Low_Pct':'Price Pattern',
    'Open_Close_Pct':'Price Pattern',
    'Price_Change':'Price Pattern',
    'Rolling_Mean_Return':'Returns',
    'Rolling_Median_Return':'Returns'
}

FEAT_DESC = {
    'RSI_14'             : 'RSI — measures momentum strength and overbought/oversold conditions',
    'MACD'               : 'MACD — captures trend direction and momentum crossovers',
    'MACD_Hist'          : 'MACD Histogram — shows momentum shift strength',
    'MACD_Signal'        : 'MACD Signal — smoothed MACD for crossover detection',
    'ATR_14'             : 'ATR — measures market volatility and price range',
    'BB_Width'           : 'Bollinger Width — indicates volatility expansion or contraction',
    'BB_Upper'           : 'Bollinger Upper Band — price resistance level',
    'BB_Lower'           : 'Bollinger Lower Band — price support level',
    'OBV'                : 'OBV — tracks volume flow relative to price direction',
    'Volume_Ratio'       : 'Volume Ratio — compares current to average trading volume',
    'Volume_MA'          : 'Volume MA — smoothed volume trend',
    'SMA_10'             : 'SMA-10 — short-term price trend average',
    'SMA_20'             : 'SMA-20 — medium-short price trend average',
    'SMA_50'             : 'SMA-50 — medium-term price trend average',
    'EMA_12'             : 'EMA-12 — fast exponential trend indicator',
    'EMA_26'             : 'EMA-26 — slow exponential trend indicator',
    'ROC'                : 'Rate of Change — measures price momentum over 10 days',
    'CCI'                : 'CCI — identifies cyclical turns in price',
    'ADX'                : 'ADX — measures trend strength regardless of direction',
    'MFI'                : 'MFI — money flow pressure using price and volume',
    'Williams_R'         : 'Williams %R — overbought and oversold oscillator',
    'Stochastic'         : 'Stochastic — compares closing price to recent range',
    'Rolling_Volatility' : 'Rolling Volatility — recent price instability measure',
    'Daily_Return'       : 'Daily Return — most recent single-day price change',
    'Return_5d'          : '5-Day Return — short-term cumulative return',
    'Return_10d'         : '10-Day Return — medium-short cumulative return',
    'Rolling_Std_10'     : 'Rolling Std-10 — price variability over 10 days',
    'Rolling_Mean_Return': 'Rolling Mean Return — average return persistence',
    'Rolling_Median_Return':'Rolling Median Return — robust return trend estimate',
    'High_Low_Pct'       : 'High-Low % — daily price range relative to close',
    'Open_Close_Pct'     : 'Open-Close % — intraday directional pressure',
    'Price_Change'       : 'Price Change — absolute daily price movement',
    'Log_Return'         : 'Log Return — logarithmic daily return for stability',
}

REGIME_MSGS = {
    'Trend'        : 'Trend indicators are dominant, suggesting a clear directional price movement is underway.',
    'Momentum'     : 'Momentum indicators are showing strong directional pressure in the market.',
    'Volatility'   : 'Volatility indicators are the primary signal, suggesting elevated market uncertainty.',
    'Volume'       : 'Volume-based indicators dominate, indicating strong market participation confirming the move.',
    'Returns'      : 'Recent return patterns are the strongest predictors, suggesting momentum continuation.',
    'Statistical'  : 'Statistical indicators show a measurable pattern in current market conditions.',
    'Price Pattern': 'Price pattern features are dominant, indicating significant intraday movement patterns.',
}

POPULAR_STOCKS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK',
    'ICICIBANK', 'KOTAKBANK', 'HINDUNILVR',
    'POWERGRID', 'BAJFINANCE', 'WIPRO',
    'HCLTECH', 'AXISBANK', 'SBIN', 'MARUTI',
    'TITAN', 'NESTLEIND', 'ASIANPAINT',
    'BAJAJFINSV', 'TECHM', 'ULTRACEMCO',
    'NTPC', 'ONGC', 'COALINDIA', 'TATASTEEL',
    'TATAMOTORS', 'SUNPHARMA', 'DRREDDY',
    'CIPLA', 'DIVISLAB', 'BRITANNIA',
    'ADANIPORTS', 'GRASIM', 'INDUSINDBK',
    'JSWSTEEL', 'LT', 'M&M', 'HEROMOTOCO',
    'EICHERMOT', 'BPCL', 'IOC'
]


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_streaming_results():
    path = os.path.join(LOGS_DIR, 'streaming_ALL.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(
        path, parse_dates=['predict_date']
    )
    return df


@st.cache_data
def load_nse_dataset():
    path = os.path.join(
        DATA_PROCESSED_DIR, 'nse_dataset.csv'
    )
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=['Date'])
    return df


@st.cache_data
def load_streaming_metrics():
    path = os.path.join(
        METRICS_DIR, 'streaming_results.csv'
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_financial_metrics():
    path = os.path.join(
        METRICS_DIR, 'financial_metrics.csv'
    )
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ============================================================
# SMART EXPLANATION GENERATOR
# ============================================================

def generate_explanation(selected_features,
                         prediction, confidence,
                         stock_name, indicators):
    """
    Generates a specific explanation for each prediction
    based on actual selected features and indicator values.
    No hardcoded generic text.
    """

    # Identify dominant feature category
    cats = Counter(
        CAT_MAP.get(f, 'Other')
        for f in selected_features
    )
    dominant_cat = cats.most_common(1)[0][0]
    cat_count    = cats.most_common(1)[0][1]

    # Top 3 features with descriptions
    top3 = selected_features[:3]
    top3_lines = ""
    for i, f in enumerate(top3, 1):
        desc = FEAT_DESC.get(f, f)
        top3_lines += f"• <b>{f}</b>: {desc}<br>"

    # RSI specific insight
    rsi_insight = ""
    if ('RSI_14' in selected_features
            and 'RSI_14' in indicators):
        rsi_val = indicators['RSI_14']
        if rsi_val > 70:
            rsi_insight = (
                f"RSI is at <b>{rsi_val:.1f}</b> "
                f"— indicating <b>overbought</b> conditions. "
            )
        elif rsi_val < 30:
            rsi_insight = (
                f"RSI is at <b>{rsi_val:.1f}</b> "
                f"— indicating <b>oversold</b> conditions. "
            )
        else:
            rsi_insight = (
                f"RSI is at <b>{rsi_val:.1f}</b> "
                f"— in neutral territory. "
            )

    # MACD specific insight
    macd_insight = ""
    if ('MACD' in selected_features
            and 'MACD' in indicators):
        macd_val    = indicators.get('MACD', 0)
        macd_signal = indicators.get('MACD_Signal', 0)
        if macd_val > macd_signal:
            macd_insight = (
                f"MACD (<b>{macd_val:.3f}</b>) is "
                f"above Signal ({macd_signal:.3f}) "
                f"— <b>bullish crossover</b>. "
            )
        else:
            macd_insight = (
                f"MACD (<b>{macd_val:.3f}</b>) is "
                f"below Signal ({macd_signal:.3f}) "
                f"— <b>bearish crossover</b>. "
            )

    # ATR specific insight
    atr_insight = ""
    if ('ATR_14' in selected_features
            and 'ATR_14' in indicators):
        atr_val = indicators['ATR_14']
        atr_insight = (
            f"ATR at <b>{atr_val:.2f}</b> "
            f"reflects current market volatility. "
        )

    # Bollinger insight
    bb_insight = ""
    if ('BB_Width' in selected_features
            and 'BB_Width' in indicators):
        bb_val = indicators['BB_Width']
        if bb_val > 0.1:
            bb_insight = (
                f"Bollinger Width at <b>{bb_val:.3f}</b> "
                f"indicates <b>high volatility</b> "
                f"and potential breakout. "
            )
        else:
            bb_insight = (
                f"Bollinger Width at <b>{bb_val:.3f}</b> "
                f"indicates <b>low volatility</b> "
                f"and consolidation phase. "
            )

    # Volume insight
    vol_insight = ""
    if ('Volume_Ratio' in selected_features
            and 'Volume_Ratio' in indicators):
        vr = indicators['Volume_Ratio']
        if vr > 1.5:
            vol_insight = (
                f"Volume Ratio at <b>{vr:.2f}x</b> "
                f"shows <b>above-average buying activity</b>. "
            )
        elif vr < 0.7:
            vol_insight = (
                f"Volume Ratio at <b>{vr:.2f}x</b> "
                f"shows <b>below-average activity</b>, "
                f"low conviction. "
            )

    # Prediction direction
    signal_word = 'positive' if prediction == 1 else 'negative'
    action      = 'BUY'      if prediction == 1 else 'SELL'

    regime_desc = REGIME_MSGS.get(
        dominant_cat,
        'Multiple indicator types contributed.'
    )

    # Combine insights
    specific_insights = (
        rsi_insight + macd_insight +
        atr_insight + bb_insight + vol_insight
    )
    if not specific_insights.strip():
        specific_insights = (
            f"The top features — "
            f"{', '.join(top3)} — "
            f"collectively indicate a "
            f"{signal_word} directional bias."
        )

    explanation = f"""
    <b>Why did the model predict {action}?</b><br><br>

    The Dynamic Feature Selection system analysed the
    last <b>252 trading days</b> of <b>{stock_name}</b>.
    From 33 available technical indicators, XGBoost
    importance ranking identified <b>{len(selected_features)}
    features</b> as most relevant under current conditions.
    <br><br>

    <b>Dominant signal category:
    {dominant_cat} ({cat_count} of top 10 features)</b><br>
    {regime_desc}
    <br><br>

    <b>Top contributing features:</b><br>
    {top3_lines}
    <br>

    <b>Indicator insights:</b><br>
    {specific_insights}
    <br><br>

    Based on these dynamically selected signals,
    the SRPClassifier predicted a
    <b>{signal_word} 20-day forward return direction</b>
    with <b>{confidence*100:.1f}% confidence</b>,
    generating a <b>{action}</b> signal.
    <br><br>

    <small style="color:#888">
    These features are specific to
    <b>{stock_name}</b> at this point in time.
    For a different stock or date, different
    features would be selected — demonstrating
    the adaptive nature of Dynamic Feature Selection.
    </small>
    """
    return explanation


# ============================================================
# LIVE PREDICTION ENGINE
# ============================================================

def run_live_prediction(ticker):
    """
    Download fresh data for any NSE stock,
    compute indicators, run dynamic feature
    selection, train SRP, return prediction.
    """

    import yfinance as yf
    import pandas_ta as ta
    from river import tree, ensemble, preprocessing
    from xgboost import XGBClassifier

    progress = st.progress(0)
    status   = st.empty()

    # ── Step 1: Download ───────────────────────────────────
    status.text("📥 Downloading market data...")
    progress.progress(10)

    try:
        ticker_sym = ticker.upper().strip()
        if not ticker_sym.endswith('.NS'):
            ticker_sym += '.NS'

        tkr  = yf.Ticker(ticker_sym)
        hist = tkr.history(
            period     = '2y',
            auto_adjust= True
        )

        if len(hist) < 100:
            progress.empty()
            status.error(
                f"❌ Not enough data for **{ticker}**. "
                f"Please check the ticker symbol. "
                f"Example: RELIANCE, TCS, INFY"
            )
            return None

        hist = hist.reset_index()
        hist['Date'] = pd.to_datetime(
            hist['Date']
        ).dt.tz_localize(None)
        hist = hist.sort_values('Date').reset_index(
            drop=True
        )

    except Exception as e:
        progress.empty()
        status.error(f"❌ Download failed: {e}")
        return None

    progress.progress(25)
    status.text(
        f"⚙️ Computing 33 technical indicators "
        f"for {ticker_sym}..."
    )

    # ── Step 2: Compute indicators ─────────────────────────
    try:
        close  = hist['Close']
        high   = hist['High']
        low    = hist['Low']
        volume = hist['Volume']

        hist['SMA_10']  = ta.sma(close, length=10)
        hist['SMA_20']  = ta.sma(close, length=20)
        hist['SMA_50']  = ta.sma(close, length=50)
        hist['EMA_12']  = ta.ema(close, length=12)
        hist['EMA_26']  = ta.ema(close, length=26)
        hist['RSI_14']  = ta.rsi(close, length=14)

        macd_df = ta.macd(close)
        if macd_df is not None:
            hist['MACD']        = macd_df.iloc[:, 0]
            hist['MACD_Signal'] = macd_df.iloc[:, 1]
            hist['MACD_Hist']   = macd_df.iloc[:, 2]
        else:
            hist['MACD'] = hist['MACD_Signal'] = \
            hist['MACD_Hist'] = 0

        hist['ROC']        = ta.roc(close, length=10)
        hist['Williams_R'] = ta.willr(
            high, low, close, length=14
        )

        stoch = ta.stoch(high, low, close)
        hist['Stochastic'] = (
            stoch.iloc[:, 0]
            if stoch is not None else 50
        )

        hist['ATR_14'] = ta.atr(
            high, low, close, length=14
        )

        bb = ta.bbands(close, length=20)
        if bb is not None:
            hist['BB_Upper'] = bb.iloc[:, 0]
            hist['BB_Lower'] = bb.iloc[:, 1]
            hist['BB_Width'] = bb.iloc[:, 3]
        else:
            hist['BB_Upper'] = hist['BB_Lower'] = close
            hist['BB_Width'] = 0

        hist['OBV']          = ta.obv(close, volume)
        hist['Volume_MA']    = volume.rolling(20).mean()
        hist['Volume_Ratio'] = (
            volume / hist['Volume_MA'].replace(0, 1)
        )
        hist['CCI'] = ta.cci(
            high, low, close, length=14
        )

        adx_df = ta.adx(high, low, close, length=14)
        hist['ADX'] = (
            adx_df.iloc[:, 0]
            if adx_df is not None else 25
        )

        hist['Daily_Return']       = close.pct_change()
        hist['Log_Return']         = np.log(
            close / close.shift(1)
        )
        hist['Return_5d']          = close.pct_change(5)
        hist['Return_10d']         = close.pct_change(10)
        hist['Rolling_Std_10']     = (
            hist['Daily_Return'].rolling(10).std()
        )
        hist['Rolling_Volatility'] = (
            hist['Daily_Return'].rolling(20).std()
        )
        hist['High_Low_Pct']    = (high - low) / close
        hist['Open_Close_Pct']  = (
            hist['Open'] - close
        ) / close
        hist['Price_Change']    = close.diff()

        tp  = (high + low + close) / 3
        mf  = tp * volume
        pos = mf.where(tp > tp.shift(1), 0)
        neg = mf.where(tp <= tp.shift(1), 0)
        hist['MFI'] = 100 - 100 / (
            1 + pos.rolling(14).sum() /
            neg.rolling(14).sum().replace(0, 1)
        )

        hist['Rolling_Mean_Return']   = (
            hist['Daily_Return'].rolling(10).mean()
        )
        hist['Rolling_Median_Return'] = (
            hist['Daily_Return'].rolling(10).median()
        )

        hist['Binary_Target'] = (
            close.shift(-20) > close
        ).astype(int)

        hist = hist.dropna().reset_index(drop=True)

        if len(hist) < 60:
            progress.empty()
            status.error(
                "Not enough clean data after "
                "indicator computation."
            )
            return None

    except Exception as e:
        progress.empty()
        status.error(
            f"Indicator computation failed: {e}"
        )
        return None

    progress.progress(50)
    status.text(
        "🔍 Running Dynamic Feature Selection "
        "(XGBoost importance on 252-day window)..."
    )

    # ── Step 3: Dynamic feature selection ─────────────────
    FEAT_COLS = [
        'SMA_10','SMA_20','SMA_50','EMA_12','EMA_26',
        'RSI_14','MACD','MACD_Signal','MACD_Hist',
        'ROC','Williams_R','Stochastic',
        'ATR_14','BB_Upper','BB_Lower','BB_Width',
        'OBV','Volume_MA','Volume_Ratio',
        'CCI','ADX','Daily_Return','Log_Return',
        'Return_5d','Return_10d','Rolling_Std_10',
        'Rolling_Volatility','High_Low_Pct',
        'Open_Close_Pct','Price_Change','MFI',
        'Rolling_Mean_Return','Rolling_Median_Return'
    ]
    FEAT_COLS = [
        f for f in FEAT_COLS if f in hist.columns
    ]

    window_data = hist.tail(252)
    X_mi = window_data[FEAT_COLS].values
    y_mi = window_data[
        'Binary_Target'
    ].values.astype(int)

    selected_features   = FEAT_COLS[:10]
    feature_importances = {}

    try:
        counts = Counter(y_mi)
        total  = len(y_mi)
        n      = len(counts)
        wmap   = {
            c: total/(n*cnt)
            for c, cnt in counts.items()
        }
        sw = np.array([wmap[i] for i in y_mi])

        xgb_fs = XGBClassifier(
            n_estimators = 100,
            random_state = 42,
            verbosity    = 0,
            eval_metric  = 'logloss',
            max_depth    = 4,
            learning_rate= 0.1,
            subsample    = 0.8
        )
        xgb_fs.fit(X_mi, y_mi, sample_weight=sw)
        importances = xgb_fs.feature_importances_

        scored = sorted(
            zip(FEAT_COLS, importances),
            key     = lambda x: x[1],
            reverse = True
        )
        selected_features   = [f for f, _ in scored[:10]]
        feature_importances = dict(scored)

    except Exception:
        selected_features   = FEAT_COLS[:10]
        feature_importances = {f: 0 for f in FEAT_COLS}

    progress.progress(70)
    status.text(
        "🤖 Training SRPClassifier on selected features..."
    )

    # ── Step 4: Train SRP ──────────────────────────────────
    try:
        srp_model = (
            preprocessing.StandardScaler() |
            ensemble.SRPClassifier(
                model    = tree.HoeffdingTreeClassifier(),
                n_models = 10,
                seed     = 42
            )
        )

        train_data = hist.iloc[:-1]
        for _, row in train_data.iterrows():
            x = {
                f: float(row[f])
                for f in selected_features
            }
            y = int(row['Binary_Target'])
            srp_model.learn_one(x, y)

        latest_row = hist.iloc[-1]
        x_pred = {
            f: float(latest_row[f])
            for f in selected_features
        }

        pred  = srp_model.predict_one(x_pred)
        proba = srp_model.predict_proba_one(x_pred)

        if proba and len(proba) > 0:
            confidence = max(proba.values())
            pred_class = max(proba, key=proba.get)
        else:
            confidence = 0.5
            pred_class = pred if pred is not None else 1

    except Exception as e:
        progress.empty()
        status.error(f"Model training failed: {e}")
        return None

    progress.progress(100)
    status.text("✅ Analysis complete!")
    time.sleep(0.5)
    status.empty()
    progress.empty()

    return {
        'ticker'            : ticker_sym,
        'stock_name'        : ticker.upper().strip(),
        'latest_date'       : hist['Date'].iloc[-1],
        'open'              : hist['Open'].iloc[-1],
        'high'              : hist['High'].iloc[-1],
        'low'               : hist['Low'].iloc[-1],
        'close'             : hist['Close'].iloc[-1],
        'volume'            : hist['Volume'].iloc[-1],
        'prev_close'        : hist['Close'].iloc[-2],
        'prediction'        : int(pred_class),
        'confidence'        : confidence,
        'selected_features' : selected_features,
        'feature_importances': feature_importances,
        'price_history'     : hist[[
            'Date','Open','High','Low',
            'Close','Volume'
        ]].tail(120),
        'indicators'        : {
            f: float(hist[f].iloc[-1])
            for f in FEAT_COLS
            if f in hist.columns
        },
        'all_features'      : FEAT_COLS,
        'n_rows_used'       : len(hist)
    }


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar():

    st.sidebar.markdown("## 📈 PULSE")
    st.sidebar.markdown(
        "*Adaptive Streaming Stock Prediction*"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Navigation")

    page = st.sidebar.radio(
        "",
        [
            "🏠 Home",
            "🤖 Stock Prediction",
            "🔍 Feature Analysis",
            "📊 Research Results",
            "ℹ️ About"
        ]
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Project:** Dynamic Feature Selection
    for Streaming Stock Market Prediction

    **Algorithm:** SRPClassifier +
    Dynamic Feature Selection

    **Guide:** Prof. I. Priyadarshini

    **Team PID 17** | KKWIEER Nashik
    """)

    return page


# ============================================================
# PAGE 1 — HOME
# ============================================================

def page_home():

    st.markdown(
        '<div class="pulse-title">📈 PULSE</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="pulse-sub">'
        'Adaptive Streaming Stock Market Prediction System'
        '<br>Dynamic Feature Selection + SRPClassifier'
        ' | NSE India</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 🔄 Streaming Learning
        Processes stock market data as a continuous
        stream — learning and adapting with each
        new trading day. No batch retraining required.
        True online learning using SRPClassifier.
        """)

    with col2:
        st.markdown("""
        ### 🔍 Dynamic Features
        At each prediction step identifies the
        TOP 10 most relevant technical indicators
        from 33 available features based on current
        market conditions using XGBoost importance.
        """)

    with col3:
        st.markdown("""
        ### 🌐 Any NSE Stock
        Works for any NSE-listed stock — not limited
        to pre-trained stocks. Downloads fresh market
        data, computes indicators, selects features
        and predicts dynamically for any ticker.
        """)

    st.markdown("---")

    st.markdown("### 🔄 How the System Works")

    flow_cols = st.columns(7)
    steps = [
        ("📥", "Market Data",
         "OHLCV + 33 indicators"),
        ("→", "", ""),
        ("🔍", "Dynamic FS",
         "Top 10 from 33 features"),
        ("→", "", ""),
        ("🤖", "SRPClassifier",
         "10 Hoeffding Trees"),
        ("→", "", ""),
        ("📊", "Prediction",
         "BUY / SELL signal"),
    ]

    for i, (icon, title, desc) in enumerate(steps):
        with flow_cols[i]:
            if icon == "→":
                st.markdown(
                    "<div style='text-align:center;"
                    "font-size:2rem;padding-top:1.5rem'>"
                    "→</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='font-size:2rem'>"
                    f"{icon}</div>"
                    f"<b>{title}</b><br>"
                    f"<small style='color:#888'>"
                    f"{desc}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    st.markdown("---")

    st.markdown("### 🏆 Validated Research Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dynamic SRP Accuracy",
              "80.01%", "+3.60% vs Static")
    m2.metric("Avg Portfolio ROI",
              "774.41%", "+190% vs Static")
    m3.metric("Sharpe Ratio",
              "12.043", "+2.019 vs Static")
    m4.metric("Research Stocks",
              "8 NSE Stocks", "2022-2024")

    st.markdown("---")

    st.markdown("""
    <div class="success-box">
    <b>🚀 Get Started:</b> Go to
    <b>🤖 Stock Prediction</b> to analyse
    <b>any NSE stock</b> using live market data,
    or explore historical research predictions
    for our 8 evaluated stocks.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Disclaimer:</b> This is a research prototype.
    Predictions are based on technical indicators only
    and should NOT be used for actual investment decisions.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# PAGE 2 — STOCK PREDICTION
# ============================================================

def page_stock_prediction(streaming_df, nse_df):

    st.markdown("## 🤖 Stock Prediction")

    tab1, tab2 = st.tabs([
        "🌐 Live Prediction — Any NSE Stock",
        "📂 Historical Analysis — Research Stocks"
    ])

    # ══════════════════════════════════════════════════════
    # TAB 1 — LIVE PREDICTION
    # ══════════════════════════════════════════════════════

    with tab1:

        st.markdown("""
        <div class="info-box">
        <b>🌐 Live Prediction — Any NSE Stock</b><br>
        Enter any NSE-listed stock ticker.
        The system will download the latest 2 years
        of market data, compute 33 technical indicators,
        dynamically select the top 10 most relevant
        features, and generate a 20-day prediction
        using SRPClassifier.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("**Quick Select (Popular Stocks)**")
            st.caption(
                "40 popular NSE stocks — "
                "or type any ticker below"
            )
            selected_popular = st.selectbox(
                "Popular NSE Stocks",
                ["— Select from list —"] + POPULAR_STOCKS,
                label_visibility='collapsed'
            )

        with col2:
            st.markdown("**Type Any NSE Ticker**")
            st.caption(
                "Enter ticker without .NS — "
                "works for any NSE listed stock"
            )
            custom_ticker = st.text_input(
                "Custom Ticker",
                placeholder=(
                    "e.g. BAJAJFINSV, ZOMATO, IRCTC..."
                ),
                label_visibility='collapsed'
            )

        # Determine final ticker
        if custom_ticker.strip():
            final_ticker = custom_ticker.strip().upper()
            st.info(
                f"📌 Analysing: **{final_ticker}.NS** "
                f"(custom input)"
            )
        elif selected_popular != "— Select from list —":
            final_ticker = selected_popular
            st.info(
                f"📌 Analysing: **{final_ticker}.NS** "
                f"(from popular list)"
            )
        else:
            final_ticker = None

        # Ticker help
        st.markdown("""
        <div class="ticker-help">
        <b>How to find the right ticker:</b>
        RELIANCE → Reliance Industries |
        TCS → Tata Consultancy Services |
        INFY → Infosys |
        HDFCBANK → HDFC Bank |
        BAJFINANCE → Bajaj Finance |
        ZOMATO → Zomato |
        IRCTC → Indian Railway Catering |
        ADANIPORTS → Adani Ports |
        Any Nifty 50 / Nifty 500 stock ticker works.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        analyse_btn = st.button(
            "🔍 ANALYSE STOCK",
            type     = "primary",
            disabled = (final_ticker is None),
            use_container_width = True
        )

        if analyse_btn and final_ticker:

            result = run_live_prediction(final_ticker)

            if result is None:
                return

            st.markdown("---")

            # ── Prediction Card ────────────────────────────
            pred       = result['prediction']
            conf       = result['confidence']
            signal     = 'BUY' if pred == 1 else 'SELL'
            card_class = (
                'pred-card-buy' if pred == 1
                else 'pred-card-sell'
            )
            s_emoji = '🟢' if pred == 1 else '🔴'
            conf_pct = conf * 100

            # Price change
            price_chg = (
                result['close'] - result['prev_close']
            )
            price_chg_pct = (
                price_chg / result['prev_close'] * 100
            )
            chg_arrow = (
                '▲' if price_chg >= 0 else '▼'
            )

            col_pred, col_info = st.columns([1, 1])

            with col_pred:
                st.markdown(
                    f'<div class="{card_class}">'
                    f'<div style="font-size:1.4rem;'
                    f'font-weight:700">'
                    f'{result["stock_name"]}</div>'
                    f'<div style="font-size:0.9rem;'
                    f'opacity:0.7">'
                    f'{result["latest_date"].strftime("%d %b %Y")}'
                    f' | NSE</div>'
                    f'<div class="signal-text">'
                    f'{s_emoji} {signal}</div>'
                    f'<div style="font-size:1.1rem;'
                    f'margin:0.5rem 0">'
                    f'Confidence: <b>{conf_pct:.1f}%</b>'
                    f'</div>'
                    f'<div style="font-size:0.9rem;'
                    f'opacity:0.8">'
                    f'Predicted 20-Day Forward Direction'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

            with col_info:
                st.markdown("#### 📋 Stock Information")
                st.metric(
                    "Close Price",
                    f"₹{result['close']:.2f}",
                    delta=(
                        f"{chg_arrow} ₹{abs(price_chg):.2f}"
                        f" ({price_chg_pct:+.2f}%)"
                    )
                )
                i1, i2 = st.columns(2)
                i1.metric("Open",  f"₹{result['open']:.2f}")
                i2.metric("High",  f"₹{result['high']:.2f}")
                i3, i4 = st.columns(2)
                i3.metric("Low",   f"₹{result['low']:.2f}")
                i4.metric(
                    "Volume",
                    f"{result['volume']:,.0f}"
                )
                st.caption(
                    f"Data: {result['n_rows_used']:,} days "
                    f"| Features used for prediction: "
                    f"{len(result['selected_features'])}"
                )

            st.markdown("---")

            # ── Price Chart ────────────────────────────────
            st.markdown("### 📈 Price History (Last 120 Days)")

            ph  = result['price_history']
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x     = ph['Date'],
                open  = ph['Open'],
                high  = ph['High'],
                low   = ph['Low'],
                close = ph['Close'],
                name  = result['stock_name'],
                increasing_line_color = '#4CAF50',
                decreasing_line_color = '#F44336'
            ))
            fig.update_layout(
                title  = (
                    f"{result['stock_name']} "
                    f"Price Chart (NSE)"
                ),
                yaxis_title = "Price (₹)",
                xaxis_title = "Date",
                height      = 400,
                xaxis_rangeslider_visible = False,
                template    = "plotly_dark"
            )
            st.plotly_chart(
                fig, use_container_width=True
            )

            st.markdown("---")

            # ── Dynamic Feature Selection ──────────────────
            st.markdown(
                "### 🔍 Dynamic Feature Selection"
            )

            fc1,fc2,fc3,fc4,fc5 = st.columns(5)
            with fc1:
                st.markdown(
                    "<div class='step-box' "
                    "style='text-align:center'>"
                    f"<b>33</b><br>Features<br>Available"
                    "</div>",
                    unsafe_allow_html=True
                )
            with fc2:
                st.markdown(
                    "<div style='text-align:center;"
                    "font-size:2rem;padding-top:1rem'>"
                    "→</div>",
                    unsafe_allow_html=True
                )
            with fc3:
                st.markdown(
                    "<div class='step-box' "
                    "style='text-align:center'>"
                    "<b>252-day Window</b><br>"
                    "XGBoost Importance"
                    "</div>",
                    unsafe_allow_html=True
                )
            with fc4:
                st.markdown(
                    "<div style='text-align:center;"
                    "font-size:2rem;padding-top:1rem'>"
                    "→</div>",
                    unsafe_allow_html=True
                )
            with fc5:
                st.markdown(
                    "<div class='step-box' "
                    "style='text-align:center'>"
                    f"<b>Top 10</b><br>"
                    f"Features Selected<br>"
                    f"<small style='color:#4CAF50'>"
                    f"for {result['stock_name']}"
                    f" TODAY</small>"
                    "</div>",
                    unsafe_allow_html=True
                )

            st.markdown("#### ✅ Selected Features")
            chips_html = ""
            for i, feat in enumerate(
                result['selected_features'], 1
            ):
                chips_html += (
                    f'<span class="feature-chip">'
                    f'{i}. {feat}</span>'
                )
            st.markdown(chips_html, unsafe_allow_html=True)

            # Feature importance chart
            if result['feature_importances']:
                sel_imp = {
                    f: result['feature_importances'].get(
                        f, 0
                    )
                    for f in result['selected_features']
                }
                sel_sorted = dict(sorted(
                    sel_imp.items(),
                    key=lambda x: x[1]
                ))

                fig2 = go.Figure(go.Bar(
                    x            = list(
                        sel_sorted.values()
                    ),
                    y            = list(
                        sel_sorted.keys()
                    ),
                    orientation  = 'h',
                    marker_color = '#4CAF50',
                    text         = [
                        f"{v:.3f}"
                        for v in sel_sorted.values()
                    ],
                    textposition = 'outside'
                ))
                max_val = max(sel_sorted.values()) \
                          if sel_sorted.values() else 1
                fig2.update_layout(
                    title   = (
                        "Feature Importance Scores "
                        "(XGBoost Gain) — "
                        f"{result['stock_name']}"
                    ),
                    xaxis_title = "Importance Score",
                    height  = 380,
                    xaxis_range = [0, max_val * 1.3],
                    template= "plotly_dark"
                )
                st.plotly_chart(
                    fig2, use_container_width=True
                )

            st.markdown("---")

            # ── Technical Indicators ───────────────────────
            st.markdown(
                "### 📉 Technical Indicators"
            )
            st.caption(
                "All 33 indicators computed from "
                "live market data. "
                "✅ = selected for this prediction."
            )

            indicators = result['indicators']

            ind_cat_map = {
                'Trend'       : [
                    'SMA_10','SMA_20','SMA_50',
                    'EMA_12','EMA_26'
                ],
                'Momentum'    : [
                    'RSI_14','MACD','MACD_Signal',
                    'MACD_Hist','ROC',
                    'Williams_R','Stochastic'
                ],
                'Volatility'  : [
                    'ATR_14','BB_Upper',
                    'BB_Lower','BB_Width'
                ],
                'Volume'      : [
                    'OBV','Volume_MA','Volume_Ratio'
                ],
                'Returns'     : [
                    'Daily_Return','Log_Return',
                    'Return_5d','Return_10d',
                    'Rolling_Std_10','Rolling_Volatility',
                    'Rolling_Mean_Return',
                    'Rolling_Median_Return'
                ],
                'Statistical' : [
                    'CCI','ADX','MFI',
                    'High_Low_Pct','Open_Close_Pct',
                    'Price_Change'
                ]
            }

            tab_cats = st.tabs(
                list(ind_cat_map.keys())
            )
            for tab_c, (cat, feats) in zip(
                tab_cats, ind_cat_map.items()
            ):
                with tab_c:
                    rows = []
                    for f in feats:
                        if f in indicators:
                            val = indicators[f]
                            sel = (
                                "✅ Selected"
                                if f in result[
                                    'selected_features'
                                ] else ""
                            )
                            desc = FEAT_DESC.get(f, f)
                            rows.append({
                                'Feature'    : f,
                                'Value'      : round(
                                    val, 4
                                ),
                                'Description': desc,
                                'Selected'   : sel
                            })
                    if rows:
                        st.dataframe(
                            pd.DataFrame(rows),
                            use_container_width = True,
                            hide_index          = True
                        )

            st.markdown("---")

            # ── Smart Explanation ──────────────────────────
            st.markdown("### 💡 Prediction Explanation")

            expl = generate_explanation(
                result['selected_features'],
                result['prediction'],
                result['confidence'],
                result['stock_name'],
                result['indicators']
            )
            st.markdown(
                f'<div class="info-box">{expl}</div>',
                unsafe_allow_html=True
            )

    # ══════════════════════════════════════════════════════
    # TAB 2 — HISTORICAL ANALYSIS
    # ══════════════════════════════════════════════════════

    with tab2:

        st.markdown("""
        <div class="info-box">
        <b>📂 Historical Research Analysis</b><br>
        Explore predictions from our research evaluation
        on 8 NSE stocks (2022-2024).
        Select a stock and date to see the exact
        prediction, selected features, and actual outcome.
        </div>
        """, unsafe_allow_html=True)

        if streaming_df is None:
            st.error(
                "Historical results not found. "
                "Run: python src/models/streaming_model.py"
                " --stock ALL"
            )
            return

        col1, col2 = st.columns(2)

        with col1:
            stocks    = sorted(
                streaming_df['stock'].unique()
            )
            sel_stock = st.selectbox(
                "Select Research Stock", stocks
            )

        stock_data = streaming_df[
            streaming_df['stock'] == sel_stock
        ].sort_values('predict_date')

        with col2:
            dates    = pd.to_datetime(
                stock_data['predict_date']
            ).dt.date.tolist()
            sel_date = st.selectbox(
                "Select Date (2022-2024)",
                dates,
                index = len(dates) - 1
            )

        sel_row = stock_data[
            pd.to_datetime(
                stock_data['predict_date']
            ).dt.date == sel_date
        ]

        if len(sel_row) == 0:
            st.warning("No data for selected date.")
            return

        row = sel_row.iloc[0]

        pred_val   = int(row['dynamic_pred'])
        conf_val   = float(row['dynamic_confidence'])
        actual_val = int(row['actual_target'])
        correct    = int(row['dynamic_correct'])

        signal_val = (
            'BUY'      if pred_val == 1
            else 'SELL' if pred_val == 0
            else 'NO TRADE'
        )
        card_cls = (
            'pred-card-buy'     if pred_val == 1
            else 'pred-card-sell' if pred_val == 0
            else 'pred-card-notrade'
        )
        emoji = (
            '🟢' if pred_val == 1
            else '🔴' if pred_val == 0
            else '⚪'
        )
        actual_str = (
            '🟢 Positive (Up)' if actual_val == 1
            else '🔴 Negative (Down)'
        )
        result_str = (
            '✅ Correct' if correct else '❌ Incorrect'
        )

        col_pred, col_hist = st.columns([1, 1])

        with col_pred:
            st.markdown("---")
            st.markdown(
                f'<div class="{card_cls}">'
                f'<div style="font-size:1.4rem;'
                f'font-weight:700">{sel_stock}</div>'
                f'<div style="font-size:0.9rem;'
                f'opacity:0.7">'
                f'{sel_date} | Regime: '
                f'{row["market_regime"]}</div>'
                f'<div class="signal-text">'
                f'{emoji} {signal_val}</div>'
                f'<div style="font-size:1.1rem;'
                f'margin:0.5rem 0">'
                f'Confidence: '
                f'<b>{conf_val*100:.1f}%</b></div>'
                f'<div style="font-size:0.9rem;'
                f'opacity:0.8">'
                f'Predicted 20-Day Forward Direction'
                f'</div>'
                f'<div style="margin-top:1rem;'
                f'font-size:1rem">'
                f'Actual: <b>{actual_str}</b><br>'
                f'Result: <b>{result_str}</b>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        with col_hist:
            st.markdown("---")
            st.markdown("### 🗓️ Recent Predictions")

            hist_view = stock_data[
                pd.to_datetime(
                    stock_data['predict_date']
                ).dt.date <= sel_date
            ].tail(10)[[
                'predict_date',
                'dynamic_signal',
                'dynamic_confidence',
                'actual_target',
                'dynamic_correct',
                'market_regime'
            ]].copy()

            hist_view['actual_target'] = hist_view[
                'actual_target'
            ].map({1:'🟢 Up', 0:'🔴 Down'})
            hist_view['dynamic_correct'] = hist_view[
                'dynamic_correct'
            ].map({1:'✅', 0:'❌'})
            hist_view['dynamic_confidence'] = hist_view[
                'dynamic_confidence'
            ].apply(lambda x: f"{x*100:.1f}%")
            hist_view['predict_date'] = pd.to_datetime(
                hist_view['predict_date']
            ).dt.date

            hist_view.columns = [
                'Date', 'Signal', 'Confidence',
                'Actual', 'Correct', 'Regime'
            ]
            st.dataframe(
                hist_view.iloc[::-1],
                use_container_width = True,
                hide_index          = True
            )

        # ── Price chart from nse_dataset ───────────────────
        if nse_df is not None:
            nse_stock = nse_df[
                nse_df['Stock_Symbol'].str.contains(
                    sel_stock, case=False, na=False
                )
            ].sort_values('Date')

            if len(nse_stock) > 0:
                sel_dt = pd.Timestamp(sel_date)
                idx_before = nse_stock[
                    nse_stock['Date'] <= sel_dt
                ].index

                if len(idx_before) > 0:
                    last_idx = idx_before[-1]
                    loc = nse_stock.index.get_loc(
                        last_idx
                    )
                    start = max(0, loc - 60)
                    chart_data = nse_stock.iloc[
                        start:loc+1
                    ]

                    st.markdown("---")
                    st.markdown(
                        "### 📈 Price Chart "
                        "(60 Days Around Prediction)"
                    )

                    cur = nse_stock.loc[last_idx]
                    ci1,ci2,ci3,ci4,ci5 = st.columns(5)
                    ci1.metric(
                        "Open", f"₹{cur['Open']:.2f}"
                    )
                    ci2.metric(
                        "High", f"₹{cur['High']:.2f}"
                    )
                    ci3.metric(
                        "Low",  f"₹{cur['Low']:.2f}"
                    )
                    ci4.metric(
                        "Close",f"₹{cur['Close']:.2f}"
                    )
                    ci5.metric(
                        "Volume",
                        f"{int(cur['Volume']):,}"
                    )

                    fig3 = go.Figure()
                    fig3.add_trace(go.Candlestick(
                        x     = chart_data['Date'],
                        open  = chart_data['Open'],
                        high  = chart_data['High'],
                        low   = chart_data['Low'],
                        close = chart_data['Close'],
                        name  = sel_stock,
                        increasing_line_color = '#4CAF50',
                        decreasing_line_color = '#F44336'
                    ))
                    fig3.add_vline(
                        x     = sel_dt,
                        line_dash  = "dash",
                        line_color = "yellow",
                        annotation_text = "Prediction Date",
                        annotation_position = "top right"
                    )
                    fig3.update_layout(
                        title  = (
                            f"{sel_stock} Price Chart"
                        ),
                        height = 400,
                        xaxis_rangeslider_visible = False,
                        template = "plotly_dark"
                    )
                    st.plotly_chart(
                        fig3, use_container_width=True
                    )

        # ── Dynamic features for this date ────────────────
        st.markdown("---")
        st.markdown(
            "### 🔍 Dynamically Selected Features"
        )
        st.caption(
            "Features the model used for this "
            "specific prediction"
        )

        if pd.notna(row.get('dynamic_features')):
            feats = [
                f.strip()
                for f in str(
                    row['dynamic_features']
                ).split(',')
            ]
            chips = ""
            for i, f in enumerate(feats, 1):
                chips += (
                    f'<span class="feature-chip">'
                    f'{i}. {f}</span>'
                )
            st.markdown(chips, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("#### Feature Descriptions")

            feat_rows = []
            for f in feats:
                feat_rows.append({
                    'Feature'    : f,
                    'Category'   : CAT_MAP.get(
                        f, 'Other'
                    ),
                    'Description': FEAT_DESC.get(f, f)
                })
            st.dataframe(
                pd.DataFrame(feat_rows),
                use_container_width = True,
                hide_index          = True
            )

            top_feat = row.get('top_mi_feature', 'N/A')
            st.markdown(f"""
            <div class="success-box">
            <b>🏆 Most Important Feature:</b>
            <code>{top_feat}</code><br>
            {FEAT_DESC.get(top_feat, top_feat)}
            <br><br>
            <b>Market Regime:</b> {row['market_regime']}
            </div>
            """, unsafe_allow_html=True)

        # ── Technical indicators from nse_dataset ─────────
        if nse_df is not None and len(nse_stock) > 0:
            if len(idx_before) > 0:
                st.markdown("---")
                st.markdown(
                    "### 📉 Technical Indicators "
                    "at Selected Date"
                )
                feat_cols_nse = [
                    c for c in nse_df.columns
                    if c not in [
                        'Date','Stock_Symbol',
                        'Stock_Name','Open','High',
                        'Low','Close','Volume',
                        'Forward_Return','Binary_Target'
                    ]
                ]
                ind_rows = []
                for f in feat_cols_nse:
                    if f in cur.index:
                        sel_marker = (
                            "✅"
                            if f in str(
                                row.get(
                                    'dynamic_features',''
                                )
                            ) else ""
                        )
                        ind_rows.append({
                            'Feature'    : f,
                            'Value'      : round(
                                float(cur[f]), 4
                            ),
                            'Category'   : CAT_MAP.get(
                                f, 'Other'
                            ),
                            'Selected'   : sel_marker,
                            'Description': FEAT_DESC.get(
                                f, ''
                            )
                        })

                st.dataframe(
                    pd.DataFrame(ind_rows),
                    use_container_width = True,
                    hide_index          = True,
                    height              = 350
                )


# ============================================================
# PAGE 3 — FEATURE ANALYSIS
# ============================================================

def page_feature_analysis(streaming_df):

    st.markdown("## 🔍 Feature Analysis")
    st.markdown(
        "Explore how Dynamic Feature Selection "
        "adapts over time. This page demonstrates "
        "the core contribution of the project."
    )

    if streaming_df is None:
        st.error("Historical results not found.")
        return

    stocks    = sorted(streaming_df['stock'].unique())
    sel_stock = st.selectbox(
        "Select Stock", stocks
    )

    stock_df = streaming_df[
        streaming_df['stock'] == sel_stock
    ].copy().sort_values('predict_date')

    # ── Feature frequency ──────────────────────────────────
    st.markdown("### 📊 Feature Selection Frequency")
    st.caption(
        "How often each feature was selected "
        "across all prediction windows for this stock"
    )

    all_feats = []
    for fs in stock_df['dynamic_features'].dropna():
        all_feats.extend(
            [f.strip() for f in str(fs).split(',')]
        )

    freq  = Counter(all_feats)
    top15 = freq.most_common(15)
    names = [f[0] for f in top15][::-1]
    vals  = [
        f[1]/len(stock_df)*100 for f in top15
    ][::-1]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x            = vals,
            y            = names,
            orientation  = 'h',
            marker_color = [
                '#4CAF50' if v > 60
                else '#FF9800' if v > 40
                else '#2196F3'
                for v in vals
            ],
            text         = [
                f"{v:.1f}%" for v in vals
            ],
            textposition = 'outside'
        ))
        fig.update_layout(
            title       = (
                f"{sel_stock} — "
                "Feature Selection Frequency"
            ),
            xaxis_title = "Selected in % of Windows",
            height      = 480,
            template    = "plotly_dark"
        )
        st.plotly_chart(
            fig, use_container_width=True
        )

    with col2:
        cat_counts = Counter()
        for feat, count in freq.items():
            cat = CAT_MAP.get(feat, 'Other')
            cat_counts[cat] += count

        fig2 = px.pie(
            values = list(cat_counts.values()),
            names  = list(cat_counts.keys()),
            title  = "Feature Category Distribution",
            color_discrete_sequence =
                px.colors.qualitative.Set2
        )
        fig2.update_layout(
            height   = 480,
            template = "plotly_dark"
        )
        st.plotly_chart(
            fig2, use_container_width=True
        )

    # ── Top feature over time ──────────────────────────────
    st.markdown("### 📅 Feature Relevance Over Time")
    st.markdown("""
    <div class="success-box">
    <b>💡 Core Insight:</b> The chart below shows
    which feature was most important each month.
    If features were static, every bar would be
    the same colour. The <b>colour changes prove
    that feature relevance shifts</b> —
    this is why Dynamic Feature Selection
    outperforms Static Feature Selection.
    </div>
    """, unsafe_allow_html=True)

    if 'top_mi_feature' in stock_df.columns:
        stock_df['month'] = pd.to_datetime(
            stock_df['predict_date']
        ).dt.to_period('M').astype(str)

        top_monthly = stock_df.groupby('month')[
            'top_mi_feature'
        ].agg(
            lambda x: x.value_counts().index[0]
            if len(x) > 0 else 'N/A'
        ).reset_index()
        top_monthly.columns = ['Month', 'Top Feature']

        fig3 = px.bar(
            top_monthly,
            x     = 'Month',
            y     = [1] * len(top_monthly),
            color = 'Top Feature',
            title = (
                f"{sel_stock} — "
                "Most Important Feature by Month"
            ),
            labels= {'y': '', 'Month': 'Month'},
            height= 350
        )
        fig3.update_yaxes(showticklabels=False)
        fig3.update_layout(
            xaxis_tickangle = -45,
            template        = "plotly_dark"
        )
        st.plotly_chart(
            fig3, use_container_width=True
        )

    # ── Feature stats ──────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Total Predictions", f"{len(stock_df):,}"
    )
    c2.metric(
        "Unique Features Used", f"{len(freq)}"
    )
    c3.metric(
        "Most Selected Feature",
        names[-1] if names else "N/A"
    )
    c4.metric(
        "Avg Features per Window",
        f"{stock_df['n_dynamic_features'].mean():.1f}"
    )

    # ── All stocks comparison ──────────────────────────────
    st.markdown("---")
    st.markdown(
        "### 🌐 Top Features Across All Research Stocks"
    )

    all_stock_feats = []
    for fs in streaming_df[
        'dynamic_features'
    ].dropna():
        all_stock_feats.extend(
            [f.strip() for f in str(fs).split(',')]
        )

    all_freq  = Counter(all_stock_feats)
    all_top15 = all_freq.most_common(15)
    a_names   = [f[0] for f in all_top15][::-1]
    a_vals    = [
        f[1]/len(streaming_df)*100
        for f in all_top15
    ][::-1]

    fig4 = go.Figure(go.Bar(
        x            = a_vals,
        y            = a_names,
        orientation  = 'h',
        marker_color = '#9C27B0',
        text         = [f"{v:.1f}%" for v in a_vals],
        textposition = 'outside'
    ))
    fig4.update_layout(
        title       = "Feature Frequency — All 8 Stocks",
        xaxis_title = "Selected in % of Windows",
        height      = 480,
        template    = "plotly_dark"
    )
    st.plotly_chart(fig4, use_container_width=True)


# ============================================================
# PAGE 4 — RESEARCH RESULTS
# ============================================================

def page_research_results(
    streaming_df, metrics_df, fin_df
):

    st.markdown("## 📊 Research Results")
    st.markdown("""
    Research evaluation comparing Dynamic Feature
    Selection vs Static baseline using walk-forward
    validation across 8 NSE stocks (2022-2024).
    """)

    # ── Algorithm journey ──────────────────────────────────
    st.markdown("### 🔬 Algorithm Development Journey")
    st.caption(
        "We systematically tested 6 approaches "
        "before arriving at the final solution"
    )

    journey = pd.DataFrame({
        'Algorithm'   : [
            'XGBoost (Batch)',
            'Hoeffding Tree (Streaming)',
            'KNN (Streaming)',
            'SRP Static (Streaming)',
            'SRP + Dynamic FS (Ours)'
        ],
        'Accuracy (%)': [
            57.42, 62.99, 70.95, 76.41, 80.01
        ],
        'Reason for Rejection / Selection': [
            'Batch learning — cannot adapt to streams',
            'Single tree — misses complex patterns',
            'No explicit drift handling — not scalable',
            'Best streaming algorithm — no dynamic FS',
            '✅ FINAL: Best algorithm + dynamic features'
        ]
    })

    colors = [
        '#FF6B6B','#FFA07A',
        '#FFD700','#90EE90','#4CAF50'
    ]
    fig = go.Figure(go.Bar(
        x            = journey['Algorithm'],
        y            = journey['Accuracy (%)'],
        marker_color = colors,
        text         = [
            f"{v:.2f}%" for v in journey['Accuracy (%)']
        ],
        textposition = 'outside'
    ))
    fig.add_hline(
        y=50, line_dash="dash",
        line_color="red",
        annotation_text="Random baseline (50%)"
    )
    fig.update_layout(
        title       = "Algorithm Accuracy Progression",
        yaxis_range = [40, 90],
        height      = 420,
        showlegend  = False,
        template    = "plotly_dark"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        journey,
        use_container_width = True,
        hide_index          = True
    )

    st.markdown("---")

    # ── Overall metrics ────────────────────────────────────
    st.markdown("### 📋 Static vs Dynamic — Overall")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ML Metrics")
        ml_data = pd.DataFrame({
            'Metric'      : [
                'Accuracy','F1 Score',
                'Precision','Recall'
            ],
            'Static SRP'  : [
                '76.41%','76.26%','76.41%','76.41%'
            ],
            'Dynamic SRP' : [
                '80.01%','79.83%','80.17%','80.01%'
            ],
            'Improvement' : [
                '+3.60%','+3.57%',
                '+3.76%','+3.60%'
            ]
        })
        st.dataframe(
            ml_data,
            use_container_width=True,
            hide_index=True
        )

    with col2:
        st.markdown("#### Financial Metrics")
        if fin_df is not None:
            st.dataframe(
                fin_df,
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")

    # ── Per stock ──────────────────────────────────────────
    st.markdown("### 📊 Per-Stock Results")

    if metrics_df is not None:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name         = 'Static SRP',
            x            = metrics_df['stock'],
            y            = metrics_df['static_accuracy'],
            marker_color = '#2196F3',
            text = [
                f"{v:.1f}%"
                for v in metrics_df['static_accuracy']
            ],
            textposition = 'outside'
        ))
        fig2.add_trace(go.Bar(
            name         = 'Dynamic SRP',
            x            = metrics_df['stock'],
            y            = metrics_df['dynamic_accuracy'],
            marker_color = '#4CAF50',
            text = [
                f"{v:.1f}%"
                for v in metrics_df['dynamic_accuracy']
            ],
            textposition = 'outside'
        ))
        fig2.update_layout(
            title      = "Per-Stock Accuracy Comparison",
            barmode    = 'group',
            height     = 420,
            yaxis_range= [60, 90],
            template   = "plotly_dark"
        )
        st.plotly_chart(
            fig2, use_container_width=True
        )

        # Improvement chart
        diff_col = (
            metrics_df['dynamic_accuracy'] -
            metrics_df['static_accuracy']
        )
        fig3 = go.Figure(go.Bar(
            x = metrics_df['stock'],
            y = diff_col,
            marker_color = [
                '#4CAF50' if v >= 0 else '#F44336'
                for v in diff_col
            ],
            text = [f"{v:+.2f}%" for v in diff_col],
            textposition = 'outside'
        ))
        fig3.add_hline(
            y=0, line_color='white', line_width=1
        )
        fig3.update_layout(
            title    = "Dynamic vs Static Improvement",
            yaxis_title = "Accuracy Difference (%)",
            height   = 350,
            template = "plotly_dark"
        )
        st.plotly_chart(
            fig3, use_container_width=True
        )

    st.markdown("---")

    # ── Year by year ───────────────────────────────────────
    st.markdown("### 📅 Year-by-Year Analysis")

    if streaming_df is not None:
        yearly = streaming_df.groupby('year').agg(
            static_acc  = ('static_correct',  'mean'),
            dynamic_acc = ('dynamic_correct', 'mean')
        ).reset_index()
        yearly['static_acc']  *= 100
        yearly['dynamic_acc'] *= 100

        fig4 = go.Figure()
        fig4.add_trace(go.Bar(
            name         = 'Static SRP',
            x            = yearly['year'].astype(str),
            y            = yearly['static_acc'],
            marker_color = '#2196F3',
            text = [
                f"{v:.1f}%"
                for v in yearly['static_acc']
            ],
            textposition = 'outside'
        ))
        fig4.add_trace(go.Bar(
            name         = 'Dynamic SRP',
            x            = yearly['year'].astype(str),
            y            = yearly['dynamic_acc'],
            marker_color = '#4CAF50',
            text = [
                f"{v:.1f}%"
                for v in yearly['dynamic_acc']
            ],
            textposition = 'outside'
        ))
        fig4.update_layout(
            title       = "Year-by-Year Accuracy",
            barmode     = 'group',
            height      = 380,
            yaxis_range = [70, 90],
            template    = "plotly_dark"
        )
        st.plotly_chart(
            fig4, use_container_width=True
        )

    # ── Evaluation graphs ──────────────────────────────────
    st.markdown("---")
    st.markdown("### 🖼️ Evaluation Graphs")

    graph_files = {
        'Accuracy Comparison'     :
            'accuracy_comparison.png',
        'Yearly Accuracy'         :
            'yearly_accuracy.png',
        'Regime Accuracy'         :
            'regime_accuracy.png',
        'Feature Frequency'       :
            'feature_frequency.png',
        'Prediction Distribution' :
            'prediction_distribution.png',
        'Portfolio Growth'        :
            'portfolio_growth.png',
    }

    cols = st.columns(2)
    for idx, (title, fname) in enumerate(
        graph_files.items()
    ):
        path = os.path.join(GRAPHS_DIR, fname)
        if os.path.exists(path):
            with cols[idx % 2]:
                st.image(
                    path,
                    caption          = title,
                    use_column_width = True
                )


# ============================================================
# PAGE 5 — ABOUT
# ============================================================

def page_about():

    st.markdown("## ℹ️ About PULSE")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Problem Statement
        Stock market data arrives as a continuous
        non-stationary stream where the predictive
        relevance of technical indicators shifts
        over time due to **concept drift**.

        Traditional batch learning systems with
        fixed feature sets cannot adapt to market
        regime changes — leading to degraded
        prediction accuracy during transitions
        from bull to bear markets.

        ### 💡 Our Solution
        **PULSE** — Adaptive streaming stock market
        prediction using Dynamic Feature Selection
        integrated with SRPClassifier.

        At every 30 streaming samples:
        - XGBoost importance computed on last 252 days
        - Top 10 most relevant features selected
        - SRPClassifier predicts using these features
        - ADWIN detects concept drift automatically
        - System adapts without any manual intervention
        """)

    with col2:
        st.markdown("""
        ### 🔍 Why Dynamic Feature Selection?

        **In a Bull Market:**
        RSI and MACD momentum indicators are
        most predictive — markets follow trends.

        **In a Bear Market:**
        ATR and Bollinger Width volatility indicators
        dominate — uncertainty drives prices.

        **In a Sideways Market:**
        OBV and Volume Ratio matter most —
        volume confirms direction.

        A **static system** uses all 33 features
        equally regardless of market regime.
        Our **dynamic system** detects which 10
        are most relevant RIGHT NOW and uses only those.

        **Result:** +3.60% accuracy improvement
        and +190% ROI improvement over static baseline.

        ### 🤖 SRPClassifier
        Streaming Random Patches (Gomes et al. 2019).
        Ensemble of 10 Hoeffding Trees.
        Double randomization: subspaces + resampling.
        Designed specifically for concept drift.
        State-of-the-art streaming classifier.
        """)

    st.markdown("---")

    st.markdown("### 📊 System Architecture")

    steps = [
        ("1", "Data Stream",
         "NSE stock OHLCV data arrives daily as a stream"),
        ("2", "Feature Engineering",
         "33 technical indicators computed from raw price data"),
        ("3", "Dynamic Feature Selection",
         "XGBoost importance on 252-day window → Top 10 selected every 30 samples"),
        ("4", "ADWIN Drift Detection",
         "Monitors prediction error — triggers immediate re-evaluation on drift"),
        ("5", "SRPClassifier",
         "Ensemble of 10 Hoeffding Trees trained on dynamically selected features"),
        ("6", "Signal Generation",
         "BUY / SELL with confidence score — NO TRADE if confidence < 55%"),
    ]

    for num, title, desc in steps:
        st.markdown(f"""
        <div class="step-box">
        <b>Step {num}: {title}</b><br>
        <span style="color:#aaa">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 👥 Team PID 17")

    t1, t2, t3, t4 = st.columns(4)
    team = [
        ("Chintan S. Mehta",  "T231301076",
         "Data Processing Lead"),
        ("Om S. Kumavat",     "T231301068",
         "Model Development Lead"),
        ("Rutuja H. Nagare",  "T231301082",
         "Evaluation Lead"),
        ("Deodatta A. Pagar", "T231301087",
         "Dynamic Feature Selection Lead"),
    ]

    for col, (name, roll, role) in zip(
        [t1, t2, t3, t4], team
    ):
        with col:
            st.markdown(f"""
            <div class="step-box"
             style="text-align:center">
            <b>{name}</b><br>
            <small style="color:#888">{roll}</small><br>
            <small style="color:#4CAF50">{role}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    **Guide:** Prof. I. Priyadarshini
    **College:** K.K. Wagh Institute of Engineering
    Education and Research, Nashik — 422003
    **Academic Year:** 2025-26
    """)

    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
    <b>⚠️ Research Disclaimer:</b>
    PULSE is an academic research prototype.
    All predictions are based on historical
    technical indicators and ML models only.
    Do NOT use for actual investment decisions.
    Past performance does not guarantee future results.
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# MAIN
# ============================================================

def main():

    streaming_df = load_streaming_results()
    nse_df       = load_nse_dataset()
    metrics_df   = load_streaming_metrics()
    fin_df       = load_financial_metrics()

    page = render_sidebar()

    if page == "🏠 Home":
        page_home()
    elif page == "🤖 Stock Prediction":
        page_stock_prediction(streaming_df, nse_df)
    elif page == "🔍 Feature Analysis":
        page_feature_analysis(streaming_df)
    elif page == "📊 Research Results":
        page_research_results(
            streaming_df, metrics_df, fin_df
        )
    elif page == "ℹ️ About":
        page_about()


if __name__ == '__main__':
    main()