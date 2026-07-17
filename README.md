# 📈 Effective Dynamic Feature Selection in Streaming Data
### A Stock Trading Use Case

> Bachelor of Engineering Final Year Project (Computer Engineering)

This project presents an adaptive machine learning framework for **Dynamic Feature Selection in Streaming Data** using stock market prediction as a real-world use case.

Unlike traditional machine learning models that rely on a fixed set of features, this system continuously re-evaluates feature importance as new market data arrives, allowing the model to adapt to changing market conditions.

The primary contribution of this project is **Dynamic Feature Selection**, while stock trading serves as the experimental application.

---

# 🚀 Project Highlights

- Dynamic Feature Selection using Mutual Information
- Static vs Dynamic Random Forest comparison
- Walk-Forward Validation (Time-Series Evaluation)
- Streaming data simulation using Sliding Windows
- Technical Indicator Feature Engineering
- Market Regime Analysis
- Confidence-based Buy / Hold / Sell predictions
- Interactive Streamlit Dashboard
- Comprehensive Performance Evaluation

---

# 📌 Problem Statement

Traditional machine learning models assume that important features remain constant throughout the entire dataset.

However, financial markets are highly dynamic.

Indicators that work well during a bull market may become ineffective during a bear market.

This project proposes a framework that continuously updates the active feature set based on recent market data, enabling better adaptation to concept drift and changing market regimes.

---

# 🏗 Project Workflow

```
Historical Stock Data
        │
        ▼
Data Collection
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Sliding Window
        │
        ▼
Dynamic Feature Selection
(Mutual Information)
        │
        ▼
Random Forest Models
        │
        ▼
Walk-Forward Validation
        │
        ▼
Performance Evaluation
        │
        ▼
Dashboard Visualization
```

---

# ⚙ Features

### Data Collection

- Yahoo Finance API
- Multiple Stocks
- Historical OHLCV Data

### Feature Engineering

Technical Indicators include:

- SMA (10,20,50)
- EMA
- RSI
- MACD
- ATR
- Bollinger Bands
- OBV
- ROC
- Price Change
- Returns
- Volume Indicators

---

### Dynamic Feature Selection

Instead of using all features throughout training,

the model periodically:

- Calculates Mutual Information
- Ranks features
- Selects Top-K Features
- Retrains using only selected features

This allows the model to adapt as market behaviour changes.

---

### Walk-Forward Validation

Instead of randomly splitting data,

the project follows a realistic evaluation strategy.

For every prediction:

- Only past information is available
- No future data leakage
- Models are periodically retrained
- Predictions simulate real-world deployment

---

### Static Baseline

Uses:

- Fixed Feature Set
- Single Training Phase
- No Feature Updates

---

### Dynamic Model

Uses:

- Sliding Window
- Mutual Information
- Dynamic Top Features
- Periodic Retraining

---

# 🛠 Technology Stack

## Programming Language

- Python

## Libraries

- Pandas
- NumPy
- Scikit-learn
- yfinance
- Matplotlib
- Seaborn
- Streamlit

---

# 📂 Project Structure

```
Pulse/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── cleaned/
│
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── visualization/
│
├── reports/
│   ├── results/
│   ├── logs/
│   └── figures/
│
├── dashboard/
│
├── requirements.txt
├── README.md
└── main.py
```

---

# 📊 Evaluation Metrics

Machine Learning Metrics

- Accuracy
- Precision
- Recall
- F1 Score

Financial Analysis

- Buy / Sell / Hold Distribution
- Year-wise Accuracy
- Market Regime Performance
- Static vs Dynamic Comparison

---

# 📈 Experimental Results

Dynamic Feature Selection demonstrated improvements over the static baseline under the same Walk-Forward Validation framework.

Example (AAPL):

| Model | Accuracy |
|---------|----------|
| Static Random Forest | 34.53% |
| Dynamic Random Forest | 36.33% |

Key observations:

- Better F1 Score
- Better performance during Bull Markets
- Better adaptation during Sideways Markets
- Dynamic Feature Selection successfully changed feature importance over time

---

# 📚 Research Contribution

The primary contribution of this work is the implementation of a **Dynamic Feature Selection Framework** for streaming data.

The framework:

- Continuously updates feature importance
- Adapts to changing data distributions
- Reduces reliance on stale features
- Supports fair evaluation using Walk-Forward Validation

Although demonstrated using stock market data, the framework can be applied to:

- Healthcare Monitoring
- Fraud Detection
- Cybersecurity
- IoT Sensor Data
- Predictive Maintenance
- Real-Time Analytics

---

# 👥 Team

- Chintan Sameer Mehta
- Om Sanjay Kumavat
- Rutuja Hitendra Nagare
- Deodatta Abhyuday Pagar

Department of Computer Engineering

K. K. Wagh Institute of Engineering Education & Research

Savitribai Phule Pune University

---

# 📖 Future Work

- Live Stock Data Streaming
- Online Learning Models
- Reinforcement Learning
- Sentiment Analysis
- Explainable AI (XAI)
- Portfolio Optimization
- Deep Learning Models
- Multi-Asset Trading

---

# 📄 License

This project is developed for academic and research purposes.

---

## ⭐ If you found this project useful, consider giving it a Star.