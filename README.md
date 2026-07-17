# 📉 Nifty 50 Market Crash Alert System

A machine learning-powered market crash prediction system that estimates the probability of significant declines in the **Nifty 50 Index** over the next **21 trading days**.

Instead of predicting future prices, the project estimates the likelihood of market crashes using an ensemble of machine learning models and presents the results through an interactive Streamlit dashboard.

---
## Streamlit App

Click the link to run the [streamlit app](https://market-crash-alert-aqxkesdzc3obw2tet8urak.streamlit.app/)
### UI
<img width="1278" height="488" alt="Screenshot 2026-07-17 at 6 47 20 PM" src="https://github.com/user-attachments/assets/9bfd79a5-420b-46d9-8431-65539346bb36" />

<img width="1290" height="468" alt="Screenshot 2026-07-17 at 6 47 52 PM" src="https://github.com/user-attachments/assets/b99b17d7-099d-4527-8cf0-54fedeecbbe3" />

<img width="1264" height="676" alt="Screenshot 2026-07-17 at 6 48 37 PM" src="https://github.com/user-attachments/assets/2d91d633-0239-4994-a0e3-6ab6673c7e60" />


---

## 🚀 Features

- Predicts probability of:
  - >5% market crash
  - >8% market crash
  - >10% market crash
- Ensemble of four machine learning models
- Live market data from Yahoo Finance
- Automatic feature engineering
- Daily model retraining
- Interactive Streamlit dashboard
- Market snapshot with macroeconomic indicators
- Probability trend visualization
- Risk categorization
- Model comparison



---

# Project Workflow

```text
Yahoo Finance Data
        │
        ▼
Feature Engineering
        │
        ▼
Crash Label Generation
        │
        ▼
Train ML Models
        │
        ▼
Optuna Hyperparameter Tuning
        │
        ▼
Probability Calibration
        │
        ▼
Model Blending
        │
        ▼
Risk Prediction
        │
        ▼
Streamlit Dashboard
```

---

# Machine Learning Models

The project combines predictions from four ensemble models:

- CatBoost
- LightGBM
- XGBoost
- Random Forest

Each model is independently trained and calibrated before producing the final prediction.

---

# Features Used

### Technical Indicators

- Daily Returns
- Rolling Returns
- Rolling Volatility
- Drawdowns
- EMA50
- EMA200
- ATR

### Macroeconomic Indicators

- India VIX
- Gold Prices
- Crude Oil Prices
- USD/INR Exchange Rate

---

# Model Training

Hyperparameters were optimized using **Optuna**.

Validation uses:

- TimeSeriesSplit
- 21-day gap
- ROC-AUC optimization

This prevents data leakage and better simulates real-world forecasting.

---

# Probability Calibration

Financial crashes are rare events.

To obtain meaningful probabilities, every model is calibrated using **Platt Scaling** (`CalibratedClassifierCV`).

The final probability is computed by averaging the calibrated predictions from all four models.

---

# Risk Levels

| Probability | Risk |
|-------------|------|
| <15% | 🟢 Low |
| 15–35% | 🟡 Moderate |
| 35–60% | 🟠 High |
| >60% | 🔴 Extreme |

---

# Tech Stack

- Python
- Streamlit
- Scikit-learn
- CatBoost
- XGBoost
- LightGBM
- Optuna
- Plotly
- Pandas
- NumPy
- Yahoo Finance API


---

# Applications

- Portfolio Risk Monitoring
- Investment Decision Support
- Market Risk Assessment
- Financial Research
- Educational Demonstration of Machine Learning in Finance

---

# Limitations

- Predictions are probabilistic, not guarantees.
- Based solely on historical market behavior.
- Cannot anticipate unforeseen geopolitical or macroeconomic events.
- Requires periodic retraining to adapt to changing market conditions.

___
