"""
Nifty 50 Market Crash Predictor — Streamlit App
=================================================
Blends CatBoost, LightGBM, XGBoost and Random Forest (each calibrated with
Platt scaling) to estimate the probability of a >5% / >8% / >10% drawdown
in the Nifty 50 over the next 21 trading days.

Hyperparameters for every model x threshold combination were tuned offline
with Optuna (100 trials each, TimeSeriesSplit(3, gap=21), maximizing ROC-AUC)
and are hardcoded below — this app does NOT re-run Optuna, it only retrains
the final models on the latest data using those tuned settings.
"""

import time
import random

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

st.set_page_config(
    page_title="Nifty 50 Crash Predictor",
    layout="wide",
    page_icon="📉",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════
# THEME
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: linear-gradient(160deg, #070B14 0%, #0F172A 40%, #0C1322 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1320px;
}
/* ── Typography ────────────────────────────────────── */
.stApp, p, h1, h2, h3, h4, h5, h6, div, li, a, label, input, textarea, select, button, th, td, caption {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Fix for Streamlit's icon ligatures (expander arrows, etc.) */
.streamlit-expanderHeader > div:first-child,
.streamlit-expanderHeader > span:first-child {
    font-family: "Material Symbols Outlined", "Streamlit-Icons", sans-serif !important;
}

/* ── Headings ──────────────────────────────────────── */
h1 {
    font-size: 1.85rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px !important;
    background: linear-gradient(135deg, #FFF 0%, #94A3B8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
h2, .stMarkdown h2 {
    font-weight: 700 !important;
    color: #F1F5F9 !important;
}
h3, .stMarkdown h3 {
    font-weight: 600 !important;
    color: #CBD5E1 !important;
    font-size: 0.95rem !important;
}

/* ── Tabs ──────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 5px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    background: transparent !important;
    color: rgba(255,255,255,0.4) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 8px 22px !important;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.7) !important;
    background: rgba(255,255,255,0.04) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(255,255,255,0.1) !important;
    color: #FFF !important;
    font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── Metrics ───────────────────────────────────────── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 16px 20px !important;
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(255,255,255,0.12);
}
[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.4) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}
[data-testid="stMetricValue"] {
    color: #F1F5F9 !important;
    font-weight: 700 !important;
}
[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}

/* ── Alerts ────────────────────────────────────────── */
.stAlert {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 14px !important;
}

/* ── Toggle ────────────────────────────────────────── */
[data-testid="stToggle"] label {
    color: rgba(255,255,255,0.55) !important;
    font-weight: 500 !important;
}

/* ── Expander ──────────────────────────────────────── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    color: rgba(255,255,255,0.6) !important;
    font-weight: 600 !important;
    padding: 14px 20px !important;
    margin-top: 1.5rem !important;
}
[data-testid="stExpander"] details {
    border: none !important;
}

/* ── DataFrame ─────────────────────────────────────── */
.stDataFrame {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}

/* ── Scrollbar ─────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.18);
}

/* ── Divider ───────────────────────────────────────── */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 1.5rem 0;
}

/* ── Text ──────────────────────────────────────────── */
p, li, .stCaption p {
    color: rgba(255,255,255,0.5) !important;
    line-height: 1.65 !important;
}
.stCaption {
    color: rgba(255,255,255,0.3) !important;
}

/* ── Plotly modebar — hidden until hover ───────────── */
.js-plotly-plot .plotly .modebar {
    opacity: 0;
    transition: opacity 0.25s ease;
}
.js-plotly-plot:hover .plotly .modebar {
    opacity: 1;
}

/* ── Spinner ───────────────────────────────────────── */
.stSpinner > div > div {
    border-top-color: rgba(255,183,3,0.5) !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
MODEL_ORDER = ["CatBoost", "LightGBM", "XGBoost", "Random Forest"]
MODEL_COLORS = {
    "CatBoost": "#FFB703",
    "LightGBM": "#2DD4BF",
    "XGBoost": "#F87171",
    "Random Forest": "#60A5FA",
    "Blended": "#FFFFFF",
}

# ---------------------------------------------------------------------------
# Optuna-tuned hyperparameters
# ---------------------------------------------------------------------------
BEST_PARAMS = {
    5: {
        "CatBoost": {'iterations': 571, 'depth': 4, 'learning_rate': 0.05661931766180711,
                     'l2_leaf_reg': 0.29404685513045253, 'bagging_temperature': 9.513300124803049,
                     'border_count': 109, 'auto_class_weights': 'Balanced', 'random_seed': 42, 'verbose': 0,
                     'allow_writing_files': False},
        "LightGBM": {'n_estimators': 573, 'max_depth': 4, 'learning_rate': 0.11460173504474114,
                     'reg_lambda': 7.1373179990944635, 'bagging_fraction': 0.7753961167775515,
                     'bagging_freq': 7, 'num_leaves': 181, 'is_unbalance': True, 'random_seed': 42,
                     'verbose': -1, 'n_jobs': -1},
        "XGBoost": {'n_estimators': 723, 'max_depth': 3, 'learning_rate': 0.13031122961680683,
                    'reg_lambda': 1.6532565653288294, 'subsample': 0.651062541315901,
                    'colsample_bytree': 0.75927480805373, 'min_child_weight': 10,
                    'scale_pos_weight': 3.70875, 'random_state': 42, 'verbosity': 0, 'n_jobs': -1},
        "Random Forest": {'n_estimators': 246, 'max_depth': 6, 'min_samples_split': 11,
                           'min_samples_leaf': 18, 'max_features': 0.7942699246485865,
                           'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
    },
    8: {
        "CatBoost": {'iterations': 915, 'depth': 4, 'learning_rate': 0.011087474024424357,
                     'l2_leaf_reg': 0.19500328134451508, 'bagging_temperature': 7.082518856813896,
                     'border_count': 80, 'auto_class_weights': 'Balanced', 'random_seed': 42, 'verbose': 0,
                     'allow_writing_files': False},
        "LightGBM": {'n_estimators': 922, 'max_depth': 4, 'learning_rate': 0.2006367115709413,
                     'reg_lambda': 1.6587762315219337, 'bagging_fraction': 0.7115397396796371,
                     'bagging_freq': 6, 'num_leaves': 179, 'is_unbalance': True, 'random_seed': 42,
                     'verbose': -1, 'n_jobs': -1},
        "XGBoost": {'n_estimators': 763, 'max_depth': 3, 'learning_rate': 0.010012856552698592,
                    'reg_lambda': 0.10685862435027241, 'subsample': 0.7845462237838345,
                    'colsample_bytree': 0.6983415964281822, 'min_child_weight': 10,
                    'scale_pos_weight': 12.264084507042254, 'random_state': 42, 'verbosity': 0, 'n_jobs': -1},
        "Random Forest": {'n_estimators': 982, 'max_depth': 5, 'min_samples_split': 11,
                           'min_samples_leaf': 17, 'max_features': 0.8525987521149532,
                           'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
    },
    10: {
        "CatBoost": {'iterations': 517, 'depth': 5, 'learning_rate': 0.03144680401056169,
                     'l2_leaf_reg': 1.086518779621556, 'bagging_temperature': 0.9428351018151659,
                     'border_count': 68, 'auto_class_weights': 'Balanced', 'random_seed': 42, 'verbose': 0,
                     'allow_writing_files': False},
        "LightGBM": {'n_estimators': 519, 'max_depth': 10, 'learning_rate': 0.23791135901632696,
                     'reg_lambda': 4.1959659512522265, 'bagging_fraction': 0.9732734868735332,
                     'bagging_freq': 4, 'num_leaves': 235, 'is_unbalance': True, 'random_seed': 42,
                     'verbose': -1, 'n_jobs': -1},
        "XGBoost": {'n_estimators': 895, 'max_depth': 7, 'learning_rate': 0.2670107487945285,
                    'reg_lambda': 0.13007201825907758, 'subsample': 0.5006723546482568,
                    'colsample_bytree': 0.8533684200132019, 'min_child_weight': 2,
                    'scale_pos_weight': 25.528169014084508, 'random_state': 42, 'verbosity': 0, 'n_jobs': -1},
        "Random Forest": {'n_estimators': 476, 'max_depth': 14, 'min_samples_split': 3,
                           'min_samples_leaf': 3, 'max_features': 0.3481637188195886,
                           'class_weight': 'balanced', 'random_state': 42, 'n_jobs': -1},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════
def _download_with_retry(ticker: str, start_date: str, end_date: str,
                          max_retries: int = 5, base_delay: float = 3.0) -> pd.DataFrame:
    """yf.download wrapped with exponential backoff + jitter.

    Streamlit Community Cloud shares outbound IPs across many apps, so
    Yahoo Finance rate-limits (YFRateLimitError / 'Too Many Requests') are
    common there even though they rarely happen locally. Retrying with
    backoff resolves most of these transparently.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if df is not None and not df.empty:
                return df
            last_err = RuntimeError(f"Empty response for {ticker}")
        except Exception as e:  # yfinance raises YFRateLimitError, JSONDecodeError, etc.
            last_err = e

        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1.5)
            time.sleep(delay)

    raise RuntimeError(f"Failed to download '{ticker}' from Yahoo Finance after "
                        f"{max_retries} attempts: {last_err}")


@st.cache_data(ttl=14400, show_spinner=False)
def fetch_raw_data(today_str: str) -> pd.DataFrame:
    """Pull Nifty 50 + macro series from Yahoo Finance, up to today."""
    start_date = "2010-01-01"
    end_date = (pd.Timestamp(today_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    data = _download_with_retry("^NSEI", start_date, end_date)
    vix = _download_with_retry("^INDIAVIX", start_date, end_date)
    gold = _download_with_retry("GC=F", start_date, end_date)
    crude = _download_with_retry("CL=F", start_date, end_date)
    usdinr = _download_with_retry("INR=X", start_date, end_date)

    for df in (data, vix, gold, crude, usdinr):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel("Ticker")

    data["Vix"] = vix["Close"]
    data["Gold"] = gold["Close"]
    data["Crude"] = crude["Close"]
    data["Usdinr"] = usdinr["Close"]

    if "Volume" in data.columns:
        data = data.drop(columns=["Volume"])

    data = data.sort_index()

    # Guard against silently training on broken/empty data (e.g. a rate
    # limit that slipped through, or Yahoo returning a mostly-NaN frame).
    # This turns a confusing downstream CatBoostError ("Target contains
    # only one unique value") into a clear, actionable message.
    min_required_rows = 400  # comfortably more than the 252+21 the model needs
    if len(data) < min_required_rows or data["Close"].isna().mean() > 0.2:
        raise RuntimeError(
            f"Downloaded market data looks incomplete ({len(data)} rows, "
            f"{data['Close'].isna().mean():.0%} missing closes). This usually "
            "means Yahoo Finance rate-limited the request. Try refreshing "
            "in a minute."
        )

    return data


def create_features(df: pd.DataFrame, threshold_pct: int) -> pd.DataFrame:
    """Exact feature set from the notebooks; crash label depends on threshold_pct."""
    df = df.copy()
    df["vix_ret_5"] = df["Vix"].ffill().pct_change(5)
    df["vix_ret_21"] = df["Vix"].ffill().pct_change(21)

    df["gold_ret_21"] = df["Gold"].ffill().pct_change(21)
    df["crude_ret_21"] = df["Crude"].ffill().pct_change(21)
    df["usdinr_ret_21"] = df["Usdinr"].ffill().pct_change(21)

    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_21"] = df["Close"].pct_change(21)
    df["ret_63"] = df["Close"].pct_change(63)

    df["vol_21"] = df["ret_1"].rolling(21).std()
    df["vol_63"] = df["ret_1"].rolling(63).std()

    df["drawdown_21"] = df["Close"] / df["Close"].rolling(21).max() - 1
    df["drawdown_63"] = df["Close"] / df["Close"].rolling(63).max() - 1
    df["drawdown_252"] = df["Close"] / df["Close"].rolling(252).max() - 1

    df["ema50"] = df["Close"].ewm(span=50).mean()
    df["ema200"] = df["Close"].ewm(span=200).mean()
    df["ema50_dist"] = df["Close"] / df["ema50"] - 1
    df["ema200_dist"] = df["Close"] / df["ema200"] - 1

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["Close"]

    future_min = pd.Series(
        [df["Low"].iloc[i + 1:i + 22].min() if i + 21 < len(df) else np.nan
         for i in range(len(df))],
        index=df.index,
    )
    future_dd = future_min / df["Close"] - 1
    df["crash"] = (future_dd < -threshold_pct / 100).astype("int64")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# MODELING
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource(ttl=86400, show_spinner=False)
def train_models(threshold_pct: int, today_str: str, _raw_data: pd.DataFrame):
    """Train + Platt-calibrate all 4 models for one crash threshold."""
    data_features = create_features(_raw_data, threshold_pct)
    train_data = data_features.iloc[252:-21, :]
    pred_data = data_features.iloc[-21:, :-1]

    X = train_data.drop(columns=["crash"])
    y = train_data["crash"]

    params = BEST_PARAMS[threshold_pct]
    cv = TimeSeriesSplit(n_splits=5, gap=21)

    estimators = {
        "CatBoost": CatBoostClassifier(**params["CatBoost"]),
        "LightGBM": lgb.LGBMClassifier(**params["LightGBM"]),
        "XGBoost": xgb.XGBClassifier(**params["XGBoost"]),
        "Random Forest": RandomForestClassifier(**params["Random Forest"]),
    }

    models = {}
    for name in MODEL_ORDER:
        calibrated = CalibratedClassifierCV(estimator=estimators[name], method="sigmoid", cv=cv)
        calibrated.fit(X, y)
        models[name] = calibrated

    return models, pred_data


def predict_probabilities(models: dict, pred_data: pd.DataFrame) -> pd.DataFrame:
    """Per-model + blended crash probability for the last 21 trading days, EMA-smoothed."""
    out = {"date": pred_data.index}
    for name in MODEL_ORDER:
        raw = models[name].predict_proba(pred_data)[:, 1] * 100
        out[name] = pd.Series(raw).ewm(span=5).mean().values

    comparison = pd.DataFrame(out)
    comparison["Blended"] = comparison[MODEL_ORDER].mean(axis=1)
    return comparison


# ═══════════════════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _risk_level(prob: float):
    """Return (label, hex_color) for a given crash probability."""
    if prob < 15:
        return "LOW RISK", "#22C55E"
    elif prob < 35:
        return "MODERATE", "#EAB308"
    elif prob < 60:
        return "HIGH RISK", "#F97316"
    else:
        return "EXTREME", "#EF4444"


def probability_card_html(prob: float, threshold: int) -> str:
    """Build a frosted-glass probability card with risk badge and progress bar."""
    level, color = _risk_level(prob)
    return f"""
    <div style="
        background: linear-gradient(150deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.015) 100%);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 32px 28px 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    ">
        <div style="position:absolute;top:0;left:0;right:0;height:3px;
            background:linear-gradient(90deg, transparent, {color}, transparent);"></div>
        <div style="color:rgba(255,255,255,0.35);font-size:10.5px;font-weight:700;
            text-transform:uppercase;letter-spacing:2.5px;margin-bottom:14px;">
            Blended P(&gt;{threshold}% drop)
        </div>
        <div style="font-size:54px;font-weight:800;color:#FFF;line-height:1;margin-bottom:18px;">
            {prob:.2f}<span style="font-size:24px;font-weight:500;color:rgba(255,255,255,0.3);">%</span>
        </div>
        <div style="width:100%;height:5px;background:rgba(255,255,255,0.06);
            border-radius:3px;margin-bottom:18px;overflow:hidden;">
            <div style="width:{min(prob, 100)}%;height:100%;
                background:linear-gradient(90deg, {color}, {color}99);border-radius:3px;"></div>
        </div>
        <span style="display:inline-block;padding:5px 16px;border-radius:20px;
            background:{color}15;color:{color};font-size:10.5px;font-weight:700;
            letter-spacing:1.5px;">{level}</span>
        <div style="color:rgba(255,255,255,0.2);font-size:10.5px;margin-top:12px;">
            Next 21 trading days
        </div>
    </div>"""


def _hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert #RRGGBB to rgba(r, g, b, a) for Plotly compatibility."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def nifty_price_chart(raw_data: pd.DataFrame, days: int = 90) -> go.Figure:
    """Compact 90-day Nifty 50 area sparkline."""
    df = raw_data["Close"].iloc[-days:]
    color = "#22C55E" if df.iloc[-1] >= df.iloc[0] else "#EF4444"
    fig = go.Figure(go.Scatter(
        x=df.index, y=df.values,
        fill="tozeroy",
        fillcolor=_hex_to_rgba(color, 0.05),
        line=dict(color=color, width=1.8),
        hovertemplate="%{x|%b %d}: <b>%{y:,.2f}</b><extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=100,
        xaxis=dict(visible=False, rangebreaks=[dict(bounds=["sat", "mon"])]),
        yaxis=dict(visible=False),
        hovermode="x",
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="rgba(255,255,255,0.08)",  # <-- Fixed (one word)
            font=dict(color="#E2E8F0", size=11),
        ),
    )
    return fig


def trend_figure(comparison: pd.DataFrame, show_models: bool) -> go.Figure:
    """Probability trend line chart — dark, minimal, weekend gaps removed."""
    fig = go.Figure()
    if show_models:
        for name in MODEL_ORDER:
            fig.add_trace(go.Scatter(
                x=comparison["date"], y=comparison[name],
                mode="lines+markers", name=name,
                line=dict(color=MODEL_COLORS[name], width=1.5),
                marker=dict(size=3, color=MODEL_COLORS[name]),
            ))
    fig.add_trace(go.Scatter(
        x=comparison["date"], y=comparison["Blended"],
        mode="lines+markers", name="Blended",
        line=dict(color="#FFFFFF", width=2.5, dash="dot"),
        marker=dict(size=4, color="#FFFFFF"),
    ))
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color="rgba(255,255,255,0.45)"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False, showline=False,
            tickformat="%b %d",
            rangebreaks=[dict(bounds=["sat", "mon"])],
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.04)",
            zeroline=False, showline=False,
            ticksuffix="%",
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.08,
            xanchor="center", x=0.5,
            font=dict(size=11, color="rgba(255,255,255,0.45)"),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1E293B",
            bordercolor="rgba(255,255,255,0.08)",  # <-- Fixed (one word)
            font=dict(color="#E2E8F0", size=11),
        ),
        margin=dict(l=10, r=10, t=45, b=10),
        height=400,
    )
    return fig


def _delta_str(current: float, previous: float) -> str:
    """Format a day-over-day delta as '+12.34 (+0.45%)'."""
    d = current - previous
    p = (d / previous) * 100 if previous else 0
    return f"{d:+.2f} ({p:+.2f}%)"


def render_threshold_tab(threshold_pct: int, raw_data: pd.DataFrame, today_str: str):
    with st.spinner(f"Training models for the {threshold_pct}% threshold…"):
        models, pred_data = train_models(threshold_pct, today_str, raw_data)
        comparison = predict_probabilities(models, pred_data)

    latest = comparison.iloc[-1]
    latest_date = pd.Timestamp(latest["date"]).strftime("%Y-%m-%d")
    st.caption(f"Latest prediction window ends: **{latest_date}**")

    show_models = st.toggle(
        "Show probability chart & model breakdown",
        key=f"toggle_{threshold_pct}", value=False,
    )

    if show_models:
        col_card, col_chart = st.columns([5, 7])
        with col_card:
            st.markdown(probability_card_html(latest["Blended"], threshold_pct), unsafe_allow_html=True)
        with col_chart:
            st.plotly_chart(trend_figure(comparison, True), use_container_width=True)

        st.markdown("**Per-model probability (today, EMA-smoothed):**")
        cols = st.columns(len(MODEL_ORDER))
        for c, name in zip(cols, MODEL_ORDER):
            c.metric(name, f"{latest[name]:.2f}%")

        with st.expander("Full table — last 21 trading days"):
            display_df = comparison.copy()
            display_df["date"] = pd.to_datetime(display_df["date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(
                display_df.style.format({c: "{:.2f}" for c in MODEL_ORDER + ["Blended"]}),
                use_container_width=True, height=420,
            )
    else:
        st.markdown(
            f'<div style="max-width:480px;margin:0 auto;">'
            f'{probability_card_html(latest["Blended"], threshold_pct)}'
            f'</div>',
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    today = pd.Timestamp.today()
    today_str = today.strftime("%Y-%m-%d")

    # ── Header ───────────────────────────────────────────────────────────
    col_head, col_date = st.columns([4, 1])
    with col_head:
        st.title("📉 Nifty 50 Crash Predictor")
        st.markdown(
            "Blended ML estimates for **>5%**, **>8%**, **>10%** drawdowns "
            "over the next 21 trading days."
        )
    with col_date:
        st.markdown(
            f'<div style="text-align:right;padding-top:8px;">'
            f'<div style="color:rgba(255,255,255,0.25);font-size:10px;'
            f'text-transform:uppercase;letter-spacing:1.5px;font-weight:600;">Today</div>'
            f'<div style="color:rgba(255,255,255,0.7);font-size:15px;font-weight:700;'
            f'margin-top:2px;">{today.strftime("%d %b")}</div>'
            f'<div style="color:rgba(255,255,255,0.35);font-size:12px;">{today.strftime("%A, %Y")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Data ─────────────────────────────────────────────────────────────
    with st.spinner("Fetching market data…"):
        try:
            raw_data = fetch_raw_data(today_str)
        except Exception as e:
            st.error(
                "⚠️ Couldn't fetch market data from Yahoo Finance right now "
                "(likely a temporary rate limit). Please refresh in a minute."
            )
            st.caption(f"Details: {e}")
            return
    if raw_data.empty:
        st.error("Could not fetch market data. Please try again shortly.")
        return

    filled = raw_data.ffill()
    latest_row = filled.iloc[-1]
    prev_row = filled.iloc[-2] if len(filled) > 1 else latest_row
    data_date = raw_data.index[-1].strftime("%Y-%m-%d")

    # ── Market Snapshot ──────────────────────────────────────────────────
    st.markdown("### Market Snapshot")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Nifty 50", f"{latest_row['Close']:,.2f}",
              _delta_str(latest_row['Close'], prev_row['Close']))
    m2.metric("India VIX", f"{latest_row['Vix']:.2f}",
              _delta_str(latest_row['Vix'], prev_row['Vix']))
    m3.metric("Gold", f"{latest_row['Gold']:,.2f}",
              _delta_str(latest_row['Gold'], prev_row['Gold']))
    m4.metric("Crude Oil", f"{latest_row['Crude']:,.2f}",
              _delta_str(latest_row['Crude'], prev_row['Crude']))
    m5.metric("USD/INR", f"{latest_row['Usdinr']:.2f}",
              _delta_str(latest_row['Usdinr'], prev_row['Usdinr']))

    # ── Nifty 90-day sparkline ───────────────────────────────────────────
    st.plotly_chart(nifty_price_chart(raw_data), use_container_width=True)
    st.caption(f"Data as of {data_date} · 90-day Nifty 50 close")

    st.markdown("---")

    # ── Threshold Tabs ───────────────────────────────────────────────────
    tab5, tab8, tab10 = st.tabs([">5% drawdown", ">8% drawdown", ">10% drawdown"])
    with tab5:
        render_threshold_tab(5, raw_data, today_str)
    with tab8:
        render_threshold_tab(8, raw_data, today_str)
    with tab10:
        render_threshold_tab(10, raw_data, today_str)

    # ── Footer ───────────────────────────────────────────────────────────
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    with st.expander("ℹ️  Methodology & Notes"):
        st.markdown("""
**Models:** CatBoost, LightGBM, XGBoost, Random Forest — each with Optuna-tuned
hyperparameters (100 trials, TimeSeriesSplit with 21-day gap, ROC-AUC objective).

**Calibration:** Every model is wrapped in `CalibratedClassifierCV(method="sigmoid")`
(Platt scaling) so that raw scores from class-weight / scale_pos_weight
imbalance handling become true probabilities.

**Blending:** Simple arithmetic mean of the four calibrated probabilities.

**Features:** 21 technical & macro features including return windows (1/5/21/63d),
realized volatility, drawdown depths, EMA distances, ATR, and VIX / Gold / Crude / USD-INR
21-day returns.

**Retraining:** Models are retrained daily on the latest data and cached for 24 h.
        """)


if __name__ == "__main__":
    main()
