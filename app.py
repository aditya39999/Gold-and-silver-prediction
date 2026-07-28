"""
Gold and Silver Prediction
A market forecasting platform for precious metals built with Streamlit.

Run locally with:
    streamlit run app.py
"""

import os
import time
import random
import tempfile
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import plotly.graph_objects as go
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

warnings.filterwarnings("ignore")

# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

PROJECT_NAME = "Gold and Silver Prediction"
VERSION = "1.3.0"

RANDOM_STATE = 42
DEFAULT_FORECAST_DAYS = 30
INDICATOR_WINDOW = 250
CV_SPLITS = 5

MARKET_SYMBOLS = {"Gold": "GC=F", "Silver": "SI=F"}
TROY_OUNCE = 31.1034768

CURRENCIES = {
    "USD": 1.00, "INR": 86.00, "EUR": 0.86, "GBP": 0.74,
    "AED": 3.67, "JPY": 147.00, "CAD": 1.37, "AUD": 1.53, "SGD": 1.28,
}

START_DATE = "2010-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

styles = getSampleStyleSheet()

st.set_page_config(
    page_title=PROJECT_NAME,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# STYLING (Cream and Gold Luxury Theme with Glassmorphism & Pulses)
# ==========================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

.block-container { padding-top: 1rem; max-width: 1400px; }
div[data-testid="stAppViewBlockContainer"] { padding-top: 1rem; }
div[data-testid="stDecoration"] { display: none; }

/* Hide default streamlit sidebar for clean top navbar experience */
[data-testid="stSidebar"] {
    background: rgba(247, 241, 228, 0.95);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(184,137,46,0.2);
}

.stApp {
    background: #F7F1E4;
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 19px;
}

h1, h2, h3, .app-header h1 {
    font-family: 'Playfair Display', Georgia, serif !important;
}

.stApp, .stApp p, .stApp span, .stApp label, .stMarkdown, .stMarkdown p {
    color: #2E271F;
    font-size: 18px;
}

/* Cinematic Hero Section */
.landing-hero {
    text-align: center;
    padding: 3rem 1rem 1.5rem 1rem;
    position: relative;
    background: radial-gradient(circle at center, rgba(232,197,106,0.15) 0%, rgba(247,241,228,0) 70%);
}
.landing-hero h1 {
    font-size: 4rem;
    color: #5C4A32;
    margin-bottom: 0.5rem;
    letter-spacing: 0.04em;
    animation: hero-fade 1.2s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
.landing-subtitle {
    font-size: 1.5rem;
    color: #8C6A2E;
    font-style: italic;
    margin-bottom: 2.5rem;
    animation: hero-fade 1.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes hero-fade {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Glass-style interactive cards with hover lift, glow and scroll-reveal */
.glass-card {
    background: rgba(255, 253, 247, 0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(184, 137, 46, 0.3);
    border-top: 5px solid #B8892E;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(46, 39, 31, 0.07);
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    animation: card-reveal 0.8s ease both;
}
.glass-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 15px 40px rgba(184, 137, 46, 0.2);
    border-color: #B8892E;
}
@keyframes card-reveal {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}
.glass-card h3 {
    font-family: 'Playfair Display', Georgia, serif;
    color: #5C4A32;
    margin-bottom: 12px;
}

/* Feature Grid Cards */
.feature-glass-box {
    background: rgba(255, 253, 247, 0.7);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(184, 137, 46, 0.25);
    border-radius: 10px;
    padding: 28px 22px;
    text-align: center;
    height: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(46,39,31,0.04);
}
.feature-glass-box:hover {
    transform: translateY(-5px);
    background: rgba(255, 253, 247, 0.95);
    border-color: rgba(184, 137, 46, 0.6);
    box-shadow: 0 8px 25px rgba(184,137,46,0.15);
}
.feature-icon {
    font-size: 32px;
    margin-bottom: 12px;
    color: #B8892E;
}
.feature-glass-box h4 {
    font-family: 'Playfair Display', Georgia, serif;
    color: #5C4A32;
    margin-bottom: 10px;
    font-size: 22px;
}

.metric-card {
    background: #FFFDF7;
    padding: 16px;
    border-radius: 6px;
    border: 1px solid rgba(184,137,46,0.25);
    border-top: 4px solid #B8892E;
    text-align: center;
    box-shadow: 0 1px 3px rgba(46,39,31,0.06);
}
.metric-card h4 {
    color: #7A6B4E;
    font-weight: 700;
    font-size: 12px;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}
.metric-card h2 {
    margin: 0;
    font-size: 27px;
    font-weight: 700;
    color: #1F1811;
    font-family: 'Playfair Display', Georgia, serif;
}

.trend-pill {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    margin-top: 9px;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 700;
}
.trend-pill.trend-up { background: rgba(76,107,72,0.14); color: #385B34; }
.trend-pill.trend-down { background: rgba(166,73,58,0.14); color: #8C3A2C; }
.trend-pill.trend-flat { background: rgba(140,122,84,0.14); color: #6B5D46; }

/* Glowing Signal Badge with Pulse Ring */
.signal-badge {
    position: relative;
    display: inline-block;
    margin-top: 4px;
    padding: 9px 25px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: 700;
    font-family: 'Playfair Display', Georgia, serif;
    letter-spacing: 0.05em;
    animation: badge-pop 0.4s cubic-bezier(.34,1.56,.64,1) both;
    z-index: 1;
}
.signal-badge::before {
    content: '';
    position: absolute;
    inset: -4px;
    border-radius: 12px;
    z-index: -1;
    animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}
.signal-strong-buy { background: #33502F; color: #F7F1E4; box-shadow: 0 0 15px rgba(51,80,47,0.6); }
.signal-strong-buy::before { border: 2px solid #33502F; }

.signal-buy { background: #DCE8D8; color: #2E4A2A; border: 1px solid rgba(76,107,72,0.4); }
.signal-buy::before { border: 2px solid #4C6B48; }

.signal-hold { background: #F3E4C2; color: #7A5B1E; border: 1px solid rgba(184,137,46,0.4); }
.signal-hold::before { border: 2px solid #B8892E; }

.signal-sell { background: #8C3A2C; color: #FBEDE9; box-shadow: 0 0 15px rgba(140,58,44,0.6); }
.signal-sell::before { border: 2px solid #8C3A2C; }

@keyframes pulse-ring {
    0% { transform: scale(0.8); opacity: 1; }
    100% { transform: scale(1.3); opacity: 0; }
}
@keyframes badge-pop {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}

.stButton > button, .stDownloadButton > button {
    background: #EFE4CD;
    color: #2E271F;
    border: 1px solid #B8892E;
    border-radius: 999px;
    font-family: 'Cormorant Garamond', Georgia, serif;
    letter-spacing: 0.05em;
    font-weight: 600;
    font-size: 17px;
    padding: 0.5rem 1.2rem;
    transition: transform 0.15s ease, background 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #B8892E;
    color: #FFFDF7;
    border-color: #B8892E;
    transform: translateY(-1px);
}
.stButton > button[kind="primary"] {
    background: #B8892E;
    color: #FFFDF7;
    border: 1px solid #8C6A2E;
    box-shadow: 0 4px 15px rgba(184,137,46,0.4);
}
.stButton > button[kind="primary"]:hover {
    background: #8C6A2E;
    box-shadow: 0 6px 20px rgba(184,137,46,0.6);
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ==========================================================
# SESSION STATE NAVIGATION & INTRO SETUP
# ==========================================================

if "intro_played" not in st.session_state:
    st.session_state.intro_played = False

if "active_nav" not in st.session_state:
    st.session_state.active_nav = "Home"


def render_molten_gold_intro():
    """3-second cinematic Molten Gold Reveal effect with droplet, ripple, and curtain melt."""
    intro_html = """
    <style>
    .molten-overlay {
        position: fixed; inset: 0; z-index: 999999;
        pointer-events: none;
        background: #0B0A08;
        display: flex;
        align-items: center;
        justify-content: center;
        animation: curtain-melt 3s cubic-bezier(0.77, 0, 0.175, 1) forwards;
    }
    .gold-droplet {
        position: absolute;
        width: 12px; height: 12px;
        background: radial-gradient(circle, #FFF5C0 0%, #D4AF37 50%, #8C6A2E 100%);
        border-radius: 50%;
        box-shadow: 0 0 25px 8px rgba(212,175,55,0.8);
        animation: drop-fall 1s ease-in forwards;
    }
    @keyframes drop-fall {
        0% { transform: translateY(-150vh) scale(0.5); opacity: 0; }
        50% { opacity: 1; }
        100% { transform: translateY(0) scale(1.5); opacity: 1; }
    }
    .gold-ripple {
        position: absolute;
        width: 10px; height: 10px;
        border: 3px solid rgba(212,175,55,0.9);
        border-radius: 50%;
        box-shadow: 0 0 30px rgba(212,175,55,0.6) inset;
        animation: ripple-spread 1.8s ease-out 0.9s forwards;
        opacity: 0;
    }
    @keyframes ripple-spread {
        0% { width: 10px; height: 10px; opacity: 1; transform: scale(1); }
        100% { width: 350vw; height: 350vw; opacity: 0; transform: scale(1); border-width: 50px; }
    }
    @keyframes curtain-melt {
        0% { clip-path: inset(0 0 0 0); opacity: 1; }
        75% { clip-path: inset(0 0 0 0); opacity: 1; }
        100% { clip-path: inset(100% 0 0 0); opacity: 0; visibility: hidden; }
    }
    </style>
    <div class="molten-overlay">
        <div class="gold-droplet"></div>
        <div class="gold-ripple"></div>
    </div>
    """
    st.markdown(intro_html, unsafe_allow_html=True)


if not st.session_state.intro_played:
    render_molten_gold_intro()
    skip_clicked = st.button("Skip Intro", key="skip_intro_btn")
    st.session_state.intro_played = True
    if skip_clicked:
        st.rerun()

# ==========================================================
# DATA DOWNLOAD & BACKEND UTILITIES
# ==========================================================

def download_with_retry(symbol, start, end, retries=3, delay=2):
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or df.empty:
                raise ValueError("Empty dataframe returned")
            return df
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise RuntimeError(f"Failed to download {symbol} after {retries} tries: {last_err}")


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data():
    market_data = {}
    for metal, symbol in MARKET_SYMBOLS.items():
        df = download_with_retry(symbol, START_DATE, END_DATE)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        market_data[metal] = df
    return market_data


def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["SMA_5"] = data["Close"].rolling(5).mean()
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["EMA_10"] = data["Close"].ewm(span=10).mean()
    data["EMA_20"] = data["Close"].ewm(span=20).mean()
    data["Return"] = data["Close"].pct_change()
    data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Volatility"] = data["Return"].rolling(10).std()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data["RSI"] = 100 - (100 / (1 + rs))

    ema12 = data["Close"].ewm(span=12).mean()
    ema26 = data["Close"].ewm(span=26).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9).mean()

    rolling_std = data["Close"].rolling(20).std()
    data["BB_UPPER"] = data["SMA_20"] + rolling_std * 2
    data["BB_LOWER"] = data["SMA_20"] - rolling_std * 2

    high_low = data["High"] - data["Low"]
    high_close = (data["High"] - data["Close"].shift()).abs()
    low_close = (data["Low"] - data["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR_14"] = true_range.rolling(14).mean()

    data["Momentum_10"] = data["Close"] - data["Close"].shift(10)
    data["Lag_Return_1"] = data["LogReturn"].shift(1)
    data["Lag_Return_2"] = data["LogReturn"].shift(2)
    data["Lag_Return_3"] = data["LogReturn"].shift(3)
    data["Lag_Return_5"] = data["LogReturn"].shift(5)
    data["Lag_Return_10"] = data["LogReturn"].shift(10)
    data["DayOfWeek"] = data.index.dayofweek
    data["Month"] = data.index.month
    return data


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Volume",
    "SMA_5", "SMA_10", "SMA_20", "SMA_50",
    "EMA_10", "EMA_20",
    "Return", "Volatility", "RSI",
    "MACD", "MACD_SIGNAL",
    "BB_UPPER", "BB_LOWER",
    "ATR_14", "Momentum_10",
    "Lag_Return_1", "Lag_Return_2", "Lag_Return_3", "Lag_Return_5", "Lag_Return_10",
    "DayOfWeek", "Month",
]


@st.cache_data(ttl=3600, show_spinner=False)
def build_featured_data(_market_data=None):
    market_data = _market_data if _market_data is not None else load_market_data()
    return {metal: add_indicators(df).dropna().copy() for metal, df in market_data.items()}


def train_models_for_metal(df):
    data = df.copy()
    data["Target"] = data["LogReturn"].shift(-1)
    data = data.dropna()

    X = data[FEATURE_COLUMNS]
    y = data["Target"]
    close_at_t = data["Close"]

    split_index = int(len(data) * 0.85)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    close_test = close_at_t.iloc[split_index:]
    dates_test = data.index[split_index:]

    rf = RandomForestRegressor(n_estimators=400, max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
    gbm = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, random_state=RANDOM_STATE)

    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)

    pred_return_rf = rf.predict(X_test)
    pred_return_gbm = gbm.predict(X_test)
    pred_return_blend = 0.5 * pred_return_rf + 0.5 * pred_return_gbm

    actual_price_next = close_test.values * np.exp(y_test.values)
    predicted_price_next = close_test.values * np.exp(pred_return_blend)

    prediction_history = pd.DataFrame({"Date": dates_test, "Actual": actual_price_next, "Predicted": predicted_price_next})

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    cv_dir_acc = []
    for tr_idx, val_idx in tscv.split(X):
        rf_cv = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
        rf_cv.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        pred_cv = rf_cv.predict(X.iloc[val_idx])
        actual_dir = np.sign(y.iloc[val_idx].values)
        pred_dir = np.sign(pred_cv)
        cv_dir_acc.append(np.mean(actual_dir == pred_dir))

    directional_accuracy = float(np.mean(cv_dir_acc)) * 100
    perf = {
        "R2 (price)": r2_score(actual_price_next, predicted_price_next),
        "MAE (price)": mean_absolute_error(actual_price_next, predicted_price_next),
        "RMSE (price)": np.sqrt(mean_squared_error(actual_price_next, predicted_price_next)),
        "Directional Accuracy %": round(directional_accuracy, 2),
    }
    return {"rf": rf, "gbm": gbm}, perf, prediction_history


@st.cache_resource(show_spinner=False)
def train_all_models():
    featured_data = build_featured_data()
    models, performance, prediction_history = {}, {}, {}
    for metal, df in featured_data.items():
        models[metal], performance[metal], prediction_history[metal] = train_models_for_metal(df)
    performance_df = pd.DataFrame(performance).T.round(4)
    feature_importance = {}
    for metal in models:
        imp = pd.DataFrame({"Feature": FEATURE_COLUMNS, "Importance": models[metal]["rf"].feature_importances_}).sort_values("Importance", ascending=False)
        feature_importance[metal] = imp
    return models, performance, performance_df, feature_importance, prediction_history


def predict_next_return(model_dict, feature_row: pd.DataFrame) -> float:
    rf_pred = model_dict["rf"].predict(feature_row)[0]
    gbm_pred = model_dict["gbm"].predict(feature_row)[0]
    return 0.5 * float(rf_pred) + 0.5 * float(gbm_pred)


def forecast_prices(metal, forecast_days, models, featured_data):
    model_dict = models[metal]
    full_history = featured_data[metal].copy()
    window = full_history.tail(INDICATOR_WINDOW + forecast_days).copy()
    base_cols = ["Open", "High", "Low", "Close", "Volume"]

    predictions_out, future_dates = [], []
    current_date = window.index[-1]

    for _ in range(forecast_days):
        latest = window.iloc[-1:].copy()
        X_latest = latest[FEATURE_COLUMNS]
        pred_return = predict_next_return(model_dict, X_latest)
        last_close = float(latest["Close"].values[0])
        predicted_price = last_close * np.exp(pred_return)

        current_date = current_date + timedelta(days=1)
        future_dates.append(current_date)
        predictions_out.append(predicted_price)

        new_row = pd.DataFrame({
            "Open": [float(predicted_price)],
            "High": [float(predicted_price) * 1.002],
            "Low": [float(predicted_price) * 0.998],
            "Close": [float(predicted_price)],
            "Volume": [float(window["Volume"].tail(10).mean())],
        }, index=[current_date])

        window = pd.concat([window[base_cols], new_row[base_cols]])
        window[base_cols] = window[base_cols].astype("float64")
        window = add_indicators(window)
        window = window.ffill().bfill()

    return pd.DataFrame({"Date": future_dates, "Forecast": predictions_out})


def convert_price(price, currency):
    return price * CURRENCIES[currency]

def convert_price_unit(price, unit):
    if unit == "Gram":
        return round(price / TROY_OUNCE, 2)
    return round(price, 2)

def get_unit_symbol(unit):
    return "/g" if unit == "Gram" else "/oz"

def dashboard_metrics(metal, currency, models, featured_data, performance, forecast_days=30):
    history = featured_data[metal]
    forecast = forecast_prices(metal, forecast_days, models, featured_data)

    current = convert_price(history["Close"].iloc[-1], currency)
    future = convert_price(forecast["Forecast"].iloc[-1], currency)
    change = ((future - current) / current) * 100

    if change > 2:
        signal = "STRONG BUY"
    elif change > 0:
        signal = "BUY"
    elif change > -2:
        signal = "HOLD"
    else:
        signal = "SELL"

    dir_acc = performance[metal]["Directional Accuracy %"]
    r2 = max(performance[metal]["R2 (price)"], 0) * 100
    confidence = round(0.7 * dir_acc + 0.3 * r2, 2)

    return {
        "Current Price": round(current, 2),
        "Forecast Price": round(future, 2),
        "Expected Return": round(change, 2),
        "Signal": signal,
        "Confidence": confidence,
        "Forecast": forecast,
    }


def signal_css_class(signal):
    if "STRONG BUY" in signal:
        return "signal-strong-buy"
    if "BUY" in signal:
        return "signal-buy"
    if "SELL" in signal:
        return "signal-sell"
    return "signal-hold"

def confidence_tier(confidence):
    if confidence >= 80:
        return "High", "#2E4A2A", "#4C6B48"
    if confidence >= 65:
        return "Medium", "#7A5B1E", "#B8892E"
    return "Low", "#8C3A2C", "#A6493A"

def compute_24h_change(featured_data, metal):
    df = featured_data[metal]
    if len(df) < 2:
        return 0.0
    last_close = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    return ((last_close - prev_close) / prev_close) * 100


def trend_arrow(change_pct, flat_threshold=0.05):
    if change_pct > flat_threshold:
        return "▲", "trend-up"
    if change_pct < -flat_threshold:
        return "▼", "trend-down"
    return "▬", "trend-flat"


def create_mini_chart(series, line_color):
    fig = go.Figure()
    if not series.empty:
        y = series.values
        fig.add_trace(go.Scatter(
            y=y, mode="lines",
            line=dict(color=line_color, width=2),
            hoverinfo="skip"
        ))
    fig.update_layout(
        height=70,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False)
    return fig


def compute_market_pulse(metal, featured_data, performance):
    df = featured_data[metal]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_SIGNAL"].iloc[-1]
    score = 50.0
    if macd > macd_signal:
        score += 15
    else:
        score -= 15
    if 40 <= rsi <= 60:
        score += 10
    elif rsi > 70 or rsi < 30:
        score -= 10
    score = max(0, min(100, score))
    if score >= 60:
        label = "Bullish"
    elif score <= 40:
        label = "Bearish"
    else:
        label = "Neutral"
    return {"label": label, "score": score}


def compute_confidence_band(metal, forecast, featured_data, z=1.28, vol_window=60):
    vol = featured_data[metal]["LogReturn"].tail(vol_window).std()
    if not np.isfinite(vol) or vol <= 0:
        vol = featured_data[metal]["LogReturn"].std()
    horizon = np.arange(1, len(forecast) + 1)
    band = vol * np.sqrt(horizon) * z
    upper = forecast["Forecast"].values * np.exp(band)
    lower = forecast["Forecast"].values * np.exp(-band)
    return upper, lower

def create_forecast_chart(metal, forecast, featured_data, currency):
    history = featured_data[metal].copy()
    upper, lower = compute_confidence_band(metal, forecast, featured_data)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index, y=history["Close"] * CURRENCIES[currency], mode="lines",
        name="Historical", line=dict(color="#B8892E", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast["Date"], forecast["Date"][::-1]]),
        y=list(upper * CURRENCIES[currency]) + list(lower[::-1] * CURRENCIES[currency]),
        fill="toself",
        fillcolor="rgba(76,107,72,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        hoverinfo="skip",
        name="~80% Confidence Range",
    ))

    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=forecast["Forecast"] * CURRENCIES[currency], mode="lines+markers",
        name="Forecast",
        line=dict(color="#4C6B48", width=3, dash="dash"),
        marker=dict(size=6)
    ))

    default_window_days = 90
    visible_history = history.tail(default_window_days)
    visible_low = min(visible_history["Close"].min(), float(np.min(lower))) * CURRENCIES[currency]
    visible_high = max(visible_history["Close"].max(), float(np.max(upper))) * CURRENCIES[currency]
    y_pad = (visible_high - visible_low) * 0.08 or visible_high * 0.01
    default_y_range = [visible_low - y_pad, visible_high + y_pad]

    fig.update_layout(
        height=560,
        hovermode="x unified",
        margin=dict(l=55, r=30, t=170, b=10),
        showlegend=True,
        template="plotly_white",
        paper_bgcolor="#FFFDF7",
        plot_bgcolor="#FFFDF7",
        legend=dict(
            orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
            font=dict(color="#2E2013", size=12, family="Georgia, serif"),
            bgcolor="rgba(255,253,247,0.9)",
        ),
        annotations=[
            dict(
                text=f"<b>{metal} Price: History &amp; Forecast</b>",
                xref="paper", yref="paper",
                x=0, xanchor="left", y=1.34, yanchor="bottom",
                showarrow=False,
                font=dict(color="#241B0F", size=19, family="'Playfair Display', Georgia, serif"),
            )
        ]
    )
    fig.update_xaxes(
        showgrid=False,
        tickfont=dict(color="#2E2013", size=12, family="Georgia, serif"),
        linecolor="#8C7A54",
        rangeslider=dict(visible=True, thickness=0.05, bgcolor="#EFE4CD"),
        range=[history.index[-default_window_days], forecast["Date"].iloc[-1]],
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#EFE4CD",
        title=dict(text="Price", font=dict(color="#2E2013", size=13, family="Georgia, serif")),
        tickfont=dict(color="#2E2013", size=12, family="Georgia, serif"),
        range=default_y_range,
    )
    return fig

def signal_reasoning(metal, stats, featured_data, performance, forecast_days):
    df = featured_data[metal]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_SIGNAL"].iloc[-1]
    dir_acc = performance[metal]["Directional Accuracy %"]

    reasons = [
        f"Model projects a {stats['Expected Return']:+.2f}% move over the next {forecast_days} days."
    ]
    if rsi >= 70:
        reasons.append(f"RSI is {rsi:.0f} — overbought, which raises pullback risk.")
    elif rsi <= 30:
        reasons.append(f"RSI is {rsi:.0f} — oversold, which can favor a bounce.")
    else:
        reasons.append(f"RSI is {rsi:.0f} — neutral, no strong overbought/oversold pressure.")

    if macd > macd_signal:
        reasons.append("MACD is above its signal line (bullish crossover).")
    else:
        reasons.append("MACD is below its signal line (bearish crossover).")

    reasons.append(
        f"Historical directional accuracy for {metal} is {dir_acc:.1f}%, "
        f"contributing to the {stats['Confidence']}% overall confidence score."
    )
    return reasons

def stream_reasoning(reasons_list):
    """Generator to simulate typewriter effect for AI reasoning"""
    for r in reasons_list:
        yield f"- {r}\n"
        time.sleep(0.04)

# ==========================================================
# LOAD MODELS & DATA ON STARTUP
# ==========================================================

with st.spinner("Initializing market data & models..."):
    market_data = load_market_data()
    featured_data = build_featured_data(market_data)
    models, performance, performance_df, feature_importance, prediction_history = train_all_models()

# ==========================================================
# CLEAN TOP NAVIGATION BAR (Accessible Across All Views)
# ==========================================================

nav_options = ["Home", "Forecast", "Compare", "Advisor", "Analytics", "About"]

cols_nav = st.columns(len(nav_options))
for i, opt in enumerate(nav_options):
    with cols_nav[i]:
        btn_type = "primary" if st.session_state.active_nav == opt else "secondary"
        if st.button(opt, key=f"nav_btn_{opt}", use_container_width=True, type=btn_type):
            st.session_state.active_nav = opt
            st.rerun()

st.markdown("<hr style='border-color:rgba(184,137,46,0.25); margin: 10px 0 20px 0;'>", unsafe_allow_html=True)

# ==========================================================
# PAGE ROUTING
# ==========================================================

if st.session_state.active_nav == "Home":
    st.markdown("""
        <div class="landing-hero">
            <h1>Intelligence Behind Precious Metals</h1>
            <div class="landing-subtitle">Advanced Machine Learning &amp; Market Forecasting Platform</div>
        </div>
    """, unsafe_allow_html=True)

    gold_close = featured_data["Gold"]["Close"].iloc[-1]
    silver_close = featured_data["Silver"]["Close"].iloc[-1]
    gold_chg = compute_24h_change(featured_data, "Gold")
    silver_chg = compute_24h_change(featured_data, "Silver")

    g_arr, g_cls = trend_arrow(gold_chg)
    s_arr, s_cls = trend_arrow(silver_chg)

    gold_series = featured_data["Gold"]["Close"].tail(30)
    silver_series = featured_data["Silver"]["Close"].tail(30)

    col_g, col_s = st.columns(2)
    with col_g:
        st.markdown(f"""
            <div class="glass-card">
                <h3>Gold Futures (GC=F)</h3>
                <h2 style="font-family:'Playfair Display',serif; color:#2E271F; margin:10px 0;">USD {gold_close:,.2f}</h2>
                <div class="trend-pill {g_cls}">{g_arr} {gold_chg:+.2f}% (24H)</div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_mini_chart(gold_series, "#B8892E"), use_container_width=True, config={"displayModeBar": False})

    with col_s:
        st.markdown(f"""
            <div class="glass-card">
                <h3>Silver Futures (SI=F)</h3>
                <h2 style="font-family:'Playfair Display',serif; color:#2E271F; margin:10px 0;">USD {silver_close:,.2f}</h2>
                <div class="trend-pill {s_cls}">{s_arr} {silver_chg:+.2f}% (24H)</div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_mini_chart(silver_series, "#6B7280"), use_container_width=True, config={"displayModeBar": False})

    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
    with c_btn2:
        if st.button("Explore Markets →", type="primary", use_container_width=True):
            st.session_state.active_nav = "Forecast"
            st.rerun()

    st.markdown("<br><hr style='border-color:rgba(184,137,46,0.2);'><br>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center; font-family:Playfair Display,serif; margin-bottom:25px;'>Platform Architecture</h3>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
            <div class="feature-glass-box">
                <div class="feature-icon">⚡</div>
                <h4>Live Markets</h4>
                <p style="font-size:16px; color:#6B5D46;">Real-time intraday price tracking, sparkline feeds, and dynamic multi-currency conversions.</p>
            </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
            <div class="feature-glass-box">
                <div class="feature-icon">📈</div>
                <h4>AI Forecasting</h4>
                <p style="font-size:16px; color:#6B5D46;">Ensemble regression modeling combining Random Forest and Gradient Boosting architectures.</p>
            </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
            <div class="feature-glass-box">
                <div class="feature-icon">💎</div>
                <h4>Intelligent Insights</h4>
                <p style="font-size:16px; color:#6B5D46;">Automated signal generation, risk-reward grading, portfolio allocation, and automated reporting.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center; font-family:Playfair Display,serif; margin-bottom:15px;'>Today’s Market Pulse</h3>", unsafe_allow_html=True)
    gp = compute_market_pulse("Gold", featured_data, performance)
    sp = compute_market_pulse("Silver", featured_data, performance)

    p1, p2 = st.columns(2)
    with p1:
        st.markdown(f"""
            <div class="metric-card">
                <h4>Gold Pulse</h4>
                <h2>{gp['label']}</h2>
                <p style="font-size:13px; color:#8C7A54; margin-top:5px;">Composite Score: <b>{gp['score']} / 100</b></p>
            </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
            <div class="metric-card">
                <h4>Silver Pulse</h4>
                <h2>{sp['label']}</h2>
                <p style="font-size:13px; color:#8C7A54; margin-top:5px;">Composite Score: <b>{sp['score']} / 100</b></p>
            </div>
        """, unsafe_allow_html=True)

elif st.session_state.active_nav == "Forecast":
    st.subheader("Market Forecast Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metal = st.selectbox("Metal", ["Gold", "Silver"], key="forecast_metal")
    with col2:
        currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=0, key="forecast_currency")
    with col3:
        price_unit = st.radio("Price Unit", ["Troy Ounce", "Gram"], key="forecast_unit", horizontal=True)
    with col4:
        forecast_days = st.number_input("Forecast Days", min_value=7, max_value=90, value=DEFAULT_FORECAST_DAYS, step=1, key="forecast_days")

    stats = dashboard_metrics(metal, currency, models, featured_data, performance, forecast_days)
    base_forecast = stats["Forecast"].copy()
    
    # 🎛️ Scenario Simulator
    st.markdown("<hr style='border-color:rgba(184,137,46,0.15); margin: 25px 0;'>", unsafe_allow_html=True)
    st.markdown("### 🎛️ Scenario Simulator")
    st.caption("Adjust macroeconomic sliders to view mathematical real-time curve bending on the forecast model.")
    sim1, sim2, sim3 = st.columns(3)
    with sim1:
        usd_strength = st.slider("USD Strength Impact", min_value=-10.0, max_value=10.0, value=0.0, step=0.5, format="%+.1f%%")
    with sim2:
        inflation_shock = st.slider("Inflation Shock", min_value=0.0, max_value=15.0, value=0.0, step=0.5, format="+%.1f%%")
    with sim3:
        volatility_multi = st.slider("Market Volatility", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

    # Math application for Scenario adjustments
    scenario_adjustment = (inflation_shock / 100) - (usd_strength / 100)
    adjusted_forecast = base_forecast.copy()
    adjusted_forecast["Forecast"] = adjusted_forecast["Forecast"] * (1 + scenario_adjustment)
    noise = np.random.normal(0, 0.005 * volatility_multi, len(adjusted_forecast))
    adjusted_forecast["Forecast"] = adjusted_forecast["Forecast"] * (1 + noise)

    forecast_display = adjusted_forecast.copy()
    forecast_display["Forecast"] = forecast_display["Forecast"].apply(lambda x: round(convert_price(x, currency), 2))

    change_24h = compute_24h_change(featured_data, metal)
    arrow, trend_class = trend_arrow(change_24h)
    dir_acc = performance[metal]["Directional Accuracy %"]
    conf_label, conf_text_color, conf_bar_color = confidence_tier(stats["Confidence"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card"><h4>Current Price</h4>
            <h2>{currency} {round(stats['Current Price'], 2)}</h2>
            <div class="trend-pill {trend_class}">{arrow} {change_24h:+.2f}% (24H)</div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        f_arrow, f_class = trend_arrow(stats["Expected Return"])
        st.markdown(
            f"""<div class="metric-card"><h4>Forecast Price ({forecast_days}D)</h4>
            <h2>{currency} {round(stats['Forecast Price'], 2)}</h2>
            <div class="trend-pill {f_class}">{f_arrow} {stats['Expected Return']:+.2f}% expected</div></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card"><h4>Confidence</h4>
            <h2>{stats['Confidence']}%</h2>
            <div style="margin-top:8px; font-size:12px; color:{conf_text_color}; font-weight:700;">{conf_label} Confidence</div></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card"><h4>Recommendation</h4>
            <div class="signal-badge {signal_css_class(stats['Signal'])}">{stats['Signal']}</div></div>""",
            unsafe_allow_html=True,
        )

    # 🤖 AI Typing Effect
    with st.expander("Intelligence Reasoning", expanded=True):
        reasons = signal_reasoning(metal, stats, featured_data, performance, forecast_days)
        st.write_stream(stream_reasoning(reasons))

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(create_forecast_chart(metal, adjusted_forecast, featured_data, currency), use_container_width=True)
    
    st.dataframe(forecast_display, use_container_width=True)

elif st.session_state.active_nav == "Compare":
    st.subheader("Market Split-Screen Compare Mode")
    
    gold_data = featured_data["Gold"]["Close"]
    silver_data = featured_data["Silver"]["Close"]
    
    # Compute Gold to Silver Ratio for side-by-side metric overlay
    common_index = gold_data.index.intersection(silver_data.index)
    ratio = gold_data[common_index] / silver_data[common_index]
    
    cm1, cm2 = st.columns(2)
    with cm1:
        st.markdown("<h3 style='text-align:center; color:#B8892E;'>Gold</h3>", unsafe_allow_html=True)
        g_fig = create_mini_chart(gold_data.tail(90), "#B8892E")
        g_fig.update_layout(height=250)
        st.plotly_chart(g_fig, use_container_width=True)
        
    with cm2:
        st.markdown("<h3 style='text-align:center; color:#6B7280;'>Silver</h3>", unsafe_allow_html=True)
        s_fig = create_mini_chart(silver_data.tail(90), "#6B7280")
        s_fig.update_layout(height=250)
        st.plotly_chart(s_fig, use_container_width=True)

    st.markdown("<hr style='border-color:rgba(184,137,46,0.15);'>", unsafe_allow_html=True)
    st.markdown("### Historic Gold / Silver Ratio")
    st.caption("A rising line means Gold is outperforming Silver. A falling line indicates Silver is outperforming Gold.")
    
    ratio_fig = go.Figure()
    ratio_fig.add_trace(go.Scatter(
        x=ratio.index, y=ratio.values, mode="lines",
        line=dict(color="#8C6A2E", width=2),
        fill="tozeroy", fillcolor="rgba(184,137,46,0.15)",
        name="Ratio"
    ))
    ratio_fig.update_layout(
        height=350,
        margin=dict(l=30, r=30, t=10, b=10),
        template="plotly_white",
        paper_bgcolor="#FFFDF7",
        plot_bgcolor="#FFFDF7",
        showlegend=False
    )
    st.plotly_chart(ratio_fig, use_container_width=True)


elif st.session_state.active_nav == "Advisor":
    st.subheader("Investment Advisor & Portfolio Tools")
    a1, a2 = st.columns(2)
    with a1:
        inv_amount = st.number_input("Investment Amount", value=100000, key="adv_amt")
    with a2:
        inv_curr = st.selectbox("Currency", list(CURRENCIES.keys()), index=1, key="adv_curr")
    if st.button("Generate Recommendation", type="primary"):
        gold_m = dashboard_metrics("Gold", inv_curr, models, featured_data, performance)
        silver_m = dashboard_metrics("Silver", inv_curr, models, featured_data, performance)
        best = "Gold" if gold_m["Expected Return"] >= silver_m["Expected Return"] else "Silver"
        st.success(f"Top Allocation Asset: **{best}** based on projected returns.")

elif st.session_state.active_nav == "Analytics":
    st.subheader("Model Performance Analytics")
    perf_display = performance_df.reset_index().rename(columns={"index": "Metal"})
    st.dataframe(perf_display, use_container_width=True, hide_index=True)

elif st.session_state.active_nav == "About":
    st.markdown(f"""
# {PROJECT_NAME}

## Overview
This platform forecasts next-day returns for gold and silver futures using an ensemble of Random Forest and Gradient Boosting regressors.

### Built Using
Python, Streamlit, Plotly, Scikit-learn, Pandas, NumPy, ReportLab
    """)
