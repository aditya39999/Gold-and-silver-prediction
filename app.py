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

import feedparser
from textblob import TextBlob

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
VERSION = "1.1.0"

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
    initial_sidebar_state="expanded",
)

# ==========================================================
# STYLING (Cream and Gold Luxury Theme)
# ==========================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&display=swap');

.block-container { padding-top: 1rem; max-width: 1400px; }
div[data-testid="stAppViewBlockContainer"] { padding-top: 1rem; }
div[data-testid="stDecoration"] { display: none; }

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

/* Landing Page Hero & Cards Styles */
.landing-hero {
    text-align: center;
    padding: 3rem 1rem 2rem 1rem;
    position: relative;
}
.landing-hero h1 {
    font-size: 3.5rem;
    color: #5C4A32;
    margin-bottom: 0.5rem;
    letter-spacing: 0.04em;
    animation: hero-fade 1.2s ease-out;
}
.landing-subtitle {
    font-size: 1.4rem;
    color: #8C6A2E;
    font-style: italic;
    margin-bottom: 2rem;
    animation: hero-fade 1.6s ease-out;
}
@keyframes hero-fade {
    from { opacity: 0; transform: translateY(15px); }
    to { opacity: 1; transform: translateY(0); }
}

.market-card-landing {
    background: #FFFDF7;
    border: 1px solid rgba(184,137,46,0.35);
    border-top: 5px solid #B8892E;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(46,39,31,0.06);
    transition: transform 0.3s ease;
}
.market-card-landing:hover {
    transform: translateY(-4px);
}
.market-card-landing h3 {
    font-family: 'Playfair Display', Georgia, serif;
    color: #5C4A32;
    margin-bottom: 10px;
}

.feature-box {
    background: #FFFDF7;
    border: 1px solid rgba(184,137,46,0.2);
    border-radius: 6px;
    padding: 20px;
    text-align: center;
    height: 100%;
}
.feature-box h4 {
    font-family: 'Playfair Display', Georgia, serif;
    color: #8C6A2E;
    margin-bottom: 8px;
}

.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px 24px;
    padding: 14px 22px;
    border-radius: 4px;
    margin-bottom: 14px;
    background: linear-gradient(120deg, #EFE4CD, #F7F1E4 60%, #EFE4CD);
    border: 1px solid rgba(184,137,46,0.35);
}
.app-header .title-block { text-align: left; }
.app-header h1 {
    margin: 0;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #8C6A2E;
    line-height: 1.2;
}
.app-header p {
    color: #6B5D46;
    margin: 2px 0 0 0;
    font-size: 13px;
    font-style: italic;
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

.signal-badge {
    display: inline-block;
    margin-top: 4px;
    padding: 9px 20px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: 700;
    font-family: 'Playfair Display', Georgia, serif;
}
.signal-badge.signal-strong-buy { background: #33502F; color: #F7F1E4; }
.signal-badge.signal-buy { background: #DCE8D8; color: #2E4A2A; border: 1px solid rgba(76,107,72,0.4); }
.signal-badge.signal-hold { background: #F3E4C2; color: #7A5B1E; border: 1px solid rgba(184,137,46,0.4); }
.signal-badge.signal-sell { background: #8C3A2C; color: #FBEDE9; }

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


# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.markdown(f"### **{PROJECT_NAME}**")
nav_selection = st.sidebar.radio(
    "Navigation",
    ["Home", "Forecast", "Advisor", "Analytics", "About"],
    index=["Home", "Forecast", "Advisor", "Analytics", "About"].index(st.session_state.active_nav)
)
st.session_state.active_nav = nav_selection

# Load models on startup
with st.spinner("Initializing market data & models..."):
    market_data = load_market_data()
    featured_data = build_featured_data(market_data)
    models, performance, performance_df, feature_importance, prediction_history = train_all_models()

# ==========================================================
# PAGE ROUTING
# ==========================================================

if st.session_state.active_nav == "Home":
    # Hero Heading
    st.markdown("""
        <div class="landing-hero">
            <h1>Intelligence Behind Precious Metals</h1>
            <div class="landing-subtitle">Advanced Machine Learning &amp; Market Forecasting Platform</div>
        </div>
    """, unsafe_allow_html=True)

    # Two Large Interactive Gold and Silver Cards
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
            <div class="market-card-landing">
                <h3>Gold Futures (GC=F)</h3>
                <h2 style="font-family:'Playfair Display',serif; color:#2E271F; margin:10px 0;">USD {gold_close:,.2f}</h2>
                <div class="trend-pill {g_cls}">{g_arr} {gold_chg:+.2f}% (24H)</div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_mini_chart(gold_series, "#B8892E"), use_container_width=True, config={"displayModeBar": False})

    with col_s:
        st.markdown(f"""
            <div class="market-card-landing">
                <h3>Silver Futures (SI=F)</h3>
                <h2 style="font-family:'Playfair Display',serif; color:#2E271F; margin:10px 0;">USD {silver_close:,.2f}</h2>
                <div class="trend-pill {s_cls}">{s_arr} {silver_chg:+.2f}% (24H)</div>
            </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(create_mini_chart(silver_series, "#6B7280"), use_container_width=True, config={"displayModeBar": False})

    # Central Call-to-Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    c_btn1, c_btn2, c_btn3 = st.columns([1, 2, 1])
    with c_btn2:
        if st.button("Enter Market Intelligence →", type="primary", use_container_width=True):
            st.session_state.active_nav = "Forecast"
            st.rerun()

    st.markdown("<br><hr style='border-color:rgba(184,137,46,0.2);'><br>", unsafe_allow_html=True)

    # Three Features Section
    st.markdown("<h3 style='text-align:center; font-family:Playfair Display,serif; margin-bottom:25px;'>Platform Architecture</h3>", unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""
            <div class="feature-box">
                <h4>Live Markets</h4>
                <p style="font-size:16px; color:#6B5D46;">Real-time intraday price tracking, sparkline feeds, and dynamic multi-currency conversions.</p>
            </div>
        """, unsafe_allow_html=True)
    with f2:
        st.markdown("""
            <div class="feature-box">
                <h4>AI Forecasting</h4>
                <p style="font-size:16px; color:#6B5D46;">Ensemble regression modeling combining Random Forest and Gradient Boosting architectures.</p>
            </div>
        """, unsafe_allow_html=True)
    with f3:
        st.markdown("""
            <div class="feature-box">
                <h4>Intelligent Insights</h4>
                <p style="font-size:16px; color:#6B5D46;">Automated signal generation, risk-reward grading, portfolio allocation, and automated reporting.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Today's Market Pulse Section
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

    from app import dashboard_metrics, signal_css_class, confidence_tier, signal_reasoning, create_forecast_chart
    stats = dashboard_metrics(metal, currency, models, featured_data, performance, forecast_days)
    forecast = stats["Forecast"].copy()
    forecast_display = forecast.copy()
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.plotly_chart(create_forecast_chart(metal, forecast, featured_data, currency), use_container_width=True)
    st.dataframe(forecast_display, use_container_width=True)

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
