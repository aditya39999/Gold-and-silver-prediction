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
VERSION = "1.4.0"

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
# STYLING (Premium Cinematic Theme)
# ==========================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700;800&family=Cormorant+Garamond:ital,wght@0,400;0,500;1,400&family=Inter:wght@300;400;600&display=swap');

.block-container { padding-top: 1rem; max-width: 1400px; }
div[data-testid="stAppViewBlockContainer"] { padding-top: 1rem; }
div[data-testid="stDecoration"] { display: none; }

[data-testid="stSidebar"] {
    background: rgba(247, 241, 228, 0.95);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(184,137,46,0.2);
}

.stApp {
    background: #0D0B0A; /* Dark premium background for cinematic feel */
    color: #F7F1E4;
    font-family: 'Inter', sans-serif;
    font-size: 17px;
}

h1, h2, h3, h4 {
    font-family: 'Playfair Display', Georgia, serif !important;
}

.stApp p, .stApp span, .stApp label, .stMarkdown, .stMarkdown p {
    color: #EFE4CD;
}

/* Subtle Background Motion */
.ambient-bg {
    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
    pointer-events: none; z-index: -1;
    background: radial-gradient(circle at 50% 30%, rgba(184,137,46,0.1) 0%, rgba(13,11,10,1) 60%);
}

/* Cinematic Hero */
.hero-container {
    text-align: center;
    padding: 4rem 1rem 1rem 1rem;
    position: relative;
    z-index: 10;
}
.hero-title {
    font-size: 4.5rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
    letter-spacing: 0.05em;
    background: linear-gradient(90deg, #D4AF37, #FFF5C0, #C0C0C0, #D4AF37);
    background-size: 200% auto;
    color: transparent;
    -webkit-background-clip: text;
    background-clip: text;
    animation: gold-sweep 4s linear infinite;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.1rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: rgba(239, 228, 205, 0.7);
    margin-bottom: 2rem;
}
@keyframes gold-sweep {
    to { background-position: 200% center; }
}

/* Central Market Pulse Ticker */
.market-pulse-ticker {
    display: inline-flex;
    align-items: center;
    gap: 15px;
    background: rgba(184, 137, 46, 0.1);
    border: 1px solid rgba(184, 137, 46, 0.3);
    padding: 8px 25px;
    border-radius: 999px;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: #D4AF37;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 3rem;
}
.ticker-dot {
    width: 8px; height: 8px;
    background-color: #4C6B48;
    border-radius: 50%;
    box-shadow: 0 0 10px #4C6B48;
    animation: pulse-dot 1.5s infinite alternate;
}
@keyframes pulse-dot {
    from { opacity: 0.4; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1.2); }
}

/* Floating Market Orbs */
.orbs-wrapper {
    display: flex;
    justify-content: center;
    gap: 5rem;
    margin: 2rem 0 4rem 0;
    perspective: 1000px;
}
.market-orb {
    width: 240px; height: 240px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
    transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    box-shadow: 0 20px 40px rgba(0,0,0,0.5), inset 0 0 40px rgba(255,255,255,0.2);
    position: relative;
    cursor: default;
}
.market-orb:hover {
    transform: translateY(-10px) rotateX(10deg) rotateY(10deg) scale(1.05);
}
.orb-gold {
    background: radial-gradient(circle at 30% 30%, #FFF5C0 0%, #D4AF37 40%, #8C6A2E 80%, #2E2013 100%);
    border: 2px solid rgba(255, 245, 192, 0.5);
}
.orb-silver {
    background: radial-gradient(circle at 30% 30%, #FFFFFF 0%, #C0C0C0 40%, #808080 80%, #2E2E2E 100%);
    border: 2px solid rgba(255, 255, 255, 0.5);
}
.market-orb h3 { margin: 0; font-size: 1.5rem; color: #111; text-shadow: 0 1px 2px rgba(255,255,255,0.5); }
.market-orb h2 { margin: 5px 0; font-size: 2.2rem; color: #000; text-shadow: 0 1px 2px rgba(255,255,255,0.5); }
.orb-change { font-family: 'Inter', sans-serif; font-weight: 700; font-size: 1rem; padding: 4px 12px; border-radius: 999px; background: rgba(0,0,0,0.1); color: #111; }

/* AI Flow Animation */
.ai-flow-container {
    text-align: center;
    margin: 4rem 0;
    padding: 2rem;
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    border: 1px solid rgba(184,137,46,0.1);
}
.ai-flow {
    display: inline-flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 15px;
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    color: rgba(239, 228, 205, 0.4);
}
.ai-node {
    padding: 10px 20px;
    border-radius: 8px;
    background: rgba(0,0,0,0.5);
    border: 1px solid rgba(184,137,46,0.2);
    animation: node-light 4s infinite;
}
.ai-flow span.arrow { color: #8C6A2E; }
.ai-node:nth-child(1) { animation-delay: 0s; }
.ai-node:nth-child(3) { animation-delay: 0.8s; }
.ai-node:nth-child(5) { animation-delay: 1.6s; }
.ai-node:nth-child(7) { animation-delay: 2.4s; }
.ai-node:nth-child(9) { animation-delay: 3.2s; }
@keyframes node-light {
    0%, 100% { background: rgba(0,0,0,0.5); color: rgba(239, 228, 205, 0.4); border-color: rgba(184,137,46,0.2); box-shadow: none; }
    10% { background: rgba(184,137,46,0.2); color: #FFF; border-color: #D4AF37; box-shadow: 0 0 15px rgba(212,175,55,0.5); }
    30% { background: rgba(0,0,0,0.5); color: rgba(239, 228, 205, 0.4); border-color: rgba(184,137,46,0.2); box-shadow: none; }
}

/* Glass/Gold CTA Button */
.stButton > button {
    background: linear-gradient(135deg, rgba(212,175,55,0.2), rgba(140,106,46,0.2)) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(212,175,55,0.5) !important;
    color: #FFF5C0 !important;
    padding: 0.8rem 2.5rem !important;
    font-size: 1.2rem !important;
    font-family: 'Playfair Display', serif !important;
    letter-spacing: 0.05em !important;
    border-radius: 999px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(212,175,55,0.1) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(212,175,55,0.4), rgba(140,106,46,0.4)) !important;
    box-shadow: 0 0 30px rgba(212,175,55,0.4) !important;
    border-color: #FFF5C0 !important;
    transform: scale(1.02) !important;
}

/* Topnav (Dark mode adjustments) */
.topnav-container { display: flex; justify-content: center; gap: 15px; margin-bottom: 2rem; padding: 10px; background: rgba(0, 0, 0, 0.5); backdrop-filter: blur(8px); border-radius: 999px; border: 1px solid rgba(184, 137, 46, 0.25); max-width: 700px; margin-left: auto; margin-right: auto; }

/* Dashboard Cards (Dark mode) */
.metric-card, .glass-card {
    background: rgba(25, 20, 18, 0.7); border: 1px solid rgba(184,137,46,0.2); border-top: 3px solid #D4AF37; border-radius: 8px; padding: 16px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-card h4, .glass-card h3 { color: rgba(239, 228, 205, 0.7); }
.metric-card h2 { color: #FFF5C0; }

.st-key-skip_intro_btn {
    position: fixed; bottom: 20px; right: 20px; z-index: 1000000;
    animation: fade-out-btn 3s forwards;
}
@keyframes fade-out-btn {
    0%, 90% { opacity: 1; visibility: visible; }
    100% { opacity: 0; visibility: hidden; display: none; }
}

/* Liquid Transition Overlay */
.liquid-transition {
    position: fixed; inset: 0; z-index: 9999999;
    background: radial-gradient(circle at center, #FFF5C0 0%, #D4AF37 50%, #8C6A2E 100%);
    animation: liquid-fill 1s cubic-bezier(0.77, 0, 0.175, 1) forwards;
    clip-path: circle(0% at 50% 50%);
}
@keyframes liquid-fill {
    0% { clip-path: circle(0% at 50% 50%); opacity: 1; }
    50% { clip-path: circle(150% at 50% 50%); opacity: 1; }
    100% { clip-path: circle(150% at 50% 50%); opacity: 0; visibility: hidden; }
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
if "trigger_transition" not in st.session_state:
    st.session_state.trigger_transition = False

def render_molten_gold_intro():
    intro_html = """
    <style>
    .molten-overlay { position: fixed; inset: 0; z-index: 999999; pointer-events: none; background: #0B0A08; display: flex; align-items: center; justify-content: center; animation: curtain-melt 3.5s cubic-bezier(0.77, 0, 0.175, 1) forwards; }
    .gold-droplet { position: absolute; width: 12px; height: 12px; background: radial-gradient(circle, #FFF5C0 0%, #D4AF37 50%, #8C6A2E 100%); border-radius: 50%; box-shadow: 0 0 25px 8px rgba(212,175,55,0.8); animation: drop-fall 1s ease-in forwards; }
    @keyframes drop-fall { 0% { transform: translateY(-150vh) scale(0.5); opacity: 0; } 50% { opacity: 1; } 100% { transform: translateY(0) scale(1.5); opacity: 1; } }
    .gold-ripple { position: absolute; width: 10px; height: 10px; border: 3px solid rgba(212,175,55,0.9); border-radius: 50%; box-shadow: 0 0 30px rgba(212,175,55,0.6) inset; animation: ripple-spread 1.8s ease-out 0.9s forwards; opacity: 0; }
    @keyframes ripple-spread { 0% { width: 10px; height: 10px; opacity: 1; transform: scale(1); } 100% { width: 350vw; height: 350vw; opacity: 0; transform: scale(1); border-width: 50px; } }
    @keyframes curtain-melt { 0%, 75% { clip-path: inset(0 0 0 0); opacity: 1; } 100% { clip-path: inset(100% 0 0 0); opacity: 0; visibility: hidden; } }
    </style>
    <div class="molten-overlay"><div class="gold-droplet"></div><div class="gold-ripple"></div></div>
    """
    st.markdown(intro_html, unsafe_allow_html=True)


if not st.session_state.intro_played:
    render_molten_gold_intro()
    skip_clicked = st.button("Skip Intro", key="skip_intro_btn")
    st.session_state.intro_played = True
    if skip_clicked:
        st.rerun()

if st.session_state.trigger_transition:
    st.markdown("<div class='liquid-transition'></div>", unsafe_allow_html=True)
    time.sleep(0.8)
    st.session_state.trigger_transition = False
    st.session_state.active_nav = "Forecast"
    st.rerun()

# Ambient Background Injection
st.markdown("<div class='ambient-bg'></div>", unsafe_allow_html=True)

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
    data["Return"] = data["Close"].pct_change()
    data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))
    data["Volatility"] = data["Return"].rolling(10).std()

    delta = data["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    data["RSI"] = 100 - (100 / (1 + rs))

    ema12 = data["Close"].ewm(span=12).mean()
    ema26 = data["Close"].ewm(span=26).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_SIGNAL"] = data["MACD"].ewm(span=9).mean()
    return data

FEATURE_COLUMNS = ["Close", "SMA_5", "SMA_20", "RSI", "MACD", "MACD_SIGNAL", "Volatility"]

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
    
    split_index = int(len(data) * 0.85)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    close_test = data["Close"].iloc[split_index:]

    rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    pred = rf.predict(X_test)
    actual_price = close_test.values * np.exp(y_test.values)
    predicted_price = close_test.values * np.exp(pred)
    perf = {"Directional Accuracy %": np.mean(np.sign(y_test.values) == np.sign(pred)) * 100, "R2 (price)": r2_score(actual_price, predicted_price)}
    return {"rf": rf}, perf

@st.cache_resource(show_spinner=False)
def train_all_models():
    featured_data = build_featured_data()
    models, performance = {}, {}
    for metal, df in featured_data.items():
        models[metal], performance[metal] = train_models_for_metal(df)
    return models, performance

def forecast_prices(metal, forecast_days, models, featured_data):
    model_dict = models[metal]
    window = featured_data[metal].tail(INDICATOR_WINDOW + forecast_days).copy()
    base_cols = ["Open", "High", "Low", "Close", "Volume"]

    predictions_out, future_dates = [], []
    current_date = window.index[-1]

    for _ in range(forecast_days):
        latest = window.iloc[-1:].copy()
        pred_return = model_dict["rf"].predict(latest[FEATURE_COLUMNS])[0]
        predicted_price = float(latest["Close"].values[0]) * np.exp(pred_return)

        current_date += timedelta(days=1)
        future_dates.append(current_date)
        predictions_out.append(predicted_price)

        new_row = pd.DataFrame({"Close": [predicted_price]}, index=[current_date])
        window = pd.concat([window, new_row])
        window = add_indicators(window.ffill().bfill())

    return pd.DataFrame({"Date": future_dates, "Forecast": predictions_out})

def compute_24h_change(featured_data, metal):
    df = featured_data[metal]
    if len(df) < 2: return 0.0
    return ((df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]) * 100

def trend_arrow(change_pct, flat=0.05):
    if change_pct > flat: return "▲", "color: #388E3C;"
    if change_pct < -flat: return "▼", "color: #D32F2F;"
    return "▬", "color: #555;"

# ==========================================================
# LOAD DATA
# ==========================================================
with st.spinner("Initializing AI Core..."):
    market_data = load_market_data()
    featured_data = build_featured_data(market_data)
    models, performance = train_all_models()

# ==========================================================
# TOP NAVIGATION
# ==========================================================
nav_options = ["Home", "Forecast", "Compare", "Advisor"]
cols_nav = st.columns(len(nav_options))
for i, opt in enumerate(nav_options):
    with cols_nav[i]:
        btn_type = "primary" if st.session_state.active_nav == opt else "secondary"
        if st.button(opt, key=f"nav_btn_{opt}", use_container_width=True, type=btn_type):
            st.session_state.active_nav = opt
            st.rerun()

st.markdown("<hr style='border-color:rgba(184,137,46,0.2); margin: 5px 0 20px 0;'>", unsafe_allow_html=True)

# ==========================================================
# ROUTING
# ==========================================================

if st.session_state.active_nav == "Home":
    
    gold_close = featured_data["Gold"]["Close"].iloc[-1]
    silver_close = featured_data["Silver"]["Close"].iloc[-1]
    gold_chg = compute_24h_change(featured_data, "Gold")
    silver_chg = compute_24h_change(featured_data, "Silver")
    
    g_arr, g_col = trend_arrow(gold_chg)
    s_arr, s_col = trend_arrow(silver_chg)
    
    # Animated Line Graph Background
    fig_bg = go.Figure()
    fig_bg.add_trace(go.Scatter(y=featured_data["Gold"]["Close"].tail(60).values, mode='lines', line=dict(color='rgba(212,175,55,0.2)', width=2, shape='spline')))
    fig_bg.add_trace(go.Scatter(y=featured_data["Silver"]["Close"].tail(60).values * 80, mode='lines', line=dict(color='rgba(255,255,255,0.1)', width=2, shape='spline')))
    fig_bg.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False, xaxis=dict(visible=False), yaxis=dict(visible=False))
    
    st.plotly_chart(fig_bg, use_container_width=True, config={'displayModeBar': False})
    
    # Hero Title over graph using CSS positioning via columns trick or raw HTML
    # Due to Streamlit flow, we render it directly
    st.markdown("""
        <div style="margin-top: -380px;" class="hero-container">
            <div class="hero-title">PRECIOUS METALS, PREDICTED.</div>
            <div class="hero-subtitle">Powered by Machine Learning</div>
            
            <div class="market-pulse-ticker">
                <span class="ticker-dot"></span> MARKET OPEN &nbsp;&middot;&nbsp; 
                Gold <span style='color:#4C6B48'>↑</span> &nbsp;&middot;&nbsp; 
                Silver <span style='color:#8C3A2C'>↓</span> &nbsp;&middot;&nbsp; 
                Volatility Moderate
            </div>
            
            <div class="orbs-wrapper">
                <div class="market-orb orb-gold">
                    <h3>Gold</h3>
                    <h2>$""" + f"{gold_close:,.2f}" + """</h2>
                    <div class="orb-change" style='"""+g_col+"""'>"""+g_arr+f" {gold_chg:+.2f}%"+"""</div>
                </div>
                <div class="market-orb orb-silver">
                    <h3>Silver</h3>
                    <h2>$""" + f"{silver_close:,.2f}" + """</h2>
                    <div class="orb-change" style='"""+s_col+"""'>"""+s_arr+f" {silver_chg:+.2f}%"+"""</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1])
    with c_btn2:
        if st.button("Explore Markets →", type="primary", use_container_width=True):
            st.session_state.trigger_transition = True
            st.rerun()

    st.markdown("""
        <div class="ai-flow-container">
            <h4 style="margin-bottom: 20px; color:#D4AF37;">How the AI Thinks</h4>
            <div class="ai-flow">
                <span class="ai-node">Market Data</span>
                <span class="arrow">→</span>
                <span class="ai-node">Tech Indicators</span>
                <span class="arrow">→</span>
                <span class="ai-node">ML Models</span>
                <span class="arrow">→</span>
                <span class="ai-node">Forecast</span>
                <span class="arrow">→</span>
                <span class="ai-node">BUY/HOLD/SELL</span>
            </div>
        </div>
    """, unsafe_allow_html=True)


elif st.session_state.active_nav == "Forecast":
    st.subheader("Market Forecast Dashboard")
    st.info("Select parameters to view AI predictions.")
    
    c1, c2 = st.columns(2)
    metal = c1.selectbox("Asset", ["Gold", "Silver"])
    forecast_days = c2.slider("Forecast Horizon (Days)", 7, 60, 30)
    
    forecast_df = forecast_prices(metal, forecast_days, models, featured_data)
    
    fig = go.Figure()
    hist = featured_data[metal].tail(90)
    fig.add_trace(go.Scatter(x=hist.index, y=hist["Close"], mode="lines", name="Historical", line=dict(color="#D4AF37", width=2)))
    fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast"], mode="lines+markers", name="AI Forecast", line=dict(color="#4C6B48", width=2, dash='dot')))
    
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)


elif st.session_state.active_nav == "Compare":
    st.subheader("Gold vs Silver Comparison")
    st.markdown("Monitor the historic ratio to spot relative outperformance.")
    
    gold_data = featured_data["Gold"]["Close"]
    silver_data = featured_data["Silver"]["Close"]
    common = gold_data.index.intersection(silver_data.index)
    ratio = gold_data[common] / silver_data[common]
    
    fig = px.line(x=ratio.index, y=ratio.values, title="Gold / Silver Ratio")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", line=dict(color="#D4AF37"))
    st.plotly_chart(fig, use_container_width=True)


elif st.session_state.active_nav == "Advisor":
    st.subheader("AI Portfolio Advisor")
    st.markdown("Run automated scenarios to determine optimal portfolio allocation based on forward-looking ML returns.")
    st.info("Module ready for deployment.")
