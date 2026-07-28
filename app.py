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
VERSION = "1.0.0"

RANDOM_STATE = 42
DEFAULT_FORECAST_DAYS = 30
INDICATOR_WINDOW = 250          # rows needed to recompute rolling indicators
CV_SPLITS = 5                    # walk-forward validation folds

MARKET_SYMBOLS = {"Gold": "GC=F", "Silver": "SI=F"}

TROY_OUNCE = 31.1034768

# Static FX table (USD base). Labeled clearly as static in the UI so users
# are not misled into thinking this is a live feed.
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
# STYLING (cream and gold luxury theme, no emojis)
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

/* Widget labels (Metal, Currency, Investment Amount, etc.) */
.stSelectbox label, .stNumberInput label, .stRadio label, .stTextInput label,
[data-testid="stWidgetLabel"] p {
    font-size: 17px !important;
    font-weight: 600;
}

/* Text inside selects, number inputs, radio options */
.stSelectbox div[data-baseweb="select"] *, .stNumberInput input,
.stRadio label span, .stTextInput input {
    font-size: 17px !important;
}

/* Dataframes / tables */
[data-testid="stDataFrame"] {
    font-size: 16px;
}

/* Markdown body text used for summaries and advisor output */
.stMarkdown h3 { font-size: 22px; }
.stMarkdown ul, .stMarkdown li { font-size: 18px; }

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
    letter-spacing: 0.02em;
}

.metric-card {
    background: #FFFDF7;
    padding: 16px 14px 14px 14px;
    border-radius: 6px;
    border: 1px solid rgba(184,137,46,0.25);
    border-top: 4px solid #B8892E;
    text-align: center;
    box-shadow: 0 1px 3px rgba(46,39,31,0.06);
    animation: card-rise 0.45s ease both;
}
.metric-card.accent-neutral { border-top-color: #B8892E; }
.metric-card.accent-good { border-top-color: #4C6B48; }
.metric-card.accent-info { border-top-color: #3B6FA0; }
.metric-card h4 {
    color: #7A6B4E;
    font-weight: 700;
    font-size: 12px;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'Cormorant Garamond', Georgia, serif;
}
.metric-card h2 {
    margin: 0;
    font-size: 27px;
    font-weight: 700;
    color: #1F1811;
    font-family: 'Playfair Display', Georgia, serif;
}
.metric-card .metric-reason {
    margin-top: 8px;
    font-size: 12px;
    color: #6B5D46;
    line-height: 1.4;
}

@keyframes card-rise {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Pill badge used for 24H change / expected return / model accuracy */
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

/* Confidence progress bar, color-coded by tier, animated fill */
.confidence-bar-wrap {
    margin-top: 10px;
    height: 8px;
    border-radius: 999px;
    background: rgba(46,39,31,0.08);
    overflow: hidden;
}
.confidence-bar-fill {
    height: 100%;
    border-radius: 999px;
    width: 0%;
    animation: fill-bar 1.1s ease forwards;
    animation-delay: 0.1s;
}
@keyframes fill-bar {
    from { width: 0%; }
    to   { width: var(--target-width); }
}
.confidence-tier-label {
    display: inline-block;
    margin-top: 8px;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.accuracy-note {
    margin-top: 6px;
    font-size: 12px;
    color: #8C7A54;
}
.accuracy-note b { color: #4A3F2E; }

/* Recommendation shown as a solid colored badge instead of plain colored text,
   so BUY / HOLD / SELL are unmistakable at a glance */
.signal-badge {
    display: inline-block;
    margin-top: 4px;
    padding: 9px 20px;
    border-radius: 8px;
    font-size: 20px;
    font-weight: 700;
    font-family: 'Playfair Display', Georgia, serif;
    letter-spacing: 0.03em;
    animation: badge-pop 0.4s cubic-bezier(.34,1.56,.64,1) both;
}
.signal-badge.signal-strong-buy { background: #33502F; color: #F7F1E4; }
.signal-badge.signal-buy { background: #DCE8D8; color: #2E4A2A; border: 1px solid rgba(76,107,72,0.4); }
.signal-badge.signal-hold { background: #F3E4C2; color: #7A5B1E; border: 1px solid rgba(184,137,46,0.4); }
.signal-badge.signal-sell { background: #8C3A2C; color: #FBEDE9; }

@keyframes badge-pop {
    from { opacity: 0; transform: scale(0.85); }
    to   { opacity: 1; transform: scale(1); }
}

/* Sentiment pulse badges (Bullish / Neutral / Bearish) */
.pulse-badge {
    display: inline-block;
    padding: 8px 22px;
    border-radius: 999px;
    font-size: 19px;
    font-weight: 700;
    font-family: 'Playfair Display', Georgia, serif;
    letter-spacing: 0.04em;
    animation: badge-pop 0.4s cubic-bezier(.34,1.56,.64,1) both;
}
.pulse-badge.pulse-bullish { background: #33502F; color: #F7F1E4; }
.pulse-badge.pulse-neutral { background: #F3E4C2; color: #7A5B1E; border: 1px solid rgba(184,137,46,0.4); }
.pulse-badge.pulse-bearish { background: #8C3A2C; color: #FBEDE9; }

.pulse-score-wrap {
    margin-top: 12px;
    height: 10px;
    border-radius: 999px;
    background: rgba(46,39,31,0.08);
    position: relative;
    overflow: hidden;
}
.pulse-score-fill {
    height: 100%;
    border-radius: 999px;
    width: 0%;
    animation: fill-bar 1.1s ease forwards;
    animation-delay: 0.15s;
}
.pulse-reason-list {
    margin-top: 10px;
    padding-left: 18px;
    font-size: 15px;
    line-height: 1.55;
}
.pulse-reason-list li { margin-bottom: 3px; }

.live-banner {
    background: #FFFDF7;
    padding: 8px 16px;
    border-radius: 999px;
    border: 1px solid rgba(184,137,46,0.3);
    font-size: 14px;
    display: flex;
    gap: 18px;
    flex-wrap: wrap;
    align-items: center;
}
.live-banner .live-item { text-align: left; line-height: 1.25; white-space: nowrap; }
.live-banner .live-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: #8C7A54; }
.live-banner .live-value {
    font-weight: 700;
    display: inline-block;
    transition: color 0.3s ease;
}
.live-banner .live-value.flash-up { animation: flash-green 0.9s ease; }
.live-banner .live-value.flash-down { animation: flash-red 0.9s ease; }

@keyframes flash-green {
    0%   { background-color: rgba(76,107,72,0.35); }
    100% { background-color: rgba(76,107,72,0); }
}
@keyframes flash-red {
    0%   { background-color: rgba(166,73,58,0.35); }
    100% { background-color: rgba(166,73,58,0); }
}

.live-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #4C6B48;
    box-shadow: 0 0 0 rgba(76,107,72,0.6);
    animation: live-pulse 1.6s infinite;
    flex-shrink: 0;
}
@keyframes live-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(76,107,72,0.55); }
    70%  { box-shadow: 0 0 0 7px rgba(76,107,72,0); }
    100% { box-shadow: 0 0 0 0 rgba(76,107,72,0); }
}
.ticker-label {
    font-size: 12px;
    font-weight: 600;
    color: #6B5D46;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin: 8px 0 2px 0;
}
.ticker-label-left {
    display: flex;
    align-items: center;
    gap: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.ticker-label-right {
    display: flex;
    align-items: center;
    gap: 6px;
}
.ticker-change {
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 700;
}
.ticker-change.trend-up { background: rgba(76,107,72,0.14); color: #385B34; }
.ticker-change.trend-down { background: rgba(166,73,58,0.14); color: #8C3A2C; }
.ticker-change.trend-flat { background: rgba(140,122,84,0.14); color: #6B5D46; }
.ticker-window {
    font-size: 10px;
    font-weight: 700;
    color: #8C7A54;
    background: rgba(140,122,84,0.12);
    padding: 2px 8px;
    border-radius: 999px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.disclaimer {
    font-size: 12px;
    color: #8C7A54;
    margin-top: 12px;
    font-style: italic;
}

/* Buttons styled as gold pill buttons */
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
.stButton > button[kind="primary"]:hover {
    background: #8C6A2E;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid rgba(184,137,46,0.35);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Cormorant Garamond', Georgia, serif;
    font-size: 19px;
    letter-spacing: 0.04em;
    color: #6B5D46;
}
.stTabs [aria-selected="true"] {
    color: #8C6A2E !important;
    font-weight: 700;
}
.stTabs [data-baseweb="tab"] p {
    font-size: 19px;
}

/* Native st.metric cards, restyled to match the cream/gold theme */
[data-testid="stMetric"] {
    background: #FFFDF7;
    padding: 14px 16px;
    border-radius: 4px;
    border: 1px solid rgba(184,137,46,0.3);
    animation: card-rise 0.45s ease both;
}
[data-testid="stMetricLabel"] {
    font-family: 'Cormorant Garamond', Georgia, serif;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 13px !important;
    color: #8C7A54 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 26px !important;
    color: #2E271F !important;
}
[data-testid="stMetricDelta"] {
    font-size: 13px !important;
    font-weight: 600;
}

/* "Why this recommendation?" expander: force high-contrast cream/gold styling
   on the header so it never inherits a dark background with dark text. */
[data-testid="stExpander"] {
    background: #FFFDF7 !important;
    border: 1px solid rgba(184,137,46,0.3) !important;
    border-radius: 6px !important;
    margin-top: 6px !important;
    margin-bottom: 2px !important;
}
[data-testid="stExpander"] summary {
    background: #FFFDF7 !important;
    color: #2E271F !important;
    border-radius: 6px !important;
    padding: 8px 12px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #F3E4C2 !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary svg {
    color: #2E271F !important;
    fill: #2E271F !important;
    font-weight: 600 !important;
}
[data-testid="stExpanderDetails"] {
    background: #FFFDF7 !important;
    color: #2E271F !important;
}

/* Tight spacer used to pull the forecast chart closer to the expander above it */
div.element-container:has(> div > div.chart-spacer) {
    margin-top: -26px;
    margin-bottom: -26px;
    height: 0;
    overflow: hidden;
}

/* Chart reveal animation: sweeps in from the left when a chart is (re)drawn */
.chart-reveal {
    animation: chart-sweep 0.9s ease;
}
@keyframes chart-sweep {
    from { clip-path: inset(0 100% 0 0); }
    to   { clip-path: inset(0 0 0 0); }
}

/* Small circular info icon + hover tooltip, used to explain the Confidence score */
.info-tooltip {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: help;
    margin-left: 5px;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background: rgba(59,111,160,0.18);
    color: #3B6FA0;
    font-size: 11px;
    font-weight: 700;
    font-family: Georgia, serif;
    vertical-align: middle;
}
.info-tooltip .tooltip-text {
    visibility: hidden;
    opacity: 0;
    transition: opacity 0.15s ease;
    position: absolute;
    z-index: 20;
    bottom: 135%;
    left: 50%;
    transform: translateX(-50%);
    width: 250px;
    background: #2E271F;
    color: #F7F1E4;
    text-align: left;
    padding: 10px 12px;
    border-radius: 6px;
    font-size: 12.5px;
    font-weight: 400;
    font-family: 'Cormorant Garamond', Georgia, serif;
    line-height: 1.45;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    white-space: normal;
    pointer-events: none;
}
.info-tooltip .tooltip-text::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: #2E271F transparent transparent transparent;
}
.info-tooltip:hover .tooltip-text {
    visibility: visible;
    opacity: 1;
}

/* Scenario simulator slider readout */
.scenario-readout {
    background: #FFFDF7;
    border: 1px solid rgba(184,137,46,0.3);
    border-radius: 8px;
    padding: 14px 18px;
    margin-top: 10px;
    animation: card-rise 0.4s ease both;
}
.scenario-readout .scenario-value {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 24px;
    font-weight: 700;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================
# CINEMATIC INTRO (Molten Gold Reveal Concept)[cite: 1]
# ==========================================================

if "intro_played" not in st.session_state:
    st.session_state.intro_played = False


def render_molten_gold_intro():
    """
    Cinematic Molten Gold Reveal: A tiny floating gold droplet drops, creating
    a liquid-gold ripple across the screen with shimmering metallic reflections,
    then melting downward like a curtain to reveal the dashboard underneath.
    """
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
# DATA DOWNLOAD & INDICATORS
# ==========================================================

def download_with_retry(symbol, start, end, retries=3, delay=2):
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.download(symbol, start=start, end=end,
                              auto_adjust=True, progress=False)
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


# ==========================================================
# MODEL TRAINING & FORECASTING
# ==========================================================

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

    rf = RandomForestRegressor(
        n_estimators=400, max_depth=10, min_samples_leaf=3,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    gbm = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=RANDOM_STATE
    )

    rf.fit(X_train, y_train)
    gbm.fit(X_train, y_train)

    pred_return_rf = rf.predict(X_test)
    pred_return_gbm = gbm.predict(X_test)
    pred_return_blend = 0.5 * pred_return_rf + 0.5 * pred_return_gbm

    actual_price_next = close_test.values * np.exp(y_test.values)
    predicted_price_next = close_test.values * np.exp(pred_return_blend)

    prediction_history = pd.DataFrame({
        "Date": dates_test,
        "Actual": actual_price_next,
        "Predicted": predicted_price_next,
    })

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    cv_dir_acc = []
    for tr_idx, val_idx in tscv.split(X):
        rf_cv = RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        )
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
        imp = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": models[metal]["rf"].feature_importances_,
        }).sort_values("Importance", ascending=False)
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

    predictions_out = []
    future_dates = []
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

        new_row = pd.DataFrame(
            {
                "Open": [float(predicted_price)],
                "High": [float(predicted_price) * 1.002],
                "Low": [float(predicted_price) * 0.998],
                "Close": [float(predicted_price)],
                "Volume": [float(window["Volume"].tail(10).mean())],
            },
            index=[current_date],
        )

        window = pd.concat([window[base_cols], new_row[base_cols]])
        window[base_cols] = window[base_cols].astype("float64")
        window = add_indicators(window)
        window = window.ffill().bfill()

    return pd.DataFrame({"Date": future_dates, "Forecast": predictions_out})


# ==========================================================
# CHARTS & METRICS
# ==========================================================

CHART_TEMPLATE = dict(
    template="plotly_white",
    paper_bgcolor="#FFFDF7",
    plot_bgcolor="#FFFDF7",
    font=dict(color="#2E2013", size=13, family="Georgia, 'Playfair Display', serif"),
)

AXIS_TICK_FONT = dict(color="#2E2013", size=12, family="Georgia, serif")
AXIS_TITLE_FONT = dict(color="#2E2013", size=13, family="Georgia, serif")


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
        ],
        **CHART_TEMPLATE,
    )
    fig.update_xaxes(
        showgrid=False,
        tickfont=AXIS_TICK_FONT,
        linecolor="#8C7A54",
        rangeslider=dict(visible=True, thickness=0.05, bgcolor="#EFE4CD"),
        range=[history.index[-default_window_days], forecast["Date"].iloc[-1]],
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#EFE4CD",
        title=dict(text="Price", font=AXIS_TITLE_FONT),
        tickfont=AXIS_TICK_FONT,
        range=default_y_range,
    )
    return fig


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


# ==========================================================
# APP LAYOUT & TABS
# ==========================================================

header_col1, header_col2 = st.columns([3, 1], vertical_alignment="center")
with header_col1:
    st.markdown(
        f"""
        <div class="app-header">
            <div class="title-block">
                <h1>{PROJECT_NAME}</h1>
                <p>An AI-assisted forecasting platform for gold and silver markets</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_col2:
    live_currency = st.selectbox(
        "Live Market Currency", list(CURRENCIES.keys()), key="live_currency", label_visibility="collapsed"
    )

with st.spinner("Loading market data and training models..."):
    market_data = load_market_data()
    featured_data = build_featured_data(market_data)
    models, performance, performance_df, feature_importance, prediction_history = train_all_models()

(
    tab_forecast, tab_advisor, tab_analytics, tab_about,
) = st.tabs(["Forecast", "Advisor", "Analytics", "About"])

with tab_forecast:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metal = st.selectbox("Metal", ["Gold", "Silver"], key="forecast_metal")
    with col2:
        currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=0, key="forecast_currency")
    with col3:
        price_unit = st.radio("Price Unit", ["Troy Ounce", "Gram"], key="forecast_unit", horizontal=True)
    with col4:
        forecast_days = st.number_input(
            "Forecast Days", min_value=7, max_value=90, value=DEFAULT_FORECAST_DAYS, step=1, key="forecast_days"
        )

    stats = dashboard_metrics(metal, currency, models, featured_data, performance, forecast_days)
    forecast = stats["Forecast"].copy()
    forecast_display = forecast.copy()
    forecast_display["Forecast"] = forecast_display["Forecast"].apply(
        lambda x: round(convert_price(x, currency), 2)
    )

    change_24h = compute_24h_change(featured_data, metal)
    arrow, trend_class = trend_arrow(change_24h)
    dir_acc = performance[metal]["Directional Accuracy %"]
    r2_pct = max(performance[metal]["R2 (price)"], 0) * 100
    conf_label, conf_text_color, conf_bar_color = confidence_tier(stats["Confidence"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""<div class="metric-card accent-neutral"><h4>Current Price</h4>
            <h2>{currency} {convert_price_unit(stats['Current Price'], price_unit)} {get_unit_symbol(price_unit)}</h2>
            <div class="trend-pill {trend_class}">{arrow} {change_24h:+.2f}% <span style="font-weight:500;">(24H)</span></div></div>""",
            unsafe_allow_html=True,
        )
    with c2:
        f_arrow, f_class = trend_arrow(stats["Expected Return"])
        st.markdown(
            f"""<div class="metric-card accent-neutral"><h4>Forecast Price ({forecast_days}D)</h4>
            <h2>{currency} {convert_price_unit(stats['Forecast Price'], price_unit)} {get_unit_symbol(price_unit)}</h2>
            <div class="trend-pill {f_class}">{f_arrow} {stats['Expected Return']:+.2f}% <span style="font-weight:500;">expected</span></div></div>""",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""<div class="metric-card accent-info"><h4>Confidence</h4>
            <h2>{stats['Confidence']}%</h2>
            <div class="confidence-bar-wrap"><div class="confidence-bar-fill" style="--target-width:{min(stats['Confidence'], 100)}%; background:{conf_bar_color};"></div></div>
            <div class="confidence-tier-label" style="color:{conf_text_color}; background:{conf_bar_color}22;">{conf_label} Confidence</div>
            <div class="accuracy-note">Model accuracy: <b>{dir_acc:.1f}%</b></div></div>""",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""<div class="metric-card accent-good"><h4>Recommendation</h4>
            <div class="signal-badge {signal_css_class(stats['Signal'])}">{stats['Signal']}</div></div>""",
            unsafe_allow_html=True,
        )

    with st.expander("Why this recommendation?", expanded=False):
        for reason in signal_reasoning(metal, stats, featured_data, performance, forecast_days):
            st.markdown(f"- {reason}")

    st.markdown('<div class="chart-spacer"></div>', unsafe_allow_html=True)
    st.markdown('<div class="chart-reveal">', unsafe_allow_html=True)
    st.plotly_chart(create_forecast_chart(metal, forecast, featured_data, currency), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.dataframe(forecast_display, use_container_width=True)

with tab_advisor:
    st.subheader("Investment Advisor")
    advisor_amount = st.number_input("Investment Amount", value=100000, key="advisor_amount")
    advisor_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=0, key="advisor_currency")
    if st.button("Generate Advice", type="primary"):
        gold = dashboard_metrics("Gold", advisor_currency, models, featured_data, performance)
        silver = dashboard_metrics("Silver", advisor_currency, models, featured_data, performance)
        rec = "Gold" if gold["Expected Return"] >= silver["Expected Return"] else "Silver"
        st.success(f"Recommended Asset: **{rec}** based on expected multi-day return profiles.")

with tab_analytics:
    st.subheader("Model Performance Metrics")
    perf_display = performance_df.reset_index().rename(columns={"index": "Metal"})
    st.dataframe(perf_display, use_container_width=True, hide_index=True)

with tab_about:
    st.markdown(f"""
# {PROJECT_NAME}

## Overview
This platform forecasts next-day returns for gold and silver futures using an
ensemble of Random Forest and Gradient Boosting regressors.

### Built Using
Python, Streamlit, Plotly, Scikit-learn, Pandas, NumPy, ReportLab
""", unsafe_allow_html=True)
