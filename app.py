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
# CINEMATIC DOORWAY INTRO (plays once per session)
# ==========================================================

if "intro_played" not in st.session_state:
    st.session_state.intro_played = False


def render_cinematic_intro():
    """
    Full-screen cream-and-gold double doors that swing open (with a touch of
    3D via rotateY + perspective), a pulsing gold glow at the seam, a
    handful of drifting gold particles, and the dashboard beneath blurring
    into focus as the doors clear. Pure CSS keyframes so it needs no JS to
    run or to clean itself up — everything fades to opacity:0 /
    pointer-events:none at the end of its own animation.
    """
    particles_html = ""
    for _ in range(22):
        left = random.uniform(2, 98)
        top = random.uniform(8, 92)
        delay = random.uniform(0, 0.9)
        size = random.uniform(2, 5)
        particles_html += (
            f'<span class="intro-particle" style="left:{left:.1f}%; top:{top:.1f}%; '
            f'width:{size:.1f}px; height:{size:.1f}px; animation-delay:{delay:.2f}s;"></span>'
        )

    intro_html = f"""
    <style>
    .intro-overlay {{
        position: fixed; inset: 0; z-index: 999999;
        pointer-events: none;
        perspective: 1800px;
        overflow: hidden;
    }}
    .intro-door {{
        position: absolute; top: 0; width: 50%; height: 100%;
        background: linear-gradient(135deg, #FBF6E9 0%, #F0E4C8 45%, #B8892E 100%);
        box-shadow: 0 0 90px rgba(184,137,46,0.5) inset, 0 0 40px rgba(184,137,46,0.35);
        border: 1px solid rgba(184,137,46,0.4);
    }}
    .intro-door::after {{
        content: "";
        position: absolute; top: 0; bottom: 0; width: 6px;
        background: linear-gradient(180deg, transparent, #B8892E 45%, #E8C56A 50%, #B8892E 55%, transparent);
        box-shadow: 0 0 24px 4px rgba(232,197,106,0.7);
    }}
    .intro-door-left {{
        left: 0; transform-origin: left center;
        animation: door-open-left 2.3s cubic-bezier(.76,0,.2,1) forwards;
    }}
    .intro-door-left::after {{ right: -3px; }}
    .intro-door-right {{
        right: 0; transform-origin: right center;
        animation: door-open-right 2.3s cubic-bezier(.76,0,.2,1) forwards;
    }}
    .intro-door-right::after {{ left: -3px; }}
    @keyframes door-open-left {{
        0%   {{ transform: translateX(0) rotateY(0deg); opacity: 1; }}
        55%  {{ transform: translateX(-6%) rotateY(-22deg); opacity: 1; }}
        100% {{ transform: translateX(-105%) rotateY(-72deg); opacity: 0; }}
    }}
    @keyframes door-open-right {{
        0%   {{ transform: translateX(0) rotateY(0deg); opacity: 1; }}
        55%  {{ transform: translateX(6%) rotateY(22deg); opacity: 1; }}
        100% {{ transform: translateX(105%) rotateY(72deg); opacity: 0; }}
    }}
    .intro-glow {{
        position: absolute; top: 50%; left: 50%; width: 340px; height: 340px;
        transform: translate(-50%, -50%);
        background: radial-gradient(circle, rgba(232,197,106,0.9) 0%, rgba(184,137,46,0.25) 45%, rgba(184,137,46,0) 70%);
        filter: blur(18px);
        animation: intro-glow-pulse 2.3s ease forwards;
    }}
    @keyframes intro-glow-pulse {{
        0%   {{ opacity: 0; transform: translate(-50%, -50%) scale(0.25); }}
        35%  {{ opacity: 1; transform: translate(-50%, -50%) scale(1.3); }}
        100% {{ opacity: 0; transform: translate(-50%, -50%) scale(2.4); }}
    }}
    .intro-particle {{
        position: absolute; border-radius: 50%;
        background: #E8C56A;
        box-shadow: 0 0 10px 3px rgba(232,197,106,0.75);
        animation: intro-particle-float 2.2s ease-out forwards;
        opacity: 0;
    }}
    @keyframes intro-particle-float {{
        0%   {{ opacity: 0; transform: translateY(0) scale(0.5); }}
        18%  {{ opacity: 1; }}
        100% {{ opacity: 0; transform: translateY(-120px) scale(1.3); }}
    }}
    [data-testid="stAppViewBlockContainer"] {{
        animation: intro-blur-focus 2.3s ease forwards;
    }}
    @keyframes intro-blur-focus {{
        0%   {{ filter: blur(20px); transform: scale(1.025); }}
        100% {{ filter: blur(0); transform: scale(1); }}
    }}
    .st-key-skip_intro_btn {{
        position: fixed !important;
        top: 22px; right: 28px;
        z-index: 1000000 !important;
        animation: card-rise 0.5s ease 0.3s both;
    }}
    .st-key-skip_intro_btn button {{
        background: rgba(255,253,247,0.85) !important;
        border: 1px solid rgba(184,137,46,0.6) !important;
        color: #6B5D46 !important;
        font-size: 13px !important;
        padding: 0.3rem 0.9rem !important;
        backdrop-filter: blur(4px);
    }}
    </style>
    <div class="intro-overlay">
        <div class="intro-glow"></div>
        {particles_html}
        <div class="intro-door intro-door-left"></div>
        <div class="intro-door intro-door-right"></div>
    </div>
    """
    st.markdown(intro_html, unsafe_allow_html=True)


if not st.session_state.intro_played:
    render_cinematic_intro()
    skip_clicked = st.button("Skip Intro", key="skip_intro_btn")
    # Mark as played immediately so the intro never replays this session,
    # whether it finishes on its own (~2.3s) or the user skips it.
    st.session_state.intro_played = True
    if skip_clicked:
        st.rerun()


def animated_counter(value_text, prefix="", suffix="", color="#1F1811", size="27px", duration=900, key=""):
    """
    Renders a number that counts up from 0 to `value_text` using a small
    inline JS animation. Falls back gracefully if the value isn't numeric.
    """
    try:
        target = float(str(value_text).replace(",", ""))
    except (TypeError, ValueError):
        st.markdown(f"<h2 style='color:{color};font-size:{size};margin:0;'>{prefix}{value_text}{suffix}</h2>", unsafe_allow_html=True)
        return

    decimals = 2 if not float(target).is_integer() else 0
    html = f"""
    <div id="counter-{key}" style="font-family:'Playfair Display',Georgia,serif;font-size:{size};font-weight:700;color:{color};margin:0;">0</div>
    <script>
    (function() {{
        const el = document.getElementById("counter-{key}");
        const target = {target};
        const decimals = {decimals};
        const duration = {duration};
        const start = performance.now();
        function step(now) {{
            const progress = Math.min((now - start) / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = target * eased;
            el.textContent = "{prefix}" + current.toFixed(decimals).replace(/\\B(?=(\\d{{3}})+(?!\\d))/g, ",") + "{suffix}";
            if (progress < 1) requestAnimationFrame(step);
        }}
        requestAnimationFrame(step);
    }})();
    </script>
    """
    import streamlit.components.v1 as components
    components.html(html, height=int(size.replace("px", "")) + 10)


# ==========================================================
# DATA DOWNLOAD (with retry + validation)
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


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Adds all technical indicators to a Close/Open/High/Low/Volume frame."""
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
# MODEL TRAINING (predicts next-day LOG RETURN)
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

    # Keep the test-period predicted-vs-actual series for the "Prediction
    # History" chart in the Analytics tab.
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


# ==========================================================
# FORECAST ENGINE (return-based, recursive, O(n))
# ==========================================================

def predict_next_return(model_dict, feature_row: pd.DataFrame) -> float:
    rf_pred = model_dict["rf"].predict(feature_row)[0]
    gbm_pred = model_dict["gbm"].predict(feature_row)[0]
    return 0.5 * float(rf_pred) + 0.5 * float(gbm_pred)


def forecast_prices(metal, forecast_days, models, featured_data):
    """
    Recursively forecasts forward by predicting the next-day log return and
    compounding it onto the last known price. Only the trailing window of
    history is recomputed for indicators each step.
    """
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
# CHARTS
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
    """
    Builds an approximate confidence range around the forecast using the
    trailing realized volatility of daily log returns, scaled by sqrt(time)
    as under a random-walk assumption. z=1.28 corresponds to ~80% coverage.
    """
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
        x=[history.index[-1]], y=[history["Close"].iloc[-1] * CURRENCIES[currency]], mode="markers",
        name="Current", marker=dict(size=11, color="#8C6A2E", symbol="diamond")
    ))

    if "SMA_20" in history.columns:
        fig.add_trace(go.Scatter(
            x=history.index, y=history["SMA_20"] * CURRENCIES[currency], mode="lines",
            name="SMA 20", line=dict(color="#A08B63", width=1.5), visible="legendonly"
        ))

    if "SMA_50" in history.columns:
        fig.add_trace(go.Scatter(
            x=history.index, y=history["SMA_50"] * CURRENCIES[currency], mode="lines",
            name="SMA 50", line=dict(color="#5C4A32", width=1.5), visible="legendonly"
        ))

    # Shaded confidence range (drawn before the forecast line so it sits underneath)
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

    fig.add_trace(go.Scatter(
        x=[forecast["Date"].iloc[0]], y=[forecast["Forecast"].iloc[0] * CURRENCIES[currency]],
        mode="markers", name="Forecast Start",
        marker=dict(size=12, color="#4C6B48", symbol="star")
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
        rangeselector=dict(
            buttons=[
                dict(count=7, label="7D", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="3M", step="month", stepmode="backward"),
                dict(count=6, label="6M", step="month", stepmode="backward"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#FFFDF7",
            activecolor="#B8892E",
            bordercolor="#B8892E",
            borderwidth=1,
            font=dict(color="#241B0F", size=12, family="Georgia, serif"),
            x=0, xanchor="left", y=1.18, yanchor="bottom",
        ),
        # Default view: recent history through the end of the forecast horizon
        range=[history.index[-default_window_days], forecast["Date"].iloc[-1]],
    )
    fig.update_yaxes(
        showgrid=True, gridcolor="#EFE4CD",
        title=dict(text="Price", font=AXIS_TITLE_FONT),
        tickfont=AXIS_TICK_FONT,
        range=default_y_range,
    )

    return fig


def create_technical_chart(metal, featured_data, indicator, currency, window_days=180):
    """Builds a standalone chart for a single technical indicator (RSI, MACD, SMA overlay, or Volatility)."""
    history = featured_data[metal].tail(window_days).copy()
    fig = go.Figure()

    if indicator == "RSI":
        fig.add_trace(go.Scatter(x=history.index, y=history["RSI"], mode="lines",
                                  name="RSI", line=dict(color="#3B6FA0", width=2)))
        fig.add_hline(y=70, line=dict(color="#8C3A2C", width=1, dash="dot"), annotation_text="Overbought (70)")
        fig.add_hline(y=30, line=dict(color="#4C6B48", width=1, dash="dot"), annotation_text="Oversold (30)")
        fig.update_yaxes(range=[0, 100], title=dict(text="RSI", font=AXIS_TITLE_FONT))
        title = f"{metal} — Relative Strength Index (RSI)"

    elif indicator == "MACD":
        fig.add_trace(go.Scatter(x=history.index, y=history["MACD"], mode="lines",
                                  name="MACD", line=dict(color="#4C6B48", width=2)))
        fig.add_trace(go.Scatter(x=history.index, y=history["MACD_SIGNAL"], mode="lines",
                                  name="Signal", line=dict(color="#B8892E", width=2, dash="dash")))
        hist = history["MACD"] - history["MACD_SIGNAL"]
        bar_colors = ["#4C6B48" if v >= 0 else "#8C3A2C" for v in hist]
        fig.add_trace(go.Bar(x=history.index, y=hist, name="Histogram",
                              marker=dict(color=bar_colors, opacity=0.4)))
        fig.update_yaxes(title=dict(text="MACD", font=AXIS_TITLE_FONT))
        title = f"{metal} — MACD"

    elif indicator == "SMA":
        fig.add_trace(go.Scatter(x=history.index, y=history["Close"] * CURRENCIES[currency], mode="lines",
                                  name="Close", line=dict(color="#2E271F", width=2)))
        fig.add_trace(go.Scatter(x=history.index, y=history["SMA_5"] * CURRENCIES[currency], mode="lines",
                                  name="SMA 5", line=dict(color="#B8892E", width=1.4)))
        fig.add_trace(go.Scatter(x=history.index, y=history["SMA_20"] * CURRENCIES[currency], mode="lines",
                                  name="SMA 20", line=dict(color="#A08B63", width=1.4)))
        fig.add_trace(go.Scatter(x=history.index, y=history["SMA_50"] * CURRENCIES[currency], mode="lines",
                                  name="SMA 50", line=dict(color="#5C4A32", width=1.4)))
        fig.update_yaxes(title=dict(text="Price", font=AXIS_TITLE_FONT))
        title = f"{metal} — Moving Averages"

    else:  # Volatility
        fig.add_trace(go.Scatter(x=history.index, y=history["Volatility"] * 100, mode="lines",
                                  name="Volatility", line=dict(color="#8C3A2C", width=2),
                                  fill="tozeroy", fillcolor="rgba(140,58,44,0.12)"))
        fig.update_yaxes(title=dict(text="10D Rolling Volatility (%)", font=AXIS_TITLE_FONT))
        title = f"{metal} — Volatility"

    fig.update_layout(
        height=340,
        margin=dict(l=55, r=30, t=60, b=10),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=11, color="#2E2013")),
        title=dict(text=title, font=dict(size=15, color="#241B0F", family="'Playfair Display', Georgia, serif")),
        **CHART_TEMPLATE,
    )
    fig.update_xaxes(showgrid=False, tickfont=AXIS_TICK_FONT, linecolor="#8C7A54")
    fig.update_yaxes(showgrid=True, gridcolor="#EFE4CD", tickfont=AXIS_TICK_FONT)
    return fig


def create_prediction_history_chart(metal, prediction_history, currency, window_days=90):
    """Predicted vs Actual price on the held-out test window."""
    df = prediction_history[metal].tail(window_days).copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Actual"] * CURRENCIES[currency], mode="lines",
                              name="Actual", line=dict(color="#2E271F", width=2)))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Predicted"] * CURRENCIES[currency], mode="lines",
                              name="Predicted", line=dict(color="#4C6B48", width=2, dash="dash")))
    fig.update_layout(
        height=360,
        margin=dict(l=55, r=30, t=60, b=10),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=11, color="#2E2013")),
        title=dict(text=f"{metal} — Predicted vs Actual (Held-out Test Window)",
                   font=dict(size=15, color="#241B0F", family="'Playfair Display', Georgia, serif")),
        **CHART_TEMPLATE,
    )
    fig.update_xaxes(showgrid=False, tickfont=AXIS_TICK_FONT, linecolor="#8C7A54")
    fig.update_yaxes(showgrid=True, gridcolor="#EFE4CD", title=dict(text="Price", font=AXIS_TITLE_FONT), tickfont=AXIS_TICK_FONT)
    return fig


# ==========================================================
# DASHBOARD METRICS
# ==========================================================

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
    """Returns (tier label, text color, bar/badge color) for the confidence score."""
    if confidence >= 80:
        return "High", "#2E4A2A", "#4C6B48"
    if confidence >= 65:
        return "Medium", "#7A5B1E", "#B8892E"
    return "Low", "#8C3A2C", "#A6493A"


def compute_24h_change(featured_data, metal):
    """Latest close vs the prior close, as a percentage."""
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
    """Plain-language breakdown of why the model landed on this signal."""
    df = featured_data[metal]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_SIGNAL"].iloc[-1]
    dir_acc = performance[metal]["Directional Accuracy %"]

    reasons = []
    reasons.append(
        f"Model projects a {stats['Expected Return']:+.2f}% move over the next {forecast_days} days."
    )
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
# AI MARKET PULSE  (Bullish / Neutral / Bearish + reasons)
# ==========================================================

def compute_market_pulse(metal, featured_data, performance, expected_return):
    """
    Combines RSI, MACD crossover, moving-average trend, recent volatility,
    and the model's directional accuracy into a single Bullish / Neutral /
    Bearish read, with a 0-100 score and plain-language reasons.
    """
    df = featured_data[metal]
    rsi = df["RSI"].iloc[-1]
    macd = df["MACD"].iloc[-1]
    macd_signal = df["MACD_SIGNAL"].iloc[-1]
    sma_20 = df["SMA_20"].iloc[-1]
    sma_50 = df["SMA_50"].iloc[-1]
    close = df["Close"].iloc[-1]
    dir_acc = performance[metal]["Directional Accuracy %"]

    score = 50.0
    reasons = []

    if expected_return > 0.5:
        score += 15
        reasons.append(f"Forecast return is positive at {expected_return:+.2f}%.")
    elif expected_return < -0.5:
        score -= 15
        reasons.append(f"Forecast return is negative at {expected_return:+.2f}%.")
    else:
        reasons.append(f"Forecast return is roughly flat at {expected_return:+.2f}%.")

    if macd > macd_signal:
        score += 10
        reasons.append("MACD sits above its signal line — bullish momentum.")
    else:
        score -= 10
        reasons.append("MACD sits below its signal line — bearish momentum.")

    if rsi >= 70:
        score -= 8
        reasons.append(f"RSI at {rsi:.0f} signals overbought conditions.")
    elif rsi <= 30:
        score += 8
        reasons.append(f"RSI at {rsi:.0f} signals oversold conditions, room to recover.")
    else:
        reasons.append(f"RSI at {rsi:.0f} is in neutral territory.")

    if close > sma_20 > sma_50:
        score += 12
        reasons.append("Price is trading above both the 20D and 50D averages — uptrend intact.")
    elif close < sma_20 < sma_50:
        score -= 12
        reasons.append("Price is trading below both the 20D and 50D averages — downtrend intact.")
    else:
        reasons.append("Price is mixed relative to its 20D/50D averages — no clear trend.")

    if dir_acc >= 55:
        score += 5
        reasons.append(f"Model's historical directional accuracy ({dir_acc:.1f}%) supports the read.")
    else:
        reasons.append(f"Model's historical directional accuracy ({dir_acc:.1f}%) is modest — treat with caution.")

    score = max(0, min(100, score))

    if score >= 62:
        label = "Bullish"
    elif score <= 38:
        label = "Bearish"
    else:
        label = "Neutral"

    return {"label": label, "score": round(score, 1), "reasons": reasons}


def pulse_css_class(label):
    return {"Bullish": "pulse-bullish", "Neutral": "pulse-neutral", "Bearish": "pulse-bearish"}[label]


def pulse_bar_color(label):
    return {"Bullish": "#4C6B48", "Neutral": "#B8892E", "Bearish": "#A6493A"}[label]


def render_market_pulse_card(metal, pulse):
    css_class = pulse_css_class(pulse["label"])
    bar_color = pulse_bar_color(pulse["label"])
    reasons_html = "".join(f"<li>{r}</li>" for r in pulse["reasons"])
    st.markdown(
        f"""
        <div class="metric-card" style="text-align:left;">
            <h4 style="text-align:center;">{metal} — AI Market Pulse</h4>
            <div style="text-align:center;">
                <span class="pulse-badge {css_class}">{pulse['label']}</span>
            </div>
            <div class="pulse-score-wrap">
                <div class="pulse-score-fill" style="background:{bar_color}; --target-width:{pulse['score']}%;"></div>
            </div>
            <div style="text-align:center;font-size:12px;color:#8C7A54;margin-top:4px;">Pulse score: <b>{pulse['score']}</b> / 100</div>
            <ul class="pulse-reason-list">{reasons_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def market_brief_text(gold_pulse, silver_pulse, news_mood=None):
    """Short combined narrative brief across both metals."""
    lines = [
        f"**Gold** is reading **{gold_pulse['label']}** (score {gold_pulse['score']}/100), "
        f"while **Silver** is reading **{silver_pulse['label']}** (score {silver_pulse['score']}/100).",
    ]
    if gold_pulse["label"] == silver_pulse["label"]:
        lines.append(f"Both metals are aligned in a {gold_pulse['label'].lower()} posture right now.")
    else:
        lines.append("The two metals are currently diverging, which can be a cue to size positions independently rather than treating them as one trade.")
    if news_mood:
        lines.append(f"Recent headline sentiment leans **{news_mood}**.")
    return "\n\n".join(lines)


def metric_card(title, value):
    st.markdown(
        f"""<div class="metric-card"><h4>{title}</h4><h2>{value}</h2></div>""",
        unsafe_allow_html=True,
    )


def export_csv(df):
    path = os.path.join(tempfile.gettempdir(), "forecast.csv")
    df.to_csv(path, index=False)
    return path


def export_excel(df):
    path = os.path.join(tempfile.gettempdir(), "forecast.xlsx")
    df.to_excel(path, index=False)
    return path


# ==========================================================
# LIVE MARKET BANNER
# ==========================================================

@st.cache_data(ttl=15, show_spinner=False)
def get_live_market():
    try:
        gold_price = yf.Ticker("GC=F").history(period="1d")["Close"].iloc[-1]
        silver_price = yf.Ticker("SI=F").history(period="1d")["Close"].iloc[-1]
        return {
            "Gold": round(float(gold_price), 2),
            "Silver": round(float(silver_price), 2),
            "USDINR": CURRENCIES["INR"],
            "Time": datetime.now().strftime("%H:%M:%S"),
        }
    except Exception:
        return {"Gold": "--", "Silver": "--", "USDINR": "--", "Time": "Offline"}


@st.cache_data(ttl=20, show_spinner=False)
def get_intraday_series(symbol, max_points=120):
    """
    Real intraday minute bars for the live sparkline. Falls back to 5-minute
    bars over the last few days when 1-minute data isn't available (e.g.
    market closed / weekend), so the chart still shows something real.
    """
    try:
        df = yf.Ticker(symbol).history(period="1d", interval="1m")
        if df is None or df.empty:
            df = yf.Ticker(symbol).history(period="5d", interval="5m")
        if df is None or df.empty:
            return pd.Series(dtype=float)
        return df["Close"].dropna().tail(max_points)
    except Exception:
        return pd.Series(dtype=float)


def create_ticker_sparkline(series, currency, line_color, fill_color):
    """Small, axis-free live line chart used for the header ticker."""
    fig = go.Figure()

    if series.empty:
        fig.add_annotation(
            text="Live data unavailable", showarrow=False,
            font=dict(color="#8C7A54", size=11),
        )
        y_range = None
    else:
        y = series.values * CURRENCIES[currency]
        x = list(series.index)
        y_min, y_max = float(np.min(y)), float(np.max(y))
        pad = (y_max - y_min) * 0.15 or max(y_max * 0.001, 0.01)
        y_range = [y_min - pad, y_max + pad]

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color=line_color, width=1.8),
            fill="tozeroy", fillcolor=fill_color,
            hovertemplate="%{y:.2f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[x[-1]], y=[y[-1]], mode="markers",
            marker=dict(size=6, color=line_color),
            showlegend=False, hoverinfo="skip",
        ))

    fig.update_layout(
        height=70,
        margin=dict(l=0, r=0, t=2, b=0),
        paper_bgcolor="#FFFDF7",
        plot_bgcolor="#FFFDF7",
        showlegend=False,
        hovermode="x",
    )
    fig.update_xaxes(visible=False, showgrid=False)
    fig.update_yaxes(visible=False, showgrid=False, zeroline=False, range=y_range)
    return fig


@st.cache_data(ttl=30, show_spinner=False)
def get_live_24h_change(symbol):
    """Previous close vs latest close, used to label the live ticker charts."""
    try:
        closes = yf.Ticker(symbol).history(period="5d", interval="1d")["Close"].dropna()
        if len(closes) < 2:
            return None
        prev, last = float(closes.iloc[-2]), float(closes.iloc[-1])
        return ((last - prev) / prev) * 100
    except Exception:
        return None


def render_live_banner(currency):
    market = get_live_market()

    # Track previous values in session state so we can flash green/red only
    # when the price actually moves between refreshes.
    prev = st.session_state.get("_prev_live_market", {})

    def fmt_with_flash(label, usd_price):
        if isinstance(usd_price, str):
            return usd_price, ""
        display = f"{round(usd_price * CURRENCIES[currency], 2)}"
        prev_val = prev.get(label)
        flash_class = ""
        if prev_val is not None and not isinstance(prev_val, str):
            if usd_price > prev_val:
                flash_class = "flash-up"
            elif usd_price < prev_val:
                flash_class = "flash-down"
        return display, flash_class

    gold_display, gold_flash = fmt_with_flash("Gold", market["Gold"])
    silver_display, silver_flash = fmt_with_flash("Silver", market["Silver"])

    st.session_state["_prev_live_market"] = {"Gold": market["Gold"], "Silver": market["Silver"]}

    st.markdown(
        f"""
        <div class="live-banner">
            <span class="live-dot"></span>
            <div class="live-item"><div class="live-label">Gold</div><div class="live-value {gold_flash}" style="color:#B8892E;">{currency} {gold_display}</div></div>
            <div class="live-item"><div class="live-label">Silver</div><div class="live-value {silver_flash}" style="color:#6B7280;">{currency} {silver_display}</div></div>
            <div class="live-item"><div class="live-label">USD/INR</div><div class="live-value" style="color:#3B6FA0;">{market['USDINR']} <small style="color:#8C7A54;font-weight:400;">(static)</small></div></div>
            <div class="live-item"><div class="live-label">Updated</div><div class="live-value">{market['Time']}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=5)
def render_live_section(currency):
    """
    Self-refreshing block: reruns every 5 seconds without re-triggering the
    full app (so models aren't retrained). Underlying data is still fetched
    at most once per cache TTL (15-20s), so this animates the live view
    without hammering the data source.
    """
    render_live_banner(currency)
    gold_series = get_intraday_series(MARKET_SYMBOLS["Gold"])
    silver_series = get_intraday_series(MARKET_SYMBOLS["Silver"])
    gold_change = get_live_24h_change(MARKET_SYMBOLS["Gold"])
    silver_change = get_live_24h_change(MARKET_SYMBOLS["Silver"])

    def ticker_header(label, change_pct):
        if change_pct is None:
            change_html = '<span class="ticker-change trend-flat">n/a</span>'
        else:
            arrow, trend_class = trend_arrow(change_pct)
            change_html = f'<span class="ticker-change {trend_class}">{arrow} {change_pct:+.2f}%</span>'
        return (
            '<div class="ticker-label">'
            f'<span class="ticker-label-left"><span class="live-dot"></span> {label} — Live</span>'
            f'<span class="ticker-label-right">{change_html}<span class="ticker-window">24H</span></span>'
            '</div>'
        )

    t1, t2 = st.columns(2)
    with t1:
        st.markdown(ticker_header("Gold", gold_change), unsafe_allow_html=True)
        st.plotly_chart(
            create_ticker_sparkline(gold_series, currency, "#B8892E", "rgba(184,137,46,0.15)"),
            use_container_width=True, config={"displayModeBar": False}, key="gold_spark",
        )
    with t2:
        st.markdown(ticker_header("Silver", silver_change), unsafe_allow_html=True)
        st.plotly_chart(
            create_ticker_sparkline(silver_series, currency, "#6B7280", "rgba(107,114,128,0.15)"),
            use_container_width=True, config={"displayModeBar": False}, key="silver_spark",
        )



# ==========================================================
# AI SUMMARY / ADVISOR
# ==========================================================

def ai_summary(metal, currency, price_unit, stats):
    current_price = convert_price_unit(stats["Current Price"], price_unit)
    forecast_price = convert_price_unit(stats["Forecast Price"], price_unit)

    return f"""
**Metal:** {metal}
**Current Price:** {currency} {current_price:.2f} {get_unit_symbol(price_unit)}
**Forecast Price (period):** {currency} {forecast_price:.2f} {get_unit_symbol(price_unit)}
**Expected Return:** {stats['Expected Return']:.2f}%
**Recommendation:** {stats['Signal']}
**Confidence:** {stats['Confidence']}%  (blend of directional accuracy and R2)
"""


def ai_investment_advisor(investment, currency, models, featured_data, performance):
    gold = dashboard_metrics("Gold", currency, models, featured_data, performance)
    silver = dashboard_metrics("Silver", currency, models, featured_data, performance)

    if gold["Expected Return"] >= silver["Expected Return"]:
        recommendation, metrics = "Gold", gold
    else:
        recommendation, metrics = "Silver", silver

    confidence = metrics["Confidence"]
    if confidence >= 80:
        risk = "Low"
    elif confidence >= 65:
        risk = "Medium"
    else:
        risk = "High"

    reasons = []
    sig = metrics["Signal"]
    if "BUY" in sig:
        reasons.append("- Bullish signal detected.")
    elif "SELL" in sig:
        reasons.append("- Bearish signal detected, consider caution.")
    else:
        reasons.append("- Neutral / HOLD signal detected.")

    reasons.append(
        "- Model confidence is high." if confidence >= 80 else
        "- Model confidence is moderate." if confidence >= 65 else
        "- Model confidence is relatively low, treat forecast as directional, not precise."
    )
    reasons.append(
        "- Positive expected return." if metrics["Expected Return"] > 0 else
        "- Negative expected return." if metrics["Expected Return"] < 0 else
        "- Neutral expected return."
    )

    return f"""
### Investment Amount
{currency} {investment:,.2f}

### Recommended Asset
{recommendation}

### Why this Recommendation
{chr(10).join(reasons)}

### Expected Return
{metrics['Expected Return']:.2f} %

### Trading Signal
{metrics['Signal']}

### Confidence
{metrics['Confidence']:.2f} %

### Risk Level
{risk}

### Suggested Holding Period
30 Days

---
*This is a model-generated estimate, not financial advice.*
"""


# ==========================================================
# INVESTMENT / PORTFOLIO / ALERTS / BACKTEST
# ==========================================================

def investment_calculator(metal, amount, currency, models, featured_data, performance):
    m = dashboard_metrics(metal, currency, models, featured_data, performance)
    units = amount / m["Current Price"]
    future_value = units * m["Forecast Price"]
    profit = future_value - amount
    roi = (profit / amount) * 100

    return pd.DataFrame({
        "Investment": [round(amount, 2)],
        "Current Price": [m["Current Price"]],
        "Forecast Price": [m["Forecast Price"]],
        "Units Purchased": [round(units, 6)],
        "Future Value": [round(future_value, 2)],
        "Profit / Loss": [round(profit, 2)],
        "ROI %": [round(roi, 2)],
    })


def compare_metals(currency, models, featured_data, performance):
    gold = dashboard_metrics("Gold", currency, models, featured_data, performance)
    silver = dashboard_metrics("Silver", currency, models, featured_data, performance)
    return pd.DataFrame({
        "Metric": ["Current Price", "Forecast Price", "Expected Return %", "Confidence %", "Recommendation"],
        "Gold": [gold["Current Price"], gold["Forecast Price"], gold["Expected Return"], gold["Confidence"], gold["Signal"]],
        "Silver": [silver["Current Price"], silver["Forecast Price"], silver["Expected Return"], silver["Confidence"], silver["Signal"]],
    })


def create_comparison_chart(currency, models, featured_data, performance):
    gold = dashboard_metrics("Gold", currency, models, featured_data, performance)
    silver = dashboard_metrics("Silver", currency, models, featured_data, performance)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Expected Return %", "Confidence %"],
        y=[gold["Expected Return"], gold["Confidence"]],
        name="Gold", marker_color="#B8892E",
    ))
    fig.add_trace(go.Bar(
        x=["Expected Return %", "Confidence %"],
        y=[silver["Expected Return"], silver["Confidence"]],
        name="Silver", marker_color="#6B7280",
    ))
    fig.update_layout(
        barmode="group",
        height=380,
        margin=dict(l=40, r=20, t=50, b=10),
        title=dict(text="Gold vs Silver — Head to Head", font=dict(size=16, color="#241B0F", family="'Playfair Display', Georgia, serif")),
        legend=dict(orientation="h", x=0, y=1.12, font=dict(size=12, color="#2E2013")),
        **CHART_TEMPLATE,
    )
    fig.update_xaxes(showgrid=False, tickfont=AXIS_TICK_FONT)
    fig.update_yaxes(showgrid=True, gridcolor="#EFE4CD", tickfont=AXIS_TICK_FONT)
    return fig


def portfolio_optimizer(investment, currency, models, featured_data, performance):
    gold = dashboard_metrics("Gold", currency, models, featured_data, performance)
    silver = dashboard_metrics("Silver", currency, models, featured_data, performance)

    gold_return = max(gold["Expected Return"], 0)
    silver_return = max(silver["Expected Return"], 0)
    total = gold_return + silver_return

    gold_weight = 50 if total == 0 else (gold_return / total) * 100
    silver_weight = 50 if total == 0 else (silver_return / total) * 100

    return pd.DataFrame({
        "Metal": ["Gold", "Silver"],
        "Allocation %": [round(gold_weight, 2), round(silver_weight, 2)],
        "Investment": [round(investment * gold_weight / 100, 2), round(investment * silver_weight / 100, 2)],
        "Expected Return %": [round(gold["Expected Return"], 2), round(silver["Expected Return"], 2)],
    })


def check_price_alert(metal, target_price, currency, models, featured_data, performance, direction="Above"):
    m = dashboard_metrics(metal, currency, models, featured_data, performance)
    current_price = m["Current Price"]
    diff = current_price - target_price
    distance_pct = (diff / target_price) * 100 if target_price else 0.0

    if direction == "Above":
        triggered = current_price >= target_price
        message = (
            f"{metal} has reached or crossed above your target price."
            if triggered else
            f"{metal} is still {abs(distance_pct):.2f}% below your target price."
        )
    else:
        triggered = current_price <= target_price
        message = (
            f"{metal} has reached or dropped below your target price."
            if triggered else
            f"{metal} is still {abs(distance_pct):.2f}% above your target price."
        )

    status = "ALERT TRIGGERED" if triggered else "ALERT NOT TRIGGERED"

    return pd.DataFrame({
        "Metal": [metal], "Current Price": [round(current_price, 2)],
        "Target Price": [round(target_price, 2)], "Direction": [direction],
        "Difference": [round(diff, 2)], "Distance %": [round(distance_pct, 2)],
        "Status": [status], "Message": [message],
    })


def strategy_backtest(metal, investment, featured_data):
    df = featured_data[metal]
    initial_price = df["Close"].iloc[0]
    latest_price = df["Close"].iloc[-1]
    units = investment / initial_price
    final_value = units * latest_price
    profit = final_value - investment
    roi = (profit / investment) * 100

    return pd.DataFrame({
        "Investment": [investment], "Initial Price": [round(initial_price, 2)],
        "Latest Price": [round(latest_price, 2)], "Units Purchased": [round(units, 4)],
        "Final Value": [round(final_value, 2)], "Profit": [round(profit, 2)], "ROI %": [round(roi, 2)],
    })


def scenario_simulation(metal, investment, price_change_pct, currency, models, featured_data, performance):
    """
    Lets the user override the model's forecast with a hypothetical price
    move (e.g. "what if Gold rises 10%?") and see the resulting position
    value. This is a manual what-if tool, independent of the ML forecast.
    """
    m = dashboard_metrics(metal, currency, models, featured_data, performance)
    current_price = m["Current Price"]
    scenario_price = current_price * (1 + price_change_pct / 100)
    units = investment / current_price
    scenario_value = units * scenario_price
    profit = scenario_value - investment
    roi = (profit / investment) * 100 if investment else 0.0

    return pd.DataFrame({
        "Metal": [metal],
        "Scenario Price Change %": [price_change_pct],
        "Investment": [round(investment, 2)],
        "Current Price": [round(current_price, 2)],
        "Scenario Price": [round(scenario_price, 2)],
        "Units Held": [round(units, 6)],
        "Scenario Value": [round(scenario_value, 2)],
        "Profit / Loss": [round(profit, 2)],
        "ROI %": [round(roi, 2)],
    })


# ==========================================================
# NEWS + SENTIMENT
# ==========================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_market_news(metal="Gold", limit=5):
    try:
        rss_url = f"https://news.google.com/rss/search?q={metal}%20price"
        feed = feedparser.parse(rss_url)
        news = [{"Title": e.title, "Link": e.link, "Published": getattr(e, "published", "")}
                for e in feed.entries[:limit]]
        if not news:
            return pd.DataFrame([{"Title": "No news found", "Link": "", "Published": ""}])
        return pd.DataFrame(news)
    except Exception as e:
        return pd.DataFrame([{"Title": f"News fetch failed: {e}", "Link": "", "Published": ""}])


def analyze_news_sentiment(news_df):
    sentiments = []
    for title in news_df["Title"]:
        polarity = TextBlob(title).sentiment.polarity
        if polarity > 0.10:
            sentiments.append("Bullish")
        elif polarity < -0.10:
            sentiments.append("Bearish")
        else:
            sentiments.append("Neutral")
    result = news_df.copy()
    result["Sentiment"] = sentiments
    return result


def market_mood(sentiment_df):
    counts = sentiment_df["Sentiment"].value_counts()
    return counts.idxmax() if len(counts) else "Neutral"


# ==========================================================
# PDF REPORT
# ==========================================================

def premium_report(metal, currency, investment, models, featured_data, performance):
    metrics = dashboard_metrics(metal, currency, models, featured_data, performance)
    filename = os.path.join(tempfile.gettempdir(), "Gold_Silver_Prediction_Report.pdf")
    doc = SimpleDocTemplate(filename)
    story = []

    story.append(Paragraph(f"<font size=22><b>{PROJECT_NAME}</b></font>", styles["Title"]))
    story.append(Paragraph("<font size=14>Market Forecast Report</font>", styles["Heading2"]))
    story.append(Spacer(1, 25))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now()}", styles["Normal"]))
    story.append(Spacer(1, 20))

    data = [
        ["Metric", "Value"],
        ["Metal", metal],
        ["Current Price", metrics["Current Price"]],
        ["Forecast Price", metrics["Forecast Price"]],
        ["Expected Return", metrics["Expected Return"]],
        ["Signal", metrics["Signal"]],
        ["Confidence", metrics["Confidence"]],
    ]
    table = Table(data, colWidths=[220, 220])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8C56A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))
    story.append(table)
    story.append(Spacer(1, 25))

    investment_df = investment_calculator(metal, investment, currency, models, featured_data, performance)
    row = investment_df.iloc[0]
    story.append(Paragraph("<b>Investment Summary</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        f"Investment : {row['Investment']}<br/>"
        f"Future Value : {row['Future Value']}<br/>"
        f"Profit / Loss : {row['Profit / Loss']}<br/>"
        f"ROI : {row['ROI %']} %",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 25))

    story.append(Paragraph("<b>Investment Advisor Notes</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))
    advisor = ai_investment_advisor(investment, currency, models, featured_data, performance).replace("\n", "<br/>")
    story.append(Paragraph(advisor, styles["BodyText"]))

    doc.build(story)
    return filename


# ==========================================================
# APP LAYOUT
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

render_live_section(live_currency)

with st.spinner("Loading market data and training models..."):
    market_data = load_market_data()
    featured_data = build_featured_data(market_data)
    models, performance, performance_df, feature_importance, prediction_history = train_all_models()

(
    tab_forecast, tab_pulse, tab_advisor, tab_comparison,
    tab_news, tab_reports, tab_analytics, tab_about,
) = st.tabs(
    ["Forecast", "Market Pulse", "Advisor", "Comparison", "Market News", "Reports", "Analytics", "About"]
)

# ---------------- FORECAST TAB ----------------
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
        confidence_tooltip = (
            f"Confidence = 0.7 &times; Directional Accuracy + 0.3 &times; R&sup2; (as %).<br><br>"
            f"Directional Accuracy ({dir_acc:.1f}%) is how often the model got the "
            f"up/down direction right, not the exact price.<br>"
            f"R&sup2; ({r2_pct:.1f}%) is how well predicted prices fit actual prices.<br><br>"
            f"It's a weighted blend, so it will not equal either number on its own — "
            f"that's why it can read higher than the accuracy figure below."
        )
        st.markdown(
            f"""<div class="metric-card accent-info"><h4>Confidence
            <span class="info-tooltip">i<span class="tooltip-text">{confidence_tooltip}</span></span></h4>
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
    st.caption(
        "Drag the range slider or use the buttons above the chart to zoom into a time period. "
        "The shaded green band around the forecast is an approximate 80% confidence range "
        "derived from recent price volatility."
    )


    st.markdown(ai_summary(metal, currency, price_unit, stats))

    st.dataframe(forecast_display, use_container_width=True)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "Download Forecast (CSV)",
            data=forecast_display.to_csv(index=False),
            file_name="forecast.csv",
            mime="text/csv",
        )
    with dl2:
        excel_path = export_excel(forecast_display)
        with open(excel_path, "rb") as f:
            st.download_button(
                "Download Forecast (Excel)",
                data=f.read(),
                file_name="forecast.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

# ---------------- MARKET PULSE TAB ----------------
with tab_pulse:
    st.subheader("AI Market Pulse")
    st.caption(
        "A rules-based read combining the model's forecast, RSI, MACD crossover, "
        "moving-average trend, and historical directional accuracy into a single "
        "Bullish / Neutral / Bearish score for each metal."
    )

    pulse_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=0, key="pulse_currency")

    gold_stats_p = dashboard_metrics("Gold", pulse_currency, models, featured_data, performance)
    silver_stats_p = dashboard_metrics("Silver", pulse_currency, models, featured_data, performance)
    gold_pulse = compute_market_pulse("Gold", featured_data, performance, gold_stats_p["Expected Return"])
    silver_pulse = compute_market_pulse("Silver", featured_data, performance, silver_stats_p["Expected Return"])

    pc1, pc2 = st.columns(2)
    with pc1:
        render_market_pulse_card("Gold", gold_pulse)
    with pc2:
        render_market_pulse_card("Silver", silver_pulse)

    st.markdown("### Market Brief")
    include_news = st.checkbox("Include latest news sentiment in the brief", value=False, key="pulse_include_news")
    news_mood = None
    if include_news:
        with st.spinner("Fetching recent headlines..."):
            news_g = analyze_news_sentiment(fetch_market_news("Gold", limit=8))
            news_s = analyze_news_sentiment(fetch_market_news("Silver", limit=8))
            combined_news = pd.concat([news_g, news_s], ignore_index=True)
            if "Sentiment" in combined_news.columns and len(combined_news):
                news_mood = market_mood(combined_news).lower()

    st.markdown(market_brief_text(gold_pulse, silver_pulse, news_mood))

# ---------------- ADVISOR TAB ----------------
with tab_advisor:
    st.subheader("Investment Advisor")
    a1, a2 = st.columns(2)
    with a1:
        advisor_amount = st.number_input("Investment Amount", value=100000, key="advisor_amount")
    with a2:
        advisor_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index("INR"), key="advisor_currency")

    if st.button("Generate Advice", type="primary"):
        st.markdown(ai_investment_advisor(advisor_amount, advisor_currency, models, featured_data, performance))

    st.markdown("---")
    st.subheader("Smart Price Alert")
    al1, al2, al3, al4 = st.columns(4)
    with al1:
        alert_metal = st.selectbox("Metal", ["Gold", "Silver"], key="alert_metal")
    with al2:
        alert_target = st.number_input("Target Price", value=350000, key="alert_target")
    with al3:
        alert_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index("INR"), key="alert_currency")
    with al4:
        alert_direction = st.radio("Trigger When", ["Above", "Below"], key="alert_direction", horizontal=True)

    if st.button("Check Alert"):
        table = check_price_alert(alert_metal, alert_target, alert_currency, models, featured_data, performance, alert_direction)
        st.dataframe(table, use_container_width=True, hide_index=True)
        row = table.iloc[0]
        if row["Status"] == "ALERT TRIGGERED":
            st.success(row["Message"])
        else:
            st.info(row["Message"])

    st.markdown("---")
    st.subheader("Scenario Simulator")
    st.caption("A manual what-if tool — override the model forecast with your own hypothetical price move.")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        scenario_metal = st.selectbox("Metal", ["Gold", "Silver"], key="scenario_metal")
    with sc2:
        scenario_investment = st.number_input("Investment Amount", value=100000, key="scenario_investment")
    with sc3:
        scenario_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index("INR"), key="scenario_currency")

    scenario_change = st.slider("Hypothetical Price Change (%)", min_value=-30, max_value=30, value=10, step=1, key="scenario_change")

    scenario_df = scenario_simulation(
        scenario_metal, scenario_investment, scenario_change, scenario_currency, models, featured_data, performance
    )
    row = scenario_df.iloc[0]
    roi_color = "#385B34" if row["ROI %"] >= 0 else "#8C3A2C"
    st.markdown(
        f"""<div class="scenario-readout">
        If {scenario_metal} moves <b>{scenario_change:+d}%</b> from {scenario_currency} {row['Current Price']:,.2f} to
        {scenario_currency} {row['Scenario Price']:,.2f}, a {scenario_currency} {scenario_investment:,.2f} position
        would be worth <span class="scenario-value">{scenario_currency} {row['Scenario Value']:,.2f}</span>,
        a <span class="scenario-value" style="color:{roi_color};">{row['ROI %']:+.2f}%</span> return.
        </div>""",
        unsafe_allow_html=True,
    )
    st.dataframe(scenario_df, use_container_width=True, hide_index=True)

# ---------------- COMPARISON TAB ----------------
with tab_comparison:
    st.subheader("Gold vs Silver Comparison")
    comparison_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=0, key="comparison_currency")

    comparison_table = compare_metals(comparison_currency, models, featured_data, performance)
    st.dataframe(comparison_table, use_container_width=True, hide_index=True)
    st.plotly_chart(create_comparison_chart(comparison_currency, models, featured_data, performance), use_container_width=True)

    st.markdown("### Suggested Allocation")
    st.caption("Weights expected-return-positive metals proportionally; splits evenly if both are non-positive.")
    portfolio_amount = st.number_input("Total Investment", value=100000, key="portfolio_amount")
    allocation_table = portfolio_optimizer(portfolio_amount, comparison_currency, models, featured_data, performance)
    st.dataframe(allocation_table, use_container_width=True, hide_index=True)

# ---------------- NEWS TAB ----------------
with tab_news:
    st.subheader("Precious Metal News")
    news_metal = st.selectbox("Metal", ["Gold", "Silver"], key="news_metal")
    if st.button("Fetch Latest News", type="primary"):
        news = fetch_market_news(news_metal)
        sentiment = analyze_news_sentiment(news)
        mood = market_mood(sentiment)
        news_column_config = {
            "Link": st.column_config.LinkColumn("Link", display_text="Open article"),
        }
        st.markdown("**Latest News**")
        st.dataframe(news, use_container_width=True, hide_index=True, column_config=news_column_config)
        st.markdown("**News Sentiment**")
        st.dataframe(sentiment, use_container_width=True, hide_index=True, column_config=news_column_config)
        st.markdown(f"**Overall Market Mood:** {mood}")

# ---------------- REPORTS TAB ----------------
with tab_reports:
    st.subheader("PDF Reports")
    r1, r2, r3 = st.columns(3)
    with r1:
        report_metal = st.selectbox("Metal", ["Gold", "Silver"], key="report_metal")
    with r2:
        report_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index("INR"), key="report_currency")
    with r3:
        report_amount = st.number_input("Investment Amount", value=100000, key="report_amount")

    if st.button("Generate PDF Report", type="primary"):
        pdf_path = premium_report(report_metal, report_currency, report_amount, models, featured_data, performance)
        with open(pdf_path, "rb") as f:
            st.download_button(
                "Download Report",
                data=f.read(),
                file_name="Gold_Silver_Prediction_Report.pdf",
                mime="application/pdf",
            )

# ---------------- ANALYTICS TAB ----------------
with tab_analytics:
    st.subheader("Model Analytics")
    an1, an2, an3 = st.columns(3)
    with an1:
        show_performance = st.button("Model Performance", use_container_width=True)
    with an2:
        show_feature_importance = st.button("Feature Importance", use_container_width=True)
    with an3:
        show_backtest = st.button("Strategy Backtest", use_container_width=True)

    if show_performance:
        perf_display = performance_df.reset_index().rename(columns={"index": "Metal"})
        st.markdown("**Model Performance**")
        st.dataframe(
            perf_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Metal": st.column_config.TextColumn("Metal", width="small"),
                "R2 (price)": st.column_config.NumberColumn("R2 (price)", format="%.4f", width="small"),
                "MAE (price)": st.column_config.NumberColumn("MAE (price)", format="%.4f", width="small"),
                "RMSE (price)": st.column_config.NumberColumn("RMSE (price)", format="%.4f", width="small"),
                "Directional Accuracy %": st.column_config.NumberColumn(
                    "Directional Accuracy %", format="%.2f", width="medium"
                ),
            },
        )

    if show_feature_importance:
        importance = feature_importance["Gold"]
        fig = px.bar(importance.head(15), x="Importance", y="Feature", orientation="h",
                     title="Top 15 Important Features (Gold)")
        fig.update_layout(**CHART_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

    if show_backtest:
        table = strategy_backtest("Gold", 100000, featured_data)
        st.markdown("**Strategy Backtest**")
        st.dataframe(table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Technical Indicators")
    ti1, ti2 = st.columns(2)
    with ti1:
        ti_metal = st.selectbox("Metal", ["Gold", "Silver"], key="ti_metal")
    with ti2:
        ti_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=0, key="ti_currency")

    ti_tab_sma, ti_tab_rsi, ti_tab_macd, ti_tab_vol = st.tabs(["Moving Averages", "RSI", "MACD", "Volatility"])
    with ti_tab_sma:
        st.markdown('<div class="chart-reveal">', unsafe_allow_html=True)
        st.plotly_chart(create_technical_chart(ti_metal, featured_data, "SMA", ti_currency), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ti_tab_rsi:
        st.markdown('<div class="chart-reveal">', unsafe_allow_html=True)
        st.plotly_chart(create_technical_chart(ti_metal, featured_data, "RSI", ti_currency), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ti_tab_macd:
        st.markdown('<div class="chart-reveal">', unsafe_allow_html=True)
        st.plotly_chart(create_technical_chart(ti_metal, featured_data, "MACD", ti_currency), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ti_tab_vol:
        st.markdown('<div class="chart-reveal">', unsafe_allow_html=True)
        st.plotly_chart(create_technical_chart(ti_metal, featured_data, "Volatility", ti_currency), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Prediction History — Predicted vs Actual")
    st.caption("Shows the model's predictions against actual prices on the held-out test window (not seen during training).")
    ph_metal = st.selectbox("Metal", ["Gold", "Silver"], key="ph_metal")
    ph_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=0, key="ph_currency")
    st.plotly_chart(create_prediction_history_chart(ph_metal, prediction_history, ph_currency), use_container_width=True)

# ---------------- ABOUT TAB ----------------
with tab_about:
    st.markdown(f"""
# {PROJECT_NAME}

## Overview
This platform forecasts next-day returns for gold and silver futures using an
ensemble of Random Forest and Gradient Boosting regressors, and translates
those forecasts into readable prices, trading signals, and investment tools.

## Model Notes
- Model predicts **next-day returns**, not raw price, so forecasts do not flatten out unrealistically over the horizon.
- Ensemble of **Random Forest** and **Gradient Boosting**.
- **Walk-forward cross-validation** and a **directional accuracy** metric are reported alongside R2.
- The forecast loop runs in O(n), recomputing only a trailing indicator window at each step.
- FX conversion uses a clearly labeled static rate table rather than a live feed.
- **AI Market Pulse** blends forecast direction, RSI, MACD crossover, moving-average trend, and directional accuracy into a Bullish/Neutral/Bearish read — it is a rules-based heuristic, not a separate ML model.
- **Prediction History** shows predicted vs actual prices on the held-out test window used to compute the reported performance metrics.

### Built Using
Python, Streamlit, Plotly, Scikit-learn, Pandas, NumPy, ReportLab

<div class="disclaimer">Disclaimer: this tool is for educational and informational purposes only and is not financial advice.</div>
""", unsafe_allow_html=True)

st.markdown(
    """<div style="text-align:center;color:#7C8AA0;font-size:13px;margin-top:40px;padding:16px 0;border-top:1px solid rgba(255,255,255,0.08);">
    Developed by Aditya Koushal
    </div>""",
    unsafe_allow_html=True,
)
