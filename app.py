"""
Gold and Silver Prediction
A market forecasting platform for precious metals built with Streamlit.

Run locally with:
    streamlit run app.py
"""

import os
import time
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

.block-container { padding-top: 2rem; max-width: 1400px; }

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

/* Markdown body text used for summaries and advisor output.
   Headings are forced to an explicit color + transparent background so they
   stay readable even if the surrounding theme (e.g. a dark browser/client
   theme) would otherwise render a dark heading background with dark text. */
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #2E271F !important;
    background: transparent !important;
}
.stMarkdown h3 { font-size: 22px; }
.stMarkdown ul, .stMarkdown li { font-size: 18px; }

/* Tight wrapper used to pull the recommendation/summary text closer to the
   chart above it (reduces the default Streamlit block gap). */
.tight-block { margin-top: -18px; }
.tight-block > div:first-child { margin-top: 0; }

/* Small inline info icon with a native browser tooltip on hover. */
.info-tip {
    display: inline-block;
    margin-left: 6px;
    font-size: 13px;
    font-weight: 700;
    color: #8C6A2E;
    border: 1px solid #B8892E;
    border-radius: 50%;
    width: 16px;
    height: 16px;
    line-height: 15px;
    text-align: center;
    cursor: help;
    background: #FFFDF7;
    vertical-align: middle;
}

.app-header {
    padding: 34px 36px;
    border-radius: 4px;
    margin-bottom: 22px;
    background: linear-gradient(120deg, #EFE4CD, #F7F1E4 60%, #EFE4CD);
    border: 1px solid rgba(184,137,46,0.35);
    text-align: center;
}
.app-header h1 {
    margin: 0;
    font-size: 34px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #8C6A2E;
}
.app-header p {
    color: #6B5D46;
    margin-top: 10px;
    font-size: 16px;
    font-style: italic;
    letter-spacing: 0.02em;
}

.metric-card {
    background: #FFFDF7;
    padding: 20px 16px;
    border-radius: 4px;
    border: 1px solid rgba(184,137,46,0.3);
    text-align: center;
}
.metric-card h4 {
    color: #8C7A54;
    font-weight: 600;
    font-size: 14px;
    margin: 0 0 8px 0;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Cormorant Garamond', Georgia, serif;
}
.metric-card h2 {
    margin: 0;
    font-size: 30px;
    font-weight: 700;
    color: #2E271F;
    font-family: 'Playfair Display', Georgia, serif;
}

.live-banner {
    background: #FFFDF7;
    padding: 16px 20px;
    border-radius: 4px;
    margin-bottom: 16px;
    border: 1px solid rgba(184,137,46,0.3);
    font-size: 19px;
}

.signal-buy { color: #4C6B48; font-weight: 700; }
.signal-hold { color: #B8892E; font-weight: 700; }
.signal-sell { color: #A6493A; font-weight: 700; }

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
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: #B8892E;
    color: #FFFDF7;
    border-color: #B8892E;
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
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


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

    return {"rf": rf, "gbm": gbm}, perf


@st.cache_resource(show_spinner=False)
def train_all_models():
    featured_data = build_featured_data()
    models, performance = {}, {}
    for metal, df in featured_data.items():
        models[metal], performance[metal] = train_models_for_metal(df)

    performance_df = pd.DataFrame(performance).T.round(4)

    feature_importance = {}
    for metal in models:
        imp = pd.DataFrame({
            "Feature": FEATURE_COLUMNS,
            "Importance": models[metal]["rf"].feature_importances_,
        }).sort_values("Importance", ascending=False)
        feature_importance[metal] = imp

    return models, performance, performance_df, feature_importance


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
    font=dict(color="#2E271F", size=13, family="Georgia, 'Playfair Display', serif"),
)


def create_forecast_chart(metal, forecast, featured_data, currency):
    history = featured_data[metal].copy().tail(180)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history.index, y=history["Close"] * CURRENCIES[currency], mode="lines",
        name="Historical", line=dict(color="#B8892E", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=[history.index[-1]], y=[history["Close"].iloc[-1] * CURRENCIES[currency]], mode="markers",
        name="Current", marker=dict(size=12, color="#8C6A2E", symbol="diamond")
    ))

    if "SMA_20" in history.columns:
        fig.add_trace(go.Scatter(
            x=history.index, y=history["SMA_20"] * CURRENCIES[currency], mode="lines",
            name="SMA 20", line=dict(color="#A08B63", width=2)
        ))

    if "SMA_50" in history.columns:
        fig.add_trace(go.Scatter(
            x=history.index, y=history["SMA_50"] * CURRENCIES[currency], mode="lines",
            name="SMA 50", line=dict(color="#5C4A32", width=2)
        ))

    fig.add_trace(go.Scatter(
        x=forecast["Date"], y=forecast["Forecast"] * CURRENCIES[currency], mode="lines+markers",
        name="Forecast",
        line=dict(color="#4C6B48", width=3, dash="dash"),
        marker=dict(size=7)
    ))

    fig.add_trace(go.Scatter(
        x=[forecast["Date"].iloc[0]], y=[forecast["Forecast"].iloc[0] * CURRENCIES[currency]],
        mode="markers", name="Forecast Start",
        marker=dict(size=12, color="#4C6B48", symbol="star")
    ))

    fig.update_layout(
        title=f"{metal} Price Forecast",
        hovermode="x unified", height=580,
        legend=dict(orientation="h", y=1.05, x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        **CHART_TEMPLATE,
    )
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EFE4CD", title="Price")

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
    if "BUY" in signal:
        return "signal-buy"
    if "SELL" in signal:
        return "signal-sell"
    return "signal-hold"


def metric_card(title, value, help_text=None):
    tooltip = f'<span class="info-tip" title="{help_text}">i</span>' if help_text else ""
    st.markdown(
        f"""<div class="metric-card"><h4>{title}{tooltip}</h4><h2>{value}</h2></div>""",
        unsafe_allow_html=True,
    )


CONFIDENCE_EXPLANATION = (
    "Confidence is not the same as directional accuracy. It is a weighted "
    "blend: 70% x directional accuracy (how often the model got the "
    "up/down direction right) + 30% x R2 of price fit (how well predicted "
    "prices track actual prices). A model can have modest directional "
    "accuracy (e.g. ~48%) but a high R2 on price level, which pulls the "
    "blended confidence score higher. See the Analytics tab for the raw "
    "Directional Accuracy % and R2 figures."
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

@st.cache_data(ttl=60, show_spinner=False)
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


def render_live_banner(currency):
    market = get_live_market()

    def fmt(usd_price):
        if isinstance(usd_price, str):
            return usd_price
        return f"{round(usd_price * CURRENCIES[currency], 2)}"

    gold_display = fmt(market["Gold"])
    silver_display = fmt(market["Silver"])

    st.markdown(
        f"""
        <div class="live-banner">
            <div style="display:flex;justify-content:space-around;flex-wrap:wrap;gap:10px;font-size:16px;">
                <div>Gold<br><span style="color:#E8C56A;font-weight:700;">{currency} {gold_display}</span></div>
                <div>Silver<br><span style="color:#CFD8DC;font-weight:700;">{currency} {silver_display}</span></div>
                <div>USD/INR<br><span style="color:#42A5F5;font-weight:700;">{market['USDINR']} <small style="color:#6b7688;">(static)</small></span></div>
                <div>Updated<br>{market['Time']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
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
**Confidence:** {stats['Confidence']}%  &mdash; a weighted blend of directional accuracy and price R2, *not* the same as raw model accuracy (hover the &#9432; on the Confidence card, or see the Analytics tab, for the breakdown)
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


def check_price_alert(metal, target_price, currency, models, featured_data, performance):
    m = dashboard_metrics(metal, currency, models, featured_data, performance)
    current_price = m["Current Price"]
    diff = current_price - target_price

    if current_price >= target_price:
        status, message = "ALERT TRIGGERED", f"{metal} has reached or crossed your target price."
    else:
        status, message = "ALERT NOT TRIGGERED", f"{metal} is still below your target price."

    return pd.DataFrame({
        "Metal": [metal], "Current Price": [round(current_price, 2)],
        "Target Price": [round(target_price, 2)], "Difference": [round(diff, 2)],
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

st.markdown(
    f"""
    <div class="app-header">
        <h1>{PROJECT_NAME}</h1>
        <p>An AI-assisted forecasting platform for gold and silver markets</p>
    </div>
    """,
    unsafe_allow_html=True,
)

live_col1, live_col2 = st.columns([5, 1])
with live_col2:
    live_currency = st.selectbox("Live Market Currency", list(CURRENCIES.keys()), key="live_currency")
render_live_banner(live_currency)

with st.spinner("Loading market data and training models..."):
    market_data = load_market_data()
    featured_data = build_featured_data(market_data)
    models, performance, performance_df, feature_importance = train_all_models()

tab_forecast, tab_advisor, tab_news, tab_reports, tab_analytics, tab_about = st.tabs(
    ["Forecast", "Advisor", "Market News", "Reports", "Analytics", "About"]
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

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Current Price", f"{currency} {convert_price_unit(stats['Current Price'], price_unit)} {get_unit_symbol(price_unit)}")
    with c2:
        metric_card("Forecast Price", f"{currency} {convert_price_unit(stats['Forecast Price'], price_unit)} {get_unit_symbol(price_unit)}")
    with c3:
        metric_card("Confidence", f"{stats['Confidence']}%", help_text=CONFIDENCE_EXPLANATION)
    with c4:
        st.markdown(
            f"""<div class="metric-card"><h4>Recommendation</h4>
            <h2 class="{signal_css_class(stats['Signal'])}">{stats['Signal']}</h2></div>""",
            unsafe_allow_html=True,
        )

    st.plotly_chart(create_forecast_chart(metal, forecast, featured_data, currency), use_container_width=True)

    st.markdown(
        f'<div class="tight-block">\n\n{ai_summary(metal, currency, price_unit, stats)}\n\n</div>',
        unsafe_allow_html=True,
    )

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
    st.subheader("Price Alert")
    al1, al2, al3 = st.columns(3)
    with al1:
        alert_metal = st.selectbox("Metal", ["Gold", "Silver"], key="alert_metal")
    with al2:
        alert_target = st.number_input("Target Price", value=350000, key="alert_target")
    with al3:
        alert_currency = st.selectbox("Currency", list(CURRENCIES.keys()), index=list(CURRENCIES.keys()).index("INR"), key="alert_currency")

    if st.button("Check Alert"):
        table = check_price_alert(alert_metal, alert_target, alert_currency, models, featured_data, performance)
        st.dataframe(table, use_container_width=True)

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
