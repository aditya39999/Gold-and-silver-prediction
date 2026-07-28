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



# ============================================================
# V3 CINEMATIC INTRO — DATA -> METAL TRANSFORMATION
# Replaces the old doorway transition with a continuous:
# market noise -> Gold/Silver formation -> forecast beam reveal.
# ============================================================
def render_data_to_metal_intro():
    import streamlit.components.v1 as components

    if st.session_state.get("_data_metal_intro_seen", False):
        return

    intro_html = r"""
    <div id="pm-intro">
      <canvas id="pm-canvas"></canvas>

      <button id="pm-skip" aria-label="Skip intro">Skip Intro</button>

      <div class="pm-noise" id="pm-noise">
        <span style="--x:8%;--y:18%;--d:0s">4047.32</span>
        <span style="--x:20%;--y:72%;--d:.7s">RSI 45.8</span>
        <span style="--x:31%;--y:29%;--d:1.4s">+0.17%</span>
        <span style="--x:71%;--y:17%;--d:.4s">58.47</span>
        <span style="--x:84%;--y:67%;--d:1.8s">MACD +</span>
        <span style="--x:63%;--y:79%;--d:1.1s">SMA 50</span>
        <span style="--x:47%;--y:13%;--d:2.1s">VOL 18.4</span>
        <span style="--x:91%;--y:38%;--d:.9s">FORECAST</span>
        <span style="--x:13%;--y:46%;--d:1.6s">30D</span>
      </div>

      <div class="pm-object pm-gold" id="pm-gold">
        <div class="pm-shine"></div>
        <div class="pm-metal-label">AU</div>
      </div>

      <div class="pm-object pm-silver" id="pm-silver">
        <div class="pm-orbit"></div>
        <div class="pm-metal-label">AG</div>
      </div>

      <main class="pm-copy" id="pm-copy">
        <div class="pm-kicker">PRECIOUS METALS • MACHINE INTELLIGENCE</div>
        <h1><span>FROM MARKET NOISE</span><b>TO MACHINE INTELLIGENCE</b></h1>
        <p>Live market structure transformed into AI-assisted Gold & Silver forecasts.</p>

        <div class="pm-live">
          <div><small>GOLD</small><strong>$4,074.50</strong><em>▲ +0.17%</em></div>
          <i></i>
          <div><small>SILVER</small><strong>$58.47</strong><em class="down">▼ −0.31%</em></div>
        </div>

        <button id="pm-enter">
          <span>ENTER MARKET</span>
          <svg viewBox="0 0 24 24"><path d="M5 12h13M13 6l6 6-6 6"/></svg>
        </button>
        <div class="pm-caption">DATA → SIGNAL → MODEL → FUTURE</div>
      </main>

      <div class="pm-beam" id="pm-beam">
        <svg viewBox="0 0 1200 260" preserveAspectRatio="none">
          <path id="beam-path" d="M0,165 C110,150 150,188 245,137 S390,175 475,112 S620,145 705,96 S845,120 920,72 S1065,100 1200,34"/>
        </svg>
      </div>
      <div class="pm-flash" id="pm-flash"></div>
    </div>

    <style>
      html,body{margin:0;background:#080806;overflow:hidden}
      #pm-intro{
        position:fixed;inset:0;z-index:2147483647;overflow:hidden;
        background:
          radial-gradient(circle at 28% 48%,rgba(198,145,35,.16),transparent 27%),
          radial-gradient(circle at 74% 47%,rgba(206,214,224,.10),transparent 26%),
          linear-gradient(135deg,#050504 0%,#0b0a07 48%,#050505 100%);
        color:#f4ead5;font-family:Arial,sans-serif;
      }
      #pm-intro:before{
        content:"";position:absolute;inset:0;opacity:.13;pointer-events:none;
        background-image:linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
                         linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px);
        background-size:48px 48px;
        mask-image:radial-gradient(circle at center,#000,transparent 78%);
      }
      #pm-canvas{position:absolute;inset:0;width:100%;height:100%}
      #pm-skip{
        position:absolute;right:28px;top:24px;z-index:20;padding:10px 18px;border-radius:999px;
        border:1px solid rgba(216,173,78,.42);background:rgba(10,10,8,.35);color:#d9c9a4;
        letter-spacing:.08em;cursor:pointer;backdrop-filter:blur(12px)
      }
      .pm-noise span{
        position:absolute;left:var(--x);top:var(--y);font:600 11px/1 monospace;color:#c9a24b;
        opacity:.18;letter-spacing:.12em;animation:noiseFloat 5s ease-in-out var(--d) infinite alternate;
      }
      @keyframes noiseFloat{to{transform:translate3d(18px,-24px,0);opacity:.48}}
      .pm-copy{
        position:absolute;left:50%;top:50%;width:min(760px,78vw);transform:translate(-50%,-50%);
        text-align:center;z-index:8;transition:opacity .55s ease,transform .8s cubic-bezier(.2,.8,.2,1)
      }
      .pm-kicker{font-size:11px;letter-spacing:.36em;color:#bba36e;margin-bottom:22px}
      .pm-copy h1{margin:0;font-family:Georgia,serif;font-weight:400;line-height:.95}
      .pm-copy h1 span{display:block;font-size:clamp(31px,4.8vw,70px);color:#f3ead7;letter-spacing:.02em}
      .pm-copy h1 b{
        display:block;margin-top:10px;font-size:clamp(33px,5.1vw,76px);font-weight:400;
        background:linear-gradient(90deg,#8f6517,#f5d87c,#b57d1d,#f2e2ad);
        -webkit-background-clip:text;color:transparent;background-size:200% auto;animation:goldFlow 4s linear infinite
      }
      @keyframes goldFlow{to{background-position:200% center}}
      .pm-copy p{margin:25px auto 22px;color:#a9a394;font-size:15px;letter-spacing:.04em}
      .pm-live{display:flex;justify-content:center;align-items:center;gap:25px;margin:26px auto 30px}
      .pm-live div{display:grid;gap:5px;min-width:130px}
      .pm-live small{font-size:9px;letter-spacing:.25em;color:#8d8779}
      .pm-live strong{font-family:Georgia,serif;font-size:19px;font-weight:400;color:#efe6d3}
      .pm-live em{font-style:normal;font-size:10px;color:#7ba17a}.pm-live em.down{color:#b77468}
      .pm-live i{width:1px;height:35px;background:linear-gradient(transparent,#806b3e,transparent)}
      #pm-enter{
        position:relative;display:inline-flex;align-items:center;gap:16px;padding:15px 28px;
        border:1px solid #8f6a20;border-radius:999px;color:#f2dfae;background:rgba(171,119,20,.11);
        letter-spacing:.18em;font-size:11px;cursor:pointer;overflow:hidden;
        box-shadow:0 0 45px rgba(185,132,26,.09);transition:.35s ease
      }
      #pm-enter:before{content:"";position:absolute;inset:-1px;transform:translateX(-110%);
        background:linear-gradient(90deg,transparent,rgba(244,211,123,.22),transparent);transition:.65s}
      #pm-enter:hover{transform:translateY(-2px);box-shadow:0 0 55px rgba(196,145,37,.2)}
      #pm-enter:hover:before{transform:translateX(110%)}
      #pm-enter svg{width:16px;fill:none;stroke:currentColor;stroke-width:1.5}
      .pm-caption{margin-top:15px;color:#625b4c;font-size:8px;letter-spacing:.32em}
      .pm-object{position:absolute;z-index:5;opacity:0;filter:blur(12px);transition:1.2s cubic-bezier(.2,.8,.2,1)}
      .pm-gold{
        width:175px;height:108px;left:8%;top:50%;border-radius:17px;transform:translateY(-50%) rotate(-9deg) scale(.55);
        background:linear-gradient(135deg,#6f4306,#dcae43 32%,#fff0a8 48%,#a66b0e 72%,#efc75f);
        box-shadow:inset 0 1px 12px #fff0a0,0 35px 90px rgba(201,144,29,.18)
      }
      .pm-silver{
        width:135px;height:135px;right:10%;top:50%;border-radius:50%;transform:translateY(-50%) scale(.55);
        background:radial-gradient(circle at 32% 28%,#fff,#cdd2d5 18%,#626970 53%,#e9ecee 72%,#747b80);
        box-shadow:inset -18px -20px 40px rgba(0,0,0,.42),0 35px 90px rgba(190,200,210,.12)
      }
      .pm-metal-label{position:absolute;inset:0;display:grid;place-items:center;font:500 24px Georgia,serif;color:rgba(20,17,10,.58);letter-spacing:.12em}
      .pm-shine{position:absolute;inset:0;border-radius:inherit;background:linear-gradient(110deg,transparent 20%,rgba(255,255,255,.55) 43%,transparent 61%);transform:translateX(-100%);animation:shine 3.6s 1.3s infinite}
      @keyframes shine{70%,100%{transform:translateX(120%)}}
      .pm-orbit{position:absolute;inset:-17px;border:1px solid rgba(215,221,225,.22);border-radius:50%;animation:orbit 6s linear infinite}
      .pm-orbit:after{content:"";position:absolute;width:5px;height:5px;border-radius:50%;background:#e8ecee;top:14px;left:15px;box-shadow:0 0 14px #fff}
      @keyframes orbit{to{transform:rotate(360deg)}}
      #pm-intro.ready .pm-object{opacity:.95;filter:blur(0)}
      #pm-intro.ready .pm-gold{transform:translateY(-50%) rotate(-9deg) scale(1)}
      #pm-intro.ready .pm-silver{transform:translateY(-50%) scale(1)}
      .pm-beam{position:absolute;left:0;right:0;bottom:11%;height:260px;opacity:0;z-index:12;pointer-events:none}
      .pm-beam svg{width:100%;height:100%;overflow:visible}
      #beam-path{fill:none;stroke:url(#x);stroke:#d8aa42;stroke-width:2;stroke-linecap:round;stroke-dasharray:1500;stroke-dashoffset:1500;filter:drop-shadow(0 0 8px rgba(229,180,65,.65))}
      .pm-flash{position:absolute;inset:0;z-index:30;pointer-events:none;background:#f4dfaa;opacity:0}
      #pm-intro.launch .pm-copy{opacity:0;transform:translate(-50%,-47%) scale(.96)}
      #pm-intro.launch .pm-noise{opacity:0;transition:.4s}
      #pm-intro.launch .pm-gold{left:44%;top:58%;transform:translate(-50%,-50%) rotate(18deg) scale(.08);opacity:0;filter:blur(8px)}
      #pm-intro.launch .pm-silver{right:44%;top:58%;transform:translate(50%,-50%) scale(.08);opacity:0;filter:blur(8px)}
      #pm-intro.launch .pm-beam{opacity:1;transition:opacity .25s .35s}
      #pm-intro.launch #beam-path{animation:drawBeam 1.35s .38s cubic-bezier(.2,.7,.2,1) forwards}
      @keyframes drawBeam{to{stroke-dashoffset:0}}
      #pm-intro.exit{animation:introExit .85s ease forwards}
      #pm-intro.exit .pm-flash{animation:flash .7s ease forwards}
      @keyframes flash{35%{opacity:.22}100%{opacity:0}}
      @keyframes introExit{65%{opacity:1;transform:scale(1.015)}100%{opacity:0;visibility:hidden;transform:scale(1.035)}}
      @media(max-width:900px){.pm-object{display:none}.pm-copy{width:88vw}.pm-copy p{font-size:13px}}
      @media(prefers-reduced-motion:reduce){*{animation-duration:.01ms!important;transition-duration:.01ms!important}}
    </style>

    <script>
      (() => {
        const root = document.getElementById('pm-intro');
        const canvas = document.getElementById('pm-canvas');
        const ctx = canvas.getContext('2d');
        let W=0,H=0,dpr=Math.min(devicePixelRatio||1,2), raf;
        const particles=[];
        function resize(){
          W=innerWidth;H=innerHeight;canvas.width=W*dpr;canvas.height=H*dpr;
          canvas.style.width=W+'px';canvas.style.height=H+'px';ctx.setTransform(dpr,0,0,dpr,0,0)
        }
        function seed(){
          particles.length=0;
          const n=Math.min(180,Math.floor(W/8));
          for(let i=0;i<n;i++) particles.push({
            x:Math.random()*W,y:Math.random()*H,
            vx:(Math.random()-.5)*.16,vy:(Math.random()-.5)*.16,
            r:Math.random()*1.35+.25,a:Math.random()*.42+.06,
            metal:Math.random()>.52
          });
        }
        function draw(){
          ctx.clearRect(0,0,W,H);
          for(const p of particles){
            p.x+=p.vx;p.y+=p.vy;
            if(p.x<0||p.x>W)p.vx*=-1;if(p.y<0||p.y>H)p.vy*=-1;
            ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
            ctx.fillStyle=p.metal?`rgba(220,171,61,${p.a})`:`rgba(206,214,220,${p.a*.72})`;ctx.fill();
          }
          raf=requestAnimationFrame(draw);
        }
        resize();seed();draw();addEventListener('resize',()=>{resize();seed()});
        setTimeout(()=>root.classList.add('ready'),250);

        function leave(){
          if(root.classList.contains('launch')) return;
          root.classList.add('launch');
          setTimeout(()=>root.classList.add('exit'),1750);
          setTimeout(()=>{
            cancelAnimationFrame(raf);
            root.remove();
            try{
              window.parent.postMessage({type:'pm_intro_complete'}, '*');
            }catch(e){}
          },2550);
        }
        document.getElementById('pm-enter').addEventListener('click',leave);
        document.getElementById('pm-skip').addEventListener('click',()=>{
          root.classList.add('exit');
          setTimeout(()=>{cancelAnimationFrame(raf);root.remove()},800);
        });
      })();
    </script>
    """
    components.html(intro_html, height=900, scrolling=False)

    # Streamlit reruns can otherwise replay an intro endlessly. Mark it seen
    # for this browser session after the component is mounted.
    st.session_state["_data_metal_intro_seen"] = True


render_data_to_metal_intro()
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

/* ==========================================================
   DASHBOARD V2 — compact command-centre layout
   ========================================================== */
[data-testid="stAppViewBlockContainer"],
[data-testid="stMainBlockContainer"] {
    opacity: 1 !important;
    filter: none !important;
}

/* Sticky primary navigation */
.stTabs [data-baseweb="tab-list"] {
    position: sticky;
    top: 2.9rem;
    z-index: 900;
    background: rgba(247,241,228,0.94);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    padding: 8px 10px;
    border: 1px solid rgba(184,137,46,.24);
    border-radius: 14px;
    box-shadow: 0 8px 24px rgba(46,39,31,.06);
    gap: 3px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding-left: 14px !important;
    padding-right: 14px !important;
}
.stTabs [data-baseweb="tab"]:nth-last-child(2) {
    background: linear-gradient(135deg,#2A241B,#171714);
    border: 1px solid #B8892E;
}
.stTabs [data-baseweb="tab"]:nth-last-child(2) p {
    color: #E8C56A !important;
    font-weight: 800 !important;
    letter-spacing: .08em;
}

/* Forecast command surface */
.forecast-command-shell {
    background: linear-gradient(135deg,rgba(255,253,247,.96),rgba(239,228,205,.78));
    border: 1px solid rgba(184,137,46,.35);
    border-radius: 18px;
    padding: 15px 18px 7px;
    box-shadow: 0 12px 35px rgba(46,39,31,.07);
    margin: 10px 0 12px;
}
.forecast-command-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .22em;
    color: #8C6A2E;
    font-weight: 800;
}

/* Compact market pulse strip */
.market-pulse-strip {
    display:grid;
    grid-template-columns:1.15fr 1.15fr .9fr .9fr;
    gap:10px;
    background:#171714;
    color:#F7F1E4;
    border:1px solid rgba(232,197,106,.38);
    border-radius:16px;
    padding:12px 16px;
    margin:8px 0 14px;
    box-shadow:0 10px 28px rgba(23,23,20,.12);
}
.market-pulse-strip .pulse-kicker {
    color:#BDAE8D;font-size:10px;letter-spacing:.16em;text-transform:uppercase;
}
.market-pulse-strip .pulse-value {
    font-family:'Playfair Display',Georgia,serif;font-size:18px;font-weight:700;margin-top:2px;
}
.market-pulse-strip .gold-value {color:#E8C56A}
.market-pulse-strip .silver-value {color:#D8DCE2}

/* Hero prediction */
.prediction-hero {
    background:#FFFDF7;
    border:1px solid rgba(184,137,46,.34);
    border-top:3px solid #B8892E;
    border-radius:14px;
    min-height:195px;
    padding:22px 24px;
    box-shadow:0 10px 28px rgba(46,39,31,.06);
}
.prediction-hero .eyebrow {
    font-size:11px;letter-spacing:.18em;text-transform:uppercase;color:#8C7A54;font-weight:800;
}
.prediction-flow {
    display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:18px 0 10px;
}
.prediction-flow .price {
    font-family:'Playfair Display',Georgia,serif;font-size:29px;font-weight:700;color:#2E271F;
}
.prediction-flow .arrow {font-size:25px;color:#B8892E}
.prediction-meta {display:flex;gap:8px;flex-wrap:wrap;margin-top:13px}
.prediction-chip {
    border-radius:999px;padding:6px 10px;font-size:12px;font-weight:800;
    background:#F2E6D8;color:#8E382C;
}
.prediction-chip.dark {background:#171714;color:#E8C56A}
.prediction-chip.soft {background:#ECEDE7;color:#355E3B}

/* AI signal card */
.ai-signal-card {
    min-height:195px;
    border-radius:14px;
    padding:22px 22px;
    color:#F7F1E4;
    background:
      radial-gradient(circle at 85% 15%,rgba(184,137,46,.23),transparent 32%),
      linear-gradient(145deg,#24211C,#171714);
    border:1px solid rgba(232,197,106,.42);
    box-shadow:0 14px 34px rgba(23,23,20,.16);
}
.ai-signal-card .eyebrow {color:#BDAE8D;font-size:11px;letter-spacing:.18em;text-transform:uppercase}
.ai-signal-card .signal {font-family:'Playfair Display',Georgia,serif;font-size:38px;color:#E8C56A;margin:10px 0 4px}
.ai-signal-card .sub {font-size:13px;color:#D7CCB5;line-height:1.55}
.ai-signal-card .beam {
    height:5px;border-radius:999px;background:#34322D;margin:16px 0 8px;overflow:hidden;
}
.ai-signal-card .beam > span {
    display:block;height:100%;border-radius:999px;
    background:linear-gradient(90deg,#8C6A2E,#E8C56A);
}

/* Advanced / Future Lab terminal layer */
.future-lab-shell {
    background:#171714;color:#F7F1E4;border:1px solid #8C6A2E;
    border-radius:18px;padding:18px 22px;margin:8px 0 16px;
    box-shadow:0 18px 50px rgba(23,23,20,.18);
}
.future-lab-shell h2 {color:#E8C56A !important;margin:0}
.future-lab-shell p {color:#CFC4AE !important}

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
    /* Do not animate/filter the Streamlit app container itself.
       This prevents reruns from leaving dashboard content faded or blurred. */
    [data-testid="stAppViewBlockContainer"] {{
        filter: none !important;
        opacity: 1 !important;
        transform: none !important;
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
    # Previous intro disabled in v3
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
# FUTURE LAB — ADVANCED INTERACTIVE MARKET EXPERIENCE
# ==========================================================

def scenario_paths(metal, days, models, featured_data, inflation=0.0, usd_strength=0.0,
                   rates=0.0, volatility_mult=1.0, correlation=0.5):
    """Digital-twin scenario layer built around the model's base forecast."""
    base = forecast_prices(metal, days, models, featured_data).copy()
    current = float(featured_data[metal]["Close"].iloc[-1])
    t = np.arange(1, days + 1) / max(days, 1)

    # Assumption sensitivities are deliberately transparent scenario controls,
    # not claims that these macro variables are direct features of the trained model.
    macro = (0.020 * inflation - 0.025 * usd_strength - 0.018 * rates) * t
    vol = float(featured_data[metal]["LogReturn"].tail(60).std()) * max(volatility_mult, 0.1)
    corr_effect = (correlation - 0.5) * (0.008 if metal == "Silver" else 0.004) * t

    base_price = base["Forecast"].values * np.exp(macro + corr_effect)
    bull = base_price * np.exp((0.65 * vol * np.sqrt(np.arange(1, days + 1))) + 0.018 * t)
    bear = base_price * np.exp((-0.65 * vol * np.sqrt(np.arange(1, days + 1))) - 0.018 * t)
    stress = base_price * np.exp((-1.35 * vol * np.sqrt(np.arange(1, days + 1))) - 0.035 * t)

    return pd.DataFrame({
        "Date": base["Date"], "Base": base_price, "Bull": bull,
        "Bear": bear, "Stress": stress, "Current": current
    })


def create_scenario_chart(paths, currency):
    fig = go.Figure()
    palette = {"Base": "#B8892E", "Bull": "#4C6B48", "Bear": "#A6493A", "Stress": "#5C4A32"}
    for name in ["Base", "Bull", "Bear", "Stress"]:
        fig.add_trace(go.Scatter(
            x=paths["Date"], y=paths[name] * CURRENCIES[currency],
            mode="lines", name=name, line=dict(width=3 if name == "Base" else 2, color=palette[name])
        ))
    fig.update_layout(height=480, hovermode="x unified", margin=dict(l=45,r=20,t=55,b=20),
                      title="Digital Twin — Scenario Universe", **CHART_TEMPLATE)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EFE4CD", title="Price")
    return fig


def monte_carlo_universe(metal, days, featured_data, models, simulations=300, seed=42):
    base = forecast_prices(metal, days, models, featured_data)
    base_prices = base["Forecast"].to_numpy(float)
    current = float(featured_data[metal]["Close"].iloc[-1])
    base_returns = np.diff(np.log(np.r_[current, base_prices]))
    hist_vol = float(featured_data[metal]["LogReturn"].tail(90).std())
    rng = np.random.default_rng(seed)

    paths = np.zeros((simulations, days))
    for i in range(simulations):
        shocks = rng.normal(0, hist_vol, days)
        simulated_returns = base_returns + shocks
        paths[i] = current * np.exp(np.cumsum(simulated_returns))
    return base["Date"], paths


def create_monte_carlo_chart(dates, paths, currency):
    fig = go.Figure()
    # Draw a representative sample to keep browser rendering fast.
    sample_n = min(100, paths.shape[0])
    for row in paths[:sample_n]:
        fig.add_trace(go.Scatter(
            x=dates, y=row * CURRENCIES[currency], mode="lines",
            line=dict(width=0.7, color="rgba(140,106,46,0.10)"),
            hoverinfo="skip", showlegend=False
        ))
    p10, p50, p90 = np.percentile(paths, [10, 50, 90], axis=0)
    fig.add_trace(go.Scatter(x=dates, y=p90*CURRENCIES[currency], mode="lines",
                             line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=dates, y=p10*CURRENCIES[currency], mode="lines",
                             fill="tonexty", fillcolor="rgba(184,137,46,0.18)",
                             line=dict(width=0), name="10–90% Probability Cloud"))
    fig.add_trace(go.Scatter(x=dates, y=p50*CURRENCIES[currency], mode="lines",
                             name="Median Future", line=dict(color="#8C6A2E", width=3)))
    fig.update_layout(height=500, title="Monte Carlo Future Universe", hovermode="x unified",
                      margin=dict(l=45,r=20,t=55,b=20), **CHART_TEMPLATE)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EFE4CD", title="Price")
    return fig


def render_prediction_beam(metal, featured_data, performance):
    row = featured_data[metal].iloc[-1]
    rsi = float(row["RSI"])
    macd_state = "BULLISH" if row["MACD"] > row["MACD_SIGNAL"] else "BEARISH"
    acc = performance[metal]["Directional Accuracy %"]
    st.markdown(f"""
    <style>
    .beam-wrap{{background:radial-gradient(circle at center,#FFF8DE,#2E271F 72%);
      border:1px solid rgba(184,137,46,.45);border-radius:18px;padding:32px 18px;overflow:hidden}}
    .beam-flow{{display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap}}
    .beam-node{{min-width:120px;padding:16px 12px;text-align:center;border-radius:14px;
      background:rgba(255,253,247,.94);border:1px solid #B8892E;box-shadow:0 0 25px rgba(232,197,106,.18);
      animation:beamPulse 2.2s ease-in-out infinite}}
    .beam-node b{{display:block;color:#8C6A2E;font-family:'Playfair Display',serif;font-size:14px}}
    .beam-node small{{color:#6B5D46}}
    .beam-line{{height:2px;min-width:35px;flex:1;background:linear-gradient(90deg,transparent,#E8C56A,transparent);
      background-size:200% 100%;animation:beamMove 1.2s linear infinite;box-shadow:0 0 12px #E8C56A}}
    @keyframes beamMove{{to{{background-position:-200% 0}}}}
    @keyframes beamPulse{{50%{{transform:translateY(-4px);box-shadow:0 0 35px rgba(232,197,106,.38)}}}}
    </style>
    <div class="beam-wrap"><div class="beam-flow">
      <div class="beam-node"><b>LIVE DATA</b><small>{metal} futures</small></div><div class="beam-line"></div>
      <div class="beam-node"><b>INDICATORS</b><small>RSI {rsi:.0f} · {macd_state}</small></div><div class="beam-line"></div>
      <div class="beam-node"><b>MODEL CORE</b><small>RF + GBM</small></div><div class="beam-line"></div>
      <div class="beam-node"><b>CONFIDENCE</b><small>{acc:.1f}% direction</small></div><div class="beam-line"></div>
      <div class="beam-node"><b>FUTURE PRICE</b><small>recursive forecast</small></div>
    </div></div>
    """, unsafe_allow_html=True)


def create_model_battle(metal, days, models, featured_data):
    """Race RF, GBM and blend forward using the same recursive feature updates."""
    history = featured_data[metal].copy()
    outputs = {}
    for mode in ["Random Forest", "Gradient Boosting", "Blend"]:
        window = history.tail(INDICATOR_WINDOW + days).copy()
        base_cols = ["Open","High","Low","Close","Volume"]
        vals, dates = [], []
        d = window.index[-1]
        for _ in range(days):
            x = window.iloc[-1:][FEATURE_COLUMNS]
            if mode == "Random Forest":
                pred = float(models[metal]["rf"].predict(x)[0])
            elif mode == "Gradient Boosting":
                pred = float(models[metal]["gbm"].predict(x)[0])
            else:
                pred = predict_next_return(models[metal], x)
            price = float(window["Close"].iloc[-1]) * np.exp(pred)
            d += timedelta(days=1)
            dates.append(d); vals.append(price)
            nr = pd.DataFrame({"Open":[price],"High":[price*1.002],"Low":[price*.998],
                               "Close":[price],"Volume":[float(window["Volume"].tail(10).mean())]}, index=[d])
            window = pd.concat([window[base_cols], nr[base_cols]])
            window = add_indicators(window.astype(float)).ffill().bfill()
        outputs[mode] = vals
    return pd.DataFrame({"Date": dates, **outputs})


def time_machine_backtest(metal, cutoff, days, featured_data):
    """Historical reveal: simple walk-forward baseline using data available at cutoff.
    It intentionally avoids retraining the expensive ensemble on every UI rerun."""
    full = featured_data[metal]
    cutoff = pd.Timestamp(cutoff)
    past = full.loc[full.index <= cutoff]
    future = full.loc[full.index > cutoff].head(days)
    if len(past) < 80 or future.empty:
        return None
    # Trend/volatility projection frozen at cutoff, then reveal actual reality.
    recent = past["LogReturn"].tail(30)
    drift = float(recent.mean())
    start = float(past["Close"].iloc[-1])
    pred = start * np.exp(np.cumsum(np.repeat(drift, len(future))))
    return pd.DataFrame({"Date": future.index, "Prediction": pred, "Reality": future["Close"].values})


def relationship_chart(featured_data, window=180):
    g = featured_data["Gold"]["Close"].pct_change()
    s = featured_data["Silver"]["Close"].pct_change()
    joined = pd.concat([g.rename("Gold"), s.rename("Silver")], axis=1).dropna().tail(window)
    rolling_corr = joined["Gold"].rolling(30).corr(joined["Silver"])
    ratio = (featured_data["Gold"]["Close"] / featured_data["Silver"]["Close"]).dropna().tail(window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rolling_corr.index, y=rolling_corr, name="30D Correlation",
                             line=dict(color="#B8892E", width=3)))
    fig.add_trace(go.Scatter(x=ratio.index, y=(ratio-ratio.mean())/ratio.std(), name="Gold/Silver Ratio (z-score)",
                             yaxis="y2", line=dict(color="#6B7280", width=2)))
    fig.update_layout(height=430, title="Gold ↔ Silver Relationship Map",
                      yaxis=dict(title="Correlation", range=[-1,1]),
                      yaxis2=dict(title="Ratio z-score", overlaying="y", side="right"),
                      margin=dict(l=45,r=45,t=55,b=20), **CHART_TEMPLATE)
    return fig, float(rolling_corr.dropna().iloc[-1])


def prediction_dna(metal, models, featured_data):
    """Transparent local influence proxy: RF importance × standardized latest feature value.
    This is an explanatory proxy, not SHAP."""
    df = featured_data[metal]
    latest = df[FEATURE_COLUMNS].iloc[-1]
    mu = df[FEATURE_COLUMNS].tail(250).mean()
    sd = df[FEATURE_COLUMNS].tail(250).std().replace(0, np.nan)
    z = ((latest - mu) / sd).fillna(0)
    imp = pd.Series(models[metal]["rf"].feature_importances_, index=FEATURE_COLUMNS)
    influence = (imp * z).sort_values(key=np.abs, ascending=False).head(10)
    fig = go.Figure(go.Bar(
        x=influence.values, y=influence.index, orientation="h",
        marker=dict(color=["#4C6B48" if v >= 0 else "#A6493A" for v in influence.values])
    ))
    fig.update_layout(height=410, title="Prediction DNA — Local Influence Proxy",
                      margin=dict(l=110,r=20,t=55,b=20), **CHART_TEMPLATE)
    return fig


def render_3d_metal_world():
    st.markdown("""
    <style>
    .metal-world{height:320px;display:flex;justify-content:center;align-items:center;gap:12vw;
      perspective:900px;background:radial-gradient(circle at 50% 50%,rgba(232,197,106,.18),transparent 55%);
      border-radius:24px;overflow:hidden}
    .metal-orb{width:170px;height:170px;border-radius:50%;display:flex;align-items:center;justify-content:center;
      font-family:'Playfair Display',serif;font-size:25px;font-weight:700;letter-spacing:.12em;
      box-shadow:inset -25px -25px 45px rgba(0,0,0,.22),inset 18px 18px 35px rgba(255,255,255,.45),0 30px 55px rgba(46,39,31,.18);
      animation:orbFloat 4s ease-in-out infinite;transition:.5s transform,.5s box-shadow}
    .metal-orb:hover{transform:rotateY(22deg) rotateX(-12deg) scale(1.12);box-shadow:0 35px 80px rgba(184,137,46,.35)}
    .gold-orb{background:radial-gradient(circle at 30% 25%,#FFF1A8,#D6A83C 38%,#8C6A2E 75%,#4A3210);color:#FFF8DE}
    .silver-orb{background:radial-gradient(circle at 30% 25%,#FFFFFF,#D8DCE2 38%,#7A808A 75%,#3D4148);color:white;animation-delay:-2s}
    @keyframes orbFloat{50%{transform:translateY(-18px) rotateY(12deg)}}
    </style>
    <div class="metal-world">
      <div class="metal-orb gold-orb">GOLD</div>
      <div class="metal-orb silver-orb">SILVER</div>
    </div>
    """, unsafe_allow_html=True)


def render_future_lab(models, featured_data, performance, performance_df):
    st.markdown("""<div class="future-lab-shell">
      <h2>FUTURE LAB // AI MARKET SIMULATION</h2>
      <p>Digital twins, Monte Carlo futures, model battles, prediction DNA and market time travel.</p>
    </div>""", unsafe_allow_html=True)
    st.caption("Experimental market-simulation and explainability workspace. Scenario controls are educational assumptions, not financial advice.")

    render_3d_metal_world()

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        lab_metal = st.selectbox("Metal", ["Gold","Silver"], key="lab_metal")
    with c2:
        lab_currency = st.selectbox("Currency", list(CURRENCIES.keys()), key="lab_currency")
    with c3:
        lab_days = st.slider("Horizon", 7, 90, 30, key="lab_days")

    st.markdown("### AI Prediction Beam")
    render_prediction_beam(lab_metal, featured_data, performance)

    st.markdown("### Digital Twin Market Engine")
    a,b,c,d,e = st.columns(5)
    with a: inflation = st.slider("Inflation shock", -2.0, 5.0, 0.0, .25, key="twin_inf")
    with b: usd = st.slider("USD strength", -3.0, 3.0, 0.0, .25, key="twin_usd")
    with c: rates = st.slider("Rate shock", -2.0, 3.0, 0.0, .25, key="twin_rates")
    with d: vol_mult = st.slider("Volatility ×", .5, 2.5, 1.0, .1, key="twin_vol")
    with e: corr = st.slider("Gold/Silver corr.", -1.0, 1.0, .5, .1, key="twin_corr")
    paths = scenario_paths(lab_metal, lab_days, models, featured_data, inflation, usd, rates, vol_mult, corr)
    st.plotly_chart(create_scenario_chart(paths, lab_currency), use_container_width=True, key="digital_twin_chart")

    st.markdown("### Monte Carlo Future Universe")
    mc1, mc2 = st.columns([1,2])
    with mc1:
        sims = st.slider("Simulations", 100, 1000, 400, 100, key="mc_sims")
        target_default = float(featured_data[lab_metal]["Close"].iloc[-1] * 1.03 * CURRENCIES[lab_currency])
        target = st.number_input("Target price", value=round(target_default,2), key="mc_target")
    dates, universe = monte_carlo_universe(lab_metal, lab_days, featured_data, models, sims)
    probability = float(np.mean(universe[:,-1] * CURRENCIES[lab_currency] >= target) * 100)
    with mc2:
        st.metric(f"Probability {lab_metal} ≥ {lab_currency} {target:,.2f}", f"{probability:.1f}%")
    st.plotly_chart(create_monte_carlo_chart(dates, universe, lab_currency), use_container_width=True, key="mc_chart")

    st.markdown("### AI Model Battle Arena")
    battle = create_model_battle(lab_metal, lab_days, models, featured_data)
    fig = go.Figure()
    for name, color in [("Random Forest","#3B6FA0"),("Gradient Boosting","#A6493A"),("Blend","#B8892E")]:
        fig.add_trace(go.Scatter(x=battle["Date"], y=battle[name]*CURRENCIES[lab_currency],
                                 mode="lines", name=name, line=dict(width=3 if name=="Blend" else 2, color=color)))
    fig.update_layout(height=430, title="RF vs GBM vs Blended Ensemble", **CHART_TEMPLATE)
    st.plotly_chart(fig, use_container_width=True, key="battle_chart")
    st.dataframe(performance_df.loc[[lab_metal]], use_container_width=True)

    st.markdown("### Prediction DNA")
    st.plotly_chart(prediction_dna(lab_metal, models, featured_data), use_container_width=True, key="dna_chart")
    st.caption("Prediction DNA uses feature importance × latest standardized feature value as a local influence proxy; it is not SHAP attribution.")

    st.markdown("### Gold ↔ Silver Relationship Map")
    rel_fig, corr_now = relationship_chart(featured_data)
    st.plotly_chart(rel_fig, use_container_width=True, key="relationship_chart")
    if abs(corr_now) < 0.2:
        st.warning(f"DIVERGENCE DETECTED — current 30-day return correlation is {corr_now:.2f}.")
    else:
        st.info(f"Current 30-day Gold/Silver return correlation: {corr_now:.2f}")

    st.markdown("### Time Machine + Reveal Reality")
    tm1, tm2 = st.columns(2)
    min_date = featured_data[lab_metal].index.min().date() + timedelta(days=365)
    max_date = featured_data[lab_metal].index.max().date() - timedelta(days=45)
    default_date = max(min_date, max_date - timedelta(days=365))
    with tm1:
        cutoff = st.date_input("Travel to date", value=default_date, min_value=min_date, max_value=max_date, key="tm_date")
    with tm2:
        reveal = st.toggle("Reveal Reality", value=False, key="tm_reveal")
    tm = time_machine_backtest(lab_metal, cutoff, min(30, lab_days), featured_data)
    if tm is not None:
        tf = go.Figure()
        tf.add_trace(go.Scatter(x=tm["Date"], y=tm["Prediction"]*CURRENCIES[lab_currency],
                                name="Prediction from cutoff", line=dict(color="#B8892E", width=3, dash="dash")))
        if reveal:
            tf.add_trace(go.Scatter(x=tm["Date"], y=tm["Reality"]*CURRENCIES[lab_currency],
                                    name="Reality", line=dict(color="#2E271F", width=3)))
        tf.update_layout(height=400, title=f"Time Machine — {cutoff}", **CHART_TEMPLATE)
        st.plotly_chart(tf, use_container_width=True, key="time_machine_chart")

    st.markdown("### Market Replay Mode")
    replay_max = min(180, len(featured_data[lab_metal]))
    replay_step = st.slider("Replay frame", 30, replay_max, replay_max, key="replay_step")
    replay_df = featured_data[lab_metal].tail(replay_max).head(replay_step)
    rf = go.Figure(go.Scatter(x=replay_df.index, y=replay_df["Close"]*CURRENCIES[lab_currency],
                              mode="lines", line=dict(color="#B8892E", width=3), name=lab_metal))
    rf.update_layout(height=360, title="Drag the frame slider to replay market history", **CHART_TEMPLATE)
    st.plotly_chart(rf, use_container_width=True, key="replay_chart")

    st.markdown("### Command Centre Mode")
    stats = dashboard_metrics(lab_metal, lab_currency, models, featured_data, performance, lab_days)
    pulse = compute_market_pulse(lab_metal, featured_data, performance, stats["Expected Return"])
    st.markdown(f"""
    <div style="background:#211B14;color:#F7F1E4;border:1px solid #B8892E;border-radius:20px;padding:28px;
      box-shadow:0 20px 60px rgba(46,39,31,.22);">
      <div style="font-size:12px;letter-spacing:.25em;color:#E8C56A">COMMAND CENTRE / {lab_metal.upper()}</div>
      <div style="display:flex;gap:35px;flex-wrap:wrap;margin-top:16px">
        <div><small style="color:#C9B98F">CURRENT</small><div style="font-size:34px;font-family:'Playfair Display'">{lab_currency} {stats['Current Price']:,.2f}</div></div>
        <div><small style="color:#C9B98F">{lab_days}D FUTURE</small><div style="font-size:34px;font-family:'Playfair Display'">{lab_currency} {stats['Forecast Price']:,.2f}</div></div>
        <div><small style="color:#C9B98F">SIGNAL</small><div style="font-size:34px;font-family:'Playfair Display';color:#E8C56A">{stats['Signal']}</div></div>
        <div><small style="color:#C9B98F">PULSE</small><div style="font-size:34px;font-family:'Playfair Display'">{pulse['label']}</div></div>
        <div><small style="color:#C9B98F">CONFIDENCE</small><div style="font-size:34px;font-family:'Playfair Display'">{stats['Confidence']:.1f}%</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)



def render_compact_market_pulse(currency, featured_data, performance):
    """One-line market pulse used on the Forecast page to reduce vertical clutter."""
    try:
        g = featured_data["Gold"]["Close"].dropna()
        s = featured_data["Silver"]["Close"].dropna()
        gp = float(g.iloc[-1]) * CURRENCIES[currency]
        sp = float(s.iloc[-1]) * CURRENCIES[currency]
        gc = ((g.iloc[-1] / g.iloc[-2]) - 1) * 100 if len(g) > 1 else 0.0
        sc = ((s.iloc[-1] / s.iloc[-2]) - 1) * 100 if len(s) > 1 else 0.0
        avg_acc = (performance["Gold"]["Directional Accuracy %"] + performance["Silver"]["Directional Accuracy %"]) / 2
        direction = "RISK-OFF" if (gc + sc) < 0 else "RISK-ON"
        st.markdown(
            f"""<div class="market-pulse-strip">
              <div><div class="pulse-kicker">Gold / Live</div><div class="pulse-value gold-value">{currency} {gp:,.2f} &nbsp; {gc:+.2f}%</div></div>
              <div><div class="pulse-kicker">Silver / Live</div><div class="pulse-value silver-value">{currency} {sp:,.2f} &nbsp; {sc:+.2f}%</div></div>
              <div><div class="pulse-kicker">Market Regime</div><div class="pulse-value">{direction}</div></div>
              <div><div class="pulse-kicker">AI Direction Accuracy</div><div class="pulse-value">{avg_acc:.1f}%</div></div>
            </div>""",
            unsafe_allow_html=True,
        )
    except Exception:
        pass


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


with st.spinner("Loading market data and training models..."):
    market_data = load_market_data()
    featured_data = build_featured_data(market_data)
    models, performance, performance_df, feature_importance, prediction_history = train_all_models()

(
    tab_forecast, tab_pulse, tab_advisor, tab_comparison,
    tab_news, tab_reports, tab_analytics, tab_future_lab, tab_about,
) = st.tabs(
    ["Forecast", "Market Pulse", "Advisor", "Comparison", "Market News", "Reports", "Analytics", "Future Lab", "About"]
)

with tab_future_lab:
    render_future_lab(models, featured_data, performance, performance_df)

# ---------------- FORECAST TAB ----------------
with tab_forecast:
    render_compact_market_pulse(live_currency, featured_data, performance)
    st.markdown("""<div class="forecast-command-shell">
      <div class="forecast-command-title">Forecast Command Bar · Configure the AI horizon</div>
    </div>""", unsafe_allow_html=True)
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

    def _set_horizon(days):
        st.session_state["forecast_days"] = days

    q1, q2, q3, q4 = st.columns([1, 1, 1, 5])
    with q1:
        st.button("7D", key="horizon_7", use_container_width=True, on_click=_set_horizon, args=(7,))
    with q2:
        st.button("30D", key="horizon_30", use_container_width=True, on_click=_set_horizon, args=(30,))
    with q3:
        st.button("90D", key="horizon_90", use_container_width=True, on_click=_set_horizon, args=(90,))
    with q4:
        st.caption("Quick horizon · or enter any value from 7–90 days above")

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

    # Stronger hierarchy: one hero prediction + one AI signal panel.
    hero_col, signal_col = st.columns([2.15, 1])
    with hero_col:
        f_arrow, f_class = trend_arrow(stats["Expected Return"])
        momentum_text = "Bearish momentum" if stats["Expected Return"] < -0.5 else ("Bullish momentum" if stats["Expected Return"] > 0.5 else "Neutral momentum")
        st.markdown(
            f"""<div class="prediction-hero">
              <div class="eyebrow">{metal.upper()} · AI PRICE PATH · {forecast_days} DAY HORIZON</div>
              <div class="prediction-flow">
                <span class="price">{currency} {convert_price_unit(stats['Current Price'], price_unit)}</span>
                <span class="arrow">→</span>
                <span class="price">{currency} {convert_price_unit(stats['Forecast Price'], price_unit)}</span>
                <span style="font-size:13px;color:#8C7A54">{get_unit_symbol(price_unit)}</span>
              </div>
              <div class="prediction-meta">
                <span class="prediction-chip">{f_arrow} {stats['Expected Return']:+.2f}% expected</span>
                <span class="prediction-chip soft">{change_24h:+.2f}% today</span>
                <span class="prediction-chip dark">{stats['Confidence']:.2f}% confidence</span>
              </div>
            </div>""",
            unsafe_allow_html=True,
        )
    with signal_col:
        signal = stats["Signal"]
        momentum_text = "Bearish momentum" if stats["Expected Return"] < -0.5 else ("Bullish momentum" if stats["Expected Return"] > 0.5 else "Neutral momentum")
        st.markdown(
            f"""<div class="ai-signal-card">
              <div class="eyebrow">AI SIGNAL ENGINE</div>
              <div class="signal">{signal}</div>
              <div class="sub">{momentum_text}<br>Projected move: <b>{stats['Expected Return']:+.2f}%</b><br>Directional accuracy: <b>{dir_acc:.1f}%</b></div>
              <div class="beam"><span style="width:{min(stats['Confidence'],100)}%"></span></div>
              <div class="sub">Confidence {stats['Confidence']:.2f}% · {conf_label}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with st.expander("Why this recommendation?", expanded=False):
        for reason in signal_reasoning(metal, stats, featured_data, performance, forecast_days):
            st.markdown(f"- {reason}")

    st.markdown('<div class="chart-reveal">', unsafe_allow_html=True)
    st.plotly_chart(create_forecast_chart(metal, forecast, featured_data, currency), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(
        "Drag the range slider or use the buttons above the chart to zoom into a time period. "
        "The shaded green band around the forecast is an approximate 80% confidence range "
        "derived from recent price volatility."
    )


    st.markdown(ai_summary(metal, currency, price_unit, stats))

    with st.expander("View forecast data & exports", expanded=False):
        st.dataframe(forecast_display, use_container_width=True, height=320)
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "Download Forecast (CSV)",
                data=forecast_display.to_csv(index=False),
                file_name="forecast.csv",
                mime="text/csv",
                key="forecast_csv_download",
            )
        with dl2:
            excel_path = export_excel(forecast_display)
            with open(excel_path, "rb") as f:
                st.download_button(
                    "Download Forecast (Excel)",
                    data=f.read(),
                    file_name="forecast.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="forecast_excel_download",
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


# V3 note:
# The old doorway intro implementation is intentionally left in this file for rollback,
# but its render call is disabled above. The active experience is Data -> Metal.
