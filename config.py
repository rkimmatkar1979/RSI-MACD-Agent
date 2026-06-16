"""
Central configuration for the Nifty 100 Swing Trading Agent.

All tunable parameters live here so the rest of the codebase stays
declarative. Values can be overridden via environment variables / a
local .env file (see .env.example).

Sections
--------
  App / UI                  — branding, titles, footer text, dark mode
  Data & Scan               — yfinance settings, concurrency
  Technical Indicators      — RSI, MACD, Fibonacci, Volume, Sector Trend
  Shortlist Tab             — scoring weights, shortlist size/threshold
  Chart Analysis Tab        — chart display window
  AI Commentary &           — LLM endpoint, model, pick counts
    Custom Analysis Tab
  Market Hours              — NSE open/close times (informational)
  Database                  — local SQLite + optional Turso (cloud)
  Authentication (Admin)    — Google OAuth, user caps, admin emails
  Scan Universe             — Nifty 100 + Gold/Silver tickers, sector map
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_setting(key, default=""):
    """Read from env (.env locally) first, then Streamlit secrets (Cloud)."""
    val = os.getenv(key)
    if val is not None:
        return val
    try:
        return str(st.secrets[key])
    except Exception:
        return default


# ---------------------------------------------------------------------------
# App / UI
# ---------------------------------------------------------------------------
APP_TITLE = "SwingEdge"
APP_PAGE_ICON = "📈"
APP_SUBTITLE = "Nifty 100 Swing Trading Agent"
APP_TAGLINE = (
    "Mathematical screening (RSI, MACD, Fibonacci retracements) "
    "+ Grok AI commentary, tuned for 2-3 week swing setups."
)

# Set to True to show the dark mode toggle in the sidebar.
DARK_MODE_ENABLED = True
DARK_MODE_NOTICE = (
    "⚠️ Dark mode is currently under development. "
    "Some UI elements may not appear as intended."
)

AUTHOR_NAME = "Rishikesh Kimmatkar"
AUTHOR_LINKEDIN_URL = "https://www.linkedin.com/in/rishikesh-kimmatkar/"
FOOTER_ABOUT = (
    f"📈 **{APP_SUBTITLE}** — mathematical screening (RSI, MACD, "
    "Fibonacci retracements) plus AI-generated commentary, for 2-3 week swing "
    "setups on NSE-listed stocks and Gold/Silver ETFs. Price data via Yahoo "
    "Finance (yfinance); AI commentary via the configured LLM API."
)
FOOTER_DISCLAIMER = (
    "⚠️ For educational and personal use only — this is **not** investment "
    "advice. Always do your own research and consult a registered financial "
    "advisor before trading."
)

# ---------------------------------------------------------------------------
# Data & Scan
# ---------------------------------------------------------------------------
# Daily candles throughout — appropriate for a 2-3 week swing trading horizon.
DATA_PERIOD = "1y"
DATA_INTERVAL = "1d"

# yfinance calls are I/O-bound; a thread pool cuts scan time dramatically
# without overwhelming Yahoo's API.
SCAN_MAX_WORKERS = 8

# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

# RSI
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

# MACD
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
# Histogram is "near a crossover" when its absolute value has shrunk below
# this fraction of its own recent (20-bar) average AND is still shrinking.
MACD_CROSSOVER_PROXIMITY_FACTOR = 0.15

# Fibonacci retracement
# 90 trading days (~4.5 months) gives a swing-relevant peak/trough window.
FIB_LOOKBACK_DAYS = 90
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618]
FIB_KEY_LEVELS = [0.5, 0.618]          # weighted higher in scoring
FIB_PROXIMITY_PCT = 0.01               # price must be within 1% of a level

# Volume
VOLUME_AVG_WINDOW = 20                 # rolling window (trading days)
VOLUME_SURGE_RATIO = 1.5               # >= this multiple counts as a surge

# Buy / Sell pressure (volume-weighted proxy — not live order-book data)
BUY_SELL_PRESSURE_WINDOW = 20          # sessions to look back

# Sector trend
SECTOR_TREND_LOOKBACK_DAYS = 10        # ~2 trading weeks
SECTOR_TREND_THRESHOLD = 0.015         # 1.5% minimum move for bonus to apply

# ---------------------------------------------------------------------------
# Shortlist Tab
# ---------------------------------------------------------------------------
# Score weights sum to 100 for the best-case combination:
#   key Fib (30) + RSI extreme (20) + MACD proximity (28)
#   + volume surge (15) + sector trend (7) = 100
# Fibonacci and MACD are primary drivers; sector trend is secondary by design.
SCORE_FIB_KEY_LEVEL = 30
SCORE_FIB_OTHER_LEVEL = 15
SCORE_RSI_EXTREME = 20
SCORE_MACD_PROXIMITY = 28
SCORE_VOLUME = 15
SCORE_SECTOR_TREND = 7

SHORTLIST_MAX_SIZE = 8
SHORTLIST_MIN_SCORE = 20               # stocks below this score are excluded

# ---------------------------------------------------------------------------
# Chart Analysis Tab
# ---------------------------------------------------------------------------
# Plots only the most recent N months of candles/volume/MACD. Fibonacci
# levels are still computed from the full FIB_LOOKBACK_DAYS window and
# drawn across this shorter view.
CHART_DISPLAY_MONTHS = 3

# ---------------------------------------------------------------------------
# AI Commentary & Custom Analysis Tab
# ---------------------------------------------------------------------------
# Any OpenAI-compatible chat completions endpoint works (Groq, xAI,
# OpenRouter, GitHub Models, ...) — ai_analyst.py uses the standard
# {"model", "messages", "temperature"} shape.
LLM_API_KEY = _get_setting("LLM_API_KEY", "")
LLM_API_URL = _get_setting("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
LLM_MODEL = _get_setting("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT_SECONDS = 60

# How many top-ranked shortlist entries get a full Entry/SL/TP write-up.
AI_TOP_PICKS_COUNT = 8
# Recent news headlines fetched per ticker (via yfinance) for AI context.
NEWS_HEADLINE_COUNT = 3
# Minimum tickers required for a Custom Analysis request (keeps each LLM
# call covering enough stocks to be worthwhile). Capped at AI_TOP_PICKS_COUNT.
CUSTOM_ANALYSIS_MIN_TICKERS = 5

# ---------------------------------------------------------------------------
# Market Hours  (informational — agent runs on demand, not on a schedule)
# ---------------------------------------------------------------------------
MARKET_TIMEZONE = "Asia/Kolkata"
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_agent.db")

# Retain only the N most recent scan dates; older rows are pruned in
# save_scan_results(). Bounds DB growth while still giving the day-over-day
# diff and score-history sparkline a few sessions to compare against.
SCAN_HISTORY_RETENTION_DAYS = 5

# Optional Turso (libSQL) cloud DB — if both vars are set, db_handler uses
# Turso instead of the local SQLite file so data survives Cloud restarts.
TURSO_DATABASE_URL = _get_setting("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = _get_setting("TURSO_AUTH_TOKEN", "")

# ---------------------------------------------------------------------------
# Authentication  (Admin Tab)
# ---------------------------------------------------------------------------
# Master switch for Google OAuth (requires Authlib + .streamlit/secrets.toml).
# When False the app skips login and per-user limits entirely.
AUTH_ENABLED = _get_setting("AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")

# First-come-first-served cap on Google accounts. Admins are exempt and can
# free up slots from the Admin tab.
AUTH_MAX_USERS = int(_get_setting("AUTH_MAX_USERS", "10"))

# Comma-separated emails (case-insensitive) that always have access and see
# the Admin tab. Managed via AUTH_ADMIN_EMAILS in .env / Streamlit secrets.
AUTH_ADMIN_EMAILS = {
    e.strip().lower() for e in _get_setting("AUTH_ADMIN_EMAILS", "").split(",") if e.strip()
}

# ---------------------------------------------------------------------------
# Scan Universe
# ---------------------------------------------------------------------------
# NOTE: NSE Indices reconstitutes the Nifty 100 semi-annually (March and
# September). Verify/update this list periodically against the official list.
NIFTY_100_TICKERS = [
    # --- Nifty 50 ---
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "HINDUNILVR.NS", "LT.NS",
    "BAJFINANCE.NS", "KOTAKBANK.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "ASIANPAINT.NS", "TITAN.NS", "ULTRACEMCO.NS", "M&M.NS",
    "NTPC.NS", "ADANIENT.NS", "ADANIPORTS.NS", "BAJAJFINSV.NS", "POWERGRID.NS",
    "WIPRO.NS", "NESTLEIND.NS", "TMPV.NS", "JSWSTEEL.NS", "TATASTEEL.NS",
    "COALINDIA.NS", "SBILIFE.NS", "HDFCLIFE.NS", "GRASIM.NS", "INDUSINDBK.NS",
    "TECHM.NS", "CIPLA.NS", "DRREDDY.NS", "EICHERMOT.NS", "BPCL.NS",
    "ONGC.NS", "APOLLOHOSP.NS", "DIVISLAB.NS", "BRITANNIA.NS", "TATACONSUM.NS",
    "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "LTIM.NS", "SHRIRAMFIN.NS", "TRENT.NS",

    # --- Nifty Next 50 ---
    "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "AMBUJACEM.NS", "BAJAJHLDNG.NS",
    "BANKBARODA.NS", "BERGEPAINT.NS", "BOSCHLTD.NS", "CANBK.NS", "CHOLAFIN.NS",
    "COLPAL.NS", "DABUR.NS", "DLF.NS", "GAIL.NS", "GODREJCP.NS",
    "HAVELLS.NS", "HAL.NS", "ICICIGI.NS", "ICICIPRULI.NS", "INDHOTEL.NS",
    "INDIGO.NS", "IOC.NS", "IRFC.NS", "JINDALSTEL.NS", "JIOFIN.NS",
    "LODHA.NS", "LTF.NS", "MARICO.NS", "MOTHERSON.NS", "MUTHOOTFIN.NS",
    "NAUKRI.NS", "PFC.NS", "PIDILITIND.NS", "PNB.NS", "RECLTD.NS",
    "SIEMENS.NS", "SRF.NS", "TATAPOWER.NS", "TIINDIA.NS", "TORNTPHARM.NS",
    "TVSMOTOR.NS", "UNITDSPR.NS", "VBL.NS", "VEDL.NS", "ETERNAL.NS",
    "ZYDUSLIFE.NS", "GODREJPROP.NS", "POLYCAB.NS", "ASHOKLEY.NS", "AUROPHARMA.NS",
]

# TATASILV.NS has no Yahoo Finance data; SILVERBEES.NS (Nippon India Silver
# ETF, the most liquid silver ETF on NSE) is used instead.
GOLD_SILVER_TICKERS = ["TATAGOLD.NS", "SILVERBEES.NS"]

# Full scan universe (102 tickers). Gold/Silver are always included in the
# shortlist regardless of score (see strategy.generate_shortlist).
SCAN_UNIVERSE = NIFTY_100_TICKERS + GOLD_SILVER_TICKERS

# Sector classification — drives the sector-trend scoring component.
# During a scan, all tickers in a sector have their recent returns averaged
# into that sector's "current trend" (see SCORE_SECTOR_TREND).
SECTOR_MAP = {
    # --- Banking ---
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "SBIN.NS": "Banking",
    "KOTAKBANK.NS": "Banking", "AXISBANK.NS": "Banking", "INDUSINDBK.NS": "Banking",
    "BANKBARODA.NS": "Banking", "CANBK.NS": "Banking", "PNB.NS": "Banking",

    # --- Financial Services (NBFC / Insurance / AMC) ---
    "BAJFINANCE.NS": "Financial Services", "BAJAJFINSV.NS": "Financial Services",
    "SBILIFE.NS": "Financial Services", "HDFCLIFE.NS": "Financial Services",
    "SHRIRAMFIN.NS": "Financial Services", "BAJAJHLDNG.NS": "Financial Services",
    "CHOLAFIN.NS": "Financial Services", "ICICIGI.NS": "Financial Services",
    "ICICIPRULI.NS": "Financial Services", "IRFC.NS": "Financial Services",
    "JIOFIN.NS": "Financial Services", "LTF.NS": "Financial Services",
    "MUTHOOTFIN.NS": "Financial Services", "PFC.NS": "Financial Services",
    "RECLTD.NS": "Financial Services",

    # --- IT ---
    "TCS.NS": "IT", "INFY.NS": "IT", "HCLTECH.NS": "IT", "WIPRO.NS": "IT",
    "TECHM.NS": "IT", "LTIM.NS": "IT", "NAUKRI.NS": "IT",

    # --- Pharma ---
    "SUNPHARMA.NS": "Pharma", "CIPLA.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "DIVISLAB.NS": "Pharma", "TORNTPHARM.NS": "Pharma", "ZYDUSLIFE.NS": "Pharma",
    "AUROPHARMA.NS": "Pharma",

    # --- Healthcare ---
    "APOLLOHOSP.NS": "Healthcare",

    # --- FMCG ---
    "ITC.NS": "FMCG", "HINDUNILVR.NS": "FMCG", "NESTLEIND.NS": "FMCG",
    "BRITANNIA.NS": "FMCG", "TATACONSUM.NS": "FMCG", "COLPAL.NS": "FMCG",
    "DABUR.NS": "FMCG", "GODREJCP.NS": "FMCG", "MARICO.NS": "FMCG",
    "UNITDSPR.NS": "FMCG", "VBL.NS": "FMCG",

    # --- Auto ---
    "MARUTI.NS": "Auto", "M&M.NS": "Auto", "TMPV.NS": "Auto",
    "EICHERMOT.NS": "Auto", "HEROMOTOCO.NS": "Auto", "BAJAJ-AUTO.NS": "Auto",
    "BOSCHLTD.NS": "Auto", "MOTHERSON.NS": "Auto", "TIINDIA.NS": "Auto",
    "TVSMOTOR.NS": "Auto", "ASHOKLEY.NS": "Auto",

    # --- Metals & Mining ---
    "JSWSTEEL.NS": "Metals & Mining", "TATASTEEL.NS": "Metals & Mining",
    "COALINDIA.NS": "Metals & Mining", "JINDALSTEL.NS": "Metals & Mining",
    "VEDL.NS": "Metals & Mining",

    # --- Oil & Gas ---
    "RELIANCE.NS": "Oil & Gas", "BPCL.NS": "Oil & Gas", "ONGC.NS": "Oil & Gas",
    "GAIL.NS": "Oil & Gas", "IOC.NS": "Oil & Gas",

    # --- Power ---
    "NTPC.NS": "Power", "POWERGRID.NS": "Power", "ADANIENSOL.NS": "Power",
    "ADANIGREEN.NS": "Power", "ADANIPOWER.NS": "Power", "TATAPOWER.NS": "Power",

    # --- Cement ---
    "ULTRACEMCO.NS": "Cement", "GRASIM.NS": "Cement", "AMBUJACEM.NS": "Cement",

    # --- Infrastructure / Capital Goods ---
    "LT.NS": "Infrastructure", "ADANIENT.NS": "Infrastructure",
    "ADANIPORTS.NS": "Infrastructure", "HAL.NS": "Infrastructure",
    "SIEMENS.NS": "Infrastructure",

    # --- Realty ---
    "DLF.NS": "Realty", "LODHA.NS": "Realty", "GODREJPROP.NS": "Realty",

    # --- Telecom ---
    "BHARTIARTL.NS": "Telecom",

    # --- Consumer Durables ---
    "ASIANPAINT.NS": "Consumer Durables", "TITAN.NS": "Consumer Durables",
    "BERGEPAINT.NS": "Consumer Durables", "HAVELLS.NS": "Consumer Durables",
    "POLYCAB.NS": "Consumer Durables",

    # --- Chemicals ---
    "PIDILITIND.NS": "Chemicals", "SRF.NS": "Chemicals",

    # --- Consumer Services ---
    "TRENT.NS": "Consumer Services", "INDHOTEL.NS": "Consumer Services",
    "INDIGO.NS": "Consumer Services", "ETERNAL.NS": "Consumer Services",

    # --- Precious Metals (always-tracked commodity ETFs) ---
    "TATAGOLD.NS": "Precious Metals", "SILVERBEES.NS": "Precious Metals",
}
