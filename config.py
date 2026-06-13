"""
Central configuration for the Nifty 100 Swing Trading Agent.

All tunable parameters live here so the rest of the codebase stays
declarative. Values can be overridden via environment variables / a
local .env file (see .env.example).
"""

import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()


def _get_setting(key, default=""):
    """Read a setting from the environment (.env locally) first, falling
    back to Streamlit secrets (.streamlit/secrets.toml, or the Cloud
    Secrets box) - Streamlit Community Cloud doesn't reliably expose
    root-level secrets as environment variables, but st.secrets always
    has them."""
    val = os.getenv(key)
    if val is not None:
        return val
    try:
        return str(st.secrets[key])
    except Exception:
        return default


# ---------------------------------------------------------------------------
# LLM API settings - any OpenAI-compatible chat completions endpoint works
# (Groq, xAI, OpenRouter, GitHub Models, ...) since ai_analyst.py uses the
# standard {"model", "messages", "temperature"} request shape and reads
# choices[0].message.content from the response.
# ---------------------------------------------------------------------------
LLM_API_KEY = _get_setting("LLM_API_KEY", "")
LLM_API_URL = _get_setting("LLM_API_URL", "https://api.groq.com/openai/v1/chat/completions")
# Default model runs on Groq's free tier - see
# https://console.groq.com/docs/models for the current list of model ids.
LLM_MODEL = _get_setting("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_TIMEOUT_SECONDS = 60

# ---------------------------------------------------------------------------
# Data settings (yfinance)
# ---------------------------------------------------------------------------
# Daily candles are used throughout - appropriate for a 2-3 week swing
# trading horizon (intraday noise is irrelevant at this timeframe).
DATA_PERIOD = "1y"
DATA_INTERVAL = "1d"

# Number of tickers fetched/analyzed concurrently during a scan. yfinance
# calls are I/O-bound, so a thread pool here cuts scan time dramatically
# without overwhelming Yahoo's API.
SCAN_MAX_WORKERS = 8

# ---------------------------------------------------------------------------
# Technical indicator parameters
# ---------------------------------------------------------------------------
RSI_PERIOD = 14
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 65

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# The MACD histogram is considered "near a crossover" when its absolute
# value has shrunk below this fraction of its own recent (20-bar) average
# magnitude AND it is still shrinking versus the prior bar.
MACD_CROSSOVER_PROXIMITY_FACTOR = 0.15

# ---------------------------------------------------------------------------
# Chart display
# ---------------------------------------------------------------------------
# The Chart Analysis tab plots only the most recent CHART_DISPLAY_MONTHS of
# candles/volume/MACD - separate from FIB_LOOKBACK_DAYS below, which still
# uses its full window to compute Fibonacci levels. Those levels (and the
# swing high/low, if within this window) are drawn across this shorter view.
CHART_DISPLAY_MONTHS = 3

# ---------------------------------------------------------------------------
# Fibonacci retracement parameters
# ---------------------------------------------------------------------------
# 90 trading days (~4.5 months) gives a swing-relevant peak/trough window.
FIB_LOOKBACK_DAYS = 90
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618]
# 50% and 61.8% are weighted higher in the scoring model.
FIB_KEY_LEVELS = [0.5, 0.618]
# Price must be within this fraction of a level to count as "at" that level.
FIB_PROXIMITY_PCT = 0.01  # 1%

# ---------------------------------------------------------------------------
# Volume parameters
# ---------------------------------------------------------------------------
VOLUME_AVG_WINDOW = 20  # rolling window (trading days) for the average-volume baseline
# Latest session volume >= this multiple of its rolling average counts as a
# "surge", confirming the move with above-average participation.
VOLUME_SURGE_RATIO = 1.5

# ---------------------------------------------------------------------------
# Buy/sell pressure parameters
# ---------------------------------------------------------------------------
# Proxy for order flow (NSE order-book depth isn't available via yfinance):
# over this many sessions, the share of total volume that traded on "buy"
# days (close >= open) vs "sell" days (close < open).
BUY_SELL_PRESSURE_WINDOW = 20

# ---------------------------------------------------------------------------
# Sector-trend parameters
# ---------------------------------------------------------------------------
# How far back (trading days) to measure each stock's own return when
# aggregating it into its sector's average "current trend" - roughly 2
# trading weeks, to capture recent industry-wide swings.
SECTOR_TREND_LOOKBACK_DAYS = 10
# A sector's average return over that window must clear this magnitude
# (either direction) for the sector-trend score bonus to apply.
SECTOR_TREND_THRESHOLD = 0.015  # 1.5%

# ---------------------------------------------------------------------------
# Scoring weights / shortlist size
# ---------------------------------------------------------------------------
# Weights sum to 100 across the best-case combination (key Fib level + RSI
# extreme + MACD crossover proximity + volume surge + aligned sector trend
# = 30+20+28+15+7 = 100). Fibonacci and MACD are the primary drivers of the
# score; sector trend is a small/secondary factor by design.
SCORE_FIB_KEY_LEVEL = 30
SCORE_FIB_OTHER_LEVEL = 15
SCORE_RSI_EXTREME = 20
SCORE_MACD_PROXIMITY = 28
SCORE_VOLUME = 15
SCORE_SECTOR_TREND = 7

SHORTLIST_MAX_SIZE = 8
SHORTLIST_MIN_SCORE = 20  # ignore stocks with a weak/no setup

# ---------------------------------------------------------------------------
# AI analyst settings
# ---------------------------------------------------------------------------
# How many top-ranked shortlist entries get a full Entry/Stop-Loss/Take-Profit
# write-up (and a news headline lookup) from the AI analyst.
AI_TOP_PICKS_COUNT = 8
# Recent news headlines fetched per ticker (via yfinance) for AI context.
NEWS_HEADLINE_COUNT = 3

# ---------------------------------------------------------------------------
# Market hours (informational only - the agent is run on demand, not on
# an automatic background schedule)
# ---------------------------------------------------------------------------
MARKET_TIMEZONE = "Asia/Kolkata"
MARKET_OPEN_TIME = "09:15"
MARKET_CLOSE_TIME = "15:30"

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trading_agent.db")

# Optional hosted Turso (libSQL) database - if both are set, db_handler uses
# Turso instead of the local SQLite file, so data survives Streamlit Cloud
# restarts/redeploys. Leave unset for local SQLite (the default).
TURSO_DATABASE_URL = _get_setting("TURSO_DATABASE_URL", "")
TURSO_AUTH_TOKEN = _get_setting("TURSO_AUTH_TOKEN", "")

# ---------------------------------------------------------------------------
# Authentication (Google OAuth via Streamlit's built-in st.login/st.user -
# requires Authlib and a [auth] / [auth.google] section in
# .streamlit/secrets.toml, see .streamlit/secrets.toml.example)
# ---------------------------------------------------------------------------
# Master switch for Google sign-in. When False, the app skips the login
# screen and per-user limits entirely (everyone gets full access, no Admin
# tab) - useful while Google OAuth isn't set up or for a private deployment.
AUTH_ENABLED = _get_setting("AUTH_ENABLED", "false").strip().lower() in ("1", "true", "yes", "on")

# Maximum number of distinct Google accounts that may use this app, on a
# first-come-first-served basis (db_handler.authorized_users tracks who has
# already logged in). Admins (below) are exempt from this cap and can free up
# slots via the in-app Admin tab.
AUTH_MAX_USERS = int(_get_setting("AUTH_MAX_USERS", "10"))

# Comma-separated Google account emails (case-insensitive) that always have
# access regardless of AUTH_MAX_USERS, and see the Admin tab to manage the
# authorized-user list.
AUTH_ADMIN_EMAILS = {
    e.strip().lower() for e in _get_setting("AUTH_ADMIN_EMAILS", "").split(",") if e.strip()
}

# ---------------------------------------------------------------------------
# Nifty 100 universe (NSE symbols, '.NS' suffix for yfinance)
#
# NOTE: NSE Indices reconstitutes the Nifty 100 semi-annually (March and
# September). This list reflects a recent representative composition -
# verify/update it periodically against the official Nifty 100 list.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Gold/Silver - permanently tracked alongside the Nifty 100 universe
# ---------------------------------------------------------------------------
# Note: TATASILV.NS has no data on Yahoo Finance (yfinance source) as of
# writing, so SILVERBEES.NS (Nippon India Silver ETF, the most liquid silver
# ETF on NSE) is used instead.
GOLD_SILVER_TICKERS = ["TATAGOLD.NS", "SILVERBEES.NS"]

# Full scan universe (102 tickers): every scan covers the Nifty 100 plus
# Gold/Silver, and Gold/Silver are always included in the shortlist
# regardless of score (see strategy.generate_shortlist).
SCAN_UNIVERSE = NIFTY_100_TICKERS + GOLD_SILVER_TICKERS

# ---------------------------------------------------------------------------
# Sector classification (drives the sector-trend scoring component)
#
# Each ticker is mapped to a broad NSE sector grouping. During a scan, all
# tickers sharing a sector have their recent returns averaged into that
# sector's "current trend" - a small additional signal (see
# SCORE_SECTOR_TREND) on top of each stock's own technicals.
# ---------------------------------------------------------------------------
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
