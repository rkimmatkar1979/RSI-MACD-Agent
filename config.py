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

from dotenv import load_dotenv

load_dotenv()


def _get_setting(key, default=""):
    """Read from env (.env locally) first, then Streamlit secrets (Cloud).

    Streamlit is imported lazily so this module can be imported outside a
    Streamlit runtime (e.g. scripts, tests, cron jobs) without crashing.
    """
    val = os.getenv(key)
    if val is not None:
        return val
    try:
        import streamlit as st  # noqa: PLC0415 — intentional lazy import
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
    "+ Grok AI commentary, tuned for 3-month swing setups."
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
    "Fibonacci retracements) plus AI-generated commentary, for 3-month swing "
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
# Daily candles throughout — appropriate for a 3-month swing trading horizon.
DATA_PERIOD = "1y"
DATA_INTERVAL = "1d"

# How long a ticker's fetched price data is cached before re-fetching from
# Yahoo Finance. yfinance's "daily" bar for the current session updates
# live while the market is open, so a short TTL (not the old 1 hour) keeps
# RSI/MACD/the chart reflecting genuinely current intraday price action.
PRICE_DATA_CACHE_TTL = 300  # 5 minutes

# yfinance calls are I/O-bound; a thread pool cuts scan time dramatically
# without overwhelming Yahoo's API.
SCAN_MAX_WORKERS = 16

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
# Used for descriptive text only - NOT for scoring (see MACD_CROSSOVER_LOOKBACK_DAYS).
MACD_CROSSOVER_PROXIMITY_FACTOR = 0.15
# How many recent sessions to scan for an actual (confirmed) bullish MACD/
# Signal crossover for scoring purposes - a crossover that "might be coming"
# doesn't count, only one that has already happened within this window.
MACD_CROSSOVER_LOOKBACK_DAYS = 10

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

# Sector trend snapshot (Shortlist tab, above the heatmap) - separate from
# SECTOR_TREND_LOOKBACK_DAYS above (which only feeds Score component 5 and is
# averaged over the shortlist). These two windows are purely informational,
# averaged across every sector in the FULL scan universe (not just
# shortlisted stocks), so every sector shows up even on a day it produced no
# qualifying setups.
SECTOR_TREND_WEEK_DAYS = 5             # ~1 trading week
SECTOR_TREND_MONTH_DAYS = 21           # ~1 trading month

# Index-level tickers used for the market-wide news feeding the sector
# forecast commentary (see ai_analyst.get_sector_outlook) - as opposed to
# ai_analyst._format_news, which fetches per-stock headlines.
MARKET_NEWS_TICKERS = ["^NSEI", "^BSESN"]

# Swing potential - is this stock actually capable of a worthwhile swing-
# trade move, regardless of whether RSI/MACD are currently confirming a
# reversal? Blends (a) room between current price and the recent swing high
# (the realistic near-term target) with (b) a smaller weighting toward this
# stock's own recent swing-leg history (median up-leg size via a zigzag scan)
# - "possibility" leads, "track record" is mixed in as a secondary check.
SWING_LOOKBACK_DAYS = 63               # ~3 trading months
SWING_REVERSAL_PCT = 0.04              # 4% reversal to count as a new zigzag pivot
SWING_TARGET_PCT = 0.075               # 7.5% (midpoint of a 7-8% target) = full credit
SWING_POSSIBILITY_WEIGHT = 0.6         # weight on room-to-target (current setup)
SWING_TRACK_RECORD_WEIGHT = 0.4        # weight on median historical up-leg size

# ---------------------------------------------------------------------------
# Shortlist Tab
# ---------------------------------------------------------------------------
# Score weights sum to 105 for the best-case combination:
#   key Fib (30) + RSI recovering (20) + confirmed MACD crossover (28)
#   + swing potential (20) + sector trend (7) = 105
# Fibonacci and MACD are primary drivers; sector trend is secondary by design.
# Volume is NOT scored (shown as context only, see strategy.score_setup).
SCORE_FIB_KEY_LEVEL = 30
SCORE_FIB_OTHER_LEVEL = 15
SCORE_RSI_EXTREME = 20
SCORE_MACD_PROXIMITY = 28              # confirmed crossover (tapered + extension-discounted)
SCORE_MACD_EARLY = 14                  # still below Signal but converging - early/watch, half weight
SCORE_VOLUME = 15
SCORE_SWING_POTENTIAL = 20
SCORE_SECTOR_TREND = 7

SHORTLIST_MAX_SIZE = 13
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
AI_TOP_PICKS_COUNT = 13
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
# Nifty 100 (top 100 by market cap on NSE) - official constituent list.
# NOTE: NSE Indices reconstitutes this semi-annually (March/September).
# Verify/update against the official list at niftyindices.com.
NIFTY_100_TICKERS = [
    # --- Banking ---
    "AXISBANK.NS", "BANKBARODA.NS", "CANBK.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "KOTAKBANK.NS", "PNB.NS", "SBIN.NS", "UNIONBANK.NS",
    # --- Financial Services ---
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "BAJAJHLDNG.NS", "CHOLAFIN.NS", "HDFCAMC.NS",
    "HDFCLIFE.NS", "IRFC.NS", "JIOFIN.NS", "MUTHOOTFIN.NS", "PFC.NS",
    "RECLTD.NS", "SBILIFE.NS", "SHRIRAMFIN.NS", "TATACAP.NS",
    # --- IT ---
    "HCLTECH.NS", "INFY.NS", "LTM.NS", "TCS.NS", "TECHM.NS",
    "WIPRO.NS",
    # --- Pharma ---
    "CIPLA.NS", "DIVISLAB.NS", "DRREDDY.NS", "SUNPHARMA.NS", "TORNTPHARM.NS",
    "ZYDUSLIFE.NS",
    # --- Healthcare ---
    "APOLLOHOSP.NS", "MAXHEALTH.NS",
    # --- FMCG ---
    "BRITANNIA.NS", "GODREJCP.NS", "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
    "TATACONSUM.NS", "UNITDSPR.NS", "VBL.NS",
    # --- Auto & Auto Ancillaries ---
    "BAJAJ-AUTO.NS", "BOSCHLTD.NS", "EICHERMOT.NS", "HYUNDAI.NS", "M&M.NS",
    "MARUTI.NS", "MOTHERSON.NS", "TVSMOTOR.NS", "TMCV.NS", "TMPV.NS",
    # --- Consumer Durables ---
    "ASIANPAINT.NS", "TITAN.NS",
    # --- Consumer Services ---
    "DMART.NS", "ETERNAL.NS", "INDHOTEL.NS", "TRENT.NS",
    # --- Infrastructure ---
    "ADANIPORTS.NS", "INDIGO.NS", "LT.NS",
    # --- Cement ---
    "AMBUJACEM.NS", "GRASIM.NS", "SHREECEM.NS", "ULTRACEMCO.NS",
    # --- Chemicals ---
    "PIDILITIND.NS", "SOLARINDS.NS",
    # --- Oil & Gas ---
    "BPCL.NS", "GAIL.NS", "IOC.NS", "ONGC.NS", "RELIANCE.NS",
    # --- Power ---
    "ADANIENSOL.NS", "ADANIGREEN.NS", "ADANIPOWER.NS", "NTPC.NS", "POWERGRID.NS",
    "TATAPOWER.NS",
    # --- Metals & Mining ---
    "ADANIENT.NS", "COALINDIA.NS", "HINDALCO.NS", "HINDZINC.NS", "JSWSTEEL.NS",
    "JINDALSTEL.NS", "TATASTEEL.NS", "VEDL.NS",
    # --- Realty ---
    "DLF.NS", "LODHA.NS",
    # --- Telecom ---
    "BHARTIARTL.NS",
    # --- Capital Goods ---
    "ABB.NS", "BEL.NS", "CGPOWER.NS", "CUMMINSIND.NS", "HAL.NS",
    "MAZDOCK.NS", "ENRIN.NS", "SIEMENS.NS",
]

# Nifty Midcap 150 (rank ~101-250 by market cap) - official constituent list.
# NOTE: reconstituted quarterly - verify/update periodically.
NIFTY_MIDCAP_150_TICKERS = [
    # --- Banking ---
    "AUBANK.NS", "BANKINDIA.NS", "MAHABANK.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
    "INDIANB.NS", "INDUSINDBK.NS", "YESBANK.NS",
    # --- Financial Services ---
    "360ONE.NS", "ABCAPITAL.NS", "AIIL.NS", "BSE.NS", "BAJAJHFL.NS",
    "GROWW.NS", "CRISIL.NS", "GICRE.NS", "HDBFS.NS", "HUDCO.NS",
    "ICICIGI.NS", "ICICIAMC.NS", "ICICIPRULI.NS", "IREDA.NS", "LTF.NS",
    "LICHSGFIN.NS", "LICI.NS", "M&MFIN.NS", "MFSL.NS", "MOTILALOFS.NS",
    "MCX.NS", "NAM-INDIA.NS", "PAYTM.NS", "POLICYBZR.NS", "SBICARD.NS",
    "SUNDARMFIN.NS", "TATAINVEST.NS", "NIACL.NS",
    # --- IT ---
    "COFORGE.NS", "HEXT.NS", "KPITTECH.NS", "LTTS.NS", "MPHASIS.NS",
    "OFSS.NS", "PERSISTENT.NS", "TATAELXSI.NS",
    # --- Pharma ---
    "ABBOTINDIA.NS", "AJANTPHARM.NS", "ALKEM.NS", "ANTHEM.NS", "AUROPHARMA.NS",
    "BIOCON.NS", "GLAXO.NS", "GLENMARK.NS", "IPCALAB.NS", "LAURUSLABS.NS",
    "LUPIN.NS", "MANKIND.NS",
    # --- Healthcare ---
    "FORTIS.NS", "MEDANTA.NS",
    # --- FMCG ---
    "AWL.NS", "COLPAL.NS", "DABUR.NS", "GODFRYPHLP.NS", "MARICO.NS",
    "PATANJALI.NS", "RADICO.NS", "UBL.NS",
    # --- Auto & Auto Ancillaries ---
    "APOLLOTYRE.NS", "ASHOKLEY.NS", "BALKRISIND.NS", "BHARATFORG.NS", "ENDURANCE.NS",
    "EXIDEIND.NS", "HEROMOTOCO.NS", "MRF.NS", "SCHAEFFLER.NS", "TIINDIA.NS",
    "UNOMINDA.NS",
    # --- Consumer Durables ---
    "ASTRAL.NS", "BERGEPAINT.NS", "BLUESTARCO.NS", "DIXON.NS", "HAVELLS.NS",
    "KALYANKJIL.NS", "LGEINDIA.NS", "VOLTAS.NS",
    # --- Consumer Services ---
    "NYKAA.NS", "ITCHOTELS.NS", "IRCTC.NS", "NAUKRI.NS", "JUBLFOOD.NS",
    "LENSKART.NS", "SWIGGY.NS", "VMM.NS",
    # --- Infrastructure ---
    "AIAENG.NS", "APARINDS.NS", "BDL.NS", "BHEL.NS", "COCHINSHIP.NS",
    "CONCOR.NS", "ESCORTS.NS", "GMRAIRPORT.NS", "JSWINFRA.NS", "KEI.NS",
    "POLYCAB.NS", "RVNL.NS", "SUPREMEIND.NS", "THERMAX.NS",
    # --- Cement ---
    "ACC.NS", "DALBHARAT.NS", "JKCEMENT.NS",
    # --- Chemicals ---
    "COROMANDEL.NS", "GODREJIND.NS", "FLUOROCHEM.NS", "LINDEINDIA.NS", "PIIND.NS",
    "SRF.NS", "UPL.NS",
    # --- Oil & Gas ---
    "ATGL.NS", "HINDPETRO.NS", "OIL.NS", "PETRONET.NS",
    # --- Power ---
    "JSWENERGY.NS", "NHPC.NS", "NLCINDIA.NS", "NTPCGREEN.NS", "SJVN.NS",
    "TORNTPOWER.NS",
    # --- Metals & Mining ---
    "APLAPOLLO.NS", "JSL.NS", "LLOYDSME.NS", "NMDC.NS", "NATIONALUM.NS",
    "SAIL.NS",
    # --- Realty ---
    "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS", "PRESTIGE.NS",
    # --- Telecom ---
    "BHARTIHEXA.NS", "INDUSTOWER.NS", "TATACOMM.NS", "IDEA.NS",
    # --- Textiles ---
    "KPRMILL.NS", "PAGEIND.NS",
    # --- Capital Goods ---
    "GVT&D.NS", "POWERINDIA.NS", "HONAUT.NS", "PREMIERENE.NS", "SUZLON.NS",
    "WAAREEENER.NS",
    # --- Other ---
    "3MINDIA.NS",
]

# Nifty Smallcap 50 (rank ~251-300 by market cap) - official constituent list.
# NOTE: reconstituted quarterly - verify/update periodically.
NIFTY_SMALLCAP_50_TICKERS = [
    # --- Banking ---
    "BANDHANBNK.NS", "CUB.NS", "KARURVYSYA.NS", "RBLBANK.NS",
    # --- Financial Services ---
    "ANANDRATHI.NS", "ANGELONE.NS", "CDSL.NS", "CHOLAHLDNG.NS", "CAMS.NS",
    "FIVESTAR.NS", "IIFL.NS", "KFINTECH.NS", "MANAPPURAM.NS", "PNBHOUSING.NS",
    "PIRAMALFIN.NS", "POONAWALLA.NS",
    # --- IT ---
    "AFFLE.NS", "REDINGTON.NS", "TATATECH.NS",
    # --- Pharma ---
    "COHANCE.NS", "GLAND.NS", "NATCOPHARM.NS", "NEULANDLAB.NS", "PPLPHARMA.NS",
    "SAILIFE.NS", "WOCKPHARMA.NS",
    # --- Healthcare ---
    "ASTERDM.NS", "LALPATHLAB.NS", "NH.NS", "SYNGENE.NS",
    # --- Auto & Auto Ancillaries ---
    "ARE&M.NS", "SONACOMS.NS",
    # --- Consumer Durables ---
    "AMBER.NS", "CROMPTON.NS", "PGEL.NS",
    # --- Consumer Services ---
    "DELHIVERY.NS",
    # --- Infrastructure ---
    "INOXWIND.NS", "KAYNES.NS", "KEC.NS", "NBCC.NS", "WELCORP.NS",
    # --- Chemicals ---
    "HSCL.NS", "NAVINFLUOR.NS", "TATACHEM.NS",
    # --- Oil & Gas ---
    "AEGISLOG.NS", "CASTROLIND.NS", "IGL.NS",
    # --- Power ---
    "CESC.NS", "RPOWER.NS",
    # --- Metals & Mining ---
    "HINDCOPPER.NS",
]

# TATASILV.NS has no Yahoo Finance data; SILVERBEES.NS (Nippon India Silver
# ETF, the most liquid silver ETF on NSE) is used instead.
GOLD_SILVER_TICKERS = ["TATAGOLD.NS", "SILVERBEES.NS"]

# Full scan universe (300 tickers = Nifty 100 + Midcap 150 + Smallcap 50,
# three disjoint official NSE tiers, + Gold/Silver). Gold/Silver are always
# included in the shortlist regardless of score (see strategy.generate_shortlist).
SCAN_UNIVERSE = (
    NIFTY_100_TICKERS + NIFTY_MIDCAP_150_TICKERS + NIFTY_SMALLCAP_50_TICKERS
    + GOLD_SILVER_TICKERS
)

# Sector classification — drives the sector-trend scoring component.
# During a scan, all tickers in a sector have their recent returns averaged
# into that sector's "current trend" (see SCORE_SECTOR_TREND).
SECTOR_MAP = {
    # --- Banking ---
    "AXISBANK.NS": "Banking", "BANKBARODA.NS": "Banking", "CANBK.NS": "Banking",
    "HDFCBANK.NS": "Banking", "ICICIBANK.NS": "Banking", "KOTAKBANK.NS": "Banking",
    "PNB.NS": "Banking", "SBIN.NS": "Banking", "UNIONBANK.NS": "Banking",
    "AUBANK.NS": "Banking", "BANKINDIA.NS": "Banking", "MAHABANK.NS": "Banking",
    "FEDERALBNK.NS": "Banking", "IDFCFIRSTB.NS": "Banking", "INDIANB.NS": "Banking",
    "INDUSINDBK.NS": "Banking", "YESBANK.NS": "Banking", "BANDHANBNK.NS": "Banking",
    "CUB.NS": "Banking", "KARURVYSYA.NS": "Banking", "RBLBANK.NS": "Banking",
    # --- Financial Services ---
    "BAJFINANCE.NS": "Financial Services", "BAJAJFINSV.NS": "Financial Services", "BAJAJHLDNG.NS": "Financial Services",
    "CHOLAFIN.NS": "Financial Services", "HDFCAMC.NS": "Financial Services", "HDFCLIFE.NS": "Financial Services",
    "IRFC.NS": "Financial Services", "JIOFIN.NS": "Financial Services", "MUTHOOTFIN.NS": "Financial Services",
    "PFC.NS": "Financial Services", "RECLTD.NS": "Financial Services", "SBILIFE.NS": "Financial Services",
    "SHRIRAMFIN.NS": "Financial Services", "TATACAP.NS": "Financial Services", "360ONE.NS": "Financial Services",
    "ABCAPITAL.NS": "Financial Services", "AIIL.NS": "Financial Services", "BSE.NS": "Financial Services",
    "BAJAJHFL.NS": "Financial Services", "GROWW.NS": "Financial Services", "CRISIL.NS": "Financial Services",
    "GICRE.NS": "Financial Services", "HDBFS.NS": "Financial Services", "HUDCO.NS": "Financial Services",
    "ICICIGI.NS": "Financial Services", "ICICIAMC.NS": "Financial Services", "ICICIPRULI.NS": "Financial Services",
    "IREDA.NS": "Financial Services", "LTF.NS": "Financial Services", "LICHSGFIN.NS": "Financial Services",
    "LICI.NS": "Financial Services", "M&MFIN.NS": "Financial Services", "MFSL.NS": "Financial Services",
    "MOTILALOFS.NS": "Financial Services", "MCX.NS": "Financial Services", "NAM-INDIA.NS": "Financial Services",
    "PAYTM.NS": "Financial Services", "POLICYBZR.NS": "Financial Services", "SBICARD.NS": "Financial Services",
    "SUNDARMFIN.NS": "Financial Services", "TATAINVEST.NS": "Financial Services", "NIACL.NS": "Financial Services",
    "ANANDRATHI.NS": "Financial Services", "ANGELONE.NS": "Financial Services", "CDSL.NS": "Financial Services",
    "CHOLAHLDNG.NS": "Financial Services", "CAMS.NS": "Financial Services", "FIVESTAR.NS": "Financial Services",
    "IIFL.NS": "Financial Services", "KFINTECH.NS": "Financial Services", "MANAPPURAM.NS": "Financial Services",
    "PNBHOUSING.NS": "Financial Services", "PIRAMALFIN.NS": "Financial Services", "POONAWALLA.NS": "Financial Services",
    # --- IT ---
    "HCLTECH.NS": "IT", "INFY.NS": "IT", "LTM.NS": "IT",
    "TCS.NS": "IT", "TECHM.NS": "IT", "WIPRO.NS": "IT",
    "COFORGE.NS": "IT", "HEXT.NS": "IT", "KPITTECH.NS": "IT",
    "LTTS.NS": "IT", "MPHASIS.NS": "IT", "OFSS.NS": "IT",
    "PERSISTENT.NS": "IT", "TATAELXSI.NS": "IT", "AFFLE.NS": "IT",
    "REDINGTON.NS": "IT", "TATATECH.NS": "IT",
    # --- Pharma ---
    "CIPLA.NS": "Pharma", "DIVISLAB.NS": "Pharma", "DRREDDY.NS": "Pharma",
    "SUNPHARMA.NS": "Pharma", "TORNTPHARM.NS": "Pharma", "ZYDUSLIFE.NS": "Pharma",
    "ABBOTINDIA.NS": "Pharma", "AJANTPHARM.NS": "Pharma", "ALKEM.NS": "Pharma",
    "ANTHEM.NS": "Pharma", "AUROPHARMA.NS": "Pharma", "BIOCON.NS": "Pharma",
    "GLAXO.NS": "Pharma", "GLENMARK.NS": "Pharma", "IPCALAB.NS": "Pharma",
    "LAURUSLABS.NS": "Pharma", "LUPIN.NS": "Pharma", "MANKIND.NS": "Pharma",
    "COHANCE.NS": "Pharma", "GLAND.NS": "Pharma", "NATCOPHARM.NS": "Pharma",
    "NEULANDLAB.NS": "Pharma", "PPLPHARMA.NS": "Pharma", "SAILIFE.NS": "Pharma",
    "WOCKPHARMA.NS": "Pharma",
    # --- Healthcare ---
    "APOLLOHOSP.NS": "Healthcare", "MAXHEALTH.NS": "Healthcare", "FORTIS.NS": "Healthcare",
    "MEDANTA.NS": "Healthcare", "ASTERDM.NS": "Healthcare", "LALPATHLAB.NS": "Healthcare",
    "NH.NS": "Healthcare", "SYNGENE.NS": "Healthcare",
    # --- FMCG ---
    "BRITANNIA.NS": "FMCG", "GODREJCP.NS": "FMCG", "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG", "NESTLEIND.NS": "FMCG", "TATACONSUM.NS": "FMCG",
    "UNITDSPR.NS": "FMCG", "VBL.NS": "FMCG", "AWL.NS": "FMCG",
    "COLPAL.NS": "FMCG", "DABUR.NS": "FMCG", "GODFRYPHLP.NS": "FMCG",
    "MARICO.NS": "FMCG", "PATANJALI.NS": "FMCG", "RADICO.NS": "FMCG",
    "UBL.NS": "FMCG",
    # --- Auto & Auto Ancillaries ---
    "BAJAJ-AUTO.NS": "Auto & Auto Ancillaries", "BOSCHLTD.NS": "Auto & Auto Ancillaries", "EICHERMOT.NS": "Auto & Auto Ancillaries",
    "HYUNDAI.NS": "Auto & Auto Ancillaries", "M&M.NS": "Auto & Auto Ancillaries", "MARUTI.NS": "Auto & Auto Ancillaries",
    "MOTHERSON.NS": "Auto & Auto Ancillaries", "TVSMOTOR.NS": "Auto & Auto Ancillaries", "TMCV.NS": "Auto & Auto Ancillaries",
    "TMPV.NS": "Auto & Auto Ancillaries", "APOLLOTYRE.NS": "Auto & Auto Ancillaries", "ASHOKLEY.NS": "Auto & Auto Ancillaries",
    "BALKRISIND.NS": "Auto & Auto Ancillaries", "BHARATFORG.NS": "Auto & Auto Ancillaries", "ENDURANCE.NS": "Auto & Auto Ancillaries",
    "EXIDEIND.NS": "Auto & Auto Ancillaries", "HEROMOTOCO.NS": "Auto & Auto Ancillaries", "MRF.NS": "Auto & Auto Ancillaries",
    "SCHAEFFLER.NS": "Auto & Auto Ancillaries", "TIINDIA.NS": "Auto & Auto Ancillaries", "UNOMINDA.NS": "Auto & Auto Ancillaries",
    "ARE&M.NS": "Auto & Auto Ancillaries", "SONACOMS.NS": "Auto & Auto Ancillaries",
    # --- Consumer Durables ---
    "ASIANPAINT.NS": "Consumer Durables", "TITAN.NS": "Consumer Durables", "ASTRAL.NS": "Consumer Durables",
    "BERGEPAINT.NS": "Consumer Durables", "BLUESTARCO.NS": "Consumer Durables", "DIXON.NS": "Consumer Durables",
    "HAVELLS.NS": "Consumer Durables", "KALYANKJIL.NS": "Consumer Durables", "LGEINDIA.NS": "Consumer Durables",
    "VOLTAS.NS": "Consumer Durables", "AMBER.NS": "Consumer Durables", "CROMPTON.NS": "Consumer Durables",
    "PGEL.NS": "Consumer Durables",
    # --- Consumer Services ---
    "DMART.NS": "Consumer Services", "ETERNAL.NS": "Consumer Services", "INDHOTEL.NS": "Consumer Services",
    "TRENT.NS": "Consumer Services", "NYKAA.NS": "Consumer Services", "ITCHOTELS.NS": "Consumer Services",
    "IRCTC.NS": "Consumer Services", "NAUKRI.NS": "Consumer Services", "JUBLFOOD.NS": "Consumer Services",
    "LENSKART.NS": "Consumer Services", "SWIGGY.NS": "Consumer Services", "VMM.NS": "Consumer Services",
    "DELHIVERY.NS": "Consumer Services",
    # --- Infrastructure ---
    "ADANIPORTS.NS": "Infrastructure", "INDIGO.NS": "Infrastructure", "LT.NS": "Infrastructure",
    "AIAENG.NS": "Infrastructure", "APARINDS.NS": "Infrastructure", "BDL.NS": "Infrastructure",
    "BHEL.NS": "Infrastructure", "COCHINSHIP.NS": "Infrastructure", "CONCOR.NS": "Infrastructure",
    "ESCORTS.NS": "Infrastructure", "GMRAIRPORT.NS": "Infrastructure", "JSWINFRA.NS": "Infrastructure",
    "KEI.NS": "Infrastructure", "POLYCAB.NS": "Infrastructure", "RVNL.NS": "Infrastructure",
    "SUPREMEIND.NS": "Infrastructure", "THERMAX.NS": "Infrastructure", "INOXWIND.NS": "Infrastructure",
    "KAYNES.NS": "Infrastructure", "KEC.NS": "Infrastructure", "NBCC.NS": "Infrastructure",
    "WELCORP.NS": "Infrastructure",
    # --- Cement ---
    "AMBUJACEM.NS": "Cement", "GRASIM.NS": "Cement", "SHREECEM.NS": "Cement",
    "ULTRACEMCO.NS": "Cement", "ACC.NS": "Cement", "DALBHARAT.NS": "Cement",
    "JKCEMENT.NS": "Cement",
    # --- Chemicals ---
    "PIDILITIND.NS": "Chemicals", "SOLARINDS.NS": "Chemicals", "COROMANDEL.NS": "Chemicals",
    "GODREJIND.NS": "Chemicals", "FLUOROCHEM.NS": "Chemicals", "LINDEINDIA.NS": "Chemicals",
    "PIIND.NS": "Chemicals", "SRF.NS": "Chemicals", "UPL.NS": "Chemicals",
    "HSCL.NS": "Chemicals", "NAVINFLUOR.NS": "Chemicals", "TATACHEM.NS": "Chemicals",
    # --- Oil & Gas ---
    "BPCL.NS": "Oil & Gas", "GAIL.NS": "Oil & Gas", "IOC.NS": "Oil & Gas",
    "ONGC.NS": "Oil & Gas", "RELIANCE.NS": "Oil & Gas", "ATGL.NS": "Oil & Gas",
    "HINDPETRO.NS": "Oil & Gas", "OIL.NS": "Oil & Gas", "PETRONET.NS": "Oil & Gas",
    "AEGISLOG.NS": "Oil & Gas", "CASTROLIND.NS": "Oil & Gas", "IGL.NS": "Oil & Gas",
    # --- Power ---
    "ADANIENSOL.NS": "Power", "ADANIGREEN.NS": "Power", "ADANIPOWER.NS": "Power",
    "NTPC.NS": "Power", "POWERGRID.NS": "Power", "TATAPOWER.NS": "Power",
    "JSWENERGY.NS": "Power", "NHPC.NS": "Power", "NLCINDIA.NS": "Power",
    "NTPCGREEN.NS": "Power", "SJVN.NS": "Power", "TORNTPOWER.NS": "Power",
    "CESC.NS": "Power", "RPOWER.NS": "Power",
    # --- Metals & Mining ---
    "ADANIENT.NS": "Metals & Mining", "COALINDIA.NS": "Metals & Mining", "HINDALCO.NS": "Metals & Mining",
    "HINDZINC.NS": "Metals & Mining", "JSWSTEEL.NS": "Metals & Mining", "JINDALSTEL.NS": "Metals & Mining",
    "TATASTEEL.NS": "Metals & Mining", "VEDL.NS": "Metals & Mining", "APLAPOLLO.NS": "Metals & Mining",
    "JSL.NS": "Metals & Mining", "LLOYDSME.NS": "Metals & Mining", "NMDC.NS": "Metals & Mining",
    "NATIONALUM.NS": "Metals & Mining", "SAIL.NS": "Metals & Mining", "HINDCOPPER.NS": "Metals & Mining",
    # --- Realty ---
    "DLF.NS": "Realty", "LODHA.NS": "Realty", "GODREJPROP.NS": "Realty",
    "OBEROIRLTY.NS": "Realty", "PHOENIXLTD.NS": "Realty", "PRESTIGE.NS": "Realty",
    # --- Telecom ---
    "BHARTIARTL.NS": "Telecom", "BHARTIHEXA.NS": "Telecom", "INDUSTOWER.NS": "Telecom",
    "TATACOMM.NS": "Telecom", "IDEA.NS": "Telecom",
    # --- Textiles ---
    "KPRMILL.NS": "Textiles", "PAGEIND.NS": "Textiles",
    # --- Capital Goods ---
    "ABB.NS": "Capital Goods", "BEL.NS": "Capital Goods", "CGPOWER.NS": "Capital Goods",
    "CUMMINSIND.NS": "Capital Goods", "HAL.NS": "Capital Goods", "MAZDOCK.NS": "Capital Goods",
    "ENRIN.NS": "Capital Goods", "SIEMENS.NS": "Capital Goods", "GVT&D.NS": "Capital Goods",
    "POWERINDIA.NS": "Capital Goods", "HONAUT.NS": "Capital Goods", "PREMIERENE.NS": "Capital Goods",
    "SUZLON.NS": "Capital Goods", "WAAREEENER.NS": "Capital Goods",
    # --- Other ---
    "3MINDIA.NS": "Other",
}
