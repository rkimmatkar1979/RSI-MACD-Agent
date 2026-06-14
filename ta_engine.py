"""
Technical analysis engine: data fetching + indicator math.

Provides:
  - fetch_data(): cached OHLCV download via yfinance
  - calculate_rsi(), calculate_macd(): indicators computed directly with
    pandas (Wilder's RSI, EMA-based MACD)
  - calculate_fibonacci_levels(): rolling peak/trough retracement levels
  - analyze_ticker(): full pipeline for a single ticker -> flat dict of
    the latest indicator values, used by strategy.py
"""

from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import pytz
import streamlit as st
import yfinance as yf

import config


def is_market_open():
    """Returns True if it is currently within NSE trading hours on a weekday (IST)."""
    tz = pytz.timezone(config.MARKET_TIMEZONE)
    now = datetime.now(tz)

    if now.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    open_h, open_m = (int(x) for x in config.MARKET_OPEN_TIME.split(":"))
    close_h, close_m = (int(x) for x in config.MARKET_CLOSE_TIME.split(":"))
    open_t = dtime(open_h, open_m)
    close_t = dtime(close_h, close_m)

    return open_t <= now.time() <= close_t


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_data(ticker, period=config.DATA_PERIOD, interval=config.DATA_INTERVAL):
    """
    Downloads OHLCV history for a ticker.

    Returns a DataFrame indexed by date with Open/High/Low/Close/Volume
    columns, or None if the data could not be fetched or is insufficient
    for the configured indicator lookbacks.
    """
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            print(f"[ta_engine] No data returned for {ticker}")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            if "Close" in df.columns.get_level_values(0):
                df.columns = df.columns.get_level_values(0)
            else:
                df.columns = df.columns.get_level_values(1)

        df = df.dropna(how="any")

        min_required = config.FIB_LOOKBACK_DAYS + config.MACD_SLOW
        if len(df) < min_required:
            print(f"[ta_engine] Insufficient history for {ticker} "
                  f"({len(df)} rows, need >= {min_required})")
            return None

        return df
    except Exception as e:
        print(f"[ta_engine] Failed to fetch data for {ticker}: {e}")
        return None


def calculate_rsi(df, period=config.RSI_PERIOD):
    """
    Adds an 'RSI' column to a copy of df using Wilder's smoothing method:

      RS  = (Wilder-smoothed average gain) / (Wilder-smoothed average loss)
      RSI = 100 - (100 / (1 + RS))
    """
    df = df.copy()
    delta = df["Close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing == an EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # Where average loss is 0 (pure up-trend), RSI is defined as 100.
    df.loc[avg_loss == 0, "RSI"] = 100.0
    return df


def calculate_macd(df, fast=config.MACD_FAST, slow=config.MACD_SLOW, signal=config.MACD_SIGNAL):
    """
    Adds 'MACD', 'MACD_SIGNAL', 'MACD_HIST' columns to a copy of df:

      MACD line   = EMA(close, fast) - EMA(close, slow)
      Signal line = EMA(MACD line, signal)
      Histogram   = MACD line - Signal line
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIGNAL"]
    return df


def calculate_fibonacci_levels(df, lookback=config.FIB_LOOKBACK_DAYS):
    """
    Computes auto-Fibonacci retracement levels from the rolling peak high
    and trough low over the most recent `lookback` bars.

    Returns (levels_dict, peak_high, trough_low) where levels_dict maps
    a label like "61.8%" to the corresponding price level.
    """
    window = df.tail(lookback)
    peak = float(window["High"].max())
    trough = float(window["Low"].min())
    diff = peak - trough

    levels = {"0.0%": peak, "100.0%": trough}
    for lvl in config.FIB_LEVELS:
        levels[f"{lvl * 100:.1f}%"] = peak - lvl * diff

    return levels, peak, trough


@st.cache_data(ttl=3600, show_spinner="📊 Fetching chart data...")
def get_chart_data(ticker):
    """
    Fetches OHLCV data and computes RSI/MACD/Fibonacci levels for a ticker
    in one cached call, keyed only by the ticker symbol - avoids
    re-running this on every Streamlit rerun (checkbox toggles, row
    selection, etc.) for the same ticker.

    Returns (df, levels, peak, trough), or (None, None, None, None) if the
    underlying price data could not be fetched.
    """
    df = fetch_data(ticker)
    if df is None:
        return None, None, None, None

    df = calculate_rsi(df)
    df = calculate_macd(df)
    levels, peak, trough = calculate_fibonacci_levels(df)
    return df, levels, peak, trough


def nearest_fib_level(price, levels):
    """Returns (level_name, level_price, distance_pct) for the level closest to price."""
    nearest_name, nearest_price = min(levels.items(), key=lambda kv: abs(price - kv[1]))
    distance_pct = abs(price - nearest_price) / price if price else np.nan
    return nearest_name, nearest_price, distance_pct


def macd_crossover_proximity(df):
    """
    Returns True if the MACD histogram is small relative to its recent
    magnitude AND still shrinking - i.e. the MACD and signal lines are
    converging toward a crossover.
    """
    hist = df["MACD_HIST"].dropna()
    if len(hist) < 10:
        return False

    latest = hist.iloc[-1]
    prev = hist.iloc[-2]
    recent_avg_abs = hist.tail(20).abs().mean()

    if recent_avg_abs == 0 or np.isnan(recent_avg_abs):
        return False

    is_small = abs(latest) < recent_avg_abs * config.MACD_CROSSOVER_PROXIMITY_FACTOR
    is_converging = abs(latest) < abs(prev)
    return bool(is_small and is_converging)


def describe_macd_pattern(df, lookback=10):
    """
    Returns a multi-sentence, plain-English description of the MACD line's
    current behaviour: its position relative to the signal line and the
    zero line, recent histogram momentum, and any recent or imminent
    crossover. Used to give both the dashboard and the AI analyst richer
    context than the raw MACD numbers alone.
    """
    macd = df["MACD"].dropna()
    signal = df["MACD_SIGNAL"].dropna()
    hist = df["MACD_HIST"].dropna()

    if len(hist) < 6:
        return "Not enough price history to characterize the MACD pattern."

    latest_macd = float(macd.iloc[-1])
    latest_signal = float(signal.iloc[-1])
    latest_hist = float(hist.iloc[-1])

    sentences = []

    if latest_hist >= 0:
        sentences.append(
            f"MACD ({latest_macd:.2f}) is currently above its Signal line "
            f"({latest_signal:.2f}) by {latest_hist:.2f}, indicating bullish momentum."
        )
    else:
        sentences.append(
            f"MACD ({latest_macd:.2f}) is currently below its Signal line "
            f"({latest_signal:.2f}) by {abs(latest_hist):.2f}, indicating bearish momentum."
        )

    if latest_macd >= 0:
        sentences.append("Both lines sit above the zero line, consistent with a broader uptrend.")
    else:
        sentences.append("Both lines sit below the zero line, consistent with a broader downtrend.")

    recent = hist.tail(5)
    if recent.iloc[-1] > recent.iloc[0]:
        sentences.append(
            "The histogram has been rising over the last 5 sessions, suggesting "
            "momentum is strengthening in the current direction."
        )
    else:
        sentences.append(
            "The histogram has been falling over the last 5 sessions, suggesting "
            "momentum is weakening (or a reversal may be developing)."
        )

    diff = (macd - signal).tail(lookback + 1)
    crossover_found = False
    for i in range(len(diff) - 1, 0, -1):
        prev_val, curr_val = diff.iloc[i - 1], diff.iloc[i]
        if prev_val == 0:
            continue
        if (prev_val < 0) != (curr_val < 0):
            bars_ago = len(diff) - 1 - i
            when = "today" if bars_ago == 0 else f"{bars_ago} session(s) ago"
            direction = (
                "bullish (MACD crossed above Signal)" if curr_val > 0
                else "bearish (MACD crossed below Signal)"
            )
            sentences.append(f"A {direction} crossover occurred {when}.")
            crossover_found = True
            break

    if not crossover_found and macd_crossover_proximity(df):
        sentences.append(
            "The MACD and Signal lines are converging and appear close to a crossover."
        )

    return " ".join(sentences)


def calculate_volume_metrics(df, window=config.VOLUME_AVG_WINDOW):
    """
    Compares the latest session's volume to its rolling average.

    Returns (latest_volume, avg_volume, volume_ratio), where volume_ratio
    is latest_volume / avg_volume (e.g. 1.5 == 50% above the rolling
    average, confirming the move with above-average participation).
    """
    latest_volume = float(df["Volume"].iloc[-1])
    avg_volume = float(df["Volume"].tail(window).mean())
    volume_ratio = latest_volume / avg_volume if avg_volume else float("nan")
    return latest_volume, avg_volume, volume_ratio


def calculate_return(df, window=config.SECTOR_TREND_LOOKBACK_DAYS):
    """
    Returns the close-to-close percentage return (decimal fraction) over
    the last `window` trading sessions, e.g. 0.02 == +2%. Used as each
    stock's contribution to its sector's average "current trend".
    """
    if len(df) <= window:
        return 0.0
    latest_close = float(df["Close"].iloc[-1])
    past_close = float(df["Close"].iloc[-1 - window])
    return (latest_close - past_close) / past_close if past_close else 0.0


def calculate_buy_sell_pressure(df, window=config.BUY_SELL_PRESSURE_WINDOW):
    """
    Proxy for buy/sell order-flow split, since NSE order-book depth isn't
    available via yfinance: over the last `window` sessions, the share of
    total volume that traded on "buy" days (close >= open) vs "sell" days
    (close < open).

    Returns (buy_pct, sell_pct), each 0-100, summing to 100.
    """
    recent = df.tail(window)
    buy_volume = float(recent.loc[recent["Close"] >= recent["Open"], "Volume"].sum())
    total_volume = float(recent["Volume"].sum())
    buy_pct = (buy_volume / total_volume * 100) if total_volume else 50.0
    sell_pct = 100 - buy_pct
    return buy_pct, sell_pct


def get_reference_session(df):
    """
    Returns (date_str, open, close) for the most recently *completed*
    trading session.

    While the market is open, the latest bar in `df` is today's still-live
    candle, so the reference session is yesterday's (df.iloc[-2]). Once the
    market has closed for the day, today's bar is final, so it becomes the
    reference session (df.iloc[-1]).
    """
    idx = -2 if (is_market_open() and len(df) >= 2) else -1
    row = df.iloc[idx]
    return df.index[idx].strftime("%Y-%m-%d"), float(row["Open"]), float(row["Close"])


def calculate_52week_range(df, close):
    """
    Computes the 52-week (i.e. full fetched history, ~1y) high/low and how
    far the current close is below the 52-week high.

    Returns (week52_high, week52_low, pct_from_52w_high), where
    pct_from_52w_high is a decimal fraction (e.g. 0.0532 == 5.32% below the
    52-week high; 0.0 == currently at the 52-week high).
    """
    week52_high = float(df["High"].max())
    week52_low = float(df["Low"].min())
    pct_from_52w_high = (week52_high - close) / week52_high if week52_high else float("nan")
    return week52_high, week52_low, pct_from_52w_high


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_news(ticker, limit=config.NEWS_HEADLINE_COUNT):
    """
    Fetches recent news headlines for a ticker via yfinance.

    Returns a list of headline strings (newest first, possibly empty).
    Never raises - news availability is best-effort context for the AI
    analyst, not part of the mathematical screen.
    """
    try:
        raw_items = yf.Ticker(ticker).news or []
        headlines = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            # Newer yfinance versions nest article fields under "content".
            content = item.get("content", item)
            title = content.get("title")
            if title:
                headlines.append(title)
            if len(headlines) >= limit:
                break
        return headlines
    except Exception as e:
        print(f"[ta_engine] Failed to fetch news for {ticker}: {e}")
        return []


def analyze_ticker(ticker):
    """
    Runs the full TA pipeline for a single ticker.

    Returns a flat dict of the latest indicator values plus the Fibonacci
    levels dict, or None if data was unavailable / insufficient.
    """
    df = fetch_data(ticker)
    if df is None:
        return None

    try:
        df = calculate_rsi(df)
        df = calculate_macd(df)
        df = df.dropna(subset=["RSI", "MACD", "MACD_SIGNAL", "MACD_HIST"])
        if df.empty:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]
        close = float(latest["Close"])
        levels, peak, trough = calculate_fibonacci_levels(df)
        level_name, level_price, distance_pct = nearest_fib_level(close, levels)
        macd_near_cross = macd_crossover_proximity(df)
        macd_pattern = describe_macd_pattern(df)
        macd_hist_direction = "up" if float(latest["MACD_HIST"]) >= float(prev["MACD_HIST"]) else "down"
        week52_high, week52_low, pct_from_52w_high = calculate_52week_range(df, close)
        latest_volume, avg_volume_20, volume_ratio = calculate_volume_metrics(df)
        buy_pct, sell_pct = calculate_buy_sell_pressure(df)
        return_nd = calculate_return(df)
        prev_session_date, prev_session_open, prev_session_close = get_reference_session(df)

        return {
            "ticker": ticker,
            "sector": config.SECTOR_MAP.get(ticker, "Other"),
            "close": close,
            "rsi": float(latest["RSI"]),
            "macd_line": float(latest["MACD"]),
            "macd_signal": float(latest["MACD_SIGNAL"]),
            "macd_hist": float(latest["MACD_HIST"]),
            "macd_hist_direction": macd_hist_direction,
            "nearest_fib_level": level_name,
            "nearest_fib_price": float(level_price),
            "fib_distance_pct": float(distance_pct),
            "fib_high": peak,
            "fib_low": trough,
            "week52_high": week52_high,
            "week52_low": week52_low,
            "pct_from_52w_high": pct_from_52w_high,
            "macd_crossover_proximity": macd_near_cross,
            "macd_pattern": macd_pattern,
            "latest_volume": latest_volume,
            "avg_volume_20": avg_volume_20,
            "volume_ratio": volume_ratio,
            "buy_pct": buy_pct,
            "sell_pct": sell_pct,
            "return_nd": return_nd,
            "prev_session_date": prev_session_date,
            "prev_session_open": prev_session_open,
            "prev_session_close": prev_session_close,
            "fib_levels": levels,
        }
    except Exception as e:
        print(f"[ta_engine] Failed to analyze {ticker}: {e}")
        return None
