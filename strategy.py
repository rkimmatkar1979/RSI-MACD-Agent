"""
Mathematical scoring + shortlist generation.

Scans the configured universe (default: config.SCAN_UNIVERSE, i.e. the
Nifty 100 plus Gold/Silver) and scores every ticker. The shortlist is the
SHORTLIST_MAX_SIZE highest-scoring Nifty 100 stocks PLUS Gold and Silver
(TATAGOLD.NS / TATASILV.NS), which are always included regardless of score.

SCORE FORMULA (max 100, computed in score_setup()):

  1. Fibonacci proximity (max SCORE_FIB_KEY_LEVEL = 30):
       30 pts if price is within FIB_PROXIMITY_PCT of a KEY level (50%/61.8%)
       15 pts (SCORE_FIB_OTHER_LEVEL) if within FIB_PROXIMITY_PCT of any
            other level (0%/23.6%/38.2%/100%)
        0 pts otherwise (mutually exclusive - only the nearest level counts)

  2. RSI extreme (max SCORE_RSI_EXTREME = 20):
       20 pts if RSI(14) <= RSI_OVERSOLD or RSI(14) >= RSI_OVERBOUGHT
        0 pts otherwise (neutral RSI)

  3. MACD crossover proximity (max SCORE_MACD_PROXIMITY = 28):
       28 pts if the MACD histogram is small AND shrinking relative to its
            recent (20-bar) average magnitude - i.e. converging toward a
            crossover
        0 pts otherwise

  4. Volume confirmation (max SCORE_VOLUME = 15):
       15 pts if latest volume >= VOLUME_SURGE_RATIO x its 20-day average
        0 pts otherwise

  5. Sector trend alignment (max SCORE_SECTOR_TREND = 7, small/secondary):
       7 pts if the stock's directional bias (bullish from an RSI-oversold
            or bullish-converging-MACD signal; bearish from RSI-overbought
            or bearish-converging-MACD) is confirmed by its sector's average
            SECTOR_TREND_LOOKBACK_DAYS-day return moving the same direction
            by at least SECTOR_TREND_THRESHOLD
        0 pts otherwise (neutral bias, or sector trend too small/opposing)

  Maximum = 30 + 20 + 28 + 15 + 7 = 100. Fibonacci (key level) and MACD
  crossover proximity are the two largest components - the primary drivers
  of the score.

Every ticker also gets baseline descriptive context (volume vs. its 20-day
average, 52-week high proximity, sector trend, and a one-line MACD trend
summary) regardless of score, so the shortlist always reads as a complete
top-N list rather than a sparse one.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import config
from ta_engine import analyze_ticker

SHORTLIST_COLUMNS = [
    "ticker", "sector", "close", "rsi", "macd_line", "macd_signal", "macd_hist",
    "macd_hist_direction", "nearest_fib_level", "nearest_fib_price", "fib_distance_pct",
    "fib_high", "fib_low", "week52_high", "week52_low", "pct_from_52w_high",
    "macd_pattern", "volume_ratio", "avg_volume_20", "buy_pct", "sell_pct",
    "sector_trend_pct", "prev_session_date", "prev_session_open", "prev_session_close",
    "score", "reasons",
]


def score_setup(analysis, sector_trend_pct):
    """
    Computes a composite score (max 100) for a single ticker's analysis dict.
    See the module docstring for the full score formula.

    Returns (score, reasons) where reasons is a list of human-readable,
    swing-trading-oriented explanations. Reasons always include baseline
    52-week-high, MACD-trend, and sector-trend context, plus an entry for
    each scoring condition that fired.

    `sector_trend_pct` is this ticker's sector's average return over
    config.SECTOR_TREND_LOOKBACK_DAYS, computed across the whole scan in
    generate_shortlist().
    """
    score = 0
    reasons = []
    bias = "neutral"  # set to "bullish"/"bearish" by RSI/MACD below

    # --- Fibonacci proximity (max 30) -------------------------------------
    if analysis["fib_distance_pct"] <= config.FIB_PROXIMITY_PCT:
        level_name = analysis["nearest_fib_level"]
        level_pct = float(level_name.strip("%")) / 100
        is_key_level = any(abs(level_pct - k) < 1e-6 for k in config.FIB_KEY_LEVELS)

        if is_key_level:
            score += config.SCORE_FIB_KEY_LEVEL
            reasons.append(
                f"Price ({analysis['close']:.2f}) is within "
                f"{analysis['fib_distance_pct'] * 100:.2f}% of the key {level_name} "
                f"Fibonacci retracement level ({analysis['nearest_fib_price']:.2f}) of "
                f"its {config.FIB_LOOKBACK_DAYS}-day range - 50%/61.8% retracements "
                "often act as support or resistance, making this a "
                "higher-probability reaction zone for a 2-3 week swing entry or exit."
            )
        else:
            score += config.SCORE_FIB_OTHER_LEVEL
            reasons.append(
                f"Price ({analysis['close']:.2f}) is within "
                f"{analysis['fib_distance_pct'] * 100:.2f}% of the {level_name} "
                f"Fibonacci retracement level ({analysis['nearest_fib_price']:.2f}) of "
                f"its {config.FIB_LOOKBACK_DAYS}-day range - a secondary level worth "
                "watching for a price reaction."
            )

    # --- RSI (max 25) ----------------------------------------------------------
    rsi = analysis["rsi"]
    if rsi <= config.RSI_OVERSOLD:
        score += config.SCORE_RSI_EXTREME
        bias = "bullish"
        reasons.append(
            f"RSI(14) is oversold at {rsi:.1f} (at or below the "
            f"{config.RSI_OVERSOLD} threshold), suggesting selling pressure may be "
            "exhausted and the stock could be due for a short-term bounce - a "
            "common entry trigger for a swing long."
        )
    elif rsi >= config.RSI_OVERBOUGHT:
        score += config.SCORE_RSI_EXTREME
        bias = "bearish"
        reasons.append(
            f"RSI(14) is overbought at {rsi:.1f} (at or above the "
            f"{config.RSI_OVERBOUGHT} threshold), suggesting buying momentum may be "
            "overextended and the stock could be due for a pullback - relevant for "
            "booking profits or a swing short."
        )
    else:
        reasons.append(
            f"RSI(14) is at {rsi:.1f} - neutral territory, with no oversold/"
            "overbought extreme currently in play."
        )

    # --- MACD crossover proximity (max 20) --------------------------------------
    if analysis["macd_crossover_proximity"]:
        score += config.SCORE_MACD_PROXIMITY
        direction = "bullish" if analysis["macd_hist"] < 0 else "bearish"
        implication = "an upside reversal" if direction == "bullish" else "a downside reversal"
        if bias == "neutral":
            bias = direction
        reasons.append(
            f"MACD histogram ({analysis['macd_hist']:.3f}) is converging toward a "
            f"{direction} crossover, which could signal {implication} forming."
        )

    # --- Volume confirmation (max 15) --------------------------------------------
    volume_ratio = analysis["volume_ratio"]
    if volume_ratio >= config.VOLUME_SURGE_RATIO:
        score += config.SCORE_VOLUME
        reasons.append(
            f"Volume is running at {volume_ratio:.2f}x its "
            f"{config.VOLUME_AVG_WINDOW}-day average ({analysis['avg_volume_20']:,.0f} "
            "shares), confirming the move with above-average participation."
        )
    else:
        reasons.append(
            f"Volume is {volume_ratio:.2f}x its {config.VOLUME_AVG_WINDOW}-day average "
            "- no unusual surge, so treat the move as lower-conviction until volume "
            "confirms it."
        )

    # --- Sector trend alignment (max 10, small/secondary factor) -----------------
    sector = analysis["sector"]
    lookback = config.SECTOR_TREND_LOOKBACK_DAYS
    if bias == "bullish" and sector_trend_pct >= config.SECTOR_TREND_THRESHOLD:
        score += config.SCORE_SECTOR_TREND
        reasons.append(
            f"The {sector} sector is up {sector_trend_pct * 100:.1f}% over the last "
            f"{lookback} sessions, a tailwind that supports this stock's bullish setup."
        )
    elif bias == "bearish" and sector_trend_pct <= -config.SECTOR_TREND_THRESHOLD:
        score += config.SCORE_SECTOR_TREND
        reasons.append(
            f"The {sector} sector is down {abs(sector_trend_pct) * 100:.1f}% over the "
            f"last {lookback} sessions, reinforcing this stock's bearish setup."
        )
    else:
        direction_word = "up" if sector_trend_pct >= 0 else "down"
        reasons.append(
            f"The {sector} sector is {direction_word} {abs(sector_trend_pct) * 100:.1f}% "
            f"over the last {lookback} sessions - no additional sector-level "
            "confirmation for this setup currently."
        )

    # --- 52-week-high context (always included) ------------------------------
    pct_from_high = analysis["pct_from_52w_high"]
    if pct_from_high <= 0.03:
        proximity_note = "near its 52-week high - a potential breakout setup on a new high"
    elif pct_from_high >= 0.30:
        proximity_note = (
            "well off its 52-week high - a deep retracement, so confirm the "
            "downtrend has stabilized before entry"
        )
    else:
        proximity_note = "trading within its broader 52-week range"
    reasons.append(
        f"Price is {pct_from_high * 100:.1f}% below its 52-week high of "
        f"{analysis['week52_high']:.2f} ({proximity_note})."
    )

    # --- MACD trend summary (always included) ---------------------------------
    macd_summary = analysis["macd_pattern"].split(". ")[0].strip()
    if macd_summary and not macd_summary.endswith("."):
        macd_summary += "."
    reasons.append(macd_summary)

    return score, reasons


def generate_shortlist(tickers=None, progress_callback=None):
    """
    Scans `tickers` (defaults to config.SCAN_UNIVERSE, i.e. the Nifty 100
    plus Gold/Silver), scores every one, and returns a DataFrame sorted by
    score (highest first).

    The shortlist is the top SHORTLIST_MAX_SIZE Nifty 100 stocks by score,
    PLUS Gold (TATAGOLD.NS) and Silver (TATASILV.NS), which are always
    included regardless of score. Setups scoring below SHORTLIST_MIN_SCORE
    that are only included to fill out the stock portion of the list get an
    extra "weaker/exploratory setup" note; Gold/Silver get a "tracked
    permanently" note instead.

    Tickers are fetched/analyzed concurrently (config.SCAN_MAX_WORKERS
    threads) since each `analyze_ticker` call is dominated by a network
    round-trip to Yahoo Finance - this cuts a full scan from minutes down to
    roughly total_time / SCAN_MAX_WORKERS.

    `progress_callback(i, total, ticker)` is invoked as each ticker
    finishes, if provided - used by the UI / CLI to show scan progress.
    Order of completion (and therefore of progress callbacks) is not
    guaranteed to match `tickers`.
    """
    if tickers is None:
        tickers = config.SCAN_UNIVERSE

    results = []
    total = len(tickers)
    completed = 0

    with ThreadPoolExecutor(max_workers=config.SCAN_MAX_WORKERS) as executor:
        future_to_ticker = {executor.submit(analyze_ticker, t): t for t in tickers}

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            completed += 1

            try:
                analysis = future.result()
            except Exception as e:
                print(f"[strategy] Failed to analyze {ticker}: {e}")
                analysis = None

            if analysis is not None:
                results.append(analysis)

            if progress_callback:
                progress_callback(completed, total, ticker)

    if not results:
        return pd.DataFrame(columns=SHORTLIST_COLUMNS)

    # --- Sector trend: average each sector's recent return across every
    # successfully-analyzed ticker in that sector (small/secondary signal,
    # see SCORE_SECTOR_TREND). -----------------------------------------------
    sector_returns = {}
    for r in results:
        sector_returns.setdefault(r["sector"], []).append(r["return_nd"])
    sector_trend = {sector: sum(vals) / len(vals) for sector, vals in sector_returns.items()}

    for r in results:
        sector_trend_pct = sector_trend[r["sector"]]
        score, reasons = score_setup(r, sector_trend_pct)
        r["sector_trend_pct"] = sector_trend_pct
        r["score"] = score
        r["reasons"] = reasons

    df = pd.DataFrame(results)

    # --- Gold/Silver are permanently tracked, regardless of score -----------
    is_commodity = df["ticker"].isin(config.GOLD_SILVER_TICKERS)
    commodities_df = df[is_commodity]
    stocks_df = df[~is_commodity]

    top_stocks = stocks_df.sort_values("score", ascending=False).head(config.SHORTLIST_MAX_SIZE)
    shortlist_df = pd.concat([top_stocks, commodities_df], ignore_index=True)
    shortlist_df = shortlist_df.sort_values("score", ascending=False).reset_index(drop=True)

    for idx, row in shortlist_df.iterrows():
        if row["ticker"] in config.GOLD_SILVER_TICKERS:
            shortlist_df.at[idx, "reasons"] = row["reasons"] + [
                "Note: tracked permanently as part of the watchlist (Gold/Silver), "
                "included in every scan regardless of score."
            ]
        elif row["score"] < config.SHORTLIST_MIN_SCORE:
            shortlist_df.at[idx, "reasons"] = row["reasons"] + [
                f"Note: composite score ({row['score']}/100) is below the "
                f"high-confidence threshold ({config.SHORTLIST_MIN_SCORE}/100) - "
                "included to complete the top setups list, so treat this as a "
                "weaker or exploratory setup."
            ]

    return shortlist_df[SHORTLIST_COLUMNS]
