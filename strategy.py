"""
Mathematical scoring + shortlist generation.

Scans the configured universe (default: config.SCAN_UNIVERSE, i.e. the
Nifty 100 plus Gold/Silver) and scores every ticker. The shortlist is the
SHORTLIST_MAX_SIZE highest-scoring Nifty 100 stocks PLUS Gold and Silver
(TATAGOLD.NS / TATASILV.NS), which are always included regardless of score.

SCORE FORMULA (max 105, computed in score_setup()):

  1. Fibonacci proximity (max SCORE_FIB_KEY_LEVEL = 30):
       30 pts if price is within FIB_PROXIMITY_PCT of a KEY level (50%/61.8%)
       15 pts (SCORE_FIB_OTHER_LEVEL) if within FIB_PROXIMITY_PCT of any
            other level (0%/23.6%/38.2%/100%)
        0 pts otherwise (mutually exclusive - only the nearest level counts)

  2. RSI recovering (max SCORE_RSI_EXTREME = 20):
       Scores ONLY when RSI(14) < 50 AND rising vs. the previous session -
       a low-but-still-falling RSI is not a confirmed reversal, so it earns
       nothing; the "rising" check is a gate, not its own graduated factor.
       Once gated in, the point value is driven by the RSI snapshot itself:
       full 20 pts at RSI <= RSI_OVERSOLD, tapering linearly to 0 pts as RSI
       approaches 50 - so the snapshot value (how oversold) carries most of
       the weight, and direction only confirms whether it counts at all.
        0 pts if RSI >= 50, or if RSI < 50 but still falling

  3. MACD confirmed bullish crossover (max SCORE_MACD_PROXIMITY = 28):
       Scores ONLY when MACD has ACTUALLY crossed above its Signal line
       within the last MACD_CROSSOVER_LOOKBACK_DAYS sessions - "converging
       toward" a crossover that hasn't happened yet is a prediction, not a
       confirmed reversal, so it earns nothing (same confirm-don't-predict
       principle as RSI above). Full 28 pts for a crossover today, tapering
       linearly to 0 pts the further back (up to the lookback window) it
       occurred - a fresher confirmation carries more weight.
        0 pts if no bullish crossover occurred in that window

  4. Swing potential (max SCORE_SWING_POTENTIAL = 20):
       Independent of whether RSI/MACD are confirming a reversal right now -
       asks whether the stock is even CAPABLE of a worthwhile (7-8%+) swing
       move. Graduated (not gated), blending two things: room between
       current price and the recent swing high (SWING_POSSIBILITY_WEIGHT,
       primary) and this stock's own median historical up-leg size via a
       zigzag scan over SWING_LOOKBACK_DAYS (SWING_TRACK_RECORD_WEIGHT,
       secondary "track record" check). Full credit at/above SWING_TARGET_PCT,
       scaling down below that.

  5. Sector trend alignment (max SCORE_SECTOR_TREND = 7, small/secondary):
       7 pts if the stock's bullish bias (from an RSI-recovering or
            confirmed-MACD-crossover signal) is confirmed by its sector's
            average SECTOR_TREND_LOOKBACK_DAYS-day return also moving up by
            at least SECTOR_TREND_THRESHOLD
        0 pts otherwise (neutral bias, or sector trend too small/opposing)

  Maximum = 30 + 20 + 28 + 20 + 7 = 105. Fibonacci (key level) and MACD
  confirmed crossover are the two largest components - the primary drivers
  of the score. Volume is NOT scored (see score_setup) - it's shown as
  descriptive context only, since a volume spike with no RSI/MACD signal
  behind it was letting weak setups qualify for the shortlist on volume
  alone.

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
                "higher-probability reaction zone for a 3-month swing entry or exit."
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

    # --- RSI (max SCORE_RSI_EXTREME) ---------------------------------------------
    # Scores ONLY when RSI is below neutral (50) AND rising vs the previous
    # session - a low-but-still-falling RSI isn't a confirmed reversal, just
    # momentum that hasn't bottomed yet, so it earns nothing. A low-and-rising
    # RSI confirms a recovery may be underway. That rising check is a gate,
    # not an independently graduated factor - once it's met, the point value
    # is driven primarily by the RSI snapshot itself (full credit at/below
    # RSI_OVERSOLD, tapering linearly to 0 as RSI approaches 50), so the
    # snapshot value carries most of the weight and the "swing" (direction)
    # only confirms whether the snapshot counts at all.
    rsi = analysis["rsi"]
    rsi_rising = analysis["rsi_direction"] == "up"
    if rsi < 50 and rsi_rising:
        if rsi <= config.RSI_OVERSOLD:
            rsi_score = config.SCORE_RSI_EXTREME
        else:
            rsi_score = config.SCORE_RSI_EXTREME * (50 - rsi) / (50 - config.RSI_OVERSOLD)
        score += rsi_score
        bias = "bullish"
        reasons.append(
            f"RSI(14) is at {rsi:.1f} and rising - below neutral (50) with "
            "upward momentum, suggesting a recovery off a low is underway, a "
            "common entry trigger for a swing long."
        )
    elif rsi < 50:
        reasons.append(
            f"RSI(14) is at {rsi:.1f} - below neutral but still falling, so "
            "this isn't a confirmed reversal yet and doesn't contribute to "
            "the score."
        )
    elif rsi >= config.RSI_OVERBOUGHT:
        reasons.append(
            f"RSI(14) is overbought at {rsi:.1f} (at or above the "
            f"{config.RSI_OVERBOUGHT} threshold) - no buy signal here, so this "
            "does not contribute to the score."
        )
    else:
        reasons.append(
            f"RSI(14) is at {rsi:.1f} - neutral-to-elevated territory, with no "
            "oversold-and-recovering setup currently in play."
        )

    # --- MACD (max SCORE_MACD_PROXIMITY) -----------------------------------------
    # Three states, not a single on/off check:
    #   1. Below Signal but converging (histogram shrinking toward zero) -
    #      an early, not-yet-confirmed setup worth watching. Smaller credit
    #      (SCORE_MACD_EARLY) since the reversal hasn't happened yet.
    #   2. Actually crossed above Signal within MACD_CROSSOVER_LOOKBACK_DAYS -
    #      a CONFIRMED reversal, not a prediction. Full weight for a crossover
    #      today, tapering off the further back it happened.
    #   3. Already crossed but MACD is well above the zero line - a
    #      mid-uptrend pullback-and-continue rather than a fresh reversal,
    #      much of the move may already be behind you, so state 2's credit
    #      gets discounted the further above zero it already is (relative to
    #      this stock's own typical histogram magnitude, macd_hist_scale -
    #      "how far above zero counts as too far" is per-stock, not fixed).
    bars_ago = analysis["macd_bullish_crossover_bars_ago"]
    if bars_ago is not None:
        hist_scale = analysis["macd_hist_scale"]
        if hist_scale > 0:
            extension_factor = max(0.0, 1 - max(0.0, analysis["macd_line"]) / hist_scale)
        else:
            extension_factor = 1.0 if analysis["macd_line"] <= 0 else 0.0
        macd_score = (
            config.SCORE_MACD_PROXIMITY
            * (1 - bars_ago / config.MACD_CROSSOVER_LOOKBACK_DAYS)
            * extension_factor
        )
        score += macd_score
        if bias == "neutral":
            bias = "bullish"
        when = "today" if bars_ago == 0 else f"{bars_ago} session(s) ago"
        if extension_factor < 0.5:
            extension_note = (
                " - though MACD is already well above zero, so this looks more like "
                "a mid-uptrend pullback-and-continue than a fresh reversal, discounted "
                "accordingly."
            )
        else:
            extension_note = ""
        reasons.append(
            f"MACD crossed above its Signal line (confirmed bullish crossover) "
            f"{when}, an actual reversal rather than an anticipated one{extension_note}"
        )
    elif analysis["macd_hist"] < 0 and analysis["macd_hist_rising_5d"]:
        score += config.SCORE_MACD_EARLY
        if bias == "neutral":
            bias = "bullish"
        reasons.append(
            f"MACD is still below its Signal line but the histogram "
            f"({analysis['macd_hist']:.3f}) has been rising over the last 5 "
            "sessions - an early, not-yet-confirmed setup worth watching for "
            "a crossover, so it earns partial credit."
        )
    else:
        reasons.append(
            f"No confirmed or converging bullish MACD setup in the last "
            f"{config.MACD_CROSSOVER_LOOKBACK_DAYS} sessions - {analysis['macd_pattern'].split('. ')[0].strip()}."
        )

    # --- Swing potential (max SCORE_SWING_POTENTIAL) -----------------------------
    # Independent of whether RSI/MACD are confirming a reversal right now -
    # this asks whether the stock is even CAPABLE of a worthwhile swing-trade
    # move. Blends two things (graduated, not gated - no cliff to 0):
    #   - "possibility": room between current price and the recent swing high
    #     (the realistic near-term target), weighted primary
    #   - "track record": this stock's own median historical up-leg size
    #     (zigzag scan over SWING_LOOKBACK_DAYS), weighted secondary
    # Full credit at/above SWING_TARGET_PCT (the 7-8% target), scaling down
    # below that - not a hard requirement, just a smaller contribution.
    room_to_target_pct = (
        max(0.0, (analysis["fib_high"] - analysis["close"]) / analysis["close"])
        if analysis["close"] else 0.0
    )
    track_record_pct = analysis["median_up_swing_pct"]
    swing_potential_pct = (
        config.SWING_POSSIBILITY_WEIGHT * room_to_target_pct
        + config.SWING_TRACK_RECORD_WEIGHT * track_record_pct
    )
    swing_score = min(
        config.SCORE_SWING_POTENTIAL,
        config.SCORE_SWING_POTENTIAL * swing_potential_pct / config.SWING_TARGET_PCT,
    )
    score += swing_score
    verdict = (
        "clears the bar for a worthwhile swing" if swing_potential_pct >= config.SWING_TARGET_PCT
        else "below the bar, so even a confirmed reversal here may not deliver a full swing-trade move"
    )
    reasons.append(
        f"Swing potential: {room_to_target_pct * 100:.1f}% room to the recent swing high, "
        f"blended with a {track_record_pct * 100:.1f}% median historical up-leg over the last "
        f"{config.SWING_LOOKBACK_DAYS} sessions, against a {config.SWING_TARGET_PCT * 100:.1f}% "
        f"target - {verdict}."
    )

    # --- Volume (always included, informational only - not scored) --------------
    # Volume surges were previously worth SCORE_VOLUME points, but that let
    # a stock qualify for the shortlist on a volume spike alone even with no
    # real RSI/MACD signal (see strategy discussion) - now purely descriptive.
    volume_ratio = analysis["volume_ratio"]
    if volume_ratio >= config.VOLUME_SURGE_RATIO:
        reasons.append(
            f"Volume is running at {volume_ratio:.2f}x its "
            f"{config.VOLUME_AVG_WINDOW}-day average ({analysis['avg_volume_20']:,.0f} "
            "shares) - above-average participation, though volume alone doesn't "
            "score a setup."
        )
    else:
        reasons.append(
            f"Volume is {volume_ratio:.2f}x its {config.VOLUME_AVG_WINDOW}-day average "
            "- no unusual surge."
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
