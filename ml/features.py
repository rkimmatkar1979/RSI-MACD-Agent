"""
Feature engineering for the ML swing-trading model.

Converts raw OHLCV + quarterly fundamentals into the continuous/ordinal
feature set the model trains and scores on - no composite "points", just the
raw signals the rule-based screener used to threshold/weight by hand.

Every feature at row (ticker, date=T) is computed using only information
available on or before T (rolling/expanding windows, no centering, no
forward fill from the future) - the same code path is safe both for
building the historical training matrix and for scoring today's live
snapshot.

Expected inputs
----------------
price_df : long-format panel, one row per (ticker, trading date):
    ticker, date, open, high, low, close, volume

fundamentals_df : long-format, one row per (ticker, fiscal quarter):
    ticker, quarter_end, promoter_holding_pct, net_profit
    (quarter_end = the official quarter-end date the results cover, e.g.
    2024-03-31 for Q4 FY24 - NOT the date results were announced. See
    build_fundamental_features() for how the publish-date lag is applied.)

index_df (optional) : a benchmark's own OHLCV-shaped history, at minimum
    columns date, close - e.g. Nifty 50 (^NSEI). Used only to compute
    market_relative_momentum_20d; if omitted, that one feature is NaN for
    every row (XGBoost handles this natively) and everything else is
    unaffected.

Note on "days" units: days_since_macd_crossover counts trading SESSIONS
(row positions), matching how MACD_CROSSOVER_LOOKBACK_DAYS is used
elsewhere in this codebase (ta_engine.macd_recent_bullish_crossover). The
triple-barrier label horizon in ml.labeling, by contrast, is calendar days,
per that requirement's spec. Don't mix the two units.
"""

from collections import deque

import numpy as np
import pandas as pd

import config

RSI_PERIOD = config.RSI_PERIOD
MACD_FAST = config.MACD_FAST
MACD_SLOW = config.MACD_SLOW
MACD_SIGNAL = config.MACD_SIGNAL
MACD_CROSSOVER_LOOKBACK_DAYS = config.MACD_CROSSOVER_LOOKBACK_DAYS
MACD_CONVERGENCE_FACTOR = config.MACD_CROSSOVER_PROXIMITY_FACTOR
MACD_HIST_SCALE_WINDOW = 20        # rolling window for this stock's "typical" |histogram| magnitude
MACD_EXTENSION_RATIO = 2.0         # MACD line > this many histogram-scales above zero => "overextended"
FIB_LOOKBACK_DAYS = config.FIB_LOOKBACK_DAYS
VOLUME_AVG_WINDOW = config.VOLUME_AVG_WINDOW
FUNDAMENTALS_LAG_DAYS = 45         # safety buffer past quarter-end before results are assumed public

MOMENTUM_WINDOW = 20               # trading sessions - used for momentum_20d, sector- and market-relative momentum
VOLATILITY_WINDOW = 20
SWING_TRACK_LOOKBACK_DAYS = config.SWING_LOOKBACK_DAYS   # 63 - "track record" window for the swing-potential rule component
SWING_REVERSAL_PCT = config.SWING_REVERSAL_PCT            # 4% - zigzag pivot threshold, same as the rule engine

MACD_STATE_OVEREXTENDED = 0
MACD_STATE_CONVERGING = 1
MACD_STATE_CROSSOVER = 2

FEATURE_COLUMNS = [
    "rsi_14",
    "is_rsi_under_50_and_rising",
    "macd_state",
    "days_since_macd_crossover",
    "pct_dist_to_fib_618",
    "pct_dist_to_fib_50",
    "volume_vs_20d_avg",
    "promoter_holding_quarterly_change",
    "profit_growth_yoy",
    # Added for the broadened-feature experiment - fill the two rule-engine
    # components (sector trend, swing track record) that were never given
    # an ML feature, plus standard momentum/volatility/market-regime signals.
    "momentum_20d",
    "volatility_20d",
    "room_to_swing_high_pct",
    "median_up_swing_pct",
    "sector_relative_momentum_20d",
    "market_relative_momentum_20d",
]


def _rsi(close, period=RSI_PERIOD):
    """Wilder's RSI, vectorized over the full series (see ta_engine.calculate_rsi for the single-value equivalent)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.where(avg_loss != 0, 100.0)


def _macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def _days_since_bullish_crossover(macd_line, signal_line):
    """
    Trading sessions since MACD last crossed ABOVE its signal line (0 =
    crossed today), scanning back through this ticker's entire available
    history. NaN until the first crossover ever observed for this ticker.
    """
    diff = macd_line - signal_line
    prev_diff = diff.shift(1)
    crossed_up = (prev_diff < 0) & (diff > 0)

    n = len(diff)
    day_idx = np.arange(n)
    cross_day = np.where(crossed_up.to_numpy(), day_idx, np.nan)
    cross_day = pd.Series(cross_day, index=diff.index).ffill()
    days_since = day_idx - cross_day.to_numpy()
    return pd.Series(days_since, index=diff.index)


def _macd_state(macd_line, signal_line, hist, days_since_cross):
    """
    Categorical MACD state per row:
      2 (crossover)    - a bullish crossover happened within the last
                          MACD_CROSSOVER_LOOKBACK_DAYS sessions and MACD is
                          NOT already far above zero - a fresh reversal.
      0 (overextended)  - either a bullish crossover happened but MACD is
                          already far above zero (the move already
                          happened), or there's no live signal at all (MACD
                          below signal and not converging).
      1 (converging)    - MACD is still below signal, but the histogram is
                          small relative to its own recent scale and
                          shrinking further - an early/watch setup.

    Rows before a ticker has enough history for hist_scale / RSI-equivalent
    warm-up default to 0 (overextended/no-signal), the same fallback bucket
    used when there's genuinely no live setup.
    """
    hist_scale = hist.abs().rolling(MACD_HIST_SCALE_WINDOW, min_periods=5).mean()
    extension_ratio = (macd_line.abs() / hist_scale).replace([np.inf, -np.inf], np.nan)

    fresh_crossover = days_since_cross <= MACD_CROSSOVER_LOOKBACK_DAYS
    extended = extension_ratio > MACD_EXTENSION_RATIO

    is_small = hist.abs() < (hist_scale * MACD_CONVERGENCE_FACTOR)
    is_shrinking = hist.abs() < hist.shift(1).abs()
    converging = (macd_line < signal_line) & is_small & is_shrinking

    state = pd.Series(MACD_STATE_OVEREXTENDED, index=macd_line.index)
    state = state.mask(converging, MACD_STATE_CONVERGING)
    state = state.mask(fresh_crossover & extended, MACD_STATE_OVEREXTENDED)
    state = state.mask(fresh_crossover & ~extended, MACD_STATE_CROSSOVER)
    return state.astype(int)


def _fib_distances(high, low, close, lookback=FIB_LOOKBACK_DAYS):
    """Rolling (trailing-only) Fibonacci 50%/61.8% retracement levels and the current close's distance from each, as a fraction of price."""
    peak = high.rolling(lookback, min_periods=lookback).max()
    trough = low.rolling(lookback, min_periods=lookback).min()
    diff = peak - trough

    level_618 = peak - 0.618 * diff
    level_50 = peak - 0.5 * diff

    pct_dist_618 = (close - level_618).abs() / close
    pct_dist_50 = (close - level_50).abs() / close
    return pct_dist_618, pct_dist_50


def _volume_ratio(volume, window=VOLUME_AVG_WINDOW):
    avg = volume.rolling(window, min_periods=window).mean()
    return volume / avg


def _momentum(close, window=MOMENTUM_WINDOW):
    return close.pct_change(window)


def _volatility(close, window=VOLATILITY_WINDOW):
    return close.pct_change().rolling(window, min_periods=window).std()


def _room_to_swing_high(high, close, lookback=SWING_TRACK_LOOKBACK_DAYS):
    """(recent swing high - close) / close - how much room is left to the local top, mirroring the rule engine's "room-to-target" half of swing potential."""
    swing_high = high.rolling(lookback, min_periods=lookback).max()
    return (swing_high - close) / close


def _median_up_swing_pct(close, lookback_days=SWING_TRACK_LOOKBACK_DAYS, reversal_pct=SWING_REVERSAL_PCT):
    """
    Trailing "track record" feature: median size of this stock's completed
    zigzag up-legs confirmed within the last `lookback_days` sessions - the
    other half of the rule engine's swing-potential score
    (ta_engine.median_up_swing_pct), but recomputed for every day instead
    of only the latest.

    Implemented as one O(n) zigzag scan of the whole series to find every
    completed leg and the session it was confirmed on, then a sliding
    window (deque) over those confirmation dates - not a fresh O(window)
    rescan at every row.
    """
    closes = close.to_numpy(dtype=float)
    n = len(closes)
    legs = []  # (confirm_idx, direction, magnitude)

    if n >= 5:
        pivot_price = closes[0]
        direction = None
        extreme_price = pivot_price
        for i in range(1, n):
            price = closes[i]
            if direction is None:
                if price >= pivot_price * (1 + reversal_pct):
                    direction, extreme_price = "up", price
                elif price <= pivot_price * (1 - reversal_pct):
                    direction, extreme_price = "down", price
                continue
            if direction == "up":
                if price > extreme_price:
                    extreme_price = price
                elif price <= extreme_price * (1 - reversal_pct):
                    legs.append((i, "up", (extreme_price - pivot_price) / pivot_price))
                    pivot_price, direction, extreme_price = extreme_price, "down", price
            else:
                if price < extreme_price:
                    extreme_price = price
                elif price >= extreme_price * (1 + reversal_pct):
                    legs.append((i, "down", (pivot_price - extreme_price) / pivot_price))
                    pivot_price, direction, extreme_price = extreme_price, "up", price

    result = np.zeros(n)
    window = deque()  # (confirm_idx, magnitude) for up-legs currently inside the trailing window
    leg_ptr = 0
    for i in range(n):
        while leg_ptr < len(legs) and legs[leg_ptr][0] <= i:
            idx, leg_dir, mag = legs[leg_ptr]
            if leg_dir == "up":
                window.append((idx, mag))
            leg_ptr += 1
        while window and window[0][0] < i - lookback_days:
            window.popleft()
        result[i] = float(np.median([m for _, m in window])) if window else 0.0

    return pd.Series(result, index=close.index)


def _ticker_price_features(g):
    g = g.sort_values("date").reset_index(drop=True)
    close, high, low, volume = g["close"], g["high"], g["low"], g["volume"]

    rsi = _rsi(close)
    macd_line, signal_line, hist = _macd(close)
    days_since_cross = _days_since_bullish_crossover(macd_line, signal_line)
    macd_state = _macd_state(macd_line, signal_line, hist, days_since_cross)
    pct_618, pct_50 = _fib_distances(high, low, close)
    vol_ratio = _volume_ratio(volume)
    momentum_20d = _momentum(close)
    volatility_20d = _volatility(close)
    room_to_swing_high = _room_to_swing_high(high, close)
    median_up_swing = _median_up_swing_pct(close)

    return pd.DataFrame({
        "ticker": g["ticker"].to_numpy(),
        "date": g["date"].to_numpy(),
        "rsi_14": rsi.to_numpy(),
        "is_rsi_under_50_and_rising": ((rsi < 50) & (rsi > rsi.shift(1))).astype(int).to_numpy(),
        "macd_state": macd_state.to_numpy(),
        "days_since_macd_crossover": days_since_cross.to_numpy(),
        "pct_dist_to_fib_618": pct_618.to_numpy(),
        "pct_dist_to_fib_50": pct_50.to_numpy(),
        "volume_vs_20d_avg": vol_ratio.to_numpy(),
        "momentum_20d": momentum_20d.to_numpy(),
        "volatility_20d": volatility_20d.to_numpy(),
        "room_to_swing_high_pct": room_to_swing_high.to_numpy(),
        "median_up_swing_pct": median_up_swing.to_numpy(),
    })


def _add_sector_relative_momentum(price_features):
    """Cross-sectional step (needs all tickers at once): each stock's momentum_20d minus the mean momentum_20d of its own sector on that same date."""
    df = price_features.copy()
    sector = df["ticker"].map(config.SECTOR_MAP).fillna("Other")
    sector_avg_momentum = df.assign(_sector=sector).groupby(["date", "_sector"])["momentum_20d"].transform("mean")
    df["sector_relative_momentum_20d"] = df["momentum_20d"] - sector_avg_momentum
    return df


def _add_market_relative_momentum(price_features, index_df):
    """Each stock's momentum_20d minus the benchmark index's own momentum_20d over the same window - NaN for every row if no index_df was supplied."""
    df = price_features.copy()
    if index_df is None:
        df["market_relative_momentum_20d"] = np.nan
        return df

    idx = index_df[["date", "close"]].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")
    idx["index_momentum_20d"] = idx["close"].pct_change(MOMENTUM_WINDOW)

    df = df.merge(idx[["date", "index_momentum_20d"]], on="date", how="left")
    df["market_relative_momentum_20d"] = df["momentum_20d"] - df["index_momentum_20d"]
    return df.drop(columns=["index_momentum_20d"])


def build_price_features(price_df, index_df=None):
    """Price/volume/technical features only (no fundamentals) - see build_feature_matrix() for the full set."""
    required = {"ticker", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(price_df.columns)
    if missing:
        raise ValueError(f"price_df missing required columns: {sorted(missing)}")

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])

    parts = [_ticker_price_features(g) for _, g in df.groupby("ticker", sort=False)]
    features = pd.concat(parts, ignore_index=True)
    features = _add_sector_relative_momentum(features)
    features = _add_market_relative_momentum(features, index_df)
    return features


def build_fundamental_features(fundamentals_df, lag_days=FUNDAMENTALS_LAG_DAYS):
    """
    Quarter-over-quarter promoter holding change and YoY profit growth,
    tagged with the earliest date each quarter's numbers are allowed to be
    used (`effective_date` = quarter_end + lag_days) - the point-in-time
    guard against lookahead bias.
    """
    required = {"ticker", "quarter_end", "promoter_holding_pct", "net_profit"}
    missing = required - set(fundamentals_df.columns)
    if missing:
        raise ValueError(f"fundamentals_df missing required columns: {sorted(missing)}")

    fdf = fundamentals_df.copy()
    fdf["quarter_end"] = pd.to_datetime(fdf["quarter_end"])
    fdf = fdf.sort_values(["ticker", "quarter_end"]).reset_index(drop=True)

    fdf["promoter_holding_quarterly_change"] = fdf.groupby("ticker")["promoter_holding_pct"].diff()

    prior_year_profit = fdf.groupby("ticker")["net_profit"].shift(4)
    fdf["profit_growth_yoy"] = (fdf["net_profit"] - prior_year_profit) / prior_year_profit.abs()

    fdf["effective_date"] = fdf["quarter_end"] + pd.Timedelta(days=lag_days)

    return fdf[["ticker", "effective_date", "promoter_holding_quarterly_change", "profit_growth_yoy"]]


def _attach_fundamentals(price_features, fundamentals_features):
    """
    As-of (backward) merge: each price row gets the most recent quarter
    whose `effective_date` is <= that row's date - i.e. forward-filled, but
    never from a quarter that hadn't legally become public yet.
    """
    left = price_features.sort_values("date").reset_index(drop=True)
    right = fundamentals_features.sort_values("effective_date").reset_index(drop=True)

    # merge_asof requires both keys on the exact same datetime64 resolution;
    # pandas can infer a coarser unit (e.g. "s" vs "ns") for an empty/sparse
    # fundamentals frame, so pin both explicitly rather than let them drift.
    left["date"] = left["date"].astype("datetime64[ns]")
    right["effective_date"] = right["effective_date"].astype("datetime64[ns]")

    merged = pd.merge_asof(
        left,
        right,
        left_on="date",
        right_on="effective_date",
        by="ticker",
        direction="backward",
    )
    return merged.drop(columns=["effective_date"])


def build_feature_matrix(price_df, fundamentals_df, index_df=None):
    """
    Full point-in-time-safe feature matrix: one row per (ticker, date) with
    columns [ticker, date] + FEATURE_COLUMNS.

    Rows for a ticker before its first fundamentals effective_date (or
    before enough price history for a given indicator's warm-up window)
    carry NaN in the affected feature(s) - XGBoost handles missing values
    natively, so these are left as NaN rather than imputed here. Same for
    market_relative_momentum_20d when index_df is omitted.
    """
    price_features = build_price_features(price_df, index_df)
    fundamentals_features = build_fundamental_features(fundamentals_df)
    matrix = _attach_fundamentals(price_features, fundamentals_features)

    matrix = matrix.sort_values(["ticker", "date"]).reset_index(drop=True)
    return matrix[["ticker", "date"] + FEATURE_COLUMNS]
