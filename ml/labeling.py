"""
Triple-barrier label engineering (Lopez de Prado-style) for the ML
swing-trading model.

For each (ticker, date=T) row: target = 1 if price hits the +8% take-profit
barrier before the -5% stop-loss barrier within the next 90 CALENDAR days;
target = 0 if the stop-loss is hit first, or if 90 days pass with neither
barrier touched.

Daily OHLC bars can't tell us which barrier was actually touched first
intraday when both are crossed on the same session - that ambiguity is
resolved conservatively in favour of the stop-loss (target = 0).

Rows whose 90-day forward window is not yet fully covered by the supplied
price history (the entry date is too close to the end of the data) are left
as NaN - a censored/unresolved outcome, not a confirmed "0" - and must be
dropped before training rather than silently treated as a miss. See
ml.train.train_and_save_model(), which does this drop.
"""

import numpy as np
import pandas as pd

TAKE_PROFIT_PCT = 0.08
STOP_LOSS_PCT = 0.05
MAX_HOLDING_DAYS = 90


def _ticker_triple_barrier(g, tp_pct, sl_pct, max_days):
    g = g.sort_values("date").reset_index(drop=True)
    dates = g["date"].to_numpy("datetime64[ns]")
    close = g["close"].to_numpy(dtype=float)
    high = g["high"].to_numpy(dtype=float)
    low = g["low"].to_numpy(dtype=float)
    n = len(g)

    labels = np.full(n, np.nan)
    last_date = dates[-1] if n else None
    horizon = np.timedelta64(max_days, "D")

    for i in range(n):
        entry_price = close[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        deadline = dates[i] + horizon

        resolved = False
        j = i + 1
        while j < n and dates[j] <= deadline:
            if low[j] <= sl_price:
                # Stop-loss (whether alone or alongside a same-day TP touch) wins ties.
                labels[i] = 0.0
                resolved = True
                break
            if high[j] >= tp_price:
                labels[i] = 1.0
                resolved = True
                break
            j += 1

        if not resolved and deadline <= last_date:
            labels[i] = 0.0  # window genuinely expired with neither barrier touched
        # else: leave NaN - window not yet fully observed in this price history

    g["target"] = labels
    return g[["ticker", "date", "target"]]


def triple_barrier_labels(price_df, tp_pct=TAKE_PROFIT_PCT, sl_pct=STOP_LOSS_PCT, max_days=MAX_HOLDING_DAYS):
    """
    Returns a DataFrame with columns [ticker, date, target] (target in
    {0.0, 1.0, NaN}), aligned 1:1 with price_df's (ticker, date) rows -
    ready to be merged onto the output of ml.features.build_feature_matrix().
    """
    required = {"ticker", "date", "close", "high", "low"}
    missing = required - set(price_df.columns)
    if missing:
        raise ValueError(f"price_df missing required columns: {sorted(missing)}")

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    parts = [
        _ticker_triple_barrier(g, tp_pct, sl_pct, max_days)
        for _, g in df.groupby("ticker", sort=False)
    ]
    return pd.concat(parts, ignore_index=True)


def _ticker_relative_triple_barrier(g, tp_pct, sl_pct, max_days):
    g = g.sort_values("date").reset_index(drop=True)
    dates = g["date"].to_numpy("datetime64[ns]")
    close = g["close"].to_numpy(dtype=float)
    index_close = g["index_close"].to_numpy(dtype=float)
    n = len(g)

    labels = np.full(n, np.nan)
    last_date = dates[-1] if n else None
    horizon = np.timedelta64(max_days, "D")

    for i in range(n):
        entry_price = close[i]
        entry_index = index_close[i]
        deadline = dates[i] + horizon

        resolved = False
        j = i + 1
        while j < n and dates[j] <= deadline:
            excess = (close[j] / entry_price) / (index_close[j] / entry_index) - 1
            if excess <= -sl_pct:
                labels[i] = 0.0
                resolved = True
                break
            if excess >= tp_pct:
                labels[i] = 1.0
                resolved = True
                break
            j += 1

        if not resolved and deadline <= last_date:
            labels[i] = 0.0

    g["target"] = labels
    return g[["ticker", "date", "target"]]


def _ticker_relative_triple_barrier_detail(g, tp_pct, sl_pct, max_days):
    g = g.sort_values("date").reset_index(drop=True)
    dates = g["date"].to_numpy("datetime64[ns]")
    close = g["close"].to_numpy(dtype=float)
    index_close = g["index_close"].to_numpy(dtype=float)
    n = len(g)

    target = np.full(n, np.nan)
    resolved_date = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    days_held = np.full(n, np.nan)
    realized_return = np.full(n, np.nan)
    mfe = np.full(n, np.nan)
    mae = np.full(n, np.nan)
    last_date = dates[-1] if n else None
    horizon = np.timedelta64(max_days, "D")

    for i in range(n):
        entry_price = close[i]
        entry_index = index_close[i]
        deadline = dates[i] + horizon

        resolved = False
        running_max, running_min = -np.inf, np.inf
        j = i + 1
        while j < n and dates[j] <= deadline:
            excess = (close[j] / entry_price) / (index_close[j] / entry_index) - 1
            running_max = max(running_max, excess)
            running_min = min(running_min, excess)
            if excess <= -sl_pct or excess >= tp_pct:
                target[i] = 0.0 if excess <= -sl_pct else 1.0
                resolved_date[i] = dates[j]
                days_held[i] = (dates[j] - dates[i]) / np.timedelta64(1, "D")
                realized_return[i] = excess
                resolved = True
                break
            j += 1

        if not resolved and deadline <= last_date:
            k = j - 1
            target[i] = 0.0
            resolved_date[i] = dates[k]
            days_held[i] = (dates[k] - dates[i]) / np.timedelta64(1, "D")
            realized_return[i] = (close[k] / entry_price) / (index_close[k] / entry_index) - 1
        # else: still censored - target and every detail field stay NaN/NaT

        if not np.isnan(target[i]):
            if running_max > -np.inf:
                mfe[i] = running_max
            if running_min < np.inf:
                mae[i] = running_min

    g["target"] = target
    g["resolved_date"] = resolved_date
    g["days_held"] = days_held
    g["realized_return"] = realized_return
    g["mfe_pct"] = mfe
    g["mae_pct"] = mae
    return g[["ticker", "date", "target", "resolved_date", "days_held", "realized_return", "mfe_pct", "mae_pct"]]


def resolve_relative_triple_barrier_detail(price_df, index_df, tp_pct=TAKE_PROFIT_PCT, sl_pct=STOP_LOSS_PCT, max_days=MAX_HOLDING_DAYS):
    """
    Paper-trading resolution helper: same barrier logic as
    relative_triple_barrier_labels(), but for every row also reports
    resolved_date, days_held, realized_return (the actual excess return at
    the moment/day the barrier was touched or the window timed out - not
    just which barrier "won"), and the path's maximum favorable/adverse
    excursion (mfe_pct/mae_pct - the best and worst excess return reached
    at any point before resolution). Rows whose window isn't fully covered
    by price_df yet are left entirely NaN/NaT (still censored), same
    convention as relative_triple_barrier_labels().

    Not used by training (which only needs the plain 0/1/NaN target) - this
    is for ml.paper_trade to record richer outcome data once a logged pick
    resolves.
    """
    required = {"ticker", "date", "close"}
    missing = required - set(price_df.columns)
    if missing:
        raise ValueError(f"price_df missing required columns: {sorted(missing)}")
    if not {"date", "close"}.issubset(index_df.columns):
        raise ValueError("index_df must have columns: date, close")

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    idx = index_df[["date", "close"]].rename(columns={"close": "index_close"}).copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")

    df = df.merge(idx, on="date", how="inner")

    empty = pd.DataFrame(columns=["ticker", "date", "target", "resolved_date", "days_held", "realized_return", "mfe_pct", "mae_pct"])
    if df.empty:
        return empty

    parts = [
        _ticker_relative_triple_barrier_detail(g, tp_pct, sl_pct, max_days)
        for _, g in df.groupby("ticker", sort=False)
    ]
    return pd.concat(parts, ignore_index=True) if parts else empty


def relative_triple_barrier_labels(price_df, index_df, tp_pct=TAKE_PROFIT_PCT, sl_pct=STOP_LOSS_PCT, max_days=MAX_HOLDING_DAYS):
    """
    Same triple-barrier construction as triple_barrier_labels(), but the
    path being tracked is the stock's cumulative return MINUS the
    benchmark index's cumulative return over the same window (excess
    return vs. e.g. Nifty 50), not the stock's raw price. The label answers
    "did this stock beat the index by +8pp before lagging it by -5pp",
    which - unlike the absolute version - doesn't conflate a stock's own
    setup with whether the whole market happened to be rallying or selling
    off over that particular 90 days.

    Uses Close-to-Close excess return only (no High/Low): a stock's
    intraday range and an index's intraday range don't combine into a
    meaningful "excess high/low", so there's no same-day-both-hit
    ambiguity to resolve here the way triple_barrier_labels() has to.

    index_df must have columns [date, close] (the benchmark's own price
    history, e.g. Nifty 50 / ^NSEI). Only dates the benchmark also traded
    on are kept.
    """
    required = {"ticker", "date", "close"}
    missing = required - set(price_df.columns)
    if missing:
        raise ValueError(f"price_df missing required columns: {sorted(missing)}")
    if not {"date", "close"}.issubset(index_df.columns):
        raise ValueError("index_df must have columns: date, close")

    df = price_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    idx = index_df[["date", "close"]].rename(columns={"close": "index_close"}).copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.sort_values("date")

    df = df.merge(idx, on="date", how="inner")

    parts = [
        _ticker_relative_triple_barrier(g, tp_pct, sl_pct, max_days)
        for _, g in df.groupby("ticker", sort=False)
    ]
    return pd.concat(parts, ignore_index=True)
