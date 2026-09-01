"""
Phase 4 - Forward Validation Dashboard (see the user's Next-Phase PRD).

Turns the ml_predictions paper-trading log into the breakdowns the PRD asks
for, so the frozen model's LIVE forward results - not just its historical
OOF backtest - can be checked over time: does a predicted 40% probability
actually win ~40% of the time, does the top-ranked group outperform, where
does the edge actually show up, and is the backtested edge (AUC ~0.585,
see ml.train) surviving contact with live data.

No Streamlit dependency here (same convention as the rest of ml/ - this
gets imported by app.py, not the other way around). Every breakdown is
gated on its own minimum sample size and reports its own n, because with a
21-day holding period this log accumulates slowly - a handful of resolved
picks is not evidence either way, and the whole point of this module is to
not quietly pretend otherwise.
"""

import json

import numpy as np
import pandas as pd

MIN_BUCKET_N = 5
MIN_REPORT_N = 20  # below this, the whole report is flagged preliminary


def _win_rate_ev(df):
    n = len(df)
    if n == 0:
        return {"n": 0, "win_rate": None, "ev_pct": None}
    win_rate = float(df["outcome"].mean())
    tp = float(df["tp_pct"].mean())
    sl = float(df["sl_pct"].mean())
    ev = win_rate * tp - (1 - win_rate) * sl
    return {"n": n, "win_rate": win_rate, "ev_pct": ev}


def _bucketed_table(df, col, q, labels, min_n=MIN_BUCKET_N):
    """Quantile-buckets df by `col` and reports win_rate/ev per bucket. Buckets under min_n are kept but flagged (not silently dropped) - the caller decides whether to display them dimmed/greyed rather than hide sample-size information."""
    valid = df[df[col].notna()]
    if len(valid) < max(min_n, q):
        return None
    try:
        bucket = pd.qcut(valid[col], q=q, labels=labels, duplicates="drop")
    except ValueError:
        return None
    rows = []
    for label, group in valid.groupby(bucket, observed=True):
        stats = _win_rate_ev(group)
        stats["bucket"] = str(label)
        stats["enough_data"] = stats["n"] >= min_n
        rows.append(stats)
    return rows


def overall_summary(df):
    resolved = df[df["resolved"] == 1]
    pending_n = int((df["resolved"] == 0).sum())
    stats = _win_rate_ev(resolved)
    breakeven = None
    if not resolved.empty:
        tp, sl = float(resolved["tp_pct"].mean()), float(resolved["sl_pct"].mean())
        breakeven = sl / (tp + sl) if (tp + sl) else None
    return {
        "total_logged": int(len(df)),
        "resolved_n": stats["n"],
        "pending_n": pending_n,
        "win_rate": stats["win_rate"],
        "ev_pct": stats["ev_pct"],
        "breakeven_win_rate": breakeven,
        "above_breakeven": (
            stats["win_rate"] >= breakeven if stats["win_rate"] is not None and breakeven is not None else None
        ),
    }


def calibration_breakdown(resolved, n_bins=5, min_n=MIN_BUCKET_N):
    """Bins by calibrated_probability and compares predicted vs actual win rate per bin - the direct answer to 'does a predicted 40% actually win ~40% of the time'."""
    valid = resolved[resolved["calibrated_probability"].notna()]
    if len(valid) < max(min_n, n_bins):
        return None
    try:
        bins = pd.qcut(valid["calibrated_probability"], q=n_bins, duplicates="drop")
    except ValueError:
        return None
    rows = []
    for interval, group in valid.groupby(bins, observed=True):
        n = len(group)
        avg_predicted = float(group["calibrated_probability"].mean())
        actual = float(group["outcome"].mean())
        tp, sl = float(group["tp_pct"].mean()), float(group["sl_pct"].mean())
        rows.append({
            "bucket": f"[{interval.left:.2f}, {interval.right:.2f}]",
            "n": n,
            "avg_predicted": avg_predicted,
            "actual_win_rate": actual,
            "gap": actual - avg_predicted,
            "ev_pct": actual * tp - (1 - actual) * sl,
            "enough_data": n >= min_n,
        })
    rows.sort(key=lambda r: r["bucket"])
    return rows


def expected_calibration_error(resolved, n_bins=5):
    """Weighted-average |predicted - actual| across calibration_breakdown's bins - a single number for 'how far off is the live calibration', comparable to the OOF ECE reported at training time. None if there isn't enough resolved data yet."""
    table = calibration_breakdown(resolved, n_bins=n_bins, min_n=1)
    if not table:
        return None
    total_n = sum(r["n"] for r in table)
    if total_n == 0:
        return None
    return sum(r["n"] * abs(r["gap"]) for r in table) / total_n


def precision_by_rank_bucket(resolved, min_n=MIN_BUCKET_N):
    """Buckets by each pick's rank WITHIN that day's shortlist (top/middle/bottom third) - answers 'does the top-ranked group outperform lower-ranked groups' using rank_shortlist, which is comparable across days of different universe sizes (unlike rank_all)."""
    return _bucketed_table(resolved, "rank_shortlist", q=3, labels=["top third", "middle third", "bottom third"], min_n=min_n)


def by_sector(resolved, min_n=MIN_BUCKET_N):
    valid = resolved[resolved["sector"].notna()]
    if valid.empty:
        return None
    rows = []
    for sector, group in valid.groupby("sector"):
        stats = _win_rate_ev(group)
        stats["bucket"] = sector
        stats["enough_data"] = stats["n"] >= min_n
        rows.append(stats)
    rows.sort(key=lambda r: -r["n"])
    return rows


def by_volatility(resolved, min_n=MIN_BUCKET_N):
    return _bucketed_table(resolved, "atr_pct", q=3, labels=["low vol", "mid vol", "high vol"], min_n=min_n)


def by_momentum(resolved, min_n=MIN_BUCKET_N):
    return _bucketed_table(resolved, "momentum_20d", q=3, labels=["weak momentum", "mid momentum", "strong momentum"], min_n=min_n)


def by_liquidity(resolved, min_n=MIN_BUCKET_N):
    return _bucketed_table(resolved, "avg_traded_value_20d", q=3, labels=["low liquidity", "mid liquidity", "high liquidity"], min_n=min_n)


def by_market_regime(resolved, min_n=MIN_BUCKET_N):
    """Parses market_regime_json (logged for analysis only, never fed to the model - see ml.paper_trade) and buckets by that day's India VIX level, relative to what's actually been observed in this log so far (not a fixed VIX threshold, which would be a separate judgment call)."""
    valid = resolved[resolved["market_regime_json"].notna()].copy()
    if valid.empty:
        return None

    def _vix(s):
        try:
            return json.loads(s).get("india_vix")
        except (TypeError, ValueError):
            return None

    valid["_india_vix"] = valid["market_regime_json"].apply(_vix)
    valid = valid[valid["_india_vix"].notna()]
    if len(valid) < max(min_n, 3):
        return None
    return _bucketed_table(valid, "_india_vix", q=3, labels=["low VIX", "mid VIX", "high VIX"], min_n=min_n)


def build_report(df, min_report_n=MIN_REPORT_N):
    """
    df: output of db_handler.get_all_ml_predictions() (ideally already
    filtered to one model_path - see that function's docstring). Returns a
    dict with `summary` plus every named breakdown (None where there isn't
    enough resolved data yet to compute it meaningfully).
    """
    summary = overall_summary(df)
    resolved = df[df["resolved"] == 1].copy()

    report = {
        "summary": summary,
        "preliminary": summary["resolved_n"] < min_report_n,
        "calibration": None,
        "calibration_error": None,
        "precision_by_rank": None,
        "by_sector": None,
        "by_volatility": None,
        "by_momentum": None,
        "by_liquidity": None,
        "by_market_regime": None,
    }
    if resolved.empty:
        return report

    report["calibration"] = calibration_breakdown(resolved)
    report["calibration_error"] = expected_calibration_error(resolved)
    report["precision_by_rank"] = precision_by_rank_bucket(resolved)
    report["by_sector"] = by_sector(resolved)
    report["by_volatility"] = by_volatility(resolved)
    report["by_momentum"] = by_momentum(resolved)
    report["by_liquidity"] = by_liquidity(resolved)
    report["by_market_regime"] = by_market_regime(resolved)
    return report
