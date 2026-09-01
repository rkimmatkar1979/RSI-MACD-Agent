"""Daily inference: score today's live feature snapshot and shortlist high-confidence setups."""

import joblib
import numpy as np
import pandas as pd

from ml.backfill import download_index_history, download_price_history, download_vix_history, empty_fundamentals
from ml.features import build_price_features_extended, build_fundamental_features, _attach_fundamentals
from ml.train import load_diagnostics

DEFAULT_TOP_PCT = 0.10


def generate_daily_shortlist(live_features_df, model_path, top_pct=DEFAULT_TOP_PCT, min_proba=None):
    """
    Scores a live (ticker, date) feature snapshot - one row per candidate
    stock as of today, built the same way as the training matrix via
    ml.features.build_feature_matrix() - and returns the top `top_pct`
    fraction by predicted probability of hitting the +8% take-profit
    before the -5% stop-loss within the next 90 days, sorted most-confident
    first.

    Selection is by RANK (top_pct), not an absolute probability cutoff.
    ml.train.precision_at_k() on the real backtest showed only a modest,
    fairly flat lift across the top 5/10/20% buckets (~1.05x) - the edge
    isn't sharply concentrated at very high probabilities, and with a
    ~42% base rate the model's raw probabilities may rarely even clear a
    threshold like 0.70. Picking today's best `top_pct` of the scanned
    universe matches how the model was actually validated; an earlier
    version of this function filtered on `proba > 0.70` instead, which
    could silently return zero stocks on many days. `min_proba` is an
    optional secondary floor (e.g. 0.5) if you also want to require the
    model lean positive, not just be relatively better than its peers.

    If the bundle has a "logistic_model" key (saved by
    ml.train.train_and_save_ensemble_model), blends XGBoost's and the
    logistic model's predictions using the bundle's own "ensemble_weight" -
    single-model bundles (no "logistic_model" key) score exactly as before.
    """
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]

    missing = [c for c in feature_cols if c not in live_features_df.columns]
    if missing:
        raise ValueError(f"live_features_df is missing model features: {missing}")

    proba = model.predict_proba(live_features_df[feature_cols])[:, 1]
    if "logistic_model" in bundle:
        w = bundle["ensemble_weight"]
        logistic_proba = bundle["logistic_model"].predict_proba(live_features_df[feature_cols])[:, 1]
        proba = w * proba + (1 - w) * logistic_proba

    result = live_features_df.copy()
    result["ml_confidence"] = proba
    result = result.sort_values("ml_confidence", ascending=False).reset_index(drop=True)

    if min_proba is not None:
        result = result[result["ml_confidence"] >= min_proba]

    top_n = max(1, int(round(len(result) * top_pct)))
    return result.head(top_n).reset_index(drop=True)


def _calibrate(score, isotonic_curve):
    """
    Maps a raw predict_proba value through the fitted isotonic curve (see
    ml.train.isotonic_calibration) via piecewise-linear interpolation
    against its stored breakpoints - the actual "calibration layer" step
    in Model -> OOF predictions -> calibration layer -> calibrated
    probability. Clamps to the curve's edge values outside the observed
    range (numpy.interp's default behavior). Returns None if the curve is
    empty/missing (older model runs trained before this existed).
    """
    if not isotonic_curve:
        return None
    xs = [p[0] for p in isotonic_curve]
    ys = [p[1] for p in isotonic_curve]
    return float(np.interp(score, xs, ys))


def score_tickers(tickers, model_path, period="1y", refresh_cache=True):
    """
    Scores every ticker in `tickers` (no top-K filtering) and returns one
    row each: ticker, date, ml_confidence, entry_price, tp_price, sl_price,
    label_kind, tp_pct, sl_pct, max_days - sorted most-confident first.
    Returns an empty DataFrame if the model hasn't been trained yet (no
    diagnostics found) or no tickers could be scored.

    Also attaches calibrated_probability - the raw ml_confidence passed
    through the model's fitted isotonic calibration curve (see
    ml.train.isotonic_calibration) - plus lift_vs_baseline,
    expected_value_pct, breakeven_win_rate, above_breakeven, and
    signal_tier all derived from that calibrated probability, not the raw
    score. This matters because a raw predict_proba from a model trained
    with scale_pos_weight (needed for ranking on an imbalanced label) is
    NOT a calibrated probability on its own - "40% raw" can empirically
    mean anywhere from ~20% to ~40% actual win rate depending on where it
    falls on the curve; calibrated_probability corrects for that. All
    calibration columns are None/NaN for models trained before this
    existed (no "isotonic_calibration" in diagnostics).

    Unlike ml.paper_trade.log_todays_picks, this does not persist anything
    and doesn't skip based on prior runs - meant for on-demand live scoring
    of a small ticker list (e.g. today's rule-based shortlist), not the
    daily paper-trading log.
    """
    diagnostics = load_diagnostics(model_path)
    if diagnostics is None:
        print(f"[infer] no diagnostics found for {model_path} - run ml.backfill first.")
        return pd.DataFrame()
    label_config = diagnostics.get("label_config")
    if label_config is None:
        print(f"[infer] {model_path}'s diagnostics predate label_config (an older model run) - "
              "re-run ml.backfill/the training driver for this model to score it live.")
        return pd.DataFrame()
    label_kind = diagnostics["label_kind"]

    price_df = download_price_history(tickers, period=period, refresh_cache=refresh_cache)
    index_df = download_index_history(period=period, refresh_cache=refresh_cache)
    vix_df = download_vix_history(period=period, refresh_cache=refresh_cache)
    price_df = price_df.copy()
    price_df["date"] = pd.to_datetime(price_df["date"])
    index_df = index_df.copy()
    index_df["date"] = pd.to_datetime(index_df["date"])

    # Extended builder (superset of every ablation-study feature group, see
    # ml.features) rather than the plain build_feature_matrix(), so this
    # works regardless of which group combination a given model was trained
    # on - generate_daily_shortlist() below only selects the columns that
    # model's own bundle actually lists.
    features = build_price_features_extended(price_df, index_df=index_df, vix_df=vix_df)
    fundamentals_features = build_fundamental_features(empty_fundamentals())
    features = _attach_fundamentals(features, fundamentals_features)
    live = features.sort_values("date").groupby("ticker").tail(1).reset_index(drop=True)
    if live.empty:
        return pd.DataFrame()

    scored = generate_daily_shortlist(live, model_path, top_pct=1.0, min_proba=None)
    scored = scored.merge(price_df[["ticker", "date", "close"]], on=["ticker", "date"], how="left")
    scored = scored.rename(columns={"close": "entry_price"})

    scored["tp_price"] = scored["entry_price"] * (1 + label_config["tp_pct"])
    scored["sl_price"] = scored["entry_price"] * (1 - label_config["sl_pct"])
    scored["label_kind"] = label_kind
    scored["tp_pct"] = label_config["tp_pct"]
    scored["sl_pct"] = label_config["sl_pct"]
    scored["max_days"] = label_config["max_days"]

    isotonic_curve = diagnostics.get("isotonic_calibration") or []
    oof_base_rate = diagnostics.get("oof_base_rate")
    breakeven_win_rate = diagnostics.get("breakeven_win_rate")
    tp_pct, sl_pct = label_config["tp_pct"], label_config["sl_pct"]

    def _tier(lift):
        if lift is None:
            return None
        if lift >= 1.2:
            return "strong"
        if lift >= 1.05:
            return "moderate"
        return "weak"

    calibrated, lift_vs_baseline, ev_pct, tiers = [], [], [], []
    for conf in scored["ml_confidence"]:
        cal = _calibrate(conf, isotonic_curve)
        if cal is None or not oof_base_rate:
            calibrated.append(None)
            lift_vs_baseline.append(None)
            ev_pct.append(None)
            tiers.append(None)
        else:
            lift = cal / oof_base_rate
            calibrated.append(cal)
            lift_vs_baseline.append(lift)
            ev_pct.append(cal * tp_pct - (1 - cal) * sl_pct)
            tiers.append(_tier(lift))

    scored["calibrated_probability"] = calibrated
    scored["lift_vs_baseline"] = lift_vs_baseline
    scored["expected_value_pct"] = ev_pct
    scored["signal_tier"] = tiers
    scored["breakeven_win_rate"] = breakeven_win_rate
    scored["above_breakeven"] = (
        scored["calibrated_probability"] >= breakeven_win_rate if breakeven_win_rate is not None else None
    )

    cols = ["ticker", "date", "ml_confidence", "entry_price", "tp_price", "sl_price",
            "label_kind", "tp_pct", "sl_pct", "max_days", "calibrated_probability",
            "lift_vs_baseline", "expected_value_pct", "signal_tier",
            "breakeven_win_rate", "above_breakeven"]
    return scored[cols].reset_index(drop=True)
