"""Daily inference: score today's live feature snapshot and shortlist high-confidence setups."""

import joblib

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
    """
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]

    missing = [c for c in feature_cols if c not in live_features_df.columns]
    if missing:
        raise ValueError(f"live_features_df is missing model features: {missing}")

    proba = model.predict_proba(live_features_df[feature_cols])[:, 1]

    result = live_features_df.copy()
    result["ml_confidence"] = proba
    result = result.sort_values("ml_confidence", ascending=False).reset_index(drop=True)

    if min_proba is not None:
        result = result[result["ml_confidence"] >= min_proba]

    top_n = max(1, int(round(len(result) * top_pct)))
    return result.head(top_n).reset_index(drop=True)
