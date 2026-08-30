"""Daily inference: score today's live feature snapshot and shortlist high-confidence setups."""

import joblib

CONFIDENCE_THRESHOLD = 0.70


def generate_daily_shortlist(live_features_df, model_path, threshold=CONFIDENCE_THRESHOLD):
    """
    Scores a live (ticker, date) feature snapshot - one row per candidate
    stock as of today, built the same way as the training matrix via
    ml.features.build_feature_matrix() - and returns only the stocks the
    model gives > `threshold` probability of hitting the +8% take-profit
    before the -5% stop-loss within the next 90 days, sorted most-confident
    first.
    """
    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]

    missing = [c for c in feature_cols if c not in live_features_df.columns]
    if missing:
        raise ValueError(f"live_features_df is missing model features: {missing}")

    proba = model.predict_proba(live_features_df[feature_cols])[:, 1]

    result = live_features_df.copy()
    result["ml_confidence"] = proba

    shortlist = result[result["ml_confidence"] > threshold]
    return shortlist.sort_values("ml_confidence", ascending=False).reset_index(drop=True)
