"""
Daily ML paper-trading log: records what the model would have picked today,
and resolves earlier picks once their outcome is known.

Two responsibilities, run every time main() is called:
  1. resolve_pending_picks() - re-checks every still-open prediction against
     fresh price history using the SAME triple-barrier function used to
     build training labels (ml.labeling), plus the richer
     resolve_relative_triple_barrier_detail() for realized_return/days_held/
     mfe_pct/mae_pct on relative-label predictions.
  2. log_todays_picks() - scores today's live feature snapshot directly off
     the model bundle (mirrors ml.infer.score_tickers's calibration path,
     but also keeps the raw XGBoost/logistic components and full-universe
     rank so the log records enough to analyze later - see Phase 3 of the
     user's Next-Phase PRD) and persists the top picks, unless today's picks
     for this model are already saved (idempotent re-runs).

Not wired to run on any OS-level schedule - triggered from
scheduler.py::run_pipeline() via run_in_background() whenever "Run Full
Scan Now" fires (or `python scheduler.py` / `python -m ml.paper_trade` is
run directly), so this rides the app's one existing manual trigger instead
of needing new scheduling infrastructure.

Run standalone:
    python -m ml.paper_trade                 # full config.SCAN_UNIVERSE
    python -m ml.paper_trade --limit 15       # quick manual test on a subset
"""

import argparse
import json
import os
import threading
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

import config
import db_handler
from ml.backfill import download_index_history, download_price_history, download_vix_history, empty_fundamentals
from ml.features import build_price_features_extended, build_fundamental_features, _attach_fundamentals
from ml.infer import DEFAULT_TOP_PCT, _calibrate
from ml.labeling import relative_triple_barrier_labels, resolve_relative_triple_barrier_detail, triple_barrier_labels
from ml.train import load_diagnostics

# Frozen model (see the user's Next-Phase PRD, Phase 2): 27 features +
# 70/30 XGBoost/logistic ensemble, fold-by-fold-validated ROBUST (4/5 folds
# improved, mean +0.0021 AUC). Paper trading is now this model's primary
# forward-validation mechanism - no further tuning until a meaningful
# sample of live outcomes has accumulated.
DEFAULT_MODEL_PATH = os.path.join("ml", "models", "tier1_full_v8_ensemble.joblib")
DEFAULT_PERIOD = "1y"  # safe margin over the largest feature warm-up window (90 trading days)

REGIME_COLS = [
    "nifty_momentum_21d", "nifty_dist_from_sma50", "nifty_dist_from_sma200",
    "nifty_volatility_21d", "india_vix", "india_vix_change_5d", "market_breadth_50dma",
]


def _to_datetime(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


def _avg_traded_value_20d(price_df):
    """Ticker -> trailing-20-session average (volume * close) as of the latest date in price_df - a simple liquidity/tradability proxy, computed locally so this stays self-contained within ml/ rather than touching ta_engine.py."""
    df = price_df.sort_values(["ticker", "date"])
    recent = df.groupby("ticker", sort=False).tail(20)
    avg_volume = recent.groupby("ticker")["volume"].mean()
    last_close = df.groupby("ticker")["close"].last()
    return (avg_volume * last_close).to_dict()


def log_todays_picks(price_df, index_df, vix_df, model_path, top_pct):
    """Scores today's live snapshot against the full universe, saves the top `top_pct` picks with full Phase-3 diagnostics, unless already saved for today."""
    diagnostics = load_diagnostics(model_path)
    if diagnostics is None:
        print(f"[paper_trade] no diagnostics found for {model_path} - skipping (run ml.backfill first).")
        return
    label_config = diagnostics["label_config"]
    label_kind = diagnostics["label_kind"]
    isotonic_curve = diagnostics.get("isotonic_calibration") or []
    oof_base_rate = diagnostics.get("oof_base_rate")

    price_df = _to_datetime(price_df, "date")
    index_df = _to_datetime(index_df, "date")

    latest_date = price_df["date"].max()
    if db_handler.has_predictions_for(str(latest_date.date()), model_path):
        print(f"[paper_trade] already logged picks for {latest_date.date()} ({model_path}) - skipping.")
        return

    features = build_price_features_extended(price_df, index_df=index_df, vix_df=vix_df)
    fundamentals_features = build_fundamental_features(empty_fundamentals())
    features = _attach_fundamentals(features, fundamentals_features)
    live = features.sort_values("date").groupby("ticker").tail(1).reset_index(drop=True)
    if live.empty:
        print("[paper_trade] no live features available today.")
        return

    bundle = joblib.load(model_path)
    model, feature_cols = bundle["model"], bundle["feature_columns"]
    missing = [c for c in feature_cols if c not in live.columns]
    if missing:
        raise ValueError(f"live features are missing model columns: {missing}")

    xgb_proba = model.predict_proba(live[feature_cols])[:, 1]
    if "logistic_model" in bundle:
        w = bundle["ensemble_weight"]
        logistic_proba = bundle["logistic_model"].predict_proba(live[feature_cols])[:, 1]
        blend_proba = w * xgb_proba + (1 - w) * logistic_proba
    else:
        logistic_proba = np.full(len(live), np.nan)
        blend_proba = xgb_proba

    live = live.copy()
    live["xgb_score"] = xgb_proba
    live["logistic_score"] = logistic_proba
    live["ml_confidence"] = blend_proba
    live["calibrated_probability"] = [_calibrate(s, isotonic_curve) for s in blend_proba]
    live["lift_vs_baseline"] = (
        live["calibrated_probability"] / oof_base_rate if oof_base_rate else np.nan
    )

    live = live.sort_values("ml_confidence", ascending=False).reset_index(drop=True)
    live["rank_all"] = live.index + 1
    universe_size = len(live)

    top_n = max(1, int(round(universe_size * top_pct)))
    picks = live.head(top_n).copy().reset_index(drop=True)
    picks["rank_shortlist"] = picks.index + 1
    picks["universe_size"] = universe_size

    picks = picks.merge(price_df[["ticker", "date", "close"]], on=["ticker", "date"], how="left")
    picks = picks.merge(
        index_df[["date", "close"]].rename(columns={"close": "index_close"}), on="date", how="left"
    )
    picks = picks[picks["close"].notna()].reset_index(drop=True)
    if picks.empty:
        print("[paper_trade] no picks generated today.")
        return

    liquidity = _avg_traded_value_20d(price_df)
    picks["avg_traded_value_20d"] = picks["ticker"].map(liquidity)
    picks["sector"] = picks["ticker"].map(config.SECTOR_MAP).fillna("Other")

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    for _, r in picks.iterrows():
        regime = {c: (None if pd.isna(r.get(c)) else float(r[c])) for c in REGIME_COLS}
        rows.append({
            "prediction_date": str(r["date"].date()),
            "ticker": r["ticker"],
            "model_path": model_path,
            "label_kind": label_kind,
            "ml_confidence": r["ml_confidence"],
            "entry_price": r["close"],
            "entry_index_price": r["index_close"] if pd.notna(r["index_close"]) else None,
            "tp_pct": label_config["tp_pct"],
            "sl_pct": label_config["sl_pct"],
            "max_days": label_config["max_days"],
            "created_at": created_at,
            "sector": r["sector"],
            "xgb_score": r["xgb_score"],
            "logistic_score": None if pd.isna(r["logistic_score"]) else r["logistic_score"],
            "calibrated_probability": r["calibrated_probability"],
            "lift_vs_baseline": None if pd.isna(r["lift_vs_baseline"]) else r["lift_vs_baseline"],
            "rank_all": r["rank_all"],
            "universe_size": r["universe_size"],
            "rank_shortlist": r["rank_shortlist"],
            "atr_pct": r.get("atr_pct") if pd.notna(r.get("atr_pct")) else None,
            "volatility_20d": r.get("volatility_20d") if pd.notna(r.get("volatility_20d")) else None,
            "momentum_5d": r.get("momentum_5d") if pd.notna(r.get("momentum_5d")) else None,
            "momentum_20d": r.get("momentum_20d") if pd.notna(r.get("momentum_20d")) else None,
            "momentum_60d": r.get("momentum_60d") if pd.notna(r.get("momentum_60d")) else None,
            "avg_traded_value_20d": r["avg_traded_value_20d"] if pd.notna(r["avg_traded_value_20d"]) else None,
            "market_regime_json": json.dumps(regime),
        })

    n = db_handler.save_ml_predictions(rows)
    print(f"[paper_trade] logged {n} pick(s) for {latest_date.date()} ({model_path}), ranked out of {universe_size}.")


def resolve_pending_picks(price_df, index_df):
    """Re-checks every unresolved prediction against fresh price history and marks any that now have a known outcome."""
    pending = db_handler.get_pending_ml_predictions()
    if pending.empty:
        print("[paper_trade] no pending predictions to resolve.")
        return

    price_df = _to_datetime(price_df, "date")
    index_df = _to_datetime(index_df, "date")
    pending["prediction_date"] = pd.to_datetime(pending["prediction_date"])

    today = str(datetime.now().date())
    updates = []
    # pending's own realized_return/days_held/mfe_pct/mae_pct columns are
    # always NULL here (resolved=0 rows) - drop them before merging so they
    # don't collide with the same-named columns detail()/labels() compute,
    # which would otherwise silently become realized_return_x/_y instead.
    pending = pending.drop(columns=["realized_return", "days_held", "mfe_pct", "mae_pct", "resolved_date"], errors="ignore")
    group_cols = ["tp_pct", "sl_pct", "max_days", "label_kind"]
    for (tp_pct, sl_pct, max_days, label_kind), group in pending.groupby(group_cols):
        tickers = group["ticker"].unique().tolist()
        sub_price = price_df[price_df["ticker"].isin(tickers)]
        if sub_price.empty:
            # This run's price_df doesn't cover these pending tickers (e.g. a
            # --limit smoke test, or a shortlist-scoped run that no longer
            # includes an older pick) - nothing to resolve them against yet.
            continue

        if label_kind == "relative":
            detail = resolve_relative_triple_barrier_detail(
                sub_price, index_df, tp_pct=tp_pct, sl_pct=sl_pct, max_days=int(max_days))
            detail = detail.rename(columns={"date": "prediction_date"})
            resolved = group.merge(detail, on=["ticker", "prediction_date"], how="left")
            resolved = resolved[resolved["target"].notna()]
            for _, r in resolved.iterrows():
                updates.append({
                    "id": int(r["id"]), "outcome": int(r["target"]), "resolved_date": today,
                    "realized_return": float(r["realized_return"]) if pd.notna(r["realized_return"]) else None,
                    "days_held": float(r["days_held"]) if pd.notna(r["days_held"]) else None,
                    "mfe_pct": float(r["mfe_pct"]) if pd.notna(r["mfe_pct"]) else None,
                    "mae_pct": float(r["mae_pct"]) if pd.notna(r["mae_pct"]) else None,
                })
        else:
            labels = triple_barrier_labels(sub_price, tp_pct=tp_pct, sl_pct=sl_pct, max_days=int(max_days))
            labels = labels.rename(columns={"date": "prediction_date"})
            resolved = group.merge(labels, on=["ticker", "prediction_date"], how="left")
            resolved = resolved[resolved["target"].notna()]
            for _, r in resolved.iterrows():
                updates.append({"id": int(r["id"]), "outcome": int(r["target"]), "resolved_date": today})

    if updates:
        db_handler.mark_ml_predictions_resolved(updates)
    print(f"[paper_trade] resolved {len(updates)}/{len(pending)} pending prediction(s).")


def main(model_path=DEFAULT_MODEL_PATH, top_pct=DEFAULT_TOP_PCT, period=DEFAULT_PERIOD, tickers=None):
    db_handler.init_db()
    tickers = tickers or config.SCAN_UNIVERSE

    price_df = download_price_history(tickers, period=period, refresh_cache=True)
    index_df = download_index_history(period=period, refresh_cache=True)
    vix_df = download_vix_history(period=period, refresh_cache=True)

    try:
        resolve_pending_picks(price_df, index_df)
    except Exception as e:
        print(f"[paper_trade] resolve_pending_picks failed: {e}")

    try:
        log_todays_picks(price_df, index_df, vix_df, model_path, top_pct)
    except Exception as e:
        print(f"[paper_trade] log_todays_picks failed: {e}")


def _safe_run(**kwargs):
    try:
        main(**kwargs)
    except Exception as e:
        print(f"[paper_trade] background run failed: {e}")


def run_in_background(**kwargs):
    """Fire-and-forget: runs main() on a daemon thread so the caller (scheduler.run_pipeline) never blocks or fails because of this."""
    threading.Thread(target=_safe_run, kwargs=kwargs, daemon=True).start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--top-pct", type=float, default=DEFAULT_TOP_PCT)
    parser.add_argument("--period", default=DEFAULT_PERIOD)
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N tickers from config.SCAN_UNIVERSE (for a quick manual test).")
    args = parser.parse_args()

    universe = config.SCAN_UNIVERSE[: args.limit] if args.limit else None
    main(model_path=args.model_path, top_pct=args.top_pct, period=args.period, tickers=universe)
