"""
Daily ML paper-trading log: records what the model would have picked today,
and resolves earlier picks once their outcome is known.

Two responsibilities, run every time main() is called:
  1. resolve_pending_picks() - re-checks every still-open prediction against
     fresh price history using the SAME triple-barrier function used to
     build training labels (ml.labeling) - its target column already means
     "unresolved" (NaN) vs "resolved" (0/1), so no new barrier-checking
     logic is needed here, just a lookup.
  2. log_todays_picks() - scores today's live feature snapshot with
     ml.infer.generate_daily_shortlist() and persists the top picks, unless
     today's picks for this model are already saved (idempotent re-runs).

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
import os
import threading
from datetime import datetime

import pandas as pd

import config
import db_handler
from ml.backfill import download_index_history, download_price_history, empty_fundamentals
from ml.features import build_feature_matrix
from ml.infer import DEFAULT_TOP_PCT, generate_daily_shortlist
from ml.labeling import relative_triple_barrier_labels, triple_barrier_labels
from ml.train import load_diagnostics

DEFAULT_MODEL_PATH = os.path.join("ml", "models", "tier1_full_v7_mom21d_tuned.joblib")
DEFAULT_PERIOD = "1y"  # safe margin over the largest feature warm-up window (90 trading days)


def _to_datetime(df, col):
    df = df.copy()
    df[col] = pd.to_datetime(df[col])
    return df


def log_todays_picks(price_df, index_df, model_path, top_pct):
    """Scores today's live snapshot and saves the top `top_pct` picks, unless already saved for today."""
    diagnostics = load_diagnostics(model_path)
    if diagnostics is None:
        print(f"[paper_trade] no diagnostics found for {model_path} - skipping (run ml.backfill first).")
        return
    label_config = diagnostics["label_config"]
    label_kind = diagnostics["label_kind"]

    price_df = _to_datetime(price_df, "date")
    index_df = _to_datetime(index_df, "date")

    latest_date = price_df["date"].max()
    if db_handler.has_predictions_for(str(latest_date.date()), model_path):
        print(f"[paper_trade] already logged picks for {latest_date.date()} ({model_path}) - skipping.")
        return

    features = build_feature_matrix(price_df, empty_fundamentals(), index_df=index_df)
    live = features.sort_values("date").groupby("ticker").tail(1).reset_index(drop=True)

    picks = generate_daily_shortlist(live, model_path, top_pct=top_pct)
    if picks.empty:
        print("[paper_trade] no picks generated today.")
        return

    picks = picks.merge(price_df[["ticker", "date", "close"]], on=["ticker", "date"], how="left")
    picks = picks.merge(
        index_df[["date", "close"]].rename(columns={"close": "index_close"}), on="date", how="left"
    )

    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        {
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
        }
        for _, r in picks.iterrows()
        if pd.notna(r["close"])
    ]

    n = db_handler.save_ml_predictions(rows)
    print(f"[paper_trade] logged {n} pick(s) for {latest_date.date()} ({model_path}).")


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
    group_cols = ["tp_pct", "sl_pct", "max_days", "label_kind"]
    for (tp_pct, sl_pct, max_days, label_kind), group in pending.groupby(group_cols):
        tickers = group["ticker"].unique().tolist()
        sub_price = price_df[price_df["ticker"].isin(tickers)]

        if label_kind == "relative":
            labels = relative_triple_barrier_labels(sub_price, index_df, tp_pct=tp_pct, sl_pct=sl_pct, max_days=int(max_days))
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

    try:
        resolve_pending_picks(price_df, index_df)
    except Exception as e:
        print(f"[paper_trade] resolve_pending_picks failed: {e}")

    try:
        log_todays_picks(price_df, index_df, model_path, top_pct)
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
