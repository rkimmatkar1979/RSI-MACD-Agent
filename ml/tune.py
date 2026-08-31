"""
Hyperparameter search for the swing-prediction XGBoost model.

Search space is informed by community guidance for noisy, short-horizon
financial data (shallow trees, heavy row/column subsampling, real L1/L2
regularization, high min_child_weight to require many samples before a
split is trusted) - notably tighter than this project's original
hand-picked XGB_PARAMS in ml/train.py, which under-regularized relative to
this guidance (subsample/colsample_bytree too high, no min_child_weight,
no reg_alpha/reg_lambda at all).

Uses the SAME purged, chronological CV splits as ml.train
(purged_time_series_splits) as the search's scoring folds - not sklearn's
default random/stratified CV - so a "better" combination found here isn't
one that simply got to peek at future data. scale_pos_weight is
recomputed per fold per candidate, same as ml.train._fit_one_fold, which
is why this is a hand-rolled loop rather than sklearn's
GridSearchCV/RandomizedSearchCV (neither supports a per-fold-varying
estimator parameter out of the box).

For speed, the search itself scores each candidate on only the LAST
`n_folds` (largest, most recent) purged splits, not all 5 - a full 5-fold
purged+baseline comparison (matching ml.train.train_and_save_model) should
be run separately on the winning candidate to get the real, reportable
number.

Run:
    python -m ml.tune --period 5y                    # full universe, cached price/index history
    python -m ml.tune --period 5y --n-iter 40 --n-folds 4   # slower, more thorough
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler
from xgboost import XGBClassifier

import config
from ml.backfill import (
    BASE_FEATURE_COLUMNS,
    download_index_history,
    download_price_history,
    empty_fundamentals,
)
from ml.features import FEATURE_COLUMNS, build_feature_matrix
from ml.labeling import MAX_HOLDING_DAYS, STOP_LOSS_PCT, TAKE_PROFIT_PCT, relative_triple_barrier_labels, triple_barrier_labels
from ml.train import purged_time_series_splits

# Ranges pulled from swing-trading-specific XGBoost tuning guidance
# (shallower/more regularized than generic defaults - the goal is a model
# that ignores noise even if that costs a bit of raw fit).
SEARCH_SPACE = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.005, 0.01, 0.02, 0.03, 0.05],
    "n_estimators": [300, 500, 800],
    "subsample": [0.5, 0.6, 0.7],
    "colsample_bytree": [0.4, 0.5, 0.6],
    "min_child_weight": [10, 30, 50, 100],
    "reg_alpha": [1, 3, 5, 10],
    "reg_lambda": [1, 3, 5, 10],
}


def _scale_pos_weight(y):
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / pos if pos else 1.0


def _fit_eval(params, X_train, y_train, X_test, y_test):
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="auc", random_state=42, n_jobs=-1,
        scale_pos_weight=_scale_pos_weight(y_train), **params,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, proba)


def search(feature_df, label_col="target", feature_cols=FEATURE_COLUMNS, n_iter=20, n_folds=3,
           random_state=42, purge_days=MAX_HOLDING_DAYS):
    """Returns a list of (mean_auc, params) sorted best-first. purge_days should match the label's own max_days horizon (see ml.train.run_cross_validation)."""
    df = feature_df.dropna(subset=[label_col]).sort_values("date").reset_index(drop=True)
    X, y = df[feature_cols], df[label_col]

    all_splits = list(purged_time_series_splits(df["date"], purge_days=purge_days))
    splits = all_splits[-n_folds:]  # largest/most recent folds - still purged, still chronological

    sampler = list(ParameterSampler(SEARCH_SPACE, n_iter=n_iter, random_state=random_state))
    results = []
    for i, params in enumerate(sampler, start=1):
        fold_scores = []
        for train_idx, test_idx, _ in splits:
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            if y_train.nunique() < 2 or y_test.nunique() < 2:
                continue
            fold_scores.append(_fit_eval(params, X_train, y_train, X_test, y_test))

        mean_auc = float(np.mean(fold_scores)) if fold_scores else float("nan")
        print(f"[{i}/{len(sampler)}] AUC={mean_auc:.4f}  {params}")
        results.append((mean_auc, params))

    results.sort(key=lambda r: r[0], reverse=True)
    return results


def main(period="5y", n_iter=20, n_folds=3, feature_set="broadened", label_kind="relative",
         tp_pct=TAKE_PROFIT_PCT, sl_pct=STOP_LOSS_PCT, max_days=MAX_HOLDING_DAYS):
    tickers = config.SCAN_UNIVERSE
    feature_cols = FEATURE_COLUMNS if feature_set == "broadened" else BASE_FEATURE_COLUMNS

    print(f"[tune] loading {period} price/index history (cache-first)...")
    price_df = download_price_history(tickers, period)
    index_df = download_index_history(period=period)

    print("[tune] building feature matrix...")
    features = build_feature_matrix(price_df, empty_fundamentals(), index_df=index_df)
    features = features[["ticker", "date"] + feature_cols]

    print(f"[tune] computing {label_kind} labels (+{tp_pct:.1%}/-{sl_pct:.1%}, {max_days}d horizon)...")
    labels = (
        relative_triple_barrier_labels(price_df, index_df, tp_pct=tp_pct, sl_pct=sl_pct, max_days=max_days) if label_kind == "relative"
        else triple_barrier_labels(price_df, tp_pct=tp_pct, sl_pct=sl_pct, max_days=max_days)
    )
    merged = features.merge(labels, on=["ticker", "date"], how="left")

    print(f"[tune] searching {n_iter} candidates over the last {n_folds} purged folds (purge_days={max_days})...")
    results = search(merged, feature_cols=feature_cols, n_iter=n_iter, n_folds=n_folds, purge_days=max_days)

    print(f"\nTop 5 parameter combinations (search-phase AUC, last-{n_folds}-fold mean):")
    for auc, params in results[:5]:
        print(f"  AUC={auc:.4f}  {params}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--n-iter", type=int, default=20)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--features", choices=["base", "broadened"], default="broadened")
    parser.add_argument("--label", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--tp-pct", type=float, default=TAKE_PROFIT_PCT)
    parser.add_argument("--sl-pct", type=float, default=STOP_LOSS_PCT)
    parser.add_argument("--max-days", type=int, default=MAX_HOLDING_DAYS)
    args = parser.parse_args()
    main(period=args.period, n_iter=args.n_iter, n_folds=args.n_folds,
         feature_set=args.features, label_kind=args.label,
         tp_pct=args.tp_pct, sl_pct=args.sl_pct, max_days=args.max_days)
