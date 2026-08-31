"""
Model training: purged, time-series-aware cross-validation, followed by a
final XGBoost classifier refit on the full labeled history.

Financial panel data breaks two assumptions ordinary k-fold CV relies on:
  1. Rows aren't independent across time - shuffled folds let the model
     train on next month and test on this month, which silently inflates
     validation scores.
  2. A label at date T depends on prices up to T + 90 days, so any training
     row within 90 days of a test fold's start date was labeled using
     price action that overlaps the test period - that's leakage even
     under a plain chronological split.

TimeSeriesSplit (chronological, unshuffled folds) addresses (1); manually
purging each fold's trailing 90 days addresses (2).
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from ml.features import FEATURE_COLUMNS

N_SPLITS = 5
PURGE_DAYS = 90

# Hyperparameters found by ml/tune.py's purged-CV search over the
# swing-trading-specific ranges (shallow trees, heavy subsampling, real
# L1/L2 regularization, high min_child_weight) - see that module's
# docstring. Winning search-phase AUC 0.522 (last-3-fold mean) vs ~0.51
# for this project's original hand-picked values, which were less
# regularized than this guidance recommends (subsample/colsample_bytree
# too high, no min_child_weight, no reg_alpha/reg_lambda at all).
XGB_PARAMS = dict(
    max_depth=2,
    learning_rate=0.005,
    n_estimators=300,
    subsample=0.5,
    colsample_bytree=0.6,
    min_child_weight=50,
    reg_alpha=1,
    reg_lambda=1,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42,
    n_jobs=-1,
)


def _scale_pos_weight(y):
    """neg/pos ratio for this split - keeps XGBoost from just predicting the majority class when TP hits are a minority outcome."""
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return neg / pos if pos else 1.0


def purged_time_series_splits(dates, n_splits=N_SPLITS, purge_days=PURGE_DAYS):
    """
    Yields (train_idx, test_idx, n_purged) - train_idx/test_idx are
    positional-index arrays over `dates` (a Series/array sorted ascending
    and aligned 1:1 with the feature matrix's row order); n_purged is how
    many rows were dropped from that fold's training set for diagnostics.

    Wraps sklearn's TimeSeriesSplit, then drops training rows dated within
    `purge_days` of the test fold's start date - those rows' triple-barrier
    labels were computed from a forward price window that bleeds into the
    test period.
    """
    dates = pd.to_datetime(pd.Series(dates).reset_index(drop=True))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for train_idx, test_idx in tscv.split(dates):
        test_start = dates.iloc[test_idx[0]]
        purge_cutoff = test_start - pd.Timedelta(days=purge_days)
        keep = (dates.iloc[train_idx] <= purge_cutoff).to_numpy()
        purged_train_idx = train_idx[keep]
        yield purged_train_idx, test_idx, len(train_idx) - len(purged_train_idx)


def _fit_one_fold(X_train, y_train):
    model = XGBClassifier(scale_pos_weight=_scale_pos_weight(y_train), **XGB_PARAMS)
    model.fit(X_train, y_train)
    return model


def _fit_logistic_baseline(X_train, y_train):
    """
    Simple linear baseline: median-impute (unlike XGBoost, sklearn's
    LogisticRegression can't handle NaN natively), scale, then fit with
    class_weight="balanced" (logistic's equivalent of scale_pos_weight).

    Purpose isn't to compete with XGBoost - it's a diagnostic: if this
    scores close to XGBoost's AUC, the ceiling is the FEATURES (there's
    only so much linearly-decodable signal in them), not the model; if
    XGBoost clears it by a wide margin, there's real nonlinear/interaction
    structure the trees are finding that a linear model can't.
    """
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000),
    )
    model.fit(X_train, y_train)
    return model


def run_cross_validation(feature_df, label_col="target", feature_cols=FEATURE_COLUMNS):
    """
    Diagnostic pass: reports out-of-fold AUC across the purged,
    chronological folds (for both the real XGBoost model and a logistic
    regression baseline, see _fit_logistic_baseline) so overfitting is
    visible before the final model is trained on the full dataset.

    Returns (fold_aucs, oof_df, baseline_fold_aucs) - fold_aucs is the
    XGBoost per-fold AUC list; oof_df has one row per test-fold observation
    with columns [ticker, date, y_true, y_pred_proba] from XGBoost,
    concatenated across every fold's (disjoint, chronological) test set -
    i.e. every prediction in oof_df came from a model that never saw that
    row during training. This is what precision_at_k() below needs:
    scoring the FINAL model on its own training data would be optimistic,
    since it saw those labels. baseline_fold_aucs is the same per-fold AUC
    list for the logistic baseline.
    """
    df = feature_df.sort_values("date").reset_index(drop=True)
    X, y = df[feature_cols], df[label_col]

    fold_aucs = []
    baseline_fold_aucs = []
    oof_parts = []
    for fold, (train_idx, test_idx, n_purged) in enumerate(purged_time_series_splits(df["date"]), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if len(X_train) == 0 or y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"[fold {fold}] skipped - insufficient class variety after purging "
                  f"(train={len(X_train)}, test={len(X_test)})")
            continue

        model = _fit_one_fold(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, proba)
        fold_aucs.append(auc)

        baseline = _fit_logistic_baseline(X_train, y_train)
        baseline_proba = baseline.predict_proba(X_test)[:, 1]
        baseline_auc = roc_auc_score(y_test, baseline_proba)
        baseline_fold_aucs.append(baseline_auc)

        print(f"[fold {fold}] train={len(X_train)} (purged {n_purged} rows) "
              f"test={len(X_test)} AUC={auc:.4f} (logistic baseline={baseline_auc:.4f})")

        fold_out = df.loc[test_idx, ["ticker", "date"]].copy() if "ticker" in df.columns else pd.DataFrame(index=test_idx)
        fold_out["y_true"] = y_test.to_numpy()
        fold_out["y_pred_proba"] = proba
        oof_parts.append(fold_out)

    if fold_aucs:
        print(f"Mean OOF AUC across {len(fold_aucs)} folds: {np.mean(fold_aucs):.4f} "
              f"(+/- {np.std(fold_aucs):.4f}) | logistic baseline: {np.mean(baseline_fold_aucs):.4f} "
              f"(+/- {np.std(baseline_fold_aucs):.4f})")
    else:
        print("No folds produced a valid AUC - check class balance / data volume.")

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(
        columns=["ticker", "date", "y_true", "y_pred_proba"])
    return fold_aucs, oof_df, baseline_fold_aucs


def precision_at_k(oof_df, k_fracs=(0.05, 0.1, 0.2)):
    """
    For each fraction in k_fracs: takes the top-k% of oof_df by predicted
    probability and reports the actual positive rate within that subset
    (precision), alongside the overall base rate and the "lift" (precision
    / base_rate). A model with zero real skill should score lift ~= 1.0 at
    every k even if its overall AUC is barely above 0.5 - a model whose
    edge is concentrated in its most-confident calls (which is the only
    way >0.70-confidence filtering in ml.infer.generate_daily_shortlist is
    useful) should show lift > 1 and rising as k shrinks.
    """
    n = len(oof_df)
    if n == 0:
        return {}

    base_rate = float(oof_df["y_true"].mean())
    ranked = oof_df.sort_values("y_pred_proba", ascending=False).reset_index(drop=True)

    results = {}
    for k in k_fracs:
        top_n = max(1, int(round(n * k)))
        subset = ranked.iloc[:top_n]
        precision = float(subset["y_true"].mean())
        results[f"top_{int(k * 100)}pct"] = {
            "n": top_n,
            "precision": precision,
            "base_rate": base_rate,
            "lift": precision / base_rate if base_rate else None,
        }
    return results


def print_feature_importance(model, feature_cols=FEATURE_COLUMNS):
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nFeature importance (final model, most relied-on first):")
    print(importance.to_string(float_format=lambda v: f"{v:.4f}"))
    return importance


def diagnostics_path(model_path):
    base, _ = os.path.splitext(model_path)
    return base + "_diagnostics.json"


def load_diagnostics(model_path):
    """Returns the diagnostics dict saved alongside model_path by train_and_save_model(), or None if it hasn't been trained yet."""
    path = diagnostics_path(model_path)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def train_and_save_model(feature_df, model_path, label_col="target", feature_cols=FEATURE_COLUMNS, label_kind="absolute"):
    """
    feature_df must already carry a `target` column (see ml.labeling,
    merged onto ml.features.build_feature_matrix's output) - NaN targets
    (censored/unresolved outcomes) are expected and are dropped here.

    1. Runs purged TimeSeriesSplit CV as a generalization sanity check.
    2. Refits a final XGBClassifier on ALL labeled rows (the model that
       actually ships gets the most data).
    3. Prints the final model's feature importances, most relied-on first.
    4. Saves {"model", "feature_columns"} to model_path via joblib, and a
       sibling *_diagnostics.json (fold AUCs, feature importance, row
       counts) that ml/../app.py's ML Predictions tab reads to display
       real results without re-running training.

    label_kind is purely descriptive (e.g. "absolute" vs "relative-to-Nifty")
    and gets stamped into the diagnostics file so the UI can show which
    label definition produced a given result.
    """
    df = feature_df.dropna(subset=[label_col]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No labeled rows to train on after dropping unresolved/NaN targets.")

    print(f"Training on {len(df)} labeled rows ({df[label_col].mean():.1%} positive) "
          f"spanning {df['date'].min().date()} to {df['date'].max().date()}")

    fold_aucs, oof_df, baseline_fold_aucs = run_cross_validation(df, label_col=label_col, feature_cols=feature_cols)

    precision_stats = precision_at_k(oof_df)
    if precision_stats:
        print("\nPrecision at top-K (out-of-fold):")
        for k, stats in precision_stats.items():
            print(f"  {k}: precision={stats['precision']:.1%} vs base_rate={stats['base_rate']:.1%} "
                  f"(lift={stats['lift']:.2f}x, n={stats['n']})")

    X_full, y_full = df[feature_cols], df[label_col]
    final_model = XGBClassifier(scale_pos_weight=_scale_pos_weight(y_full), **XGB_PARAMS)
    final_model.fit(X_full, y_full)

    importance = print_feature_importance(final_model, feature_cols)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": list(feature_cols)}, model_path)

    diagnostics = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "label_kind": label_kind,
        "n_rows": int(len(df)),
        "positive_rate": float(df[label_col].mean()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
        "fold_aucs": [float(a) for a in fold_aucs],
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else None,
        "baseline_fold_aucs": [float(a) for a in baseline_fold_aucs],
        "baseline_mean_auc": float(np.mean(baseline_fold_aucs)) if baseline_fold_aucs else None,
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "precision_at_k": precision_stats,
    }
    with open(diagnostics_path(model_path), "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved diagnostics to {diagnostics_path(model_path)}")
    return final_model
