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
from sklearn.isotonic import IsotonicRegression
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


def _fit_one_fold(X_train, y_train, xgb_params=XGB_PARAMS, use_scale_pos_weight=True):
    spw = _scale_pos_weight(y_train) if use_scale_pos_weight else 1.0
    model = XGBClassifier(scale_pos_weight=spw, **xgb_params)
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


def run_cross_validation(feature_df, label_col="target", feature_cols=FEATURE_COLUMNS,
                          purge_days=PURGE_DAYS, xgb_params=XGB_PARAMS, use_scale_pos_weight=True):
    """
    Diagnostic pass: reports out-of-fold AUC across the purged,
    chronological folds (for both the real XGBoost model and a logistic
    regression baseline, see _fit_logistic_baseline) so overfitting is
    visible before the final model is trained on the full dataset.

    purge_days should match (or exceed) the label's own forward-looking
    horizon (see purged_time_series_splits) - callers using a shorter
    label window than the 90-day default MUST pass the matching purge_days
    or training rows too close to a test fold's start will leak label
    information from inside the test window.

    xgb_params defaults to this module's XGB_PARAMS (tuned for the
    original 90-day label via ml.tune) - pass a different dict for a
    differently-tuned label/horizon rather than overwriting the module
    constant, since multiple horizons can be in active use at once.

    use_scale_pos_weight=False disables the neg/pos class-imbalance
    correction entirely (scale_pos_weight=1.0 for every fold) - useful for
    an A/B comparison against the default, since scale_pos_weight improves
    ranking/AUC on imbalanced labels at a known cost to how well the raw
    predict_proba output reflects true probabilities (see calibration_table
    and isotonic_calibration below).

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
    for fold, (train_idx, test_idx, n_purged) in enumerate(purged_time_series_splits(df["date"], purge_days=purge_days), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if len(X_train) == 0 or y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"[fold {fold}] skipped - insufficient class variety after purging "
                  f"(train={len(X_train)}, test={len(X_test)})")
            continue

        model = _fit_one_fold(X_train, y_train, xgb_params=xgb_params, use_scale_pos_weight=use_scale_pos_weight)
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


def run_cross_validation_ensemble(feature_df, label_col="target", feature_cols=FEATURE_COLUMNS,
                                   purge_days=PURGE_DAYS, xgb_params=XGB_PARAMS, ensemble_weight=0.7):
    """
    Same purged-CV loop as run_cross_validation, but blends each fold's
    XGBoost and logistic-baseline predictions (ensemble_weight on XGBoost,
    1-ensemble_weight on logistic) into ONE out-of-fold probability per
    row, rather than treating the logistic fit as a diagnostic-only
    baseline. XGBoost is always fit with use_scale_pos_weight=False here -
    the ensemble was only ever validated with it disabled (see this
    session's A/B test), not with it on.

    Returns (blend_fold_aucs, oof_df, xgb_only_fold_aucs) - oof_df's
    y_pred_proba column is the BLENDED probability, ready to feed straight
    into calibration_table()/isotonic_calibration() so the calibration
    curve reflects what actually gets shipped, not the XGBoost-only output.
    xgb_only_fold_aucs is kept for the printed fold-by-fold comparison.
    """
    df = feature_df.sort_values("date").reset_index(drop=True)
    X, y = df[feature_cols], df[label_col]

    blend_fold_aucs, xgb_only_fold_aucs = [], []
    oof_parts = []
    for fold, (train_idx, test_idx, n_purged) in enumerate(purged_time_series_splits(df["date"], purge_days=purge_days), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        if len(X_train) == 0 or y_train.nunique() < 2 or y_test.nunique() < 2:
            print(f"[fold {fold}] skipped - insufficient class variety after purging "
                  f"(train={len(X_train)}, test={len(X_test)})")
            continue

        xgb_model = _fit_one_fold(X_train, y_train, xgb_params=xgb_params, use_scale_pos_weight=False)
        xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_auc = roc_auc_score(y_test, xgb_proba)
        xgb_only_fold_aucs.append(xgb_auc)

        log_model = _fit_logistic_baseline(X_train, y_train)
        log_proba = log_model.predict_proba(X_test)[:, 1]

        blend_proba = ensemble_weight * xgb_proba + (1 - ensemble_weight) * log_proba
        blend_auc = roc_auc_score(y_test, blend_proba)
        blend_fold_aucs.append(blend_auc)

        print(f"[fold {fold}] train={len(X_train)} (purged {n_purged} rows) test={len(X_test)} "
              f"xgb_auc={xgb_auc:.4f} blend_auc={blend_auc:.4f} ({'+' if blend_auc >= xgb_auc else ''}{blend_auc - xgb_auc:.4f})")

        fold_out = df.loc[test_idx, ["ticker", "date"]].copy() if "ticker" in df.columns else pd.DataFrame(index=test_idx)
        fold_out["y_true"] = y_test.to_numpy()
        fold_out["y_pred_proba"] = blend_proba
        oof_parts.append(fold_out)

    if blend_fold_aucs:
        print(f"Mean OOF AUC: xgb-only={np.mean(xgb_only_fold_aucs):.4f} (+/- {np.std(xgb_only_fold_aucs):.4f}) "
              f"| blend={np.mean(blend_fold_aucs):.4f} (+/- {np.std(blend_fold_aucs):.4f})")
    else:
        print("No folds produced a valid AUC - check class balance / data volume.")

    oof_df = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame(
        columns=["ticker", "date", "y_true", "y_pred_proba"])
    return blend_fold_aucs, oof_df, xgb_only_fold_aucs


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


def calibration_table(oof_df, tp_pct=None, sl_pct=None, n_bins=20):
    """
    Bins oof_df's out-of-fold predicted probabilities into n_bins quantile
    buckets (pd.qcut, duplicates dropped if the score distribution is too
    narrow/spiky to support n_bins distinct edges) and reports, per bucket:
    the score range, row count, empirical win rate (mean y_true - "of
    predictions scored in this range, how many actually hit the target"),
    lift vs the overall base rate, and - if tp_pct/sl_pct are given -
    expected value per trade (win_rate*tp_pct - (1-win_rate)*sl_pct).

    This is NOT the same thing as precision_at_k: that buckets by RANK
    (today's top 5/10/20%), this buckets by the raw SCORE value itself, so
    a live prediction can be looked up by "which historical bucket does
    THIS score fall into" regardless of how many other stocks were scored
    alongside it that day. Returns [] if oof_df is empty.
    """
    if oof_df.empty:
        return []

    base_rate = float(oof_df["y_true"].mean())
    binned = pd.qcut(oof_df["y_pred_proba"], q=n_bins, duplicates="drop")

    bins = []
    for interval, group in oof_df.groupby(binned, observed=True):
        win_rate = float(group["y_true"].mean())
        entry = {
            "low": float(interval.left),
            "high": float(interval.right),
            "n": int(len(group)),
            "win_rate": win_rate,
            "lift": win_rate / base_rate if base_rate else None,
        }
        if tp_pct is not None and sl_pct is not None:
            entry["expected_value"] = win_rate * tp_pct - (1 - win_rate) * sl_pct
        bins.append(entry)

    bins.sort(key=lambda b: b["low"])
    return bins


def isotonic_calibration(oof_df):
    """
    Fits isotonic regression - monotonic, non-parametric - on the
    out-of-fold (raw predicted probability, actual outcome) pairs: the
    standard way to turn a model's raw predict_proba into an actual
    calibrated probability without assuming a parametric shape (unlike
    Platt/sigmoid scaling, isotonic only enforces "higher raw score =>
    higher-or-equal calibrated probability" and lets the data set
    everything else - appropriate here since nothing about XGBoost with
    scale_pos_weight suggests its raw output follows a sigmoid-shaped
    miscalibration).

    Architecture: Model -> OOF predictions -> this calibration layer ->
    calibrated probability. Returns the fitted curve as a list of [x, y]
    breakpoints (JSON-serializable; interpolate with a plain increasing
    piecewise-linear lookup at inference time - see ml.infer._calibrate)
    rather than persisting the sklearn object itself, so diagnostics.json
    stays the single source of truth with no second joblib file to keep in
    sync. Returns [] if oof_df is empty.
    """
    if oof_df.empty:
        return []
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(oof_df["y_pred_proba"], oof_df["y_true"])
    return [[float(x), float(y)] for x, y in zip(iso.X_thresholds_, iso.y_thresholds_)]


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


def train_and_save_model(feature_df, model_path, label_col="target", feature_cols=FEATURE_COLUMNS,
                          label_kind="absolute", purge_days=PURGE_DAYS, label_config=None, xgb_params=XGB_PARAMS,
                          use_scale_pos_weight=True):
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
    label definition produced a given result. purge_days MUST match (or
    exceed) the label's own max_days horizon - see run_cross_validation.
    label_config, if given (e.g. {"tp_pct":..., "sl_pct":..., "max_days":...}),
    is stamped into diagnostics as-is so runs with a non-default label
    horizon stay distinguishable from each other after the fact. xgb_params
    defaults to this module's XGB_PARAMS (see run_cross_validation) and is
    also stamped into diagnostics for the same reason.
    """
    df = feature_df.dropna(subset=[label_col]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No labeled rows to train on after dropping unresolved/NaN targets.")

    print(f"Training on {len(df)} labeled rows ({df[label_col].mean():.1%} positive) "
          f"spanning {df['date'].min().date()} to {df['date'].max().date()}")

    fold_aucs, oof_df, baseline_fold_aucs = run_cross_validation(
        df, label_col=label_col, feature_cols=feature_cols, purge_days=purge_days, xgb_params=xgb_params,
        use_scale_pos_weight=use_scale_pos_weight)

    precision_stats = precision_at_k(oof_df)
    if precision_stats:
        print("\nPrecision at top-K (out-of-fold):")
        for k, stats in precision_stats.items():
            print(f"  {k}: precision={stats['precision']:.1%} vs base_rate={stats['base_rate']:.1%} "
                  f"(lift={stats['lift']:.2f}x, n={stats['n']})")

    _lc = label_config or {}
    calibration = calibration_table(oof_df, tp_pct=_lc.get("tp_pct"), sl_pct=_lc.get("sl_pct"))
    breakeven_win_rate = (
        _lc["sl_pct"] / (_lc["tp_pct"] + _lc["sl_pct"])
        if "tp_pct" in _lc and "sl_pct" in _lc else None
    )
    if calibration:
        print(f"\nCalibration ({len(calibration)} score bins, out-of-fold)"
              + (f" - breakeven win rate {breakeven_win_rate:.1%}:" if breakeven_win_rate is not None else ":"))
        for b in calibration:
            print(f"  [{b['low']:.3f}, {b['high']:.3f}] n={b['n']:>6d} win_rate={b['win_rate']:.1%} lift={b['lift']:.2f}x"
                  + (f" ev={b['expected_value']:+.2%}" if "expected_value" in b else ""))

    oof_base_rate = float(oof_df["y_true"].mean()) if not oof_df.empty else None
    isotonic_curve = isotonic_calibration(oof_df)
    if isotonic_curve:
        print(f"\nIsotonic calibration fit on {len(oof_df)} OOF predictions "
              f"({len(isotonic_curve)} breakpoints).")

    X_full, y_full = df[feature_cols], df[label_col]
    final_model = XGBClassifier(
        scale_pos_weight=(_scale_pos_weight(y_full) if use_scale_pos_weight else 1.0), **xgb_params)
    final_model.fit(X_full, y_full)

    importance = print_feature_importance(final_model, feature_cols)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({"model": final_model, "feature_columns": list(feature_cols)}, model_path)

    diagnostics = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "label_kind": label_kind,
        "label_config": label_config,
        "purge_days": purge_days,
        "xgb_params": xgb_params,
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
        "calibration": calibration,
        "breakeven_win_rate": breakeven_win_rate,
        "isotonic_calibration": isotonic_curve,
        "oof_base_rate": oof_base_rate,
        "use_scale_pos_weight": use_scale_pos_weight,
    }
    with open(diagnostics_path(model_path), "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved diagnostics to {diagnostics_path(model_path)}")
    return final_model


def train_and_save_ensemble_model(feature_df, model_path, label_col="target", feature_cols=FEATURE_COLUMNS,
                                   label_kind="absolute", purge_days=PURGE_DAYS, label_config=None,
                                   xgb_params=XGB_PARAMS, ensemble_weight=0.7):
    """
    Same shape/outputs as train_and_save_model(), but for an XGBoost +
    logistic-regression BLEND rather than XGBoost alone - only use this
    after a fold-by-fold robustness check on the blend weight (a pooled/
    average AUC improvement isn't enough on its own, see this session's
    ensemble validation). Always runs with use_scale_pos_weight=False (the
    ensemble was only validated that way).

    Saves {"model", "logistic_model", "ensemble_weight", "feature_columns"}
    - ml.infer.score_tickers()/generate_daily_shortlist() detect the
      "logistic_model" key and blend automatically; single-model bundles
      from train_and_save_model() are unaffected and keep working as-is.

    Calibration (calibration_table/isotonic_calibration) is fit on the
    BLENDED out-of-fold predictions, not XGBoost's alone, so the
    calibration curve matches what actually gets shipped.
    """
    df = feature_df.dropna(subset=[label_col]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("No labeled rows to train on after dropping unresolved/NaN targets.")

    print(f"Training ensemble on {len(df)} labeled rows ({df[label_col].mean():.1%} positive) "
          f"spanning {df['date'].min().date()} to {df['date'].max().date()} (ensemble_weight={ensemble_weight})")

    fold_aucs, oof_df, xgb_only_fold_aucs = run_cross_validation_ensemble(
        df, label_col=label_col, feature_cols=feature_cols, purge_days=purge_days,
        xgb_params=xgb_params, ensemble_weight=ensemble_weight)

    precision_stats = precision_at_k(oof_df)
    if precision_stats:
        print("\nPrecision at top-K (out-of-fold, blended):")
        for k, stats in precision_stats.items():
            print(f"  {k}: precision={stats['precision']:.1%} vs base_rate={stats['base_rate']:.1%} "
                  f"(lift={stats['lift']:.2f}x, n={stats['n']})")

    _lc = label_config or {}
    calibration = calibration_table(oof_df, tp_pct=_lc.get("tp_pct"), sl_pct=_lc.get("sl_pct"))
    breakeven_win_rate = (
        _lc["sl_pct"] / (_lc["tp_pct"] + _lc["sl_pct"])
        if "tp_pct" in _lc and "sl_pct" in _lc else None
    )
    if calibration:
        print(f"\nCalibration ({len(calibration)} score bins, out-of-fold, blended)"
              + (f" - breakeven win rate {breakeven_win_rate:.1%}:" if breakeven_win_rate is not None else ":"))
        for b in calibration:
            print(f"  [{b['low']:.3f}, {b['high']:.3f}] n={b['n']:>6d} win_rate={b['win_rate']:.1%} lift={b['lift']:.2f}x"
                  + (f" ev={b['expected_value']:+.2%}" if "expected_value" in b else ""))

    oof_base_rate = float(oof_df["y_true"].mean()) if not oof_df.empty else None
    isotonic_curve = isotonic_calibration(oof_df)
    if isotonic_curve:
        print(f"\nIsotonic calibration fit on {len(oof_df)} blended OOF predictions "
              f"({len(isotonic_curve)} breakpoints).")

    X_full, y_full = df[feature_cols], df[label_col]
    final_xgb = XGBClassifier(scale_pos_weight=1.0, **xgb_params)
    final_xgb.fit(X_full, y_full)
    final_logistic = _fit_logistic_baseline(X_full, y_full)

    importance = print_feature_importance(final_xgb, feature_cols)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({
        "model": final_xgb,
        "logistic_model": final_logistic,
        "ensemble_weight": ensemble_weight,
        "feature_columns": list(feature_cols),
    }, model_path)

    diagnostics = {
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "label_kind": label_kind,
        "label_config": label_config,
        "purge_days": purge_days,
        "xgb_params": xgb_params,
        "ensemble_weight": ensemble_weight,
        "n_rows": int(len(df)),
        "positive_rate": float(df[label_col].mean()),
        "date_range": [str(df["date"].min().date()), str(df["date"].max().date())],
        "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else None,
        "fold_aucs": [float(a) for a in fold_aucs],
        "mean_auc": float(np.mean(fold_aucs)) if fold_aucs else None,
        "std_auc": float(np.std(fold_aucs)) if fold_aucs else None,
        "xgb_only_fold_aucs": [float(a) for a in xgb_only_fold_aucs],
        "xgb_only_mean_auc": float(np.mean(xgb_only_fold_aucs)) if xgb_only_fold_aucs else None,
        "feature_importance": {k: float(v) for k, v in importance.items()},
        "precision_at_k": precision_stats,
        "calibration": calibration,
        "breakeven_win_rate": breakeven_win_rate,
        "isotonic_calibration": isotonic_curve,
        "oof_base_rate": oof_base_rate,
        "use_scale_pos_weight": False,
    }
    with open(diagnostics_path(model_path), "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nSaved ensemble model to {model_path}")
    print(f"Saved diagnostics to {diagnostics_path(model_path)}")
    return final_xgb, final_logistic
