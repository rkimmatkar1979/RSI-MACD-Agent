"""
Tier-1 real-data check: downloads full daily OHLCV history for a ticker
list via yfinance, builds a real (fundamentals-blind) feature matrix,
triple-barrier-labels it, and trains + reports a real out-of-fold AUC and
feature importance.

Fundamentals are intentionally skipped here (promoter_holding_quarterly_change
/ profit_growth_yoy come back NaN for every row, which XGBoost handles
natively - see ml/features.py) because yfinance/screener.in only expose a
handful of recent quarters, not enough point-in-time history to align
across a multi-year backtest. That's Tier 2, and needs its own data source.

Two experiment axes, both defaulted to the "broadened" version:
  --features {base,broadened}  base = the original 9-feature spec;
      broadened = + momentum/volatility/swing-track-record/sector- and
      market-relative momentum (see ml/features.py FEATURE_COLUMNS).
  --label {absolute,relative}  absolute = stock hits +8%/-5% in raw price;
      relative = stock beats/lags the Nifty 50 index by +8pp/-5pp over the
      same 90 days (see ml.labeling.relative_triple_barrier_labels).

Run:
    python -m ml.backfill                       # full config.SCAN_UNIVERSE, 5y history, broadened features + relative label
    python -m ml.backfill --limit 30 --period 2y # quick smoke run on a subset
    python -m ml.backfill --features base --label absolute  # reproduce the original Tier-1 experiment
    python -m ml.backfill --tp-pct 0.05 --sl-pct 0.03 --max-days 21  # shorter, momentum-favorable horizon
"""

import argparse
import os

import pandas as pd
import yfinance as yf

import config
from ml.features import FEATURE_COLUMNS, build_feature_matrix
from ml.labeling import MAX_HOLDING_DAYS, STOP_LOSS_PCT, TAKE_PROFIT_PCT, relative_triple_barrier_labels, triple_barrier_labels
from ml.train import train_and_save_model

DEFAULT_PERIOD = "5y"
DEFAULT_MODEL_PATH = os.path.join("ml", "models", "tier1_model.joblib")
INDEX_TICKER = "^NSEI"  # Nifty 50 index - benchmark for the relative label and market_relative_momentum_20d
VIX_TICKER = "^INDIAVIX"  # regime feature - available on yfinance with full 5y+ history
CACHE_DIR = os.path.join("ml", "data")

BASE_FEATURE_COLUMNS = [
    "rsi_14",
    # rsi_distance_from_50 / rsi_slope and macd_hist_norm /
    # macd_extension_ratio replace the original spec's discrete
    # is_rsi_under_50_and_rising / macd_state - those two are no longer
    # emitted by ml.features.build_feature_matrix() as model-facing columns
    # (see ml/features.py FEATURE_COLUMNS "Round 4" comment), so this list
    # must track that swap to stay runnable.
    "rsi_distance_from_50",
    "rsi_slope",
    "macd_hist_norm",
    "macd_extension_ratio",
    "days_since_macd_crossover",
    "pct_dist_to_fib_618",
    "pct_dist_to_fib_50",
    "volume_vs_20d_avg",
    "promoter_holding_quarterly_change",
    "profit_growth_yoy",
]


def _price_cache_path(period):
    return os.path.join(CACHE_DIR, f"price_history_{period}.pkl")


def _index_cache_path(period):
    return os.path.join(CACHE_DIR, f"index_history_{period}.pkl")


def _download_price_history_fresh(tickers, period):
    frames = []
    failed = []

    for i, ticker in enumerate(tickers, start=1):
        try:
            df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                failed.append(ticker)
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index().rename(columns={
                "Date": "date", "Open": "open", "High": "high",
                "Low": "low", "Close": "close", "Volume": "volume",
            })
            df["ticker"] = ticker
            frames.append(df[["ticker", "date", "open", "high", "low", "close", "volume"]].dropna())
        except Exception as e:
            print(f"[backfill] {ticker} failed: {e}")
            failed.append(ticker)

        if i % 25 == 0 or i == len(tickers):
            print(f"[backfill] fetched {i}/{len(tickers)} tickers ({len(failed)} failed so far)")

    if failed:
        print(f"[backfill] {len(failed)} tickers had no usable data: {failed}")
    if not frames:
        raise RuntimeError("No price data could be downloaded for any ticker.")

    return pd.concat(frames, ignore_index=True)


def download_price_history(tickers, period=DEFAULT_PERIOD, use_cache=True, refresh_cache=False):
    """
    Sequentially downloads daily OHLCV for each ticker, skipping (and
    reporting) any that fail or have no data.

    Caches the full download to ml/data/price_history_{period}.pkl (raw
    downloaded data, not source - see .gitignore) so repeated experiments
    against the same universe/period don't re-hit yfinance every time; a
    cache is only reused if it already covers every requested ticker.
    """
    cache_path = _price_cache_path(period)
    if use_cache and not refresh_cache and os.path.exists(cache_path):
        cached = pd.read_pickle(cache_path)
        if set(tickers).issubset(set(cached["ticker"].unique())):
            print(f"[backfill] using cached price history from {cache_path}")
            return cached[cached["ticker"].isin(tickers)].reset_index(drop=True)
        print(f"[backfill] cache at {cache_path} doesn't cover all requested tickers - re-downloading")

    data = _download_price_history_fresh(tickers, period)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        data.to_pickle(cache_path)
    return data


def empty_fundamentals():
    """Placeholder fundamentals frame with the right schema and zero rows - see module docstring."""
    return pd.DataFrame({
        "ticker": pd.Series(dtype=str),
        "quarter_end": pd.Series(dtype="datetime64[ns]"),
        "promoter_holding_pct": pd.Series(dtype=float),
        "net_profit": pd.Series(dtype=float),
    })


def download_index_history(ticker=INDEX_TICKER, period=DEFAULT_PERIOD, use_cache=True, refresh_cache=False):
    """Benchmark index history (date, close only) - used for the relative label and market_relative_momentum_20d. Cached the same way as download_price_history()."""
    cache_path = _index_cache_path(period)
    if use_cache and not refresh_cache and os.path.exists(cache_path):
        print(f"[backfill] using cached index history from {cache_path}")
        return pd.read_pickle(cache_path)

    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"Could not download benchmark index data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "date", "Close": "close"})
    result = df[["date", "close"]].dropna()

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        result.to_pickle(cache_path)
    return result


def _vix_cache_path(period):
    return os.path.join(CACHE_DIR, f"vix_history_{period}.pkl")


def download_vix_history(ticker=VIX_TICKER, period=DEFAULT_PERIOD, use_cache=True, refresh_cache=False):
    """India VIX history (date, close only) - regime feature (see ml.features REGIME_FEATURE_COLUMNS). Same shape/caching pattern as download_index_history()."""
    cache_path = _vix_cache_path(period)
    if use_cache and not refresh_cache and os.path.exists(cache_path):
        print(f"[backfill] using cached VIX history from {cache_path}")
        return pd.read_pickle(cache_path)

    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"Could not download VIX data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index().rename(columns={"Date": "date", "Close": "close"})
    result = df[["date", "close"]].dropna()

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        result.to_pickle(cache_path)
    return result


def main(tickers=None, period=DEFAULT_PERIOD, model_path=DEFAULT_MODEL_PATH,
         feature_set="broadened", label_kind="relative", refresh_cache=False,
         tp_pct=TAKE_PROFIT_PCT, sl_pct=STOP_LOSS_PCT, max_days=MAX_HOLDING_DAYS):
    tickers = tickers or config.SCAN_UNIVERSE
    feature_cols = FEATURE_COLUMNS if feature_set == "broadened" else BASE_FEATURE_COLUMNS

    print(f"[backfill] downloading {period} of daily history for {len(tickers)} tickers...")
    price_df = download_price_history(tickers, period, refresh_cache=refresh_cache)
    print(f"[backfill] {len(price_df)} price rows across {price_df['ticker'].nunique()} tickers")

    print(f"[backfill] downloading benchmark index ({INDEX_TICKER}) history...")
    index_df = download_index_history(period=period, refresh_cache=refresh_cache)

    print(f"[backfill] building feature matrix (fundamentals-blind, feature_set={feature_set})...")
    features = build_feature_matrix(price_df, empty_fundamentals(), index_df=index_df)
    features = features[["ticker", "date"] + feature_cols]

    print(f"[backfill] computing {label_kind} triple-barrier labels "
          f"(+{tp_pct:.1%}/-{sl_pct:.1%}, scans forward up to {max_days} days per row)...")
    if label_kind == "relative":
        labels = relative_triple_barrier_labels(price_df, index_df, tp_pct=tp_pct, sl_pct=sl_pct, max_days=max_days)
    else:
        labels = triple_barrier_labels(price_df, tp_pct=tp_pct, sl_pct=sl_pct, max_days=max_days)

    merged = features.merge(labels, on=["ticker", "date"], how="left")
    resolved = merged["target"].notna()
    print(f"[backfill] {resolved.sum()} labeled rows ({(~resolved).sum()} unresolved/censored), "
          f"{merged.loc[resolved, 'target'].mean():.1%} positive")

    # Purge window must match the label's own forward-looking horizon (see
    # ml.train.run_cross_validation) - a training row within max_days of a
    # test fold's start was labeled using price action overlapping the test
    # window, regardless of what the DEFAULT 90-day horizon used to be.
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    train_and_save_model(
        merged, model_path, feature_cols=feature_cols, label_kind=label_kind,
        purge_days=max_days, label_config={"tp_pct": tp_pct, "sl_pct": sl_pct, "max_days": max_days},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N tickers from config.SCAN_UNIVERSE (for a quick run).")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance history period, e.g. 2y, 5y, max.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--features", choices=["base", "broadened"], default="broadened")
    parser.add_argument("--label", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore any cached price/index history and re-download from yfinance.")
    parser.add_argument("--tp-pct", type=float, default=TAKE_PROFIT_PCT, help=f"Take-profit barrier as a fraction, e.g. 0.05 for 5%% (default {TAKE_PROFIT_PCT}).")
    parser.add_argument("--sl-pct", type=float, default=STOP_LOSS_PCT, help=f"Stop-loss barrier as a fraction (default {STOP_LOSS_PCT}).")
    parser.add_argument("--max-days", type=int, default=MAX_HOLDING_DAYS, help=f"Label horizon in calendar days; also used as the purged-CV purge window (default {MAX_HOLDING_DAYS}).")
    args = parser.parse_args()

    universe = config.SCAN_UNIVERSE[: args.limit] if args.limit else None
    main(tickers=universe, period=args.period, model_path=args.model_path,
         feature_set=args.features, label_kind=args.label, refresh_cache=args.refresh_cache,
         tp_pct=args.tp_pct, sl_pct=args.sl_pct, max_days=args.max_days)
