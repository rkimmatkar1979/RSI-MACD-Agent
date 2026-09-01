"""
On-demand pipeline runner.

This agent does NOT run on any OS-level/cron schedule (no such
infrastructure exists for this deployment). The full pipeline
(scan -> score -> AI commentary -> persist) is triggered by one of:

  - the Streamlit dashboard's "Run Full Scan Now" button (blocking, full-page
    progress view - see app.py's scan_in_progress flow),
  - maybe_auto_scan() below, called once per app.py page load - a lightweight
    "whoever loads the page first each trading day kicks off that day's scan
    in the background" substitute for real scheduling, or
  - directly from the command line: `python scheduler.py`

`is_market_open()` is kept as an informational helper so the dashboard can
indicate whether the data reflects a live session or the last closed
session.
"""

import threading
from datetime import datetime

import config
import db_handler
from ai_analyst import get_ai_recommendations, get_sector_outlook
from ml.paper_trade import run_in_background as run_ml_paper_trade_in_background
from strategy import generate_shortlist
from ta_engine import is_market_open, is_trading_day  # re-exported for callers (e.g. app.py)


def run_pipeline(tickers=None, progress_callback=None):
    """
    Executes the full pipeline once: scan -> score -> AI commentary
    (per-stock + sector-wide) -> persist.

    Returns (shortlist_df, ai_commentary, scan_date).
    """
    db_handler.init_db()

    if tickers is None:
        tickers = config.SCAN_UNIVERSE

    shortlist, sector_trend_df = generate_shortlist(tickers=tickers, progress_callback=progress_callback)
    ai_commentary = get_ai_recommendations(shortlist)
    sector_outlook = get_sector_outlook(sector_trend_df)
    scan_date = db_handler.save_scan_results(
        shortlist, ai_commentary, universe_size=len(tickers),
        sector_trend_df=sector_trend_df, sector_outlook=sector_outlook,
    )

    # Fire-and-forget: logs/resolves the experimental ML model's paper-trade
    # picks on a background thread, riding this same manual trigger instead
    # of needing separate scheduling infrastructure. Never blocks or can
    # fail this function - see ml/paper_trade.py's own internal try/except.
    #
    # Scoped to just today's rule-based shortlist (not config.SCAN_UNIVERSE)
    # and top_pct=1.0 (log every one of them, not just the model's own
    # top-10%) - this deliberately makes it track the SAME stocks the ML
    # tab already shows confidence for, rather than independently scoring
    # the full universe.
    if not shortlist.empty:
        run_ml_paper_trade_in_background(tickers=shortlist["ticker"].tolist(), top_pct=1.0)

    return shortlist, ai_commentary, scan_date


# In-process guard for maybe_auto_scan() below: an in-memory dict (not the
# DB) because the DB check alone isn't enough - the first triggered scan
# takes a couple of minutes to land, and every other visitor loading the
# page during that window would still see "no scan for today yet" and would
# each kick off their own duplicate scan without this. Streamlit apps this
# size run as a single process, so a plain module-level dict is a valid
# process-wide lock; it resets (and a fresh auto-scan can fire again) if the
# process restarts, which is fine - worst case is one extra scan that day.
_auto_scan_state = {"triggered_date": None, "lock": threading.Lock()}


def maybe_auto_scan():
    """
    Kicks off run_pipeline() on a background thread the first time anyone
    loads the app on a trading day with no scan yet recorded for that date -
    a "no scheduling infra" substitute for a real daily cron: whichever
    visitor happens to load the page first each day triggers that day's scan
    for everyone, instead of requiring someone to remember to click
    "Run Full Scan Now".

    Non-blocking and silent on failure (mirrors ml.paper_trade's
    run_in_background) - the triggering visitor keeps seeing the previous
    day's results immediately; the fresh scan lands a couple of minutes
    later via save_scan_results' own cache-clearing (picked up on that
    visitor's, or anyone else's, next rerun).

    Returns True if this call actually started a new auto-scan (so the
    caller can show a "today's scan just started" notice), False otherwise
    (not a trading day, already triggered this process, or today's scan
    already exists).
    """
    if not is_trading_day():
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    with _auto_scan_state["lock"]:
        if _auto_scan_state["triggered_date"] == today:
            return False
        _auto_scan_state["triggered_date"] = today

    if today in db_handler.get_available_scan_dates():
        return False

    def _safe_auto_run():
        try:
            run_pipeline()
        except Exception as e:
            print(f"[scheduler] Auto-scan failed: {e}")

    threading.Thread(target=_safe_auto_run, daemon=True).start()
    return True


if __name__ == "__main__":
    print(f"Market open (IST): {is_market_open()}")
    print(f"Scanning {len(config.SCAN_UNIVERSE)} tickers... this can take a few minutes.\n")

    def _progress(i, total, ticker):
        print(f"[{i:3d}/{total}] {ticker}")

    shortlist_df, commentary, scan_date = run_pipeline(progress_callback=_progress)

    print(f"\nScan complete for {scan_date}. {len(shortlist_df)} setup(s) found.\n")
    if not shortlist_df.empty:
        print(shortlist_df[
            ["ticker", "close", "rsi", "macd_hist", "nearest_fib_level", "score"]
        ].to_string(index=False))

    print("\n--- AI Commentary ---\n")
    print(commentary)
