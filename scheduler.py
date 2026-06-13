"""
On-demand pipeline runner.

By design this agent does NOT run on an automatic background schedule.
The full pipeline (scan -> score -> AI commentary -> persist) is triggered
manually:

  - from the Streamlit dashboard via the "Run Full Scan Now" button, or
  - directly from the command line: `python scheduler.py`

`is_market_open()` is kept as an informational helper so the dashboard can
indicate whether the data reflects a live session or the last closed
session.
"""

import config
import db_handler
from ai_analyst import get_ai_recommendations
from strategy import generate_shortlist
from ta_engine import is_market_open  # re-exported for callers (e.g. app.py)


def run_pipeline(tickers=None, progress_callback=None):
    """
    Executes the full pipeline once: scan -> score -> AI commentary -> persist.

    Returns (shortlist_df, ai_commentary, scan_date).
    """
    db_handler.init_db()

    if tickers is None:
        tickers = config.SCAN_UNIVERSE

    shortlist = generate_shortlist(tickers=tickers, progress_callback=progress_callback)
    ai_commentary = get_ai_recommendations(shortlist)
    scan_date = db_handler.save_scan_results(shortlist, ai_commentary, universe_size=len(tickers))

    return shortlist, ai_commentary, scan_date


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
