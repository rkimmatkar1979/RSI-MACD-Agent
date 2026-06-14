"""
LLM analyst integration (any OpenAI-compatible chat completions API).

Takes the mathematical shortlist produced by strategy.py, compresses it
into a dense plain-text block, and asks the configured LLM for a concise
swing-trading execution plan (Entry / Stop-Loss / Take-Profit) for the
top-ranked picks (config.AI_TOP_PICKS_COUNT), calibrated for a 2-3 week
holding horizon. Recent news headlines for those picks are fetched via
yfinance and included as extra context.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config
import db_handler
from ta_engine import fetch_news

# Explanation of the decimal-valued columns sent to the AI, also surfaced to
# the user in app.py so both sides interpret the numbers the same way.
NOTATION_NOTE = (
    "Notation: 'MACD-Signal diff' is the MACD histogram value (MACD line minus "
    "Signal line) in price-point decimals - positive means the MACD line is "
    "ABOVE the signal line (bullish momentum), negative means it is BELOW "
    "(bearish momentum), and values near zero mean the lines are converging "
    "toward a crossover; the '(up)'/'(down)' tag shows whether this histogram "
    "value rose or fell vs. the previous session. '% from 52W high' is how far "
    "the current price sits below its 52-week high as a decimal fraction "
    "(0.00 = at the 52-week high, 0.10 = 10% below it). 'Volume ratio' is the "
    "latest session's volume divided by its 20-day average volume (1.00 = "
    "exactly average, 1.5 = 50% above average). 'Buy/Sell pressure' is a "
    "volume-weighted proxy (NOT live order-book data): over the last 20 "
    "sessions, the % of total volume that traded on up days (close >= open) "
    "vs down days. 'Sector trend' is the average return of all scanned stocks "
    "in the same sector over the last 10 sessions, giving industry-level "
    "context for the pick."
)


def _format_shortlist(shortlist_df):
    """Compresses the shortlist DataFrame into one dense line per ticker."""
    lines = []
    for _, row in shortlist_df.iterrows():
        reasons = "; ".join(row["reasons"]) if row["reasons"] else "-"
        macd_diff = row["macd_hist"]
        macd_side = "MACD above Signal/bullish" if macd_diff > 0 else "MACD below Signal/bearish"
        lines.append(
            f"{row['ticker']} ({row['sector']}) | CMP {row['close']:.2f} | RSI {row['rsi']:.1f} | "
            f"MACD {row['macd_line']:.3f}/{row['macd_signal']:.3f} "
            f"(MACD-Signal diff {macd_diff:.3f} ({row['macd_hist_direction']}), {macd_side}) | "
            f"90D Fib range {row['fib_low']:.2f}-{row['fib_high']:.2f}, nearest {row['nearest_fib_level']} "
            f"@ {row['nearest_fib_price']:.2f} ({row['fib_distance_pct'] * 100:.2f}% away) | "
            f"52W range {row['week52_low']:.2f}-{row['week52_high']:.2f} "
            f"({row['pct_from_52w_high'] * 100:.2f}% from 52W high) | "
            f"Volume {row['volume_ratio']:.2f}x its 20D average | "
            f"Buy/Sell pressure {row['buy_pct']:.0f}%/{row['sell_pct']:.0f}% | "
            f"Sector trend {row['sector_trend_pct'] * 100:+.2f}% (10D) | "
            f"Prev session ({row['prev_session_date']}) O {row['prev_session_open']:.2f} / "
            f"C {row['prev_session_close']:.2f} | "
            f"Score {row['score']}/100 | Signals: {reasons}"
        )
        # The first sentence of macd_pattern duplicates the MACD trend
        # summary already included in `reasons` (see strategy.score_setup) -
        # only send the remaining sentences, if any, to avoid repeating it.
        macd_pattern_rest = ". ".join(row["macd_pattern"].split(". ")[1:])
        if macd_pattern_rest:
            lines.append(f"    MACD pattern detail: {macd_pattern_rest}")
    return "\n".join(lines)


def _format_news(tickers):
    """
    Fetches recent headlines for each ticker and formats them as a block.

    Tickers are fetched concurrently (config.SCAN_MAX_WORKERS threads) since
    each is a separate network round-trip to Yahoo Finance - fetching
    AI_TOP_PICKS_COUNT tickers sequentially was the dominant cost of an AI
    commentary request. Output is still assembled in `tickers` order so the
    formatted block (and therefore the resulting prompt hash) stays
    deterministic regardless of fetch completion order.

    Returns a plain-text section listing up to config.NEWS_HEADLINE_COUNT
    headlines per ticker, or a note that none were found.
    """
    headlines_by_ticker = {}
    with ThreadPoolExecutor(max_workers=min(len(tickers), config.SCAN_MAX_WORKERS)) as executor:
        future_to_ticker = {executor.submit(fetch_news, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                headlines_by_ticker[ticker] = future.result()
            except Exception as e:
                print(f"[ai_analyst] Failed to fetch news for {ticker}: {e}")
                headlines_by_ticker[ticker] = []

    lines = []
    for ticker in tickers:
        headlines = headlines_by_ticker.get(ticker, [])
        if headlines:
            lines.append(f"{ticker}:")
            for headline in headlines:
                lines.append(f"  - {headline}")
        else:
            lines.append(f"{ticker}: (no recent news available)")
    return "\n".join(lines)


def get_ai_recommendations(shortlist_df):
    """
    Sends the shortlist to xAI's Grok model and returns its commentary as
    plain text/markdown. On any failure, returns a human-readable fallback
    message instead of raising - the mathematical shortlist remains valid
    and usable even if the AI call fails.
    """
    if shortlist_df is None or shortlist_df.empty:
        return "No qualifying technical setups were found in this scan."

    if not config.LLM_API_KEY or config.LLM_API_KEY == "your_llm_api_key_here":
        return (
            "AI analysis skipped: LLM_API_KEY is not configured (it is empty or "
            "still set to the placeholder value from .env.example). Get a free "
            "API key (e.g. from https://console.groq.com/keys) and set "
            "LLM_API_KEY in your .env file to enable AI commentary. "
            "LLM_API_URL/LLM_MODEL can also be changed there to point at a "
            "different OpenAI-compatible provider."
        )

    # Only the top picks get a write-up below, so only send their data -
    # the rest of the shortlist would be pure overhead with no effect on
    # the output.
    top_picks_df = shortlist_df.head(config.AI_TOP_PICKS_COUNT)
    top_picks = top_picks_df["ticker"].tolist()
    shortlist_text = _format_shortlist(top_picks_df)
    news_text = _format_news(top_picks)

    prompt = (
        "You are reviewing a mathematically pre-screened shortlist of NSE-listed "
        "stocks (plus Gold/Silver ETFs, always included). Each line shows the "
        "current market price (CMP), sector, RSI(14), MACD line/signal, the 90-day "
        "Fibonacci retracement range with the nearest level, the 52-week price "
        "range, volume vs its 20-day average, buy/sell pressure, sector trend, the "
        "previous session's open/close, a composite technical score (0-100), and "
        "the specific signals that triggered the screen.\n\n"
        f"{NOTATION_NOTE}\n\n"
        f"Shortlist:\n{shortlist_text}\n\n"
        f"Recent news headlines for these stocks:\n{news_text}\n\n"
        f"For all {len(top_picks)} stocks above ({', '.join(top_picks)}), write a "
        "concise professional trading note as a markdown bullet list (NOT "
        "paragraphs of prose). Use a '### TICKER' heading for each stock, "
        "followed by exactly 4 bullet points:\n"
        "- **Why**: the confluence of signals (Fibonacci level, RSI, MACD, "
        "volume, sector trend) that makes this a buy/sell candidate right now.\n"
        "- **Entry**: WHEN to enter - e.g. immediately at CMP, on a pullback to "
        "a specific level, or on a breakout above a specific level.\n"
        "- **Stop-Loss**: a specific level with a brief rationale.\n"
        "- **Take-Profit**: a specific target.\n\n"
        "Keep each bullet to 1-2 short sentences. Calibrate all levels for a "
        "SWING TRADE with a 2-3 WEEK holding horizon (not an intraday or scalp "
        "trade) - use the given Fibonacci range, 52-week range, and current "
        "price action to justify the levels. If a news headline is material "
        "(e.g. earnings, regulatory action, M&A, guidance change), briefly "
        "factor it into the relevant bullet; ignore routine or irrelevant "
        "headlines."
    )

    # If a previous scan sent this exact same shortlist/news data, reuse its
    # commentary instead of calling the LLM again - the prompt (and therefore
    # the analysis) would be identical.
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cached = db_handler.get_cached_ai_commentary(prompt_hash)
    if cached is not None:
        return cached

    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": config.LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert quantitative swing-trading analyst covering "
                    "Indian equity markets (NSE). You write concise, specific, "
                    "actionable trade plans based strictly on the technical data "
                    "you are given."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
    }

    try:
        response = requests.post(
            config.LLM_API_URL,
            headers=headers,
            json=payload,
            timeout=config.LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        db_handler.save_ai_commentary_cache(prompt_hash, content)
        return content

    except requests.exceptions.Timeout:
        return (
            "AI analysis unavailable: the LLM API request timed out. "
            "The mathematical shortlist above is still valid."
        )
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("error", e.response.text)
        except ValueError:
            detail = e.response.text
        return (
            f"AI analysis unavailable: LLM API returned "
            f"{e.response.status_code} error: {detail}. "
            "The mathematical shortlist above is still valid."
        )
    except requests.exceptions.RequestException as e:
        return (
            f"AI analysis unavailable: could not reach the LLM API ({e}). "
            "The mathematical shortlist above is still valid."
        )
    except (KeyError, IndexError, ValueError) as e:
        return f"AI analysis unavailable: unexpected response format from LLM API ({e})."
