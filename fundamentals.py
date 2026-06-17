"""Company fundamentals via yfinance + screener.in — cached data layer for the AI Commentary tab."""

import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

_PL_ROWS = [
    "Total Revenue",
    "Gross Profit",
    "EBITDA",
    "Operating Income",
    "Net Income",
]

_CF_ROWS = [
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
    "Free Cash Flow",
]

_SCREENER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

# One requests.Session per thread — reuses the TCP/TLS connection to
# screener.in within a prefetch batch instead of opening a fresh socket
# for every ticker.
_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(_SCREENER_HEADERS)
        _thread_local.session = s
    return _thread_local.session


def _ticker_to_screener(ticker: str) -> str:
    return ticker.replace(".NS", "").replace(".BO", "")


def _parse_screener_section(rows, header_cells) -> tuple[dict, list] | None:
    """Extract last-4-column data from a screener.in table's row list."""
    last4 = header_cells[1:][-4:]
    if not last4:
        return None
    col_indices = [header_cells.index(q) for q in last4]
    data = {}
    for row in rows[1:]:
        cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
        if not cells:
            continue
        label = cells[0].rstrip("+").strip()
        if not label or label == "No. of Shareholders":
            continue
        data[label] = [cells[ci] if ci < len(cells) else "—" for ci in col_indices]
    return (data, last4) if data else None


@st.cache_data(ttl=604800, show_spinner=False)
def _scrape_screener_page(ticker: str) -> dict:
    """Fetch screener.in once per ticker and parse both Shareholding and Cash Flow sections.

    Returns {"shareholding": (data, cols) | None, "cashflow": (data, cols) | None}.
    Single HTTP request covers what used to be two separate fetches.
    """
    slug = _ticker_to_screener(ticker)
    result: dict = {"shareholding": None, "cashflow": None}
    for suffix in ("/consolidated/", "/"):
        try:
            url = f"https://www.screener.in/company/{slug}{suffix}"
            r = _get_session().get(url, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for section in soup.find_all("section"):
                h2 = section.find("h2")
                if not h2:
                    continue
                h2_text = h2.get_text()
                is_sh = "Shareholding" in h2_text
                is_cf = "Cash Flow" in h2_text
                if not is_sh and not is_cf:
                    continue
                table = section.find("table")
                if not table:
                    continue
                rows = table.find_all("tr")
                if not rows:
                    continue
                header_cells = [td.get_text(strip=True) for td in rows[0].find_all(["th", "td"])]
                parsed = _parse_screener_section(rows, header_cells)
                if parsed:
                    result["shareholding" if is_sh else "cashflow"] = parsed
            break  # page fetched successfully; no need to try non-consolidated fallback
        except Exception as e:
            print(f"[fundamentals] screener scrape failed for {ticker}: {e}")
    return result


def _screener_to_df(raw) -> pd.DataFrame | None:
    if raw is None:
        return None
    data, cols = raw
    return pd.DataFrame(data, index=cols).T


@st.cache_data(ttl=604800, show_spinner=False)
def get_shareholding(ticker: str) -> pd.DataFrame | None:
    """Quarterly shareholding pattern (Promoters/FII/DII/Public) from screener.in."""
    return _screener_to_df(_scrape_screener_page(ticker).get("shareholding"))


@st.cache_data(ttl=604800, show_spinner=False)
def get_screener_cashflow(ticker: str) -> pd.DataFrame | None:
    """Annual cash flow from screener.in."""
    return _screener_to_df(_scrape_screener_page(ticker).get("cashflow"))


@st.cache_data(ttl=604800, show_spinner=False)
def get_company_basics(ticker: str):
    """
    Returns a dict with:
      info              — yfinance .info dict (key ratios, metadata)
      quarterly_pl      — quarterly income statement (last 4 quarters)
      annual_cashflow   — annual cash flow from yfinance (fallback)
      screener_cashflow — annual cash flow from screener.in (primary)
      shareholding      — quarterly Promoter/FII/DII/Public % from screener.in
    Returns None on failure.

    yfinance calls (info, quarterly P&L, annual CF) and the screener.in fetch
    run in parallel — wall time ≈ slowest single call rather than their sum.
    """
    try:
        def _fetch_info():
            try:
                return yf.Ticker(ticker).info or {}
            except Exception:
                return {}

        def _fetch_q_pl():
            try:
                return yf.Ticker(ticker).quarterly_income_stmt
            except Exception:
                return pd.DataFrame()

        def _fetch_annual_cf():
            try:
                return yf.Ticker(ticker).cashflow
            except Exception:
                return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=4) as ex:
            f_info = ex.submit(_fetch_info)
            f_qpl = ex.submit(_fetch_q_pl)
            f_cf = ex.submit(_fetch_annual_cf)
            f_screener = ex.submit(_scrape_screener_page, ticker)
            info = f_info.result()
            q_pl = f_qpl.result()
            annual_cf = f_cf.result()
            screener = f_screener.result()

        return {
            "info": info,
            "quarterly_pl": q_pl,
            "annual_cashflow": annual_cf,
            "screener_cashflow": _screener_to_df(screener.get("cashflow")),
            "shareholding": _screener_to_df(screener.get("shareholding")),
        }
    except Exception as e:
        print(f"[fundamentals] {ticker}: {e}")
        return None


def prefetch_all(tickers):
    """Warms get_company_basics for all tickers in parallel (8 outer workers,
    each spawning 4 inner workers for yfinance + screener — runs once per scan
    then hits cache for the rest of the week)."""
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as ex:
        list(ex.map(get_company_basics, tickers))
