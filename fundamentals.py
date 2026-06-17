"""Company fundamentals via yfinance + screener.in — cached data layer for the AI Commentary tab."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

# Rows to extract from quarterly income statement (in display order)
_PL_ROWS = [
    "Total Revenue",
    "Gross Profit",
    "EBITDA",
    "Operating Income",
    "Net Income",
]

# Rows to extract from annual cash flow statement
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


def _ticker_to_screener(ticker: str) -> str:
    """Convert NSE ticker (RELIANCE.NS) to screener.in slug (RELIANCE)."""
    return ticker.replace(".NS", "").replace(".BO", "")


@st.cache_data(ttl=604800, show_spinner=False)
def get_shareholding(ticker: str) -> pd.DataFrame | None:
    """
    Scrapes the last 4 quarters of shareholding pattern from screener.in.
    Returns a DataFrame (rows = Promoters/FIIs/DIIs/Government/Public,
    cols = quarter labels) or None on failure.
    """
    slug = _ticker_to_screener(ticker)
    for suffix in ("/consolidated/", "/"):
        try:
            url = f"https://www.screener.in/company/{slug}{suffix}"
            r = requests.get(url, headers=_SCREENER_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for section in soup.find_all("section"):
                h2 = section.find("h2")
                if not h2 or "Shareholding" not in h2.text:
                    continue
                table = section.find("table")
                if not table:
                    break
                rows = table.find_all("tr")
                if not rows:
                    break
                # First row is the header (quarter labels)
                header_cells = [td.get_text(strip=True) for td in rows[0].find_all(["th", "td"])]
                # Last 4 quarter columns (skip the empty label cell at index 0)
                quarter_cols = header_cells[1:][-4:]
                col_indices = [header_cells.index(q) for q in quarter_cols]

                data = {}
                for row in rows[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
                    if not cells:
                        continue
                    label = cells[0].rstrip("+").strip()
                    if label in ("No. of Shareholders", ""):
                        continue
                    values = []
                    for ci in col_indices:
                        val = cells[ci] if ci < len(cells) else "—"
                        values.append(val)
                    data[label] = values

                if data:
                    return pd.DataFrame(data, index=quarter_cols).T
            break
        except Exception as e:
            print(f"[fundamentals] shareholding scrape failed for {ticker}: {e}")
    return None


@st.cache_data(ttl=604800, show_spinner=False)
def get_screener_cashflow(ticker: str) -> pd.DataFrame | None:
    """
    Scrapes the annual Cash Flow statement from screener.in.
    Returns a DataFrame (rows = CF line items, cols = FY labels, last 4 FYs)
    or None on failure.
    """
    slug = _ticker_to_screener(ticker)
    for suffix in ("/consolidated/", "/"):
        try:
            url = f"https://www.screener.in/company/{slug}{suffix}"
            r = requests.get(url, headers=_SCREENER_HEADERS, timeout=10)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for section in soup.find_all("section"):
                h2 = section.find("h2")
                if not h2 or "Cash Flow" not in h2.text:
                    continue
                table = section.find("table")
                if not table:
                    break
                rows = table.find_all("tr")
                if not rows:
                    break
                header_cells = [td.get_text(strip=True) for td in rows[0].find_all(["th", "td"])]
                fy_cols = header_cells[1:][-4:]
                col_indices = [header_cells.index(q) for q in fy_cols]
                data = {}
                for row in rows[1:]:
                    cells = [td.get_text(strip=True) for td in row.find_all(["th", "td"])]
                    if not cells:
                        continue
                    label = cells[0].strip()
                    if not label:
                        continue
                    values = [cells[ci] if ci < len(cells) else "—" for ci in col_indices]
                    data[label] = values
                if data:
                    return pd.DataFrame(data, index=fy_cols).T
            break
        except Exception as e:
            print(f"[fundamentals] cashflow scrape failed for {ticker}: {e}")
    return None


@st.cache_data(ttl=604800, show_spinner=False)
def get_company_basics(ticker: str):
    """
    Returns a dict with:
      info              — yfinance .info dict (key ratios, metadata)
      quarterly_pl      — quarterly income statement (last 4 quarters)
      annual_cashflow   — annual cash flow statement from yfinance (fallback)
      screener_cashflow — annual cash flow from screener.in (primary)
      shareholding      — quarterly Promoter/FII/DII/Public % from screener.in
    Returns None on failure.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        try:
            q_pl = t.quarterly_income_stmt
        except Exception:
            q_pl = pd.DataFrame()

        try:
            annual_cf = t.cashflow
        except Exception:
            annual_cf = pd.DataFrame()

        shareholding = get_shareholding(ticker)
        screener_cf = get_screener_cashflow(ticker)

        return {
            "info": info,
            "quarterly_pl": q_pl,
            "annual_cashflow": annual_cf,
            "screener_cashflow": screener_cf,
            "shareholding": shareholding,
        }
    except Exception as e:
        print(f"[fundamentals] {ticker}: {e}")
        return None


def prefetch_all(tickers):
    """Warms get_company_basics (and its screener sub-calls) for all tickers in parallel."""
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as ex:
        list(ex.map(get_company_basics, tickers))
