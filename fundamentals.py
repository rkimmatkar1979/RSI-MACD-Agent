"""Company fundamentals via yfinance — cached data layer for the AI Commentary tab."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
import yfinance as yf

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


@st.cache_data(ttl=3600, show_spinner=False)
def get_company_basics(ticker: str):
    """
    Returns a dict with:
      info            — yfinance .info dict (key ratios, metadata)
      quarterly_pl    — quarterly income statement (last 4 quarters)
      annual_cashflow — annual cash flow statement (last 4 FYs)
                        Quarterly CF is unavailable for NSE stocks via yfinance.
      major_holders   — % insiders / % institutions DataFrame
    Returns None on failure.
    """
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}

        try:
            q_pl = t.quarterly_income_stmt   # rows=line items, cols=quarter end dates
        except Exception:
            q_pl = pd.DataFrame()

        try:
            annual_cf = t.cashflow           # annual; quarterly CF unavailable for NSE
        except Exception:
            annual_cf = pd.DataFrame()

        try:
            major_holders = t.major_holders
        except Exception:
            major_holders = None

        return {
            "info": info,
            "quarterly_pl": q_pl,
            "annual_cashflow": annual_cf,
            "major_holders": major_holders,
        }
    except Exception as e:
        print(f"[fundamentals] {ticker}: {e}")
        return None


def prefetch_all(tickers):
    """Warms the get_company_basics cache for all tickers in parallel (8 threads)."""
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as ex:
        list(ex.map(get_company_basics, tickers))
