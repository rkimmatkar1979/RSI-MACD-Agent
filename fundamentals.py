"""Company fundamentals via yfinance — cached data layer for the AI Commentary tab."""

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
import yfinance as yf

_PL_ROWS = [
    "Total Revenue",
    "Gross Profit",
    "Operating Income",
    "EBITDA",
    "Net Income Common Stockholders",
    "Net Income",
]

_CF_ROWS = [
    "Operating Cash Flow",
    "Investing Cash Flow",
    "Financing Cash Flow",
    "Capital Expenditure",
    "Free Cash Flow",
]


@st.cache_data(ttl=3600, show_spinner=False)
def get_company_basics(ticker: str):
    """Returns {info, quarterly_pl, quarterly_cashflow, major_holders} or None. Cached 1 hour."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        try:
            q_pl = t.quarterly_financials
        except Exception:
            q_pl = pd.DataFrame()
        try:
            q_cf = t.quarterly_cashflow
        except Exception:
            q_cf = pd.DataFrame()
        try:
            major_holders = t.major_holders
        except Exception:
            major_holders = None
        return {
            "info": info,
            "quarterly_pl": q_pl,
            "quarterly_cashflow": q_cf,
            "major_holders": major_holders,
        }
    except Exception as e:
        print(f"[fundamentals] {ticker}: {e}")
        return None


def prefetch_all(tickers):
    """Warms the get_company_basics cache for all tickers in parallel (8 threads)."""
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as ex:
        list(ex.map(get_company_basics, tickers))
