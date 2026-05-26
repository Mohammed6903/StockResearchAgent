"""Index membership. Constituents pulled (and cached) from Wikipedia.

Supports S&P 500 (US, plain symbols) and NIFTY 50 (India, `.NS` symbols for yfinance).
Falls back to a small static set per index if the fetch fails, so a scan can still run
offline. The union with the watchlist is handled by the universe resolver.
"""

from __future__ import annotations

from .cache import cached

_FALLBACK_SP500 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "COST", "ORCL", "BAC",
]

_FALLBACK_NIFTY50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "HINDUNILVR.NS",
    "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS", "AXISBANK.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS", "SUNPHARMA.NS",
    "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
]


def _symbols_from_wikipedia(url: str, suffix: str = "") -> list[str]:
    """Return the 'Symbol' column from the first Wikipedia table that has one."""
    import pandas as pd

    tables = pd.read_html(url)
    for tbl in tables:
        cols = {str(c).strip().lower(): c for c in tbl.columns}
        if "symbol" in cols:
            syms = tbl[cols["symbol"]].astype(str).str.strip()
            out = {
                (s.replace(".", "-") + suffix if suffix == "" else s + suffix)
                for s in syms
                if s and s.lower() != "nan"
            }
            return sorted(out)
    return []


def sp500_tickers() -> list[str]:
    def produce() -> list[str]:
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            return _symbols_from_wikipedia(url) or list(_FALLBACK_SP500)
        except Exception:
            return list(_FALLBACK_SP500)

    return cached("index:sp500", produce) or list(_FALLBACK_SP500)


def nifty50_tickers() -> list[str]:
    def produce() -> list[str]:
        try:
            url = "https://en.wikipedia.org/wiki/NIFTY_50"
            return _symbols_from_wikipedia(url, suffix=".NS") or list(_FALLBACK_NIFTY50)
        except Exception:
            return list(_FALLBACK_NIFTY50)

    return cached("index:nifty50", produce) or list(_FALLBACK_NIFTY50)


def index_tickers(name: str) -> list[str]:
    key = (name or "").upper()
    if key == "SP500":
        return sp500_tickers()
    if key == "NIFTY50":
        return nifty50_tickers()
    return []
