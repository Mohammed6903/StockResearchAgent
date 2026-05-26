"""Index membership. S&P 500 constituents pulled (and cached) from Wikipedia.

Falls back to a small static set of large caps if the fetch fails, so a scan can
still run offline. The union with the watchlist is handled by the universe resolver.
"""

from __future__ import annotations

from .cache import cached

_FALLBACK_SP500 = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "BRK-B", "JPM", "V",
    "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "COST", "ORCL", "BAC",
]


def sp500_tickers() -> list[str]:
    def produce() -> list[str]:
        try:
            import pandas as pd

            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            syms = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False)
            return sorted({s.strip().upper() for s in syms if s.strip()})
        except Exception:
            return list(_FALLBACK_SP500)

    out = cached("index:sp500", produce)
    return out or list(_FALLBACK_SP500)


def index_tickers(name: str) -> list[str]:
    if (name or "").upper() == "SP500":
        return sp500_tickers()
    return []
