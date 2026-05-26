"""yfinance adapter: forensic fundamentals + price history. Returns typed models.

All network calls are wrapped so a single bad ticker never raises into the pipeline.
Missing fields stay None (flagged downstream as data warnings) rather than guessed.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..models import Fundamentals
from .cache import cached


def _get_info(ticker: str) -> dict:
    def produce() -> dict:
        try:
            info = yf.Ticker(ticker).get_info()
            return info if isinstance(info, dict) else {}
        except Exception:
            return {}

    return cached(f"info:{ticker}", produce)


def get_fundamentals(ticker: str) -> Fundamentals:
    i = _get_info(ticker)
    if not i:
        return Fundamentals()
    return Fundamentals(
        name=i.get("longName") or i.get("shortName"),
        sector=i.get("sector"),
        industry=i.get("industry"),
        market_cap=i.get("marketCap"),
        pe_ratio=i.get("trailingPE"),
        forward_pe=i.get("forwardPE"),
        pb_ratio=i.get("priceToBook"),
        ps_ratio=i.get("priceToSalesTrailing12Months"),
        peg_ratio=i.get("trailingPegRatio") or i.get("pegRatio"),
        debt_to_equity=i.get("debtToEquity"),
        current_ratio=i.get("currentRatio"),
        quick_ratio=i.get("quickRatio"),
        gross_margin=i.get("grossMargins"),
        operating_margin=i.get("operatingMargins"),
        profit_margin=i.get("profitMargins"),
        roe=i.get("returnOnEquity"),
        roa=i.get("returnOnAssets"),
        revenue_growth=i.get("revenueGrowth"),
        earnings_growth=i.get("earningsGrowth"),
        free_cashflow=i.get("freeCashflow"),
        total_debt=i.get("totalDebt"),
        total_cash=i.get("totalCash"),
        dividend_yield=i.get("dividendYield"),
    )


def get_closes(ticker: str, days: int) -> pd.Series:
    """Daily close prices for the lookback window as a date-indexed float Series.

    Empty Series on failure or no data.
    """

    def produce() -> dict:
        try:
            df = yf.Ticker(ticker).history(period=f"{max(days, 30)}d", auto_adjust=True)
            if df is None or df.empty or "Close" not in df:
                return {}
            closes = df["Close"].dropna()
            return {ts.strftime("%Y-%m-%d"): float(v) for ts, v in closes.items()}
        except Exception:
            return {}

    data = cached(f"closes:{ticker}:{days}", produce)
    if not data:
        return pd.Series(dtype=float)
    s = pd.Series(data, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()
