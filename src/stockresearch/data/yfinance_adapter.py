"""yfinance adapter: forensic fundamentals + price history. Returns typed models.

All network calls are wrapped so a single bad ticker never raises into the pipeline.
Missing fields stay None (flagged downstream as data warnings) rather than guessed.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..models import Fundamentals, Statements
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


# Statement line item -> (yfinance statement attr, row label).
_STATEMENT_MAP: dict[str, tuple[str, str]] = {
    "total_assets": ("balance_sheet", "Total Assets"),
    "total_liabilities": ("balance_sheet", "Total Liabilities Net Minority Interest"),
    "total_equity": ("balance_sheet", "Stockholders Equity"),
    "current_assets": ("balance_sheet", "Current Assets"),
    "current_liabilities": ("balance_sheet", "Current Liabilities"),
    "inventory": ("balance_sheet", "Inventory"),
    "total_debt": ("balance_sheet", "Total Debt"),
    "cash": ("balance_sheet", "Cash And Cash Equivalents"),
    "revenue": ("income_stmt", "Total Revenue"),
    "gross_profit": ("income_stmt", "Gross Profit"),
    "operating_income": ("income_stmt", "Operating Income"),
    "net_income": ("income_stmt", "Net Income"),
    "interest_expense": ("income_stmt", "Interest Expense"),
    "operating_cashflow": ("cashflow", "Operating Cash Flow"),
    "capex": ("cashflow", "Capital Expenditure"),
    "free_cashflow": ("cashflow", "Free Cash Flow"),
}


def get_statements(ticker: str) -> Statements:
    """Latest annual balance-sheet / income / cash-flow line items. Missing rows stay None."""

    def produce() -> dict:
        out: dict = {}
        try:
            tk = yf.Ticker(ticker)
            frames = {
                "balance_sheet": tk.balance_sheet,
                "income_stmt": tk.income_stmt,
                "cashflow": tk.cashflow,
            }
        except Exception:
            return {}
        for field, (stmt, row) in _STATEMENT_MAP.items():
            df = frames.get(stmt)
            try:
                if df is None or df.empty or row not in df.index:
                    continue
                val = df.loc[row].iloc[0]
                if val is not None and not pd.isna(val):
                    out[field] = float(val)
            except Exception:
                continue
        try:
            bs = frames["balance_sheet"]
            if bs is not None and not bs.empty:
                out["period"] = bs.columns[0].strftime("%Y-%m-%d")
        except Exception:
            pass
        return out

    data = cached(f"statements:{ticker}", produce)
    return Statements(**data) if data else Statements()


def price_on_or_after(ticker: str, target_iso: str) -> float | None:
    """First available close on/after target_iso (handles weekends/holidays). None if none."""
    from datetime import date, timedelta

    def produce() -> dict:
        try:
            start = date.fromisoformat(target_iso)
            end = start + timedelta(days=10)
            df = yf.Ticker(ticker).history(
                start=start.isoformat(), end=end.isoformat(), auto_adjust=True
            )
            if df is None or df.empty or "Close" not in df:
                return {}
            return {"price": float(df["Close"].dropna().iloc[0])}
        except Exception:
            return {}

    data = cached(f"priceon:{ticker}:{target_iso}", produce)
    return data.get("price")


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
