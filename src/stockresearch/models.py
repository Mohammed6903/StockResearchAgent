"""Pydantic schemas shared across the system. Used as ADK structured outputs too."""

from __future__ import annotations

from datetime import date as Date
from enum import Enum

from pydantic import BaseModel, Field


class Lean(str, Enum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class Action(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


class QuantMetrics(BaseModel):
    """Deterministically computed market metrics. None means data unavailable."""

    last_price: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_1y: float | None = None
    annual_volatility: float | None = None
    beta: float | None = None
    alpha: float | None = None          # annualized, vs benchmark
    sharpe: float | None = None
    max_drawdown: float | None = None


class Fundamentals(BaseModel):
    """Forensic fundamentals pulled from the data source. None means unavailable."""

    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    pe_ratio: float | None = None
    forward_pe: float | None = None
    pb_ratio: float | None = None
    ps_ratio: float | None = None
    peg_ratio: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    quick_ratio: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    profit_margin: float | None = None
    roe: float | None = None
    roa: float | None = None
    revenue_growth: float | None = None
    earnings_growth: float | None = None
    free_cashflow: float | None = None
    total_debt: float | None = None
    total_cash: float | None = None
    dividend_yield: float | None = None


class Statements(BaseModel):
    """Key raw line items from the latest annual filing. None = not reported.

    These are the actual statement figures the ratios are derived from, so explanations can
    show e.g. current_ratio = current_assets / current_liabilities with real numbers.
    """

    period: str | None = None  # e.g. "2026-03-31"
    # Balance sheet
    total_assets: float | None = None
    total_liabilities: float | None = None
    total_equity: float | None = None
    current_assets: float | None = None
    current_liabilities: float | None = None
    inventory: float | None = None
    total_debt: float | None = None
    cash: float | None = None
    # Income statement
    revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    interest_expense: float | None = None
    # Cash flow
    operating_cashflow: float | None = None
    capex: float | None = None
    free_cashflow: float | None = None


class NewsItem(BaseModel):
    title: str
    source: str | None = None
    url: str | None = None
    summary: str | None = None


class TickerSnapshot(BaseModel):
    """Everything gathered for one ticker before LLM reasoning."""

    ticker: str
    fundamentals: Fundamentals = Field(default_factory=Fundamentals)
    statements: Statements = Field(default_factory=Statements)
    metrics: QuantMetrics = Field(default_factory=QuantMetrics)
    news: list[NewsItem] = Field(default_factory=list)
    news_text: str = ""
    data_warnings: list[str] = Field(default_factory=list)


class MacroSignal(BaseModel):
    headline: str
    summary: str = ""
    regions: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    affected_tickers: list[str] = Field(default_factory=list)
    impact: Lean = Lean.NEUTRAL
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    sources: list[NewsItem] = Field(default_factory=list)
    corroborated: bool = True  # backed by >= 2 independent publisher domains


class TickerAnalysis(BaseModel):
    ticker: str
    name: str | None = None
    sector: str | None = None
    lean: Lean = Lean.NEUTRAL
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    score: float = Field(0.0, description="composite rank score, higher = more attractive")
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    reasoning: str = ""
    metrics: QuantMetrics = Field(default_factory=QuantMetrics)
    factors: dict[str, float] = Field(default_factory=dict)  # value/quality/momentum/growth 0..1
    sources: list[NewsItem] = Field(default_factory=list)
    data_warnings: list[str] = Field(default_factory=list)


class Explanation(BaseModel):
    """Beginner-facing teaching layer for one ticker's analysis (built on demand)."""

    ticker: str
    metric_lines: list[str] = Field(default_factory=list)  # one per metric, glossary + value
    walkthrough: str = ""                                   # how numbers + news → the verdict
    news_links: list[str] = Field(default_factory=list)     # news → stock transmission, plain


class Recommendation(BaseModel):
    """An actionable, capital-sized call, logged for later win-rate evaluation."""

    run_date: Date
    ticker: str
    action: Action = Action.HOLD
    shares: float = 0.0          # shares to trade (always >= 0; direction is `action`)
    amount: float = 0.0          # cash value of the trade, in account currency
    target_pct: float = 0.0      # intended position size as a fraction of capital
    price_at_rec: float | None = None
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    lean: Lean = Lean.NEUTRAL
    score: float = 0.0
    rationale: str = ""
    notes: list[str] = Field(default_factory=list)


class DailyReport(BaseModel):
    run_date: Date
    universe_size: int
    macro_summary: str = ""
    macro_signals: list[MacroSignal] = Field(default_factory=list)
    analyses: list[TickerAnalysis] = Field(default_factory=list)
    snapshots: list[TickerSnapshot] = Field(default_factory=list)  # inputs, for training export
    errors: list[str] = Field(default_factory=list)
