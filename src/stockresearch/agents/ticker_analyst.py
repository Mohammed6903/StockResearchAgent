"""Per-ticker analyst worker with a reflection pass.

Pipeline per ticker:
  1. Assemble a factual snapshot (fundamentals + locally-computed quant + grounded news).
  2. Analyst reasons -> AnalystVerdict (lean/score/pros/cons/reasoning).
  3. Critic reflects: checks the verdict's claims against the snapshot facts, revises once.
The deterministic facts (metrics, identity) are always taken from the snapshot, never
from the model, then merged into the final TickerAnalysis.
"""

from __future__ import annotations

import json

import pandas as pd

from ..config import load_settings
from ..data.news_adapter import get_company_news
from ..data.verify import verify_fundamentals
from ..data.yfinance_adapter import get_closes, get_fundamentals
from ..models import MacroSignal, QuantMetrics, TickerAnalysis, TickerSnapshot
from ..quant.metrics import compute_metrics
from .reason import reason
from .schemas import AnalystVerdict, CritiqueResult


def build_snapshot(
    ticker: str, benchmark_closes: pd.Series, *, fetch_news: bool = True, verify: bool = False
) -> TickerSnapshot:
    q = load_settings().quant
    warnings: list[str] = []

    fundamentals = get_fundamentals(ticker)
    if fundamentals.name is None:
        warnings.append("fundamentals unavailable")
    elif verify:
        warnings.extend(verify_fundamentals(ticker, fundamentals))

    closes = get_closes(ticker, q.lookback_days)
    if closes.empty:
        warnings.append("price history unavailable")
        metrics = QuantMetrics()
    else:
        metrics = compute_metrics(closes, benchmark_closes, q.risk_free_rate)
        if metrics.beta is None:
            warnings.append("insufficient overlap for beta/alpha")

    news_text, news_items = ("", [])
    if fetch_news:
        news_text, news_items = get_company_news(ticker, fundamentals.name)

    return TickerSnapshot(
        ticker=ticker, fundamentals=fundamentals, metrics=metrics,
        news=news_items, news_text=news_text, data_warnings=warnings,
    )


def _facts_block(snap: TickerSnapshot, macro_signals: list[MacroSignal]) -> str:
    fund = {k: v for k, v in snap.fundamentals.model_dump().items() if v is not None}
    met = {k: v for k, v in snap.metrics.model_dump().items() if v is not None}
    news_text = snap.news_text
    relevant_macro = [
        s.model_dump(include={"headline", "impact", "sectors", "summary"})
        for s in macro_signals
        if (snap.fundamentals.sector in s.sectors)
        or (snap.ticker in s.affected_tickers)
        or not s.sectors
    ][:6]
    return (
        f"TICKER: {snap.ticker} ({snap.fundamentals.name or 'unknown'})\n"
        f"SECTOR: {snap.fundamentals.sector or 'unknown'}\n"
        f"FUNDAMENTALS: {json.dumps(fund, default=str)}\n"
        f"QUANT METRICS (computed deterministically): {json.dumps(met, default=str)}\n"
        f"RECENT NEWS: {news_text or 'none found'}\n"
        f"RELEVANT MACRO SIGNALS: {json.dumps(relevant_macro, default=str)}\n"
        f"DATA WARNINGS: {snap.data_warnings or 'none'}\n"
    )


def _analyze(facts: str) -> AnalystVerdict:
    prompt = (
        "You are an equity research analyst. Based ONLY on the facts below, produce an "
        "analytical verdict. Weigh valuation (P/E, P/B, PEG), profitability/margins, "
        "leverage (debt/equity, current ratio), growth, risk-adjusted performance "
        "(Sharpe, beta, alpha, volatility, drawdown), recent news and relevant macro. "
        "Cite specific numbers in your reasoning. Be balanced: list real pros AND cons. "
        "If key data is missing, lower your confidence. This is research, not advice.\n\n"
        f"{facts}"
    )
    return reason(prompt, AnalystVerdict, temperature=0.3)


def _reflect(facts: str, verdict: AnalystVerdict) -> AnalystVerdict:
    prompt = (
        "You are a skeptical reviewer checking another analyst's verdict for rigor. "
        "Verify every claim is supported by the facts. Flag: numbers contradicted by the "
        "data, pros/cons not grounded in facts, lean/score inconsistent with the evidence, "
        "or overconfidence given data warnings. Return a corrected verdict (revised). "
        "If the verdict is already sound, set needs_revision=false and echo it in revised.\n\n"
        f"FACTS:\n{facts}\n\nVERDICT UNDER REVIEW:\n{verdict.model_dump_json()}\n"
    )
    critique = reason(prompt, CritiqueResult, temperature=0.1)
    return critique.revised


def analyze_ticker(
    ticker: str, benchmark_closes: pd.Series, macro_signals: list[MacroSignal],
    *, reflect: bool = True, fetch_news: bool = True, verify: bool = False,
) -> TickerAnalysis:
    snap = build_snapshot(ticker, benchmark_closes, fetch_news=fetch_news, verify=verify)
    facts = _facts_block(snap, macro_signals)
    verdict = _analyze(facts)
    if reflect:
        verdict = _reflect(facts, verdict)
    return TickerAnalysis(
        ticker=snap.ticker,
        name=snap.fundamentals.name,
        sector=snap.fundamentals.sector,
        lean=verdict.lean,
        confidence=verdict.confidence,
        score=verdict.score,
        pros=verdict.pros,
        cons=verdict.cons,
        reasoning=verdict.reasoning,
        metrics=snap.metrics,
        sources=snap.news,
        data_warnings=snap.data_warnings,
    )
