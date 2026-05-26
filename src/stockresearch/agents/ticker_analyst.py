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
from datetime import date

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
    today = date.today().isoformat()
    prompt = (
        f"Today's date is {today}. You are an equity research analyst. Produce an analytical "
        "verdict using ONLY the facts below.\n"
        "CRITICAL RULES ON RECENCY:\n"
        "- The FUNDAMENTALS and QUANT METRICS are LIVE as of today; treat them as current.\n"
        "- RECENT NEWS items may be mis-dated by the search. Treat as current ONLY items "
        "clearly dated within the last ~1-2 months. IGNORE anything older (e.g. a prior-year "
        "'Q4 FY24' result) and NEVER describe a prior-year event as recent.\n"
        "- Do NOT use any prior/training knowledge of specific past events, earnings quarters, "
        "capital raises, litigation, or figures that are not present below.\n"
        "- Do NOT attribute a live fundamental to a named fiscal quarter unless that period is "
        "explicitly and recently dated in the news.\n"
        "- If RECENT NEWS conflicts with the live fundamentals (e.g. news says profit fell but "
        "earnings_growth is positive), PREFER the live fundamentals and note the conflict.\n\n"
        "Weigh valuation (P/E, P/B, PEG), profitability/margins, leverage (debt/equity, current "
        "ratio), growth, risk-adjusted performance (Sharpe, beta, alpha, volatility, drawdown), "
        "and genuinely recent news/macro. Cite specific numbers. Be balanced: real pros AND "
        "cons. If key data is missing or news isn't recent, lower confidence. Research, not "
        f"advice.\n\n{facts}"
    )
    return reason(prompt, AnalystVerdict, temperature=0.3)


def _reflect(facts: str, verdict: AnalystVerdict) -> AnalystVerdict:
    today = date.today().isoformat()
    prompt = (
        f"Today's date is {today}. You are a skeptical reviewer checking another analyst's "
        "verdict for rigor. Verify every claim is supported by the facts. Flag and REMOVE:\n"
        "- any claim citing a specific past fiscal quarter, dated event, or figure that is "
        "NOT present in the live fundamentals or in news dated within the last ~1-2 months "
        "(these are stale model recollections or mis-dated search hits);\n"
        "- any prior-year event described as 'recent';\n"
        "- numbers contradicted by the live data, pros/cons not grounded in the facts, "
        "lean/score inconsistent with the evidence, or overconfidence given data warnings.\n"
        "Return a corrected verdict (revised). If already sound, set needs_revision=false and "
        "echo it in revised.\n\n"
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
