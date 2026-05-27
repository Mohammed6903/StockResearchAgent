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
from ..data.yfinance_adapter import get_closes, get_fundamentals, get_statements
from ..models import MacroSignal, QuantMetrics, TickerAnalysis, TickerSnapshot
from ..quant.metrics import compute_metrics
from ..quant.score import ScoreResult, composite_score, lean_from_score
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

    statements = get_statements(ticker)

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
        ticker=ticker, fundamentals=fundamentals, statements=statements, metrics=metrics,
        news=news_items, news_text=news_text, data_warnings=warnings,
    )


def _facts_block(snap: TickerSnapshot, macro_signals: list[MacroSignal]) -> str:
    fund = {k: v for k, v in snap.fundamentals.model_dump().items() if v is not None}
    stmts = {k: v for k, v in snap.statements.model_dump().items() if v is not None}
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
        f"STATEMENT LINE ITEMS (latest filing): {json.dumps(stmts, default=str)}\n"
        f"QUANT METRICS (computed deterministically): {json.dumps(met, default=str)}\n"
        f"RECENT NEWS: {news_text or 'none found'}\n"
        f"RELEVANT MACRO SIGNALS: {json.dumps(relevant_macro, default=str)}\n"
        f"DATA WARNINGS: {snap.data_warnings or 'none'}\n"
    )


_NUDGE_BAND = 0.10  # max the LLM may move the deterministic composite score


def _analyze(facts: str, sr: ScoreResult) -> AnalystVerdict:
    today = date.today().isoformat()
    lo, hi = round(sr.score - _NUDGE_BAND, 3), round(sr.score + _NUDGE_BAND, 3)
    prompt = (
        f"Today's date is {today}. You are an equity research analyst. A QUANTITATIVE composite "
        f"score has already been computed from the live metrics:\n"
        f"  composite score = {sr.score:.2f} (0 = weak, 1 = strong)\n"
        f"  factor scores (0-1) = {sr.factors}\n\n"
        "Your job is to EXPLAIN this score and write balanced pros/cons grounded in the facts "
        "below — NOT to invent your own rating. Set `score` to the composite, which you may "
        f"adjust by AT MOST ±{_NUDGE_BAND:.2f} (i.e. within [{lo}, {hi}]) ONLY if genuinely "
        "recent, material news justifies it; otherwise echo the composite exactly. Do not "
        "output a score outside that band. (`lean`/`confidence` you return are ignored — the "
        "system derives them from the score.)\n"
        "RECENCY RULES: fundamentals/metrics are live as of today; treat RECENT NEWS as current "
        "only if dated within the last ~1-2 months — ignore prior-year events and never call "
        "them recent; do not use training-data recollections; if news conflicts with the live "
        "fundamentals, prefer the fundamentals and note it. Cite specific numbers.\n\n"
        f"{facts}"
    )
    return reason(prompt, AnalystVerdict, temperature=0.3)


def _reflect(facts: str, verdict: AnalystVerdict) -> AnalystVerdict:
    today = date.today().isoformat()
    prompt = (
        f"Today's date is {today}. You are a skeptical reviewer checking another analyst's "
        "pros/cons/reasoning for rigor (the numeric score is set elsewhere — keep `score` as "
        "given). Flag and REMOVE:\n"
        "- any claim citing a specific past fiscal quarter, dated event, or figure that is "
        "NOT present in the live fundamentals or in news dated within the last ~1-2 months;\n"
        "- any prior-year event described as 'recent';\n"
        "- numbers contradicted by the live data or pros/cons not grounded in the facts.\n"
        "Return a corrected verdict (revised), preserving its `score`. If already sound, set "
        "needs_revision=false and echo it in revised.\n\n"
        f"FACTS:\n{facts}\n\nVERDICT UNDER REVIEW:\n{verdict.model_dump_json()}\n"
    )
    critique = reason(prompt, CritiqueResult, temperature=0.1)
    return critique.revised


def analyze_from_snapshot(
    snap: TickerSnapshot, macro_signals: list[MacroSignal], *, reflect: bool = True,
) -> TickerAnalysis:
    """The reasoning step, given an already-built snapshot. The score/lean/confidence come from
    the deterministic factor model; the LLM explains it and may nudge the score within a band."""
    sr = composite_score(snap.fundamentals, snap.metrics)
    facts = _facts_block(snap, macro_signals)
    verdict = _analyze(facts, sr)
    if reflect:
        verdict = _reflect(facts, verdict)

    if sr.factors:  # have quant data: clamp the LLM's score into the allowed band
        lo, hi = sr.score - _NUDGE_BAND, sr.score + _NUDGE_BAND
        final_score = max(0.0, min(1.0, max(lo, min(hi, verdict.score))))
    else:           # no quant data at all: stay neutral, low confidence
        final_score = 0.5

    return TickerAnalysis(
        ticker=snap.ticker,
        name=snap.fundamentals.name,
        sector=snap.fundamentals.sector,
        lean=lean_from_score(final_score),
        confidence=sr.confidence,
        score=round(final_score, 4),
        pros=verdict.pros,
        cons=verdict.cons,
        reasoning=verdict.reasoning,
        metrics=snap.metrics,
        factors=sr.factors,
        sources=snap.news,
        data_warnings=snap.data_warnings,
    )


def analyze_ticker(
    ticker: str, benchmark_closes: pd.Series, macro_signals: list[MacroSignal],
    *, reflect: bool = True, fetch_news: bool = True, verify: bool = False,
) -> TickerAnalysis:
    snap = build_snapshot(ticker, benchmark_closes, fetch_news=fetch_news, verify=verify)
    return analyze_from_snapshot(snap, macro_signals, reflect=reflect)
