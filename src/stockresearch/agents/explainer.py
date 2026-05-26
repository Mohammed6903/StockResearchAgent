"""Beginner explainer: turns one ticker's analysis into a teaching layer.

Two parts:
  - Deterministic metric lines: each metric's authoritative glossary definition + this
    stock's real value + a plain "good/ok/concern" read + (where possible) the derivation
    from the actual statement line items. No LLM, so definitions/formulas can't drift.
  - One LLM pass that narrates, for a beginner, how the numbers and *recent* news led to the
    verdict, plus the news → stock transmission. Narration only — it never restates formulas.
"""

from __future__ import annotations

from .. import glossary
from ..models import Explanation, Statements, TickerAnalysis, TickerSnapshot
from .reason import reason
from .schemas import TeachingNote

# Coarse thresholds purely for a beginner "read"; deliberately simple, not advice.
_GOOD = "✓ generally healthy"
_OK = "→ middling / context-dependent"
_CONCERN = "⚠ worth scrutiny"


def _read(term_key: str, value: float, higher_is_better: bool | None) -> str:
    if higher_is_better is None:
        return _OK
    bands = {
        "current_ratio": (1.0, 1.5), "quick_ratio": (0.5, 1.0),
        "debt_to_equity": (1.0, 2.0), "roe": (0.12, 0.20), "roa": (0.05, 0.10),
        "gross_margin": (0.20, 0.40), "operating_margin": (0.10, 0.20),
        "profit_margin": (0.08, 0.15), "sharpe": (0.0, 1.0),
        "pe_ratio": (15, 30), "forward_pe": (15, 30), "peg_ratio": (1.0, 2.0),
    }
    lo, hi = bands.get(term_key, (0.0, 0.0))
    if higher_is_better:
        return _GOOD if value >= hi else (_OK if value >= lo else _CONCERN)
    return _GOOD if value <= lo else (_OK if value <= hi else _CONCERN)


def _derivation(term_key: str, s: Statements) -> str:
    """Show the ratio computed from the real statement lines, when both are present."""
    def have(*xs):
        return all(x is not None for x in xs)

    if term_key == "current_ratio" and have(s.current_assets, s.current_liabilities):
        return f" (= current assets {s.current_assets:,.0f} ÷ current liabilities {s.current_liabilities:,.0f})"
    if term_key == "quick_ratio" and have(s.current_assets, s.inventory, s.current_liabilities):
        return f" (= (current assets {s.current_assets:,.0f} − inventory {s.inventory:,.0f}) ÷ {s.current_liabilities:,.0f})"
    if term_key == "debt_to_equity" and have(s.total_debt, s.total_equity):
        return f" (= total debt {s.total_debt:,.0f} ÷ equity {s.total_equity:,.0f})"
    if term_key == "roe" and have(s.net_income, s.total_equity):
        return f" (= net income {s.net_income:,.0f} ÷ equity {s.total_equity:,.0f})"
    if term_key == "roa" and have(s.net_income, s.total_assets):
        return f" (= net income {s.net_income:,.0f} ÷ total assets {s.total_assets:,.0f})"
    if term_key == "profit_margin" and have(s.net_income, s.revenue):
        return f" (= net income {s.net_income:,.0f} ÷ revenue {s.revenue:,.0f})"
    if term_key == "free_cashflow" and have(s.operating_cashflow, s.capex):
        return f" (= operating cash flow {s.operating_cashflow:,.0f} − capex {abs(s.capex):,.0f})"
    return ""


def _metric_lines(snap: TickerSnapshot) -> list[str]:
    values: dict[str, float | None] = {
        **snap.fundamentals.model_dump(),
        **snap.metrics.model_dump(),
    }
    lines: list[str] = []
    for term in glossary.all_terms():
        v = values.get(term.key)
        if v is None:
            continue
        is_pct = term.key in {
            "roe", "roa", "gross_margin", "operating_margin", "profit_margin",
            "annual_volatility", "alpha", "dividend_yield",
        }
        shown = f"{v * 100:.1f}%" if is_pct else f"{v:,.2f}"
        read = _read(term.key, v, term.higher_is_better)
        deriv = _derivation(term.key, snap.statements)
        lines.append(
            f"{term.name}: {shown} {read}{deriv}\n    {term.definition} "
            f"[{term.formula}]"
        )
    return lines


def _teaching(snap: TickerSnapshot, analysis: TickerAnalysis) -> TeachingNote:
    facts = {
        "ticker": snap.ticker,
        "name": snap.fundamentals.name,
        "verdict": analysis.lean.value,
        "confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "pros": analysis.pros,
        "cons": analysis.cons,
        "recent_news": snap.news_text,
    }
    prompt = (
        "You are a patient finance teacher explaining to a BEGINNER. Using only the facts "
        "below, write a plain-English walkthrough of how the financial metrics and the "
        "RECENT news led to this verdict — connect specific numbers to the conclusion in "
        "simple language (no jargon without a quick gloss). Do NOT restate formulas (those "
        "are shown separately). Then, for each material recent news item, add one bullet "
        "explaining the chain from the event to this company's actual drivers (e.g. "
        "'rates up → borrowing costlier → hurts because the firm is highly indebted'). "
        "If the news isn't genuinely recent, say so rather than inventing relevance.\n\n"
        f"FACTS:\n{facts}\n"
    )
    return reason(prompt, TeachingNote, temperature=0.3)


def explain_analysis(snap: TickerSnapshot, analysis: TickerAnalysis) -> Explanation:
    note = _teaching(snap, analysis)
    return Explanation(
        ticker=snap.ticker,
        metric_lines=_metric_lines(snap),
        walkthrough=note.walkthrough,
        news_links=note.news_links,
    )
