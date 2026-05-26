"""Macro/political analyst: scans grounded news, then structures market-moving signals.

Two steps because Gemini grounding and response_schema can't be combined in one call:
(1) grounded search produces attributable headline text + real source URLs, (2) a
structured pass distills it into typed signals. To keep signals auditable, the structured
pass attributes each signal to sources by *id* (indexing the numbered source list); we then
resolve those ids back to the real URLs deterministically, so the model never invents a URL.
Signals the model fails to attribute are flagged as unsourced.
"""

from __future__ import annotations

from urllib.parse import urlparse

from ..data.news_adapter import get_macro_news
from ..models import MacroSignal, NewsItem
from .reason import reason
from .schemas import MacroDraftList, MacroSignalDraft

# A signal must be backed by at least this many distinct publisher domains to count as
# corroborated. Uncorroborated signals keep their evidence but have confidence capped, so a
# single-source claim can't present as high-confidence.
_MIN_DOMAINS_FOR_CORROBORATION = 2
_UNCORROBORATED_MAX_CONFIDENCE = 0.5


def _resolve_sources(draft: MacroSignalDraft, sources: list[NewsItem]) -> list[NewsItem]:
    resolved: list[NewsItem] = []
    for i in draft.source_ids:
        if 0 <= i < len(sources):
            resolved.append(sources[i])
    return resolved


def _domain(item: NewsItem) -> str:
    """Best-effort publisher domain. Grounding URLs are opaque Vertex redirects, so the
    publisher domain lives in the title; fall back to the URL host only if needed."""
    if item.title and "." in item.title and " " not in item.title.strip():
        return item.title.strip().lower()
    if item.url:
        host = urlparse(item.url).netloc.lower()
        if host and "vertexaisearch" not in host:
            return host
    return (item.title or item.url or "").strip().lower()


def _distinct_domains(sources: list[NewsItem]) -> set[str]:
    return {d for d in (_domain(s) for s in sources) if d}


def run_macro_scan() -> tuple[str, list[MacroSignal]]:
    grounded_text, sources = get_macro_news()
    if not grounded_text:
        return "", []

    numbered = "\n".join(f"[{i}] {s.title} — {s.url}" for i, s in enumerate(sources))
    prompt = (
        "You are a macro/political analyst for an equity research desk. Below is grounded "
        "news from a web search, plus a numbered list of the sources it came from. Extract "
        "ONLY items likely to move stock markets or change the outlook for specific "
        "sectors/companies (monetary policy, regulation, geopolitics, elections, trade, "
        "commodities, major fiscal moves). Ignore everything non-market. For each signal: "
        "set impact (bullish/bearish/neutral for the affected area), tag sectors/regions, "
        "list affected tickers only if clearly implied, and set source_ids to the ids of "
        "the sources that support it. Only cite ids that actually support the claim.\n\n"
        f"GROUNDED NEWS:\n{grounded_text}\n\nSOURCES:\n{numbered}\n"
    )
    draft = reason(prompt, MacroDraftList, temperature=0.2)

    signals: list[MacroSignal] = []
    for d in draft.signals:
        resolved = _resolve_sources(d, sources)
        corroborated = len(_distinct_domains(resolved)) >= _MIN_DOMAINS_FOR_CORROBORATION
        confidence = d.confidence
        if not corroborated:
            confidence = min(confidence, _UNCORROBORATED_MAX_CONFIDENCE)
        signals.append(
            MacroSignal(
                headline=d.headline, summary=d.summary, regions=d.regions,
                sectors=d.sectors, affected_tickers=d.affected_tickers,
                impact=d.impact, confidence=confidence, sources=resolved,
                corroborated=corroborated,
            )
        )
    return draft.summary, signals
