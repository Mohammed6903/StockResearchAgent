"""Tests for grounding-citation surfacing: macro source-id resolution + report rendering."""

from datetime import date

from stockresearch.agents import macro_agent
from stockresearch.agents.schemas import MacroDraftList, MacroSignalDraft
from stockresearch.models import (
    DailyReport, Lean, MacroSignal, NewsItem, TickerAnalysis,
)
from stockresearch.report import to_markdown


def test_macro_resolves_source_ids_to_real_urls(monkeypatch):
    sources = [
        NewsItem(title="Reuters: Fed holds", url="https://reuters.com/a"),
        NewsItem(title="WSJ: tariffs", url="https://wsj.com/b"),
    ]
    monkeypatch.setattr(macro_agent, "get_macro_news", lambda: ("grounded text", sources))
    draft = MacroDraftList(
        summary="cautious",
        signals=[
            MacroSignalDraft(headline="Fed steady", impact=Lean.NEUTRAL, source_ids=[0]),
            MacroSignalDraft(headline="Tariff risk", impact=Lean.BEARISH, source_ids=[1, 99]),
            MacroSignalDraft(headline="Rumor", impact=Lean.BULLISH, source_ids=[]),
        ],
    )
    monkeypatch.setattr(macro_agent, "reason", lambda *a, **k: draft)

    summary, signals = macro_agent.run_macro_scan()
    assert summary == "cautious"
    # id 0 -> reuters
    assert [s.url for s in signals[0].sources] == ["https://reuters.com/a"]
    # id 99 is out of range and dropped; only the valid one remains
    assert [s.url for s in signals[1].sources] == ["https://wsj.com/b"]
    # no ids -> unsourced
    assert signals[2].sources == []


def test_two_distinct_domains_is_corroborated(monkeypatch):
    sources = [
        NewsItem(title="reuters.com", url="https://v/a"),
        NewsItem(title="wsj.com", url="https://v/b"),
    ]
    monkeypatch.setattr(macro_agent, "get_macro_news", lambda: ("t", sources))
    draft = MacroDraftList(signals=[
        MacroSignalDraft(headline="H", impact=Lean.BEARISH, confidence=0.9, source_ids=[0, 1]),
    ])
    monkeypatch.setattr(macro_agent, "reason", lambda *a, **k: draft)

    _, signals = macro_agent.run_macro_scan()
    assert signals[0].corroborated is True
    assert signals[0].confidence == 0.9  # unchanged


def test_single_domain_caps_confidence(monkeypatch):
    # Two sources but same publisher domain -> not independent corroboration.
    sources = [
        NewsItem(title="reuters.com", url="https://v/a"),
        NewsItem(title="reuters.com", url="https://v/b"),
    ]
    monkeypatch.setattr(macro_agent, "get_macro_news", lambda: ("t", sources))
    draft = MacroDraftList(signals=[
        MacroSignalDraft(headline="H", impact=Lean.BEARISH, confidence=0.95, source_ids=[0, 1]),
    ])
    monkeypatch.setattr(macro_agent, "reason", lambda *a, **k: draft)

    _, signals = macro_agent.run_macro_scan()
    assert signals[0].corroborated is False
    assert signals[0].confidence == 0.5  # capped


def test_unsourced_signal_is_uncorroborated(monkeypatch):
    monkeypatch.setattr(macro_agent, "get_macro_news", lambda: ("t", []))
    draft = MacroDraftList(signals=[
        MacroSignalDraft(headline="H", impact=Lean.BULLISH, confidence=0.8, source_ids=[]),
    ])
    monkeypatch.setattr(macro_agent, "reason", lambda *a, **k: draft)
    # get_macro_news returns no sources -> early return path; force via non-empty text
    monkeypatch.setattr(macro_agent, "get_macro_news", lambda: ("nonempty", []))

    _, signals = macro_agent.run_macro_scan()
    assert signals[0].corroborated is False
    assert signals[0].confidence == 0.5


def test_report_renders_citations():
    report = DailyReport(
        run_date=date(2026, 5, 26),
        universe_size=1,
        macro_summary="m",
        macro_signals=[
            MacroSignal(headline="Fed steady", impact=Lean.NEUTRAL,
                        sources=[NewsItem(title="Reuters", url="https://reuters.com/a")]),
            MacroSignal(headline="Unsourced claim", impact=Lean.BULLISH),
        ],
        analyses=[
            TickerAnalysis(ticker="AAPL", lean=Lean.BULLISH, score=0.7,
                           sources=[NewsItem(title="9to5mac", url="https://9to5mac.com/x")]),
        ],
    )
    md = to_markdown(report)
    assert "[Reuters](https://reuters.com/a)" in md
    assert "_Sources: unattributed_" in md
    assert "[9to5mac](https://9to5mac.com/x)" in md
