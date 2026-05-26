"""Agent pipeline tests with mocked LLM + data sources (no network)."""

import pandas as pd
import pytest

from stockresearch.agents import orchestrator, ticker_analyst
from stockresearch.agents.schemas import AnalystVerdict
from stockresearch.models import Fundamentals, Lean, QuantMetrics


@pytest.fixture
def fake_data(monkeypatch):
    monkeypatch.setattr(
        ticker_analyst, "get_fundamentals",
        lambda t: Fundamentals(name=f"{t} Inc", sector="Technology", pe_ratio=20.0),
    )
    monkeypatch.setattr(
        ticker_analyst, "get_closes",
        lambda t, d: pd.Series(
            [100 + i for i in range(60)],
            index=pd.date_range("2024-01-01", periods=60, freq="D"), dtype=float,
        ),
    )
    monkeypatch.setattr(
        ticker_analyst, "get_company_news", lambda t, n: ("calm news", []),
    )


def test_analyze_ticker_merges_facts_not_model(fake_data, monkeypatch):
    # LLM tries to claim a wrong lean; metrics/identity must come from snapshot.
    verdict = AnalystVerdict(lean=Lean.BULLISH, confidence=0.7, score=0.6,
                             pros=["cheap"], cons=["slow"], reasoning="ok")
    monkeypatch.setattr(ticker_analyst, "_analyze", lambda facts: verdict)

    res = ticker_analyst.analyze_ticker("AAPL", pd.Series(dtype=float), [], reflect=False)
    assert res.ticker == "AAPL"
    assert res.name == "AAPL Inc"
    assert res.sector == "Technology"
    assert res.lean == Lean.BULLISH
    assert res.metrics.last_price == 159.0  # computed from fake closes, not the LLM


def test_reflection_can_override_verdict(fake_data, monkeypatch):
    bad = AnalystVerdict(lean=Lean.BULLISH, confidence=0.9, score=0.9, reasoning="hype")
    good = AnalystVerdict(lean=Lean.NEUTRAL, confidence=0.4, score=0.4, reasoning="grounded")
    monkeypatch.setattr(ticker_analyst, "_analyze", lambda f: bad)
    monkeypatch.setattr(
        ticker_analyst, "_reflect", lambda f, v: good,
    )
    res = ticker_analyst.analyze_ticker("MSFT", pd.Series(dtype=float), [], reflect=True)
    assert res.lean == Lean.NEUTRAL and res.confidence == 0.4


def test_orchestrator_isolates_bad_ticker(monkeypatch):
    monkeypatch.setattr(orchestrator, "run_macro_scan", lambda: ("", []))
    monkeypatch.setattr(orchestrator, "get_closes", lambda t, d: pd.Series(dtype=float))

    def fake_analyze(t, bench, macro, reflect, fetch_news, verify):
        if t == "BOOM":
            raise RuntimeError("data feed exploded")
        return ticker_analyst.TickerAnalysis(ticker=t, score=0.5, metrics=QuantMetrics())

    monkeypatch.setattr(orchestrator, "analyze_ticker", fake_analyze)

    report = orchestrator.run_daily(["GOOD1", "BOOM", "GOOD2"], skip_macro=True)
    tickers = {a.ticker for a in report.analyses}
    assert tickers == {"GOOD1", "GOOD2"}
    assert any("BOOM" in e for e in report.errors)
    assert report.universe_size == 3
