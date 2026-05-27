"""Agent pipeline tests with mocked LLM + data sources (no network)."""

import pandas as pd
import pytest

from stockresearch.agents import orchestrator, ticker_analyst
from stockresearch.agents.schemas import AnalystVerdict
from stockresearch.models import Fundamentals, Lean, QuantMetrics
from stockresearch.quant.score import lean_from_score


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


def test_metrics_and_identity_come_from_snapshot_not_model(fake_data, monkeypatch):
    # The LLM provides narrative only; identity/metrics/score come from the deterministic side.
    verdict = AnalystVerdict(lean=Lean.BULLISH, confidence=0.7, score=0.6,
                             pros=["cheap"], cons=["slow"], reasoning="ok")
    monkeypatch.setattr(ticker_analyst, "_analyze", lambda facts, sr: verdict)

    res = ticker_analyst.analyze_ticker("AAPL", pd.Series(dtype=float), [], reflect=False)
    assert res.ticker == "AAPL"
    assert res.name == "AAPL Inc"
    assert res.sector == "Technology"
    assert res.metrics.last_price == 159.0  # computed from fake closes, not the LLM
    assert res.factors  # deterministic factor breakdown present
    # lean is derived from the (deterministic) score, not whatever the LLM claimed
    assert res.lean == lean_from_score(res.score)


def test_llm_score_nudge_is_clamped_to_band(fake_data, monkeypatch):
    from stockresearch.quant.score import composite_score

    # The LLM tries to force the score to 1.0; it must be clamped to composite + 0.10.
    sr = composite_score(
        ticker_analyst.get_fundamentals("AAPL"),
        ticker_analyst.compute_metrics(ticker_analyst.get_closes("AAPL", 365),
                                       pd.Series(dtype=float), 0.045),
    )
    greedy = AnalystVerdict(score=1.0, reasoning="moon")
    monkeypatch.setattr(ticker_analyst, "_analyze", lambda facts, s: greedy)

    res = ticker_analyst.analyze_ticker("AAPL", pd.Series(dtype=float), [], reflect=False)
    assert abs(res.score - round(min(1.0, sr.score + 0.10), 4)) < 1e-6


def test_orchestrator_isolates_bad_ticker(monkeypatch):
    from stockresearch.models import TickerSnapshot

    monkeypatch.setattr(orchestrator, "run_macro_scan", lambda: ("", []))
    monkeypatch.setattr(orchestrator, "get_closes", lambda t, d: pd.Series(dtype=float))

    def fake_build(t, bench, *, fetch_news=True, verify=False):
        if t == "BOOM":
            raise RuntimeError("data feed exploded")
        return TickerSnapshot(ticker=t)

    def fake_analyze(snap, macro, *, reflect=True):
        return ticker_analyst.TickerAnalysis(ticker=snap.ticker, score=0.5, metrics=QuantMetrics())

    monkeypatch.setattr(orchestrator, "build_snapshot", fake_build)
    monkeypatch.setattr(orchestrator, "analyze_from_snapshot", fake_analyze)

    report = orchestrator.run_daily(["GOOD1", "BOOM", "GOOD2"], skip_macro=True)
    tickers = {a.ticker for a in report.analyses}
    assert tickers == {"GOOD1", "GOOD2"}
    assert any("BOOM" in e for e in report.errors)
    assert report.universe_size == 3
