"""Store round-trip tests against a temp DB (config db_path is monkeypatched)."""

from datetime import date

import pytest

from stockresearch import config
from stockresearch.models import (
    DailyReport, Lean, MacroSignal, QuantMetrics, TickerAnalysis,
)


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = config.load_settings()
    monkeypatch.setattr(s.runtime, "db_path", str(tmp_path / "test.db"))
    config.load_settings.cache_clear()
    monkeypatch.setattr(config, "load_settings", lambda: s)
    from stockresearch.store import db
    import importlib
    importlib.reload(db)
    return db


def _report():
    return DailyReport(
        run_date=date(2026, 5, 26),
        universe_size=2,
        macro_summary="calm",
        macro_signals=[MacroSignal(headline="Fed holds", summary="x", impact=Lean.NEUTRAL)],
        analyses=[
            TickerAnalysis(ticker="AAPL", name="Apple", lean=Lean.BULLISH, score=0.8,
                           metrics=QuantMetrics(last_price=200, beta=1.2, sharpe=1.5)),
            TickerAnalysis(ticker="XYZ", lean=Lean.BEARISH, score=0.1),
        ],
    )


def test_save_and_load_roundtrip(store):
    store.save_report(_report())
    loaded = store.load_report(date(2026, 5, 26))
    assert loaded is not None
    assert loaded.universe_size == 2
    assert loaded.analyses[0].ticker == "AAPL"


def test_top_picks_ordered_by_score(store):
    store.save_report(_report())
    picks = store.top_picks(date(2026, 5, 26))
    assert [p["ticker"] for p in picks] == ["AAPL", "XYZ"]


def test_save_is_idempotent(store):
    store.save_report(_report())
    store.save_report(_report())
    assert len(store.list_runs()) == 1


def test_ticker_history(store):
    store.save_report(_report())
    hist = store.ticker_history("AAPL")
    assert len(hist) == 1 and hist[0]["lean"] == "bullish"
