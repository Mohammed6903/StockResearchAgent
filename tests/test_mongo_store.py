"""MongoStore round-trip tests using mongomock (no real MongoDB needed)."""

from datetime import date

import pytest

from stockresearch.models import (
    DailyReport, Lean, MacroSignal, QuantMetrics, TickerAnalysis, TickerSnapshot,
)

mongomock = pytest.importorskip("mongomock")


@pytest.fixture
def store():
    from stockresearch.store.mongo import MongoStore

    return MongoStore(client=mongomock.MongoClient(), dbname="test")


def _report():
    return DailyReport(
        run_date=date(2026, 5, 26),
        universe_size=2,
        macro_summary="cautious",
        macro_signals=[MacroSignal(headline="RBI holds", impact=Lean.NEUTRAL)],
        analyses=[
            TickerAnalysis(ticker="RELIANCE.NS", name="Reliance", lean=Lean.BULLISH,
                           score=0.8, confidence=0.7,
                           metrics=QuantMetrics(last_price=1300, beta=1.1, sharpe=1.2)),
            TickerAnalysis(ticker="TCS.NS", lean=Lean.BEARISH, score=0.2, confidence=0.6,
                           metrics=QuantMetrics(last_price=3000)),
        ],
        snapshots=[TickerSnapshot(ticker="RELIANCE.NS"), TickerSnapshot(ticker="TCS.NS")],
    )


def test_save_and_load_roundtrip(store):
    store.save_report(_report())
    loaded = store.load_report(date(2026, 5, 26))
    assert loaded is not None
    assert loaded.universe_size == 2
    assert loaded.analyses[0].ticker == "RELIANCE.NS"
    assert loaded.analyses[0].lean == Lean.BULLISH
    assert len(loaded.snapshots) == 2  # inputs preserved for training export


def test_save_is_idempotent(store):
    store.save_report(_report())
    store.save_report(_report())
    assert len(store.list_runs()) == 1


def test_ticker_history_and_all_calls(store):
    store.save_report(_report())
    hist = store.ticker_history("reliance.ns")
    assert len(hist) == 1 and hist[0]["lean"] == "bullish" and hist[0]["beta"] == 1.1
    calls = store.all_calls()
    assert {c["ticker"] for c in calls} == {"RELIANCE.NS", "TCS.NS"}


def test_top_picks_ordered(store):
    store.save_report(_report())
    assert [p["ticker"] for p in store.top_picks(date(2026, 5, 26))] == ["RELIANCE.NS", "TCS.NS"]


def test_outcomes_roundtrip(store):
    store.upsert_outcome("2026-05-26", "RELIANCE.NS", 30, "bullish", 0.7, 1300, 1430, 0.10, True)
    store.upsert_outcome("2026-05-26", "RELIANCE.NS", 30, "bullish", 0.7, 1300, 1450, 0.115, True)
    rows = store.outcomes_rows()
    assert len(rows) == 1  # upsert replaced
    assert rows[0]["correct"] == 1 and abs(rows[0]["forward_return"] - 0.115) < 1e-9


def test_get_store_falls_back_to_sqlite_without_uri(monkeypatch):
    import stockresearch.store as store_pkg

    monkeypatch.delenv("MONGODB_URI", raising=False)
    store_pkg.get_store.cache_clear()
    assert store_pkg.active_backend_name() == "sqlite"
