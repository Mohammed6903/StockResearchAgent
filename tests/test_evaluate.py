"""Tests for outcome scoring / calibration / export against a temp DB (no network)."""

import json
from datetime import date

import pytest

from stockresearch import config
from stockresearch.models import DailyReport, Lean, QuantMetrics, TickerAnalysis, TickerSnapshot


@pytest.fixture
def env(tmp_path, monkeypatch):
    s = config.load_settings()
    monkeypatch.setattr(s.runtime, "db_path", str(tmp_path / "t.db"))
    config.load_settings.cache_clear()
    monkeypatch.setattr(config, "load_settings", lambda: s)
    import importlib

    from stockresearch import evaluate
    from stockresearch.store import db
    importlib.reload(db)
    importlib.reload(evaluate)
    return db, evaluate


def _seed(db):
    report = DailyReport(
        run_date=date(2026, 1, 1),
        universe_size=2,
        analyses=[
            TickerAnalysis(ticker="UP", lean=Lean.BULLISH, confidence=0.8, score=0.9,
                           metrics=QuantMetrics(last_price=100.0)),
            TickerAnalysis(ticker="DN", lean=Lean.BEARISH, confidence=0.6, score=0.2,
                           metrics=QuantMetrics(last_price=200.0)),
        ],
        snapshots=[TickerSnapshot(ticker="UP"), TickerSnapshot(ticker="DN")],
    )
    db.save_report(report)


def test_is_correct_logic(env):
    _, evaluate = env
    assert evaluate._is_correct("bullish", 0.10) is True
    assert evaluate._is_correct("bullish", 0.01) is False
    assert evaluate._is_correct("bearish", -0.10) is True
    assert evaluate._is_correct("neutral", 0.00) is True
    assert evaluate._is_correct("neutral", 0.20) is False


def test_score_and_calibration(env, monkeypatch):
    db, evaluate = env
    _seed(db)
    # UP rose 20%, DN fell 25% -> both leans correct.
    prices = {"UP": 120.0, "DN": 150.0}
    monkeypatch.setattr(evaluate, "price_on_or_after", lambda t, d: prices[t])

    n = evaluate.score_matured(today=date(2026, 6, 1))  # well past 90d
    assert n == 2 * len(evaluate.HORIZONS)

    c = evaluate.calibration()
    assert c["overall_accuracy"] == 1.0
    assert c["by_lean"]["bullish"]["correct"] == c["by_lean"]["bullish"]["n"]


def test_not_matured_scores_nothing(env, monkeypatch):
    db, evaluate = env
    _seed(db)
    monkeypatch.setattr(evaluate, "price_on_or_after", lambda t, d: 999.0)
    # Only 3 days after the call: no horizon (min 7d) matured.
    assert evaluate.score_matured(today=date(2026, 1, 4)) == 0


def test_export_jsonl(env, monkeypatch, tmp_path):
    db, evaluate = env
    _seed(db)
    prices = {"UP": 120.0, "DN": 150.0}
    monkeypatch.setattr(evaluate, "price_on_or_after", lambda t, d: prices[t])
    evaluate.score_matured(today=date(2026, 6, 1))

    out = tmp_path / "train.jsonl"
    rows = evaluate.export_jsonl(str(out))
    assert rows == 2
    lines = [json.loads(line) for line in out.read_text().splitlines()]
    up = next(r for r in lines if r["ticker"] == "UP")
    assert up["verdict"]["lean"] == "bullish"
    assert up["inputs"]["ticker"] == "UP"
    assert "return_30d" in up["outcomes"]
