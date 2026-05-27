"""Advisor tests: action + sizing are deterministic from the analysis score (no LLM)."""

from datetime import date

import pytest

from stockresearch.agents import advisor
from stockresearch.models import Action, QuantMetrics, TickerAnalysis, TickerSnapshot
from stockresearch.quant.score import lean_from_score


def _snap(price):
    return TickerSnapshot(ticker="AAA", metrics=QuantMetrics(last_price=price))


def _analysis(score, factors=True):
    return TickerAnalysis(
        ticker="AAA", score=score, lean=lean_from_score(score), confidence=0.7,
        factors={"value": 0.5, "quality": 0.5} if factors else {},
    )


def test_strong_score_buys_sized_by_conviction():
    rec = advisor.advise(_snap(100), _analysis(0.8), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.BUY
    # frac=(0.8-0.6)/0.4=0.5 -> target 0.10*(0.4+0.3)=0.07 -> 7000/100 = 70 shares
    assert rec.shares == 70 and rec.amount == 7000


def test_higher_score_buys_more():
    low = advisor.advise(_snap(100), _analysis(0.62), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    high = advisor.advise(_snap(100), _analysis(0.95), capital=100_000,
                          max_position_pct=0.10, held_qty=0)
    assert high.shares > low.shares  # conviction scales size


def test_buy_when_already_at_target_holds():
    rec = advisor.advise(_snap(100), _analysis(0.8), capital=100_000,
                         max_position_pct=0.10, held_qty=70)
    assert rec.action == Action.HOLD and rec.shares == 0


def test_weak_score_full_exit_when_held():
    rec = advisor.advise(_snap(100), _analysis(0.2), capital=100_000,
                         max_position_pct=0.10, held_qty=50)
    assert rec.action == Action.SELL and rec.shares == 50


def test_weak_score_no_holding_holds_avoid():
    rec = advisor.advise(_snap(100), _analysis(0.2), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.HOLD
    assert any("avoid" in n for n in rec.notes)


def test_middling_score_holds():
    rec = advisor.advise(_snap(100), _analysis(0.50), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.HOLD


def test_no_capital_abstains():
    rec = advisor.advise(_snap(100), _analysis(0.8), capital=0,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.HOLD and any("capital" in n for n in rec.notes)


def test_missing_price_abstains():
    rec = advisor.advise(_snap(None), _analysis(0.8), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.HOLD and any("price" in n for n in rec.notes)


def test_no_factors_abstains():
    rec = advisor.advise(_snap(100), _analysis(0.8, factors=False), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.HOLD and any("quant data" in n for n in rec.notes)


@pytest.fixture
def env(tmp_path, monkeypatch):
    from stockresearch import config
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


def test_winrate_scoring(env, monkeypatch):
    from stockresearch.models import Recommendation

    db, evaluate = env
    db.save_recommendation(Recommendation(
        run_date=date(2026, 1, 1), ticker="UP", action=Action.BUY, shares=10, amount=1000,
        target_pct=0.1, price_at_rec=100.0, confidence=0.8))
    db.save_recommendation(Recommendation(
        run_date=date(2026, 1, 1), ticker="DN", action=Action.SELL, shares=10, amount=2000,
        target_pct=0.0, price_at_rec=200.0, confidence=0.6))

    prices = {"UP": 120.0, "DN": 150.0}  # UP +20% (buy win), DN -25% (sell win)
    monkeypatch.setattr(evaluate, "price_on_or_after", lambda t, d: prices[t])

    scored = evaluate.score_recommendations(today=date(2026, 6, 1))
    assert scored == 2 * len(evaluate.HORIZONS)
    w = evaluate.winrate()
    assert w["win_rate"] == 1.0
