"""Advisor sizing tests (mocked LLM) + win-rate scoring (synthetic prices)."""

from datetime import date

import pytest

from stockresearch.agents import advisor
from stockresearch.agents.schemas import AdviceVerdict
from stockresearch.models import Action, Lean, QuantMetrics, TickerAnalysis, TickerSnapshot


def _snap(price):
    return TickerSnapshot(ticker="AAA", metrics=QuantMetrics(last_price=price))


def _analysis():
    return TickerAnalysis(ticker="AAA", lean=Lean.BULLISH, score=0.8, confidence=0.7)


def _mock_verdict(monkeypatch, action, target_pct, conf=0.7):
    monkeypatch.setattr(
        advisor, "_verdict",
        lambda *a, **k: AdviceVerdict(action=action, target_pct_of_capital=target_pct,
                                      confidence=conf, rationale="r"),
    )


def test_buy_sizes_against_capital(monkeypatch):
    _mock_verdict(monkeypatch, Action.BUY, 0.10)
    rec = advisor.advise(_snap(100), _analysis(), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.BUY
    assert rec.shares == 100      # 10% of 100k = 10k / 100 = 100 shares
    assert rec.amount == 100 * 100
    assert rec.target_pct == 0.10


def test_target_pct_clamped_to_cap(monkeypatch):
    _mock_verdict(monkeypatch, Action.BUY, 0.50)  # asks for 50%, cap is 10%
    rec = advisor.advise(_snap(100), _analysis(), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.target_pct == 0.10 and rec.shares == 100
    assert any("cap" in n for n in rec.notes)


def test_buy_when_already_at_target_becomes_hold(monkeypatch):
    _mock_verdict(monkeypatch, Action.BUY, 0.10)
    # already hold 100 shares = target -> nothing to buy
    rec = advisor.advise(_snap(100), _analysis(), capital=100_000,
                         max_position_pct=0.10, held_qty=100)
    assert rec.action == Action.HOLD and rec.shares == 0


def test_sell_with_no_holding_becomes_hold(monkeypatch):
    _mock_verdict(monkeypatch, Action.SELL, 0.0)
    rec = advisor.advise(_snap(100), _analysis(), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.HOLD
    assert any("no holding" in n for n in rec.notes)


def test_sell_full_exit(monkeypatch):
    _mock_verdict(monkeypatch, Action.SELL, 0.0)
    rec = advisor.advise(_snap(100), _analysis(), capital=100_000,
                         max_position_pct=0.10, held_qty=50)
    assert rec.action == Action.SELL and rec.shares == 50


def test_no_capital_abstains(monkeypatch):
    _mock_verdict(monkeypatch, Action.BUY, 0.10)
    rec = advisor.advise(_snap(100), _analysis(), capital=0,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.HOLD and rec.shares == 0
    assert any("capital" in n for n in rec.notes)


def test_missing_price_abstains(monkeypatch):
    _mock_verdict(monkeypatch, Action.BUY, 0.10)
    rec = advisor.advise(_snap(None), _analysis(), capital=100_000,
                         max_position_pct=0.10, held_qty=0)
    assert rec.action == Action.HOLD
    assert any("price" in n for n in rec.notes)


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
    assert w["by_action"]["buy"]["wins"] == w["by_action"]["buy"]["n"]
