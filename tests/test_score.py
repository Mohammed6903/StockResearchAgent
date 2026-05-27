"""Tests for the deterministic composite factor score — it must differentiate, not hedge."""

from stockresearch.models import Fundamentals, Lean, QuantMetrics
from stockresearch.quant.score import composite_score, lean_from_score


def _strong():
    # cheap, profitable, low debt, good momentum, growing
    return (
        Fundamentals(pe_ratio=12, forward_pe=10, peg_ratio=0.9, pb_ratio=2, ps_ratio=2,
                     roe=0.28, roa=0.14, operating_margin=0.30, profit_margin=0.18,
                     gross_margin=0.5, debt_to_equity=15, current_ratio=2.2,
                     revenue_growth=0.22, earnings_growth=0.30),
        QuantMetrics(return_1y=0.35, sharpe=1.6, alpha=0.18, max_drawdown=-0.08,
                     annual_volatility=0.18, last_price=100),
    )


def _weak():
    # expensive, unprofitable, heavily indebted, poor momentum, shrinking
    return (
        Fundamentals(pe_ratio=70, forward_pe=55, peg_ratio=4, pb_ratio=15, ps_ratio=14,
                     roe=0.01, roa=0.005, operating_margin=0.02, profit_margin=0.01,
                     gross_margin=0.05, debt_to_equity=180, current_ratio=0.4,
                     revenue_growth=-0.15, earnings_growth=-0.4),
        QuantMetrics(return_1y=-0.35, sharpe=-0.8, alpha=-0.2, max_drawdown=-0.6,
                     annual_volatility=0.7, last_price=100),
    )


def test_strong_scores_high_weak_scores_low():
    strong = composite_score(*_strong())
    weak = composite_score(*_weak())
    assert strong.score > 0.7
    assert weak.score < 0.3
    assert strong.score - weak.score > 0.4  # genuinely differentiated, not both ~0.5


def test_lean_thresholds():
    assert lean_from_score(0.75) == Lean.BULLISH
    assert lean_from_score(0.30) == Lean.BEARISH
    assert lean_from_score(0.50) == Lean.NEUTRAL


def test_confidence_rises_with_completeness():
    full = composite_score(*_strong())
    sparse = composite_score(Fundamentals(pe_ratio=12), QuantMetrics(last_price=100))
    assert full.confidence > sparse.confidence


def test_no_data_is_neutral_low_confidence():
    r = composite_score(Fundamentals(), QuantMetrics())
    assert r.score == 0.5 and r.factors == {} and r.confidence <= 0.2


def test_factor_breakdown_present():
    r = composite_score(*_strong())
    assert set(r.factors) == {"value", "quality", "momentum", "growth"}
    assert all(0.0 <= v <= 1.0 for v in r.factors.values())
