"""Deterministic composite factor score — the decision signal.

The LLM tends to hedge every stock to "neutral / ~0.5 / 60%". So the score, lean, and
confidence are computed here from the real metrics instead: a 0–1 composite built from four
factor groups (value, quality, momentum/risk, growth). Each metric is mapped to 0–1 via an
anchored ramp (good→1, bad→0); factors average their available metrics; the composite is a
weighted blend of the factors that have data. This spreads scores out across stocks and is
fully reproducible. The LLM is only allowed to nudge the composite within a small band.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import Fundamentals, Lean, QuantMetrics

# Score thresholds for direction. Balanced: clear winners/losers get a call, middles HOLD.
BUY_THRESHOLD = 0.60
SELL_THRESHOLD = 0.40


def lean_from_score(score: float) -> Lean:
    if score >= BUY_THRESHOLD:
        return Lean.BULLISH
    if score <= SELL_THRESHOLD:
        return Lean.BEARISH
    return Lean.NEUTRAL

# (attribute, good_value, bad_value). good→1.0, bad→0.0, linear & clamped between.
# Lower-is-better metrics simply have good < bad.
_VALUE = [
    ("pe_ratio", 10.0, 60.0), ("forward_pe", 10.0, 45.0), ("peg_ratio", 0.8, 3.0),
    ("pb_ratio", 1.0, 12.0), ("ps_ratio", 1.0, 12.0),
]
_QUALITY = [
    ("roe", 0.25, 0.0), ("roa", 0.12, 0.0), ("operating_margin", 0.25, 0.0),
    ("profit_margin", 0.15, 0.0), ("gross_margin", 0.45, 0.05),
    ("debt_to_equity", 10.0, 150.0), ("current_ratio", 2.0, 0.5),
]
_MOMENTUM = [
    ("return_1y", 0.30, -0.30), ("sharpe", 1.5, -0.5), ("alpha", 0.15, -0.15),
    ("max_drawdown", -0.10, -0.50), ("annual_volatility", 0.15, 0.60),
]
_GROWTH = [
    ("revenue_growth", 0.20, -0.10), ("earnings_growth", 0.25, -0.30),
]
_FACTORS = {"value": _VALUE, "quality": _QUALITY, "momentum": _MOMENTUM, "growth": _GROWTH}
_WEIGHTS = {"value": 0.30, "quality": 0.30, "momentum": 0.20, "growth": 0.20}
_TOTAL_METRICS = sum(len(v) for v in _FACTORS.values())


@dataclass
class ScoreResult:
    score: float                       # composite 0..1
    factors: dict[str, float] = field(default_factory=dict)  # per-factor 0..1 (present only)
    completeness: float = 0.0          # fraction of metrics available
    confidence: float = 0.5


def _ramp(x: float, good: float, bad: float) -> float:
    if good == bad:
        return 0.5
    t = (x - bad) / (good - bad)
    return max(0.0, min(1.0, t))


def _factor_score(specs: list[tuple], values: dict) -> tuple[float | None, int]:
    scores = []
    for attr, good, bad in specs:
        v = values.get(attr)
        if v is not None:
            scores.append(_ramp(float(v), good, bad))
    if not scores:
        return None, 0
    return sum(scores) / len(scores), len(scores)


def composite_score(fundamentals: Fundamentals, metrics: QuantMetrics) -> ScoreResult:
    values = {**fundamentals.model_dump(), **metrics.model_dump()}

    factor_scores: dict[str, float] = {}
    present_metrics = 0
    for name, specs in _FACTORS.items():
        fs, n = _factor_score(specs, values)
        present_metrics += n
        if fs is not None:
            factor_scores[name] = fs

    if not factor_scores:
        return ScoreResult(score=0.5, factors={}, completeness=0.0, confidence=0.1)

    wsum = sum(_WEIGHTS[f] for f in factor_scores)
    composite = sum(factor_scores[f] * _WEIGHTS[f] for f in factor_scores) / wsum

    completeness = present_metrics / _TOTAL_METRICS
    decisiveness = abs(composite - 0.5) * 2  # 0 at neutral, 1 at extremes
    confidence = max(0.1, min(0.95, 0.25 + 0.45 * completeness + 0.30 * decisiveness))

    return ScoreResult(
        score=round(composite, 4), factors={k: round(v, 4) for k, v in factor_scores.items()},
        completeness=round(completeness, 3), confidence=round(confidence, 3),
    )
