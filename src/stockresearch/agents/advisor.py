"""Turn the (deterministic, factor-based) analysis score into an actionable, capital-sized
BUY/SELL/HOLD call.

Direction and sizing are derived **deterministically** from the composite score, the current
holding, and the configured capital — no LLM step, because letting the model choose the action
just collapses everything to a hedged HOLD. Conviction scales the position toward the
per-position cap. The rationale reuses the analyst's narrative plus a templated action line.
"""

from __future__ import annotations

import math
from datetime import date as Date

from ..models import Action, Recommendation, TickerAnalysis, TickerSnapshot
from ..quant.score import BUY_THRESHOLD, SELL_THRESHOLD

_FULL_EXIT_SCORE = 0.25  # at/below this, sell the entire holding


def _buy_target_pct(score: float, max_position_pct: float) -> float:
    """Scale position size with conviction: ~40% of the cap at the buy threshold, full cap as
    the score approaches 1."""
    frac = (score - BUY_THRESHOLD) / max(1e-9, 1.0 - BUY_THRESHOLD)
    return max_position_pct * (0.4 + 0.6 * max(0.0, min(1.0, frac)))


def advise(snapshot: TickerSnapshot, analysis: TickerAnalysis, *, capital: float,
           max_position_pct: float, held_qty: float, run_date: Date | None = None) -> Recommendation:
    run_date = run_date or Date.today()
    price = snapshot.metrics.last_price
    score = analysis.score
    rec = Recommendation(
        run_date=run_date, ticker=snapshot.ticker, lean=analysis.lean, score=score,
        confidence=analysis.confidence, price_at_rec=price, rationale=analysis.reasoning,
    )
    notes: list[str] = []

    if not capital or capital <= 0:
        rec.notes = ["no capital configured; set account.capital"]
        return rec
    if price is None or price <= 0:
        rec.notes = ["price unavailable; cannot size a trade"]
        return rec
    if not analysis.factors:
        rec.notes = ["insufficient quant data to score; no actionable call"]
        return rec

    if score >= BUY_THRESHOLD:
        target_pct = _buy_target_pct(score, max_position_pct)
        target_shares = math.floor(target_pct * capital / price)
        delta = target_shares - held_qty
        if delta > 0:
            rec.action = Action.BUY
            rec.shares = float(delta)
            rec.target_pct = round(target_pct, 4)
        else:
            rec.action = Action.HOLD
            notes.append("attractive, but already at/above target size — no buy needed")
    elif score <= SELL_THRESHOLD:
        if held_qty > 0:
            if score <= _FULL_EXIT_SCORE:
                sell = held_qty
            else:
                frac = (SELL_THRESHOLD - score) / max(1e-9, SELL_THRESHOLD - _FULL_EXIT_SCORE)
                sell = math.ceil(held_qty * (0.5 + 0.5 * max(0.0, min(1.0, frac))))
            rec.action = Action.SELL
            rec.shares = float(min(held_qty, sell))
        else:
            rec.action = Action.HOLD
            notes.append("weak score — avoid; nothing held to sell")
    else:
        rec.action = Action.HOLD
        notes.append("middling score — no clear edge either way")

    rec.amount = round(rec.shares * price, 2)
    rec.notes = notes
    return rec
