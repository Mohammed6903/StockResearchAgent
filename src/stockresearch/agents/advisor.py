"""Turn an analysis into an actionable, capital-sized BUY/SELL/HOLD recommendation.

The LLM chooses the direction and a *target position size* (as a fraction of capital), aware
of the capital scale and the current holding. Python then does all the sizing arithmetic and
enforces the hard constraints — the per-position cap, "can't sell what you don't own", and
"can't buy without a price" — so no share count is ever hallucinated.
"""

from __future__ import annotations

import math
from datetime import date as Date

from ..models import Action, Recommendation, TickerAnalysis, TickerSnapshot
from .reason import reason
from .schemas import AdviceVerdict


def _verdict(snapshot: TickerSnapshot, analysis: TickerAnalysis, *, capital: float,
             max_position_pct: float, held_qty: float, price: float) -> AdviceVerdict:
    held_value = held_qty * price
    max_position_value = capital * max_position_pct
    facts = {
        "ticker": snapshot.ticker,
        "name": analysis.name,
        "analysis_lean": analysis.lean.value,
        "analysis_score": analysis.score,
        "analysis_confidence": analysis.confidence,
        "reasoning": analysis.reasoning,
        "pros": analysis.pros,
        "cons": analysis.cons,
        "price": price,
        "total_capital": capital,
        "max_position_value": max_position_value,
        "max_position_pct": max_position_pct,
        "current_holding_shares": held_qty,
        "current_holding_value": held_value,
        "current_weight_pct": (held_value / capital) if capital else 0.0,
    }
    prompt = (
        "You are a portfolio manager turning the research below into ONE actionable call for "
        "this single stock. Decide action (buy/sell/hold) and the target position size for "
        f"this stock as a fraction of total capital. You MUST NOT exceed the per-position cap "
        f"of {max_position_pct:.0%} of capital. Be proportionate to the conviction (score and "
        "confidence) and mindful of the capital scale and the current holding shown. "
        "SELL only makes sense if there is a current holding. This is research sizing, not "
        "financial advice.\n\n"
        f"FACTS:\n{facts}\n"
    )
    return reason(prompt, AdviceVerdict, temperature=0.2)


def advise(snapshot: TickerSnapshot, analysis: TickerAnalysis, *, capital: float,
           max_position_pct: float, held_qty: float, run_date: Date | None = None) -> Recommendation:
    run_date = run_date or Date.today()
    price = snapshot.metrics.last_price
    notes: list[str] = []

    base = Recommendation(
        run_date=run_date, ticker=snapshot.ticker, lean=analysis.lean,
        score=analysis.score, confidence=analysis.confidence, price_at_rec=price,
    )

    if not capital or capital <= 0:
        base.notes = ["no capital configured; set account.capital"]
        return base
    if price is None or price <= 0:
        base.notes = ["price unavailable; cannot size a trade"]
        return base

    v = _verdict(snapshot, analysis, capital=capital, max_position_pct=max_position_pct,
                 held_qty=held_qty, price=price)
    base.confidence = v.confidence
    base.rationale = v.rationale

    target_pct = max(0.0, min(v.target_pct_of_capital, max_position_pct))
    if v.target_pct_of_capital > max_position_pct:
        notes.append(f"target trimmed to per-position cap of {max_position_pct:.0%}")
    target_shares = math.floor(target_pct * capital / price)
    delta = target_shares - held_qty  # +want to buy, -want to sell

    if v.action == Action.BUY:
        if delta > 0:
            base.action = Action.BUY
            base.shares = float(delta)
        else:
            base.action = Action.HOLD
            notes.append("already at/above target size; no buy needed")
    elif v.action == Action.SELL:
        if held_qty <= 0:
            base.action = Action.HOLD
            notes.append("no holding to sell")
        else:
            sell_shares = held_qty if target_shares >= held_qty else (held_qty - target_shares)
            sell_shares = min(held_qty, max(0.0, sell_shares)) or held_qty
            base.action = Action.SELL
            base.shares = float(sell_shares)
    else:
        base.action = Action.HOLD

    base.target_pct = target_pct
    base.amount = round(base.shares * price, 2)
    base.notes = notes
    return base
