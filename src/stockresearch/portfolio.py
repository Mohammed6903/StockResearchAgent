"""Personal-finance layer: value/P&L per holding plus beginner-oriented portfolio checks
(concentration, sector spread). Deterministic; pulls live price + sector from yfinance and
the latest stored lean from the DB. Framed as education, never as advice."""

from __future__ import annotations

from . import manage
from .data.yfinance_adapter import get_closes, get_fundamentals
from .store import get_store

db = get_store()

# Beginner rule-of-thumb thresholds for the flags (not advice — just prompts to think).
_CONCENTRATION_WARN = 0.25   # any single holding above 25% of the portfolio
_SECTOR_WARN = 0.40          # any single sector above 40%


def review() -> dict:
    """Return per-holding rows + portfolio-level summary and flags. Empty if no holdings."""
    holdings = manage.list_holdings()
    if not holdings:
        return {}

    rows: list[dict] = []
    for h in holdings:
        closes = get_closes(h["ticker"], 30)
        price = float(closes.iloc[-1]) if len(closes) else None
        fund = get_fundamentals(h["ticker"])
        value = price * h["qty"] if price is not None else None
        cost = h["avg_price"] * h["qty"]
        pl = (value - cost) if value is not None else None
        hist = db.ticker_history(h["ticker"])
        rows.append({
            "ticker": h["ticker"],
            "qty": h["qty"],
            "avg_price": h["avg_price"],
            "price": price,
            "value": value,
            "cost": cost,
            "pl": pl,
            "pl_pct": (pl / cost) if (pl is not None and cost) else None,
            "sector": fund.sector or "Unknown",
            "latest_lean": hist[-1]["lean"] if hist else None,
        })

    total_value = sum(r["value"] for r in rows if r["value"] is not None) or 0.0
    for r in rows:
        r["weight"] = (r["value"] / total_value) if (r["value"] and total_value) else None

    sector_weights: dict[str, float] = {}
    for r in rows:
        if r["weight"]:
            sector_weights[r["sector"]] = sector_weights.get(r["sector"], 0.0) + r["weight"]

    flags: list[str] = []
    for r in rows:
        if r["weight"] and r["weight"] > _CONCENTRATION_WARN:
            flags.append(
                f"{r['ticker']} is {r['weight']:.0%} of the portfolio — concentrated in one "
                "stock; a single bad surprise hits hard."
            )
    for sector, w in sector_weights.items():
        if w > _SECTOR_WARN:
            flags.append(
                f"{w:.0%} of the portfolio is in {sector} — heavy sector exposure; consider "
                "whether you're diversified across sectors."
            )
    if len(rows) < 5:
        flags.append(
            f"Only {len(rows)} holding(s) — few positions means less diversification. "
            "Beginners often spread across more names/sectors."
        )

    total_cost = sum(r["cost"] for r in rows) or 0.0
    return {
        "rows": sorted(rows, key=lambda r: r["value"] or 0, reverse=True),
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pl": total_value - total_cost,
        "sector_weights": sector_weights,
        "flags": flags,
    }
