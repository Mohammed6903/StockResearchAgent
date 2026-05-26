"""Outcome scoring, calibration, and training-data export. All deterministic — no LLM.

Turns the daily run history into a feedback loop and a labeled dataset:
- `score_matured`: for each past call, once enough time has elapsed, compute the realized
  forward return and whether the lean was directionally correct.
- `calibration`: does stated confidence track real hit-rate? does each lean actually predict
  the right direction?
- `export_jsonl`: dump {inputs (snapshot), verdict, outcomes} rows for model training/eval.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

from .data.yfinance_adapter import price_on_or_after
from .store import get_store

db = get_store()

HORIZONS = (7, 30, 90)
# Directional band: |return| <= this counts as "flat" (neutral correct); outside it is a move.
_FLAT_BAND = 0.03


def _is_correct(lean: str, fwd: float) -> bool:
    if lean == "bullish":
        return fwd > _FLAT_BAND
    if lean == "bearish":
        return fwd < -_FLAT_BAND
    return abs(fwd) <= _FLAT_BAND  # neutral


def score_matured(today: date | None = None) -> int:
    """Compute and persist outcomes for calls old enough for each horizon. Returns #scored."""
    today = today or date.today()
    scored = 0
    for call in db.all_calls():
        rd = date.fromisoformat(call["run_date"])
        price_then = call["last_price"]
        if not price_then:
            continue
        for h in HORIZONS:
            target = rd + timedelta(days=h)
            if target > today:
                continue  # not matured yet
            price_later = price_on_or_after(call["ticker"], target.isoformat())
            if not price_later:
                continue
            fwd = price_later / price_then - 1.0
            lean = call["lean"] or "neutral"
            db.upsert_outcome(
                call["run_date"], call["ticker"], h, lean, call["confidence"],
                price_then, price_later, fwd, _is_correct(lean, fwd),
            )
            scored += 1
    return scored


def calibration() -> dict:
    """Aggregate accuracy by confidence bucket and by lean. Empty dict if no matured data."""
    rows = db.outcomes_rows()
    if not rows:
        return {}
    buckets = {"low (<50%)": (0.0, 0.5), "med (50-70%)": (0.5, 0.7), "high (>=70%)": (0.7, 1.01)}
    by_conf: dict[str, dict] = {b: {"n": 0, "correct": 0} for b in buckets}
    by_lean: dict[str, dict] = {}
    for r in rows:
        c = r["confidence"] or 0.0
        for label, (lo, hi) in buckets.items():
            if lo <= c < hi:
                by_conf[label]["n"] += 1
                by_conf[label]["correct"] += int(r["correct"] or 0)
                break
        lean = r["lean"] or "neutral"
        d = by_lean.setdefault(lean, {"n": 0, "correct": 0, "ret_sum": 0.0})
        d["n"] += 1
        d["correct"] += int(r["correct"] or 0)
        d["ret_sum"] += r["forward_return"] or 0.0
    total = len(rows)
    overall = sum(int(r["correct"] or 0) for r in rows) / total
    return {"total": total, "overall_accuracy": overall, "by_confidence": by_conf,
            "by_lean": by_lean}


def export_jsonl(path: str) -> int:
    """Write one JSONL row per (run, ticker) joining inputs + verdict + matured outcomes.

    Returns the number of rows written. This is the labeled dataset for training/eval.
    """
    outcomes: dict[tuple[str, str], dict] = {}
    for o in db.outcomes_rows():
        outcomes.setdefault((o["run_date"], o["ticker"]), {})[f"return_{o['horizon_days']}d"] = (
            o["forward_return"]
        )

    rows_written = 0
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for run in db.list_runs():
            report = db.load_report(date.fromisoformat(run["run_date"]))
            if not report:
                continue
            snap_by_ticker = {s.ticker: s for s in report.snapshots}
            for a in report.analyses:
                snap = snap_by_ticker.get(a.ticker)
                record = {
                    "run_date": report.run_date.isoformat(),
                    "ticker": a.ticker,
                    "inputs": snap.model_dump() if snap else None,
                    "verdict": {
                        "lean": a.lean.value, "score": a.score, "confidence": a.confidence,
                        "pros": a.pros, "cons": a.cons, "reasoning": a.reasoning,
                    },
                    "outcomes": outcomes.get((report.run_date.isoformat(), a.ticker), {}),
                }
                f.write(json.dumps(record, default=str) + "\n")
                rows_written += 1
    return rows_written
