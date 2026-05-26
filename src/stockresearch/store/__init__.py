"""Storage backend selection.

`get_store()` returns the active backend: a MongoStore when MONGODB_URI is set and reachable
(centralized store for the training pipeline), otherwise the local SQLite `db` module
(always works offline). Both expose the same method surface, so callers are backend-agnostic.
"""

from __future__ import annotations

import logging
import os
from datetime import date as Date
from functools import lru_cache

from ..models import DailyReport, MacroSignal, TickerAnalysis, TickerSnapshot

log = logging.getLogger("stockresearch")


@lru_cache(maxsize=1)
def get_store():
    uri = os.environ.get("MONGODB_URI")
    if uri:
        try:
            from .mongo import MongoStore

            store = MongoStore(uri, os.environ.get("MONGODB_DB", "stockresearch"))
            log.info("Using MongoDB backend.")
            return store
        except Exception as e:  # unreachable / auth / driver missing -> fall back
            log.warning("MongoDB unavailable (%s); falling back to SQLite.", e)
    from . import db

    return db


def active_backend_name() -> str:
    store = get_store()
    return "mongodb" if type(store).__name__ == "MongoStore" else "sqlite"


def persist_analyses(
    run_date: Date,
    analyses: list[TickerAnalysis],
    snapshots: list[TickerSnapshot],
    *,
    macro_summary: str | None = None,
    macro_signals: list[MacroSignal] | None = None,
    universe_size: int | None = None,
    errors: list[str] | None = None,
) -> DailyReport:
    """Merge analyses/snapshots into the day's report (upsert per ticker, latest wins).

    Used by both `scan` and `analyze` so a day's record is the union of everything analyzed
    that day, regardless of which command produced it — nothing is lost or duplicated.
    """
    store = get_store()
    report = store.load_report(run_date) or DailyReport(run_date=run_date, universe_size=0)

    a_by = {a.ticker: a for a in report.analyses}
    s_by = {s.ticker: s for s in report.snapshots}
    for a in analyses:
        a_by[a.ticker] = a
    for s in snapshots:
        s_by[s.ticker] = s
    report.analyses = sorted(a_by.values(), key=lambda a: a.score, reverse=True)
    report.snapshots = list(s_by.values())

    if macro_summary is not None:
        report.macro_summary = macro_summary
    if macro_signals is not None:
        report.macro_signals = macro_signals
    if errors is not None:
        report.errors = errors
    if universe_size is not None:
        report.universe_size = universe_size
    elif not report.universe_size:
        report.universe_size = len(report.analyses)

    store.save_report(report)
    return report
