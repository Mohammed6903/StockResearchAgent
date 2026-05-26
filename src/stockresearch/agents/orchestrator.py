"""Daily pipeline orchestrator: macro scan -> resolve universe -> parallel per-ticker
analysis (with reflection) -> ranked DailyReport. Per-ticker failures are isolated."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as Date

from ..config import load_settings
from ..data.yfinance_adapter import get_closes
from ..models import DailyReport, TickerAnalysis
from .macro_agent import run_macro_scan
from .ticker_analyst import analyze_ticker
from .universe import resolve_universe

log = logging.getLogger("stockresearch")


def run_daily(
    tickers: list[str] | None = None,
    *,
    run_date: Date | None = None,
    reflect: bool = True,
    fetch_news: bool = True,
    skip_macro: bool = False,
    verify: bool = False,
    progress=None,
) -> DailyReport:
    settings = load_settings()
    run_date = run_date or Date.today()
    universe = tickers or resolve_universe()

    macro_summary, macro_signals = ("", [])
    if not skip_macro:
        try:
            macro_summary, macro_signals = run_macro_scan()
        except Exception as e:  # macro is best-effort; never block the run
            log.warning("macro scan failed: %s", e)

    benchmark_closes = get_closes(settings.universe.benchmark, settings.quant.lookback_days)

    analyses: list[TickerAnalysis] = []
    errors: list[str] = []

    def work(t: str) -> TickerAnalysis:
        return analyze_ticker(
            t, benchmark_closes, macro_signals,
            reflect=reflect, fetch_news=fetch_news, verify=verify,
        )

    with ThreadPoolExecutor(max_workers=settings.runtime.concurrency) as pool:
        futures = {pool.submit(work, t): t for t in universe}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                analyses.append(fut.result())
            except Exception as e:
                msg = f"{t}: {e}"
                errors.append(msg)
                log.warning("ticker analysis failed for %s", msg)
            if progress:
                progress(t)

    analyses.sort(key=lambda a: a.score, reverse=True)
    return DailyReport(
        run_date=run_date,
        universe_size=len(universe),
        macro_summary=macro_summary,
        macro_signals=macro_signals,
        analyses=analyses,
        errors=errors,
    )


def run_single(ticker: str, *, reflect: bool = True, verify: bool = True) -> TickerAnalysis:
    settings = load_settings()
    benchmark_closes = get_closes(settings.universe.benchmark, settings.quant.lookback_days)
    macro_summary, macro_signals = run_macro_scan()
    _ = macro_summary
    return analyze_ticker(
        ticker.upper(), benchmark_closes, macro_signals, reflect=reflect, verify=verify
    )
