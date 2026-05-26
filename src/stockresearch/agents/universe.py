"""Universe resolver: candidate tickers = watchlist ∪ index, capped for cost safety."""

from __future__ import annotations

from ..config import load_settings, load_watchlist
from ..data.index_membership import index_tickers


def resolve_universe(extra: list[str] | None = None) -> list[str]:
    s = load_settings().universe
    watch = load_watchlist()
    idx = index_tickers(s.index)
    seen: dict[str, None] = {}
    for t in [*watch, *(extra or []), *idx]:
        seen.setdefault(t.strip().upper(), None)
    tickers = list(seen)
    return tickers[: s.max_tickers]
