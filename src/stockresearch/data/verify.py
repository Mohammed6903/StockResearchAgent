"""Cross-check yfinance fundamentals against a second free source (Alpha Vantage).

Tier-2 hardening: yfinance/Yahoo can be stale or wrong. We fetch the same ratios from an
independent source and flag any material disagreement as a data warning, rather than
trusting one source blindly. The comparison logic is pure and source-agnostic so it can be
unit-tested without network; the Alpha Vantage adapter is the (optional) data provider.

Requires a free Alpha Vantage API key in env var ALPHAVANTAGE_API_KEY. Without it,
verification is a no-op (returns a single informational note). The free tier is rate
limited (~25 req/day), so verification is opt-in for scans and on-demand for single-ticker
deep dives.
"""

from __future__ import annotations

import os
import urllib.request

from ..models import Fundamentals
from .cache import cached

# Our Fundamentals field -> human label + whether the value is a fraction (margins/yields).
_COMPARABLE: dict[str, str] = {
    "pe_ratio": "P/E",
    "forward_pe": "Forward P/E",
    "peg_ratio": "PEG",
    "pb_ratio": "P/B",
    "ps_ratio": "P/S",
    "profit_margin": "profit margin",
    "operating_margin": "operating margin",
    "roa": "ROA",
    "roe": "ROE",
    "market_cap": "market cap",
    "dividend_yield": "dividend yield",
}

# Alpha Vantage OVERVIEW key -> our Fundamentals field name.
_AV_MAP: dict[str, str] = {
    "PERatio": "pe_ratio",
    "ForwardPE": "forward_pe",
    "PEGRatio": "peg_ratio",
    "PriceToBookRatio": "pb_ratio",
    "PriceToSalesRatioTTM": "ps_ratio",
    "ProfitMargin": "profit_margin",
    "OperatingMarginTTM": "operating_margin",
    "ReturnOnAssetsTTM": "roa",
    "ReturnOnEquityTTM": "roe",
    "MarketCapitalization": "market_cap",
    "DividendYield": "dividend_yield",
}


def _rel_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom if denom else 0.0


def compare_fundamentals(
    ours: Fundamentals, other: dict[str, float], *, tolerance: float = 0.10
) -> list[str]:
    """Return a warning per comparable field whose two sources differ by > tolerance.

    `other` maps our Fundamentals field names -> the second source's numeric value.
    Pure function: no network, no globals — this is the unit-tested core.
    """
    warnings: list[str] = []
    for field, label in _COMPARABLE.items():
        a = getattr(ours, field, None)
        b = other.get(field)
        if a is None or b is None:
            continue
        if _rel_diff(float(a), float(b)) > tolerance:
            warnings.append(
                f"{label} mismatch: Yahoo {float(a):.4g} vs 2nd source {float(b):.4g} "
                f"({_rel_diff(float(a), float(b)) * 100:.0f}% diff)"
            )
    return warnings


def _parse(v) -> float | None:
    if v in (None, "None", "-", "", "NaN"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def alphavantage_overview(ticker: str) -> dict[str, float] | None:
    """Fetch Alpha Vantage OVERVIEW and map to our field names. None if unavailable."""
    key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        return None

    def produce() -> dict:
        url = (
            "https://www.alphavantage.co/query?function=OVERVIEW"
            f"&symbol={ticker}&apikey={key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as r:  # noqa: S310 (trusted host)
                import json

                return json.loads(r.read().decode())
        except Exception:
            return {}

    raw = cached(f"av_overview:{ticker}", produce)
    if not raw or "Symbol" not in raw:
        return None
    out: dict[str, float] = {}
    for av_key, our_field in _AV_MAP.items():
        val = _parse(raw.get(av_key))
        if val is not None:
            out[our_field] = val
    return out


def verify_fundamentals(ticker: str, ours: Fundamentals, *, tolerance: float = 0.10) -> list[str]:
    """Cross-check against the configured second source; return discrepancy warnings.

    Returns an informational note (not an error) when no second source is configured, so
    the report is explicit about whether verification actually happened.
    """
    other = alphavantage_overview(ticker)
    if other is None:
        return ["fundamentals not cross-checked (no 2nd source / API key)"]
    return compare_fundamentals(ours, other, tolerance=tolerance)
