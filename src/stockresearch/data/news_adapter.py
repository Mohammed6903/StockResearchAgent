"""News via Gemini Google Search grounding.

We query the fast model with grounding enabled and harvest the grounding metadata
(real source titles + URLs) plus the model's short synthesis. This gives grounded,
attributable headlines rather than free-floating LLM text.
"""

from __future__ import annotations

from datetime import date

from google.genai import types

from ..config import load_settings
from ..gemini import get_client
from ..models import NewsItem
from .cache import cached

_SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())


def _today() -> str:
    return date.today().isoformat()


def _region() -> str:
    return (load_settings().universe.region or "global").strip()


def _is_focused(region: str) -> bool:
    return bool(region) and region.lower() != "global"


def _grounded(query: str) -> tuple[str, list[NewsItem]]:
    client = get_client()
    model = load_settings().vertex.fast_model
    resp = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(tools=[_SEARCH_TOOL], temperature=0.2),
    )
    text = resp.text or ""
    items: list[NewsItem] = []
    try:
        meta = resp.candidates[0].grounding_metadata
        for chunk in (meta.grounding_chunks or []):
            web = getattr(chunk, "web", None)
            if web and web.title:
                items.append(NewsItem(title=web.title, source=web.title, url=web.uri))
    except (AttributeError, IndexError, TypeError):
        pass
    return text, items


def get_company_news(ticker: str, name: str | None) -> tuple[str, list[NewsItem]]:
    """Returns (synthesis, sources) for recent material news about one company."""
    label = name or ticker

    region = _region()
    market = f" (listed in {region})" if _is_focused(region) else ""
    today = _today()

    def produce() -> dict:
        q = (
            f"Today is {today}. Find ONLY news published within the last ~30 days about "
            f"{label} ({ticker}){market} that could affect its stock: earnings, guidance, "
            "regulation, litigation, management changes, product, M&A. Include the "
            "publication date of each item. Critically: IGNORE anything older than ~1 month "
            "and never describe a prior-year event (e.g. an FY24 result) as current. If there "
            "is no material news in the last ~30 days, reply exactly 'No material recent news.' "
            "Summarize in 3-4 sentences."
        )
        text, items = _grounded(q)
        return {"text": text, "items": [i.model_dump() for i in items]}

    data = cached(f"news:{ticker}:{region}:v2", produce)
    return data.get("text", ""), [NewsItem(**i) for i in data.get("items", [])]


def get_macro_news() -> tuple[str, list[NewsItem]]:
    """Returns (synthesis, sources) for political/macro headlines that move markets."""

    region = _region()
    today = _today()
    if _is_focused(region):
        scope = (
            f"that are likely to move the {region} stock market — both {region}-specific "
            f"events (e.g. the central bank/RBI, regulators/SEBI, the Union Budget, "
            f"elections, sector policy, monsoon) AND global events that materially affect "
            f"{region} markets (US Fed policy, crude oil prices, FII/FPI flows, USD/INR, "
            f"global risk sentiment)"
        )
    else:
        scope = "that are likely to move stock markets or change the outlook for sectors"

    def produce() -> dict:
        q = (
            f"Today is {today}. What are the most important political, geopolitical, "
            f"regulatory, central-bank, and macroeconomic news headlines from the last ~7 days "
            f"{scope}? Include each item's date. IGNORE anything older than ~2 weeks and never "
            "present a prior-year event as current. List each with a one-line market-impact "
            "note. Ignore non-market news."
        )
        text, items = _grounded(q)
        return {"text": text, "items": [i.model_dump() for i in items]}

    data = cached(f"news:macro:{region}:v2", produce)
    return data.get("text", ""), [NewsItem(**i) for i in data.get("items", [])]
