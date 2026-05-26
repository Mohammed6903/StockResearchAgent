"""News via Gemini Google Search grounding.

We query the fast model with grounding enabled and harvest the grounding metadata
(real source titles + URLs) plus the model's short synthesis. This gives grounded,
attributable headlines rather than free-floating LLM text.
"""

from __future__ import annotations

from google.genai import types

from ..config import load_settings
from ..gemini import get_client
from ..models import NewsItem
from .cache import cached

_SEARCH_TOOL = types.Tool(google_search=types.GoogleSearch())


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

    def produce() -> dict:
        q = (
            f"Find the most material, recent news (last ~2 weeks) about {label} ({ticker}) "
            "that could affect its stock: earnings, guidance, regulation, litigation, "
            "management changes, product, M&A. Summarize in 3-4 sentences. Skip fluff."
        )
        text, items = _grounded(q)
        return {"text": text, "items": [i.model_dump() for i in items]}

    data = cached(f"news:{ticker}", produce)
    return data.get("text", ""), [NewsItem(**i) for i in data.get("items", [])]


def get_macro_news() -> tuple[str, list[NewsItem]]:
    """Returns (synthesis, sources) for political/macro headlines that move markets."""

    def produce() -> dict:
        q = (
            "What are the most important political, geopolitical, regulatory, central-bank, "
            "and macroeconomic news headlines from the last few days that are likely to move "
            "stock markets or change the outlook for specific sectors or companies? "
            "List each with a one-line explanation of the market impact. Ignore non-market news."
        )
        text, items = _grounded(q)
        return {"text": text, "items": [i.model_dump() for i in items]}

    data = cached("news:macro", produce)
    return data.get("text", ""), [NewsItem(**i) for i in data.get("items", [])]
