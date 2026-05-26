"""Structured LLM reasoning over google-genai (the layer ADK builds on).

Each call hands Gemini a fully-assembled, factual payload and asks for a typed result
via response_schema. The model interprets verified numbers and grounded news — it is
never asked to produce a metric itself. JSON is parsed straight into Pydantic models.
"""

from __future__ import annotations

from typing import TypeVar

from google.genai import types
from pydantic import BaseModel

from ..config import load_settings
from ..gemini import get_client

T = TypeVar("T", bound=BaseModel)


def reason(prompt: str, schema: type[T], *, use_fast: bool = False, temperature: float = 0.3) -> T:
    settings = load_settings().vertex
    model = settings.fast_model if use_fast else settings.model
    resp = get_client().models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
            temperature=temperature,
        ),
    )
    parsed = getattr(resp, "parsed", None)
    if isinstance(parsed, schema):
        return parsed
    return schema.model_validate_json(resp.text)
