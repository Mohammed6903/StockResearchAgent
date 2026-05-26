"""Vertex AI / Gemini client helpers. Centralizes env setup so agents and tools share it."""

from __future__ import annotations

import os
from functools import lru_cache

from google import genai

from .config import load_settings


def configure_vertex_env() -> None:
    """Set the env vars ADK and google-genai read for Vertex mode. Idempotent."""
    s = load_settings().vertex
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", s.project)
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", s.location)


@lru_cache(maxsize=1)
def get_client() -> genai.Client:
    s = load_settings().vertex
    return genai.Client(vertexai=True, project=s.project, location=s.location)
