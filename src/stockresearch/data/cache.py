"""Tiny day-level JSON disk cache to avoid refetching within a run / day."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from ..config import load_settings


def _cache_dir() -> Path:
    d = load_settings().cache_dir / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(key: str) -> Path:
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return _cache_dir() / f"{h}.json"


def cached(key: str, producer: Callable[[], Any]) -> Any:
    """Return cached JSON for key if fresh, else call producer() and store it."""
    ttl = load_settings().runtime.cache_ttl_hours * 3600
    p = _path(key)
    if p.exists() and (time.time() - p.stat().st_mtime) < ttl:
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    value = producer()
    try:
        p.write_text(json.dumps(value))
    except (TypeError, OSError):
        pass
    return value
