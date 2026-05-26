"""Storage backend selection.

`get_store()` returns the active backend: a MongoStore when MONGODB_URI is set and reachable
(centralized store for the training pipeline), otherwise the local SQLite `db` module
(always works offline). Both expose the same method surface, so callers are backend-agnostic.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

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
