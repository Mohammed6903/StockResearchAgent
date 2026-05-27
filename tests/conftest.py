"""Test hygiene: never touch a real MongoDB. Forces the local SQLite backend for every test
(the dev shell may export MONGODB_URI), and clears the store-selection cache around each test."""

import pytest


@pytest.fixture(autouse=True)
def _force_local_store(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.delenv("MONGODB_DB", raising=False)
    import stockresearch.store as store_pkg

    def _clear():
        clear = getattr(store_pkg.get_store, "cache_clear", None)
        if clear:
            clear()

    _clear()
    yield
    _clear()
