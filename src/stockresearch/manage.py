"""Read/write helpers for the watchlist and settings, used by the `watch`/`config` CLI.

Edits round-trip through ruamel.yaml so user comments and formatting in the config files
survive. Every settings change is re-validated against the Settings schema before it is
written, so an invalid value can't corrupt the config. `config set` is restricted to a
curated allowlist of daily-use keys; everything remains readable via `get`/`show`.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from .config import Settings, _resolve_project, load_watchlist, project_root

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=2, offset=0)

# key -> (human description, coercion function). Only these may be changed via `config set`.
SETTABLE: dict[str, tuple[str, Any]] = {
    "universe.index": ("index to union with watchlist (SP500 | NIFTY50 | NONE)", "_index"),
    "universe.benchmark": ("benchmark ticker for beta/alpha (e.g. SPY, ^NSEI)", "_upper"),
    "universe.max_tickers": ("max tickers scanned per run", "_int"),
    "universe.region": ("market focus for macro/news (e.g. India, US, global)", "_str"),
    "vertex.model": ("Gemini model for reasoning/synthesis", "_str"),
    "vertex.fast_model": ("Gemini model for macro/news triage", "_str"),
}

_VALID_INDEXES = {"SP500", "NIFTY50", "NONE"}


def _settings_path() -> Path:
    return project_root() / "config" / "settings.yaml"


def _watchlist_path() -> Path:
    return project_root() / "config" / "watchlist.yaml"


def _load(path: Path):
    return _yaml.load(path.read_text())


def _dump(path: Path, data) -> None:
    with path.open("w") as f:
        _yaml.dump(data, f)


# ---- watchlist -------------------------------------------------------------------------

def list_watchlist() -> list[str]:
    return load_watchlist()


def _write_tickers(data, tickers: list[str]) -> None:
    data["tickers"] = tickers
    _dump(_watchlist_path(), data)


def add_watchlist(tickers: list[str]) -> tuple[list[str], list[str]]:
    """Return (newly_added, full_list)."""
    data = _load(_watchlist_path())
    current = [str(t).strip().upper() for t in (data.get("tickers") or [])]
    added = []
    for t in (x.strip().upper() for x in tickers):
        if t and t not in current:
            current.append(t)
            added.append(t)
    _write_tickers(data, current)
    return added, current


def remove_watchlist(tickers: list[str]) -> tuple[list[str], list[str]]:
    """Return (removed, full_list)."""
    data = _load(_watchlist_path())
    current = [str(t).strip().upper() for t in (data.get("tickers") or [])]
    drop = {x.strip().upper() for x in tickers}
    removed = [t for t in current if t in drop]
    current = [t for t in current if t not in drop]
    _write_tickers(data, current)
    return removed, current


def clear_watchlist() -> None:
    data = _load(_watchlist_path())
    _write_tickers(data, [])


# ---- settings --------------------------------------------------------------------------

def _coerce(key: str, value: str):
    kind = SETTABLE[key][1]
    if kind == "_int":
        return int(value)
    if kind == "_upper":
        return value.strip().upper()
    if kind == "_index":
        v = value.strip().upper()
        if v not in _VALID_INDEXES:
            raise ValueError(f"universe.index must be one of {', '.join(sorted(_VALID_INDEXES))}")
        return v
    return value


def _navigate(root, key: str):
    parts = key.split(".")
    node = root
    for p in parts[:-1]:
        if p not in node:
            raise KeyError(key)
        node = node[p]
    return node, parts[-1]


def get_setting(key: str):
    data = _load(_settings_path())
    node, leaf = _navigate(data, key)
    if leaf not in node:
        raise KeyError(key)
    return node[leaf]


def show_settings() -> dict:
    return dict(_load(_settings_path()))


def _validate(raw_mapping) -> None:
    """Validate a settings mapping against the schema (resolving the project so the
    required field is satisfied), without persisting the resolved project."""
    plain = copy.deepcopy(dict(raw_mapping))
    plain.setdefault("vertex", {})
    plain["vertex"] = dict(plain["vertex"])
    plain["vertex"]["project"] = _resolve_project(plain["vertex"].get("project"))
    Settings(**plain)


def set_setting(key: str, value: str):
    if key not in SETTABLE:
        allowed = ", ".join(sorted(SETTABLE))
        raise ValueError(f"'{key}' is not settable via CLI. Allowed: {allowed}")
    coerced = _coerce(key, value)
    data = _load(_settings_path())
    node, leaf = _navigate(data, key)
    node[leaf] = coerced
    _validate(data)
    _dump(_settings_path(), data)
    return coerced
