"""Configuration loading. Resolves project root, settings.yaml, and watchlist.yaml."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_PLACEHOLDER_PROJECTS = {"", "your-gcp-project-id", "YOUR_GCP_PROJECT"}


def project_root() -> Path:
    # src/stockresearch/config.py -> project root is three parents up.
    return Path(__file__).resolve().parents[2]


def _adc_quota_project() -> str | None:
    """Fall back to the Application Default Credentials quota project, so the tool runs
    locally with zero config and the committed settings.yaml can stay free of a real ID."""
    path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    try:
        return json.loads(path.read_text()).get("quota_project_id")
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_project(yaml_value: str | None) -> str:
    """Project precedence: GOOGLE_CLOUD_PROJECT env > settings.yaml > ADC quota project."""
    env = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if env:
        return env
    if yaml_value and yaml_value not in _PLACEHOLDER_PROJECTS:
        return yaml_value
    adc = _adc_quota_project()
    if adc:
        return adc
    raise RuntimeError(
        "No GCP project configured. Set vertex.project in config/settings.yaml, export "
        "GOOGLE_CLOUD_PROJECT, or run `gcloud auth application-default login`."
    )


class VertexConfig(BaseModel):
    project: str
    location: str = "us-central1"
    model: str = "gemini-2.5-pro"
    fast_model: str = "gemini-2.5-flash"


class UniverseConfig(BaseModel):
    index: str = "SP500"
    benchmark: str = "SPY"
    max_tickers: int = 50


class QuantConfig(BaseModel):
    lookback_days: int = 365
    risk_free_rate: float = 0.045


class RuntimeConfig(BaseModel):
    concurrency: int = 4
    cache_ttl_hours: int = 12
    reports_dir: str = "reports"
    db_path: str = ".cache/stockresearch.db"


class Settings(BaseModel):
    vertex: VertexConfig
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @property
    def reports_path(self) -> Path:
        return _resolve(self.runtime.reports_dir)

    @property
    def db_full_path(self) -> Path:
        return _resolve(self.runtime.db_path)

    @property
    def cache_dir(self) -> Path:
        return _resolve(".cache")


def _resolve(rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else project_root() / p


@lru_cache(maxsize=1)
def load_settings() -> Settings:
    raw = yaml.safe_load((project_root() / "config" / "settings.yaml").read_text())
    raw.setdefault("vertex", {})
    raw["vertex"]["project"] = _resolve_project(raw["vertex"].get("project"))
    return Settings(**raw)


@lru_cache(maxsize=1)
def load_watchlist() -> list[str]:
    raw = yaml.safe_load((project_root() / "config" / "watchlist.yaml").read_text())
    return [str(t).strip().upper() for t in (raw.get("tickers") or [])]
