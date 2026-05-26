"""Tests for watch/config management: round-trip edits, comment preservation, validation."""

import pytest

from stockresearch import config, manage

_SETTINGS = """\
vertex:
  # keep this comment
  project: your-gcp-project-id
  location: us-central1
  model: gemini-2.5-pro
  fast_model: gemini-2.5-flash
universe:
  index: SP500
  benchmark: SPY
  max_tickers: 50
quant:
  lookback_days: 365
  risk_free_rate: 0.045
runtime:
  concurrency: 4
  cache_ttl_hours: 12
  reports_dir: reports
  db_path: .cache/stockresearch.db
"""

_WATCHLIST = """\
# my tickers
tickers:
  - AAPL
  - MSFT
"""


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "settings.yaml").write_text(_SETTINGS)
    (cfgdir / "watchlist.yaml").write_text(_WATCHLIST)
    monkeypatch.setattr(manage, "project_root", lambda: tmp_path)
    monkeypatch.setattr(config, "project_root", lambda: tmp_path)
    config.load_watchlist.cache_clear()
    config.load_settings.cache_clear()
    # ensure project resolution doesn't depend on a real ADC file
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-proj")
    return tmp_path


def test_watch_add_dedupes_and_uppercases(cfg):
    added, full = manage.add_watchlist(["nvda", "AAPL"])  # aapl already present
    assert added == ["NVDA"]
    assert full == ["AAPL", "MSFT", "NVDA"]


def test_watch_remove(cfg):
    removed, full = manage.remove_watchlist(["msft", "ZZZ"])
    assert removed == ["MSFT"]
    assert full == ["AAPL"]


def test_watch_clear(cfg):
    manage.clear_watchlist()
    assert manage.list_watchlist() == []


def test_config_set_persists_and_preserves_comment(cfg):
    manage.set_setting("universe.benchmark", "qqq")
    assert manage.get_setting("universe.benchmark") == "QQQ"
    assert "# keep this comment" in (cfg / "config" / "settings.yaml").read_text()


def test_config_set_coerces_int(cfg):
    manage.set_setting("universe.max_tickers", "10")
    assert manage.get_setting("universe.max_tickers") == 10


def test_config_set_rejects_non_allowlisted_key(cfg):
    with pytest.raises(ValueError, match="not settable"):
        manage.set_setting("runtime.concurrency", "8")


def test_config_set_validates_index_value(cfg):
    with pytest.raises(ValueError, match="SP500 or NONE"):
        manage.set_setting("universe.index", "DOWJONES")


def test_config_set_bad_int_raises(cfg):
    with pytest.raises(ValueError):
        manage.set_setting("universe.max_tickers", "lots")


def test_get_reads_readonly_keys(cfg):
    assert manage.get_setting("quant.risk_free_rate") == 0.045
