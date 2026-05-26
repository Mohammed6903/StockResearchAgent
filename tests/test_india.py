"""Tests for India/NIFTY support: index membership, settable index value, region prompts."""

import pytest

from stockresearch.data import index_membership as idx
from stockresearch.data import news_adapter


def test_nifty_fallback_when_fetch_fails(monkeypatch):
    monkeypatch.setattr(idx, "cached", lambda key, producer: producer())
    monkeypatch.setattr(
        idx, "_symbols_from_wikipedia", lambda *a, **k: (_ for _ in ()).throw(RuntimeError())
    )
    out = idx.nifty50_tickers()
    assert "RELIANCE.NS" in out and all(t.endswith(".NS") for t in out)


def test_index_tickers_routing(monkeypatch):
    monkeypatch.setattr(idx, "sp500_tickers", lambda: ["AAPL"])
    monkeypatch.setattr(idx, "nifty50_tickers", lambda: ["RELIANCE.NS"])
    assert idx.index_tickers("NIFTY50") == ["RELIANCE.NS"]
    assert idx.index_tickers("sp500") == ["AAPL"]
    assert idx.index_tickers("NONE") == []


def test_wikipedia_parser_appends_suffix(monkeypatch):
    import pandas as pd

    df = pd.DataFrame({"Company": ["Reliance", "TCS"], "Symbol": ["RELIANCE", "TCS"]})
    monkeypatch.setattr("pandas.read_html", lambda url: [df])
    out = idx._symbols_from_wikipedia("http://x", suffix=".NS")
    assert out == ["RELIANCE.NS", "TCS.NS"]


def test_manage_allows_nifty50_index(tmp_path, monkeypatch):
    from stockresearch import config, manage

    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "settings.yaml").write_text(
        "vertex:\n  project: p\n  location: us-central1\n  model: m\n  fast_model: f\n"
        "universe:\n  index: SP500\n  benchmark: SPY\n  max_tickers: 50\n  region: global\n"
        "quant:\n  lookback_days: 365\n  risk_free_rate: 0.045\n"
        "runtime:\n  concurrency: 4\n  cache_ttl_hours: 12\n"
        "  reports_dir: reports\n  db_path: .cache/x.db\n"
    )
    (cfgdir / "watchlist.yaml").write_text("tickers: [RELIANCE.NS]\n")
    monkeypatch.setattr(manage, "project_root", lambda: tmp_path)
    monkeypatch.setattr(config, "project_root", lambda: tmp_path)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "p")

    manage.set_setting("universe.index", "nifty50")
    assert manage.get_setting("universe.index") == "NIFTY50"
    manage.set_setting("universe.region", "India")
    assert manage.get_setting("universe.region") == "India"


def test_macro_prompt_is_india_focused(monkeypatch):
    captured = {}
    monkeypatch.setattr(news_adapter, "_region", lambda: "India")
    monkeypatch.setattr(news_adapter, "cached", lambda key, producer: producer())

    def fake_grounded(q):
        captured["q"] = q
        return "text", []

    monkeypatch.setattr(news_adapter, "_grounded", fake_grounded)
    news_adapter.get_macro_news()
    assert "India" in captured["q"]
    assert "RBI" in captured["q"] or "SEBI" in captured["q"]


def test_macro_prompt_global_when_not_focused(monkeypatch):
    captured = {}
    monkeypatch.setattr(news_adapter, "_region", lambda: "global")
    monkeypatch.setattr(news_adapter, "cached", lambda key, producer: producer())

    def fake_grounded(q):
        captured["q"] = q
        return "text", []

    monkeypatch.setattr(news_adapter, "_grounded", fake_grounded)
    news_adapter.get_macro_news()
    assert "India" not in captured["q"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
