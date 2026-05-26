"""Portfolio tests: holdings round-trip + review math/flags with mocked prices & DB."""

import pandas as pd
import pytest

from stockresearch import config, manage
from stockresearch import portfolio as pf
from stockresearch.models import Fundamentals


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfgdir = tmp_path / "config"
    cfgdir.mkdir()
    (cfgdir / "portfolio.yaml").write_text("holdings: []\n")
    monkeypatch.setattr(manage, "project_root", lambda: tmp_path)
    monkeypatch.setattr(config, "project_root", lambda: tmp_path)
    return tmp_path


def test_add_update_remove_holding(env):
    manage.add_holding("reliance.ns", 10, 1200)
    manage.add_holding("TCS.NS", 5, 3000)
    assert {h["ticker"] for h in manage.list_holdings()} == {"RELIANCE.NS", "TCS.NS"}
    # upsert replaces
    manage.add_holding("RELIANCE.NS", 20, 1100)
    rel = next(h for h in manage.list_holdings() if h["ticker"] == "RELIANCE.NS")
    assert rel["qty"] == 20 and rel["avg_price"] == 1100
    removed, _ = manage.remove_holding("TCS.NS")
    assert removed and len(manage.list_holdings()) == 1


def test_review_math_and_flags(env, monkeypatch):
    manage.add_holding("AAA", 10, 100)   # cost 1000
    manage.add_holding("BBB", 10, 100)   # cost 1000

    prices = {"AAA": 150.0, "BBB": 50.0}
    monkeypatch.setattr(pf, "get_closes", lambda t, d: pd.Series([prices[t]]))
    monkeypatch.setattr(pf, "get_fundamentals", lambda t: Fundamentals(sector="Tech"))
    monkeypatch.setattr(pf.db, "ticker_history", lambda t: [])

    r = pf.review()
    # AAA value 1500 (+50%), BBB value 500 (-50%); total 2000
    aaa = next(h for h in r["rows"] if h["ticker"] == "AAA")
    assert aaa["value"] == 1500 and abs(aaa["pl_pct"] - 0.5) < 1e-9
    assert abs(r["total_value"] - 2000) < 1e-9
    assert abs(r["total_pl"] - 0) < 1e-9  # +500 and -500
    # AAA is 75% -> concentration flag; Tech 100% -> sector flag; <5 holdings flag
    assert any("AAA is 75%" in f for f in r["flags"])
    assert any("Tech" in f for f in r["flags"])
    assert any("holding(s)" in f for f in r["flags"])


def test_review_empty(env):
    assert pf.review() == {}
