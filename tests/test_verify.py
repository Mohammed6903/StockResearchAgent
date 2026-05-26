"""Tests for the fundamentals cross-check (pure comparison + adapter mapping, no network)."""

from stockresearch.data import verify
from stockresearch.models import Fundamentals


def test_no_warning_when_sources_agree():
    ours = Fundamentals(pe_ratio=37.6, profit_margin=0.271, roe=1.50)
    other = {"pe_ratio": 37.9, "profit_margin": 0.268, "roe": 1.46}
    assert verify.compare_fundamentals(ours, other, tolerance=0.10) == []


def test_warns_on_material_disagreement():
    ours = Fundamentals(pe_ratio=37.6, pb_ratio=50.0)
    other = {"pe_ratio": 25.0, "pb_ratio": 50.5}  # P/E differs ~33%, P/B agrees
    warns = verify.compare_fundamentals(ours, other, tolerance=0.10)
    assert len(warns) == 1
    assert "P/E mismatch" in warns[0]
    assert "Yahoo 37.6" in warns[0]


def test_skips_fields_missing_in_either_source():
    ours = Fundamentals(pe_ratio=37.6)  # only P/E present on our side
    other = {"roe": 0.99, "roa": 0.5}   # no P/E on the other side
    assert verify.compare_fundamentals(ours, other) == []


def test_tolerance_boundary():
    ours = Fundamentals(pe_ratio=100.0)
    other = {"pe_ratio": 111.5}  # ~10.3% diff -> over a 10% tolerance
    assert verify.compare_fundamentals(ours, other, tolerance=0.10)
    assert verify.compare_fundamentals(ours, other, tolerance=0.15) == []


def test_verify_no_op_without_api_key(monkeypatch):
    monkeypatch.delenv("ALPHAVANTAGE_API_KEY", raising=False)
    notes = verify.verify_fundamentals("AAPL", Fundamentals(pe_ratio=10))
    assert notes == ["fundamentals not cross-checked (no 2nd source / API key)"]


def test_alphavantage_overview_maps_and_parses(monkeypatch):
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "demo")
    raw = {
        "Symbol": "AAPL", "PERatio": "37.65", "ProfitMargin": "0.271",
        "ReturnOnEquityTTM": "1.50", "DividendYield": "None", "PEGRatio": "-",
    }
    monkeypatch.setattr(verify, "cached", lambda key, producer: raw)
    out = verify.alphavantage_overview("AAPL")
    assert out["pe_ratio"] == 37.65
    assert out["profit_margin"] == 0.271
    assert out["roe"] == 1.50
    assert "dividend_yield" not in out  # "None" filtered
    assert "peg_ratio" not in out       # "-" filtered
