"""Unit tests for deterministic quant metrics using synthetic series with known answers."""

import numpy as np
import pandas as pd

from stockresearch.quant import metrics as m


def _series(values):
    idx = pd.date_range("2024-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx, dtype=float)


def test_window_return_simple():
    s = _series([100, 110])
    assert abs(m._window_return(s, 1) - 0.1) < 1e-9


def test_volatility_constant_growth_is_zero():
    # Constant 1% daily growth -> zero return variance -> zero volatility.
    closes = _series([100 * (1.01 ** i) for i in range(50)])
    assert m.annual_volatility(closes) < 1e-9


def test_max_drawdown():
    s = _series([100, 120, 60, 90])
    # peak 120 -> trough 60 = -50%
    assert abs(m.max_drawdown(s) - (-0.5)) < 1e-9


def test_beta_one_when_asset_equals_benchmark():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, 200)
    prices = 100 * np.cumprod(1 + rets)
    s = _series(prices)
    beta, alpha = m.beta_alpha(s, s.copy(), risk_free_rate=0.0)
    assert abs(beta - 1.0) < 1e-6
    assert abs(alpha) < 1e-6


def test_beta_two_when_asset_doubles_benchmark_moves():
    rng = np.random.default_rng(1)
    b_rets = rng.normal(0.0, 0.01, 300)
    a_rets = 2.0 * b_rets  # asset moves exactly 2x benchmark, no idiosyncratic noise
    bench = _series(100 * np.cumprod(1 + b_rets))
    asset = _series(100 * np.cumprod(1 + a_rets))
    beta, alpha = m.beta_alpha(asset, bench, risk_free_rate=0.0)
    assert abs(beta - 2.0) < 0.05


def test_sharpe_positive_for_steady_gains():
    closes = _series([100 * (1.001 ** i) for i in range(100)])
    sr = m.sharpe(closes, risk_free_rate=0.0)
    assert sr is not None and sr > 0


def test_empty_series_yields_empty_metrics():
    qm = m.compute_metrics(pd.Series(dtype=float), pd.Series(dtype=float), 0.045)
    assert qm.beta is None and qm.last_price is None


def test_insufficient_overlap_returns_none_beta():
    s = _series([100, 101, 102])
    beta, alpha = m.beta_alpha(s, s.copy(), 0.0)
    assert beta is None and alpha is None
