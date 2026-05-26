"""Deterministic market metrics. Pure functions over price series — no LLM, no network.

Returns are simple daily pct changes. Beta/alpha come from an OLS regression of the
asset's excess daily returns on the benchmark's excess daily returns (CAPM), with alpha
annualized. Sharpe uses annualized excess return over annualized volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..models import QuantMetrics

TRADING_DAYS = 252


def _daily_returns(closes: pd.Series) -> pd.Series:
    return closes.pct_change().dropna()


def _window_return(closes: pd.Series, days: int) -> float | None:
    if len(closes) < 2:
        return None
    end = closes.iloc[-1]
    start_idx = max(0, len(closes) - 1 - days)
    start = closes.iloc[start_idx]
    if start == 0:
        return None
    return float(end / start - 1.0)


def annual_volatility(closes: pd.Series) -> float | None:
    r = _daily_returns(closes)
    if len(r) < 2:
        return None
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(closes: pd.Series) -> float | None:
    if len(closes) < 2:
        return None
    running_max = closes.cummax()
    drawdown = closes / running_max - 1.0
    return float(drawdown.min())


def sharpe(closes: pd.Series, risk_free_rate: float) -> float | None:
    r = _daily_returns(closes)
    if len(r) < 2:
        return None
    vol = r.std(ddof=1)
    if vol == 0:
        return None
    daily_rf = risk_free_rate / TRADING_DAYS
    excess = r.mean() - daily_rf
    return float((excess / vol) * np.sqrt(TRADING_DAYS))


def beta_alpha(
    closes: pd.Series, benchmark: pd.Series, risk_free_rate: float
) -> tuple[float | None, float | None]:
    """CAPM beta and annualized alpha via OLS on aligned daily excess returns."""
    ra = _daily_returns(closes)
    rb = _daily_returns(benchmark)
    joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(joined) < 20:
        return None, None
    daily_rf = risk_free_rate / TRADING_DAYS
    x = (joined.iloc[:, 1] - daily_rf).to_numpy()
    y = (joined.iloc[:, 0] - daily_rf).to_numpy()
    var = np.var(x, ddof=1)
    if var == 0:
        return None, None
    beta = float(np.cov(x, y, ddof=1)[0, 1] / var)
    daily_alpha = float(np.mean(y) - beta * np.mean(x))
    alpha_annual = daily_alpha * TRADING_DAYS
    return beta, alpha_annual


def compute_metrics(
    closes: pd.Series,
    benchmark: pd.Series,
    risk_free_rate: float,
) -> QuantMetrics:
    if closes.empty:
        return QuantMetrics()
    beta, alpha = beta_alpha(closes, benchmark, risk_free_rate)
    return QuantMetrics(
        last_price=float(closes.iloc[-1]),
        return_1m=_window_return(closes, 21),
        return_3m=_window_return(closes, 63),
        return_1y=_window_return(closes, TRADING_DAYS),
        annual_volatility=annual_volatility(closes),
        beta=beta,
        alpha=alpha,
        sharpe=sharpe(closes, risk_free_rate),
        max_drawdown=max_drawdown(closes),
    )
