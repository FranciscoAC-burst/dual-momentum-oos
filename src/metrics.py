"""Performance metrics for daily return series and equity curves.

Conventions
-----------
* Data are daily and annualized with ``TRADING_DAYS_PER_YEAR`` (252).
* Non-finite observations (``NaN`` / ``inf``) are dropped before computing,
  so a series produced by ``pct_change()`` can be passed as is.
* Standard deviations use ``ddof=1`` (sample, the pandas default).
* Drawdown is reported as a positive number (``0.15`` = 15% decline).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "calculate_cagr",
    "calculate_sharpe",
    "calculate_max_drawdown",
    "calculate_max_drawdown_duration",
    "calculate_mar",
    "calculate_excess_return",
    "calculate_information_ratio",
    "calculate_daily_mean_return",
    "calculate_skewness",
    "calculate_all_metrics",
]

#: Trading days per year used for all annualization.
TRADING_DAYS_PER_YEAR: int = 252

_SQRT_TRADING_DAYS: float = float(np.sqrt(TRADING_DAYS_PER_YEAR))
_ROUND_DECIMALS: int = 4

#: Standard deviations below this threshold are treated as zero. An exact
#: ``== 0.0`` check misses series that are constant only numerically (e.g.
#: ~1e-19 float residue), which would produce absurd ratios (~1e16). Real
#: daily volatility is never below ~1e-8, so this only catches noise.
_ZERO_DISPERSION: float = 1e-12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _clean_values(series: pd.Series) -> np.ndarray:
    """Return ``series`` as float64 with non-numeric/non-finite values dropped."""
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    values = numeric.to_numpy(dtype="float64")
    return values[np.isfinite(values)]


def _align_pair(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """Inner-join two return series on their index and drop non-finite pairs."""
    strategy = pd.to_numeric(pd.Series(strategy_returns), errors="coerce")
    benchmark = pd.to_numeric(pd.Series(benchmark_returns), errors="coerce")
    strategy, benchmark = strategy.align(benchmark, join="inner")

    strategy_values = strategy.to_numpy(dtype="float64")
    benchmark_values = benchmark.to_numpy(dtype="float64")
    mask = np.isfinite(strategy_values) & np.isfinite(benchmark_values)
    return strategy_values[mask], benchmark_values[mask]


def _round(value: float) -> float:
    return round(float(value), _ROUND_DECIMALS)


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------
def calculate_cagr(equity_curve: pd.Series, initial_capital: float = 1.0) -> float:
    """Compound annual growth rate of an equity curve.

    Computed as ``(equity[-1] / initial_capital) ** (252 / n) - 1``, where
    ``n`` is the number of points in the curve.

    CAGR is measured against ``initial_capital`` rather than ``equity[0]``
    on purpose: the first point may already reflect the opening commission
    (``equity[0] = 1 - cost``), and dividing by it would cancel that cost.

    Returns ``0.0`` if the curve has ``<= 1`` point or
    ``initial_capital <= 0``, and ``-1.0`` if final equity is ``<= 0``
    (bankruptcy).
    """
    values = _clean_values(equity_curve)
    total_days = values.size
    if total_days <= 1:
        return 0.0
    if initial_capital <= 0.0:
        return 0.0

    final = values[-1]
    if final <= 0.0:
        return -1.0

    exponent = TRADING_DAYS_PER_YEAR / total_days
    return float((final / initial_capital) ** exponent - 1.0)


def calculate_sharpe(returns: pd.Series, risk_free: float = 0.0) -> float:
    """Annualized Sharpe ratio of daily returns.

    ``risk_free`` is an **annual** rate, converted to daily by dividing by
    252 before subtracting it. The result is annualized by ``sqrt(252)``.
    Returns ``0.0`` with fewer than two observations or (numerically) zero
    volatility.
    """
    values = _clean_values(returns)
    if values.size < 2:
        return 0.0

    daily_risk_free = float(risk_free) / TRADING_DAYS_PER_YEAR
    excess = values - daily_risk_free

    volatility = excess.std(ddof=1)
    if volatility < _ZERO_DISPERSION or not np.isfinite(volatility):
        return 0.0

    return float(excess.mean() / volatility * _SQRT_TRADING_DAYS)


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline, returned as a **positive** number.

    ``0.15`` means a 15% drawdown; a curve that never declines returns
    ``0.0``.
    """
    values = _clean_values(equity_curve)
    if values.size == 0:
        return 0.0

    running_max = np.maximum.accumulate(values)
    positive_peak = running_max > 0.0

    ratio = np.ones_like(values)
    np.divide(values, running_max, out=ratio, where=positive_peak)
    drawdowns = np.where(positive_peak, ratio - 1.0, 0.0)

    # ``abs`` avoids returning ``-0.0`` when the curve never declines.
    return float(abs(np.minimum(drawdowns.min(), 0.0)))


def calculate_max_drawdown_duration(equity_curve: pd.Series) -> int:
    """Length of the longest drawdown, in consecutive trading days underwater.

    A day is underwater when equity is strictly below its running maximum.
    Neither the peak day nor the recovery day is counted; an unrecovered
    drawdown runs to the end of the series. Returns ``0`` if the curve is
    never underwater.
    """
    values = _clean_values(equity_curve)
    if values.size == 0:
        return 0

    running_max = np.maximum.accumulate(values)
    underwater = values < running_max
    if not underwater.any():
        return 0

    # Each non-underwater day bumps the episode id, so consecutive
    # underwater days share the same id.
    episode_id = np.cumsum(~underwater)
    lengths = np.bincount(episode_id[underwater])
    return int(lengths.max())


def calculate_mar(cagr: float, max_dd: float) -> float:
    """MAR ratio: ``cagr / |max_dd|``.

    Accepts drawdown with either sign convention. If ``max_dd`` is zero,
    returns ``inf`` when ``cagr > 0`` and ``0.0`` otherwise.
    """
    cagr_value = float(cagr)
    drawdown = abs(float(max_dd))

    if drawdown == 0.0:
        return float("inf") if cagr_value > 0.0 else 0.0

    return float(cagr_value / drawdown)


def calculate_excess_return(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Annualized difference between mean daily strategy and benchmark returns.

    Series are aligned on their common index. Returns ``0.0`` if they do
    not overlap.
    """
    strategy, benchmark = _align_pair(strategy_returns, benchmark_returns)
    if strategy.size == 0:
        return 0.0

    return float((strategy.mean() - benchmark.mean()) * TRADING_DAYS_PER_YEAR)


def calculate_information_ratio(
    strategy_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Annualized information ratio versus a benchmark.

    ``mean(active) / std(active) * sqrt(252)``, where ``active`` is the
    daily strategy-minus-benchmark return on the common index. Returns
    ``0.0`` with fewer than two overlapping observations or (numerically)
    zero tracking error.
    """
    strategy, benchmark = _align_pair(strategy_returns, benchmark_returns)
    if strategy.size < 2:
        return 0.0

    active = strategy - benchmark
    tracking_error = active.std(ddof=1)
    if tracking_error < _ZERO_DISPERSION or not np.isfinite(tracking_error):
        return 0.0

    return float(active.mean() / tracking_error * _SQRT_TRADING_DAYS)


def calculate_daily_mean_return(returns: pd.Series) -> float:
    """Arithmetic mean of daily returns; ``0.0`` for an empty series."""
    values = _clean_values(returns)
    if values.size == 0:
        return 0.0
    return float(values.mean())


def calculate_skewness(returns: pd.Series) -> float:
    """Sample skewness of daily returns.

    Uses pandas ``Series.skew()`` (adjusted Fisher-Pearson). Returns ``0.0``
    with fewer than three observations or (numerically) zero dispersion,
    where skewness is undefined.
    """
    values = _clean_values(returns)
    if values.size < 3:
        return 0.0

    volatility = values.std(ddof=1)
    if volatility < _ZERO_DISPERSION or not np.isfinite(volatility):
        return 0.0

    skew = pd.Series(values).skew()
    if not np.isfinite(skew):
        return 0.0
    return float(skew)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------
def calculate_all_metrics(
    returns: pd.Series,
    equity_curve: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    initial_capital: float = 1.0,
) -> dict:
    """Compute all metrics and return them in a dict.

    Float values are rounded to 4 decimals; ``max_drawdown_duration`` is an
    ``int`` (days). ``initial_capital`` is passed to :func:`calculate_cagr`
    so the opening commission penalizes CAGR. The MAR ratio is computed from
    the unrounded CAGR and drawdown, then rounded. ``excess_return`` and
    ``information_ratio`` are ``None`` when ``benchmark_returns`` is
    ``None``.
    """
    cagr = calculate_cagr(equity_curve, initial_capital)
    max_drawdown = calculate_max_drawdown(equity_curve)

    metrics: dict = {
        "cagr": _round(cagr),
        "sharpe_ratio": _round(calculate_sharpe(returns)),
        "max_drawdown": _round(max_drawdown),
        "max_drawdown_duration": calculate_max_drawdown_duration(equity_curve),
        "mar_ratio": _round(calculate_mar(cagr, max_drawdown)),
        "daily_mean_return": _round(calculate_daily_mean_return(returns)),
        "skewness": _round(calculate_skewness(returns)),
        "excess_return": None,
        "information_ratio": None,
    }

    if benchmark_returns is not None:
        metrics["excess_return"] = _round(
            calculate_excess_return(returns, benchmark_returns)
        )
        metrics["information_ratio"] = _round(
            calculate_information_ratio(returns, benchmark_returns)
        )

    return metrics
