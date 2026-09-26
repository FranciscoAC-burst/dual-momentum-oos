"""Unit tests for ``src.metrics`` on synthetic series.

Expected values are derived by hand, never by calling the implementation.
"""

import ast
import math
import sys
from pathlib import Path

# Allow running this file directly, not only via pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    TRADING_DAYS_PER_YEAR,
    calculate_all_metrics,
    calculate_cagr,
    calculate_daily_mean_return,
    calculate_excess_return,
    calculate_information_ratio,
    calculate_mar,
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_sharpe,
    calculate_skewness,
)

SOURCE_PATH = Path(__file__).resolve().parents[1] / "src" / "metrics.py"

# Annualized Sharpe of the alternating 0.02 / 0.00 series (252 values):
# mean 0.01, std(ddof=1) = 0.01 * sqrt(252/251), so Sharpe = 0.01 / std * sqrt(252) = sqrt(251).
SQRT_251 = math.sqrt(251.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def constant_returns() -> pd.Series:
    return pd.Series(np.full(TRADING_DAYS_PER_YEAR, 0.01))


@pytest.fixture
def constant_equity() -> pd.Series:
    """253 points compounding 1% per day, from 1.0 to 1.01 ** 252."""
    exponents = np.arange(TRADING_DAYS_PER_YEAR + 1, dtype=float)
    return pd.Series(np.power(1.01, exponents))


@pytest.fixture
def crash_equity() -> pd.Series:
    """50% drawdown lasting 3 days, then full recovery; starts and ends at 1.0 (CAGR 0)."""
    return pd.Series([1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0])


@pytest.fixture
def alternating_returns() -> pd.Series:
    """Alternates 2% / 0%: mean 1%, Sharpe sqrt(251) (see SQRT_251)."""
    return pd.Series([0.02, 0.00] * (TRADING_DAYS_PER_YEAR // 2))


# ---------------------------------------------------------------------------
# calculate_cagr
# ---------------------------------------------------------------------------
def test_cagr_constant_daily_return(constant_equity):
    """Annualizes over the number of points (253), not returns (252)."""
    expected = (1.01**TRADING_DAYS_PER_YEAR) ** (252.0 / 253.0) - 1.0
    assert calculate_cagr(constant_equity) == pytest.approx(expected, rel=1e-12)


def test_cagr_annualises_over_the_sample_length():
    equity = pd.Series(np.linspace(1.0, 1.21, TRADING_DAYS_PER_YEAR // 2))
    expected = 1.21 ** (252.0 / (TRADING_DAYS_PER_YEAR // 2)) - 1.0
    assert expected == pytest.approx(1.21**2 - 1.0, rel=1e-12)
    assert calculate_cagr(equity) == pytest.approx(expected, rel=1e-12)


def test_cagr_is_zero_when_final_equals_initial_capital():
    equity = pd.Series([1.0, 1.1, 0.9, 1.0])
    assert calculate_cagr(equity) == pytest.approx(0.0, abs=1e-15)


def test_cagr_negative_performance_is_negative():
    """126 points are half a year, so -19% annualizes to 0.81 ** 2 - 1."""
    equity = pd.Series(np.linspace(1.0, 0.81, TRADING_DAYS_PER_YEAR // 2))
    assert calculate_cagr(equity) == pytest.approx(0.81**2 - 1.0, rel=1e-12)


def test_cagr_measures_from_initial_capital_not_the_first_point():
    """The first point (0.99) is net of the opening commission; measuring from 1.0 penalizes it."""
    equity = pd.Series([0.99, 1.05, 1.10])
    from_capital = calculate_cagr(equity, initial_capital=1.0)
    from_first_point = calculate_cagr(equity, initial_capital=equity.iloc[0])
    assert from_capital < from_first_point


def test_cagr_higher_initial_capital_lowers_the_result():
    equity = pd.Series([1.0, 1.1, 1.2])
    assert calculate_cagr(equity, initial_capital=1.0) > calculate_cagr(
        equity, initial_capital=1.5
    )


def test_cagr_default_initial_capital_is_one():
    equity = pd.Series([1.0, 1.1, 1.2])
    assert calculate_cagr(equity) == calculate_cagr(equity, initial_capital=1.0)


def test_cagr_bankruptcy_returns_minus_one():
    equity = pd.Series([1.0, 0.5, 0.0, 0.0])
    assert calculate_cagr(equity) == -1.0


@pytest.mark.parametrize(
    "equity",
    [
        pd.Series([], dtype=float),
        pd.Series([1.0]),
    ],
    ids=["empty-series", "single-point"],
)
def test_cagr_short_series_returns_zero(equity):
    assert calculate_cagr(equity) == 0.0


def test_cagr_non_positive_initial_capital_returns_zero():
    equity = pd.Series([1.0, 1.1, 1.2])
    assert calculate_cagr(equity, initial_capital=0.0) == 0.0
    assert calculate_cagr(equity, initial_capital=-1.0) == 0.0


def test_cagr_ignores_nan_observations():
    equity = pd.Series([1.0, np.nan, 1.21])
    assert calculate_cagr(equity) == pytest.approx(
        calculate_cagr(pd.Series([1.0, 1.21])), rel=1e-12
    )


# ---------------------------------------------------------------------------
# calculate_sharpe
# ---------------------------------------------------------------------------
def test_sharpe_zero_volatility_returns_zero(constant_returns):
    assert calculate_sharpe(constant_returns) == 0.0


def test_sharpe_alternating_series_is_sqrt_251(alternating_returns):
    assert calculate_sharpe(alternating_returns) == pytest.approx(SQRT_251, rel=1e-12)


def test_sharpe_zero_mean_series_is_zero():
    returns = pd.Series([0.01, -0.01] * (TRADING_DAYS_PER_YEAR // 2))
    assert calculate_sharpe(returns) == pytest.approx(0.0, abs=1e-12)


def test_sharpe_risk_free_is_annual_and_reduces_the_ratio(alternating_returns):
    """0.0252 annual = 0.0001 daily: mean excess 0.0099, same std, so Sharpe scales by 0.99."""
    result = calculate_sharpe(alternating_returns, risk_free=0.0252)
    assert result == pytest.approx(0.99 * SQRT_251, rel=1e-12)
    assert result < calculate_sharpe(alternating_returns)


def test_sharpe_is_zero_for_numerically_constant_returns():
    """``0.1 + 0.2 != 0.3`` in floating point, so std is ~1e-17 rather than exactly 0."""
    returns = pd.Series([0.1 + 0.2, 0.3] * (TRADING_DAYS_PER_YEAR // 2))
    dispersion = returns.std(ddof=1)
    assert 0.0 < dispersion < 1e-12, "case must exercise the threshold, not exact zero"
    assert calculate_sharpe(returns) == 0.0


def test_sharpe_keeps_a_genuinely_low_volatility_series():
    """The float-noise threshold must not zero out a small but real volatility."""
    returns = pd.Series([0.0001, 0.0002] * (TRADING_DAYS_PER_YEAR // 2))
    assert calculate_sharpe(returns) == pytest.approx(0.00015 / 0.00005 * SQRT_251)


def test_sharpe_annualisation_factor_is_sqrt_252():
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015, 0.0, 0.007, -0.003])
    daily_sharpe = returns.mean() / returns.std(ddof=1)
    expected = daily_sharpe * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert calculate_sharpe(returns) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    "returns",
    [pd.Series([], dtype=float), pd.Series([0.01])],
    ids=["empty-series", "single-observation"],
)
def test_sharpe_short_series_returns_zero(returns):
    assert calculate_sharpe(returns) == 0.0


# ---------------------------------------------------------------------------
# calculate_max_drawdown
# ---------------------------------------------------------------------------
def test_max_drawdown_is_positive_on_a_50_percent_crash(crash_equity):
    assert calculate_max_drawdown(crash_equity) == pytest.approx(0.5, rel=1e-12)


def test_max_drawdown_matches_the_documented_15_percent_example():
    equity = pd.Series([100.0, 85.0, 95.0])
    assert calculate_max_drawdown(equity) == pytest.approx(0.15, rel=1e-12)


def test_max_drawdown_is_zero_for_a_monotonic_curve(constant_equity):
    result = calculate_max_drawdown(constant_equity)
    assert result == 0.0
    # A negative zero would show up in reports as "-0.0".
    assert math.copysign(1.0, result) > 0.0


def test_max_drawdown_takes_the_deepest_of_several_episodes():
    equity = pd.Series([100.0, 90.0, 100.0, 50.0, 100.0])
    assert calculate_max_drawdown(equity) == pytest.approx(0.5, rel=1e-12)


def test_max_drawdown_of_empty_series_is_zero():
    assert calculate_max_drawdown(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# calculate_max_drawdown_duration
# ---------------------------------------------------------------------------
def test_duration_counts_full_days_under_water(crash_equity):
    """Below the peak at indices 2, 3 and 4 before recovering."""
    assert calculate_max_drawdown_duration(crash_equity) == 3


def test_duration_is_zero_without_drawdown(constant_equity):
    assert calculate_max_drawdown_duration(constant_equity) == 0


def test_duration_counts_to_the_end_when_never_recovered():
    """Under water at indices 2 and 3, through the end of the sample."""
    equity = pd.Series([100.0, 120.0, 90.0, 80.0])
    assert calculate_max_drawdown_duration(equity) == 2


def test_duration_takes_the_longest_not_the_deepest_episode():
    """A shallow 3-day episode beats a 1-day 50% drop."""
    equity = pd.Series([100.0, 90.0, 95.0, 99.0, 100.0, 50.0, 100.0])
    assert calculate_max_drawdown_duration(equity) == 3
    assert calculate_max_drawdown(equity) == pytest.approx(0.5, rel=1e-12)


def test_duration_excludes_the_recovery_day():
    """Returning exactly to the peak counts as recovered."""
    equity = pd.Series([100.0, 50.0, 100.0])
    assert calculate_max_drawdown_duration(equity) == 1


def test_duration_of_empty_series_is_zero():
    assert calculate_max_drawdown_duration(pd.Series([], dtype=float)) == 0


def test_duration_returns_a_python_int(crash_equity):
    assert isinstance(calculate_max_drawdown_duration(crash_equity), int)


# ---------------------------------------------------------------------------
# calculate_mar
# ---------------------------------------------------------------------------
def test_mar_is_cagr_over_drawdown():
    assert calculate_mar(0.20, 0.10) == pytest.approx(2.0, rel=1e-12)


def test_mar_is_infinite_when_there_is_no_drawdown():
    assert math.isinf(calculate_mar(0.20, 0.0))
    assert calculate_mar(0.20, 0.0) > 0


@pytest.mark.parametrize("cagr", [0.0, -0.15], ids=["zero-cagr", "negative-cagr"])
def test_mar_is_zero_when_no_drawdown_and_no_gain(cagr):
    assert calculate_mar(cagr, 0.0) == 0.0


def test_mar_accepts_a_negative_drawdown_convention():
    assert calculate_mar(0.20, -0.10) == pytest.approx(2.0, rel=1e-12)


# ---------------------------------------------------------------------------
# calculate_daily_mean_return
# ---------------------------------------------------------------------------
def test_daily_mean_return_of_a_constant_series():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert calculate_daily_mean_return(returns) == pytest.approx(0.01, rel=1e-12)


def test_daily_mean_return_matches_the_arithmetic_mean():
    returns = pd.Series([0.02, -0.01, 0.03, 0.00, -0.04])
    assert calculate_daily_mean_return(returns) == pytest.approx(
        returns.mean(), rel=1e-12
    )


def test_daily_mean_return_of_a_zero_mean_series():
    returns = pd.Series([0.01, -0.01] * (TRADING_DAYS_PER_YEAR // 2))
    assert calculate_daily_mean_return(returns) == pytest.approx(0.0, abs=1e-15)


def test_daily_mean_return_ignores_nan_observations():
    returns = pd.Series([0.02, np.nan, 0.04])
    assert calculate_daily_mean_return(returns) == pytest.approx(0.03, rel=1e-12)


def test_daily_mean_return_of_empty_series_is_zero():
    assert calculate_daily_mean_return(pd.Series([], dtype=float)) == 0.0


# ---------------------------------------------------------------------------
# calculate_skewness
# ---------------------------------------------------------------------------
def test_skewness_of_a_symmetric_series_is_zero():
    returns = pd.Series([-0.02, -0.01, 0.0, 0.01, 0.02])
    assert calculate_skewness(returns) == pytest.approx(0.0, abs=1e-12)


def test_skewness_matches_pandas_skew():
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.15, 0.0, 0.007, -0.003])
    assert calculate_skewness(returns) == pytest.approx(returns.skew(), rel=1e-12)


def test_positive_tail_gives_positive_skewness():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])
    assert calculate_skewness(returns) > 0.0


def test_negative_tail_gives_negative_skewness():
    returns = pd.Series([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5])
    assert calculate_skewness(returns) < 0.0


def test_skewness_of_a_constant_series_is_zero():
    returns = pd.Series([0.01] * 10)
    assert calculate_skewness(returns) == 0.0


def test_skewness_of_numerically_constant_series_is_zero():
    returns = pd.Series([0.1 + 0.2, 0.3] * 5)
    assert 0.0 < returns.std(ddof=1) < 1e-12
    assert calculate_skewness(returns) == 0.0


@pytest.mark.parametrize(
    "returns",
    [pd.Series([], dtype=float), pd.Series([0.01]), pd.Series([0.01, 0.02])],
    ids=["empty", "one", "two"],
)
def test_skewness_needs_at_least_three_observations(returns):
    assert calculate_skewness(returns) == 0.0


def test_skewness_ignores_nan_observations():
    with_nan = pd.Series([0.0, 0.0, np.nan, 0.0, 0.5])
    clean = pd.Series([0.0, 0.0, 0.0, 0.5])
    assert calculate_skewness(with_nan) == pytest.approx(
        calculate_skewness(clean), rel=1e-12
    )


# ---------------------------------------------------------------------------
# calculate_excess_return
# ---------------------------------------------------------------------------
def test_excess_return_is_zero_against_an_identical_benchmark(alternating_returns):
    assert calculate_excess_return(
        alternating_returns, alternating_returns.copy()
    ) == pytest.approx(0.0, abs=1e-12)


def test_excess_return_is_the_annualised_mean_difference(alternating_returns):
    """Mean 0.01 vs a constant 0.005 benchmark: 0.005 * 252 = 1.26."""
    benchmark = pd.Series(np.full(TRADING_DAYS_PER_YEAR, 0.005))
    assert calculate_excess_return(alternating_returns, benchmark) == pytest.approx(
        1.26, rel=1e-12
    )


def test_excess_return_is_negative_when_underperforming():
    strategy = pd.Series(np.full(TRADING_DAYS_PER_YEAR, 0.001))
    benchmark = pd.Series(np.full(TRADING_DAYS_PER_YEAR, 0.002))
    assert calculate_excess_return(strategy, benchmark) == pytest.approx(
        -0.252, rel=1e-12
    )


def test_excess_return_aligns_on_the_common_index():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    strategy = pd.Series([0.01, 0.01, 0.01, 0.01], index=dates)
    benchmark = pd.Series([0.005, 0.005], index=dates[1:3])
    assert calculate_excess_return(strategy, benchmark) == pytest.approx(
        0.005 * TRADING_DAYS_PER_YEAR, rel=1e-12
    )


def test_excess_return_without_overlap_is_zero():
    strategy = pd.Series([0.01, 0.02], index=[0, 1])
    benchmark = pd.Series([0.01, 0.02], index=[5, 6])
    assert calculate_excess_return(strategy, benchmark) == 0.0


# ---------------------------------------------------------------------------
# calculate_information_ratio
# ---------------------------------------------------------------------------
def test_information_ratio_is_zero_when_tracking_error_is_zero(alternating_returns):
    assert (
        calculate_information_ratio(alternating_returns, alternating_returns.copy())
        == 0.0
    )


def test_information_ratio_is_zero_for_a_constant_active_return(alternating_returns):
    """A scalar offset leaves ~1e-19 float noise in the active std; IR must not blow up to ~1e16."""
    benchmark = alternating_returns - 0.001
    assert calculate_information_ratio(alternating_returns, benchmark) == 0.0


def test_information_ratio_against_a_constant_benchmark(alternating_returns):
    """Active return alternates 0.015 / -0.005 (mean 0.005, same std): IR = 0.5 * sqrt(251)."""
    benchmark = pd.Series(np.full(TRADING_DAYS_PER_YEAR, 0.005))
    assert calculate_information_ratio(
        alternating_returns, benchmark
    ) == pytest.approx(0.5 * SQRT_251, rel=1e-12)


def test_information_ratio_equals_sharpe_against_a_zero_benchmark(alternating_returns):
    benchmark = pd.Series(np.zeros(TRADING_DAYS_PER_YEAR))
    assert calculate_information_ratio(
        alternating_returns, benchmark
    ) == pytest.approx(calculate_sharpe(alternating_returns), rel=1e-12)


def test_information_ratio_without_enough_overlap_is_zero():
    strategy = pd.Series([0.01, 0.02], index=[0, 1])
    benchmark = pd.Series([0.005], index=[1])
    assert calculate_information_ratio(strategy, benchmark) == 0.0


# ---------------------------------------------------------------------------
# calculate_all_metrics
# ---------------------------------------------------------------------------
EXPECTED_KEYS = {
    "cagr",
    "sharpe_ratio",
    "max_drawdown",
    "max_drawdown_duration",
    "mar_ratio",
    "daily_mean_return",
    "skewness",
    "excess_return",
    "information_ratio",
}


def test_all_metrics_returns_every_key(constant_returns, constant_equity):
    metrics = calculate_all_metrics(constant_returns, constant_equity)
    assert set(metrics) == EXPECTED_KEYS


def test_all_metrics_keys_are_lowercase(constant_returns, constant_equity):
    metrics = calculate_all_metrics(constant_returns, constant_equity)
    assert all(key == key.lower() for key in metrics)


def test_all_metrics_without_benchmark_nulls_the_relative_metrics(
    constant_returns, constant_equity
):
    metrics = calculate_all_metrics(constant_returns, constant_equity)
    assert metrics["excess_return"] is None
    assert metrics["information_ratio"] is None


def test_all_metrics_values_on_the_constant_curve(constant_returns, constant_equity):
    metrics = calculate_all_metrics(constant_returns, constant_equity)
    expected_cagr = (1.01**TRADING_DAYS_PER_YEAR) ** (252.0 / 253.0) - 1.0
    assert metrics["cagr"] == round(expected_cagr, 4)
    assert metrics["sharpe_ratio"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["max_drawdown_duration"] == 0
    assert math.isinf(metrics["mar_ratio"])


def test_all_metrics_propagates_initial_capital(constant_returns, constant_equity):
    base = calculate_all_metrics(constant_returns, constant_equity)
    scaled = calculate_all_metrics(
        constant_returns, constant_equity, initial_capital=2.0
    )
    assert scaled["cagr"] < base["cagr"]
    # Drawdown is relative, so initial capital does not affect it.
    assert scaled["max_drawdown"] == base["max_drawdown"]


def test_all_metrics_includes_daily_mean_and_skewness():
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.15, 0.0, 0.007, -0.003])
    equity = (1.0 + returns).cumprod()
    metrics = calculate_all_metrics(returns, equity)

    assert metrics["daily_mean_return"] == round(
        calculate_daily_mean_return(returns), 4
    )
    assert metrics["skewness"] == round(calculate_skewness(returns), 4)


def test_all_metrics_daily_mean_and_skewness_are_rounded():
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.15, 0.0, 0.007, -0.003])
    equity = (1.0 + returns).cumprod()
    metrics = calculate_all_metrics(returns, equity)

    for key in ("daily_mean_return", "skewness"):
        assert metrics[key] == round(metrics[key], 4)


def test_all_metrics_values_on_the_crash_curve(crash_equity):
    returns = crash_equity.pct_change().dropna()
    metrics = calculate_all_metrics(returns, crash_equity)
    assert metrics["max_drawdown"] == 0.5
    assert metrics["max_drawdown_duration"] == 3
    assert metrics["cagr"] == 0.0
    # Zero CAGR with a non-zero drawdown: MAR = 0 / 0.5 = 0.0.
    assert metrics["mar_ratio"] == 0.0


def test_all_metrics_with_benchmark_fills_the_relative_metrics(alternating_returns):
    equity = (1.0 + alternating_returns).cumprod()
    benchmark = pd.Series(np.full(TRADING_DAYS_PER_YEAR, 0.005))
    metrics = calculate_all_metrics(alternating_returns, equity, benchmark)
    assert metrics["excess_return"] == round(1.26, 4)
    assert metrics["information_ratio"] == round(0.5 * SQRT_251, 4)


def test_all_metrics_with_identical_benchmark_zeroes_both(alternating_returns):
    equity = (1.0 + alternating_returns).cumprod()
    metrics = calculate_all_metrics(
        alternating_returns, equity, alternating_returns.copy()
    )
    assert metrics["excess_return"] == 0.0
    assert metrics["information_ratio"] == 0.0


def test_all_metrics_rounds_floats_to_four_decimals(alternating_returns):
    equity = (1.0 + alternating_returns).cumprod()
    benchmark = pd.Series(np.full(TRADING_DAYS_PER_YEAR, 0.005))
    metrics = calculate_all_metrics(alternating_returns, equity, benchmark)

    floats = {
        key: value
        for key, value in metrics.items()
        if isinstance(value, float) and math.isfinite(value)
    }
    assert floats, "test must inspect at least one finite float"
    assert all(value == round(value, 4) for value in floats.values())


def test_all_metrics_duration_stays_an_int(crash_equity):
    returns = crash_equity.pct_change().dropna()
    metrics = calculate_all_metrics(returns, crash_equity)
    assert isinstance(metrics["max_drawdown_duration"], int)


def test_all_metrics_does_not_mutate_its_inputs(alternating_returns):
    equity = (1.0 + alternating_returns).cumprod()
    benchmark = pd.Series(np.full(TRADING_DAYS_PER_YEAR, 0.005))
    returns_before = alternating_returns.copy()
    equity_before = equity.copy()

    calculate_all_metrics(alternating_returns, equity, benchmark)

    pd.testing.assert_series_equal(alternating_returns, returns_before)
    pd.testing.assert_series_equal(equity, equity_before)


# ---------------------------------------------------------------------------
# Vectorization contract
# ---------------------------------------------------------------------------
def test_metrics_module_has_no_explicit_loops():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    nodes = list(ast.walk(tree))

    loop_nodes = [
        node
        for node in nodes
        if isinstance(
            node,
            (
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        )
    ]
    assert loop_nodes == [], "src/metrics.py must be fully vectorized"


def test_metrics_module_does_not_iterate_row_by_row():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not attributes & {"iterrows", "itertuples", "applymap", "iteritems"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
