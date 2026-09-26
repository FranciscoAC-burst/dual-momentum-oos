"""Smoke tests for ``src.plots.generate_tearsheet``.

Checks that a non-empty PNG is written and optional branches do not raise; visuals are not checked.
"""

import sys
from pathlib import Path

# Allow running this file directly, not only via pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.plots import generate_tearsheet


def _synthetic_returns(periods: int = 504, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    values = rng.normal(loc=0.0005, scale=0.01, size=periods)
    index = pd.bdate_range("2021-01-04", periods=periods)
    return pd.Series(values, index=index, name="net_returns")


def _equity_from_returns(returns: pd.Series) -> pd.Series:
    return (1.0 + returns).cumprod().rename("equity_curve")


def test_tearsheet_png_is_created_with_content(tmp_path):
    returns = _synthetic_returns()
    equity = _equity_from_returns(returns)
    output = tmp_path / "tearsheet.png"

    generate_tearsheet(equity, returns, output_path=str(output))

    assert output.exists()
    assert output.stat().st_size > 0


def test_tearsheet_with_benchmark(tmp_path):
    returns = _synthetic_returns(seed=1)
    equity = _equity_from_returns(returns)
    bench_returns = _synthetic_returns(seed=2)
    bench_equity = _equity_from_returns(bench_returns)
    output = tmp_path / "tearsheet_bench.png"

    generate_tearsheet(
        equity,
        returns,
        benchmark_equity=bench_equity,
        benchmark_returns=bench_returns,
        output_path=str(output),
    )

    assert output.exists()
    assert output.stat().st_size > 0


def test_tearsheet_without_benchmark_does_not_raise(tmp_path):
    returns = _synthetic_returns(seed=7)
    equity = _equity_from_returns(returns)
    output = tmp_path / "no_bench.png"

    generate_tearsheet(equity, returns, benchmark_equity=None, output_path=str(output))
    assert output.stat().st_size > 0


def test_tearsheet_creates_missing_output_directory(tmp_path):
    returns = _synthetic_returns(seed=3)
    equity = _equity_from_returns(returns)
    output = tmp_path / "nested" / "dir" / "tearsheet.png"

    generate_tearsheet(equity, returns, output_path=str(output))

    assert output.exists()
    assert output.stat().st_size > 0


def test_tearsheet_with_high_return_triggers_semilog(tmp_path):
    """A total return above +300% switches to a semilog scale."""
    index = pd.bdate_range("2020-01-01", periods=300)
    returns = pd.Series(np.full(300, 0.01), index=index)
    equity = _equity_from_returns(returns)  # ~19x by the end
    output = tmp_path / "semilog.png"

    generate_tearsheet(equity, returns, output_path=str(output))
    assert output.stat().st_size > 0


def test_tearsheet_handles_a_short_series(tmp_path):
    """Shorter than a full rolling-Sharpe window."""
    index = pd.bdate_range("2022-01-03", periods=5)
    returns = pd.Series([0.01, -0.02, 0.015, 0.0, -0.005], index=index)
    equity = _equity_from_returns(returns)
    output = tmp_path / "short.png"

    generate_tearsheet(equity, returns, output_path=str(output))
    assert output.stat().st_size > 0


def test_tearsheet_handles_a_flat_series(tmp_path):
    """All-zero returns (zero variance) must not break the KDE or percentiles."""
    index = pd.bdate_range("2022-01-03", periods=260)
    returns = pd.Series(np.zeros(260), index=index)
    equity = _equity_from_returns(returns)
    output = tmp_path / "flat.png"

    generate_tearsheet(equity, returns, output_path=str(output))
    assert output.stat().st_size > 0


def test_tearsheet_uses_the_agg_backend():
    import matplotlib

    assert matplotlib.get_backend().lower() == "agg"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
