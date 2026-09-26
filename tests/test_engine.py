"""Unit tests for ``src.engine.VectorEngine``.

Expected values are derived by hand from literal prices and weights, never by calling the engine.
"""

import ast
import sys
from pathlib import Path

# Allow running this file directly, not only via pytest.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from src.engine import VectorEngine
from src.metrics import calculate_all_metrics

SOURCE_PATH = Path(__file__).resolve().parents[1] / "src" / "engine.py"

RTOL = 1e-12


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------
def make_index(periods: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=periods)


def make_ohlc(close, open_=None, columns=("A",)) -> dict:
    """Build the OHLC dict; ``open`` defaults to ``close`` (no overnight gap)."""
    names = list(columns)
    close_frame = pd.DataFrame(close, columns=names, dtype="float64")
    close_frame.index = make_index(len(close_frame))

    if open_ is None:
        open_frame = close_frame.copy()
    else:
        open_frame = pd.DataFrame(open_, columns=names, dtype="float64")
        open_frame.index = close_frame.index

    return {"open": open_frame, "close": close_frame}


def make_weights(values, index, columns=("A",)) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=list(columns), dtype="float64")


def constant_weight_case(weight: float, periods: int = 4, price: float = 100.0):
    """Flat prices and a constant weight: isolates costs from market P&L."""
    ohlc = make_ohlc([price] * periods)
    weights = make_weights([weight] * periods, ohlc["close"].index)
    return ohlc, weights


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
def test_default_construction():
    engine = VectorEngine()
    assert engine.cost == 0.0005
    assert engine.timing == "close"
    assert engine.borrow_rate_annual == 0.0
    assert engine.borrow_rate_daily == 0.0


def test_borrow_rate_is_divided_by_252():
    engine = VectorEngine(borrow_rate_annual=0.0252)
    assert engine.borrow_rate_daily == pytest.approx(0.0252 / 252.0, rel=RTOL)


@pytest.mark.parametrize("timing", ["close", "open"])
def test_valid_timings_are_accepted(timing):
    assert VectorEngine(timing=timing).timing == timing


@pytest.mark.parametrize(
    "timing",
    ["mid", "Close", "OPEN", "", "vwap", None],
    ids=["mid", "capitalized", "uppercase", "empty", "vwap", "none"],
)
def test_invalid_timing_raises_assertion(timing):
    with pytest.raises(AssertionError):
        VectorEngine(timing=timing)


# ---------------------------------------------------------------------------
# Input validation and alignment
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dropped", ["open", "close"], ids=["no-open", "no-close"])
def test_missing_ohlc_key_raises_assertion(dropped):
    ohlc, weights = constant_weight_case(1.0)
    del ohlc[dropped]
    with pytest.raises(AssertionError):
        VectorEngine().run(ohlc, weights)


def test_extra_ohlc_keys_are_ignored():
    ohlc, weights = constant_weight_case(1.0)
    enriched = dict(ohlc)
    enriched["high"] = ohlc["close"] * 1.01
    enriched["volume"] = ohlc["close"] * 0.0

    engine = VectorEngine(cost=0.0)
    base = engine.run(ohlc, weights)["net_returns"]
    with_extras = engine.run(enriched, weights)["net_returns"]
    pd.testing.assert_series_equal(base, with_extras)


def test_mismatched_ohlc_index_raises_assertion():
    ohlc, weights = constant_weight_case(1.0)
    ohlc["open"] = ohlc["open"].set_axis(make_index(len(ohlc["open"])).shift(1, freq="D"))
    with pytest.raises(AssertionError):
        VectorEngine().run(ohlc, weights)


def test_mismatched_ohlc_columns_raises_assertion():
    ohlc, weights = constant_weight_case(1.0)
    ohlc["open"] = ohlc["open"].rename(columns={"A": "Z"})
    with pytest.raises(AssertionError):
        VectorEngine().run(ohlc, weights)


def test_non_dataframe_weights_raise_assertion():
    ohlc, _ = constant_weight_case(1.0)
    with pytest.raises(AssertionError):
        VectorEngine().run(ohlc, pd.Series([1.0, 1.0, 1.0, 1.0]))


def test_weights_are_reindexed_to_the_close_frame():
    """Reindexed: B = [0, 1, 1, 0, 0], A = 0; column C and the out-of-range date drop out."""
    dates = make_index(5)
    close = pd.DataFrame(
        {
            "A": [100.0, 100.0, 100.0, 100.0, 100.0],
            "B": [100.0, 110.0, 121.0, 133.1, 146.41],
        },
        index=dates,
    )
    ohlc = {"open": close.copy(), "close": close}

    sparse_weights = pd.DataFrame(
        {"B": [1.0, 1.0, 1.0], "C": [5.0, 5.0, 5.0]},
        index=[dates[1], dates[2], pd.Timestamp("2024-02-15")],
    )

    result = VectorEngine(cost=0.0).run(ohlc, sparse_weights)

    # Held position w.shift(1) = [0, 0, 1, 1, 0] applied to B's returns.
    expected_returns = np.array([0.0, 0.0, 121.0 / 110.0 - 1.0, 133.1 / 121.0 - 1.0, 0.0])
    np.testing.assert_allclose(
        result["net_returns"].to_numpy(), expected_returns, rtol=RTOL
    )
    np.testing.assert_allclose(
        result["turnover"].to_numpy(), np.array([0.0, 1.0, 0.0, 1.0, 0.0]), rtol=RTOL
    )
    assert result["equity_curve"].index.equals(dates)


def test_output_series_share_the_close_index():
    ohlc, weights = constant_weight_case(1.0, periods=6)
    result = VectorEngine().run(ohlc, weights)
    index = ohlc["close"].index
    for key in ("equity_curve", "net_returns", "turnover"):
        assert len(result[key]) == len(index)
        assert result[key].index.equals(index)


def test_run_does_not_mutate_its_inputs():
    ohlc, weights = constant_weight_case(2.0)
    ohlc_before = {"open": ohlc["open"].copy(), "close": ohlc["close"].copy()}
    weights_before = weights.copy()

    VectorEngine(cost=0.001, borrow_rate_annual=0.03).run(ohlc, weights)

    pd.testing.assert_frame_equal(ohlc["open"], ohlc_before["open"])
    pd.testing.assert_frame_equal(ohlc["close"], ohlc_before["close"])
    pd.testing.assert_frame_equal(weights, weights_before)


# ---------------------------------------------------------------------------
# Zero weights
# ---------------------------------------------------------------------------
def test_zero_weights_produce_flat_equity_and_zero_costs():
    ohlc = make_ohlc([100.0, 130.0, 90.0, 115.0, 101.0])
    weights = make_weights([0.0] * 5, ohlc["close"].index)

    result = VectorEngine(cost=0.01, borrow_rate_annual=0.10).run(ohlc, weights)

    np.testing.assert_allclose(result["net_returns"].to_numpy(), np.zeros(5), atol=0.0)
    np.testing.assert_allclose(result["turnover"].to_numpy(), np.zeros(5), atol=0.0)
    np.testing.assert_allclose(result["equity_curve"].to_numpy(), np.ones(5), atol=0.0)

    metrics = result["metrics"]
    assert metrics["cagr"] == 0.0
    assert metrics["max_drawdown"] == 0.0
    assert metrics["max_drawdown_duration"] == 0
    assert metrics["sharpe_ratio"] == 0.0


def test_zero_weights_are_flat_even_with_volatile_prices_and_open_timing():
    ohlc = make_ohlc([100.0, 130.0, 90.0], open_=[95.0, 140.0, 80.0])
    weights = make_weights([0.0] * 3, ohlc["close"].index)

    result = VectorEngine(timing="open", cost=0.05).run(ohlc, weights)
    np.testing.assert_allclose(result["equity_curve"].to_numpy(), np.ones(3), atol=0.0)


# ---------------------------------------------------------------------------
# Timing: close vs open with overnight gaps
# ---------------------------------------------------------------------------
# Closes compound exactly 10% per day.
GAP_CLOSE = [100.0, 110.0, 121.0, 133.1]
# Opens gap away from the previous close.
GAP_OPEN = [100.0, 105.0, 115.0, 130.0]


def test_close_timing_captures_the_full_close_to_close_return():
    ohlc = make_ohlc(GAP_CLOSE, open_=GAP_OPEN)
    weights = make_weights([1.0] * 4, ohlc["close"].index)

    result = VectorEngine(timing="close", cost=0.0).run(ohlc, weights)

    expected = np.array(
        [
            0.0,
            110.0 / 100.0 - 1.0,
            121.0 / 110.0 - 1.0,
            133.1 / 121.0 - 1.0,
        ]
    )
    np.testing.assert_allclose(result["net_returns"].to_numpy(), expected, rtol=RTOL)


def test_open_timing_splits_overnight_and_intraday_legs():
    """The previous position earns the overnight gap; the new one only the intraday leg."""
    ohlc = make_ohlc(GAP_CLOSE, open_=GAP_OPEN)
    weights = make_weights([1.0] * 4, ohlc["close"].index)

    result = VectorEngine(timing="open", cost=0.0).run(ohlc, weights)

    expected = np.array(
        [
            # t0: shift(2) and shift(1) are both 0, and open == close.
            0.0,
            # t1: no previous position yet, only the intraday leg 105 -> 110.
            110.0 / 105.0 - 1.0,
            # t2: gap 110 -> 115 on the previous position, intraday 115 -> 121.
            (115.0 / 110.0 - 1.0) + (121.0 / 115.0 - 1.0),
            # t3: gap 121 -> 130, intraday 130 -> 133.1.
            (130.0 / 121.0 - 1.0) + (133.1 / 130.0 - 1.0),
        ]
    )
    np.testing.assert_allclose(result["net_returns"].to_numpy(), expected, rtol=RTOL)


def test_open_and_close_timing_diverge_with_overnight_gaps():
    ohlc = make_ohlc(GAP_CLOSE, open_=GAP_OPEN)
    weights = make_weights([1.0] * 4, ohlc["close"].index)

    by_close = VectorEngine(timing="close", cost=0.0).run(ohlc, weights)
    by_open = VectorEngine(timing="open", cost=0.0).run(ohlc, weights)

    close_returns = by_close["net_returns"].to_numpy()
    open_returns = by_open["net_returns"].to_numpy()

    assert not np.allclose(close_returns, open_returns)
    assert by_close["equity_curve"].iloc[-1] != pytest.approx(
        by_open["equity_curve"].iloc[-1]
    )


def test_open_timing_equals_close_timing_without_overnight_gaps():
    close = [100.0, 110.0, 121.0, 133.1]
    open_without_gaps = [100.0, 100.0, 110.0, 121.0]
    ohlc = make_ohlc(close, open_=open_without_gaps)
    weights = make_weights([1.0] * 4, ohlc["close"].index)

    by_close = VectorEngine(timing="close", cost=0.0).run(ohlc, weights)
    by_open = VectorEngine(timing="open", cost=0.0).run(ohlc, weights)

    np.testing.assert_allclose(
        by_open["net_returns"].to_numpy(),
        by_close["net_returns"].to_numpy(),
        rtol=RTOL,
    )


def test_turnover_is_independent_of_timing():
    """Turnover is measured on target weights, not on execution."""
    ohlc = make_ohlc(GAP_CLOSE, open_=GAP_OPEN)
    weights = make_weights([0.0, 1.0, 0.5, 1.0], ohlc["close"].index)

    by_close = VectorEngine(timing="close").run(ohlc, weights)
    by_open = VectorEngine(timing="open").run(ohlc, weights)
    pd.testing.assert_series_equal(by_close["turnover"], by_open["turnover"])


# ---------------------------------------------------------------------------
# Turnover and transaction costs
# ---------------------------------------------------------------------------
def test_entering_a_full_position_costs_exactly_cost():
    cost = 0.001
    ohlc, _ = constant_weight_case(0.0)
    weights = make_weights([0.0, 1.0, 1.0, 1.0], ohlc["close"].index)

    result = VectorEngine(cost=cost).run(ohlc, weights)

    np.testing.assert_allclose(
        result["turnover"].to_numpy(), np.array([0.0, 1.0, 0.0, 0.0]), rtol=RTOL
    )
    np.testing.assert_allclose(
        result["net_returns"].to_numpy(),
        np.array([0.0, -cost, 0.0, 0.0]),
        rtol=RTOL,
    )
    assert 1.0 - result["equity_curve"].iloc[-1] == pytest.approx(cost, rel=RTOL)


def test_round_trip_is_charged_on_both_legs():
    cost = 0.002
    ohlc, _ = constant_weight_case(0.0, periods=5)
    weights = make_weights([0.0, 1.0, 1.0, 0.0, 0.0], ohlc["close"].index)

    result = VectorEngine(cost=cost).run(ohlc, weights)

    np.testing.assert_allclose(
        result["turnover"].to_numpy(),
        np.array([0.0, 1.0, 0.0, 1.0, 0.0]),
        rtol=RTOL,
    )
    assert 1.0 - result["equity_curve"].iloc[-1] == pytest.approx(
        1.0 - (1.0 - cost) ** 2, rel=RTOL
    )


def test_turnover_sums_absolute_changes_across_assets():
    """Building A from cash on day 0 moves 1.0; rotating A into B moves 2.0."""
    dates = make_index(3)
    close = pd.DataFrame(
        {"A": [100.0, 100.0, 100.0], "B": [100.0, 100.0, 100.0]}, index=dates
    )
    ohlc = {"open": close.copy(), "close": close}
    weights = pd.DataFrame(
        {"A": [1.0, 0.0, 0.0], "B": [0.0, 1.0, 1.0]}, index=dates, dtype="float64"
    )

    result = VectorEngine(cost=0.001).run(ohlc, weights)
    np.testing.assert_allclose(
        result["turnover"].to_numpy(), np.array([1.0, 2.0, 0.0]), rtol=RTOL
    )


def test_the_initial_portfolio_is_charged():
    """Building ``|w_0|`` from cash is charged on day 0: equity starts at ``1 - cost``."""
    ohlc, weights = constant_weight_case(1.0)
    result = VectorEngine(cost=0.01).run(ohlc, weights)
    assert result["turnover"].iloc[0] == pytest.approx(1.0, rel=RTOL)
    assert result["net_returns"].iloc[0] == pytest.approx(-0.01, rel=RTOL)
    assert result["equity_curve"].iloc[0] == pytest.approx(0.99, rel=RTOL)


def test_opening_commission_is_reflected_in_the_equity_level():
    """The N-point equity curve starts at the first NAV, already net of the opening cost."""
    ohlc = make_ohlc([100.0, 101.0, 103.0, 102.0, 105.0])
    weights = make_weights([1.0] * 5, ohlc["close"].index)

    free = VectorEngine(cost=0.0).run(ohlc, weights)["equity_curve"]
    charged = VectorEngine(cost=0.01).run(ohlc, weights)["equity_curve"]

    assert charged.iloc[0] == pytest.approx(free.iloc[0] * (1.0 - 0.01), rel=RTOL)
    assert (charged.to_numpy() < free.to_numpy()).all()


def test_opening_commission_lowers_the_cagr():
    """CAGR is measured from an initial capital of 1.0, so the opening cost lowers it."""
    idx = pd.bdate_range("2024-01-02", periods=252)
    close = pd.DataFrame(
        np.power(1.0005, np.arange(252.0)), index=idx, columns=["A"]
    )
    ohlc = {"open": close.copy(), "close": close}
    weights = pd.DataFrame([1.0] * 252, index=idx, columns=["A"], dtype="float64")

    cagr_free = VectorEngine(cost=0.0).run(ohlc, weights)["metrics"]["cagr"]
    cagr_charged = VectorEngine(cost=0.005).run(ohlc, weights)["metrics"]["cagr"]

    assert cagr_charged < cagr_free


def test_zero_cost_engine_charges_nothing():
    ohlc, _ = constant_weight_case(0.0)
    weights = make_weights([0.0, 1.0, 0.0, 1.0], ohlc["close"].index)
    result = VectorEngine(cost=0.0).run(ohlc, weights)
    np.testing.assert_allclose(result["equity_curve"].to_numpy(), np.ones(4), atol=0.0)


# ---------------------------------------------------------------------------
# Leverage financing cost
# ---------------------------------------------------------------------------
def test_leverage_of_two_pays_the_daily_borrow_rate():
    """Weight 2.0 borrows 1.0 and pays ``borrow_rate_daily`` on it each day."""
    borrow_rate_annual = 0.0252
    borrow_rate_daily = borrow_rate_annual / 252.0
    ohlc, weights = constant_weight_case(2.0, periods=4)

    result = VectorEngine(
        cost=0.0, borrow_rate_annual=borrow_rate_annual
    ).run(ohlc, weights)

    # The day-0 build is free at cost=0.0; financing accrues from day 1 on the held position.
    np.testing.assert_allclose(
        result["turnover"].to_numpy(), np.array([2.0, 0.0, 0.0, 0.0]), rtol=RTOL
    )
    np.testing.assert_allclose(
        result["net_returns"].to_numpy(),
        np.array([0.0, -borrow_rate_daily, -borrow_rate_daily, -borrow_rate_daily]),
        rtol=RTOL,
    )
    np.testing.assert_allclose(
        result["equity_curve"].to_numpy(),
        np.array(
            [
                1.0,
                (1.0 - borrow_rate_daily),
                (1.0 - borrow_rate_daily) ** 2,
                (1.0 - borrow_rate_daily) ** 3,
            ]
        ),
        rtol=RTOL,
    )


def test_fully_invested_portfolio_pays_no_financing():
    ohlc, weights = constant_weight_case(1.0, periods=4)
    result = VectorEngine(cost=0.0, borrow_rate_annual=0.50).run(ohlc, weights)
    np.testing.assert_allclose(result["equity_curve"].to_numpy(), np.ones(4), atol=0.0)


def test_financing_scales_with_the_borrowed_capital():
    """Weight 3.0 borrows 2.0, twice as much as weight 2.0."""
    borrow_rate_daily = 0.0252 / 252.0
    ohlc, weights = constant_weight_case(3.0, periods=3)
    result = VectorEngine(cost=0.0, borrow_rate_annual=0.0252).run(ohlc, weights)
    np.testing.assert_allclose(
        result["net_returns"].to_numpy(),
        np.array([0.0, -2.0 * borrow_rate_daily, -2.0 * borrow_rate_daily]),
        rtol=RTOL,
    )


def test_financing_uses_gross_not_net_exposure():
    """A market-neutral 1.5 / -1.5 book (gross 3.0) borrows 2.0, not 0.0."""
    borrow_rate_daily = 0.0252 / 252.0
    dates = make_index(3)
    close = pd.DataFrame(
        {"A": [100.0, 100.0, 100.0], "B": [100.0, 100.0, 100.0]}, index=dates
    )
    ohlc = {"open": close.copy(), "close": close}
    weights = pd.DataFrame(
        {"A": [1.5, 1.5, 1.5], "B": [-1.5, -1.5, -1.5]}, index=dates, dtype="float64"
    )

    result = VectorEngine(cost=0.0, borrow_rate_annual=0.0252).run(ohlc, weights)
    np.testing.assert_allclose(
        result["net_returns"].to_numpy(),
        np.array([0.0, -2.0 * borrow_rate_daily, -2.0 * borrow_rate_daily]),
        rtol=RTOL,
    )


def test_no_financing_when_borrow_rate_is_zero():
    ohlc, weights = constant_weight_case(5.0, periods=4)
    result = VectorEngine(cost=0.0, borrow_rate_annual=0.0).run(ohlc, weights)
    np.testing.assert_allclose(result["equity_curve"].to_numpy(), np.ones(4), atol=0.0)


# ---------------------------------------------------------------------------
# Bankruptcy (absorbing barrier)
# ---------------------------------------------------------------------------
def test_minus_105_percent_return_wipes_the_account_out():
    """3.0x leverage on a 35% drop: 3.0 * -0.35 = -105%."""
    ohlc = make_ohlc([100.0, 100.0, 65.0, 130.0, 130.0])
    weights = make_weights([3.0] * 5, ohlc["close"].index)

    result = VectorEngine(cost=0.0, borrow_rate_annual=0.0).run(ohlc, weights)

    assert result["net_returns"].iloc[2] == pytest.approx(-1.05, rel=RTOL)
    np.testing.assert_allclose(
        result["equity_curve"].to_numpy(),
        np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
        atol=0.0,
    )


def test_a_wiped_out_account_does_not_revive_on_a_rally():
    ohlc = make_ohlc([100.0, 100.0, 65.0, 130.0, 130.0])
    weights = make_weights([3.0] * 5, ohlc["close"].index)

    result = VectorEngine(cost=0.0).run(ohlc, weights)

    # Post-bankruptcy returns are NaN (excluded from stats) even as the asset doubles.
    assert np.isnan(result["net_returns"].iloc[3])
    assert np.isnan(result["net_returns"].iloc[4])
    equity = result["equity_curve"].to_numpy()
    assert (equity[2:] == 0.0).all()
    assert equity[1] > 0.0


def test_returns_after_bankruptcy_do_not_count_in_the_sharpe():
    from src.metrics import calculate_sharpe

    ohlc = make_ohlc([100.0, 100.0, 65.0, 130.0, 130.0])
    weights = make_weights([3.0] * 5, ohlc["close"].index)

    result = VectorEngine(cost=0.0).run(ohlc, weights)
    net = result["net_returns"]

    # Live returns are the first three: [0.0, 0.0, -1.05].
    survivors = net.iloc[:3]
    assert not survivors.isna().any()
    assert net.iloc[3:].isna().all()

    assert result["metrics"]["sharpe_ratio"] == pytest.approx(
        round(calculate_sharpe(survivors), 4), rel=RTOL
    )

    # Counting the +300% rebound would give a different Sharpe.
    revived = pd.Series([0.0, 0.0, -1.05, 3.0, 0.0])
    assert calculate_sharpe(survivors) != pytest.approx(calculate_sharpe(revived))


def test_exactly_minus_100_percent_also_wipes_out():
    """2.0x leverage on a 50% drop: 1 + r = 0.0 already triggers the barrier."""
    ohlc = make_ohlc([100.0, 50.0, 100.0])
    weights = make_weights([2.0] * 3, ohlc["close"].index)

    result = VectorEngine(cost=0.0).run(ohlc, weights)

    assert result["net_returns"].iloc[1] == pytest.approx(-1.0, rel=RTOL)
    np.testing.assert_allclose(
        result["equity_curve"].to_numpy(), np.array([1.0, 0.0, 0.0]), atol=0.0
    )


def test_bankruptcy_metrics_report_total_loss():
    ohlc = make_ohlc([100.0, 100.0, 65.0, 130.0, 130.0])
    weights = make_weights([3.0] * 5, ohlc["close"].index)

    metrics = VectorEngine(cost=0.0).run(ohlc, weights)["metrics"]

    assert metrics["cagr"] == -1.0
    assert metrics["max_drawdown"] == 1.0
    # Under water at indices 2, 3 and 4, through the end of the sample.
    assert metrics["max_drawdown_duration"] == 3


def test_a_deep_but_survivable_loss_does_not_trigger_the_barrier():
    """3.0x leverage on a 33% drop is -99%: the account survives and can recover."""
    ohlc = make_ohlc([100.0, 100.0, 67.0, 134.0])
    weights = make_weights([3.0] * 4, ohlc["close"].index)

    result = VectorEngine(cost=0.0).run(ohlc, weights)

    assert result["net_returns"].iloc[2] == pytest.approx(3.0 * (67.0 / 100.0 - 1.0))
    assert result["equity_curve"].iloc[-2] > 0.0
    assert result["equity_curve"].iloc[-1] > result["equity_curve"].iloc[-2]


# ---------------------------------------------------------------------------
# Output contract and delegation to src.metrics
# ---------------------------------------------------------------------------
def _alignment_scenarios():
    normal = make_ohlc([100.0, 101.0, 103.0, 99.0, 105.0, 104.0])
    normal_w = make_weights([1.0, 1.0, 0.5, 0.5, 1.0, 1.0], normal["close"].index)

    levered = make_ohlc([100.0, 98.0, 102.0, 101.0])
    levered_w = make_weights([2.0, 2.0, 2.0, 2.0], levered["close"].index)

    ruin = make_ohlc([100.0, 100.0, 60.0, 130.0, 130.0])
    ruin_w = make_weights([3.0] * 5, ruin["close"].index)

    gapped = make_ohlc([100.0, 110.0, 121.0], open_=[100.0, 105.0, 115.0])
    gapped_w = make_weights([1.0, 1.0, 1.0], gapped["close"].index)

    return [
        ("normal", VectorEngine(cost=0.001), normal, normal_w),
        ("levered", VectorEngine(cost=0.0005, borrow_rate_annual=0.03), levered, levered_w),
        ("bankruptcy", VectorEngine(cost=0.0), ruin, ruin_w),
        ("timing_open", VectorEngine(timing="open", cost=0.001), gapped, gapped_w),
    ]


@pytest.mark.parametrize(
    "engine, ohlc, weights",
    [scenario[1:] for scenario in _alignment_scenarios()],
    ids=[scenario[0] for scenario in _alignment_scenarios()],
)
def test_all_series_share_length_and_index_with_close(engine, ohlc, weights):
    result = engine.run(ohlc, weights)
    close_index = ohlc["close"].index

    for key in ("net_returns", "equity_curve", "turnover"):
        assert len(result[key]) == len(ohlc["close"])
        assert result[key].index.equals(close_index)


def test_run_returns_exactly_the_documented_keys():
    ohlc, weights = constant_weight_case(1.0)
    result = VectorEngine().run(ohlc, weights)
    assert set(result) == {"metrics", "equity_curve", "net_returns", "turnover"}
    assert isinstance(result["metrics"], dict)
    assert isinstance(result["equity_curve"], pd.Series)
    assert isinstance(result["net_returns"], pd.Series)
    assert isinstance(result["turnover"], pd.Series)


def test_output_series_are_named():
    ohlc, weights = constant_weight_case(1.0)
    result = VectorEngine().run(ohlc, weights)
    assert result["equity_curve"].name == "equity_curve"
    assert result["net_returns"].name == "net_returns"
    assert result["turnover"].name == "turnover"


def test_metrics_are_delegated_to_calculate_all_metrics():
    ohlc = make_ohlc([100.0, 104.0, 99.0, 107.0, 103.0])
    weights = make_weights([1.0, 1.0, 0.5, 0.5, 1.0], ohlc["close"].index)

    result = VectorEngine(cost=0.001, borrow_rate_annual=0.02).run(ohlc, weights)
    expected = calculate_all_metrics(
        result["net_returns"], result["equity_curve"], None
    )
    assert result["metrics"] == expected


def test_benchmark_is_forwarded_to_the_metrics_block():
    ohlc = make_ohlc([100.0, 104.0, 99.0, 107.0, 103.0])
    weights = make_weights([1.0, 1.0, 0.5, 0.5, 1.0], ohlc["close"].index)
    benchmark = pd.Series([0.0, 0.01, -0.02, 0.03, -0.01], index=ohlc["close"].index)

    result = VectorEngine(cost=0.0).run(ohlc, weights, benchmark)

    assert result["metrics"]["excess_return"] is not None
    assert result["metrics"]["information_ratio"] is not None
    expected = calculate_all_metrics(
        result["net_returns"], result["equity_curve"], benchmark
    )
    assert result["metrics"] == expected


def test_without_benchmark_the_relative_metrics_are_none():
    ohlc, weights = constant_weight_case(1.0)
    metrics = VectorEngine().run(ohlc, weights)["metrics"]
    assert metrics["excess_return"] is None
    assert metrics["information_ratio"] is None


def test_the_engine_is_deterministic():
    ohlc = make_ohlc([100.0, 104.0, 99.0, 107.0, 103.0], open_=[99.0, 105.0, 98.0, 108.0, 102.0])
    weights = make_weights([1.0, 2.0, 0.5, -1.0, 1.0], ohlc["close"].index)
    engine = VectorEngine(cost=0.0005, timing="open", borrow_rate_annual=0.03)

    first = engine.run(ohlc, weights)
    second = engine.run(ohlc, weights)

    pd.testing.assert_series_equal(first["net_returns"], second["net_returns"])
    pd.testing.assert_series_equal(first["equity_curve"], second["equity_curve"])
    assert first["metrics"] == second["metrics"]


# ---------------------------------------------------------------------------
# Vectorization contract
# ---------------------------------------------------------------------------
def test_engine_module_has_no_explicit_loops():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    loop_nodes = [
        node
        for node in ast.walk(tree)
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
    assert loop_nodes == [], "src/engine.py must be fully vectorized"


def test_engine_module_does_not_iterate_dataframes():
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    assert not attributes & {"iterrows", "itertuples", "applymap", "iteritems", "apply"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
