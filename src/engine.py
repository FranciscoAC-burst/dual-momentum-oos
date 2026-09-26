"""Vectorized, deterministic execution engine.

Turns a target-weight matrix and OHLC prices into net returns, an equity
curve, turnover and the metrics block from :mod:`src.metrics`.

Conventions
-----------
* Fully vectorized with NumPy/Pandas; identical inputs always produce
  identical outputs.
* Initial capital is an implicit ``1.0``: the equity curve is
  ``(1.0 + net_returns).cumprod()``, so its first point is the NAV at the
  close of the first session. All returned series share the length and
  ``DatetimeIndex`` of ``ohlc["close"]``.
* ``cost`` is a fraction of turnover (``0.0005`` = 5 bps):
  ``transaction_costs = turnover * cost``.
* The initial portfolio is built from cash at ``t=0``:
  ``turnover.iloc[0] = |w_0|.sum()`` and its cost is charged to
  ``net_returns.iloc[0]``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .metrics import TRADING_DAYS_PER_YEAR, calculate_all_metrics

__all__ = ["VectorEngine"]

_VALID_TIMINGS: frozenset = frozenset({"close", "open"})
_REQUIRED_OHLC_KEYS: frozenset = frozenset({"open", "close"})


class VectorEngine:
    """Vectorized backtesting engine with transaction and financing costs.

    Parameters
    ----------
    cost:
        Transaction cost as a fraction of daily turnover
        (``0.0005`` = 5 bps).
    timing:
        ``"close"`` executes each decision in the same day's closing
        auction; ``"open"`` executes it at the next day's open.
    borrow_rate_annual:
        Annual rate charged on gross exposure above 100% of capital.
        Converted to a daily rate by dividing by 252.
    """

    def __init__(
        self,
        cost: float = 0.0005,
        timing: str = "close",
        borrow_rate_annual: float = 0.0,
    ) -> None:
        assert timing in _VALID_TIMINGS, (
            f"timing must be one of {sorted(_VALID_TIMINGS)}; got {timing!r}"
        )

        self.cost: float = float(cost)
        self.timing: str = timing
        self.borrow_rate_annual: float = float(borrow_rate_annual)
        self.borrow_rate_daily: float = self.borrow_rate_annual / float(
            TRADING_DAYS_PER_YEAR
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(cost={self.cost!r}, "
            f"timing={self.timing!r}, "
            f"borrow_rate_annual={self.borrow_rate_annual!r})"
        )

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def run(
        self,
        ohlc: dict[str, pd.DataFrame],
        target_weights: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> dict:
        """Run the backtest.

        Parameters
        ----------
        ohlc:
            Dict of wide DataFrames (dates x tickers). Must contain
            ``"open"`` and ``"close"`` with identical index and columns;
            other keys are ignored.
        target_weights:
            Weights decided at each close (dates x tickers). Reindexed to
            ``ohlc["close"]``; missing dates or tickers are treated as flat.
        benchmark_returns:
            Optional daily benchmark returns, used for excess return and
            information ratio.

        Returns
        -------
        dict
            ``metrics`` (from :func:`src.metrics.calculate_all_metrics`),
            ``equity_curve``, ``net_returns`` and ``turnover``. The three
            series share the length and index of ``ohlc["close"]``. If the
            account is wiped out, equity stays at ``0.0`` and later net
            returns are ``NaN``.
        """
        open_prices, close_prices = self._validate_ohlc(ohlc)
        weights = self._align_weights(target_weights, close_prices)

        gross_returns = self._gross_returns(open_prices, close_prices, weights)
        turnover = self._turnover(weights)
        transaction_costs = turnover * self.cost
        financing_costs = self._financing_costs(weights)

        net_returns = (gross_returns - transaction_costs - financing_costs).rename(
            "net_returns"
        )
        equity_curve, net_returns = self._apply_ruin_barrier(net_returns)

        # CAGR is measured against the implicit starting capital of 1.0, not
        # against the first equity point, which already reflects the opening
        # commission.
        metrics = calculate_all_metrics(
            net_returns, equity_curve, benchmark_returns, initial_capital=1.0
        )

        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "net_returns": net_returns,
            "turnover": turnover,
        }

    # -----------------------------------------------------------------
    # Validation and alignment
    # -----------------------------------------------------------------
    @staticmethod
    def _validate_ohlc(
        ohlc: dict[str, pd.DataFrame],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Validate the ``ohlc`` contract and return ``(open, close)`` as float64."""
        assert isinstance(ohlc, dict), (
            f"ohlc must be a dict of DataFrames; got {type(ohlc).__name__}"
        )

        missing = _REQUIRED_OHLC_KEYS.difference(ohlc)
        assert not missing, (
            f"ohlc must contain keys {sorted(_REQUIRED_OHLC_KEYS)}; "
            f"missing {sorted(missing)}"
        )

        open_prices = ohlc["open"]
        close_prices = ohlc["close"]

        assert isinstance(open_prices, pd.DataFrame), "ohlc['open'] must be a DataFrame"
        assert isinstance(close_prices, pd.DataFrame), (
            "ohlc['close'] must be a DataFrame"
        )
        assert open_prices.index.equals(close_prices.index), (
            "ohlc['open'] and ohlc['close'] must share the same index"
        )
        assert open_prices.columns.equals(close_prices.columns), (
            "ohlc['open'] and ohlc['close'] must share the same columns"
        )

        return open_prices.astype("float64"), close_prices.astype("float64")

    @staticmethod
    def _align_weights(
        target_weights: pd.DataFrame, close_prices: pd.DataFrame
    ) -> pd.DataFrame:
        """Reindex weights to the close prices; missing dates/tickers become 0.0."""
        assert isinstance(target_weights, pd.DataFrame), (
            "target_weights must be a DataFrame"
        )

        aligned = target_weights.reindex(
            index=close_prices.index, columns=close_prices.columns
        )
        return aligned.astype("float64").fillna(0.0)

    # -----------------------------------------------------------------
    # Return components
    # -----------------------------------------------------------------
    def _gross_returns(
        self,
        open_prices: pd.DataFrame,
        close_prices: pd.DataFrame,
        weights: pd.DataFrame,
    ) -> pd.Series:
        """Daily gross portfolio return under the configured execution timing.

        ``close``
            The decision taken at the close of ``t`` is filled in that same
            auction and earns the full ``Close(t) -> Close(t+1)`` return.

        ``open``
            The decision taken after the close of ``t`` is filled at the
            open of ``t+1``: the previous position bears the overnight gap
            and the new position earns the intraday leg.
        """
        if self.timing == "close":
            # Equivalent to ``close_prices.pct_change()`` without the
            # deprecated ``fill_method`` behavior.
            asset_returns = (close_prices / close_prices.shift(1) - 1.0).fillna(0.0)
            positions = weights.shift(1).fillna(0.0)
            return (positions * asset_returns).sum(axis=1)

        ret_overnight = (open_prices / close_prices.shift(1) - 1.0).fillna(0.0)
        ret_intraday = (close_prices / open_prices - 1.0).fillna(0.0)

        previous_positions = weights.shift(2).fillna(0.0)
        incoming_positions = weights.shift(1).fillna(0.0)

        overnight_pnl = (previous_positions * ret_overnight).sum(axis=1)
        intraday_pnl = (incoming_positions * ret_intraday).sum(axis=1)
        return overnight_pnl + intraday_pnl

    @staticmethod
    def _turnover(weights: pd.DataFrame) -> pd.Series:
        """Daily turnover: sum of absolute weight changes.

        The first row is measured against an all-cash portfolio, so building
        the initial positions generates turnover and pays costs.
        """
        changes = (weights - weights.shift(1).fillna(0.0)).abs()
        return changes.sum(axis=1).rename("turnover")

    def _financing_costs(self, weights: pd.DataFrame) -> pd.Series:
        """Daily cost of financing the gross exposure held above 100% of capital."""
        gross_exposure = weights.abs().sum(axis=1).shift(1).fillna(0.0)
        borrowed_capital = (gross_exposure - 1.0).clip(lower=0.0)
        return borrowed_capital * self.borrow_rate_daily

    # -----------------------------------------------------------------
    # Equity curve and bankruptcy barrier
    # -----------------------------------------------------------------
    @staticmethod
    def _apply_ruin_barrier(net_returns: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Compound the equity curve and apply an absorbing bankruptcy barrier.

        The account is wiped out on the first day where ``1 + r <= 0``:
        equity drops to ``0.0`` and stays there, so a later market rebound
        cannot revive a liquidated account. Returns after that day are set
        to ``NaN`` so they do not enter return-based metrics; the
        bankruptcy-day return itself is kept.

        Returns ``(equity_curve, truncated_net_returns)``.
        """
        returns = net_returns.to_numpy(dtype="float64")
        growth = 1.0 + returns

        wiped_out = growth <= 0.0
        alive = np.cumsum(wiped_out) == 0

        # Neutralize the bankruptcy-day factor so the cumulative product
        # cannot flip sign; that day is forced to 0.0 right after.
        compounding = np.where(alive, growth, 1.0)
        equity = np.cumprod(compounding)
        equity = np.where(alive & (equity > 0.0), equity, 0.0)

        # Days strictly after bankruptcy (the bankruptcy day keeps its return).
        post_bankruptcy = ~alive & ~wiped_out
        surviving_returns = np.where(post_bankruptcy, np.nan, returns)

        index = net_returns.index
        equity_curve = pd.Series(equity, index=index, name="equity_curve")
        truncated_returns = pd.Series(
            surviving_returns, index=index, name="net_returns"
        )
        return equity_curve, truncated_returns
