"""Antonacci (2012, rev. 2016) — Risk Premia Harvesting Through Dual Momentum.

Out-of-sample replication of the **modular dual momentum** system: four pairs
of assets, each targeting a different risk premium. Every month, within each
module, the asset with the higher 12-month total return is selected (*relative
momentum*) and held only if that return beats T-bills over the same 12 months
(*absolute momentum*); otherwise the module is parked in T-bills. Each module
gets a fixed 25% of the portfolio.

Implementation decisions
------------------------
* **Universe (ETFs on the paper's indices, under their current names).**

  ============  ==================================  ==================================
  Module        Asset A                             Asset B
  ============  ==================================  ==================================
  Equities      SPY  (MSCI US)                      ACWX (EAFE+ = MSCI ACWI ex US)
  Credit        HYG  (ICE BofA US Cash Pay HY)      IGIB (Bloomberg US Intermediate Credit)
  REITs         USRT (FTSE Nareit Equity REITs)     REM  (FTSE Nareit Mortgage REITs)
  Stress        VGLT (Bloomberg US Long Treasury)   GLD  (gold, London PM fix)
  ============  ==================================  ==================================

  The ETFs track the paper's index or a close proxy, not the exact series (SPY
  is the S&P 500 rather than MSCI US, HYG tracks the iBoxx USD Liquid HY, IGIB
  tracked Intermediate Credit only until 2018). ETF expense ratios are already
  reflected in prices.
* **T-bills (safe asset and hurdle).** SGOV (iShares 0-3 Month Treasury Bond)
  only trades from May 2020. Before that, BIL (SPDR Bloomberg 1-3 Month T-Bill)
  is used and spliced into SGOV at the close of SGOV's first full month (June
  2020), rescaling SGOV so the series is continuous. The resulting column is
  ``TBILL``.
* **Total-return prices.** ``oos_data.parquet`` is built with
  ``scripts/convert_factset.py --total-return``: the close is FactSet's *Total
  Return (Gross)* index and the open is scaled by the same daily factor, so
  bond and T-bill coupons reach the strategy and dividends fall in the
  overnight gap on the ex-date.
* **Signal and execution.** The signal uses the **close of the last trading
  day of month t** and is executed at the **open of the first trading day of
  month t+1** (``timing='open'``). The paper does not specify execution and,
  with no skip month, trading at the signal close itself would introduce
  look-ahead bias. 12-month lookback with **no skip month**, as in the paper.
* **OOS period.** The paper's data end in December 2011. The first OOS signal
  is the December 2011 close (Dec 2010 → Dec 2011 return) and the first
  position is opened at the 3 January 2012 open. 2010-2011 only warms up the
  lookback.
* **Costs.** ``cost_bps = 5`` (liquid ETFs) on turnover. The paper deducts no
  costs; a sensitivity at 0, 10 and 20 bps is included.
* **Rebalancing.** The engine holds target weights constant between signals,
  so each module weighs exactly 25% throughout the month (implicit, cost-free
  rebalancing). This engine convention applies equally to the benchmark.
* **Benchmark.** The **no-momentum** basket of Table 14: the 8 risky assets and
  T-bills equally weighted (1/9 each, "all nine assets" in the paper). The
  absolute-momentum-only variant applies the filter to each asset of that same
  basket.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import VectorEngine  # noqa: E402
from src.metrics import TRADING_DAYS_PER_YEAR  # noqa: E402
from src.plots import generate_tearsheet  # noqa: E402

__all__ = ["generate_target_weights", "build_weights", "load_ohlc", "main"]

# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

#: Formation period in months, no skip month (paper, Section 2).
LOOKBACK_MONTHS: int = 12

#: Cost in bps on turnover (liquid ETFs).
COST_BPS: float = 5.0

#: 10 bps per side matches the 20 bps per switch the author deducts in his
#: 2014 paper.
COST_SENSITIVITY_BPS: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)

#: Signal at the month-end close, execution at the next session's open.
TIMING: str = "open"

#: The paper's data end in December 2011.
FIRST_SIGNAL_MONTH: str = "2011-12"

#: The paper's modules. A/B follow the order of Table 1 onwards.
MODULES: pd.DataFrame = pd.DataFrame(
    {
        "asset_a": ["SPY-US", "HYG-US", "USRT-US", "VGLT-US"],
        "asset_b": ["ACWX-US", "IGIB-US", "REM-US", "GLD-US"],
        "index_a": [
            "MSCI US",
            "ICE BofA US Cash Pay High Yield",
            "FTSE Nareit Equity REITs",
            "Bloomberg US Long Treasury",
        ],
        "index_b": [
            "EAFE+ (MSCI ACWI ex US)",
            "Bloomberg US Intermediate Credit",
            "FTSE Nareit Mortgage REITs",
            "Gold (London PM fix)",
        ],
        "premium": [
            "Equity / sovereign risk",
            "Credit risk",
            "Real estate risk",
            "Economic stress",
        ],
    },
    index=pd.Index(["Equities", "Credit", "REITs", "Stress"], name="module"),
)

RISK_ASSETS: list[str] = MODULES[["asset_a", "asset_b"]].to_numpy().ravel().tolist()

#: Synthetic T-bill column and the ETFs it is spliced from (old, new).
CASH: str = "TBILL"
CASH_SOURCES: tuple[str, str] = ("BIL-US", "SGOV-US")

#: OOS drawdown episodes (peak and trough of the SPY close, checked against the data).
STRESS_WINDOWS: pd.DataFrame = pd.DataFrame(
    {
        "start": pd.to_datetime(["2015-05-21", "2018-09-20", "2020-02-19", "2022-01-03", "2025-02-19"]),
        "end": pd.to_datetime(["2016-02-11", "2018-12-24", "2020-03-23", "2022-10-12", "2025-04-08"]),
    },
    index=pd.Index(
        ["2015-16 correction", "Q4 2018", "COVID-19", "2022 bear market", "2025 tariff shock"], name="episode"
    ),
)

DATA_PATH: Path = BASE_DIR / "oos_data.parquet"
RESULTS_DIR: Path = BASE_DIR / "results"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Confirmed last trading days of each month in a daily index."""
    # A day is a month-end if the next session falls in another month. The last
    # sample date never qualifies: its month may be incomplete and, even if it
    # were not, there is no later open at which to execute the signal.
    periods = index.to_period("M")
    is_month_end = np.append(periods[1:] != periods[:-1], False)
    return index[is_month_end]


def _cash_splice_date(sgov_close: pd.Series) -> pd.Timestamp:
    """Close of SGOV's first full month; SGOV is used from the next session on."""
    first = sgov_close.first_valid_index()
    month_ends = _month_end_dates(sgov_close.index)
    return month_ends[month_ends.to_period("M") > first.to_period("M")][0]


def load_ohlc(parquet_path: Path = DATA_PATH) -> tuple[dict[str, pd.DataFrame], pd.Timestamp]:
    """Load the long parquet; return the engine's wide OHLC panel and the T-bill splice date.

    The panel holds the 8 risky assets plus the synthetic ``TBILL`` column (BIL
    up to the splice, rescaled SGOV afterwards), restricted to dates on which
    every asset has a price.
    """
    panel = pd.read_parquet(parquet_path)
    panel["date"] = pd.to_datetime(panel["date"])
    open_all = panel.pivot(index="date", columns="ticker", values="open").sort_index()
    close_all = panel.pivot(index="date", columns="ticker", values="close").sort_index()

    old, new = CASH_SOURCES
    splice = _cash_splice_date(close_all[new])
    scale = close_all.at[splice, old] / close_all.at[splice, new]
    after = close_all.index > splice

    open_prices = open_all[RISK_ASSETS].assign(**{CASH: open_all[old].where(~after, open_all[new] * scale)})
    close_prices = close_all[RISK_ASSETS].assign(**{CASH: close_all[old].where(~after, close_all[new] * scale)})

    complete = open_prices.notna().all(axis=1) & close_prices.notna().all(axis=1)
    ohlc = {
        "open": open_prices.loc[complete].astype("float64"),
        "close": close_prices.loc[complete].astype("float64"),
    }
    return ohlc, splice


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
def _momentum(close: pd.DataFrame) -> pd.DataFrame:
    """``LOOKBACK_MONTHS`` total return at each month-end close."""
    monthly = close.loc[_month_end_dates(close.index)]
    return monthly / monthly.shift(LOOKBACK_MONTHS) - 1.0


def build_weights(
    ohlc: dict[str, pd.DataFrame],
    *,
    relative: bool = True,
    absolute: bool = True,
    modules: list[str] | None = None,
) -> pd.DataFrame:
    """Daily target weights for the four variants of the paper's Table 14.

    * ``relative`` on: each module weighs ``1 / n_modules``, fully allocated to
      the asset with the higher 12-month return. Off: the Table 14 basket
      ("all nine assets"), risky assets and T-bills equally weighted at
      ``1 / (2 * n_modules + 1)`` each (1/9 with four modules).
    * ``absolute`` on: a position is held only if its 12-month return beats
      T-bills; otherwise that slice goes to T-bills.

    Both on is dual momentum; both off is the no-momentum basket. ``modules``
    restricts the portfolio to a subset (a single module gets 100%).

    New weights appear on the month-end close row that generates them and are
    forward-filled until the next signal; with ``timing='open'`` the engine
    executes them at the next open. Weights are 0 until the lookback is full.
    """
    close = ohlc["close"]
    table = MODULES if modules is None else MODULES.loc[list(modules)]
    mom = _momentum(close)

    mom_a = mom[table["asset_a"]].to_numpy()
    mom_b = mom[table["asset_b"]].to_numpy()
    hurdle = mom[[CASH]].to_numpy()
    valid = np.isfinite(mom_a).all(axis=1) & np.isfinite(mom_b).all(axis=1) & np.isfinite(hurdle[:, 0])

    if relative:
        # Tie (unlikely with continuous prices): asset A wins.
        share_a = (mom_a >= mom_b).astype("float64")
        share_b = 1.0 - share_a
        sleeve = 1.0 / len(table)
    else:
        # Table 14 basket: each risky asset gets one slice and T-bills take the
        # remainder (w_cash below).
        share_a = share_b = np.ones(mom_a.shape)
        sleeve = 1.0 / (2 * len(table) + 1)

    if absolute:
        pass_a, pass_b = mom_a > hurdle, mom_b > hurdle
    else:
        pass_a = pass_b = np.ones(mom_a.shape, dtype=bool)

    live = valid[:, None]
    w_a = sleeve * share_a * pass_a * live
    w_b = sleeve * share_b * pass_b * live
    w_cash = np.where(valid, 1.0 - w_a.sum(axis=1) - w_b.sum(axis=1), 0.0)

    monthly = pd.DataFrame(
        np.column_stack([w_a, w_b, w_cash]),
        index=mom.index,
        columns=[*table["asset_a"], *table["asset_b"], CASH],
    ).reindex(columns=close.columns, fill_value=0.0)
    return monthly.reindex(close.index).ffill().fillna(0.0)


def generate_target_weights(ohlc: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Target weights of Antonacci's dual momentum (4 modules, 25% each)."""
    return build_weights(ohlc, relative=True, absolute=True)


def evaluation_start(ohlc: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Close of the first OOS signal (last trading day of December 2011)."""
    month_ends = _month_end_dates(ohlc["close"].index)
    start = month_ends[month_ends.to_period("M") == pd.Period(FIRST_SIGNAL_MONTH, "M")][0]
    assert np.isfinite(_momentum(ohlc["close"]).loc[start]).all(), "Incomplete lookback at the first signal"
    return start


def _slice_ohlc(ohlc: dict[str, pd.DataFrame], start: pd.Timestamp) -> dict[str, pd.DataFrame]:
    return {"open": ohlc["open"].loc[start:], "close": ohlc["close"].loc[start:]}


# ---------------------------------------------------------------------------
# Supplementary statistics
# ---------------------------------------------------------------------------
def _drawdown(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def _annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _monthly_curve(curve: pd.Series) -> pd.Series:
    return curve.loc[_month_end_dates(curve.index)]


def _monthly_returns(curve: pd.Series) -> pd.Series:
    """Monthly returns; the first point is the Dec 2011 signal close, so the first return is Jan 2012."""
    monthly = _monthly_curve(curve)
    return (monthly / monthly.shift(1) - 1.0).iloc[1:]


def _annual_returns(returns: pd.Series) -> pd.Series:
    annual = (1.0 + returns.fillna(0.0)).groupby(returns.index.year).prod() - 1.0
    annual.index.name = "year"
    return annual


def _annual_sharpe(returns: pd.Series) -> pd.Series:
    """Calendar-year Sharpe (rf=0) from daily returns, same convention as the engine."""
    by_year = returns.dropna().groupby(returns.dropna().index.year)
    sharpe = by_year.mean() / by_year.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe.index.name = "year"
    return sharpe.dropna()


def _summary(result: dict, tbill_close: pd.Series) -> dict:
    """Stats block comparable with the paper's tables."""
    # ``sharpe_excess_tbill`` follows the paper (monthly excess return over
    # T-bills, annualized); ``sharpe_ratio`` is the engine's (daily, rf=0).
    metrics = result["metrics"]
    equity = result["equity_curve"]
    monthly = _monthly_returns(equity)
    excess = monthly - _monthly_returns(tbill_close).reindex(monthly.index)
    return {
        "worst_month_date": monthly.idxmin().strftime("%Y-%m"),
        "cagr": metrics["cagr"],
        "volatility": round(_annualised_volatility(result["net_returns"]), 4),
        "sharpe_ratio": metrics["sharpe_ratio"],
        "sharpe_excess_tbill": round(float(excess.mean() / excess.std(ddof=1) * np.sqrt(12.0)), 4),
        "max_drawdown": metrics["max_drawdown"],
        "mar_ratio": metrics["mar_ratio"],
        "pct_positive_months": round(float((monthly > 0.0).mean()), 4),
        "worst_month": round(float(monthly.min()), 4),
        "total_turnover": round(float(result["turnover"].sum()), 4),
    }


def _period_stats(net_returns: pd.Series, equity: pd.Series) -> dict:
    """CAGR, volatility and drawdown of a sub-period, measured from its own starting capital."""
    start_capital = float(equity.iloc[0]) / (1.0 + float(net_returns.iloc[0]))
    growth = float(equity.iloc[-1]) / start_capital
    cagr = growth ** (TRADING_DAYS_PER_YEAR / len(equity)) - 1.0
    vol = _annualised_volatility(net_returns)
    return {
        "cagr": round(cagr, 4),
        "volatility": round(vol, 4),
        "return_to_vol": round(cagr / vol, 4) if vol > 0.0 else None,
        "max_drawdown": round(abs(float(_drawdown(equity / start_capital).min())), 4),
    }


def _asset_statistics(close: pd.DataFrame) -> pd.DataFrame:
    """Buy-and-hold stats per asset (total return, no costs)."""
    returns = close.pct_change().iloc[1:]
    growth = close.iloc[-1] / close.iloc[0]
    return pd.DataFrame(
        {
            "cagr": growth ** (TRADING_DAYS_PER_YEAR / len(close)) - 1.0,
            "volatility": returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR),
            "max_drawdown": (close / close.cummax() - 1.0).min().abs(),
        }
    ).round(4)


def _max_drawdown_episode(equity: pd.Series) -> dict:
    """Peak and trough of the maximum drawdown (deepest underwater point of the daily curve)."""
    trough = _drawdown(equity).idxmin()
    peak = equity.loc[:trough].idxmax()
    return {"peak": str(peak.date()), "trough": str(trough.date())}


def _module_positions(weights: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Role of each module at each monthly signal: 'A', 'B' or 'T-bills'."""
    monthly = weights.loc[month_ends]
    in_a = monthly[MODULES["asset_a"]].to_numpy() > 0.0
    in_b = monthly[MODULES["asset_b"]].to_numpy() > 0.0
    roles = np.where(in_a, "A", np.where(in_b, "B", "T-bills"))
    return pd.DataFrame(roles, index=month_ends, columns=MODULES.index)


def _longest_drawdown(equity: pd.Series) -> dict:
    """Peak, recovery and length of the longest underwater period."""
    # Each day at a high opens an episode; underwater days share the id of the
    # preceding peak (same convention as the engine).
    underwater = equity < equity.cummax()
    episode = (~underwater).cumsum()
    lengths = underwater.groupby(episode).sum()
    longest = lengths.idxmax()
    in_episode = episode == longest
    peak = equity.index[in_episode & ~underwater][0]
    last_underwater = equity.index[in_episode & underwater][-1]
    after = equity.index[equity.index > last_underwater]
    return {
        "peak": str(peak.date()),
        "recovery": str(after[0].date()) if len(after) else None,
        "sessions": int(lengths.max()),
    }


def _stress_returns(curves: pd.DataFrame) -> pd.DataFrame:
    """Return of each curve from peak to trough of each drawdown episode."""
    start_pos = curves.index.get_indexer(STRESS_WINDOWS["start"], method="ffill")
    end_pos = curves.index.get_indexer(STRESS_WINDOWS["end"], method="ffill")
    values = curves.to_numpy()
    returns = values[end_pos] / values[start_pos] - 1.0
    return pd.DataFrame(returns, index=STRESS_WINDOWS.index, columns=curves.columns).round(4)


def _stale_month_ends(ohlc: dict[str, pd.DataFrame], month_ends: pd.DatetimeIndex) -> dict:
    """Count month-end closes equal to the previous close while the open changed."""
    # Signature of a close FactSet did not update. With two-decimal prices it
    # can also be a genuine unchanged close, so this is only a warning.
    close, open_ = ohlc["close"], ohlc["open"]
    stale = (close.diff() == 0.0) & (open_.diff() != 0.0)
    return {k: int(v) for k, v in stale.loc[month_ends, RISK_ASSETS].sum().items()}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def build_diagnostics(
    ohlc: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    runs: dict[str, dict],
    module_runs: dict[str, dict],
    module_relative_runs: dict[str, dict],
    cost_runs: dict[float, dict],
    splice: pd.Timestamp,
) -> dict:
    """Supplementary diagnostics for metrics.json (decomposition, modules, alpha decay, stress, costs)."""
    close = ohlc["close"]
    tbill = close[CASH]
    spy = close["SPY-US"]
    month_ends = _month_end_dates(close.index)
    years = len(close) / TRADING_DAYS_PER_YEAR

    # --- Table 14 decomposition ----------------------------------------------
    decomposition = {name: _summary(run, tbill) for name, run in runs.items()}

    # --- Standalone modules, holdings and switches (Tables 1-11) -------------
    positions = _module_positions(weights, month_ends)
    switches_per_year = (positions != positions.shift(1)).iloc[1:].sum() / years

    # Arithmetic attribution: the portfolio is the average of the modules (bar
    # cost netting on the shared T-bill leg), so each module contributes 1/4 of
    # its mean return. The cost of the absolute filter is the difference between
    # the dual module and the same module with relative momentum only.
    module_net = pd.DataFrame({m: r["net_returns"] for m, r in module_runs.items()})
    module_rel_net = pd.DataFrame({m: r["net_returns"] for m, r in module_relative_runs.items()})
    scale = TRADING_DAYS_PER_YEAR / len(MODULES)
    contribution = module_net.mean() * scale
    filter_cost = (module_net.mean() - module_rel_net.mean()) * scale

    # Role of each module in each holding month: set by the previous month-end
    # signal, so the index is shifted forward one month.
    held = positions.set_axis(positions.index.to_period("M") + 1)
    role_ticker = {"A": "asset_a", "B": "asset_b"}
    holding_share = (
        positions.melt(var_name="module", value_name="role")
        .pipe(lambda d: pd.crosstab(d["module"], d["role"], normalize="index"))
        .reindex(index=MODULES.index, columns=["A", "B", "T-bills"], fill_value=0.0)
    )
    asset_stats = _asset_statistics(close)

    module_monthly = pd.DataFrame({m: _monthly_returns(r["equity_curve"]) for m, r in module_runs.items()})
    spy_monthly = _monthly_returns(spy).reindex(module_monthly.index)
    corr_spy = module_monthly.corrwith(spy_monthly)

    module_summaries = {m: _summary(module_runs[m], tbill) for m in MODULES.index}
    worst_role = {
        m: held.at[pd.Period(module_summaries[m]["worst_month_date"], "M"), m] for m in MODULES.index
    }
    modules = {
        m: {
            "asset_a": MODULES.at[m, "asset_a"],
            "asset_b": MODULES.at[m, "asset_b"],
            "dual": module_summaries[m],
            "relative_only": _summary(module_relative_runs[m], tbill),
            "worst_month_holding": MODULES.at[m, role_ticker[worst_role[m]]] if worst_role[m] in role_ticker else CASH,
            "contribution_annual": round(float(contribution[m]), 4),
            "absolute_filter_cost_annual": round(float(filter_cost[m]), 4),
            "buy_and_hold_a": asset_stats.loc[MODULES.at[m, "asset_a"]].to_dict(),
            "buy_and_hold_b": asset_stats.loc[MODULES.at[m, "asset_b"]].to_dict(),
            "switches_per_year": round(float(switches_per_year[m]), 2),
            "pct_a": round(float(holding_share.at[m, "A"]), 4),
            "pct_b": round(float(holding_share.at[m, "B"]), 4),
            "pct_cash": round(float(holding_share.at[m, "T-bills"]), 4),
            "corr_spy_monthly": round(float(corr_spy[m]), 4),
        }
        for m in MODULES.index
    }

    # --- Current position (last confirmed signal) ----------------------------
    last_signal = month_ends[-1]
    last_roles = positions.loc[last_signal]
    current = {
        m: (MODULES.at[m, "asset_a"] if role == "A" else MODULES.at[m, "asset_b"] if role == "B" else CASH)
        for m, role in last_roles.items()
    }

    # --- Alpha decay: OOS halves ---------------------------------------------
    dual, none = runs["dual"], runs["none"]
    mid = len(close) // 2
    halves = {
        label: {
            "start": str(close.index[sl][0].date()),
            "end": str(close.index[sl][-1].date()),
            "strategy": _period_stats(dual["net_returns"].iloc[sl], dual["equity_curve"].iloc[sl]),
            "benchmark": _period_stats(none["net_returns"].iloc[sl], none["equity_curve"].iloc[sl]),
            "tbill_cagr": round(float((tbill.iloc[sl].iloc[-1] / tbill.iloc[sl].iloc[0])
                                      ** (TRADING_DAYS_PER_YEAR / len(tbill.iloc[sl])) - 1.0), 4),
        }
        for label, sl in (("first_half", slice(None, mid)), ("second_half", slice(mid, None)))
    }

    # --- Calendar years and drawdown episodes --------------------------------
    annual = pd.DataFrame(
        {
            "dual": _annual_returns(dual["net_returns"]),
            "none": _annual_returns(none["net_returns"]),
            "spy": _annual_returns(spy.pct_change()),
            "tbill": _annual_returns(tbill.pct_change()),
        }
    )
    curves = pd.DataFrame(
        {"dual": dual["equity_curve"], "none": none["equity_curve"], "spy": spy, "tbill": tbill}
    )
    stress = _stress_returns(curves)

    # --- Exposure and costs --------------------------------------------------
    exposure_risk = weights[RISK_ASSETS].sum(axis=1)
    cost_table = {
        f"{bps:g}": {"cagr": r["metrics"]["cagr"], "sharpe_ratio": r["metrics"]["sharpe_ratio"],
                     "max_drawdown": r["metrics"]["max_drawdown"]}
        for bps, r in cost_runs.items()
    }

    return {
        "period_start": str(close.index[0].date()),
        "first_execution": str(close.index[1].date()),
        "period_end": str(close.index[-1].date()),
        "last_signal": str(last_signal.date()),
        "n_days": int(len(close)),
        "n_signal_months": int(len(month_ends)),
        "years": round(years, 2),
        "cash_splice_date": str(splice.date()),
        "decomposition": decomposition,
        "modules": modules,
        "module_correlations": module_monthly.corr().round(4).to_dict(),
        "current_positions": current,
        "max_drawdown_episode": _max_drawdown_episode(dual["equity_curve"]),
        "longest_drawdown": _longest_drawdown(dual["equity_curve"]),
        "avg_risk_exposure": round(float(exposure_risk.mean()), 4),
        "pct_months_fully_in_tbills": round(float((positions == "T-bills").all(axis=1).mean()), 4),
        # Holding months (the month after the signal) fully in T-bills.
        "months_fully_in_tbills": [
            str(p) for p in (positions.index[(positions == "T-bills").all(axis=1)].to_period("M") + 1)
        ],
        "tbill_cagr": round(float(asset_stats.at[CASH, "cagr"]), 4),
        "spy_buy_and_hold": asset_stats.loc["SPY-US"].to_dict(),
        "cost_sensitivity_bps": cost_table,
        "alpha_decay": halves,
        "annual_returns": {col: {int(y): round(float(v), 4) for y, v in annual[col].items()} for col in annual},
        "annual_sharpe": {
            name: {int(y): round(float(v), 4) for y, v in _annual_sharpe(run["net_returns"]).items()}
            for name, run in (("dual", dual), ("none", none))
        },
        "stress_episodes": {
            ep: {"start": str(STRESS_WINDOWS.at[ep, "start"].date()), "end": str(STRESS_WINDOWS.at[ep, "end"].date()),
                 **stress.loc[ep].to_dict()}
            for ep in STRESS_WINDOWS.index
        },
        "stale_month_end_closes": _stale_month_ends(ohlc, month_ends),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> dict:
    """Run the full backtest and write ``results/metrics.json`` and ``results/tearsheet.png``."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    full_ohlc, splice = load_ohlc()
    start = evaluation_start(full_ohlc)
    ohlc = _slice_ohlc(full_ohlc, start)

    # The four Table 14 variants, computed on the full history (the 12-month
    # lookback needs 2010-2011) and then trimmed to the OOS period.
    variants = {
        "none": build_weights(full_ohlc, relative=False, absolute=False),
        "absolute": build_weights(full_ohlc, relative=False, absolute=True),
        "relative": build_weights(full_ohlc, relative=True, absolute=False),
        "dual": generate_target_weights(full_ohlc),
    }
    variants = {name: w.loc[start:] for name, w in variants.items()}

    engine = VectorEngine(cost=COST_BPS / 10_000.0, timing=TIMING)

    # Benchmark: the no-momentum basket, through the same engine at the same cost.
    benchmark = engine.run(ohlc, variants["none"])
    bench_returns = benchmark["net_returns"]
    runs = {
        name: benchmark if name == "none" else engine.run(ohlc, w, benchmark_returns=bench_returns)
        for name, w in variants.items()
    }
    strategy = runs["dual"]

    module_runs = {
        m: engine.run(ohlc, build_weights(full_ohlc, modules=[m]).loc[start:]) for m in MODULES.index
    }
    module_relative_runs = {
        m: engine.run(ohlc, build_weights(full_ohlc, modules=[m], absolute=False).loc[start:])
        for m in MODULES.index
    }
    cost_runs = {
        bps: VectorEngine(cost=bps / 10_000.0, timing=TIMING).run(ohlc, variants["dual"])
        for bps in COST_SENSITIVITY_BPS
    }

    diagnostics = build_diagnostics(
        ohlc, variants["dual"], runs, module_runs, module_relative_runs, cost_runs, splice
    )

    payload = {
        "config": {
            "paper": "Antonacci (2012, rev. 2016) — Risk Premia Harvesting Through Dual Momentum",
            "lookback_months": LOOKBACK_MONTHS,
            "skip_month": False,
            "timing": TIMING,
            "execution": "signal at the close of the last trading day of month t; entry at the open "
                         "of the first trading day of month t+1",
            "cost_bps": COST_BPS,
            "modules": MODULES[["asset_a", "asset_b", "index_a", "index_b"]].to_dict(orient="index"),
            "cash": {"column": CASH, "sources": list(CASH_SOURCES), "splice_date": str(splice.date())},
            "prices": "FactSet Total Return (Gross): dividends reinvested",
            "benchmark": "equal-weight 9-asset basket (8 risky assets + T-bills) without momentum (Table 14)",
        },
        "strategy": strategy["metrics"],
        "benchmark": benchmark["metrics"],
        "diagnostics": diagnostics,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    generate_tearsheet(
        equity_curve=strategy["equity_curve"],
        net_returns=strategy["net_returns"],
        benchmark_equity=benchmark["equity_curve"],
        benchmark_returns=bench_returns,
        output_path=str(RESULTS_DIR / "tearsheet.png"),
        title="Antonacci (2016) Dual Momentum — Out-of-Sample Backtest",
        footnote=(
            f"Net of {COST_BPS:g} bps transaction costs · Total return (dividends reinvested) · "
            "Signal at month-end close, execution at next session's open · "
            "Benchmark: equal-weight 9-asset basket (8 risky assets + T-bills) without momentum · Data: FactSet"
        ),
        strategy_label="Dual Momentum",
        benchmark_label="No-momentum basket",
    )

    d = diagnostics["decomposition"]
    print(f"OOS period: {diagnostics['period_start']} -> {diagnostics['period_end']} ({diagnostics['years']} years)")
    print("\n".join(
        f"  {name:<9} CAGR {d[name]['cagr']:+.2%}  Vol {d[name]['volatility']:.2%}  "
        f"Sharpe {d[name]['sharpe_ratio']:.2f} (excess over T-bills {d[name]['sharpe_excess_tbill']:.2f})  "
        f"MaxDD {d[name]['max_drawdown']:.2%}"
        for name in ("dual", "relative", "absolute", "none")
    ))
    print(f"Results in {RESULTS_DIR}")
    return payload


if __name__ == "__main__":
    main()
