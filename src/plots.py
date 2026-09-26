"""Backtest tearsheet generation.

Draws a single-figure diagnostic panel from the equity curve and net returns
produced by :class:`src.engine.VectorEngine`.

Conventions:

* Headless: the matplotlib ``Agg`` backend is set at import time and the
  figure is always closed after saving, so it is safe for batch runs.
* Header metrics are computed with :mod:`src.metrics`, so they match
  ``metrics.json``.
* Equity curves are rebased to 1.0 at their first observation so the strategy
  and the benchmark are comparable.
* Percentage axes never display ``-0.0%``.
* The benchmark is optional; without it, the affected panels plot the
  strategy only.
"""

from __future__ import annotations

import os
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Headless backend; must be set before importing pyplot.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from .metrics import (
    calculate_cagr,
    calculate_mar,
    calculate_max_drawdown,
    calculate_max_drawdown_duration,
    calculate_sharpe,
)

__all__ = ["generate_tearsheet"]

TRADING_DAYS_PER_YEAR: int = 252

#: Rolling Sharpe window (trading days): one year is less noisy than six
#: months for monthly-rebalanced strategies.
_ROLLING_WINDOW: int = 252

#: Cumulative return above which the equity axis switches to a log scale
#: (300%, i.e. a final equity of 4.0x).
_SEMILOG_THRESHOLD: float = 3.0

#: Calendar years with fewer sessions than this are flagged as partial.
_PARTIAL_YEAR_SESSIONS: int = 240

#: Percentile of |monthly return| that sets the heatmap color scale, so a
#: single extreme month does not wash out the other cells.
_HEATMAP_PERCENTILE: float = 95.0

#: Minimum heatmap color scale (1%), for near-flat series.
_MIN_COLOR_SCALE: float = 0.01

_MONTH_LABELS: list[str] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

#: Last heatmap column, holding the full-year return.
_ANNUAL_COLUMN: str = "Year"

_STRATEGY_COLOR = "#1f4e79"
_BENCHMARK_COLOR = "#7f7f7f"
_DRAWDOWN_COLOR = "#e15759"
_POSITIVE_COLOR = "#59a14f"
_NEGATIVE_COLOR = "#e15759"
_MUTED_TEXT = "#555555"
_TABLE_EDGE = "#d0d0d0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _format_pct(value: float, decimals: int = 1) -> str:
    """Format a fraction as a percentage; never returns ``-0.0%``."""
    if not np.isfinite(value):
        return "—"
    scaled = round(float(value) * 100.0, decimals) + 0.0  # -0.0 + 0.0 == 0.0
    return f"{scaled:.{decimals}f}%"


def _percent_formatter(decimals: int = 0) -> FuncFormatter:
    return FuncFormatter(lambda value, _pos: _format_pct(value, decimals))


def _rotate_date_labels(ax: plt.Axes, degrees: int = 45) -> None:
    ax.tick_params(axis="x", labelrotation=degrees)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")


def _despine(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)


def _clean_series(series: pd.Series) -> pd.Series:
    """Numeric copy of ``series`` with NaN/inf dropped."""
    numeric = pd.to_numeric(pd.Series(series), errors="coerce")
    return numeric[np.isfinite(numeric)]


def _rebase(equity: pd.Series) -> pd.Series:
    equity = _clean_series(equity)
    if equity.empty or equity.iloc[0] == 0.0:
        return equity
    return equity / equity.iloc[0]


def _drawdown(equity: pd.Series) -> pd.Series:
    equity = _clean_series(equity)
    if equity.empty:
        return equity
    running_max = equity.cummax()
    return equity / running_max - 1.0


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling Sharpe ratio (rf = 0)."""
    returns = _clean_series(returns)
    if returns.empty:
        return returns
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std(ddof=1)
    sharpe = rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return sharpe.replace([np.inf, -np.inf], np.nan)


def _initial_capital(equity: pd.Series, returns: Optional[pd.Series]) -> float:
    """Capital before the first session, so the CAGR matches the engine.

    The engine reports ``equity[0] = capital * (1 + r[0])``; undoing that first
    return recovers the starting capital, so the opening trade cost is
    reflected in the CAGR, as in ``metrics.json``.
    """
    equity = _clean_series(equity)
    if equity.empty:
        return 1.0
    first = float(equity.iloc[0])
    if returns is None:
        return first
    numeric = pd.to_numeric(pd.Series(returns), errors="coerce")
    first_return = numeric.get(equity.index[0], np.nan)
    if np.isfinite(first_return) and first_return > -1.0:
        return first / (1.0 + float(first_return))
    return first


def _key_metrics(equity: pd.Series, returns: pd.Series) -> dict[str, str]:
    cagr = calculate_cagr(equity, _initial_capital(equity, returns))
    max_dd = calculate_max_drawdown(equity)
    clean = _clean_series(returns)
    volatility = float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if clean.size > 1 else 0.0
    calmar = calculate_mar(cagr, max_dd)
    return {
        "CAGR": _format_pct(cagr, 2),
        "Volatility": _format_pct(volatility, 2),
        "Sharpe (rf = 0)": f"{calculate_sharpe(returns):.2f}",
        "Max Drawdown": _format_pct(-max_dd, 2),
        "Calmar": f"{calmar:.2f}" if np.isfinite(calmar) else "—",
        "Longest DD": f"{calculate_max_drawdown_duration(equity)} sessions",
    }


def _period_label(equity: pd.Series) -> str:
    """Period subtitle, e.g. 'Dec 2011 – Sep 2026 · 14.7 years'."""
    clean = _clean_series(equity)
    if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
        return ""
    years = len(clean) / TRADING_DAYS_PER_YEAR
    return f"{clean.index[0]:%b %Y} – {clean.index[-1]:%b %Y} · {years:.1f} years"


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _plot_key_metrics(
    ax: plt.Axes,
    equity: pd.Series,
    returns: pd.Series,
    benchmark_equity: Optional[pd.Series],
    benchmark_returns: Optional[pd.Series],
    strategy_label: str,
    benchmark_label: str,
) -> None:
    ax.axis("off")
    rows = {strategy_label: _key_metrics(equity, returns)}
    if benchmark_equity is not None and not _clean_series(benchmark_equity).empty:
        bench_returns = (
            benchmark_returns
            if benchmark_returns is not None
            else _clean_series(benchmark_equity).pct_change()
        )
        rows[benchmark_label] = _key_metrics(benchmark_equity, bench_returns)

    frame = pd.DataFrame(rows).T
    table = ax.table(
        cellText=frame.to_numpy(),
        rowLabels=frame.index.tolist(),
        colLabels=frame.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 2.0)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(_TABLE_EDGE)
        if row == 0:
            cell.set_facecolor(_STRATEGY_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        elif col == -1:
            cell.set_text_props(fontweight="bold", ha="left")
            cell.set_facecolor("#f2f2f2")
        elif row == 1:
            cell.set_text_props(fontweight="bold", color=_STRATEGY_COLOR)


def _annotate_last_value(ax: plt.Axes, curve: pd.Series, color: str) -> None:
    if curve.empty:
        return
    ax.annotate(
        f"{curve.iloc[-1]:.2f}x",
        xy=(curve.index[-1], curve.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        color=color,
        fontsize=9,
        fontweight="bold",
        va="center",
    )


def _plot_equity_curve(
    ax: plt.Axes,
    equity: pd.Series,
    benchmark_equity: Optional[pd.Series],
    strategy_label: str,
    benchmark_label: str,
) -> None:
    strategy = _rebase(equity)
    ax.plot(strategy.index, strategy.to_numpy(), color=_STRATEGY_COLOR,
            linewidth=1.6, label=strategy_label)
    _annotate_last_value(ax, strategy, _STRATEGY_COLOR)

    if benchmark_equity is not None:
        benchmark = _rebase(benchmark_equity)
        if not benchmark.empty:
            ax.plot(benchmark.index, benchmark.to_numpy(), color=_BENCHMARK_COLOR,
                    linewidth=1.4, linestyle="--", label=benchmark_label)
            _annotate_last_value(ax, benchmark, _BENCHMARK_COLOR)

    if not strategy.empty and strategy.max() > (1.0 + _SEMILOG_THRESHOLD):
        ax.set_yscale("log")
        ax.set_ylabel("Growth of $1 (log scale)")
    else:
        ax.set_ylabel("Growth of $1")

    ax.set_title("Cumulative Return", fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    _despine(ax)
    _rotate_date_labels(ax)


def _plot_underwater(ax: plt.Axes, equity: pd.Series) -> None:
    drawdown = _drawdown(equity)
    ax.set_title("Underwater Plot (Drawdown)", fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)
    _despine(ax)

    if drawdown.empty:
        return

    values = drawdown.to_numpy()
    ax.fill_between(drawdown.index, values, 0.0, color=_DRAWDOWN_COLOR, alpha=0.4)
    ax.plot(drawdown.index, values, color=_DRAWDOWN_COLOR, linewidth=1.0)
    ax.yaxis.set_major_formatter(_percent_formatter(0))
    # Non-degenerate lower bound: with no drawdown the axis would collapse to (0, 0).
    lower = float(values.min())
    ax.set_ylim(bottom=min(lower * 1.12, -0.01), top=0.0)

    if lower < 0.0:
        trough = drawdown.idxmin()
        # Put the label left of the trough when it falls late in the sample.
        late = drawdown.index.get_loc(trough) > 0.75 * len(drawdown)
        ax.scatter([trough], [lower], color=_DRAWDOWN_COLOR, s=20, zorder=3)
        when = f" ({trough:%b %Y})" if isinstance(trough, pd.Timestamp) else ""
        ax.annotate(
            f"Max DD {_format_pct(lower, 1)}{when}",
            xy=(trough, lower),
            xytext=(-8, 0) if late else (8, 0),
            textcoords="offset points",
            ha="right" if late else "left",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=_DRAWDOWN_COLOR,
        )
    _rotate_date_labels(ax)


def _plot_rolling_sharpe(ax: plt.Axes, returns: pd.Series) -> None:
    sharpe = _rolling_sharpe(returns, _ROLLING_WINDOW)
    ax.set_title(f"Rolling Sharpe Ratio ({_ROLLING_WINDOW}-day, annualized)", fontweight="bold")
    ax.set_ylabel("Sharpe ratio")
    ax.grid(True, alpha=0.3)
    _despine(ax)

    valid = sharpe.dropna() if not sharpe.empty else sharpe
    if not valid.empty:
        ax.plot(valid.index, valid.to_numpy(), color=_STRATEGY_COLOR, linewidth=1.3)

    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="-")
    ax.axhline(1.0, color=_POSITIVE_COLOR, linewidth=0.9, linestyle="--")
    _rotate_date_labels(ax)


def _plot_return_distribution(ax: plt.Axes, returns: pd.Series) -> None:
    clean = _clean_series(returns)
    ax.set_title("Daily Return Distribution", fontweight="bold")
    ax.set_xlabel("Daily return")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    _despine(ax)

    if clean.empty:
        return

    values = clean.to_numpy()
    # seaborn cannot fit a KDE with < 2 points or zero variance.
    use_kde = values.size >= 2 and np.std(values) > 0.0
    sns.histplot(values, kde=use_kde, ax=ax, color=_STRATEGY_COLOR,
                 alpha=0.5, edgecolor="white")

    mean_return = float(values.mean())
    var_95 = float(np.percentile(values, 5))
    ax.axvline(mean_return, color=_POSITIVE_COLOR, linestyle="--", linewidth=1.3,
               label=f"Mean {_format_pct(mean_return, 2)}")
    ax.axvline(var_95, color=_NEGATIVE_COLOR, linestyle="--", linewidth=1.3,
               label=f"VaR 95% {_format_pct(var_95, 2)}")
    ax.xaxis.set_major_formatter(_percent_formatter(1))
    ax.legend(loc="upper right", frameon=False)


def _plot_annual_returns(ax: plt.Axes, returns: pd.Series) -> None:
    clean = _clean_series(returns)
    ax.set_title("Annual Returns", fontweight="bold")
    ax.set_ylabel("Return")
    ax.grid(True, axis="y", alpha=0.3)
    ax.grid(False, axis="x")
    _despine(ax)

    if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
        return

    annual = clean.resample("YE").apply(lambda r: (1.0 + r).prod() - 1.0)
    if annual.empty:
        return

    sessions = clean.resample("YE").count()
    partial = sessions.to_numpy() < _PARTIAL_YEAR_SESSIONS
    labels = [f"{ts.year}*" if is_partial else str(ts.year)
              for ts, is_partial in zip(annual.index, partial)]
    values = annual.to_numpy()
    colors = [_POSITIVE_COLOR if v >= 0 else _NEGATIVE_COLOR for v in values]
    positions = np.arange(len(values))

    ax.bar(positions, values, color=colors, alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(_percent_formatter(0))
    _rotate_date_labels(ax)
    if partial.any():
        ax.set_title("Annual Returns (* partial year)", fontweight="bold")

    # Symmetric, non-degenerate y-range (an all-zero sample would collapse it).
    span = float(np.abs(values).max())
    if span == 0.0:
        span = 0.01
    offset = span * 0.02
    for pos, value in zip(positions, values):
        # Skip labels for flat years (e.g. a year holding only the start-up session).
        if abs(value) < 0.0005:
            continue
        va = "bottom" if value >= 0 else "top"
        ax.text(pos, value + (offset if value >= 0 else -offset),
                _format_pct(value, 1), ha="center", va=va, fontsize=8)
    ax.set_ylim(-span * 1.25, span * 1.25)


def _monthly_return_table(returns: pd.Series) -> pd.DataFrame:
    """Year x month return matrix plus a final compounded full-year column."""
    monthly = returns.resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0)
    frame = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "ret": monthly.to_numpy(),
        }
    )
    table = frame.pivot(index="year", columns="month", values="ret")
    table = table.reindex(columns=range(1, 13))
    table.columns = _MONTH_LABELS
    annual = returns.resample("YE").apply(lambda r: (1.0 + r).prod() - 1.0)
    table[_ANNUAL_COLUMN] = annual.set_axis(annual.index.year).reindex(table.index)
    return table


def _color_scale(values: np.ndarray, percentile: Optional[float]) -> float:
    """Symmetric half-width of the color scale, as a fraction."""
    finite = np.abs(values[np.isfinite(values)])
    if finite.size == 0:
        return _MIN_COLOR_SCALE
    scale = np.percentile(finite, percentile) if percentile is not None else finite.max()
    return max(float(scale), _MIN_COLOR_SCALE)


def _plot_monthly_heatmap(ax: plt.Axes, returns: pd.Series) -> None:
    clean = _clean_series(returns)
    ax.set_title("Monthly Returns (%)", fontweight="bold")

    if clean.empty or not isinstance(clean.index, pd.DatetimeIndex):
        ax.axis("off")
        return

    table = _monthly_return_table(clean)
    if table.empty:
        ax.axis("off")
        return

    # Months and the full-year column use separate color scales, since a year
    # is not comparable with a month. The monthly scale is capped at a
    # percentile so a single extreme month does not wash out the rest.
    month_scale = _color_scale(table[_MONTH_LABELS].to_numpy(), _HEATMAP_PERCENTILE)
    year_scale = _color_scale(table[_ANNUAL_COLUMN].to_numpy(), None)

    values = table * 100.0
    labels = values.round(1) + 0.0  # avoid "-0.0" in the cells
    year_only = np.zeros(values.shape, dtype=bool)
    year_only[:, -1] = True

    common = dict(
        ax=ax,
        cmap="RdYlGn",
        center=0.0,
        annot=labels.to_numpy(),
        fmt=".1f",
        annot_kws={"size": 8},
        linewidths=0.5,
        linecolor="white",
        cbar=False,
    )
    sns.heatmap(values, mask=year_only, vmin=-month_scale * 100.0,
                vmax=month_scale * 100.0, **common)
    sns.heatmap(values, mask=~year_only, vmin=-year_scale * 100.0,
                vmax=year_scale * 100.0, **common)

    ax.axvline(len(_MONTH_LABELS), color="white", linewidth=3.0)
    ax.grid(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_tearsheet(
    equity_curve: pd.Series,
    net_returns: pd.Series,
    benchmark_equity: Optional[pd.Series] = None,
    benchmark_returns: Optional[pd.Series] = None,
    output_path: str = "results/tearsheet.png",
    title: Optional[str] = None,
    footnote: Optional[str] = None,
    strategy_label: str = "Strategy",
    benchmark_label: str = "Benchmark",
) -> None:
    """Render a backtest tearsheet and save it as a PNG.

    The figure has a key-metrics header and six panels: cumulative return,
    underwater (drawdown), rolling Sharpe, daily return distribution, annual
    returns and a monthly returns heatmap.

    Args:
        equity_curve: Strategy equity curve.
        net_returns: Strategy daily net returns.
        benchmark_equity: Optional benchmark equity curve. If ``None``, the
            affected panels plot the strategy only.
        benchmark_returns: Optional benchmark daily returns; derived from
            ``benchmark_equity`` when omitted.
        output_path: Destination PNG path; parent directories are created.
        title: Figure title (defaults to "Backtest Tearsheet").
        footnote: Optional footer note (e.g. costs, execution, data source).
        strategy_label, benchmark_label: Series names used in the legend and
            the metrics header.
    """
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 18), dpi=150)
    grid = fig.add_gridspec(4, 2, height_ratios=[0.20, 1.0, 1.0, 1.05])
    kpi_ax = fig.add_subplot(grid[0, :])
    axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(2)] for row in range(1, 4)])

    try:
        _plot_key_metrics(kpi_ax, equity_curve, net_returns, benchmark_equity,
                          benchmark_returns, strategy_label, benchmark_label)
        _plot_equity_curve(axes[0, 0], equity_curve, benchmark_equity,
                           strategy_label, benchmark_label)
        _plot_underwater(axes[0, 1], equity_curve)
        _plot_rolling_sharpe(axes[1, 0], net_returns)
        _plot_return_distribution(axes[1, 1], net_returns)
        _plot_annual_returns(axes[2, 0], net_returns)
        _plot_monthly_heatmap(axes[2, 1], net_returns)

        plt.tight_layout(rect=(0, 0.02, 1, 0.955))
        fig.suptitle(title or "Backtest Tearsheet", fontsize=17, fontweight="bold", y=0.992)
        period = _period_label(equity_curve)
        if period:
            fig.text(0.5, 0.968, period, ha="center", va="center", fontsize=11, color=_MUTED_TEXT)
        if footnote:
            fig.text(0.01, 0.004, footnote, ha="left", va="bottom", fontsize=9, color=_MUTED_TEXT)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)
