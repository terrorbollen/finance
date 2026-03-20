"""Visualise backtest results: price chart with signal overlays + equity curve."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if TYPE_CHECKING:
    from datetime import date

from backtesting.results import BacktestResult, Signal

# ── colour / style constants ─────────────────────────────────────────────────
_BUY_COLOR = "#2ecc71"  # green
_SELL_COLOR = "#e74c3c"  # red
_HOLD_COLOR = "#95a5a6"  # grey
_PRICE_COLOR = "#2c3e50"
_EQUITY_COLORS = ["#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#f39c12", "#2ecc71", "#16a085"]
_CONF_BASE = "#5d6d7e"  # neutral slate for confidence legend circles

_MIN_MARKER_SIZE = 40
_MAX_MARKER_SIZE = 220


def _marker_sizes(confidences: list[float]) -> list[float]:
    """Scale confidence (0-1) linearly to marker-point² area."""
    return [_MIN_MARKER_SIZE + (_MAX_MARKER_SIZE - _MIN_MARKER_SIZE) * c for c in confidences]


def _conf_colors(confidences: list[float], base_hex: str) -> list[tuple]:
    """Blend each confidence from pale (low) to full base colour (high).

    At confidence=0 the marker is ~85% blended with white; at confidence=1
    it is the pure base colour.
    """
    r, g, b = mcolors.to_rgb(base_hex)
    result = []
    for c in confidences:
        t = 0.25 + 0.75 * c  # range [0.25, 1.0]
        result.append((1.0 - t * (1.0 - r), 1.0 - t * (1.0 - g), 1.0 - t * (1.0 - b)))
    return result


def plot_backtest(
    result: BacktestResult,
    save_path: str | None = None,
    show: bool = True,
) -> None:
    """Plot price + signals (top) and equity curve (bottom) for a backtest result.

    Parameters
    ----------
    result:     BacktestResult returned by Backtester.run()
    save_path:  If given, save the figure to this path (PNG, PDF, …).
    show:       Call plt.show() at the end (set False when saving only).
    """
    if not result.daily_predictions:
        print("No predictions to plot.")
        return

    available_horizons = sorted(result.horizon_metrics.keys())
    if not available_horizons:
        print("No horizon metrics to plot.")
        return

    # Use median horizon for the price panel (all horizons share the same signal for now)
    horizon = available_horizons[len(available_horizons) // 2]

    # ── collect price + signal series ─────────────────────────────────────────
    dates: list[date] = []
    prices: list[float] = []
    buy_dates: list[date] = []
    buy_prices: list[float] = []
    buy_confs: list[float] = []
    sell_dates: list[date] = []
    sell_prices: list[float] = []
    sell_confs: list[float] = []
    hold_dates: list[date] = []
    hold_prices: list[float] = []
    hold_confs: list[float] = []

    for dp in result.daily_predictions:
        dates.append(dp.date)
        prices.append(dp.current_price)

        pred = dp.predictions.get(horizon)
        if pred is None:
            continue
        conf = pred.confidence
        if pred.predicted_signal == Signal.BUY:
            buy_dates.append(dp.date)
            buy_prices.append(dp.current_price)
            buy_confs.append(conf)
        elif pred.predicted_signal == Signal.SELL:
            sell_dates.append(dp.date)
            sell_prices.append(dp.current_price)
            sell_confs.append(conf)
        else:
            hold_dates.append(dp.date)
            hold_prices.append(dp.current_price)
            hold_confs.append(conf)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig, (ax_price, ax_equity) = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        gridspec_kw={"height_ratios": [3, 2]},
        sharex=False,
    )
    fig.suptitle(
        f"{result.ticker}  |  {result.start_date} – {result.end_date}  |  signals (consensus)",
        fontsize=13,
        fontweight="bold",
    )

    # ── top panel: price + signals ────────────────────────────────────────────
    ax_price.plot(dates, prices, color=_PRICE_COLOR, linewidth=1.2, zorder=2, label="Price")

    if hold_dates:
        ax_price.scatter(
            hold_dates,
            hold_prices,
            s=_marker_sizes(hold_confs),
            c=_conf_colors(hold_confs, _HOLD_COLOR),
            alpha=0.5,
            zorder=3,
            marker="o",
            label="HOLD",
        )
    if buy_dates:
        ax_price.scatter(
            buy_dates,
            buy_prices,
            s=_marker_sizes(buy_confs),
            c=_conf_colors(buy_confs, _BUY_COLOR),
            alpha=0.9,
            zorder=4,
            marker="^",
            label="BUY",
        )
    if sell_dates:
        ax_price.scatter(
            sell_dates,
            sell_prices,
            s=_marker_sizes(sell_confs),
            c=_conf_colors(sell_confs, _SELL_COLOR),
            alpha=0.9,
            zorder=4,
            marker="v",
            label="SELL",
        )

    # confidence legend
    for conf_pct in [50, 70, 90]:
        size = _MIN_MARKER_SIZE + (_MAX_MARKER_SIZE - _MIN_MARKER_SIZE) * (conf_pct / 100)
        color = _conf_colors([conf_pct / 100], _CONF_BASE)[0]
        ax_price.scatter([], [], s=size, color=color, alpha=0.85, label=f"conf {conf_pct}%")

    ax_price.set_ylabel("Price", fontsize=10)
    ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_price.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_price.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_price.legend(loc="upper left", fontsize=8, ncol=2)
    ax_price.grid(True, alpha=0.3)

    # ── bottom panel: equity curves ───────────────────────────────────────────
    for idx, h in enumerate(available_horizons):
        m = result.horizon_metrics.get(h)
        if m is None or not m.equity_curve:
            continue
        eq_dates = [d for d, _ in m.equity_curve]
        eq_values = [v for _, v in m.equity_curve]
        eq_color = _EQUITY_COLORS[idx % len(_EQUITY_COLORS)]
        ax_equity.plot(
            eq_dates,
            eq_values,
            color=eq_color,
            linewidth=1.4,
            label=f"{h}-day (net {m.net_total_return:+.1f}%)",
        )

    bh = result.buy_hold_return
    ax_equity.axhline(
        bh, color="black", linestyle="--", linewidth=1, label=f"Buy & Hold {bh:+.1f}%"
    )
    ax_equity.axhline(0, color="grey", linestyle=":", linewidth=0.8)
    ax_equity.set_ylabel("Cumulative net return (%)", fontsize=10)
    ax_equity.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_equity.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax_equity.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax_equity.legend(loc="upper left", fontsize=8, ncol=2)
    ax_equity.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)
