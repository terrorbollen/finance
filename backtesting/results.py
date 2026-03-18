"""Dataclasses for storing backtest predictions and results."""

import json
from dataclasses import dataclass, field
from datetime import date

from signals.direction import Direction as Signal


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation results for a single horizon's trade sequence.

    Produced by randomly shuffling the realized trade returns N times and
    computing full-path metrics (total return, max drawdown, Sharpe) for each
    permutation.  This reveals whether the observed backtest result is a
    genuine edge or just a lucky ordering of the same trades.
    """

    n_simulations: int

    # Total return distribution
    mean_total_return: float
    std_total_return: float
    p5_total_return: float
    p95_total_return: float
    # Percentile rank of the observed result (0–100). >50 means the strategy
    # outperformed the average random ordering.
    observed_total_return_pct: float

    # Max drawdown distribution (values are negative percentages)
    mean_max_drawdown: float
    p5_max_drawdown: float
    p95_max_drawdown: float

    # Sharpe ratio distribution
    mean_sharpe: float
    p5_sharpe: float
    p95_sharpe: float
    observed_sharpe_pct: float

    def to_dict(self) -> dict:
        return {
            "n_simulations": self.n_simulations,
            "mean_total_return": round(self.mean_total_return, 4),
            "std_total_return": round(self.std_total_return, 4),
            "p5_total_return": round(self.p5_total_return, 4),
            "p95_total_return": round(self.p95_total_return, 4),
            "observed_total_return_pct": round(self.observed_total_return_pct, 1),
            "mean_max_drawdown": round(self.mean_max_drawdown, 4),
            "p5_max_drawdown": round(self.p5_max_drawdown, 4),
            "p95_max_drawdown": round(self.p95_max_drawdown, 4),
            "mean_sharpe": round(self.mean_sharpe, 4),
            "p5_sharpe": round(self.p5_sharpe, 4),
            "p95_sharpe": round(self.p95_sharpe, 4),
            "observed_sharpe_pct": round(self.observed_sharpe_pct, 1),
        }


@dataclass
class HorizonPrediction:
    """Single horizon prediction made on a specific date."""

    prediction_date: date
    horizon_days: int
    predicted_signal: Signal
    confidence: float  # 0-1
    predicted_price_change: float  # percentage

    # Actual outcomes (filled in after horizon passes)
    actual_signal: Signal | None = None
    actual_price_change: float | None = None
    target_date: date | None = None

    # Volume data for slippage modeling (today / rolling_20d_avg; None if unavailable)
    relative_volume: float | None = None

    # ADX(14) on the prediction date — used for regime filtering (None if unavailable)
    adx: float | None = None

    # Full softmax output [p_buy, p_hold, p_sell] (None if not available)
    all_probs: tuple[float, float, float] | None = None  # [p_buy, p_hold, p_sell]

    @property
    def is_correct(self) -> bool | None:
        """Check if the predicted signal matches the actual outcome."""
        if self.actual_signal is None:
            return None
        return self.predicted_signal == self.actual_signal

    def to_dict(self) -> dict:
        return {
            "prediction_date": self.prediction_date.isoformat(),
            "horizon_days": self.horizon_days,
            "predicted_signal": self.predicted_signal.name,
            "confidence": self.confidence,
            "predicted_price_change": self.predicted_price_change,
            "actual_signal": self.actual_signal.name if self.actual_signal else None,
            "actual_price_change": self.actual_price_change,
            "target_date": self.target_date.isoformat() if self.target_date else None,
            "is_correct": self.is_correct,
            "relative_volume": self.relative_volume,
            "adx": self.adx,
            "all_probs": list(self.all_probs) if self.all_probs is not None else None,
        }


@dataclass
class DailyPrediction:
    """All predictions made on a single day for multiple horizons."""

    date: date
    current_price: float
    predictions: dict[int, HorizonPrediction] = field(default_factory=dict)  # horizon -> prediction

    def add_prediction(self, prediction: HorizonPrediction):
        self.predictions[prediction.horizon_days] = prediction

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "current_price": self.current_price,
            "predictions": {h: p.to_dict() for h, p in self.predictions.items()},
        }


@dataclass
class ClassMetrics:
    """Precision and recall for a single class."""

    precision: float
    recall: float
    support: int  # number of actual instances

    def to_dict(self) -> dict:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "support": self.support,
        }


@dataclass
class TradeRecord:
    """Single simulated trade outcome."""
    date: date
    signal: Signal
    gross_pct: float
    net_pct: float
    is_winner: bool

    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "signal": self.signal.name,
            "gross_pct": self.gross_pct,
            "net_pct": self.net_pct,
            "is_winner": self.is_winner,
        }


@dataclass
class HorizonMetrics:
    """Aggregated metrics for a single prediction horizon."""

    horizon_days: int
    total_predictions: int

    # Signal accuracy
    accuracy: float
    class_metrics: dict[Signal, ClassMetrics] = field(default_factory=dict)

    # Confidence calibration (confidence_bucket -> actual_accuracy)
    calibration: dict[str, float] = field(default_factory=dict)

    # Price prediction metrics
    price_mae: float = 0.0
    price_rmse: float = 0.0

    # Simulated trading metrics
    win_rate: float = 0.0
    total_return: float = 0.0  # gross return (%)
    net_total_return: float = 0.0  # after transaction costs (%)
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0  # negative percentage
    calmar_ratio: float = 0.0

    # Reliability / significance
    trade_count: int = 0
    low_sample: bool = False  # True when trade_count < 30
    win_rate_pvalue: float = 1.0  # one-sided binomial test vs 50% chance

    # Equity curve: sorted (date, cumulative_net_return_pct) pairs
    equity_curve: list[tuple[date, float]] = field(default_factory=list)

    # Calibration quality
    brier_score: float = 0.0
    ece: float = 0.0  # Expected Calibration Error

    # Per-class ROC AUC (one-vs-rest)
    roc_auc: dict[Signal, float] = field(default_factory=dict)

    # Bootstrap 95% confidence intervals
    sharpe_ci: tuple[float, float] = (0.0, 0.0)
    win_rate_ci: tuple[float, float] = (0.0, 0.0)
    net_return_ci: tuple[float, float] = (0.0, 0.0)

    # Temporal and regime breakdowns
    temporal_accuracy: list[tuple[date, float]] = field(default_factory=list)  # (month_start, accuracy)
    regime_metrics: dict[str, dict] = field(default_factory=dict)  # ADX regime → metrics

    # Per-trade records
    trades: list[TradeRecord] = field(default_factory=list)

    # Benjamini-Hochberg FDR-corrected p-value (set by BacktestResult after all horizons computed)
    bh_corrected_pvalue: float = 1.0

    # Monte Carlo simulation over trade-return permutations (None when trade_count < 5)
    monte_carlo: MonteCarloResult | None = None

    def to_dict(self) -> dict:
        return {
            "horizon_days": self.horizon_days,
            "total_predictions": self.total_predictions,
            "accuracy": self.accuracy,
            "class_metrics": {s.name: m.to_dict() for s, m in self.class_metrics.items()},
            "calibration": self.calibration,
            "price_mae": self.price_mae,
            "price_rmse": self.price_rmse,
            "win_rate": self.win_rate,
            "total_return": self.total_return,
            "net_total_return": self.net_total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "trade_count": self.trade_count,
            "low_sample": self.low_sample,
            "win_rate_pvalue": self.win_rate_pvalue,
            "equity_curve": [(d.isoformat(), v) for d, v in self.equity_curve],
            "brier_score": self.brier_score,
            "ece": self.ece,
            "roc_auc": {s.name: v for s, v in self.roc_auc.items()},
            "sharpe_ci": list(self.sharpe_ci),
            "win_rate_ci": list(self.win_rate_ci),
            "net_return_ci": list(self.net_return_ci),
            "temporal_accuracy": [(d.isoformat(), v) for d, v in self.temporal_accuracy],
            "regime_metrics": self.regime_metrics,
            "trades": [t.to_dict() for t in self.trades],
            "bh_corrected_pvalue": self.bh_corrected_pvalue,
            "monte_carlo": self.monte_carlo.to_dict() if self.monte_carlo is not None else None,
        }


@dataclass
class BacktestResult:
    """Complete backtest results with all predictions and metrics."""

    ticker: str
    start_date: date
    end_date: date
    trading_days: int
    buy_hold_return: float
    leverage: float = 1.0
    benchmark_return: float | None = None  # OMXS30 return over the same period (None if unavailable)

    daily_predictions: list[DailyPrediction] = field(default_factory=list)
    horizon_metrics: dict[int, HorizonMetrics] = field(default_factory=dict)

    def summary(self) -> str:
        """Generate a human-readable summary of backtest results."""
        leverage_str = f"{self.leverage:.1f}x leverage" if self.leverage != 1.0 else "no leverage"
        lines = [
            "=" * 80,
            f"{'BACKTEST RESULTS: ' + self.ticker:^80}",
            "=" * 80,
            f"Period: {self.start_date} to {self.end_date} ({self.trading_days} trading days)  |  {leverage_str}",
            "",
            "SIGNAL ACCURACY BY HORIZON (Primary Metrics)",
            "-" * 80,
            f"{'Horizon':<10} {'Accuracy':<12} {'BUY Prec/Rec':<16} {'HOLD Prec/Rec':<16} {'SELL Prec/Rec':<16}",
        ]

        for horizon in sorted(self.horizon_metrics.keys()):
            metrics = self.horizon_metrics[horizon]

            buy_metrics = metrics.class_metrics.get(Signal.BUY)
            hold_metrics = metrics.class_metrics.get(Signal.HOLD)
            sell_metrics = metrics.class_metrics.get(Signal.SELL)

            buy_str = (
                f"{buy_metrics.precision * 100:.1f}%/{buy_metrics.recall * 100:.1f}%"
                if buy_metrics
                else "N/A"
            )
            hold_str = (
                f"{hold_metrics.precision * 100:.1f}%/{hold_metrics.recall * 100:.1f}%"
                if hold_metrics
                else "N/A"
            )
            sell_str = (
                f"{sell_metrics.precision * 100:.1f}%/{sell_metrics.recall * 100:.1f}%"
                if sell_metrics
                else "N/A"
            )

            lines.append(
                f"{horizon}-day{'':<5} {metrics.accuracy * 100:.1f}%{'':<7} {buy_str:<16} {hold_str:<16} {sell_str:<16}"
            )

        lines.extend(
            [
                "",
                "CONFIDENCE CALIBRATION",
                "-" * 80,
                f"{'Confidence':<15} {'Actual Accuracy':<20} (Is model well-calibrated?)",
            ]
        )

        # Get calibration from first horizon with data
        if self.horizon_metrics:
            first_metrics = next(iter(self.horizon_metrics.values()))
            for bucket, actual in sorted(first_metrics.calibration.items()):
                lines.append(f"{bucket:<15} {actual * 100:.1f}%")

        lines.extend(
            [
                "",
                "SIMULATED TRADING PERFORMANCE (per horizon)",
                "-" * 80,
                "  ** p<0.01  * p<0.05  ⚠ <30 trades (low sample — treat metrics with caution)",
                f"{'Horizon':<10} {'Trades':<10} {'Win Rate':<14} {'Gross Ret':<12} {'Net Ret':<12} {'Sharpe':<8} {'Max DD':<10} {'Calmar':<8}",
            ]
        )

        for horizon in sorted(self.horizon_metrics.keys()):
            m = self.horizon_metrics[horizon]
            dd_str = f"{m.max_drawdown:+.1f}%" if m.max_drawdown != 0.0 else "N/A"
            calmar_str = f"{m.calmar_ratio:.2f}" if m.calmar_ratio != 0.0 else "N/A"

            # Significance marker for win rate
            if m.win_rate_pvalue < 0.01:
                sig = "**"
            elif m.win_rate_pvalue < 0.05:
                sig = "*"
            else:
                sig = ""
            low_marker = " ⚠" if m.low_sample else ""
            win_str = f"{m.win_rate * 100:.1f}%{sig}"
            trade_str = f"{m.trade_count}{low_marker}"

            lines.append(
                f"{horizon}-day{'':<5} {trade_str:<10} {win_str:<14} "
                f"{m.total_return:+.2f}%{'':<5} {m.net_total_return:+.2f}%{'':<5} "
                f"{m.sharpe_ratio:.2f}{'':<4} {dd_str:<10} {calmar_str}"
            )

        lines.append("")
        lines.append("BENCHMARK COMPARISON")
        lines.append("-" * 80)
        lines.append(f"Buy & Hold ({self.ticker + '):':20s} {self.buy_hold_return:+.2f}%")
        if self.benchmark_return is not None:
            lines.append(f"Index      (OMXS30):          {self.benchmark_return:+.2f}%")
        lines.append("(Strategy net returns shown per horizon in table above)")

        lines.append("")
        lines.append("VALIDATION DIAGNOSTICS")
        lines.append("-" * 80)
        lines.append(f"{'Horizon':<10} {'Brier':<8} {'ECE':<8} {'Win CI (95%)':<20} {'Sharpe CI (95%)':<20} {'BH p-val':<10} {'Regime coverage'}")
        lines.append("-" * 80)
        for horizon in sorted(self.horizon_metrics.keys()):
            m = self.horizon_metrics[horizon]
            win_ci = f"[{m.win_rate_ci[0]*100:.1f}%, {m.win_rate_ci[1]*100:.1f}%]" if m.win_rate_ci != (0.0, 0.0) else "N/A"
            sharpe_ci = f"[{m.sharpe_ci[0]:.2f}, {m.sharpe_ci[1]:.2f}]" if m.sharpe_ci != (0.0, 0.0) else "N/A"
            bh_str = f"{m.bh_corrected_pvalue:.3f}"
            regime_str = ", ".join(f"{k}:{v['n_trades']}t" for k, v in m.regime_metrics.items()) if m.regime_metrics else "N/A (no ADX)"
            lines.append(f"{horizon}-day{'':<5} {m.brier_score:.3f}   {m.ece:.3f}   {win_ci:<20} {sharpe_ci:<20} {bh_str:<10} {regime_str}")
        lines.extend(["", "MONTE CARLO SIMULATION (trade-order permutations)"])
        lines.append("-" * 80)
        lines.append(
            "Shuffles realized trade returns 1 000 times to test path-dependency."
        )
        lines.append(
            "Observed pct = where your result ranks vs random orderings (>50 is above average)."
        )
        lines.append(
            f"{'Horizon':<10} {'MC Return mean [p5,p95]':<30} {'Obs pct':<10} {'MC Sharpe mean [p5,p95]':<30} {'Obs pct'}"
        )
        lines.append("-" * 80)
        for horizon in sorted(self.horizon_metrics.keys()):
            m = self.horizon_metrics[horizon]
            if m.monte_carlo is None:
                lines.append(f"{horizon}-day{'':<5} N/A (fewer than 5 trades)")
                continue
            mc = m.monte_carlo
            ret_str = (
                f"{mc.mean_total_return:+.1f}% [{mc.p5_total_return:+.1f}%, {mc.p95_total_return:+.1f}%]"
            )
            sharpe_str = (
                f"{mc.mean_sharpe:.2f} [{mc.p5_sharpe:.2f}, {mc.p95_sharpe:.2f}]"
            )
            lines.append(
                f"{horizon}-day{'':<5} {ret_str:<30} {mc.observed_total_return_pct:.0f}%{'':<5} "
                f"{sharpe_str:<30} {mc.observed_sharpe_pct:.0f}%"
            )
        lines.append("=" * 80)

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON export."""
        return {
            "ticker": self.ticker,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "trading_days": self.trading_days,
            "buy_hold_return": self.buy_hold_return,
            "benchmark_return": self.benchmark_return,
            "daily_predictions": [d.to_dict() for d in self.daily_predictions],
            "horizon_metrics": {h: m.to_dict() for h, m in self.horizon_metrics.items()},
        }

    def export_json(self, path: str):
        """Export results to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def export_equity_curve_csv(self, path: str, horizon: int):
        """Export cumulative equity curve for a given horizon to CSV.

        Columns: date, cumulative_net_return_pct
        """
        import csv

        metrics = self.horizon_metrics.get(horizon)
        if metrics is None or not metrics.equity_curve:
            return

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["date", "cumulative_net_return_pct"])
            for dt, value in metrics.equity_curve:
                writer.writerow([dt.isoformat(), value])

    def export_trades_csv(self, path: str) -> None:
        """Export per-trade records for all horizons to CSV."""
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["horizon_days", "date", "signal", "gross_pct", "net_pct", "is_winner"])
            for h, m in sorted(self.horizon_metrics.items()):
                for t in m.trades:
                    writer.writerow([h, t.date.isoformat(), t.signal.name,
                                      round(t.gross_pct, 4), round(t.net_pct, 4), t.is_winner])

    def export_csv(self, path: str):
        """Export predictions to CSV file."""
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "prediction_date",
                    "horizon_days",
                    "predicted_signal",
                    "confidence",
                    "predicted_price_change",
                    "actual_signal",
                    "actual_price_change",
                    "target_date",
                    "is_correct",
                ]
            )

            for daily in self.daily_predictions:
                for _horizon, pred in daily.predictions.items():
                    writer.writerow(
                        [
                            pred.prediction_date.isoformat(),
                            pred.horizon_days,
                            pred.predicted_signal.name,
                            pred.confidence,
                            pred.predicted_price_change,
                            pred.actual_signal.name if pred.actual_signal else "",
                            pred.actual_price_change
                            if pred.actual_price_change is not None
                            else "",
                            pred.target_date.isoformat() if pred.target_date else "",
                            pred.is_correct if pred.is_correct is not None else "",
                        ]
                    )
