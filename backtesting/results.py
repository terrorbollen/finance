"""Dataclasses for storing backtest predictions and results."""

import json
from dataclasses import dataclass, field
from datetime import date

from signals.direction import Direction as Signal


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
