"""Metrics calculation for backtesting results."""

import math
from datetime import date
from typing import TypedDict

import numpy as np
from scipy import stats

from backtesting.results import (
    ClassMetrics,
    HorizonMetrics,
    HorizonPrediction,
    Signal,
)


class TradingMetrics(TypedDict):
    """Return type of _calculate_trading_metrics — maps directly onto HorizonMetrics fields."""

    win_rate: float
    total_return: float
    net_total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    trade_count: int
    low_sample: bool
    win_rate_pvalue: float
    equity_curve: list[tuple[date, float]]

# Maximum per-trade slippage cost as a percentage (0.5%)
_MAX_SLIPPAGE_PCT = 0.5

# Minimum trades needed for metrics to be considered reliable
MIN_TRADES_FOR_RELIABILITY = 30


class MetricsCalculator:
    """Calculate metrics from backtest predictions."""

    def __init__(
        self,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
        commission_pct: float = 0.001,
        slippage_factor: float = 0.0,
        leverage: float = 1.0,
    ):
        """
        Initialize metrics calculator.

        Args:
            buy_threshold: Price change threshold for BUY signal (default 2%)
            sell_threshold: Price change threshold for SELL signal (default -2%)
            commission_pct: One-way commission as decimal (default 0.1% = 0.001)
            slippage_factor: Scaling constant for volume-based slippage.
                             slippage_pct = slippage_factor / sqrt(relative_volume),
                             capped at 0.5%. Set to 0 to disable.
            leverage: Leverage multiplier applied to each trade (default 1.0 = no leverage).
                      Commission and slippage are also scaled by leverage since they apply
                      to the full leveraged notional.
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.commission_pct = commission_pct
        self.slippage_factor = slippage_factor
        self.leverage = leverage

    def calculate_horizon_metrics(
        self,
        predictions: list[HorizonPrediction],
        horizon: int,
    ) -> HorizonMetrics:
        """
        Calculate all metrics for a single horizon.

        Args:
            predictions: List of predictions for this horizon
            horizon: The horizon in days

        Returns:
            HorizonMetrics with all calculated metrics
        """
        # Filter to predictions with actual outcomes
        completed = [p for p in predictions if p.actual_signal is not None]

        if not completed:
            return HorizonMetrics(
                horizon_days=horizon,
                total_predictions=0,
                accuracy=0.0,
            )

        # Calculate signal accuracy
        accuracy = self._calculate_accuracy(completed)

        # Calculate per-class precision/recall
        class_metrics = self._calculate_class_metrics(completed)

        # Calculate confidence calibration
        calibration = self._calculate_calibration(completed)

        # Calculate price prediction metrics
        price_mae, price_rmse = self._calculate_price_metrics(completed)

        # Calculate simulated trading returns
        trading = self._calculate_trading_metrics(completed)

        return HorizonMetrics(
            horizon_days=horizon,
            total_predictions=len(completed),
            accuracy=accuracy,
            class_metrics=class_metrics,
            calibration=calibration,
            price_mae=price_mae,
            price_rmse=price_rmse,
            **trading,
        )

    def _calculate_accuracy(self, predictions: list[HorizonPrediction]) -> float:
        """Calculate overall signal accuracy."""
        correct = sum(1 for p in predictions if p.is_correct)
        return correct / len(predictions) if predictions else 0.0

    def _calculate_class_metrics(
        self, predictions: list[HorizonPrediction]
    ) -> dict[Signal, ClassMetrics]:
        """Calculate precision and recall for each signal class."""
        metrics = {}

        for signal_class in Signal:
            # True positives: predicted this class and it was correct
            tp = sum(
                1
                for p in predictions
                if p.predicted_signal == signal_class and p.actual_signal == signal_class
            )

            # False positives: predicted this class but it was wrong
            fp = sum(
                1
                for p in predictions
                if p.predicted_signal == signal_class and p.actual_signal != signal_class
            )

            # False negatives: actual was this class but predicted something else
            fn = sum(
                1
                for p in predictions
                if p.actual_signal == signal_class and p.predicted_signal != signal_class
            )

            # Support: total actual instances of this class
            support = sum(1 for p in predictions if p.actual_signal == signal_class)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            metrics[signal_class] = ClassMetrics(
                precision=precision,
                recall=recall,
                support=support,
            )

        return metrics

    def _calculate_calibration(self, predictions: list[HorizonPrediction]) -> dict[str, float]:
        """
        Calculate confidence calibration.

        Groups predictions by confidence buckets and calculates
        actual accuracy for each bucket.
        """
        buckets = {
            "50-60%": (0.50, 0.60),
            "60-70%": (0.60, 0.70),
            "70-80%": (0.70, 0.80),
            "80-90%": (0.80, 0.90),
            "90-100%": (0.90, 1.01),
        }

        calibration = {}

        for bucket_name, (low, high) in buckets.items():
            bucket_preds = [p for p in predictions if low <= p.confidence < high]

            if bucket_preds:
                accuracy = sum(1 for p in bucket_preds if p.is_correct) / len(bucket_preds)
                calibration[bucket_name] = accuracy

        return calibration

    def _calculate_price_metrics(self, predictions: list[HorizonPrediction]) -> tuple[float, float]:
        """Calculate MAE and RMSE for price predictions."""
        # Filter predictions with price data
        with_prices = [p for p in predictions if p.actual_price_change is not None]

        if not with_prices:
            return 0.0, 0.0

        errors = [
            abs(p.predicted_price_change - p.actual_price_change)
            for p in with_prices
            if p.actual_price_change is not None
        ]
        squared_errors = [
            (p.predicted_price_change - p.actual_price_change) ** 2
            for p in with_prices
            if p.actual_price_change is not None
        ]

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(squared_errors))

        return float(mae), float(rmse)

    def _calculate_trading_metrics(self, predictions: list[HorizonPrediction]) -> TradingMetrics:
        """
        Calculate simulated trading metrics including risk-adjusted returns.

        Strategy:
        - BUY signal: go long
        - SELL signal: go short
        - HOLD signal: no position

        Commission is applied as a round-trip cost (entry + exit).

        Returns:
            Dict of trading metrics for HorizonMetrics fields.
        """
        gross_returns: list[float] = []
        net_returns: list[float] = []
        equity_points: list[tuple[date, float]] = []  # (prediction_date, net_return)
        # Commission and slippage apply to the full leveraged notional
        round_trip_cost = 2 * self.commission_pct * self.leverage * 100

        for p in predictions:
            if p.actual_price_change is None:
                continue

            if p.predicted_signal == Signal.BUY:
                gross = p.actual_price_change * self.leverage
            elif p.predicted_signal == Signal.SELL:
                gross = -p.actual_price_change * self.leverage
            else:
                continue  # HOLD: no trade

            gross_returns.append(gross)

            # Volume-based slippage (round-trip: applied at both entry and exit)
            slippage_cost = 0.0
            if self.slippage_factor > 0 and p.relative_volume is not None:
                rel_vol = max(p.relative_volume, 1e-6)  # guard against division by zero
                one_way_slippage = min(
                    self.slippage_factor / math.sqrt(rel_vol),
                    _MAX_SLIPPAGE_PCT,
                )
                slippage_cost = 2 * one_way_slippage * self.leverage  # round-trip on notional

            net = gross - round_trip_cost - slippage_cost
            net_returns.append(net)
            equity_points.append((p.prediction_date, net))

        empty: TradingMetrics = {
            "win_rate": 0.0,
            "total_return": 0.0,
            "net_total_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "calmar_ratio": 0.0,
            "trade_count": 0,
            "low_sample": True,
            "win_rate_pvalue": 1.0,
            "equity_curve": [],
        }
        if not gross_returns:
            return empty

        trade_count = len(gross_returns)
        wins = sum(1 for r in gross_returns if r > 0)
        sharpe, sortino, max_dd, calmar = self._calculate_risk_metrics(net_returns)

        # Binomial test: is win rate significantly above 50% chance?
        pvalue = float(stats.binomtest(wins, trade_count, p=0.5, alternative="greater").pvalue)

        # Build equity curve: cumulative net returns sorted chronologically
        equity_points.sort(key=lambda x: x[0])
        cumulative = 0.0
        equity_curve: list[tuple[date, float]] = []
        for dt, ret in equity_points:
            cumulative += ret
            equity_curve.append((dt, round(cumulative, 4)))

        return {
            "win_rate": wins / trade_count,
            "total_return": float(sum(gross_returns)),
            "net_total_return": float(sum(net_returns)),
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "calmar_ratio": calmar,
            "trade_count": trade_count,
            "low_sample": trade_count < MIN_TRADES_FOR_RELIABILITY,
            "win_rate_pvalue": pvalue,
            "equity_curve": equity_curve,
        }

    def _calculate_risk_metrics(
        self, returns_pct: list[float]
    ) -> tuple[float, float, float, float]:
        """
        Calculate Sharpe, Sortino, max drawdown, and Calmar ratio.

        Args:
            returns_pct: Per-trade returns as percentages (e.g. 2.0 = 2%)

        Returns:
            (sharpe_ratio, sortino_ratio, max_drawdown_pct, calmar_ratio)
        """
        if len(returns_pct) < 2:
            return 0.0, 0.0, 0.0, 0.0

        arr = np.array(returns_pct)
        mean_r = float(np.mean(arr))
        std_r = float(np.std(arr, ddof=1))

        ann_factor = np.sqrt(252)
        sharpe = float((mean_r / std_r) * ann_factor) if std_r > 1e-8 else 0.0

        downside = arr[arr < 0]
        downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-8
        sortino = float((mean_r / downside_std) * ann_factor) if downside_std > 1e-8 else 0.0

        # Max drawdown from cumulative returns
        cum = np.cumsum(arr)
        running_max = np.maximum.accumulate(cum)
        max_dd = float(np.min(cum - running_max))  # negative percentage

        total_return = float(np.sum(arr))
        calmar = total_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

        return sharpe, sortino, max_dd, calmar

    def determine_actual_signal(self, price_change: float) -> Signal:
        """
        Determine the actual signal based on price change.

        Args:
            price_change: Actual price change as a decimal (e.g., 0.02 for 2%)

        Returns:
            Signal enum value
        """
        if price_change > self.buy_threshold:
            return Signal.BUY
        elif price_change < self.sell_threshold:
            return Signal.SELL
        else:
            return Signal.HOLD
