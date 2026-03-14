"""Metrics calculation for backtesting results."""

import numpy as np
from collections import defaultdict
from typing import Optional

from backtesting.results import (
    Signal,
    HorizonPrediction,
    HorizonMetrics,
    ClassMetrics,
)


class MetricsCalculator:
    """Calculate metrics from backtest predictions."""

    def __init__(
        self,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
    ):
        """
        Initialize metrics calculator.

        Args:
            buy_threshold: Price change threshold for BUY signal (default 2%)
            sell_threshold: Price change threshold for SELL signal (default -2%)
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

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
        win_rate, total_return = self._calculate_trading_metrics(completed)

        return HorizonMetrics(
            horizon_days=horizon,
            total_predictions=len(completed),
            accuracy=accuracy,
            class_metrics=class_metrics,
            calibration=calibration,
            price_mae=price_mae,
            price_rmse=price_rmse,
            win_rate=win_rate,
            total_return=total_return,
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
                1 for p in predictions
                if p.predicted_signal == signal_class and p.actual_signal == signal_class
            )

            # False positives: predicted this class but it was wrong
            fp = sum(
                1 for p in predictions
                if p.predicted_signal == signal_class and p.actual_signal != signal_class
            )

            # False negatives: actual was this class but predicted something else
            fn = sum(
                1 for p in predictions
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

    def _calculate_calibration(
        self, predictions: list[HorizonPrediction]
    ) -> dict[str, float]:
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
            bucket_preds = [
                p for p in predictions
                if low <= p.confidence < high
            ]

            if bucket_preds:
                accuracy = sum(1 for p in bucket_preds if p.is_correct) / len(bucket_preds)
                calibration[bucket_name] = accuracy

        return calibration

    def _calculate_price_metrics(
        self, predictions: list[HorizonPrediction]
    ) -> tuple[float, float]:
        """Calculate MAE and RMSE for price predictions."""
        # Filter predictions with price data
        with_prices = [
            p for p in predictions
            if p.actual_price_change is not None
        ]

        if not with_prices:
            return 0.0, 0.0

        errors = [
            abs(p.predicted_price_change - p.actual_price_change)
            for p in with_prices
        ]
        squared_errors = [
            (p.predicted_price_change - p.actual_price_change) ** 2
            for p in with_prices
        ]

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(squared_errors))

        return float(mae), float(rmse)

    def _calculate_trading_metrics(
        self, predictions: list[HorizonPrediction]
    ) -> tuple[float, float]:
        """
        Calculate simulated trading metrics.

        Simple strategy:
        - BUY signal: go long
        - SELL signal: go short
        - HOLD signal: no position

        Returns:
            Tuple of (win_rate, total_return_percentage)
        """
        trades = []

        for p in predictions:
            if p.actual_price_change is None:
                continue

            if p.predicted_signal == Signal.BUY:
                # Long position: profit if price goes up
                trade_return = p.actual_price_change
                trades.append(trade_return)
            elif p.predicted_signal == Signal.SELL:
                # Short position: profit if price goes down
                trade_return = -p.actual_price_change
                trades.append(trade_return)
            # HOLD: no trade

        if not trades:
            return 0.0, 0.0

        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades)
        total_return = sum(trades)

        return win_rate, total_return

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
