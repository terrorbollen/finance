"""Metrics calculation for backtesting results."""

import math
from collections import defaultdict
from datetime import date
from typing import TypedDict

import numpy as np
from scipy import stats

from backtesting.results import (
    ClassMetrics,
    HorizonMetrics,
    HorizonPrediction,
    Signal,
    TradeRecord,
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
    sharpe_ci: tuple[float, float]
    win_rate_ci: tuple[float, float]
    net_return_ci: tuple[float, float]
    trades: list  # list[TradeRecord] — typed as list to avoid circular import

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
        enforce_position_cooldown: bool = False,
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
            enforce_position_cooldown: If True, after a non-HOLD trade skip the next
                                       horizon predictions to avoid overlapping positions.
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.commission_pct = commission_pct
        self.slippage_factor = slippage_factor
        self.leverage = leverage
        self.enforce_position_cooldown = enforce_position_cooldown

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
        trading = self._calculate_trading_metrics(completed, horizon=horizon)

        metrics = HorizonMetrics(
            horizon_days=horizon,
            total_predictions=len(completed),
            accuracy=accuracy,
            class_metrics=class_metrics,
            calibration=calibration,
            price_mae=price_mae,
            price_rmse=price_rmse,
            **trading,
        )

        # Additional validation diagnostics
        metrics.brier_score = self._calculate_brier_score(completed)
        metrics.ece = self._calculate_ece(calibration, completed)
        metrics.roc_auc = self._calculate_roc_auc(completed)
        metrics.temporal_accuracy = self._calculate_temporal_accuracy(completed)
        metrics.regime_metrics = self._calculate_regime_metrics(completed)

        return metrics

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

    def _calculate_trading_metrics(
        self,
        predictions: list[HorizonPrediction],
        horizon: int = 1,
    ) -> TradingMetrics:
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
        # Apply position cooldown: after a non-HOLD trade, skip the next
        # `horizon` predictions to avoid overlapping positions.
        if self.enforce_position_cooldown and horizon > 1:
            sorted_preds = sorted(predictions, key=lambda p: p.prediction_date)
            cooldown_remaining = 0
            filtered: list[HorizonPrediction] = []
            for p in sorted_preds:
                if cooldown_remaining > 0:
                    cooldown_remaining -= 1
                    continue
                filtered.append(p)
                if p.predicted_signal != Signal.HOLD and p.actual_price_change is not None:
                    cooldown_remaining = horizon - 1
            predictions = filtered

        gross_returns: list[float] = []
        net_returns: list[float] = []
        equity_points: list[tuple[date, float]] = []  # (prediction_date, net_return)
        trades_list: list = []
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
            trades_list.append(TradeRecord(
                date=p.prediction_date,
                signal=p.predicted_signal,
                gross_pct=round(gross, 4),
                net_pct=round(net, 4),
                is_winner=(net > 0),
            ))

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
            "sharpe_ci": (0.0, 0.0),
            "win_rate_ci": (0.0, 0.0),
            "net_return_ci": (0.0, 0.0),
            "trades": [],
        }
        if not gross_returns:
            return empty

        trade_count = len(gross_returns)
        wins = sum(1 for r in gross_returns if r > 0)
        sharpe, sortino, max_dd, calmar = self._calculate_risk_metrics(net_returns)

        # Binomial test: is win rate significantly above 50% chance?
        pvalue = float(stats.binomtest(wins, trade_count, p=0.5, alternative="greater").pvalue)

        # Bootstrap 95% confidence intervals
        def _sharpe_fn(arr: np.ndarray) -> float:
            s = float(np.std(arr, ddof=1))
            return float((np.mean(arr) / s) * np.sqrt(252)) if s > 1e-8 else 0.0

        sharpe_ci = self._bootstrap_ci(net_returns, _sharpe_fn)
        win_rate_ci = self._bootstrap_ci(
            net_returns,
            lambda arr: float(np.sum(arr > 0) / len(arr)),
        )
        net_return_ci = self._bootstrap_ci(
            net_returns,
            lambda arr: float(np.sum(arr)),
        )

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
            "sharpe_ci": sharpe_ci,
            "win_rate_ci": win_rate_ci,
            "net_return_ci": net_return_ci,
            "trades": trades_list,
        }

    def _bootstrap_ci(
        self,
        values: list[float],
        stat_fn,
        n_boot: int = 500,
    ) -> tuple[float, float]:
        """95% bootstrap confidence interval for a statistic."""
        if len(values) < 5:
            return (0.0, 0.0)
        rng = np.random.default_rng(42)
        arr = np.array(values)
        n = len(arr)
        boot_stats = sorted(
            stat_fn(arr[rng.integers(0, n, size=n)]) for _ in range(n_boot)
        )
        return (boot_stats[int(0.025 * n_boot)], boot_stats[int(0.975 * n_boot)])

    def _calculate_brier_score(self, predictions: list[HorizonPrediction]) -> float:
        """Multi-class Brier score. Uses all_probs if available, else falls back to confidence."""
        signal_to_idx = {Signal.BUY: 0, Signal.HOLD: 1, Signal.SELL: 2}
        total = 0.0
        n = 0
        for p in predictions:
            if p.actual_signal is None:
                continue
            actual_idx = signal_to_idx[p.actual_signal]
            if p.all_probs is not None:
                probs = p.all_probs
            else:
                # Fallback: confidence on predicted class, remainder split equally
                pred_idx = signal_to_idx[p.predicted_signal]
                remainder = (1.0 - p.confidence) / 2.0
                probs_list = [remainder, remainder, remainder]
                probs_list[pred_idx] = p.confidence
                probs = (probs_list[0], probs_list[1], probs_list[2])
            one_hot = [1.0 if i == actual_idx else 0.0 for i in range(3)]
            total += sum((probs[i] - one_hot[i]) ** 2 for i in range(3))
            n += 1
        return total / n if n > 0 else 0.0

    def _calculate_ece(
        self,
        calibration: dict[str, float],
        predictions: list[HorizonPrediction],
    ) -> float:
        """Expected Calibration Error: weighted mean |accuracy - avg_confidence| per bucket."""
        bucket_ranges = {
            "50-60%": (0.50, 0.60), "60-70%": (0.60, 0.70),
            "70-80%": (0.70, 0.80), "80-90%": (0.80, 0.90), "90-100%": (0.90, 1.01),
        }
        n = len(predictions)
        if n == 0:
            return 0.0
        ece = 0.0
        for bucket_name, actual_acc in calibration.items():
            lo, hi = bucket_ranges[bucket_name]
            bucket_preds = [p for p in predictions if lo <= p.confidence < hi]
            if bucket_preds:
                avg_conf = float(np.mean([p.confidence for p in bucket_preds]))
                ece += (len(bucket_preds) / n) * abs(actual_acc - avg_conf)
        return ece

    def _calculate_roc_auc(self, predictions: list[HorizonPrediction]) -> dict[Signal, float]:
        """One-vs-rest ROC AUC per signal class. Requires all_probs to be set."""
        from sklearn.metrics import roc_auc_score

        preds = [p for p in predictions if p.all_probs is not None and p.actual_signal is not None]
        if len(preds) < 10:
            return {}

        signal_to_idx = {Signal.BUY: 0, Signal.HOLD: 1, Signal.SELL: 2}
        result: dict[Signal, float] = {}
        for signal in Signal:
            idx = signal_to_idx[signal]
            y_true = [1 if p.actual_signal == signal else 0 for p in preds]
            if sum(y_true) == 0 or sum(y_true) == len(y_true):
                continue
            y_score = [p.all_probs[idx] for p in preds]  # type: ignore[index]
            try:
                result[signal] = float(roc_auc_score(y_true, y_score))
            except Exception:
                pass
        return result

    def _calculate_temporal_accuracy(
        self, predictions: list[HorizonPrediction]
    ) -> list[tuple[date, float]]:
        """Monthly accuracy: (first_of_month, accuracy) sorted chronologically."""
        monthly: dict[tuple[int, int], list[bool]] = defaultdict(list)
        for p in predictions:
            if p.is_correct is not None:
                monthly[(p.prediction_date.year, p.prediction_date.month)].append(p.is_correct)
        return [
            (date(y, m, 1), sum(correct) / len(correct))
            for (y, m), correct in sorted(monthly.items())
            if correct
        ]

    def _calculate_regime_metrics(
        self, predictions: list[HorizonPrediction]
    ) -> dict[str, dict]:
        """Split accuracy and win rate by ADX regime (ranging <20, trending 20-40, strong >40)."""
        regimes = {
            "ranging": lambda adx: adx < 20,
            "trending": lambda adx: 20 <= adx < 40,
            "strong_trend": lambda adx: adx >= 40,
        }
        result: dict[str, dict] = {}
        for name, condition in regimes.items():
            regime_preds = [p for p in predictions if p.adx is not None and condition(p.adx)]
            if not regime_preds:
                continue
            accuracy = sum(1 for p in regime_preds if p.is_correct) / len(regime_preds)
            trades = [
                p for p in regime_preds
                if p.predicted_signal != Signal.HOLD and p.actual_price_change is not None
            ]
            wins = sum(
                1 for p in trades
                if (p.predicted_signal == Signal.BUY and p.actual_price_change > 0)
                or (p.predicted_signal == Signal.SELL and p.actual_price_change < 0)
            )
            result[name] = {
                "accuracy": round(accuracy, 4),
                "win_rate": round(wins / len(trades), 4) if trades else 0.0,
                "n_predictions": len(regime_preds),
                "n_trades": len(trades),
            }
        return result

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

        bars_per_year = 252
        ann_factor = np.sqrt(bars_per_year)
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
