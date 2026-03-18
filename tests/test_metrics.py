"""Tests for MetricsCalculator — accuracy, class metrics, risk ratios, trading P&L."""

import math
from datetime import date

import numpy as np
import pytest

from backtesting.metrics import MetricsCalculator
from backtesting.results import HorizonPrediction, Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pred(
    predicted: Signal,
    actual: Signal,
    actual_change: float = 0.0,
    predicted_change: float = 0.0,
    confidence: float = 0.75,
    relative_volume: float | None = None,
) -> HorizonPrediction:
    p = HorizonPrediction(
        prediction_date=date(2024, 1, 2),
        horizon_days=5,
        predicted_signal=predicted,
        confidence=confidence,
        predicted_price_change=predicted_change,
        relative_volume=relative_volume,
    )
    p.actual_signal = actual
    p.actual_price_change = actual_change
    return p


def _calc(**kwargs) -> MetricsCalculator:
    defaults = {
        "buy_threshold": 0.02,
        "sell_threshold": -0.02,
        "commission_pct": 0.0,
        "slippage_factor": 0.0,
    }
    defaults.update(kwargs)
    return MetricsCalculator(**defaults)


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------


class TestAccuracy:
    def test_all_correct(self):
        preds = [_pred(Signal.BUY, Signal.BUY) for _ in range(4)]
        assert _calc()._calculate_accuracy(preds) == 1.0

    def test_all_wrong(self):
        preds = [_pred(Signal.BUY, Signal.SELL) for _ in range(4)]
        assert _calc()._calculate_accuracy(preds) == 0.0

    def test_partial(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY),
            _pred(Signal.BUY, Signal.BUY),
            _pred(Signal.BUY, Signal.BUY),
            _pred(Signal.SELL, Signal.BUY),
        ]
        assert _calc()._calculate_accuracy(preds) == pytest.approx(0.75)

    def test_empty(self):
        assert _calc()._calculate_accuracy([]) == 0.0


# ---------------------------------------------------------------------------
# Class metrics (precision / recall / support)
# ---------------------------------------------------------------------------


class TestClassMetrics:
    def test_perfect_buy_precision_recall(self):
        preds = [_pred(Signal.BUY, Signal.BUY) for _ in range(3)]
        m = _calc()._calculate_class_metrics(preds)
        assert m[Signal.BUY].precision == 1.0
        assert m[Signal.BUY].recall == 1.0
        assert m[Signal.BUY].support == 3

    def test_no_buy_predictions(self):
        preds = [_pred(Signal.HOLD, Signal.BUY)]
        m = _calc()._calculate_class_metrics(preds)
        # precision undefined (0 predicted BUY) → 0.0
        assert m[Signal.BUY].precision == 0.0
        # recall: 1 actual BUY, 0 predicted → 0.0
        assert m[Signal.BUY].recall == 0.0
        assert m[Signal.BUY].support == 1

    def test_mixed(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY),  # TP for BUY
            _pred(Signal.BUY, Signal.SELL),  # FP for BUY, FN for SELL
            _pred(Signal.SELL, Signal.SELL),  # TP for SELL
        ]
        m = _calc()._calculate_class_metrics(preds)
        # BUY: tp=1, fp=1 → precision=0.5; tp=1, fn=0 → recall=1.0
        assert m[Signal.BUY].precision == pytest.approx(0.5)
        assert m[Signal.BUY].recall == pytest.approx(1.0)
        # SELL: tp=1, fp=0 → precision=1.0; tp=1, fn=1 → recall=0.5
        assert m[Signal.SELL].precision == pytest.approx(1.0)
        assert m[Signal.SELL].recall == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Calibration buckets
# ---------------------------------------------------------------------------


class TestCalibration:
    def test_high_confidence_bucket_populated(self):
        # All predictions in 80-90% bucket, all correct
        preds = [_pred(Signal.BUY, Signal.BUY, confidence=0.85) for _ in range(5)]
        cal = _calc()._calculate_calibration(preds)
        assert "80-90%" in cal
        assert cal["80-90%"] == pytest.approx(1.0)

    def test_bucket_accuracy_below_one_when_wrong(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY, confidence=0.65),
            _pred(Signal.BUY, Signal.SELL, confidence=0.65),
        ]
        cal = _calc()._calculate_calibration(preds)
        assert cal["60-70%"] == pytest.approx(0.5)

    def test_empty_bucket_not_included(self):
        preds = [_pred(Signal.BUY, Signal.BUY, confidence=0.55)]
        cal = _calc()._calculate_calibration(preds)
        assert "80-90%" not in cal


# ---------------------------------------------------------------------------
# Price metrics (MAE / RMSE)
# ---------------------------------------------------------------------------


class TestPriceMetrics:
    def test_perfect_prediction(self):
        p = _pred(Signal.BUY, Signal.BUY, actual_change=3.0, predicted_change=3.0)
        mae, rmse = _calc()._calculate_price_metrics([p])
        assert mae == pytest.approx(0.0)
        assert rmse == pytest.approx(0.0)

    def test_known_mae_rmse(self):
        # errors: 1.0 and 3.0 → MAE=2.0, RMSE=sqrt((1+9)/2)=sqrt(5)
        preds = [
            _pred(Signal.BUY, Signal.BUY, actual_change=2.0, predicted_change=1.0),
            _pred(Signal.BUY, Signal.BUY, actual_change=2.0, predicted_change=5.0),
        ]
        mae, rmse = _calc()._calculate_price_metrics(preds)
        assert mae == pytest.approx(2.0)
        assert rmse == pytest.approx(math.sqrt(5))

    def test_no_price_data(self):
        p = HorizonPrediction(
            prediction_date=date(2024, 1, 2),
            horizon_days=5,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=2.0,
        )
        # actual_price_change is None
        mae, rmse = _calc()._calculate_price_metrics([p])
        assert mae == 0.0
        assert rmse == 0.0


# ---------------------------------------------------------------------------
# Risk metrics (Sharpe / Sortino / max drawdown / Calmar)
# ---------------------------------------------------------------------------


class TestRiskMetrics:
    def test_single_return_gives_zeros(self):
        sharpe, sortino, dd, calmar = _calc()._calculate_risk_metrics([2.0])
        assert sharpe == 0.0
        assert sortino == 0.0
        assert dd == 0.0
        assert calmar == 0.0

    def test_all_positive_returns_zero_drawdown(self):
        returns = [1.0, 2.0, 3.0, 1.5]
        _, _, max_dd, _ = _calc()._calculate_risk_metrics(returns)
        assert max_dd == pytest.approx(0.0)

    def test_drawdown_known_sequence(self):
        # cumsum: [3, 1, 2, -1] → drawdowns at each point vs running max
        # cum: [3, 4, 3, 2], running_max: [3, 4, 4, 4]
        # diff: [0, 0, -1, -2] → max_dd = -2
        returns = [3.0, 1.0, -1.0, -1.0]
        _, _, max_dd, _ = _calc()._calculate_risk_metrics(returns)
        assert max_dd == pytest.approx(-2.0)

    def test_sharpe_sign_positive_for_positive_mean(self):
        returns = [1.0, 2.0, 1.5, 2.0, 1.0]
        sharpe, _, _, _ = _calc()._calculate_risk_metrics(returns)
        assert sharpe > 0.0

    def test_sharpe_sign_negative_for_negative_mean(self):
        returns = [-1.0, -2.0, -1.5, -2.0]
        sharpe, _, _, _ = _calc()._calculate_risk_metrics(returns)
        assert sharpe < 0.0

    def test_sharpe_formula(self):
        returns = [1.0, 3.0, 2.0, 4.0]
        arr = np.array(returns)
        expected = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))
        sharpe, _, _, _ = _calc()._calculate_risk_metrics(returns)
        assert sharpe == pytest.approx(expected, rel=1e-6)

    def test_sortino_higher_than_sharpe_when_few_losses(self):
        # Mostly gains with two small losses → downside std << total std → sortino > sharpe
        # Need >=2 downside returns so downside_std is computed (not the 1e-8 fallback)
        returns = [2.0, 3.0, 1.0, 2.5, -0.1, -0.2]
        sharpe, sortino, _, _ = _calc()._calculate_risk_metrics(returns)
        assert sortino > sharpe

    def test_calmar_positive_when_profit_and_drawdown(self):
        returns = [3.0, -1.0, 2.0, -0.5]
        _, _, max_dd, calmar = _calc()._calculate_risk_metrics(returns)
        assert max_dd < 0
        assert calmar == pytest.approx(sum(returns) / abs(max_dd))

    def test_calmar_zero_when_no_drawdown(self):
        # No drawdown → calmar = 0 (guard against div by zero)
        returns = [1.0, 1.0, 1.0]
        _, _, _, calmar = _calc()._calculate_risk_metrics(returns)
        # max_dd = 0 → calmar protected → 0.0
        assert calmar == 0.0


# ---------------------------------------------------------------------------
# Trading metrics (win rate, gross/net returns, direction)
# ---------------------------------------------------------------------------


class TestTradingMetrics:
    def test_all_hold_no_trades(self):
        preds = [_pred(Signal.HOLD, Signal.HOLD, actual_change=2.0) for _ in range(5)]
        result = _calc()._calculate_trading_metrics(preds)
        assert result["win_rate"] == 0.0
        assert result["total_return"] == 0.0
        assert result["net_total_return"] == 0.0

    def test_buy_long_gains_on_up_move(self):
        pred = _pred(Signal.BUY, Signal.BUY, actual_change=3.0)
        result = _calc()._calculate_trading_metrics([pred])
        assert result["total_return"] == pytest.approx(3.0)

    def test_sell_short_gains_on_down_move(self):
        # SELL signal; price falls 3% → profit for short
        pred = _pred(Signal.SELL, Signal.SELL, actual_change=-3.0)
        result = _calc()._calculate_trading_metrics([pred])
        assert result["total_return"] == pytest.approx(3.0)

    def test_sell_short_loses_on_up_move(self):
        pred = _pred(Signal.SELL, Signal.BUY, actual_change=2.0)
        result = _calc()._calculate_trading_metrics([pred])
        assert result["total_return"] == pytest.approx(-2.0)

    def test_commission_reduces_net_return(self):
        calc = _calc(commission_pct=0.001)  # 0.1% one-way
        pred = _pred(Signal.BUY, Signal.BUY, actual_change=2.0)
        result = calc._calculate_trading_metrics([pred])
        round_trip = 2 * 0.001 * 100  # 0.2%
        assert result["net_total_return"] == pytest.approx(2.0 - round_trip)

    def test_win_rate_calculation(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY, actual_change=2.0),  # win
            _pred(Signal.BUY, Signal.BUY, actual_change=1.0),  # win
            _pred(Signal.BUY, Signal.SELL, actual_change=-1.0),  # loss
            _pred(Signal.BUY, Signal.SELL, actual_change=-2.0),  # loss
        ]
        result = _calc()._calculate_trading_metrics(preds)
        assert result["win_rate"] == pytest.approx(0.5)

    def test_empty_predictions_returns_zeros(self):
        result = _calc()._calculate_trading_metrics([])
        assert result["win_rate"] == 0.0
        assert result["total_return"] == 0.0
        assert result["net_total_return"] == 0.0
        assert result["trade_count"] == 0
        assert result["equity_curve"] == []

    def test_predictions_without_actual_skipped(self):
        p = HorizonPrediction(
            prediction_date=date(2024, 1, 2),
            horizon_days=5,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=2.0,
        )
        # actual_price_change is None
        result = _calc()._calculate_trading_metrics([p])
        assert result["total_return"] == 0.0


# ---------------------------------------------------------------------------
# determine_actual_signal
# ---------------------------------------------------------------------------


class TestDetermineActualSignal:
    def test_above_buy_threshold(self):
        calc = MetricsCalculator(buy_threshold=0.02, sell_threshold=-0.02)
        assert calc.determine_actual_signal(0.03) == Signal.BUY

    def test_below_sell_threshold(self):
        calc = MetricsCalculator(buy_threshold=0.02, sell_threshold=-0.02)
        assert calc.determine_actual_signal(-0.03) == Signal.SELL

    def test_in_neutral_zone(self):
        calc = MetricsCalculator(buy_threshold=0.02, sell_threshold=-0.02)
        assert calc.determine_actual_signal(0.01) == Signal.HOLD

    def test_exactly_at_buy_threshold_is_hold(self):
        calc = MetricsCalculator(buy_threshold=0.02, sell_threshold=-0.02)
        # strictly greater than threshold → BUY; equal → HOLD
        assert calc.determine_actual_signal(0.02) == Signal.HOLD

    def test_exactly_at_sell_threshold_is_hold(self):
        calc = MetricsCalculator(buy_threshold=0.02, sell_threshold=-0.02)
        assert calc.determine_actual_signal(-0.02) == Signal.HOLD


# ---------------------------------------------------------------------------
# calculate_horizon_metrics (integration)
# ---------------------------------------------------------------------------


class TestCalculateHorizonMetrics:
    def test_empty_list_returns_zero_metrics(self):
        m = _calc().calculate_horizon_metrics([], horizon=5)
        assert m.total_predictions == 0
        assert m.accuracy == 0.0

    def test_no_completed_predictions(self):
        p = HorizonPrediction(
            prediction_date=date(2024, 1, 2),
            horizon_days=5,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=2.0,
        )
        # actual_signal is None — not completed
        m = _calc().calculate_horizon_metrics([p], horizon=5)
        assert m.total_predictions == 0

    def test_populated_metrics(self):
        preds = [
            _pred(Signal.BUY, Signal.BUY, actual_change=3.0, predicted_change=2.5),
            _pred(Signal.BUY, Signal.BUY, actual_change=2.0, predicted_change=2.0),
            _pred(Signal.SELL, Signal.SELL, actual_change=-2.0, predicted_change=-1.5),
            _pred(Signal.BUY, Signal.SELL, actual_change=-1.0, predicted_change=2.0),
        ]
        m = _calc().calculate_horizon_metrics(preds, horizon=5)
        assert m.total_predictions == 4
        assert m.accuracy == pytest.approx(0.75)
        assert m.horizon_days == 5
        assert m.win_rate > 0.0
        assert m.price_mae > 0.0
        assert Signal.BUY in m.class_metrics
