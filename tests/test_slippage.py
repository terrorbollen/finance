"""Tests for volume-based slippage modeling (task B4)."""

from datetime import date

import pytest

from backtesting.metrics import _MAX_SLIPPAGE_PCT, MetricsCalculator
from backtesting.results import HorizonPrediction, Signal


def _make_prediction(
    *,
    signal: Signal = Signal.BUY,
    actual_change: float = 2.0,
    relative_volume: float | None = None,
) -> HorizonPrediction:
    """Helper to build a minimal HorizonPrediction."""
    p = HorizonPrediction(
        prediction_date=date(2024, 1, 2),
        horizon_days=1,
        predicted_signal=signal,
        confidence=0.8,
        predicted_price_change=actual_change,
        relative_volume=relative_volume,
    )
    p.actual_price_change = actual_change
    p.actual_signal = Signal.BUY if actual_change > 0 else Signal.SELL
    return p


class TestZeroSlippageWhenVolumeUnavailable:
    """Slippage must be zero when relative_volume is None."""

    def test_no_slippage_without_volume(self):
        """Net return equals gross return minus commission when volume unavailable."""
        calc = MetricsCalculator(commission_pct=0.001, slippage_factor=0.1)
        pred = _make_prediction(actual_change=2.0, relative_volume=None)

        result = calc._calculate_trading_metrics([pred])

        gross = result["total_return"]
        net = result["net_total_return"]
        round_trip_commission = 2 * 0.001 * 100  # 0.2%
        assert abs((gross - net) - round_trip_commission) < 1e-9

    def test_zero_slippage_factor_no_slippage(self):
        """Even with volume present, zero slippage_factor means zero slippage."""
        calc = MetricsCalculator(commission_pct=0.001, slippage_factor=0.0)
        pred = _make_prediction(actual_change=2.0, relative_volume=0.1)  # very low volume

        result = calc._calculate_trading_metrics([pred])

        gross = result["total_return"]
        net = result["net_total_return"]
        round_trip_commission = 2 * 0.001 * 100
        assert abs((gross - net) - round_trip_commission) < 1e-9


class TestHigherSlippageOnLowVolume:
    """Low relative volume should produce higher slippage than normal volume."""

    def test_low_volume_higher_slippage_than_normal(self):
        """A day with relative_volume=0.25 has higher slippage than rel_vol=1.0."""
        calc = MetricsCalculator(commission_pct=0.0, slippage_factor=0.1)

        pred_normal = _make_prediction(actual_change=2.0, relative_volume=1.0)
        pred_thin = _make_prediction(actual_change=2.0, relative_volume=0.25)

        result_normal = calc._calculate_trading_metrics([pred_normal])
        result_thin = calc._calculate_trading_metrics([pred_thin])

        # Both have same gross return; thin-market should have lower net return
        assert result_thin["net_total_return"] < result_normal["net_total_return"]

    def test_normal_volume_slippage_equals_factor(self):
        """At relative_volume=1.0, one-way slippage equals slippage_factor."""
        slippage_factor = 0.1
        calc = MetricsCalculator(commission_pct=0.0, slippage_factor=slippage_factor)
        pred = _make_prediction(actual_change=2.0, relative_volume=1.0)

        result = calc._calculate_trading_metrics([pred])

        gross = result["total_return"]
        net = result["net_total_return"]
        expected_round_trip_slippage = 2 * slippage_factor  # entry + exit
        assert abs((gross - net) - expected_round_trip_slippage) < 1e-9

    def test_very_low_volume_capped_at_max(self):
        """Extremely low volume triggers the 0.5% per-trade slippage cap."""
        calc = MetricsCalculator(commission_pct=0.0, slippage_factor=0.1)
        # relative_volume=0.001 → uncapped: 0.1/sqrt(0.001)≈3.16% >> 0.5% cap
        pred = _make_prediction(actual_change=2.0, relative_volume=0.001)

        result = calc._calculate_trading_metrics([pred])

        gross = result["total_return"]
        net = result["net_total_return"]
        round_trip_slippage = gross - net
        # Should be exactly 2 * MAX (round-trip)
        expected = 2 * _MAX_SLIPPAGE_PCT
        assert abs(round_trip_slippage - expected) < 1e-9


class TestSlippageCap:
    """Slippage must never exceed 0.5% per one-way trade."""

    def test_cap_enforced(self):
        """High slippage_factor with low volume is capped at 0.5% one-way."""
        calc = MetricsCalculator(commission_pct=0.0, slippage_factor=10.0)
        pred = _make_prediction(actual_change=5.0, relative_volume=0.01)

        result = calc._calculate_trading_metrics([pred])

        gross = result["total_return"]
        net = result["net_total_return"]
        round_trip_slippage = gross - net
        assert round_trip_slippage <= 2 * _MAX_SLIPPAGE_PCT + 1e-9

    def test_max_slippage_constant_value(self):
        """The cap constant should be 0.5%."""
        assert _MAX_SLIPPAGE_PCT == 0.5


class TestRelativeVolumeField:
    """HorizonPrediction.relative_volume field behaves correctly."""

    def test_default_is_none(self):
        pred = HorizonPrediction(
            prediction_date=date(2024, 1, 2),
            horizon_days=1,
            predicted_signal=Signal.HOLD,
            confidence=0.5,
            predicted_price_change=0.0,
        )
        assert pred.relative_volume is None

    def test_set_and_serialise(self):
        pred = HorizonPrediction(
            prediction_date=date(2024, 1, 2),
            horizon_days=1,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=1.5,
            relative_volume=1.23,
        )
        d = pred.to_dict()
        assert d["relative_volume"] == pytest.approx(1.23)
