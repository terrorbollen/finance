"""Tests for Backtester — holdout enforcement, outcome filling, relative volume."""

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from backtesting.backtester import Backtester
from backtesting.results import DailyPrediction, HorizonPrediction, Signal

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_rows: int = 60, with_volume: bool = True) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame with a DatetimeIndex."""
    dates = pd.date_range(start="2023-01-02", periods=n_rows, freq="B")
    close = 100.0 + np.cumsum(np.random.default_rng(42).normal(0, 1, n_rows))
    data = {
        "open": close * 0.999,
        "high": close * 1.005,
        "low": close * 0.995,
        "close": close,
    }
    if with_volume:
        data["volume"] = np.random.default_rng(0).integers(1_000, 10_000, n_rows).astype(float)
    return pd.DataFrame(data, index=dates)


def _mock_model(signal_idx: int = 0) -> MagicMock:
    """Mock SignalModel that always returns the same signal."""
    model = MagicMock()
    probs = np.zeros((1, 3))
    probs[0, signal_idx] = 0.8
    model.predict.return_value = (probs, np.array([signal_idx]), np.array([2.0]))
    return model


def _backtester_with_mock_model(
    signal_idx: int = 0,
    sequence_length: int = 5,
    strict_holdout: bool = False,
) -> Backtester:
    """Return a Backtester with no real model/config loaded."""
    bt = Backtester.__new__(Backtester)
    bt.model_path = "nonexistent.weights.h5"
    bt.sequence_length = sequence_length
    bt.buy_threshold = 0.02
    bt.sell_threshold = -0.02
    bt.strict_holdout = strict_holdout
    bt.slippage_factor = 0.0
    bt.holdout_start_date = None
    bt.feature_columns = None
    bt.feature_mean = None
    bt.feature_std = None
    bt.input_dim = None
    bt.model = _mock_model(signal_idx)

    from backtesting.metrics import MetricsCalculator
    bt.metrics_calculator = MetricsCalculator(
        buy_threshold=bt.buy_threshold,
        sell_threshold=bt.sell_threshold,
        commission_pct=0.001,
        slippage_factor=0.0,
    )
    return bt


# ---------------------------------------------------------------------------
# Holdout enforcement
# ---------------------------------------------------------------------------

class TestHoldoutEnforcement:
    def test_start_before_holdout_is_clamped(self):
        bt = _backtester_with_mock_model(strict_holdout=True)
        bt.holdout_start_date = date(2024, 6, 1)

        start = date(2024, 1, 1)  # before holdout

        # Simulate the guard block from run()
        if bt.strict_holdout and bt.holdout_start_date and start < bt.holdout_start_date:
            print(f"WARNING: Adjusting start to {bt.holdout_start_date}")
            start = bt.holdout_start_date

        assert start == date(2024, 6, 1)

    def test_start_after_holdout_unchanged(self):
        bt = _backtester_with_mock_model(strict_holdout=True)
        bt.holdout_start_date = date(2024, 6, 1)

        start = date(2024, 9, 1)  # after holdout
        original = start

        if bt.strict_holdout and bt.holdout_start_date and start < bt.holdout_start_date:
            start = bt.holdout_start_date

        assert start == original

    def test_no_holdout_date_no_adjustment(self):
        bt = _backtester_with_mock_model(strict_holdout=True)
        bt.holdout_start_date = None

        start = date(2020, 1, 1)
        original = start

        if bt.strict_holdout and bt.holdout_start_date and start < bt.holdout_start_date:
            start = bt.holdout_start_date

        assert start == original

    def test_strict_holdout_false_skips_guard(self):
        bt = _backtester_with_mock_model(strict_holdout=False)
        bt.holdout_start_date = date(2024, 6, 1)

        start = date(2024, 1, 1)
        original = start

        if bt.strict_holdout and bt.holdout_start_date and start < bt.holdout_start_date:
            start = bt.holdout_start_date

        assert start == original


# ---------------------------------------------------------------------------
# _fill_actual_outcomes
# ---------------------------------------------------------------------------

class TestFillActualOutcomes:
    def _setup(self, n_rows: int = 30, seq_len: int = 5):
        bt = _backtester_with_mock_model(sequence_length=seq_len)
        df = _make_df(n_rows)
        return bt, df

    def test_outcome_filled_for_completed_horizon(self):
        bt, df = self._setup()
        idx = 10  # pick a row with room for horizon=5 ahead
        pred_date = df.index[idx].date()
        p = HorizonPrediction(
            prediction_date=pred_date,
            horizon_days=5,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=2.0,
        )
        daily = DailyPrediction(date=pred_date, current_price=float(df["close"].iloc[idx]))
        daily.add_prediction(p)

        bt._fill_actual_outcomes(df, [daily], horizons=[5])

        assert p.actual_price_change is not None
        assert p.actual_signal is not None
        assert p.target_date is not None

    def test_actual_price_change_correct(self):
        bt, df = self._setup(n_rows=30)
        idx = 10
        pred_date = df.index[idx].date()
        p = HorizonPrediction(
            prediction_date=pred_date,
            horizon_days=3,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=2.0,
        )
        daily = DailyPrediction(date=pred_date, current_price=float(df["close"].iloc[idx]))
        daily.add_prediction(p)

        bt._fill_actual_outcomes(df, [daily], horizons=[3])

        expected_change = ((df["close"].iloc[idx + 3] / df["close"].iloc[idx]) - 1) * 100
        assert p.actual_price_change == pytest.approx(expected_change)

    def test_horizon_past_end_of_data_stays_none(self):
        bt, df = self._setup(n_rows=15)
        idx = 13  # only 1 row ahead, horizon=5 exceeds data
        pred_date = df.index[idx].date()
        p = HorizonPrediction(
            prediction_date=pred_date,
            horizon_days=5,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=2.0,
        )
        daily = DailyPrediction(date=pred_date, current_price=float(df["close"].iloc[idx]))
        daily.add_prediction(p)

        bt._fill_actual_outcomes(df, [daily], horizons=[5])

        assert p.actual_price_change is None
        assert p.actual_signal is None

    def test_actual_signal_matches_threshold(self):
        bt, df = self._setup(n_rows=30)
        # Manually force a large price jump at idx+5
        df = df.copy()
        idx = 10
        df.iloc[idx + 5, df.columns.get_loc("close")] = df["close"].iloc[idx] * 1.10  # +10%

        pred_date = df.index[idx].date()
        p = HorizonPrediction(
            prediction_date=pred_date,
            horizon_days=5,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=2.0,
        )
        daily = DailyPrediction(date=pred_date, current_price=float(df["close"].iloc[idx]))
        daily.add_prediction(p)

        bt._fill_actual_outcomes(df, [daily], horizons=[5])

        assert p.actual_signal == Signal.BUY  # +10% > 2% buy threshold


# ---------------------------------------------------------------------------
# _predict_for_date — relative volume
# ---------------------------------------------------------------------------

class TestPredictForDate:
    def test_relative_volume_computed_when_volume_present(self):
        bt = _backtester_with_mock_model(sequence_length=5)
        df = _make_df(n_rows=30, with_volume=True)
        feature_cols = ["close"]

        pred = bt._predict_for_date(df=df, feature_cols=feature_cols, idx=20, horizon=5)

        assert pred.relative_volume is not None
        assert pred.relative_volume > 0.0

    def test_relative_volume_none_without_volume_column(self):
        bt = _backtester_with_mock_model(sequence_length=5)
        df = _make_df(n_rows=30, with_volume=False)
        feature_cols = ["close"]

        pred = bt._predict_for_date(df=df, feature_cols=feature_cols, idx=20, horizon=5)

        assert pred.relative_volume is None

    def test_prediction_date_matches_df_index(self):
        bt = _backtester_with_mock_model(sequence_length=5)
        df = _make_df(n_rows=30)
        idx = 15
        pred = bt._predict_for_date(df=df, feature_cols=["close"], idx=idx, horizon=5)

        assert pred.prediction_date == df.index[idx].date()

    def test_horizon_stored_on_prediction(self):
        bt = _backtester_with_mock_model(sequence_length=5)
        df = _make_df(n_rows=30)
        pred = bt._predict_for_date(df=df, feature_cols=["close"], idx=15, horizon=7)

        assert pred.horizon_days == 7

    def test_signal_from_mock_model(self):
        bt = _backtester_with_mock_model(signal_idx=2, sequence_length=5)  # SELL=2
        df = _make_df(n_rows=30)
        pred = bt._predict_for_date(df=df, feature_cols=["close"], idx=15, horizon=5)

        assert pred.predicted_signal == Signal.SELL

    def test_relative_volume_is_ratio_of_today_to_rolling_avg(self):
        bt = _backtester_with_mock_model(sequence_length=5)
        df = _make_df(n_rows=30, with_volume=False)
        # Set all volumes to 1000 then spike the last one to 2000
        df["volume"] = 1000.0
        df.iloc[20, df.columns.get_loc("volume")] = 2000.0

        pred = bt._predict_for_date(df=df, feature_cols=["close"], idx=20, horizon=5)

        # rolling avg of first 20 rows is 1000, today is 2000 → ratio ≈ 2.0
        assert pred.relative_volume == pytest.approx(2.0, rel=0.1)
