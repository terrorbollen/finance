"""Tests for Backtester — holdout enforcement, outcome filling, relative volume."""

from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import backtesting.backtester as bt_module
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
    model.prediction_horizons = [5, 10, 20]

    def _predict(X: np.ndarray):
        batch = X.shape[0]
        probs = np.zeros((batch, 3))
        probs[:, signal_idx] = 0.8
        return probs, np.full(batch, signal_idx), np.full(batch, 2.0)

    def _predict_per_horizon(X: np.ndarray):
        batch = X.shape[0]
        probs = np.zeros((batch, 3))
        probs[:, signal_idx] = 0.8
        horizon_probs = [probs.copy() for _ in model.prediction_horizons]
        horizon_classes = [np.full(batch, signal_idx) for _ in model.prediction_horizons]
        return horizon_probs, horizon_classes, np.full(batch, 2.0)

    model.predict.side_effect = _predict
    model.predict_per_horizon.side_effect = _predict_per_horizon
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
    bt.prediction_horizons = None
    bt.retrain_every = None
    bt.retrain_epochs = 20
    bt.enforce_position_cooldown = False
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
# Holdout enforcement — calls real bt.run() so changes to the guard are caught
# ---------------------------------------------------------------------------


def _patch_fetcher(monkeypatch, df: pd.DataFrame) -> None:
    """Patch StockDataFetcher and FeatureEngineer so run() uses synthetic data."""
    mock_fetcher = MagicMock()
    mock_fetcher.fetch.return_value = df
    mock_fetcher.fetch_cross_asset_data.return_value = {}
    monkeypatch.setattr(bt_module, "StockDataFetcher", lambda **_: mock_fetcher)
    monkeypatch.setattr(
        bt_module,
        "FeatureEngineer",
        lambda df, **_: MagicMock(
            add_all_features=lambda: df,
            get_feature_columns=lambda: ["close"],
        ),
    )


class TestHoldoutEnforcement:
    def _bt(self, strict_holdout: bool, holdout_start_date: date | None) -> Backtester:
        bt = _backtester_with_mock_model(strict_holdout=strict_holdout, sequence_length=5)
        bt.holdout_start_date = holdout_start_date
        bt.feature_columns = None  # use whatever columns come back
        return bt

    def test_start_before_holdout_is_clamped(self, monkeypatch):
        """Result start_date is moved forward to holdout_start_date."""
        df = _make_df(n_rows=60)
        _patch_fetcher(monkeypatch, df)

        bt = self._bt(strict_holdout=True, holdout_start_date=date(2023, 3, 1))
        result = bt.run("FAKE.ST", start_date=date(2023, 1, 2), end_date=date(2023, 3, 31))
        assert result.start_date >= date(2023, 3, 1)

    def test_start_after_holdout_unchanged(self, monkeypatch):
        """When start is already past holdout, it is not modified."""
        df = _make_df(n_rows=60)
        _patch_fetcher(monkeypatch, df)

        # Use dates within the synthetic df's range (60 business days from 2023-01-02)
        requested_start = date(2023, 2, 1)
        bt = self._bt(strict_holdout=True, holdout_start_date=date(2023, 1, 15))
        result = bt.run("FAKE.ST", start_date=requested_start, end_date=date(2023, 3, 31))
        assert result.start_date == requested_start

    def test_no_holdout_date_does_not_adjust(self, monkeypatch):
        """With no holdout date, any start is accepted."""
        df = _make_df(n_rows=60)
        _patch_fetcher(monkeypatch, df)

        requested_start = date(2023, 1, 2)
        bt = self._bt(strict_holdout=True, holdout_start_date=None)
        result = bt.run("FAKE.ST", start_date=requested_start, end_date=date(2023, 3, 31))
        assert result.start_date == requested_start

    def test_strict_holdout_false_accepts_early_start(self, monkeypatch):
        """strict_holdout=False never adjusts the start date."""
        df = _make_df(n_rows=60)
        _patch_fetcher(monkeypatch, df)

        requested_start = date(2023, 1, 2)
        bt = self._bt(strict_holdout=False, holdout_start_date=date(2023, 6, 1))
        result = bt.run("FAKE.ST", start_date=requested_start, end_date=date(2023, 3, 31))
        assert result.start_date == requested_start


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

    def test_gap_in_data_does_not_shift_target_date(self):
        """When yfinance drops a session, the target price must still reflect the
        correct business-day horizon, not a shifted row offset."""
        bt, df = self._setup(n_rows=30)
        idx = 10
        pred_date = df.index[idx].date()
        horizon = 5

        # Drop row idx+2 to simulate a yfinance data gap
        df_gap = pd.concat([df.iloc[: idx + 2], df.iloc[idx + 3 :]]).copy()
        df_gap.index = pd.DatetimeIndex(df_gap.index)

        # Price at the correct 5-BDay target in the gapped frame
        from pandas.tseries.offsets import BDay

        target_date = (df.index[idx] + BDay(horizon)).date()
        # After the drop the target date row still exists in df_gap (we dropped idx+2, not idx+5)
        target_price_correct = float(df["close"].loc[df.index[idx] + BDay(horizon)])
        pred_price = float(df["close"].iloc[idx])
        expected_change = ((target_price_correct / pred_price) - 1) * 100

        p = HorizonPrediction(
            prediction_date=pred_date,
            horizon_days=horizon,
            predicted_signal=Signal.BUY,
            confidence=0.7,
            predicted_price_change=2.0,
        )
        daily = DailyPrediction(date=pred_date, current_price=pred_price)
        daily.add_prediction(p)

        bt._fill_actual_outcomes(df_gap, [daily], horizons=[horizon])

        assert p.actual_price_change == pytest.approx(expected_change, rel=1e-6)
        assert p.target_date == target_date

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


# ---------------------------------------------------------------------------
# Walk-forward retraining — mocks _retrain_model so no training occurs
# ---------------------------------------------------------------------------


class TestWalkForwardRetrain:
    """Tests the chunked prediction / retraining coordination in Backtester.run().

    _retrain_model is mocked to a no-op so these tests run in milliseconds.
    What is verified: when retraining fires, how many times, at what dates,
    and that the prediction count is always correct.
    """

    def _bt(self, retrain_every: int) -> Backtester:
        bt = _backtester_with_mock_model(sequence_length=5)
        bt.retrain_every = retrain_every
        return bt

    def _patch_retrain(self, monkeypatch, bt: Backtester) -> list:
        """Replace _retrain_model with a recorder that returns immediately."""
        calls: list = []
        monkeypatch.setattr(bt, "_retrain_model", lambda df_raw, d: calls.append(d))
        return calls

    def test_no_retrain_when_disabled(self, monkeypatch):
        """retrain_every=None → _retrain_model is never called."""
        bt = _backtester_with_mock_model(sequence_length=5)
        bt.retrain_every = None
        calls = self._patch_retrain(monkeypatch, bt)

        df = _make_df(n_rows=80)
        _patch_fetcher(monkeypatch, df)
        bt.run("FAKE.ST", start_date=date(2023, 1, 2), end_date=date(2023, 3, 31))

        assert calls == []

    def test_retrain_count_matches_chunk_boundaries(self, monkeypatch):
        """Number of retrains equals number of chunks minus one (first chunk never retrains)."""
        retrain_every = 20
        bt = self._bt(retrain_every)
        calls = self._patch_retrain(monkeypatch, bt)

        df = _make_df(n_rows=80)
        _patch_fetcher(monkeypatch, df)
        result = bt.run("FAKE.ST", start_date=date(2023, 1, 2), end_date=date(2023, 3, 31))

        n_days = result.trading_days
        expected_retrains = (n_days - 1) // retrain_every  # chunks - 1
        assert len(calls) == expected_retrains

    def test_retrain_cutoff_dates_are_strictly_increasing(self, monkeypatch):
        """Each retrain uses a later cutoff date than the previous one."""
        bt = self._bt(retrain_every=15)
        calls = self._patch_retrain(monkeypatch, bt)

        df = _make_df(n_rows=80)
        _patch_fetcher(monkeypatch, df)
        bt.run("FAKE.ST", start_date=date(2023, 1, 2), end_date=date(2023, 3, 31))

        assert len(calls) >= 2, "Need at least 2 retrains to test ordering"
        assert calls == sorted(calls)
        assert len(set(calls)) == len(calls)  # all unique

    def test_prediction_count_equals_trading_days(self, monkeypatch):
        """Total daily predictions equal trading_days regardless of chunk size."""
        bt = self._bt(retrain_every=10)
        self._patch_retrain(monkeypatch, bt)

        df = _make_df(n_rows=80)
        _patch_fetcher(monkeypatch, df)
        result = bt.run("FAKE.ST", start_date=date(2023, 1, 2), end_date=date(2023, 3, 31))

        assert len(result.daily_predictions) == result.trading_days

    def test_single_chunk_triggers_no_retrain(self, monkeypatch):
        """When retrain_every >= trading_days, only one chunk exists → no retrains."""
        bt = self._bt(retrain_every=1000)
        calls = self._patch_retrain(monkeypatch, bt)

        df = _make_df(n_rows=80)
        _patch_fetcher(monkeypatch, df)
        bt.run("FAKE.ST", start_date=date(2023, 1, 2), end_date=date(2023, 3, 31))

        assert calls == []
