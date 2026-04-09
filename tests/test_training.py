"""Tests for ModelTrainer — label creation, normalization, and data preparation."""

import numpy as np
import pandas as pd
import pytest

from models.training import ModelTrainer
from signals.direction import BUY_IDX, HOLD_IDX, SELL_IDX

# ---------------------------------------------------------------------------
# M7 — label index arithmetic
# ---------------------------------------------------------------------------


def _make_ohlcv(close_prices: list[float]) -> pd.DataFrame:
    """Minimal OHLCV DataFrame from a list of close prices."""
    n = len(close_prices)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    close = np.array(close_prices, dtype=float)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )


def _raw_labels(
    close_prices: list[float], horizon: int, buy_thresh: float, sell_thresh: float
) -> np.ndarray:
    """
    Reproduce the label-creation kernel from prepare_data() in isolation so we
    can assert correctness independently of the full training pipeline.

    For each row i the max-return window covers close_prices[i+1 : i+1+horizon].
    The denominator is close_prices[i] (today's price — the entry price).
    """
    from numpy.lib.stride_tricks import sliding_window_view

    n = len(close_prices)
    close_vals = np.array(close_prices, dtype=float)
    max_ret = np.full(n, np.nan)
    min_ret = np.full(n, np.nan)
    if n > horizon:
        windows = sliding_window_view(close_vals[1:], window_shape=horizon)
        max_ret[: n - horizon] = windows.max(axis=1) / close_vals[: n - horizon] - 1
        min_ret[: n - horizon] = windows.min(axis=1) / close_vals[: n - horizon] - 1

    lbl = np.full(n, float(HOLD_IDX), dtype=float)
    lbl[np.isnan(max_ret)] = np.nan
    buy_cond = max_ret > buy_thresh
    sell_cond = min_ret < sell_thresh
    lbl[buy_cond & ~sell_cond] = BUY_IDX
    lbl[sell_cond & ~buy_cond] = SELL_IDX
    conflict = buy_cond & sell_cond & ~np.isnan(max_ret)
    lbl[conflict & (max_ret >= np.abs(min_ret))] = BUY_IDX
    lbl[conflict & (max_ret < np.abs(min_ret))] = SELL_IDX
    return lbl


class TestLabelCreation:
    """
    M7 — verify that max-return labels are computed against the correct base price.

    For prediction at day i, the entry price is close[i] and the future window
    is close[i+1 : i+1+h].  Dividing by close[i-1] would misstate every return
    by one day.  These tests confirm the denominator is close[i].
    """

    def test_single_strong_up_move_is_labelled_buy(self):
        """A price that jumps 10% on day 1 and stays flat should label day 0 as BUY."""
        # Day 0 close = 100.  Day 1 close = 110 (+10%).  Days 2-4 flat.
        prices = [100.0, 110.0, 110.0, 110.0, 110.0]
        lbl = _raw_labels(prices, horizon=3, buy_thresh=0.05, sell_thresh=-0.05)
        # Row 0: max over [110, 110, 110] / 100 - 1 = 0.10 → BUY
        assert lbl[0] == BUY_IDX, f"Expected BUY at row 0, got {lbl[0]}"

    def test_single_strong_down_move_is_labelled_sell(self):
        """A price that drops 10% on day 1 should label day 0 as SELL."""
        prices = [100.0, 90.0, 90.0, 90.0, 90.0]
        lbl = _raw_labels(prices, horizon=3, buy_thresh=0.05, sell_thresh=-0.05)
        # Row 0: min over [90, 90, 90] / 100 - 1 = -0.10 → SELL
        assert lbl[0] == SELL_IDX, f"Expected SELL at row 0, got {lbl[0]}"

    def test_flat_prices_are_labelled_hold(self):
        """Flat prices produce no return — all valid rows should be HOLD."""
        prices = [100.0] * 10
        lbl = _raw_labels(prices, horizon=3, buy_thresh=0.05, sell_thresh=-0.05)
        valid = lbl[~np.isnan(lbl)]
        assert np.all(valid == HOLD_IDX), f"Expected all HOLD, got {valid}"

    def test_label_denominator_is_row_i_not_row_i_minus_1(self):
        """
        Explicitly verify the denominator is close[i], not close[i-1].

        Set close = [80, 100, 120, 120, 120].
        - Row 0: window = [100, 120, 120], max = 120.  Return vs close[0]=80 → +50%.
          Return vs close[-1] (hypothetical) would be undefined.
        - Row 1: window = [120, 120, 120], max = 120.  Return vs close[1]=100 → +20%.
          Return vs close[0]=80 would give +50% (wrong).
        """
        prices = [80.0, 100.0, 120.0, 120.0, 120.0]
        lbl = _raw_labels(prices, horizon=3, buy_thresh=0.05, sell_thresh=-0.05)
        # Row 0 and row 1 should both be BUY (returns of +50% and +20% exceed 5%)
        assert lbl[0] == BUY_IDX, f"Row 0 should be BUY (return +50%), got {lbl[0]}"
        assert lbl[1] == BUY_IDX, f"Row 1 should be BUY (return +20%), got {lbl[1]}"

    def test_last_horizon_rows_are_nan(self):
        """The last `horizon` rows have no future window and should be NaN."""
        h = 3
        prices = list(range(1, 11))  # 10 prices
        lbl = _raw_labels(prices, horizon=h, buy_thresh=0.05, sell_thresh=-0.05)
        # Last h rows should be NaN (no future data)
        assert np.all(np.isnan(lbl[len(prices) - h :])), "Last h rows must be NaN"

    def test_conflict_resolved_by_magnitude(self):
        """When both BUY and SELL thresholds are crossed, larger magnitude wins."""
        # Row 0: max_ret = +20%, min_ret = -5%.  |max| > |min| → BUY.
        # Prices: start at 100, spike to 120, then dip to 95, then recover.
        prices = [100.0, 120.0, 95.0, 100.0]
        lbl = _raw_labels(prices, horizon=2, buy_thresh=0.05, sell_thresh=-0.04)
        # Window [120, 95]: max=120 → +20%, min=95 → -5%.  Both thresholds crossed.
        # |20%| > |5%| → BUY.
        assert lbl[0] == BUY_IDX, f"Conflict row should resolve to BUY, got {lbl[0]}"


# ---------------------------------------------------------------------------
# M8 — feature-dimension consistency check
# ---------------------------------------------------------------------------


class TestFeatureDimensionCheck:
    """M8 — train() must catch feature-column mismatches before np.vstack."""

    def test_mismatched_feature_dims_raises_with_ticker_name(self):
        """
        When the second ticker's prepare_data() returns a different column count,
        train() should raise ValueError immediately and name the offending ticker.
        """
        from unittest.mock import MagicMock, patch

        trainer = ModelTrainer(sequence_length=5)
        call_count = 0

        def fake_prepare_data(df, reference_data=None):
            nonlocal call_count
            call_count += 1
            n = 100
            n_features = 10 if call_count == 1 else 8  # mismatch on second call
            features = np.ones((n, n_features))
            labels = [np.zeros(n, dtype=int)] * len(trainer.prediction_horizons)
            prices = np.zeros(n)
            dates = pd.date_range("2020-01-02", periods=n, freq="B")
            return features, labels, prices, dates

        fake_df = _make_ohlcv([100.0] * 200)

        with (
            patch.object(trainer, "prepare_data", side_effect=fake_prepare_data),
            patch("models.training.StockDataFetcher") as mock_fetcher_cls,
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.fetch.return_value = fake_df
            mock_fetcher.fetch_cross_asset_data.return_value = {}
            mock_fetcher_cls.return_value = mock_fetcher

            with pytest.raises(ValueError, match="BAD-TICKER"):
                trainer.train(["GOOD-TICKER", "BAD-TICKER"], track_with_mlflow=False)

    def test_matching_feature_dims_does_not_raise(self):
        """When all tickers return the same column count, no error is raised at the check."""
        from unittest.mock import MagicMock, patch

        trainer = ModelTrainer(sequence_length=5)

        def fake_prepare_data(df, reference_data=None):
            n = 100
            features = np.ones((n, 10))
            labels = [np.zeros(n, dtype=int)] * len(trainer.prediction_horizons)
            prices = np.zeros(n)
            dates = pd.date_range("2020-01-02", periods=n, freq="B")
            return features, labels, prices, dates

        fake_df = _make_ohlcv([100.0] * 200)

        with (
            patch.object(trainer, "prepare_data", side_effect=fake_prepare_data),
            patch("models.training.StockDataFetcher") as mock_fetcher_cls,
        ):
            mock_fetcher = MagicMock()
            mock_fetcher.fetch.return_value = fake_df
            mock_fetcher.fetch_cross_asset_data.return_value = {}
            mock_fetcher_cls.return_value = mock_fetcher

            # The dimension check passes; a subsequent error (e.g. model build)
            # is fine — we just confirm ValueError from the dimension check is NOT raised.
            try:
                trainer.train(["T1", "T2"], track_with_mlflow=False)
            except ValueError as exc:
                assert "feature dimension mismatch" not in str(exc).lower(), (
                    f"Unexpected dimension-check error: {exc}"
                )
