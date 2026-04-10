"""Tests for the FeatureEngineer class."""

import numpy as np
import pandas as pd
import pytest

from data.features import FEATURE_COLUMNS, FeatureEngineer

EXPECTED_FEATURES = FEATURE_COLUMNS


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """300 rows of synthetic OHLCV data — enough for all rolling windows."""
    rng = np.random.default_rng(42)
    n = 300
    close = 100 * np.cumprod(1 + rng.standard_normal(n) * 0.01)
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))
    open_ = close * (1 + rng.standard_normal(n) * 0.005)
    volume = rng.integers(100_000, 1_000_000, n).astype(float)
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


class TestFeatureEngineer:
    def test_returns_dataframe(self, sample_ohlcv_data):
        result = FeatureEngineer(sample_ohlcv_data).add_all_features()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_no_nan(self, sample_ohlcv_data):
        result = FeatureEngineer(sample_ohlcv_data).add_all_features()
        assert not result.isna().any().any()

    def test_no_inf(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        result = eng.add_all_features()
        for col in eng.get_feature_columns():
            assert not np.isinf(result[col]).any(), f"{col} contains inf"

    def test_feature_columns(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_all_features()
        assert eng.get_feature_columns() == EXPECTED_FEATURES

    def test_ohlcv_excluded(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_all_features()
        for col in ("open", "high", "low", "close", "volume"):
            assert col not in eng.get_feature_columns()

    def test_features_array_shape(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_all_features()
        arr = eng.get_features_array()
        assert arr.shape == (len(eng.df), len(EXPECTED_FEATURES))


class TestRSI:
    def test_rsi_range(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_rsi()
        rsi = eng.df["rsi"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100


class TestMACDHistogram:
    def test_column_present(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_macd_histogram()
        assert "macd_histogram" in eng.df.columns

    def test_no_nan_after_warmup(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_macd_histogram()
        assert eng.df["macd_histogram"].dropna().notna().all()


class TestMomentum:
    def test_column_present(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_momentum()
        assert "momentum_10" in eng.df.columns

    def test_matches_manual_calculation(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_momentum()
        expected = sample_ohlcv_data["close"].pct_change(10) * 100
        pd.testing.assert_series_equal(
            eng.df["momentum_10"], expected, check_names=False, rtol=1e-6
        )


class TestATR:
    def test_atr_positive(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_atr()
        assert (eng.df["atr"].dropna() > 0).all()

    def test_atr_is_percentage(self, sample_ohlcv_data):
        """ATR as % of price should be small (typically 0.5–5% for equities)."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_atr()
        atr = eng.df["atr"].dropna()
        assert atr.max() < 50, "ATR % unexpectedly large"


class TestBBPosition:
    def test_mostly_in_range(self, sample_ohlcv_data):
        """Most values should be within [0, 1]; outliers outside bands are fine."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_bb_position()
        bp = eng.df["bb_position"].dropna()
        within = ((bp >= -0.5) & (bp <= 1.5)).mean()
        assert within > 0.9

    def test_constant_price_yields_midpoint(self):
        """When all prices are identical the band collapses to zero width; bb_position must be 0.5, not NaN."""
        dates = pd.date_range("2023-01-02", periods=40, freq="B")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000,
            },
            index=dates,
        )
        eng = FeatureEngineer(df)
        eng._add_bb_position()
        bp = eng.df["bb_position"]
        # All values should be exactly 0.5 — no NaNs produced for constant-price rows
        assert bp.notna().all()
        assert bp.eq(0.5).all()


class TestVolumeRatio:
    def test_positive(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_volume_ratio()
        assert (eng.df["volume_ratio"].dropna() > 0).all()

    def test_clipped_at_10(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_volume_ratio()
        assert eng.df["volume_ratio"].max() <= 10


class TestADX:
    def test_range(self, sample_ohlcv_data):
        eng = FeatureEngineer(sample_ohlcv_data)
        eng._add_adx()
        adx = eng.df["adx_14"].dropna()
        assert adx.min() >= 0
        assert adx.max() <= 100


