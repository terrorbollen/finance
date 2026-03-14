"""Tests for the FeatureEngineer class."""

import numpy as np
import pandas as pd

from data.features import FeatureEngineer


class TestFeatureEngineer:
    """Tests for FeatureEngineer functionality."""

    def test_add_all_features_returns_dataframe(self, sample_ohlcv_data: pd.DataFrame):
        """Test that add_all_features returns a DataFrame."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.add_all_features()

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_add_all_features_no_nan(self, sample_ohlcv_data: pd.DataFrame):
        """Test that add_all_features returns data with no NaN values."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        result = engineer.add_all_features()

        assert not result.isna().any().any(), "DataFrame should have no NaN values after dropna"

    def test_rsi_range(self, sample_ohlcv_data: pd.DataFrame):
        """Test that RSI values are in valid range [0, 100]."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        engineer.add_rsi()

        # Drop NaN values from the rolling window calculation
        rsi = engineer.df["rsi"].dropna()

        assert rsi.min() >= 0, "RSI should be >= 0"
        assert rsi.max() <= 100, "RSI should be <= 100"

    def test_feature_columns_exclude_ohlcv(self, sample_ohlcv_data: pd.DataFrame):
        """Test that get_feature_columns excludes OHLCV columns."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        engineer.add_all_features()

        feature_cols = engineer.get_feature_columns()
        ohlcv_cols = ["open", "high", "low", "close", "volume"]

        for col in ohlcv_cols:
            assert col not in feature_cols, f"{col} should not be in feature columns"

    def test_feature_columns_added(self, sample_ohlcv_data: pd.DataFrame):
        """Test that expected feature columns are added."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        engineer.add_all_features()

        feature_cols = engineer.get_feature_columns()

        # Check for key feature groups
        expected_patterns = [
            "sma_",
            "ema_",
            "rsi",
            "macd",
            "bb_",
            "volume_",
            "roc_",
            "atr",
            "returns",
        ]

        for pattern in expected_patterns:
            matching = [c for c in feature_cols if pattern in c]
            assert len(matching) > 0, f"Expected feature with pattern '{pattern}'"

    def test_moving_averages_ratios(self, sample_ohlcv_data: pd.DataFrame):
        """Test that moving average features are ratios (normalized)."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        engineer.add_moving_averages()

        # SMA/EMA ratios should be small values centered around 0
        for period in [5, 10, 20, 50, 200]:
            sma_col = f"sma_{period}_ratio"
            ema_col = f"ema_{period}_ratio"

            if sma_col in engineer.df.columns:
                sma_vals = engineer.df[sma_col].dropna()
                # Ratios should typically be within [-0.5, 0.5] for normal market conditions
                assert sma_vals.abs().max() < 1.0, f"{sma_col} should be a normalized ratio"

            if ema_col in engineer.df.columns:
                ema_vals = engineer.df[ema_col].dropna()
                assert ema_vals.abs().max() < 1.0, f"{ema_col} should be a normalized ratio"

    def test_bollinger_bands_position_range(self, sample_ohlcv_data: pd.DataFrame):
        """Test that Bollinger Band position is in valid range."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        engineer.add_bollinger_bands()

        bb_position = engineer.df["bb_position"].dropna()

        # Most values should be between 0 and 1, with some outliers allowed
        within_range = ((bb_position >= -0.5) & (bb_position <= 1.5)).sum()
        total = len(bb_position)

        assert within_range / total > 0.9, "Most bb_position values should be near [0, 1]"

    def test_get_features_array_shape(self, sample_ohlcv_data: pd.DataFrame):
        """Test that get_features_array returns correct shape."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        engineer.add_all_features()

        features_array = engineer.get_features_array()
        feature_cols = engineer.get_feature_columns()

        assert features_array.shape[1] == len(feature_cols)
        assert features_array.shape[0] == len(engineer.df)

    def test_no_inf_values(self, sample_ohlcv_data: pd.DataFrame):
        """Test that features don't contain infinite values."""
        engineer = FeatureEngineer(sample_ohlcv_data)
        engineer.add_all_features()

        feature_cols = engineer.get_feature_columns()
        for col in feature_cols:
            vals = engineer.df[col]
            assert not np.isinf(vals).any(), f"Column {col} contains infinite values"


class TestMarketRegimeFeatures:
    """Tests for F1 – market regime indicators (adx_14, vol_regime)."""

    def test_adx_column_present(self, sample_ohlcv_data: pd.DataFrame):
        """adx_14 column must be created."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_market_regime_features()
        assert "adx_14" in eng.df.columns

    def test_adx_range(self, sample_ohlcv_data: pd.DataFrame):
        """ADX values should be in [0, 100]."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_market_regime_features()
        adx = eng.df["adx_14"].dropna()
        assert adx.min() >= 0.0, "ADX should be >= 0"
        assert adx.max() <= 100.0, "ADX should be <= 100"

    def test_vol_regime_column_present(self, sample_ohlcv_data: pd.DataFrame):
        """vol_regime column must be created."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_market_regime_features()
        assert "vol_regime" in eng.df.columns

    def test_vol_regime_range(self, sample_ohlcv_data: pd.DataFrame):
        """vol_regime values must lie in [0, 1]."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_market_regime_features()
        vr = eng.df["vol_regime"].dropna()
        assert len(vr) > 0, "vol_regime should have non-NaN values"
        assert vr.min() >= 0.0, "vol_regime should be >= 0"
        assert vr.max() <= 1.0, "vol_regime should be <= 1"

    def test_market_regime_in_feature_columns(self, sample_ohlcv_data: pd.DataFrame):
        """adx_14 and vol_regime must appear in get_feature_columns()."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_all_features()
        cols = eng.get_feature_columns()
        assert "adx_14" in cols
        assert "vol_regime" in cols

    def test_no_nan_after_add_all(self, sample_ohlcv_data: pd.DataFrame):
        """add_all_features drops NaNs; adx/vol_regime must not reintroduce them."""
        eng = FeatureEngineer(sample_ohlcv_data)
        result = eng.add_all_features()
        assert not result["adx_14"].isna().any()
        assert not result["vol_regime"].isna().any()


class TestCrossAssetFeatures:
    """Tests for F2 – cross-asset features (omxs30_corr, usdsek_ret, eursek_ret)."""

    def _make_ref_df(self, index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
        """Build a tiny reference DataFrame with a ``close`` column."""
        rng = np.random.default_rng(seed)
        prices = 100 * np.cumprod(1 + rng.standard_normal(len(index)) * 0.01)
        return pd.DataFrame({"close": prices}, index=index)

    def test_cross_asset_columns_present_with_data(self, sample_ohlcv_data: pd.DataFrame):
        """All three cross-asset columns should exist when reference data is supplied."""
        idx = sample_ohlcv_data.index
        ref = {
            "omxs30": self._make_ref_df(idx, seed=1),
            "usdsek": self._make_ref_df(idx, seed=2),
            "eursek": self._make_ref_df(idx, seed=3),
        }
        eng = FeatureEngineer(sample_ohlcv_data, reference_data=ref)
        eng.add_cross_asset_features()
        for col in ("omxs30_corr", "usdsek_ret", "eursek_ret"):
            assert col in eng.df.columns, f"{col} missing"

    def test_cross_asset_columns_present_without_data(self, sample_ohlcv_data: pd.DataFrame):
        """Columns must exist (filled with 0) when no reference data is provided."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_cross_asset_features()
        for col in ("omxs30_corr", "usdsek_ret", "eursek_ret"):
            assert col in eng.df.columns, f"{col} missing without reference data"

    def test_no_nan_in_cross_asset_columns(self, sample_ohlcv_data: pd.DataFrame):
        """Cross-asset columns must not contain NaN after the method runs."""
        idx = sample_ohlcv_data.index
        ref = {
            "omxs30": self._make_ref_df(idx, seed=1),
            "usdsek": self._make_ref_df(idx, seed=2),
            "eursek": self._make_ref_df(idx, seed=3),
        }
        eng = FeatureEngineer(sample_ohlcv_data, reference_data=ref)
        eng.add_price_features()  # needed for stock returns
        eng.add_cross_asset_features()
        for col in ("omxs30_corr", "usdsek_ret", "eursek_ret"):
            assert not eng.df[col].isna().any(), f"{col} contains NaN"

    def test_omxs30_corr_range(self, sample_ohlcv_data: pd.DataFrame):
        """Correlation must be in [-1, 1]."""
        idx = sample_ohlcv_data.index
        ref = {"omxs30": self._make_ref_df(idx, seed=10)}
        eng = FeatureEngineer(sample_ohlcv_data, reference_data=ref)
        eng.add_price_features()
        eng.add_cross_asset_features()
        corr = eng.df["omxs30_corr"]
        assert corr.min() >= -1.0 - 1e-9
        assert corr.max() <= 1.0 + 1e-9

    def test_cross_asset_in_feature_columns(self, sample_ohlcv_data: pd.DataFrame):
        """All three cross-asset columns must appear in get_feature_columns()."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_all_features()
        cols = eng.get_feature_columns()
        for col in ("omxs30_corr", "usdsek_ret", "eursek_ret"):
            assert col in cols, f"{col} not in feature columns"


class TestCalendarFeatures:
    """Tests for F3 – calendar effects."""

    def test_calendar_columns_present(self, sample_ohlcv_data: pd.DataFrame):
        """All four calendar columns must be created."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_calendar_features()
        for col in ("day_of_week", "month", "is_month_end", "is_quarter_end"):
            assert col in eng.df.columns, f"{col} missing"

    def test_day_of_week_range(self, sample_ohlcv_data: pd.DataFrame):
        """day_of_week must be in [0, 1]."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_calendar_features()
        dow = eng.df["day_of_week"]
        assert dow.min() >= 0.0
        assert dow.max() <= 1.0

    def test_month_range(self, sample_ohlcv_data: pd.DataFrame):
        """month must be in [0, 1]."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_calendar_features()
        m = eng.df["month"]
        assert m.min() >= 0.0
        assert m.max() <= 1.0

    def test_is_month_end_binary(self, sample_ohlcv_data: pd.DataFrame):
        """is_month_end must only contain 0 or 1."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_calendar_features()
        vals = eng.df["is_month_end"].unique()
        assert set(vals).issubset({0, 1})

    def test_is_quarter_end_binary(self, sample_ohlcv_data: pd.DataFrame):
        """is_quarter_end must only contain 0 or 1."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_calendar_features()
        vals = eng.df["is_quarter_end"].unique()
        assert set(vals).issubset({0, 1})

    def test_quarter_end_subset_of_month_end(self, sample_ohlcv_data: pd.DataFrame):
        """Every quarter-end day must also be a month-end day."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_calendar_features()
        qe = eng.df["is_quarter_end"] == 1
        me = eng.df["is_month_end"] == 1
        assert (qe & ~me).sum() == 0, "quarter-end must be a subset of month-end"

    def test_no_nan_in_calendar_columns(self, sample_ohlcv_data: pd.DataFrame):
        """Calendar columns must not contain NaN."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_calendar_features()
        for col in ("day_of_week", "month", "is_month_end", "is_quarter_end"):
            assert not eng.df[col].isna().any(), f"{col} contains NaN"

    def test_calendar_in_feature_columns(self, sample_ohlcv_data: pd.DataFrame):
        """All four calendar columns must appear in get_feature_columns()."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_all_features()
        cols = eng.get_feature_columns()
        for col in ("day_of_week", "month", "is_month_end", "is_quarter_end"):
            assert col in cols, f"{col} not in feature columns"

    def test_quarter_end_only_in_quarter_months(self, sample_ohlcv_data: pd.DataFrame):
        """is_quarter_end must only be set in Mar/Jun/Sep/Dec."""
        eng = FeatureEngineer(sample_ohlcv_data)
        eng.add_calendar_features()
        qe_idx = eng.df.index[eng.df["is_quarter_end"] == 1]
        quarter_end_months = {3, 6, 9, 12}
        for dt in qe_idx:
            assert dt.month in quarter_end_months, (
                f"quarter_end set on {dt} which is month {dt.month}"
            )
