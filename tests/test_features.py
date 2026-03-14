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
