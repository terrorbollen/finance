"""Pytest fixtures for the trading signal test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n_days = 300

    # Generate realistic price data with random walk
    base_price = 100.0
    returns = np.random.randn(n_days) * 0.02  # 2% daily volatility
    close = base_price * np.cumprod(1 + returns)

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n_days) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n_days) * 0.01))
    open_ = close * (1 + np.random.randn(n_days) * 0.005)

    # Ensure high >= close >= low
    high = np.maximum(high, np.maximum(close, open_))
    low = np.minimum(low, np.minimum(close, open_))

    # Generate volume
    volume = np.random.randint(100000, 1000000, n_days).astype(float)

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    # Add datetime index
    df.index = pd.date_range(start="2023-01-01", periods=n_days, freq="D")

    return df


@pytest.fixture
def sample_features_df(sample_ohlcv_data: pd.DataFrame) -> pd.DataFrame:
    """Generate sample DataFrame with features added."""
    from data.features import FeatureEngineer

    engineer = FeatureEngineer(sample_ohlcv_data)
    return engineer.add_all_features()


@pytest.fixture
def sample_confidences() -> tuple[np.ndarray, np.ndarray]:
    """Generate sample confidence scores and correctness labels."""
    np.random.seed(42)
    n_samples = 200

    # Generate confidences with varying accuracy by bucket
    confidences = np.random.uniform(0, 1, n_samples)

    # Higher confidences should be more accurate
    correct_probs = 0.3 + 0.5 * confidences  # 30% to 80% accuracy
    correct = np.random.random(n_samples) < correct_probs

    return confidences, correct.astype(float)
