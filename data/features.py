"""Feature engineering module for technical indicators."""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Calculates technical indicators and features for ML models."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with price data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
        """
        self.df = df.copy()

    def add_all_features(self) -> pd.DataFrame:
        """Add all technical indicators to the DataFrame."""
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_volume_features()
        self.add_momentum_features()
        self.add_price_features()

        # Drop rows with NaN values from indicator calculations
        self.df = self.df.dropna()

        return self.df

    def add_moving_averages(
        self, periods: list[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """Add Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)."""
        for period in periods:
            self.df[f"sma_{period}"] = self.df["close"].rolling(window=period).mean()
            self.df[f"ema_{period}"] = (
                self.df["close"].ewm(span=period, adjust=False).mean()
            )

        return self.df

    def add_rsi(self, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index (RSI).

        RSI ranges from 0-100:
        - Above 70: Overbought (potential sell signal)
        - Below 30: Oversold (potential buy signal)
        """
        delta = self.df["close"].diff()

        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        self.df["rsi"] = 100 - (100 / (1 + rs))

        return self.df

    def add_macd(
        self, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD).

        Components:
        - MACD line: Fast EMA - Slow EMA
        - Signal line: EMA of MACD line
        - Histogram: MACD - Signal (positive = bullish, negative = bearish)
        """
        ema_fast = self.df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["close"].ewm(span=slow, adjust=False).mean()

        self.df["macd"] = ema_fast - ema_slow
        self.df["macd_signal"] = self.df["macd"].ewm(span=signal, adjust=False).mean()
        self.df["macd_histogram"] = self.df["macd"] - self.df["macd_signal"]

        return self.df

    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands.

        Bands expand during volatility and contract during consolidation.
        Price near upper band may indicate overbought, lower band oversold.
        """
        sma = self.df["close"].rolling(window=period).mean()
        std = self.df["close"].rolling(window=period).std()

        self.df["bb_middle"] = sma
        self.df["bb_upper"] = sma + (std * std_dev)
        self.df["bb_lower"] = sma - (std * std_dev)
        self.df["bb_width"] = (self.df["bb_upper"] - self.df["bb_lower"]) / sma
        self.df["bb_position"] = (self.df["close"] - self.df["bb_lower"]) / (
            self.df["bb_upper"] - self.df["bb_lower"]
        )

        return self.df

    def add_volume_features(self) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume moving average
        self.df["volume_sma_20"] = self.df["volume"].rolling(window=20).mean()

        # Volume ratio (current vs average)
        self.df["volume_ratio"] = self.df["volume"] / self.df["volume_sma_20"]

        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df["close"].iloc[i] > self.df["close"].iloc[i - 1]:
                obv.append(obv[-1] + self.df["volume"].iloc[i])
            elif self.df["close"].iloc[i] < self.df["close"].iloc[i - 1]:
                obv.append(obv[-1] - self.df["volume"].iloc[i])
            else:
                obv.append(obv[-1])

        self.df["obv"] = obv

        return self.df

    def add_momentum_features(self) -> pd.DataFrame:
        """Add momentum-based features."""
        # Price Rate of Change (ROC)
        for period in [5, 10, 20]:
            self.df[f"roc_{period}"] = (
                (self.df["close"] - self.df["close"].shift(period))
                / self.df["close"].shift(period)
            ) * 100

        # Average True Range (ATR) - volatility measure
        high_low = self.df["high"] - self.df["low"]
        high_close = np.abs(self.df["high"] - self.df["close"].shift())
        low_close = np.abs(self.df["low"] - self.df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df["atr"] = true_range.rolling(window=14).mean()

        return self.df

    def add_price_features(self) -> pd.DataFrame:
        """Add price-derived features."""
        # Daily returns
        self.df["returns"] = self.df["close"].pct_change()

        # Log returns
        self.df["log_returns"] = np.log(self.df["close"] / self.df["close"].shift(1))

        # High-Low range
        self.df["hl_range"] = (self.df["high"] - self.df["low"]) / self.df["close"]

        # Distance from high/low
        self.df["dist_from_high"] = (
            self.df["high"].rolling(20).max() - self.df["close"]
        ) / self.df["close"]
        self.df["dist_from_low"] = (
            self.df["close"] - self.df["low"].rolling(20).min()
        ) / self.df["close"]

        return self.df

    def get_feature_columns(self) -> list[str]:
        """Return list of feature column names (excluding OHLCV)."""
        excluded = ["open", "high", "low", "close", "volume"]
        return [col for col in self.df.columns if col not in excluded]

    def get_features_array(self) -> np.ndarray:
        """Return features as numpy array for ML model input."""
        feature_cols = self.get_feature_columns()
        return self.df[feature_cols].values
