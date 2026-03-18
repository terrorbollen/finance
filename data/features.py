"""Feature engineering module for technical indicators."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Calculates technical indicators and features for ML models.

    Produces 8 features chosen for low mutual redundancy:
        rsi, macd_histogram, momentum_10, returns,
        atr, bb_position, volume_ratio, adx_14
    """

    def __init__(self, df: pd.DataFrame, reference_data: dict[str, pd.DataFrame] | None = None):
        """
        Initialize with price data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            reference_data: Unused — kept for interface compatibility.
        """
        self.df = df.copy()

    def add_all_features(self) -> pd.DataFrame:
        """Compute all 8 features and return the enriched DataFrame."""
        self._add_rsi()
        self._add_macd_histogram()
        self._add_momentum()
        self._add_returns()
        self._add_atr()
        self._add_bb_position()
        self._add_volume_ratio()
        self._add_adx()

        self.df = self.df.dropna()
        return self.df

    # ------------------------------------------------------------------
    # Individual indicator methods
    # ------------------------------------------------------------------

    def _add_rsi(self, period: int = 14) -> None:
        """RSI(14) — overbought/oversold oscillator, range 0–100."""
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        self.df["rsi"] = 100 - (100 / (1 + rs))

    def _add_macd_histogram(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        """MACD histogram (MACD line minus signal line), normalised by price.

        Positive = bullish momentum accelerating; negative = bearish.
        """
        ema_fast = self.df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["close"].ewm(span=slow, adjust=False).mean()
        macd_line = (ema_fast - ema_slow) / self.df["close"] * 100
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        self.df["macd_histogram"] = macd_line - signal_line

    def _add_momentum(self, period: int = 10) -> None:
        """10-day price momentum: percentage change over the lookback period."""
        self.df["momentum_10"] = self.df["close"].pct_change(periods=period) * 100

    def _add_returns(self) -> None:
        """1-day percentage return."""
        self.df["returns"] = self.df["close"].pct_change() * 100

    def _add_atr(self, period: int = 14) -> None:
        """ATR(14) expressed as % of price — scale-independent volatility.

        Used both as a feature and by SignalGenerator for stop-loss sizing.
        """
        high_low = self.df["high"] - self.df["low"]
        high_close = (self.df["high"] - self.df["close"].shift()).abs()
        low_close = (self.df["low"] - self.df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df["atr"] = true_range.rolling(window=period).mean() / self.df["close"] * 100

    def _add_bb_position(self, period: int = 20, std_dev: float = 2.0) -> None:
        """Price position within Bollinger Bands, range 0–1.

        0 = at lower band (oversold region), 1 = at upper band (overbought region).
        """
        sma = self.df["close"].rolling(window=period).mean()
        std = self.df["close"].rolling(window=period).std()
        bb_upper = sma + std * std_dev
        bb_lower = sma - std * std_dev
        band_width = bb_upper - bb_lower
        self.df["bb_position"] = (self.df["close"] - bb_lower).where(band_width > 0, 0.5) / band_width.where(band_width > 0, 1.0)

    def _add_volume_ratio(self, period: int = 20) -> None:
        """Current volume relative to its rolling average, clipped at 10x.

        Falls back to 1.0 (neutral) when volume data is unavailable (e.g. indexes).
        """
        if "volume" not in self.df.columns or self.df["volume"].isna().all() or (self.df["volume"] == 0).all():
            self.df["volume_ratio"] = 1.0
            return
        vol_avg = self.df["volume"].rolling(window=period).mean()
        raw = (self.df["volume"] / vol_avg).clip(upper=10)
        self.df["volume_ratio"] = raw.fillna(1.0)

    def _add_adx(self, period: int = 14) -> None:
        """ADX(14) — trend strength, range 0–100.

        Low (<20) = ranging market; high (>40) = strong trend.
        Uses Wilder's smoothing (equivalent to RMA).
        """
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)

        dm_plus = high.diff()
        dm_minus = -low.diff()
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

        alpha = 1.0 / period
        atr_w = tr.ewm(alpha=alpha, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(alpha=alpha, adjust=False).mean() / atr_w
        di_minus = 100 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / atr_w

        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).fillna(0)
        self.df["adx_14"] = dx.ewm(alpha=alpha, adjust=False).mean()

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_feature_columns(self) -> list[str]:
        """Return feature column names (excludes raw OHLCV columns)."""
        excluded = {"open", "high", "low", "close", "volume"}
        return [col for col in self.df.columns if col not in excluded]

    def get_features_array(self) -> np.ndarray:
        """Return features as a numpy array for ML input."""
        return self.df[self.get_feature_columns()].values
