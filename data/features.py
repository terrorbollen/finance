"""Feature engineering module for technical indicators."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Calculates technical indicators and features for ML models."""

    def __init__(self, df: pd.DataFrame, reference_data: dict[str, pd.DataFrame] | None = None):
        """
        Initialize with price data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            reference_data: Optional dict of reference DataFrames keyed by name
                            (e.g. {"omxs30": omxs30_df, "usdsek": usdsek_df}).
                            Each DataFrame must have at minimum a ``close`` column
                            and share the same DatetimeIndex as ``df``.
        """
        self.df = df.copy()
        self.reference_data = reference_data or {}

    def add_all_features(self) -> pd.DataFrame:
        """Add all technical indicators to the DataFrame."""
        self.add_moving_averages()
        self.add_rsi()
        self.add_macd()
        self.add_bollinger_bands()
        self.add_volume_features()
        self.add_momentum_features()
        self.add_price_features()
        self.add_market_regime_features()
        self.add_cross_asset_features()
        self.add_calendar_features()

        # Drop rows with NaN values from indicator calculations
        self.df = self.df.dropna()

        return self.df

    def add_moving_averages(self, periods: list[int] | None = None) -> pd.DataFrame:
        """Add Moving Average features as price ratios (normalized)."""
        if periods is None:
            periods = [5, 10, 20, 50, 200]
        for period in periods:
            sma = self.df["close"].rolling(window=period).mean()
            ema = self.df["close"].ewm(span=period, adjust=False).mean()

            # Store as ratio to current price (normalized, scale-independent)
            self.df[f"sma_{period}_ratio"] = (self.df["close"] - sma) / sma
            self.df[f"ema_{period}_ratio"] = (self.df["close"] - ema) / ema

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

    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Add Moving Average Convergence Divergence (MACD) - normalized.

        Components:
        - MACD line: Fast EMA - Slow EMA (normalized by price)
        - Signal line: EMA of MACD line
        - Histogram: MACD - Signal (positive = bullish, negative = bearish)
        """
        ema_fast = self.df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df["close"].ewm(span=slow, adjust=False).mean()

        # Normalize MACD by price to make it scale-independent
        macd = ema_fast - ema_slow
        self.df["macd"] = macd / self.df["close"] * 100  # As percentage
        self.df["macd_signal"] = self.df["macd"].ewm(span=signal, adjust=False).mean()
        self.df["macd_histogram"] = self.df["macd"] - self.df["macd_signal"]

        return self.df

    def add_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        Add Bollinger Bands (normalized features only).

        Bands expand during volatility and contract during consolidation.
        Price near upper band may indicate overbought, lower band oversold.
        """
        sma = self.df["close"].rolling(window=period).mean()
        std = self.df["close"].rolling(window=period).std()

        bb_upper = sma + (std * std_dev)
        bb_lower = sma - (std * std_dev)

        # Only store normalized/ratio features (not raw price levels)
        self.df["bb_width"] = (bb_upper - bb_lower) / sma
        self.df["bb_position"] = (self.df["close"] - bb_lower) / (bb_upper - bb_lower)

        return self.df

    def add_volume_features(self) -> pd.DataFrame:
        """Add volume-based features (normalized to avoid huge values)."""
        # Volume moving average (for internal calculation only)
        volume_sma_20 = self.df["volume"].rolling(window=20).mean()

        # Volume ratio (current vs average) - already normalized
        # Clip to avoid extreme values
        self.df["volume_ratio"] = (self.df["volume"] / volume_sma_20).clip(-10, 10)

        # Volume trend (is volume increasing or decreasing?)
        # Use log difference instead of pct_change to avoid inf
        vol_log = pd.Series(np.log1p(volume_sma_20), index=self.df.index)
        self.df["volume_trend"] = (vol_log - vol_log.shift(5)).clip(-2, 2)

        # On-Balance Volume (OBV) - use normalized rate of change instead of cumulative
        obv = [0]
        for i in range(1, len(self.df)):
            if self.df["close"].iloc[i] > self.df["close"].iloc[i - 1]:
                obv.append(obv[-1] + self.df["volume"].iloc[i])
            elif self.df["close"].iloc[i] < self.df["close"].iloc[i - 1]:
                obv.append(obv[-1] - self.df["volume"].iloc[i])
            else:
                obv.append(obv[-1])

        obv_series = pd.Series(obv, index=self.df.index)
        # Normalize OBV: use log difference instead of pct_change
        obv_log = pd.Series(np.log1p(np.abs(obv_series)) * np.sign(obv_series), index=self.df.index)
        self.df["obv_roc"] = (obv_log - obv_log.shift(10)).clip(-5, 5)

        return self.df

    def add_momentum_features(self) -> pd.DataFrame:
        """Add momentum-based features."""
        # Price Rate of Change (ROC) - already normalized as percentage
        for period in [5, 10, 20]:
            self.df[f"roc_{period}"] = (
                (self.df["close"] - self.df["close"].shift(period)) / self.df["close"].shift(period)
            ) * 100

        # Average True Range (ATR) - normalize by price for scale independence
        high_low = self.df["high"] - self.df["low"]
        high_close = (self.df["high"] - self.df["close"].shift()).abs()
        low_close = (self.df["low"] - self.df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=14).mean()
        self.df["atr"] = atr / self.df["close"] * 100  # As percentage of price

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
        self.df["dist_from_low"] = (self.df["close"] - self.df["low"].rolling(20).min()) / self.df[
            "close"
        ]

        return self.df

    # ------------------------------------------------------------------
    # F1 – Market regime indicators
    # ------------------------------------------------------------------

    def add_market_regime_features(self, adx_period: int = 14) -> pd.DataFrame:
        """
        Add market-regime features: ADX trend strength and volatility regime.

        New columns
        -----------
        adx_14
            Average Directional Index over ``adx_period`` bars (0-100).
            Higher values indicate a stronger trend.
        vol_regime
            Rolling 20-day return std expressed as its percentile rank within
            the trailing 252-day window.  Values near 1 = high-vol regime,
            near 0 = low-vol regime.
        """
        # --- ADX -----------------------------------------------------------
        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # True Range
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)

        # Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()

        # Only positive when that direction dominates
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)

        # Smoothed using Wilder's EMA (equivalent to RMA)
        alpha = 1.0 / adx_period
        atr_wilder = tr.ewm(alpha=alpha, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(alpha=alpha, adjust=False).mean() / atr_wilder
        di_minus = 100 * dm_minus.ewm(alpha=alpha, adjust=False).mean() / atr_wilder

        dx = (100 * (di_plus - di_minus).abs() / (di_plus + di_minus)).fillna(0)
        self.df["adx_14"] = dx.ewm(alpha=alpha, adjust=False).mean()

        # --- Volatility regime ---------------------------------------------
        returns = close.pct_change()
        vol_20 = returns.rolling(window=20).std()

        def _rolling_percentile(s: pd.Series, window: int) -> pd.Series:
            """Rank each value within its trailing ``window``-length history."""

            def _rank(arr: np.ndarray) -> float:
                if len(arr) < 2:
                    return np.nan
                current = arr[-1]
                history = arr[:-1]
                return float(np.sum(history <= current) / len(history))

            return s.rolling(window=window, min_periods=window).apply(_rank, raw=True)

        self.df["vol_regime"] = _rolling_percentile(vol_20, 252)

        return self.df

    # ------------------------------------------------------------------
    # F2 – Cross-asset features
    # ------------------------------------------------------------------

    def add_cross_asset_features(self, corr_window: int = 20) -> pd.DataFrame:
        """
        Add cross-asset features using reference data supplied at construction.

        Expected keys in ``self.reference_data``
        -----------------------------------------
        ``omxs30``
            DataFrame with a ``close`` column for the OMXS30 index.
        ``usdsek``
            DataFrame with a ``close`` column for USD/SEK.
        ``eursek``
            DataFrame with a ``close`` column for EUR/SEK.

        New columns
        -----------
        omxs30_corr
            Rolling 20-day Pearson correlation of the stock's daily returns
            with OMXS30 daily returns.  Missing values filled with 0.
        usdsek_ret
            Daily return of USD/SEK.  Missing values filled with 0.
        eursek_ret
            Daily return of EUR/SEK.  Missing values filled with 0.
        """
        stock_returns = self.df["close"].pct_change()

        # OMXS30 correlation
        if "omxs30" in self.reference_data:
            omxs30_close = self.reference_data["omxs30"]["close"].reindex(self.df.index)
            omxs30_close = omxs30_close.ffill()
            omxs30_returns = omxs30_close.pct_change()
            self.df["omxs30_corr"] = (
                stock_returns.rolling(window=corr_window)
                .corr(omxs30_returns)
                .fillna(0)
            )
        else:
            self.df["omxs30_corr"] = 0.0

        # USD/SEK daily return
        if "usdsek" in self.reference_data:
            usdsek_close = self.reference_data["usdsek"]["close"].reindex(self.df.index)
            usdsek_close = usdsek_close.ffill()
            self.df["usdsek_ret"] = usdsek_close.pct_change().fillna(0)
        else:
            self.df["usdsek_ret"] = 0.0

        # EUR/SEK daily return
        if "eursek" in self.reference_data:
            eursek_close = self.reference_data["eursek"]["close"].reindex(self.df.index)
            eursek_close = eursek_close.ffill()
            self.df["eursek_ret"] = eursek_close.pct_change().fillna(0)
        else:
            self.df["eursek_ret"] = 0.0

        return self.df

    # ------------------------------------------------------------------
    # F3 – Calendar effects
    # ------------------------------------------------------------------

    def add_calendar_features(self) -> pd.DataFrame:
        """
        Add calendar-effect features derived from the DataFrame's DatetimeIndex.

        New columns
        -----------
        day_of_week
            Day of the week normalized to [0, 1]: Monday = 0.0, Friday = 1.0.
        month
            Month normalized to [0, 1]: January ≈ 0.0, December ≈ 1.0.
        is_month_end
            1 if the row is the last trading day of the calendar month, else 0.
        is_quarter_end
            1 if the row is the last trading day of a quarter-end month
            (March, June, September, December), else 0.
        """
        idx = self.df.index

        # day_of_week: 0=Mon … 4=Fri → normalize to [0, 1]
        # Clip to [0, 4] in case the index contains weekend dates (e.g. in tests).
        dow = np.clip(idx.dayofweek, 0, 4)
        self.df["day_of_week"] = dow / 4.0

        # month: 1-12 → normalize to [0, 1]
        self.df["month"] = (idx.month - 1) / 11.0

        # is_month_end: last trading day of each calendar month
        # A row is month-end when the next row's month differs (or it's the last row).
        next_month = pd.Series(idx.month, index=idx).shift(-1)
        self.df["is_month_end"] = (pd.Series(idx.month, index=idx) != next_month).astype(int)

        # is_quarter_end: month-end AND it's a quarter-end month
        quarter_end_months = {3, 6, 9, 12}
        is_qtr_month = pd.Series(idx.month, index=idx).isin(quarter_end_months)
        self.df["is_quarter_end"] = (
            (self.df["is_month_end"] == 1) & is_qtr_month
        ).astype(int)

        return self.df

    def get_feature_columns(self) -> list[str]:
        """Return list of feature column names (excluding OHLCV)."""
        excluded = ["open", "high", "low", "close", "volume"]
        return [col for col in self.df.columns if col not in excluded]

    def get_features_array(self) -> np.ndarray:
        """Return features as numpy array for ML model input."""
        feature_cols = self.get_feature_columns()
        return self.df[feature_cols].values
