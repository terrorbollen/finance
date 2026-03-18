"""Feature engineering module for technical indicators."""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Calculates technical indicators and features for ML models.

    Core features: rsi, macd_histogram, momentum_10, returns,
                   atr, bb_position, volume_ratio, adx_14
    Volatility:    vix_level, vix_1d_change, vix_stock_corr,
                   vstoxx_level, vstoxx_1d_change
    Macro:         oil_level, oil_1d_change, rate_level, rate_1d_change
    """

    def __init__(self, df: pd.DataFrame, reference_data: dict[str, pd.DataFrame] | None = None):
        """
        Initialize with price data.

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            reference_data: Optional cross-asset reference data (OMXS30, FX, VIX, VSTOXX).
                            Keys: "omxs30", "usdsek", "eursek", "vix", "vstoxx".
        """
        self.df = df.copy()
        self.reference_data = reference_data or {}

    def add_all_features(self) -> pd.DataFrame:
        """Compute all features and return the enriched DataFrame."""
        self._add_rsi()
        self._add_macd_histogram()
        self._add_momentum()
        self._add_returns()
        self._add_atr()
        self._add_bb_position()
        self._add_volume_ratio()
        self._add_adx()
        self._add_volatility_features()
        self._add_macro_features()

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
        self.df["bb_position"] = (self.df["close"] - bb_lower).where(
            band_width > 0, 0.5
        ) / band_width.where(band_width > 0, 1.0)

    def _add_volume_ratio(self, period: int = 20) -> None:
        """Current volume relative to its rolling average, clipped at 10x.

        Falls back to 1.0 (neutral) when volume data is unavailable (e.g. indexes).
        """
        if (
            "volume" not in self.df.columns
            or self.df["volume"].isna().all()
            or (self.df["volume"] == 0).all()
        ):
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

    def _add_volatility_features(self, corr_window: int = 20) -> None:
        """VIX and VSTOXX volatility regime features.

        Uses VIX (global fear gauge) and VSTOXX (European volatility index,
        more relevant for Swedish stocks) from reference_data if available.
        Falls back gracefully to neutral values when data is missing.

        Features added:
            vix_level      — VIX close normalised by its 252-day rolling mean,
                             so values >1 mean elevated fear relative to recent history.
            vix_1d_change  — Daily % change in VIX: captures sudden regime shifts.
            vix_stock_corr — 20-day rolling correlation between stock daily returns
                             and VIX daily changes: negative = risk-off stock (moves
                             against fear), positive = fear-amplified stock.
            vstoxx_level   — Same normalisation applied to VSTOXX (European vol index).
        """
        stock_returns = self.df["close"].pct_change()

        for key, col_prefix in (("vix", "vix"), ("vstoxx", "vstoxx")):
            ref = self.reference_data.get(key)
            if ref is None or ref.empty or (ref["close"] == 0).all():
                # No data available — fill with neutral values so dropna() keeps the rows.
                self.df[f"{col_prefix}_level"] = 1.0
                self.df[f"{col_prefix}_1d_change"] = 0.0
                if col_prefix == "vix":
                    self.df["vix_stock_corr"] = 0.0
                continue

            vol_close = ref["close"].reindex(self.df.index).ffill().bfill()

            # Normalised level: current / 252-day rolling mean (>1 = elevated fear)
            rolling_mean = vol_close.rolling(window=252, min_periods=20).mean()
            self.df[f"{col_prefix}_level"] = (
                vol_close / rolling_mean.where(rolling_mean > 0, 1.0)
            ).clip(0, 5)

            # Daily % change in volatility index
            self.df[f"{col_prefix}_1d_change"] = vol_close.pct_change() * 100

            # Rolling correlation between stock returns and vol-index changes (VIX only)
            if col_prefix == "vix":
                vol_changes = vol_close.pct_change()
                self.df["vix_stock_corr"] = stock_returns.rolling(
                    window=corr_window, min_periods=10
                ).corr(vol_changes)

    def _add_macro_features(self) -> None:
        """Brent crude oil and 10Y interest rate macro features.

        Uses Brent crude (BZ=F) and US 10Y Treasury yield (^TNX) from
        reference_data if available. Falls back gracefully to neutral values
        when data is missing so dropna() never removes rows.

        Features added:
            oil_level      — Brent close normalised by its 252-day rolling mean
                             (>1 = oil expensive relative to recent history;
                             relevant for energy costs and inflation regime).
            oil_1d_change  — Daily % change in Brent price (captures supply shocks).
            rate_level     — 10Y Treasury yield in % (e.g. 4.5 means 4.5%).
                             Used directly; already scale-independent as a rate.
            rate_1d_change — Daily change in 10Y yield in percentage points
                             (positive = rates rising; tightening financial conditions).
        """
        # --- Oil (Brent crude, BZ=F) ---
        oil_ref = self.reference_data.get("oil")
        if oil_ref is None or oil_ref.empty or (oil_ref["close"] == 0).all():
            self.df["oil_level"] = 1.0
            self.df["oil_1d_change"] = 0.0
        else:
            oil_close = oil_ref["close"].reindex(self.df.index).ffill().bfill()
            rolling_mean = oil_close.rolling(window=252, min_periods=20).mean()
            self.df["oil_level"] = (oil_close / rolling_mean.where(rolling_mean > 0, 1.0)).clip(
                0, 5
            )
            self.df["oil_1d_change"] = oil_close.pct_change() * 100

        # --- Interest rates (US 10Y Treasury yield, ^TNX) ---
        rate_ref = self.reference_data.get("rates")
        if rate_ref is None or rate_ref.empty or (rate_ref["close"] == 0).all():
            self.df["rate_level"] = 0.0
            self.df["rate_1d_change"] = 0.0
        else:
            rate_close = rate_ref["close"].reindex(self.df.index).ffill().bfill()
            self.df["rate_level"] = rate_close
            self.df["rate_1d_change"] = rate_close.diff()

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
