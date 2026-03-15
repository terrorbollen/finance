"""Signal generation logic combining model predictions into actionable signals."""

import os

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from data.features import FeatureEngineer
from data.fetcher import StockDataFetcher
from models.config import ModelConfig
from models.signal_model import SignalModel
from signals.calibration import ConfidenceCalibrator, DirectionalCalibrator
from signals.direction import Direction


class Signal(BaseModel):
    """Trading signal with all relevant information."""

    model_config = ConfigDict(use_enum_values=False)

    ticker: str
    direction: Direction
    confidence: float  # 0-100% (calibrated if calibrator available)
    current_price: float
    entry_price: float
    target_price: float
    stop_loss: float
    predicted_change: float  # Percentage
    timestamp: str
    position_size: float = 1.0  # fraction of capital (0-1), Kelly-based
    raw_confidence: float | None = None  # 0-100% (raw model output)
    is_calibrated: bool = False

    @field_validator("confidence")
    @classmethod
    def confidence_in_range(cls, v: float) -> float:
        if not 0 <= v <= 100:
            raise ValueError(f"confidence must be 0-100, got {v:.1f}")
        return v

    @field_validator("position_size")
    @classmethod
    def position_size_in_range(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError(f"position_size must be 0-1, got {v:.3f}")
        return v

    @field_validator("current_price", "entry_price", "target_price", "stop_loss")
    @classmethod
    def prices_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Price values must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def check_price_logic(self) -> "Signal":
        if self.direction == Direction.BUY:
            if self.stop_loss >= self.current_price:
                raise ValueError(
                    f"BUY signal: stop_loss ({self.stop_loss:.2f}) must be below current_price ({self.current_price:.2f})"
                )
            if self.target_price <= self.current_price:
                raise ValueError(
                    f"BUY signal: target_price ({self.target_price:.2f}) must be above current_price ({self.current_price:.2f})"
                )
        elif self.direction == Direction.SELL:
            if self.stop_loss <= self.current_price:
                raise ValueError(
                    f"SELL signal: stop_loss ({self.stop_loss:.2f}) must be above current_price ({self.current_price:.2f})"
                )
            if self.target_price >= self.current_price:
                raise ValueError(
                    f"SELL signal: target_price ({self.target_price:.2f}) must be below current_price ({self.current_price:.2f})"
                )
        return self

    def __str__(self) -> str:
        """Format signal for display."""
        if self.is_calibrated and self.raw_confidence is not None:
            confidence_str = f"{self.confidence:.1f}% (raw: {self.raw_confidence:.1f}%)"
        else:
            confidence_str = f"{self.confidence:.1f}%"

        lines = [
            f"{'=' * 50}",
            f"  SIGNAL: {self.ticker}",
            f"{'=' * 50}",
            f"  Direction:        {self.direction.value}",
            f"  Confidence:       {confidence_str}",
            f"  Position Size:    {self.position_size * 100:.1f}% of capital (Kelly)",
            f"  Current Price:    {self.current_price:.2f}",
            f"  Entry Price:      {self.entry_price:.2f}",
            f"  Target Price:     {self.target_price:.2f} ({self.predicted_change:+.2f}%)",
            f"  Stop Loss:        {self.stop_loss:.2f}",
            f"  Generated:        {self.timestamp}",
            f"{'=' * 50}",
        ]
        return "\n".join(lines)


class SignalGenerator:
    """Generates trading signals using the trained model."""

    def __init__(
        self,
        model_path: str | None = None,
        sequence_length: int = 20,
        stop_loss_pct: float = 0.05,
        calibration_path: str | None = None,
        min_confidence: float | None = 55.0,
        atr_multiplier: float = 2.0,
        take_profit_atr_multiplier: float = 3.0,
        max_position_size: float = 0.25,
        max_drawdown_pct: float | None = None,
        max_positions: int | None = None,
    ):
        """
        Initialize the signal generator.

        Args:
            model_path: Path to trained model weights. Defaults to checkpoint.
            sequence_length: Sequence length used during training
            stop_loss_pct: Fallback stop loss percentage if ATR unavailable
            calibration_path: Path to calibration JSON. Defaults to checkpoint file.
            min_confidence: Minimum calibrated confidence (0-100) to trade (default 55.0).
                            Signals below this threshold are forced to HOLD.
            atr_multiplier: Stop loss distance as a multiple of ATR (default: 2.0)
            take_profit_atr_multiplier: Take-profit distance as a multiple of ATR (default: 3.0).
            max_position_size: Hard cap on Kelly position size as fraction of capital (default: 0.25)
            max_drawdown_pct: Maximum portfolio drawdown (positive %) before new BUY/SELL signals
                              are suppressed. E.g. 10.0 halts trading when the portfolio is down
                              10% from its peak. None disables this limit.
            max_positions: Maximum number of concurrent open positions. Signals for new positions
                           are forced to HOLD once this count is reached. None disables this limit.
        """
        paths = ModelConfig.checkpoint_paths()
        self.model_path = model_path or paths["weights"]
        self.sequence_length = sequence_length
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence
        self.atr_multiplier = atr_multiplier
        self.take_profit_atr_multiplier = take_profit_atr_multiplier
        self.max_position_size = max_position_size
        self.max_drawdown_pct = max_drawdown_pct
        self.max_positions = max_positions
        self.model: SignalModel | None = None
        self.fetcher = StockDataFetcher(period="1y")

        # Normalization params (should match training)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.feature_columns: list[str] | None = None
        self.input_dim: int | None = None
        self.prediction_horizons: list[int] | None = None

        # Confidence calibration
        self.calibrator: ConfidenceCalibrator | None = None
        self.directional_calibrator: DirectionalCalibrator | None = None
        self.calibration_path = calibration_path or paths["calibration"]
        self.directional_calibration_path = (
            calibration_path.replace(".json", "_directional.json")
            if calibration_path
            else paths["calibration_directional"]
        )

        # Load training config if available
        self._load_config()
        self._load_calibrator()

    def _load_config(self) -> None:
        """Load training configuration from file."""
        config_path = self.model_path.replace(".weights.h5", "_config.json")
        if os.path.exists(config_path):
            cfg = ModelConfig.load(config_path)
            self.feature_columns = cfg.feature_columns
            self.feature_mean = cfg.feature_mean_array
            self.feature_std = cfg.feature_std_array
            self.sequence_length = cfg.sequence_length
            self.input_dim = cfg.input_dim
            self.prediction_horizons = cfg.prediction_horizons

    def _load_calibrator(self) -> None:
        """Load confidence calibrators if available. Prefers directional calibrator."""
        if os.path.exists(self.directional_calibration_path):
            try:
                self.directional_calibrator = DirectionalCalibrator.load(
                    self.directional_calibration_path
                )
                print(
                    f"Loaded directional calibration from {self.directional_calibration_path}"
                )
            except Exception as e:
                print(f"Warning: Could not load directional calibration: {e}")
                self.directional_calibrator = None

        if os.path.exists(self.calibration_path):
            try:
                self.calibrator = ConfidenceCalibrator.load(self.calibration_path)
                print(f"Loaded confidence calibration from {self.calibration_path}")
            except Exception as e:
                print(f"Warning: Could not load calibration: {e}")
                self.calibrator = None

    def _load_model(self, input_dim: int) -> None:
        """Load the trained model."""
        self.model = SignalModel(
            input_dim=input_dim,
            sequence_length=self.sequence_length,
            prediction_horizons=self.prediction_horizons,
        )
        try:
            self.model.load(self.model_path)
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using untrained model (predictions will be random)")

    def generate(
        self,
        ticker: str,
        portfolio_drawdown: float = 0.0,
        open_position_count: int = 0,
    ) -> Signal:
        """
        Generate a trading signal for a given ticker.

        Args:
            ticker: Stock ticker symbol
            portfolio_drawdown: Current portfolio drawdown as a positive percentage
                                (e.g. 8.5 means the portfolio is 8.5% below its peak).
                                Used to enforce max_drawdown_pct. Defaults to 0.
            open_position_count: Number of currently open positions. Used to enforce
                                 max_positions. Defaults to 0.

        Returns:
            Signal object with direction, confidence, and price targets
        """
        # Fetch data
        df = self.fetcher.fetch(ticker)

        # Add features (including cross-asset reference data)
        ref_data = self.fetcher.fetch_cross_asset_data(pd.DatetimeIndex(df.index))
        engineer = FeatureEngineer(df, reference_data=ref_data)
        df_features = engineer.add_all_features()

        # Use feature columns from training config if available
        if self.feature_columns is not None:
            available_cols = [c for c in self.feature_columns if c in df_features.columns]
            missing = set(self.feature_columns) - set(df_features.columns)
            if missing:
                missing_pct = len(missing) / len(self.feature_columns)
                if missing_pct > 0.1:
                    raise ValueError(
                        f"Too many features missing from training config ({len(missing)}/{len(self.feature_columns)}): {missing}"
                    )
                print(f"Warning: {len(missing)} minor features missing from training: {missing}")
            feature_cols = available_cols
        else:
            feature_cols = engineer.get_feature_columns()

        features = df_features[feature_cols].values

        # Initialize model if needed
        if self.model is None:
            input_dim = self.input_dim if self.input_dim else len(feature_cols)
            self._load_model(input_dim=input_dim)
        if self.model is None:
            raise RuntimeError("Failed to initialize model")

        # Normalize features using training statistics if available
        if self.feature_mean is not None and len(self.feature_mean) == len(feature_cols):
            features_norm = (features - self.feature_mean) / self.feature_std
        else:
            print(
                "Warning: Training normalization stats not available — falling back to "
                "current-window statistics. Predictions may be unreliable. Re-train the model."
            )
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
            features_norm = (features - mean) / std

        # Create sequence from most recent data
        if len(features_norm) < self.sequence_length:
            raise ValueError(
                f"Not enough data points. Need {self.sequence_length}, got {len(features_norm)}"
            )

        X = features_norm[-self.sequence_length :].reshape(1, self.sequence_length, -1)

        # Get predictions
        signal_probs, signal_class, price_target = self.model.predict(X)

        # Extract values
        direction_idx = signal_class[0]
        direction = [Direction.BUY, Direction.HOLD, Direction.SELL][direction_idx]
        raw_confidence = float(signal_probs[0][direction_idx])
        predicted_change = float(price_target[0])

        # Apply confidence calibration — directional takes priority over global
        if (
            self.directional_calibrator is not None
            and self.directional_calibrator.is_fitted_for(direction.value)
        ):
            calibrated_confidence = self.directional_calibrator.calibrate(
                direction.value, raw_confidence
            )
            confidence = calibrated_confidence * 100
            raw_confidence_pct = raw_confidence * 100
            is_calibrated = True
        elif self.calibrator is not None and self.calibrator.is_fitted:
            calibrated_confidence = self.calibrator.calibrate(raw_confidence)
            confidence = calibrated_confidence * 100
            raw_confidence_pct = raw_confidence * 100
            is_calibrated = True
        else:
            confidence = raw_confidence * 100
            raw_confidence_pct = None
            is_calibrated = False

        # Confidence threshold filtering: force HOLD if below minimum
        if self.min_confidence is not None and confidence < self.min_confidence:
            direction = Direction.HOLD

        # Portfolio risk limits: suppress new directional trades when limits are breached
        direction = self._apply_portfolio_limits(
            direction, ticker, portfolio_drawdown, open_position_count
        )

        # Current price
        current_price = float(df_features["close"].iloc[-1])

        # Calculate price targets
        entry_price = current_price

        # ATR-based stop loss and take-profit (atr column is already % of price)
        if "atr" in df_features.columns:
            atr_pct = float(df_features["atr"].iloc[-1])
            stop_distance = (atr_pct / 100) * self.atr_multiplier
            take_profit_distance = (atr_pct / 100) * self.take_profit_atr_multiplier
            use_atr_target = True
        else:
            stop_distance = self.stop_loss_pct
            take_profit_distance = abs(predicted_change / 100)
            use_atr_target = False

        if direction == Direction.SELL:
            stop_loss = current_price * (1 + stop_distance)
            if use_atr_target:
                target_price = current_price * (1 - take_profit_distance)
                predicted_change = -take_profit_distance * 100
            else:
                target_price = current_price * (1 + predicted_change / 100)
        elif direction == Direction.BUY:
            stop_loss = current_price * (1 - stop_distance)
            if use_atr_target:
                target_price = current_price * (1 + take_profit_distance)
                predicted_change = take_profit_distance * 100
            else:
                target_price = current_price * (1 + predicted_change / 100)
        else:  # HOLD
            stop_loss = current_price * (1 - stop_distance)
            target_price = current_price * (1 + predicted_change / 100)

        # Kelly criterion position sizing
        # f* = (p*b - q) / b, where p=win_prob, q=1-p, b=gain/loss ratio
        # Use half-Kelly and cap at max_position_size for safety
        position_size = self._kelly_position_size(
            confidence_pct=confidence,
            predicted_change_pct=abs(predicted_change),
            stop_distance_pct=stop_distance * 100,
            direction=direction,
        )

        # Create signal
        return Signal(
            ticker=ticker,
            direction=direction,
            confidence=confidence,
            current_price=current_price,
            entry_price=entry_price,
            target_price=target_price,
            stop_loss=stop_loss,
            predicted_change=predicted_change,
            timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            position_size=position_size,
            raw_confidence=raw_confidence_pct,
            is_calibrated=is_calibrated,
        )

    def _apply_portfolio_limits(
        self,
        direction: Direction,
        ticker: str,
        portfolio_drawdown: float,
        open_position_count: int,
    ) -> Direction:
        """
        Enforce portfolio-level risk limits, returning HOLD if a limit is breached.

        Limits only apply to BUY and SELL directions; HOLD passes through unchanged.

        Args:
            direction: Proposed signal direction
            ticker: Ticker label used in warning messages
            portfolio_drawdown: Current drawdown as positive % from peak
            open_position_count: Number of currently open positions

        Returns:
            Original direction, or HOLD if a limit is breached
        """
        if direction == Direction.HOLD:
            return direction

        if self.max_drawdown_pct is not None and portfolio_drawdown >= self.max_drawdown_pct:
            print(
                f"Portfolio drawdown {portfolio_drawdown:.1f}% >= limit {self.max_drawdown_pct:.1f}% — forcing HOLD for {ticker}"
            )
            return Direction.HOLD

        if self.max_positions is not None and open_position_count >= self.max_positions:
            print(
                f"Open positions {open_position_count} >= limit {self.max_positions} — forcing HOLD for {ticker}"
            )
            return Direction.HOLD

        return direction

    def _kelly_position_size(
        self,
        confidence_pct: float,
        predicted_change_pct: float,
        stop_distance_pct: float,
        direction: Direction,
    ) -> float:
        """
        Compute half-Kelly position size capped at max_position_size.

        Kelly fraction: f* = (p*b - q) / b
          p = win probability (confidence)
          q = 1 - p
          b = expected gain / expected loss (reward-to-risk ratio)

        Uses half-Kelly (0.5 * f*) to reduce variance.

        Args:
            confidence_pct: Model confidence in 0-100 range
            predicted_change_pct: Absolute predicted price move (%)
            stop_distance_pct: Stop loss distance from entry (%)
            direction: Signal direction (HOLD returns 0)

        Returns:
            Position size as fraction of capital in [0, max_position_size]
        """
        if direction == Direction.HOLD:
            return 0.0

        p = confidence_pct / 100.0
        q = 1.0 - p

        # Reward-to-risk ratio; guard against zero stop
        if stop_distance_pct < 1e-6:
            stop_distance_pct = self.stop_loss_pct * 100
        b = max(predicted_change_pct, 1e-6) / stop_distance_pct

        kelly_f = (p * b - q) / b

        # Negative Kelly means edge is against us → no trade
        if kelly_f <= 0:
            return 0.0

        # Half-Kelly, capped at max_position_size
        return min(0.5 * kelly_f, self.max_position_size)

    def scan(
        self,
        tickers: list[str],
        portfolio_drawdown: float = 0.0,
        open_position_count: int = 0,
    ) -> list[Signal]:
        """
        Generate signals for multiple tickers.

        Args:
            tickers: List of ticker symbols
            portfolio_drawdown: Current portfolio drawdown as a positive percentage.
                                Passed to each generate() call for drawdown limit enforcement.
            open_position_count: Number of currently open positions. The count is NOT
                                 incremented automatically as signals are generated — the
                                 caller is responsible for tracking live positions.

        Returns:
            List of Signal objects sorted by confidence descending
        """
        signals = []
        for ticker in tickers:
            try:
                signal = self.generate(
                    ticker,
                    portfolio_drawdown=portfolio_drawdown,
                    open_position_count=open_position_count,
                )
                signals.append(signal)
            except Exception as e:
                print(f"Error generating signal for {ticker}: {e}")

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
