"""Signal generation logic combining model predictions into actionable signals."""

import json
import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from data.features import FeatureEngineer
from data.fetcher import StockDataFetcher
from models.signal_model import SignalModel
from signals.calibration import ConfidenceCalibrator


class Direction(Enum):
    """Trading signal direction."""

    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"


@dataclass
class Signal:
    """Trading signal with all relevant information."""

    ticker: str
    direction: Direction
    confidence: float  # 0-100% (calibrated if calibrator available)
    current_price: float
    entry_price: float
    target_price: float
    stop_loss: float
    predicted_change: float  # Percentage
    timestamp: str
    raw_confidence: float | None = None  # 0-100% (raw model output)
    is_calibrated: bool = False

    def __str__(self) -> str:
        """Format signal for display."""
        if self.is_calibrated and self.raw_confidence is not None:
            confidence_str = f"{self.confidence:.1f}% (raw: {self.raw_confidence:.1f}%)"
        else:
            confidence_str = f"{self.confidence:.1f}%"

        lines = [
            f"{'='*50}",
            f"  SIGNAL: {self.ticker}",
            f"{'='*50}",
            f"  Direction:        {self.direction.value}",
            f"  Confidence:       {confidence_str}",
            f"  Current Price:    {self.current_price:.2f}",
            f"  Entry Price:      {self.entry_price:.2f}",
            f"  Target Price:     {self.target_price:.2f} ({self.predicted_change:+.2f}%)",
            f"  Stop Loss:        {self.stop_loss:.2f}",
            f"  Generated:        {self.timestamp}",
            f"{'='*50}",
        ]
        return "\n".join(lines)


class SignalGenerator:
    """Generates trading signals using the trained model."""

    def __init__(
        self,
        model_path: str = "checkpoints/signal_model.weights.h5",
        sequence_length: int = 20,
        stop_loss_pct: float = 0.05,
        calibration_path: str | None = None,
        min_confidence: float | None = None,
        atr_multiplier: float = 2.0,
    ):
        """
        Initialize the signal generator.

        Args:
            model_path: Path to trained model weights
            sequence_length: Sequence length used during training
            stop_loss_pct: Fallback stop loss percentage if ATR unavailable
            calibration_path: Path to calibration JSON (default: checkpoints/calibration.json)
            min_confidence: Minimum calibrated confidence (0-100) to trade.
                            Signals below this threshold are forced to HOLD.
            atr_multiplier: Stop loss distance as a multiple of ATR (default: 2.0)
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.stop_loss_pct = stop_loss_pct
        self.min_confidence = min_confidence
        self.atr_multiplier = atr_multiplier
        self.model: SignalModel | None = None
        self.fetcher = StockDataFetcher(period="1y")

        # Normalization params (should match training)
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.feature_columns: list[str] | None = None
        self.input_dim: int | None = None

        # Confidence calibration
        self.calibrator: ConfidenceCalibrator | None = None
        self.calibration_path = calibration_path or "checkpoints/calibration.json"

        # Load training config if available
        self._load_config()
        self._load_calibrator()

    def _load_config(self):
        """Load training configuration from file."""
        config_path = self.model_path.replace(".weights.h5", "_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.feature_columns = config.get("feature_columns")
            self.feature_mean = np.array(config.get("feature_mean"))
            self.feature_std = np.array(config.get("feature_std"))
            self.sequence_length = config.get("sequence_length", self.sequence_length)
            self.input_dim = config.get("input_dim")

    def _load_calibrator(self):
        """Load confidence calibrator if available."""
        if os.path.exists(self.calibration_path):
            try:
                self.calibrator = ConfidenceCalibrator.load(self.calibration_path)
                print(f"Loaded confidence calibration from {self.calibration_path}")
            except Exception as e:
                print(f"Warning: Could not load calibration: {e}")
                self.calibrator = None

    def _load_model(self, input_dim: int):
        """Load the trained model."""
        self.model = SignalModel(
            input_dim=input_dim, sequence_length=self.sequence_length
        )
        try:
            self.model.load(self.model_path)
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using untrained model (predictions will be random)")

    def generate(self, ticker: str) -> Signal:
        """
        Generate a trading signal for a given ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Signal object with direction, confidence, and price targets
        """
        # Fetch data
        df = self.fetcher.fetch(ticker)

        # Add features
        engineer = FeatureEngineer(df)
        df_features = engineer.add_all_features()

        # Use feature columns from training config if available
        if self.feature_columns is not None:
            # Filter to only columns that exist in the data
            available_cols = [c for c in self.feature_columns if c in df_features.columns]
            if len(available_cols) != len(self.feature_columns):
                missing = set(self.feature_columns) - set(available_cols)
                print(f"Warning: Missing features from training: {missing}")
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
            # Fallback to current data statistics
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
            features_norm = (features - mean) / std

        # Create sequence from most recent data
        if len(features_norm) < self.sequence_length:
            raise ValueError(
                f"Not enough data points. Need {self.sequence_length}, got {len(features_norm)}"
            )

        X = features_norm[-self.sequence_length :].reshape(
            1, self.sequence_length, -1
        )

        # Get predictions
        signal_probs, signal_class, price_target = self.model.predict(X)

        # Extract values
        direction_idx = signal_class[0]
        direction = [Direction.BUY, Direction.HOLD, Direction.SELL][direction_idx]
        raw_confidence = float(signal_probs[0][direction_idx])
        predicted_change = float(price_target[0])

        # Apply confidence calibration if available
        if self.calibrator is not None and self.calibrator.is_fitted:
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

        # Current price
        current_price = float(df_features["close"].iloc[-1])

        # Calculate price targets
        entry_price = current_price
        target_price = current_price * (1 + predicted_change / 100)

        # ATR-based dynamic stop loss (atr column is already % of price)
        if "atr" in df_features.columns:
            atr_pct = float(df_features["atr"].iloc[-1])
            stop_distance = (atr_pct / 100) * self.atr_multiplier
        else:
            stop_distance = self.stop_loss_pct

        if direction == Direction.SELL:
            stop_loss = current_price * (1 + stop_distance)
        else:
            stop_loss = current_price * (1 - stop_distance)

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
            raw_confidence=raw_confidence_pct,
            is_calibrated=is_calibrated,
        )

    def scan(self, tickers: list[str]) -> list[Signal]:
        """
        Generate signals for multiple tickers.

        Args:
            tickers: List of ticker symbols

        Returns:
            List of Signal objects
        """
        signals = []
        for ticker in tickers:
            try:
                signal = self.generate(ticker)
                signals.append(signal)
            except Exception as e:
                print(f"Error generating signal for {ticker}: {e}")

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
