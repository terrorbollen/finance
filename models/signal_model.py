"""TensorFlow model for trading signal classification and price prediction."""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Optional
import os


class SignalModel:
    """
    Multi-output model for trading signals.

    Outputs:
    - Signal classification: Buy (0), Hold (1), Sell (2)
    - Price target: Predicted price change percentage
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 20,
        hidden_units: list[int] = [128, 64, 32],
    ):
        """
        Initialize the model architecture.

        Args:
            input_dim: Number of input features
            sequence_length: Number of time steps to look back
            hidden_units: List of hidden layer sizes
        """
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units
        self.model: Optional[keras.Model] = None
        self._build_model()

    def _build_model(self):
        """Build the TensorFlow model."""
        # Input layer
        inputs = keras.Input(shape=(self.sequence_length, self.input_dim))

        # LSTM layers for sequence processing
        x = keras.layers.LSTM(self.hidden_units[0], return_sequences=True)(inputs)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.LSTM(self.hidden_units[1], return_sequences=False)(x)
        x = keras.layers.Dropout(0.2)(x)

        # Dense layers
        x = keras.layers.Dense(self.hidden_units[2], activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)

        # Signal classification head (Buy=0, Hold=1, Sell=2)
        signal_output = keras.layers.Dense(3, activation="softmax", name="signal")(x)

        # Price target regression head (predicted % change)
        price_output = keras.layers.Dense(1, activation="linear", name="price_target")(x)

        # Build model
        self.model = keras.Model(
            inputs=inputs, outputs=[signal_output, price_output], name="signal_model"
        )

        # Compile with appropriate losses
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                "signal": "sparse_categorical_crossentropy",
                "price_target": "mse",
            },
            loss_weights={"signal": 1.0, "price_target": 0.5},
            metrics={"signal": "accuracy", "price_target": "mae"},
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions from the model.

        Args:
            X: Input features of shape (batch, sequence_length, features)

        Returns:
            Tuple of (signal_probs, signal_class, price_target)
            - signal_probs: Softmax probabilities for each class
            - signal_class: Predicted class (0=Buy, 1=Hold, 2=Sell)
            - price_target: Predicted price change percentage
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        signal_probs, price_target = self.model.predict(X, verbose=0)
        signal_class = np.argmax(signal_probs, axis=1)

        return signal_probs, signal_class, price_target.flatten()

    def save(self, path: str):
        """Save model weights to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save_weights(path)

    def load(self, path: str):
        """Load model weights from disk."""
        if self.model is None:
            self._build_model()
        self.model.load_weights(path)

    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


def create_sequences(
    features: np.ndarray, labels: np.ndarray, price_changes: np.ndarray, seq_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM input.

    Args:
        features: Feature array of shape (samples, features)
        labels: Signal labels (0=Buy, 1=Hold, 2=Sell)
        price_changes: Future price change percentages
        seq_length: Number of time steps per sequence

    Returns:
        Tuple of (X_sequences, y_signals, y_prices)
    """
    X, y_signal, y_price = [], [], []

    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length : i])
        y_signal.append(labels[i])
        y_price.append(price_changes[i])

    return np.array(X), np.array(y_signal), np.array(y_price)
