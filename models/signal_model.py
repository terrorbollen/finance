"""TensorFlow model for trading signal classification and price prediction."""

import os

import numpy as np
from tensorflow import keras

from models.losses import sparse_focal_loss


class SignalModel:
    """
    Multi-output model for trading signals.

    Outputs:
    - Signal classification per horizon: Buy (0), Hold (1), Sell (2)
    - Price target: Predicted price change percentage (middle horizon)

    At inference, a consensus vote across horizons determines the final signal:
    2+ horizons must agree on Buy or Sell, otherwise Hold is emitted.
    """

    def __init__(
        self,
        input_dim: int,
        sequence_length: int = 20,
        hidden_units: list[int] | None = None,
        dropout_rate: float = 0.3,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        prediction_horizons: list[int] | None = None,
    ):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units if hidden_units is not None else [64, 32]
        self.dropout_rate = dropout_rate
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.prediction_horizons = prediction_horizons if prediction_horizons is not None else [5, 10, 20]
        self.model: keras.Model | None = None
        self._build_model()

    def _build_model(self):
        """Build the TensorFlow model with one signal head per horizon."""
        inputs = keras.Input(shape=(self.sequence_length, self.input_dim))

        # Shared LSTM backbone
        x = keras.layers.LSTM(self.hidden_units[0], return_sequences=False)(inputs)
        x = keras.layers.Dropout(self.dropout_rate)(x)
        x = keras.layers.Dense(self.hidden_units[1], activation="relu")(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(self.dropout_rate)(x)

        # One signal classification head per horizon
        signal_outputs = []
        for h in self.prediction_horizons:
            head = keras.layers.Dense(3, activation="softmax", name=f"signal_{h}d")(x)
            signal_outputs.append(head)

        # Price target regression head (middle horizon)
        price_output = keras.layers.Dense(1, activation="linear", name="price_target")(x)

        self.model = keras.Model(
            inputs=inputs,
            outputs=signal_outputs + [price_output],
            name="signal_model",
        )

        # Build loss and metric dicts
        if self.use_focal_loss:
            signal_loss = sparse_focal_loss(gamma=self.focal_gamma, alpha=self.focal_alpha)
        else:
            signal_loss = "sparse_categorical_crossentropy"

        losses = {f"signal_{h}d": signal_loss for h in self.prediction_horizons}
        losses["price_target"] = "mse"

        loss_weights = {f"signal_{h}d": 1.0 for h in self.prediction_horizons}
        loss_weights["price_target"] = 0.1

        metrics = {f"signal_{h}d": "accuracy" for h in self.prediction_horizons}
        metrics["price_target"] = "mae"

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics,
        )

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions using majority-vote consensus across horizons.

        Args:
            X: Input features of shape (batch, sequence_length, features)

        Returns:
            Tuple of (signal_probs, signal_class, price_target)
            - signal_probs: Averaged softmax probabilities across horizons (batch, 3)
            - signal_class: Consensus class (0=Buy, 1=Hold, 2=Sell)
            - price_target: Predicted price change percentage
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        outputs = self.model.predict(X, verbose=0)
        # outputs = [signal_5d, signal_10d, ..., price_target]
        signal_probs_list = outputs[:-1]   # list of (batch, 3)
        price_target = outputs[-1]          # (batch, 1)

        batch_size = len(X)

        # Per-horizon argmax votes: (n_horizons, batch)
        votes = np.stack([np.argmax(p, axis=1) for p in signal_probs_list], axis=1)  # (batch, n_horizons)

        # Average probabilities for confidence reporting
        avg_probs = np.mean(np.stack(signal_probs_list, axis=0), axis=0)  # (batch, 3)

        # Majority-vote consensus: need 2+ of 3 horizons to agree on Buy or Sell
        signal_class = np.ones(batch_size, dtype=int)  # default HOLD
        for i in range(batch_size):
            v = votes[i]
            if np.sum(v == 0) >= 2:
                signal_class[i] = 0  # BUY
            elif np.sum(v == 2) >= 2:
                signal_class[i] = 2  # SELL

        return avg_probs, signal_class, price_target.flatten()

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
        assert self.model is not None
        self.model.load_weights(path)

    def summary(self):
        """Print model summary."""
        if self.model:
            self.model.summary()


def create_sequences(
    features: np.ndarray,
    labels_list: list[np.ndarray],
    price_changes: np.ndarray,
    seq_length: int,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Create sequences for LSTM input.

    Args:
        features: Feature array of shape (samples, features)
        labels_list: List of signal label arrays per horizon (each shape (samples,))
        price_changes: Future price change percentages
        seq_length: Number of time steps per sequence

    Returns:
        Tuple of (X_sequences, y_signals_list, y_prices)
    """
    X, y_price = [], []
    y_signals: list[list[int]] = [[] for _ in labels_list]

    for i in range(seq_length, len(features)):
        X.append(features[i - seq_length : i])
        y_price.append(price_changes[i])
        for j, labels in enumerate(labels_list):
            y_signals[j].append(labels[i])

    return (
        np.array(X),
        [np.array(ys) for ys in y_signals],
        np.array(y_price),
    )
