"""TensorFlow model for trading signal classification and price prediction."""

import os

import numpy as np
from tensorflow import keras

from models.direction import BUY_IDX, HOLD_IDX, SELL_IDX
from models.losses import balanced_focal_loss, sparse_focal_loss


def _apply_majority_vote(
    signal_probs_list: list[np.ndarray],
    horizon_classes: list[np.ndarray],
    price_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Majority-vote consensus across horizon heads.

    Returns (avg_probs, signal_class, price_target) matching the SignalModel.predict()
    contract.  Used by both SignalModel and EnsembleSignalModel so the logic lives in
    one place.
    """
    batch_size = len(price_target)
    votes = np.stack(horizon_classes, axis=1)  # (batch, n_horizons)
    avg_probs = np.mean(np.stack(signal_probs_list, axis=0), axis=0)  # (batch, 3)
    n_horizons = len(signal_probs_list)
    majority = n_horizons // 2 + 1
    signal_class = np.full(batch_size, HOLD_IDX, dtype=int)
    for i in range(batch_size):
        v = votes[i]
        if np.sum(v == BUY_IDX) >= majority:
            signal_class[i] = BUY_IDX
        elif np.sum(v == SELL_IDX) >= majority:
            signal_class[i] = SELL_IDX
    return avg_probs, signal_class, price_target


class SignalModel:
    """
    Multi-output model for trading signals.

    Outputs:
    - Signal classification per horizon: Buy (0), Hold (1), Sell (2)
    - Price target: Predicted price change percentage (middle horizon)

    At inference, a consensus vote across horizons determines the final signal:
    2+ horizons must agree on Buy or Sell, otherwise Hold is emitted.

    The ``backbone`` parameter selects the recurrent layer:
    - ``"lstm"`` (default): LSTM with multi-head attention and GAP
    - ``"gru"``: GRU with GAP (no attention) — complementary to LSTM in an ensemble
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
        class_weights: list[float] | None = None,
        prediction_horizons: list[int] | None = None,
        backbone: str = "lstm",
    ):
        if backbone not in ("lstm", "gru"):
            raise ValueError(f"backbone must be 'lstm' or 'gru', got {backbone!r}")
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_units = hidden_units if hidden_units is not None else [64, 32]
        self.dropout_rate = dropout_rate
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.class_weights = class_weights
        self.prediction_horizons = (
            prediction_horizons if prediction_horizons is not None else [5, 10, 20]
        )
        self.backbone = backbone
        self.model: keras.Model | None = None
        self._build_model()

    def _build_model(self):
        """Build the TensorFlow model with one signal head per horizon."""
        inputs = keras.Input(shape=(self.sequence_length, self.input_dim))

        # Recurrent backbone — LSTM uses attention for weighted sequence aggregation;
        # GRU uses plain GAP, making the two architectures genuinely complementary.
        if self.backbone == "lstm":
            x = keras.layers.LSTM(self.hidden_units[0], return_sequences=True)(inputs)
            x = keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=self.hidden_units[0] // 4
            )(x, x)
            x = keras.layers.GlobalAveragePooling1D()(x)
        else:  # gru
            x = keras.layers.GRU(self.hidden_units[0], return_sequences=True)(inputs)
            x = keras.layers.GlobalAveragePooling1D()(x)

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

        # Build loss and metric dicts.
        # If class_weights are provided, use balanced_focal_loss which applies
        # per-class alpha weighting (down-weights the dominant HOLD class).
        # Falls back to scalar-alpha focal loss when weights aren't available.
        if self.use_focal_loss:
            if self.class_weights is not None:
                signal_loss = balanced_focal_loss(
                    gamma=self.focal_gamma, class_weights=self.class_weights
                )
            else:
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

    def predict_per_horizon(
        self, X: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """
        Generate per-horizon predictions from each classification head.

        Args:
            X: Input features of shape (batch, sequence_length, features)

        Returns:
            Tuple of (horizon_probs, horizon_classes, price_target)
            - horizon_probs: List of softmax probability arrays, one per horizon,
              each of shape (batch, 3). Ordered to match self.prediction_horizons.
            - horizon_classes: List of predicted class arrays, one per horizon,
              each of shape (batch,). Values: 0=Buy, 1=Hold, 2=Sell.
            - price_target: Predicted price change percentage, shape (batch,)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded")

        outputs = self.model.predict(X, verbose=0)
        # outputs = [signal_5d, signal_10d, ..., price_target]
        signal_probs_list: list[np.ndarray] = outputs[:-1]  # list of (batch, 3)
        price_target: np.ndarray = outputs[-1]  # (batch, 1)

        horizon_classes = [np.argmax(p, axis=1) for p in signal_probs_list]

        return signal_probs_list, horizon_classes, price_target.flatten()

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate consensus predictions by majority-voting across all horizon heads.

        Calls predict_per_horizon() internally and aggregates results. Use
        predict_per_horizon() directly when per-horizon signal quality matters
        (e.g. in the backtester via B10).

        Args:
            X: Input features of shape (batch, sequence_length, features)

        Returns:
            Tuple of (signal_probs, signal_class, price_target)
            - signal_probs: Averaged softmax probabilities across horizons (batch, 3)
            - signal_class: Consensus class (0=Buy, 1=Hold, 2=Sell)
            - price_target: Predicted price change percentage, shape (batch,)
        """
        signal_probs_list, horizon_classes, price_target = self.predict_per_horizon(X)
        return _apply_majority_vote(signal_probs_list, horizon_classes, price_target)

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


class EnsembleSignalModel:
    """Ensemble of an LSTM-backbone and a GRU-backbone SignalModel.

    Predictions are formed by averaging per-horizon softmax probabilities from
    both models before applying the majority-vote consensus.  Each model is
    trained and saved independently; the ensemble is assembled at inference time.

    The public interface (``predict``, ``predict_per_horizon``, ``save``, ``load``,
    ``summary``) is identical to ``SignalModel`` so all callers work unchanged.
    """

    def __init__(self, lstm_model: SignalModel, gru_model: SignalModel) -> None:
        self.lstm_model = lstm_model
        self.gru_model = gru_model
        self.prediction_horizons = lstm_model.prediction_horizons

    def predict_per_horizon(
        self, X: np.ndarray
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
        """Average per-horizon probabilities from both backbones."""
        lstm_probs, _, lstm_price = self.lstm_model.predict_per_horizon(X)
        gru_probs, _, gru_price = self.gru_model.predict_per_horizon(X)
        avg_probs = [(lp + gp) / 2 for lp, gp in zip(lstm_probs, gru_probs, strict=True)]
        avg_classes = [np.argmax(p, axis=1) for p in avg_probs]
        return avg_probs, avg_classes, (lstm_price + gru_price) / 2

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Consensus signal from the averaged ensemble probabilities."""
        signal_probs_list, horizon_classes, price_target = self.predict_per_horizon(X)
        return _apply_majority_vote(signal_probs_list, horizon_classes, price_target)

    def save(self, path: str) -> None:
        """Save LSTM weights to ``path`` and GRU weights to the ``_gru`` variant."""
        self.lstm_model.save(path)
        self.gru_model.save(path.replace(".weights.h5", "_gru.weights.h5"))

    def load(self, path: str) -> None:
        """Load both model weights from the paths produced by ``save``."""
        self.lstm_model.load(path)
        self.gru_model.load(path.replace(".weights.h5", "_gru.weights.h5"))

    def summary(self) -> None:
        print("=== LSTM backbone ===")
        self.lstm_model.summary()
        print("=== GRU backbone ===")
        self.gru_model.summary()


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
