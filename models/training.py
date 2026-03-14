"""Training pipeline for the signal model."""

import json
import os

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from data.features import FeatureEngineer
from data.fetcher import StockDataFetcher
from models.mlflow_tracking import (
    log_hyperparameters,
    log_metrics,
    log_model_artifact,
    log_training_history,
    setup_mlflow,
    training_run,
)
from models.signal_model import SignalModel, create_sequences


class ModelTrainer:
    """Handles model training with time-series aware data splitting."""

    def __init__(
        self,
        sequence_length: int = 20,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        prediction_horizon: int = 5,
        buy_threshold: float = 0.015,  # Lowered from 0.02 to get more BUY signals
        sell_threshold: float = -0.015,  # Raised from -0.02 to get more SELL signals
        use_adaptive_thresholds: bool = True,  # Adapt thresholds to volatility
    ):
        """
        Initialize the trainer.

        Args:
            sequence_length: Number of time steps for LSTM input
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation (rest is test)
            prediction_horizon: Days ahead to predict for price target
            buy_threshold: Min % gain to label as Buy (default lowered for balance)
            sell_threshold: Max % loss to label as Sell (default raised for balance)
            use_adaptive_thresholds: If True, adjust thresholds based on stock volatility
        """
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.model: SignalModel | None = None
        self.feature_columns: list[str] = []

    def prepare_data(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and labels from raw price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            Tuple of (features, labels, price_changes)
        """
        # Add technical indicators
        engineer = FeatureEngineer(df)
        df_features = engineer.add_all_features()

        # Store feature columns BEFORE adding labels
        self.feature_columns = engineer.get_feature_columns()

        # Calculate future returns for labels
        df_features["future_return"] = (
            df_features["close"].shift(-self.prediction_horizon)
            / df_features["close"]
            - 1
        )

        # Determine thresholds
        if self.use_adaptive_thresholds:
            # Adaptive thresholds based on stock's volatility
            # Use rolling std of returns as volatility measure
            volatility = df_features["close"].pct_change().rolling(20).std().median()
            # Scale thresholds by volatility (typical volatility ~0.01-0.02)
            scale = max(volatility / 0.015, 0.5)  # Don't go below 0.5x
            buy_thresh = self.buy_threshold * scale
            sell_thresh = self.sell_threshold * scale
        else:
            buy_thresh = self.buy_threshold
            sell_thresh = self.sell_threshold

        # Create labels: 0=Buy, 1=Hold, 2=Sell
        df_features["label"] = 1  # Default: Hold
        df_features.loc[df_features["future_return"] > buy_thresh, "label"] = 0
        df_features.loc[df_features["future_return"] < sell_thresh, "label"] = 2

        # Drop rows with NaN
        df_features = df_features.dropna()

        # Extract arrays
        features = df_features[self.feature_columns].values
        labels = df_features["label"].to_numpy().astype(int)
        price_changes = df_features["future_return"].to_numpy() * 100  # As percentage

        # Replace any inf values with large finite values, then clip
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        return features, labels, price_changes

    def train(
        self,
        tickers: list[str],
        epochs: int = 50,
        batch_size: int = 32,
        model_path: str = "checkpoints/signal_model.weights.h5",
        use_focal_loss: bool = True,
        track_with_mlflow: bool = True,
    ) -> dict:
        """
        Train the model on given tickers.

        Args:
            tickers: List of ticker symbols to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_path: Path to save best model weights
            track_with_mlflow: Whether to track experiment with MLflow

        Returns:
            Dictionary with training history and metrics
        """
        # Setup MLflow tracking if enabled
        if track_with_mlflow:
            setup_mlflow(experiment_name="signal-model-training")

        # Fetch and combine data from all tickers
        fetcher = StockDataFetcher(period="5y")
        all_features, all_labels, all_prices = [], [], []

        for ticker in tickers:
            try:
                df = fetcher.fetch(ticker)
                features, labels, prices = self.prepare_data(df)
                all_features.append(features)
                all_labels.append(labels)
                all_prices.append(prices)
                print(f"Loaded {len(features)} samples from {ticker}")
            except Exception as e:
                print(f"Error loading {ticker}: {e}")

        if not all_features:
            raise ValueError("No data loaded for training")

        # Combine all data
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        prices = np.concatenate(all_prices)

        # Print class distribution BEFORE training
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        print("\n" + "=" * 50)
        print("CLASS DISTRIBUTION (before sequences):")
        print("=" * 50)
        label_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
        for label, count in zip(unique, counts, strict=False):
            pct = count / total * 100
            print(f"  {label_names.get(label, label)}: {count:,} samples ({pct:.1f}%)")
        print("=" * 50 + "\n")

        # Normalize features
        self.feature_mean = np.nanmean(features, axis=0)
        self.feature_std = np.nanstd(features, axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std

        # Replace any remaining inf/nan with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences
        X, y_signal, y_price = create_sequences(
            features, labels, prices, self.sequence_length
        )

        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1, 2]),
            y=y_signal
        )
        class_weight_dict = dict(enumerate(class_weights))
        print("Class weights (to balance training):")
        for label, weight in class_weight_dict.items():
            print(f"  {label_names.get(label, label)}: {weight:.3f}")
        print()

        # Convert class weights to sample weights (for multi-output model compatibility)
        sample_weights = np.array([class_weight_dict[label] for label in y_signal])

        # Time-series split (no shuffling to preserve temporal order)
        n = len(X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        X_train, y_signal_train, y_price_train, sw_train = (
            X[:train_end],
            y_signal[:train_end],
            y_price[:train_end],
            sample_weights[:train_end],
        )
        X_val, y_signal_val, y_price_val = (
            X[train_end:val_end],
            y_signal[train_end:val_end],
            y_price[train_end:val_end],
        )
        X_test, y_signal_test, y_price_test = (
            X[val_end:],
            y_signal[val_end:],
            y_price[val_end:],
        )

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Build model
        input_dim = X.shape[2]

        def _do_training():
            """Inner function to run training (optionally wrapped in MLflow context)."""
            self.model = SignalModel(
                input_dim=input_dim,
                sequence_length=self.sequence_length,
                use_focal_loss=use_focal_loss,
            )

            if use_focal_loss:
                print("Using Focal Loss (handles class imbalance internally)")

            # Callbacks
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            callbacks = [
                keras.callbacks.ModelCheckpoint(
                    model_path, save_best_only=True, save_weights_only=True, monitor="val_loss"
                ),
                keras.callbacks.EarlyStopping(
                    patience=10, restore_best_weights=True, monitor="val_loss"
                ),
                keras.callbacks.ReduceLROnPlateau(
                    factor=0.5, patience=5, min_lr=1e-6, monitor="val_loss"
                ),
            ]

            # Train model
            # When using focal loss, don't use sample weights (focal loss handles imbalance)
            # Otherwise, use sample weights for class balancing
            fit_kwargs = {
                "x": X_train,
                "y": [y_signal_train, y_price_train],
                "validation_data": (X_val, [y_signal_val, y_price_val]),
                "epochs": epochs,
                "batch_size": batch_size,
                "callbacks": callbacks,
                "verbose": 1,
            }

            if not use_focal_loss:
                # Only use sample weights with standard cross-entropy
                fit_kwargs["sample_weight"] = [sw_train, np.ones_like(sw_train)]

            assert self.model.model is not None
            history = self.model.model.fit(**fit_kwargs)

            # Evaluate on test set
            test_results: dict[str, float] = self.model.model.evaluate(
                X_test,
                [y_signal_test, y_price_test],
                verbose=0,
                return_dict=True,
            )

            print("\nTest Results:")
            print(f"  Signal Accuracy: {test_results['signal_accuracy']:.4f}")
            print(f"  Price MAE: {test_results['price_target_mae']:.4f}%")

            # Save training config for inference
            config_path = model_path.replace(".weights.h5", "_config.json")
            config = {
                "feature_columns": self.feature_columns,
                "feature_mean": self.feature_mean.tolist(),
                "feature_std": self.feature_std.tolist(),
                "sequence_length": self.sequence_length,
                "input_dim": input_dim,
            }
            with open(config_path, "w") as f:
                json.dump(config, f)
            print(f"Saved model config to {config_path}")

            return history, test_results, config_path

        # Run training with or without MLflow tracking
        if track_with_mlflow:
            run_name = f"train-{'-'.join(tickers[:3])}"
            if len(tickers) > 3:
                run_name += f"-+{len(tickers)-3}"

            with training_run(run_name=run_name, tags={"tickers": ",".join(tickers)}):
                # Log hyperparameters
                log_hyperparameters({
                    "sequence_length": self.sequence_length,
                    "train_ratio": self.train_ratio,
                    "val_ratio": self.val_ratio,
                    "prediction_horizon": self.prediction_horizon,
                    "buy_threshold": self.buy_threshold,
                    "sell_threshold": self.sell_threshold,
                    "use_adaptive_thresholds": self.use_adaptive_thresholds,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "use_focal_loss": use_focal_loss,
                    "num_tickers": len(tickers),
                    "train_samples": len(X_train),
                    "val_samples": len(X_val),
                    "test_samples": len(X_test),
                    "input_dim": input_dim,
                })

                history, test_results, config_path = _do_training()

                # Log training history per epoch
                log_training_history(history.history)

                # Log final test metrics
                log_metrics({
                    "test_loss": test_results["loss"],
                    "test_signal_accuracy": test_results["signal_accuracy"],
                    "test_price_mae": test_results["price_target_mae"],
                })

                # Log model artifacts
                log_model_artifact(model_path, artifact_path="model")
                log_model_artifact(config_path, artifact_path="model")
        else:
            history, test_results, _ = _do_training()

        return {
            "history": history.history,
            "test_loss": test_results["loss"],
            "test_signal_accuracy": test_results["signal_accuracy"],
            "test_price_mae": test_results["price_target_mae"],
        }
