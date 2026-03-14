"""Training pipeline for the signal model."""

import json
import numpy as np
import pandas as pd
from tensorflow import keras
from typing import Optional
import os

from models.signal_model import SignalModel, create_sequences
from data.fetcher import StockDataFetcher
from data.features import FeatureEngineer


class ModelTrainer:
    """Handles model training with time-series aware data splitting."""

    def __init__(
        self,
        sequence_length: int = 20,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        prediction_horizon: int = 5,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
    ):
        """
        Initialize the trainer.

        Args:
            sequence_length: Number of time steps for LSTM input
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation (rest is test)
            prediction_horizon: Days ahead to predict for price target
            buy_threshold: Min % gain to label as Buy
            sell_threshold: Max % loss to label as Sell
        """
        self.sequence_length = sequence_length
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.model: Optional[SignalModel] = None
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

        # Create labels: 0=Buy, 1=Hold, 2=Sell
        df_features["label"] = 1  # Default: Hold
        df_features.loc[df_features["future_return"] > self.buy_threshold, "label"] = 0
        df_features.loc[
            df_features["future_return"] < self.sell_threshold, "label"
        ] = 2

        # Drop rows with NaN
        df_features = df_features.dropna()

        # Extract arrays
        features = df_features[self.feature_columns].values
        labels = df_features["label"].values.astype(int)
        price_changes = (df_features["future_return"].values * 100)  # As percentage

        return features, labels, price_changes

    def train(
        self,
        tickers: list[str],
        epochs: int = 50,
        batch_size: int = 32,
        model_path: str = "checkpoints/signal_model.weights.h5",
    ) -> dict:
        """
        Train the model on given tickers.

        Args:
            tickers: List of ticker symbols to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_path: Path to save best model weights

        Returns:
            Dictionary with training history and metrics
        """
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

        # Normalize features
        self.feature_mean = features.mean(axis=0)
        self.feature_std = features.std(axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std

        # Create sequences
        X, y_signal, y_price = create_sequences(
            features, labels, prices, self.sequence_length
        )

        # Time-series split (no shuffling to preserve temporal order)
        n = len(X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        X_train, y_signal_train, y_price_train = (
            X[:train_end],
            y_signal[:train_end],
            y_price[:train_end],
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
        self.model = SignalModel(
            input_dim=input_dim, sequence_length=self.sequence_length
        )

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

        # Train
        history = self.model.model.fit(
            X_train,
            {"signal": y_signal_train, "price_target": y_price_train},
            validation_data=(
                X_val,
                {"signal": y_signal_val, "price_target": y_price_val},
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        # Evaluate on test set
        test_results = self.model.model.evaluate(
            X_test,
            {"signal": y_signal_test, "price_target": y_price_test},
            verbose=0,
            return_dict=True,
        )

        print(f"\nTest Results:")
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

        return {
            "history": history.history,
            "test_loss": test_results["loss"],
            "test_signal_accuracy": test_results["signal_accuracy"],
            "test_price_mae": test_results["price_target_mae"],
        }
