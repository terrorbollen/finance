"""Walk-forward training for adaptive model retraining."""

import json
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
from tensorflow import keras

from data.features import FeatureEngineer
from data.fetcher import StockDataFetcher
from models.mlflow_tracking import (
    log_hyperparameters,
    log_metrics,
    log_model_artifact,
    setup_mlflow,
    training_run,
)
from models.signal_model import SignalModel, create_sequences


@dataclass
class WindowResult:
    """Results from training on a single walk-forward window."""

    window_id: int
    train_start_idx: int
    train_end_idx: int
    val_start_idx: int
    val_end_idx: int
    train_samples: int
    val_samples: int
    val_accuracy: float
    val_loss: float
    class_distribution: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward training."""

    windows: list[WindowResult] = field(default_factory=list)
    mean_val_accuracy: float = 0.0
    std_val_accuracy: float = 0.0
    best_window_accuracy: float = 0.0
    worst_window_accuracy: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "WALK-FORWARD TRAINING RESULTS",
            "=" * 60,
            f"Total windows: {len(self.windows)}",
            f"Mean validation accuracy: {self.mean_val_accuracy:.4f}",
            f"Std validation accuracy: {self.std_val_accuracy:.4f}",
            f"Best window accuracy: {self.best_window_accuracy:.4f}",
            f"Worst window accuracy: {self.worst_window_accuracy:.4f}",
            "",
            "Per-window results:",
            "-" * 60,
        ]

        for w in self.windows:
            lines.append(
                f"  Window {w.window_id}: acc={w.val_accuracy:.4f}, "
                f"train={w.train_samples}, val={w.val_samples}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


class WalkForwardTrainer:
    """
    Walk-forward training with periodic model retraining.

    Walk-forward validation simulates real trading conditions by:
    1. Training on historical data up to a point
    2. Validating on the next N days
    3. Expanding the training window and repeating

    This helps detect model degradation and adapt to changing market regimes.
    """

    def __init__(
        self,
        initial_train_days: int = 500,
        validation_days: int = 60,
        step_days: int = 60,
        sequence_length: int = 20,
        prediction_horizon: int = 5,
        buy_threshold: float = 0.015,
        sell_threshold: float = -0.015,
    ):
        """
        Initialize walk-forward trainer.

        Args:
            initial_train_days: Number of days for initial training window (~2 years)
            validation_days: Size of validation window (~3 months)
            step_days: How many days to step forward each iteration
            sequence_length: LSTM sequence length
            prediction_horizon: Days ahead to predict
            buy_threshold: Price change threshold for BUY label
            sell_threshold: Price change threshold for SELL label
        """
        self.initial_train_days = initial_train_days
        self.validation_days = validation_days
        self.step_days = step_days
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self.feature_columns: list[str] = []
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def prepare_data(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and labels from raw price data."""
        engineer = FeatureEngineer(df)
        df_features = engineer.add_all_features()

        self.feature_columns = engineer.get_feature_columns()

        # Calculate future returns
        df_features["future_return"] = (
            df_features["close"].shift(-self.prediction_horizon) / df_features["close"] - 1
        )

        # Create labels
        df_features["label"] = 1  # Default: Hold
        df_features.loc[df_features["future_return"] > self.buy_threshold, "label"] = 0
        df_features.loc[df_features["future_return"] < self.sell_threshold, "label"] = 2

        df_features = df_features.dropna()

        features = df_features[self.feature_columns].values
        labels = df_features["label"].to_numpy().astype(int)
        price_changes = df_features["future_return"].to_numpy() * 100

        # Handle inf/nan
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        return features, labels, price_changes

    def generate_windows(self, n_samples: int) -> list[tuple[int, int, int, int]]:
        """
        Generate train/validation window indices.

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_start, train_end, val_start, val_end) tuples
        """
        windows = []

        # Account for sequence length at the start
        train_start = 0
        train_end = self.initial_train_days

        while train_end + self.validation_days <= n_samples:
            val_start = train_end
            val_end = val_start + self.validation_days

            windows.append((train_start, train_end, val_start, val_end))

            # Step forward (expanding window - include previous validation data)
            train_end = val_end

        return windows

    def train_window(
        self,
        X: np.ndarray,
        y_signal: np.ndarray,
        y_price: np.ndarray,
        train_start: int,
        train_end: int,
        val_start: int,
        val_end: int,
        epochs: int = 30,
        batch_size: int = 32,
    ) -> tuple[SignalModel, dict]:
        """
        Train model on a single window.

        Args:
            X: Full sequence array
            y_signal: Full signal labels
            y_price: Full price changes
            train_start, train_end: Training window indices
            val_start, val_end: Validation window indices
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Tuple of (trained model, validation metrics)
        """
        X_train = X[train_start:train_end]
        y_signal_train = y_signal[train_start:train_end]
        y_price_train = y_price[train_start:train_end]

        X_val = X[val_start:val_end]
        y_signal_val = y_signal[val_start:val_end]
        y_price_val = y_price[val_start:val_end]

        # Build model (with focal loss)
        input_dim = X.shape[2]
        model = SignalModel(
            input_dim=input_dim,
            sequence_length=self.sequence_length,
            use_focal_loss=True,
        )

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, patience=3, min_lr=1e-6, monitor="val_loss"
            ),
        ]

        assert model.model is not None
        # Train
        model.model.fit(
            X_train,
            [y_signal_train, y_price_train],
            validation_data=(X_val, [y_signal_val, y_price_val]),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        # Evaluate
        val_results = cast(
            dict[str, float],
            model.model.evaluate(X_val, [y_signal_val, y_price_val], verbose=0, return_dict=True),
        )

        return model, val_results

    def run(
        self,
        tickers: list[str],
        epochs: int = 30,
        batch_size: int = 32,
        model_path: str = "checkpoints/signal_model.weights.h5",
        verbose: bool = True,
        track_with_mlflow: bool = True,
        tags: dict[str, str] | None = None,
    ) -> WalkForwardResult:
        """
        Run walk-forward training on given tickers.

        Args:
            tickers: List of ticker symbols
            epochs: Training epochs per window
            batch_size: Batch size
            model_path: Path to save final model
            verbose: Print progress
            track_with_mlflow: Whether to track experiment with MLflow

        Returns:
            WalkForwardResult with all window metrics
        """
        # Setup MLflow tracking if enabled
        if track_with_mlflow:
            setup_mlflow()
        # Fetch and combine data
        fetcher = StockDataFetcher(period="5y")
        all_features, all_labels, all_prices = [], [], []

        for ticker in tickers:
            try:
                df = fetcher.fetch(ticker)
                features, labels, prices = self.prepare_data(df)
                all_features.append(features)
                all_labels.append(labels)
                all_prices.append(prices)
                if verbose:
                    print(f"Loaded {len(features)} samples from {ticker}")
            except Exception as e:
                print(f"Error loading {ticker}: {e}")

        if not all_features:
            raise ValueError("No data loaded")

        # Combine and normalize
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        prices = np.concatenate(all_prices)

        self.feature_mean = np.nanmean(features, axis=0)
        self.feature_std = np.nanstd(features, axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences
        X, y_signal, y_price = create_sequences(features, labels, prices, self.sequence_length)

        if verbose:
            print(f"\nTotal sequences: {len(X)}")
            unique, counts = np.unique(y_signal, return_counts=True)
            print(f"Class distribution: {dict(zip(unique, counts, strict=False))}")

        # Generate windows
        windows = self.generate_windows(len(X))
        if verbose:
            print(f"Generated {len(windows)} walk-forward windows\n")

        def _run_walk_forward():
            """Inner function to run walk-forward training."""
            results = []
            best_model = None
            best_accuracy = 0.0

            for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
                if verbose:
                    print(
                        f"Window {i + 1}/{len(windows)}: "
                        f"train[{train_start}:{train_end}] "
                        f"val[{val_start}:{val_end}]"
                    )

                # Train on window (optionally wrapped in nested MLflow run)
                if track_with_mlflow:
                    with training_run(run_name=f"window-{i + 1}", nested=True):
                        log_hyperparameters(
                            {
                                "window_id": i + 1,
                                "train_start": train_start,
                                "train_end": train_end,
                                "val_start": val_start,
                                "val_end": val_end,
                                "train_samples": train_end - train_start,
                                "val_samples": val_end - val_start,
                            }
                        )

                        model, val_metrics = self.train_window(
                            X,
                            y_signal,
                            y_price,
                            train_start,
                            train_end,
                            val_start,
                            val_end,
                            epochs=epochs,
                            batch_size=batch_size,
                        )

                        log_metrics(
                            {
                                "val_accuracy": val_metrics["signal_accuracy"],
                                "val_loss": val_metrics["loss"],
                            }
                        )
                else:
                    model, val_metrics = self.train_window(
                        X,
                        y_signal,
                        y_price,
                        train_start,
                        train_end,
                        val_start,
                        val_end,
                        epochs=epochs,
                        batch_size=batch_size,
                    )

                # Get class distribution in validation set
                val_labels = y_signal[val_start:val_end]
                unique, counts = np.unique(val_labels, return_counts=True)
                class_dist = {int(k): int(v) for k, v in zip(unique, counts, strict=False)}

                window_result = WindowResult(
                    window_id=i + 1,
                    train_start_idx=train_start,
                    train_end_idx=train_end,
                    val_start_idx=val_start,
                    val_end_idx=val_end,
                    train_samples=train_end - train_start,
                    val_samples=val_end - val_start,
                    val_accuracy=val_metrics["signal_accuracy"],
                    val_loss=val_metrics["loss"],
                    class_distribution=class_dist,
                )
                results.append(window_result)

                if verbose:
                    print(f"  Validation accuracy: {window_result.val_accuracy:.4f}")

                # Keep best model
                if window_result.val_accuracy > best_accuracy:
                    best_accuracy = window_result.val_accuracy
                    best_model = model

            return results, best_model

        # Run walk-forward training with or without MLflow tracking
        if track_with_mlflow:
            run_name = f"walk-forward-{'-'.join(tickers[:3])}"
            if len(tickers) > 3:
                run_name += f"-+{len(tickers) - 3}"

            run_tags = {"tickers": ",".join(tickers), "run_type": "walk-forward"}
            if tags:
                run_tags.update(tags)

            with training_run(run_name=run_name, tags=run_tags):
                # Log hyperparameters
                log_hyperparameters(
                    {
                        "initial_train_days": self.initial_train_days,
                        "validation_days": self.validation_days,
                        "step_days": self.step_days,
                        "sequence_length": self.sequence_length,
                        "prediction_horizon": self.prediction_horizon,
                        "buy_threshold": self.buy_threshold,
                        "sell_threshold": self.sell_threshold,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "num_tickers": len(tickers),
                        "num_windows": len(windows),
                        "total_sequences": len(X),
                    }
                )

                results, best_model = _run_walk_forward()

                # Save best model and log artifacts
                if best_model is not None:
                    best_model.save(model_path)

                    config_path = model_path.replace(".weights.h5", "_config.json")
                    if self.feature_mean is None or self.feature_std is None:
                        raise RuntimeError("feature_mean/std not set; call prepare_data first")
                    config = {
                        "feature_columns": self.feature_columns,
                        "feature_mean": self.feature_mean.tolist(),
                        "feature_std": self.feature_std.tolist(),
                        "sequence_length": self.sequence_length,
                        "input_dim": X.shape[2],
                        "walk_forward": True,
                        "num_windows": len(windows),
                    }
                    with open(config_path, "w") as f:
                        json.dump(config, f)

                    if verbose:
                        print(f"\nSaved best model to {model_path}")

                    # Log model artifacts
                    log_model_artifact(model_path, artifact_path="model")
                    log_model_artifact(config_path, artifact_path="model")

                # Log aggregated metrics
                accuracies = [r.val_accuracy for r in results]
                log_metrics(
                    {
                        "mean_val_accuracy": float(np.mean(accuracies)),
                        "std_val_accuracy": float(np.std(accuracies)),
                        "best_window_accuracy": float(np.max(accuracies)),
                        "worst_window_accuracy": float(np.min(accuracies)),
                    }
                )
        else:
            results, best_model = _run_walk_forward()

            # Save best model
            if best_model is not None:
                best_model.save(model_path)

                if self.feature_mean is None or self.feature_std is None:
                    raise RuntimeError("feature_mean/std not set; call prepare_data first")
                config_path = model_path.replace(".weights.h5", "_config.json")
                config = {
                    "feature_columns": self.feature_columns,
                    "feature_mean": self.feature_mean.tolist(),
                    "feature_std": self.feature_std.tolist(),
                    "sequence_length": self.sequence_length,
                    "input_dim": X.shape[2],
                    "walk_forward": True,
                    "num_windows": len(windows),
                }
                with open(config_path, "w") as f:
                    json.dump(config, f)

                if verbose:
                    print(f"\nSaved best model to {model_path}")

        # Aggregate results
        accuracies = [r.val_accuracy for r in results]
        wf_result = WalkForwardResult(
            windows=results,
            mean_val_accuracy=float(np.mean(accuracies)),
            std_val_accuracy=float(np.std(accuracies)),
            best_window_accuracy=float(np.max(accuracies)),
            worst_window_accuracy=float(np.min(accuracies)),
        )

        return wf_result
