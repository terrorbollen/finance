"""Walk-forward training for adaptive model retraining."""

import json
from dataclasses import dataclass, field
from datetime import date
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
    class_distribution: dict[int, int] = field(default_factory=dict)


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
        sequence_length: int,
        prediction_horizons: list[int],
        buy_threshold: float,
        sell_threshold: float,
        initial_train_days: int = 500,
        validation_days: int = 60,
        step_days: int = 60,
        purge_gap: int | None = None,
        embargo_gap: int | None = None,
    ):
        """
        Initialize walk-forward trainer.

        Args:
            sequence_length: LSTM sequence length — must match the training config.
            prediction_horizons: Days ahead to predict — must match the training config.
            buy_threshold: Price change threshold for BUY label — must match the training config.
            sell_threshold: Price change threshold for SELL label — must match the training config.
            initial_train_days: Number of days for initial training window (~2 years)
            validation_days: Size of validation window (~3 months)
            step_days: How many days to step forward each iteration
            purge_gap: Bars to skip between end of training and start of validation to
                prevent label leakage from overlapping sequences.  Defaults to
                ``max(prediction_horizons)`` (longest horizon label look-ahead).
            embargo_gap: Bars to skip at the very start of each new training window
                where sequences would overlap with the previous validation period
                (combinatorial purged CV embargo concept).  Defaults to
                ``sequence_length`` (typically 20).
        """
        self.initial_train_days = initial_train_days
        self.validation_days = validation_days
        self.step_days = step_days
        self.sequence_length = sequence_length
        self.prediction_horizons = (
            prediction_horizons if prediction_horizons is not None else [5, 10, 20]
        )
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        # Default purge_gap to longest horizon.
        # embargo_gap defaults to 0: for expanding-window walk-forward we want all
        # historical data in every fold. Embargo is appropriate for cross-validation
        # but shrinks training windows to near-zero in walk-forward mode.
        self.purge_gap: int = max(self.prediction_horizons) if purge_gap is None else purge_gap
        self.embargo_gap: int = 0 if embargo_gap is None else embargo_gap

        self.feature_columns: list[str] = []
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def prepare_data(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        """Prepare features and labels from raw price data.

        Args:
            df: Raw OHLCV DataFrame for one ticker.

        Returns:
            Tuple of (features, labels_list, price_changes) where labels_list
            contains one label array per horizon in self.prediction_horizons.
        """
        engineer = FeatureEngineer(df)
        df_features = engineer.add_all_features()

        self.feature_columns = engineer.get_feature_columns()

        # Create labels for each prediction horizon
        for h in self.prediction_horizons:
            future_ret = df_features["close"].shift(-h) / df_features["close"] - 1
            df_features[f"_future_return_{h}d"] = future_ret
            lbl = pd.Series(1, index=df_features.index, dtype=float)  # Hold
            lbl[future_ret.isna()] = float("nan")
            lbl[future_ret > self.buy_threshold] = 0  # Buy
            lbl[future_ret < self.sell_threshold] = 2  # Sell
            df_features[f"_label_{h}d"] = lbl

        # Price target regression uses the middle horizon
        mid_h = self.prediction_horizons[len(self.prediction_horizons) // 2]
        df_features["_future_return"] = df_features[f"_future_return_{mid_h}d"]

        df_features = df_features.dropna()

        features = df_features[self.feature_columns].values
        labels_list: list[np.ndarray] = [
            df_features[f"_label_{h}d"].to_numpy().astype(int) for h in self.prediction_horizons
        ]
        price_changes = df_features["_future_return"].to_numpy() * 100

        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        return features, labels_list, price_changes

    def generate_windows(self, n_samples: int) -> list[tuple[int, int, int, int]]:
        """
        Generate train/validation window indices with purge and embargo gaps.

        The purge gap skips ``purge_gap`` bars between the end of the training
        window and the start of the validation window.  This prevents leakage
        caused by LSTM sequences that overlap the train/val boundary sharing
        the same future-return labels.

        The embargo gap advances the training-window start by ``embargo_gap``
        bars on every fold beyond the first.  This prevents the new training
        window from reusing rows that were part of the previous validation set
        and whose sequences could overlap with it (combinatorial purged CV).

        Args:
            n_samples: Total number of samples

        Returns:
            List of (train_start, train_end, val_start, val_end) tuples where
            val_start already incorporates the purge gap and train_start
            incorporates the embargo gap for folds > 0.
        """
        windows = []

        train_start = 0
        train_end = self.initial_train_days

        while True:
            # Apply purge gap: skip rows between training end and val start
            val_start = train_end + self.purge_gap
            val_end = val_start + self.validation_days

            if val_end > n_samples:
                break

            windows.append((train_start, train_end, val_start, val_end))

            # Step forward: the training window expands to include the just-seen
            # validation data (expanding-window walk-forward).
            prev_val_start = val_start  # keep for embargo calculation
            train_end = val_end

            # Embargo gap: advance train_start so that sequences at the boundary
            # of the previous validation period (which share future-return labels
            # with validation rows) are excluded from the next training fold.
            # The embargo covers ``embargo_gap`` rows starting from prev_val_start.
            # When embargo_gap == 0 the formula collapses to prev_val_start which
            # is <= the existing train_start = 0, so train_start never advances —
            # preserving the original expanding-window behaviour.
            if self.embargo_gap > 0:
                new_train_start = prev_val_start + self.embargo_gap
                train_start = max(train_start, new_train_start)

        return windows

    def train_window(
        self,
        X: np.ndarray,
        y_signal_list: list[np.ndarray],
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
            y_signal_list: List of signal label arrays, one per horizon
            y_price: Full price changes
            train_start, train_end: Training window indices
            val_start, val_end: Validation window indices
            epochs: Training epochs
            batch_size: Batch size

        Returns:
            Tuple of (trained model, validation metrics dict)
        """
        X_train = X[train_start:train_end]
        y_signal_train = [ys[train_start:train_end] for ys in y_signal_list]
        y_price_train = y_price[train_start:train_end]

        X_val = X[val_start:val_end]
        y_signal_val = [ys[val_start:val_end] for ys in y_signal_list]
        y_price_val = y_price[val_start:val_end]

        # Build model matching the prediction_horizons used during data preparation
        input_dim = X.shape[2]
        model = SignalModel(
            input_dim=input_dim,
            sequence_length=self.sequence_length,
            use_focal_loss=True,
            prediction_horizons=self.prediction_horizons,
        )

        # Build named output dicts to match Keras model output names
        y_train_dict = {
            f"signal_{h}d": ys
            for h, ys in zip(self.prediction_horizons, y_signal_train, strict=False)
        }
        y_train_dict["price_target"] = y_price_train
        y_val_dict = {
            f"signal_{h}d": ys
            for h, ys in zip(self.prediction_horizons, y_signal_val, strict=False)
        }
        y_val_dict["price_target"] = y_price_val

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
        model.model.fit(
            X_train,
            y_train_dict,
            validation_data=(X_val, y_val_dict),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        val_results = cast(
            dict[str, float],
            model.model.evaluate(X_val, y_val_dict, verbose=0, return_dict=True),
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
        fetcher = StockDataFetcher(period="max")
        all_features: list[np.ndarray] = []
        all_labels: list[list[np.ndarray]] = []  # [ticker][horizon_idx] -> array
        all_prices: list[np.ndarray] = []

        for ticker in tickers:
            try:
                df = fetcher.fetch(ticker)
                features, labels_list, prices = self.prepare_data(df)
                all_features.append(features)
                all_labels.append(labels_list)
                all_prices.append(prices)
                if verbose:
                    print(f"Loaded {len(features)} samples from {ticker}")
            except Exception as e:
                print(f"Error loading {ticker}: {e}")

        if not all_features:
            raise ValueError("No data loaded")

        # Combine data across tickers
        features = np.vstack(all_features)
        # Concatenate each horizon's labels separately across tickers
        labels_list_combined: list[np.ndarray] = [
            np.concatenate([t[i] for t in all_labels]) for i in range(len(self.prediction_horizons))
        ]
        prices = np.concatenate(all_prices)

        # Normalize using only the first training window to avoid look-ahead bias.
        # Using the full dataset for normalization would leak validation/test statistics
        # into the training normalization constants for every fold.
        train_end_raw = min(self.initial_train_days, len(features))
        self.feature_mean = np.nanmean(features[:train_end_raw], axis=0)
        self.feature_std = np.nanstd(features[:train_end_raw], axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences — pass list of label arrays (one per horizon)
        X, y_signal_list, y_price = create_sequences(
            features, labels_list_combined, prices, self.sequence_length
        )

        if verbose:
            print(f"\nTotal sequences: {len(X)}")
            unique, counts = np.unique(y_signal_list[0], return_counts=True)
            print(f"Class distribution (first horizon): {dict(zip(unique, counts, strict=False))}")

        # Generate windows
        windows = self.generate_windows(len(X))
        if not windows:
            raise ValueError(
                f"Walk-forward produced 0 windows. "
                f"purge_gap={self.purge_gap} + embargo_gap={self.embargo_gap} "
                f"may exceed validation_days={self.validation_days}, "
                f"or n_samples={len(X)} < initial_train_days={self.initial_train_days}. "
                f"Try reducing purge_gap/embargo_gap or increasing the date range."
            )
        if verbose:
            print(f"Generated {len(windows)} walk-forward windows\n")

        # Canonical accuracy key: first horizon
        first_h = self.prediction_horizons[0]
        acc_key = f"signal_{first_h}d_accuracy"

        def _run_walk_forward():
            """Inner function to run walk-forward training."""
            results = []
            best_model = None
            best_accuracy = 0.0

            for i, (train_start, train_end, val_start, val_end) in enumerate(windows):
                if verbose:
                    gap_info = (
                        f" [purge={self.purge_gap}, embargo={self.embargo_gap}]"
                        if self.purge_gap or self.embargo_gap
                        else ""
                    )
                    print(
                        f"Window {i + 1}/{len(windows)}: "
                        f"train[{train_start}:{train_end}] "
                        f"val[{val_start}:{val_end}]"
                        f"{gap_info}"
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
                            y_signal_list,
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
                                "val_accuracy": val_metrics.get(acc_key, float("nan")),
                                "val_loss": val_metrics["loss"],
                            }
                        )
                else:
                    model, val_metrics = self.train_window(
                        X,
                        y_signal_list,
                        y_price,
                        train_start,
                        train_end,
                        val_start,
                        val_end,
                        epochs=epochs,
                        batch_size=batch_size,
                    )

                # Get class distribution in validation set (first horizon)
                val_labels = y_signal_list[0][val_start:val_end]
                unique, counts = np.unique(val_labels, return_counts=True)
                class_dist = {int(k): int(v) for k, v in zip(unique, counts, strict=False)}

                val_acc = val_metrics.get(acc_key, float("nan"))
                window_result = WindowResult(
                    window_id=i + 1,
                    train_start_idx=train_start,
                    train_end_idx=train_end,
                    val_start_idx=val_start,
                    val_end_idx=val_end,
                    train_samples=train_end - train_start,
                    val_samples=val_end - val_start,
                    val_accuracy=val_acc,
                    val_loss=val_metrics["loss"],
                    class_distribution=class_dist,
                )
                results.append(window_result)

                if verbose:
                    print(f"  Validation accuracy ({first_h}d): {window_result.val_accuracy:.4f}")

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
                        "prediction_horizons": str(self.prediction_horizons),
                        "buy_threshold": self.buy_threshold,
                        "sell_threshold": self.sell_threshold,
                        "purge_gap": self.purge_gap,
                        "embargo_gap": self.embargo_gap,
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
                    from models.config import ModelConfig  # noqa: PLC0415

                    cfg_obj = ModelConfig(
                        feature_columns=self.feature_columns,
                        feature_mean=self.feature_mean.tolist(),
                        feature_std=self.feature_std.tolist(),
                        sequence_length=self.sequence_length,
                        input_dim=X.shape[2],
                        training_fetch_date=date.today(),
                        holdout_start_date=date.today(),  # walk-forward uses all data
                        buy_threshold=self.buy_threshold,
                        sell_threshold=self.sell_threshold,
                        prediction_horizons=self.prediction_horizons,
                        tickers=tickers,
                    )
                    cfg_obj.save(config_path)

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
                    "prediction_horizons": self.prediction_horizons,
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
