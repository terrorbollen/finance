"""Training pipeline for the signal model."""

import os
from datetime import date

import mlflow
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from data.features import FeatureEngineer
from data.fetcher import StockDataFetcher
from models.config import ModelConfig
from models.mlflow_tracking import (
    log_hyperparameters,
    log_metrics,
    log_model_artifact,
    setup_mlflow,
    training_run,
)
from models.signal_model import SignalModel, create_sequences


class ModelTrainer:
    """Handles model training with time-series aware data splitting."""

    # Bars per trading day and data fetch period per interval.
    # Stockholm exchange trades ~7 hours/day on Yahoo Finance hourly data.
    _INTERVAL_BARS_PER_DAY: dict[str, int] = {"1d": 1, "1h": 7}
    _INTERVAL_PERIOD: dict[str, str] = {"1d": "5y", "1h": "2y"}

    def __init__(
        self,
        sequence_length: int | None = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        prediction_horizon: int | None = None,
        buy_threshold: float = 0.015,
        sell_threshold: float = -0.015,
        use_adaptive_thresholds: bool = True,
        interval: str = "1d",
    ):
        """
        Initialize the trainer.

        Args:
            sequence_length: LSTM lookback window (bars). Defaults to 20 days worth of bars.
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation (rest is test)
            prediction_horizon: Bars ahead to predict. Defaults to 5 days worth of bars.
            buy_threshold: Min % gain to label as Buy
            sell_threshold: Max % loss to label as Sell
            use_adaptive_thresholds: If True, adjust thresholds based on stock volatility
            interval: Data interval — "1d" (daily) or "1h" (hourly)
        """
        self.interval = interval
        bars = self._INTERVAL_BARS_PER_DAY.get(interval, 1)
        self.sequence_length = sequence_length if sequence_length is not None else 20 * bars
        self.prediction_horizon = prediction_horizon if prediction_horizon is not None else 5 * bars
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.model: SignalModel | None = None
        self.feature_columns: list[str] = []
        # Set by prepare_data(); declared here so the object's full shape is visible
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        reference_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DatetimeIndex]:
        """
        Prepare features and labels from raw price data.

        Args:
            df: DataFrame with OHLCV data
            reference_data: Optional cross-asset reference data for F2 features.

        Returns:
            Tuple of (features, labels, price_changes, date_index)
        """
        # Add technical indicators
        engineer = FeatureEngineer(df, reference_data=reference_data)
        df_features = engineer.add_all_features()

        # Store feature columns BEFORE adding labels
        self.feature_columns = engineer.get_feature_columns()

        # Calculate future returns for labels
        df_features["future_return"] = (
            df_features["close"].shift(-self.prediction_horizon) / df_features["close"] - 1
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
        date_index = pd.DatetimeIndex(df_features.index)

        # Replace any inf values with large finite values, then clip
        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        return features, labels, price_changes, date_index

    def train(
        self,
        tickers: list[str],
        epochs: int = 50,
        batch_size: int = 32,
        model_path: str | None = None,
        use_focal_loss: bool = True,
        track_with_mlflow: bool = True,
        tags: dict[str, str] | None = None,
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
        # Resolve model path based on interval if not explicitly provided
        suffix = "" if self.interval == "1d" else f"_{self.interval}"
        if model_path is None:
            model_path = f"checkpoints/signal_model{suffix}.weights.h5"

        # Setup MLflow tracking if enabled
        if track_with_mlflow:
            setup_mlflow()

        # Fetch and combine data from all tickers
        period = self._INTERVAL_PERIOD.get(self.interval, "5y")
        fetcher = StockDataFetcher(period=period, interval=self.interval)
        all_features, all_labels, all_prices, all_dates = [], [], [], []

        loaded_tickers: list[str] = []
        for ticker in tickers:
            try:
                df = fetcher.fetch(ticker)
                ref_data = fetcher.fetch_cross_asset_data(pd.DatetimeIndex(df.index))
                features, labels, prices, dates = self.prepare_data(df, reference_data=ref_data)
                if len(features) == 0:
                    print(f"Skipping {ticker}: no usable samples")
                    continue
                all_features.append(features)
                all_labels.append(labels)
                all_prices.append(prices)
                all_dates.append(dates)
                loaded_tickers.append(ticker)
                print(f"Loaded {len(features)} samples from {ticker}")
            except Exception as e:
                print(f"Error loading {ticker}: {e}")

        if not all_features:
            raise ValueError("No data loaded for training")

        # Combine all data
        print(f"\nCombining data from {len(all_features)} tickers...")
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)
        prices = np.concatenate(all_prices)
        print(f"Total samples: {len(features):,}  |  Features: {features.shape[1]}")

        # Compute the actual holdout start: the latest calendar date that appears
        # in any ticker's training or validation split.  We look at each ticker's
        # date index independently so that the split index maps cleanly to a date.
        latest_val_date = max(
            dates[min(int(len(dates) * (self.train_ratio + self.val_ratio)), len(dates) - 1)]
            for dates in all_dates
        )

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

        # Normalize features using training data only to avoid look-ahead bias.
        # Compute the raw-feature index where training ends so we don't include
        # val/test statistics in the mean/std.
        print("Normalizing features...")
        n_sequences = len(features) - self.sequence_length + 1
        train_end_raw = int(n_sequences * self.train_ratio) + self.sequence_length - 1
        train_end_raw = min(train_end_raw, len(features))
        self.feature_mean = np.nanmean(features[:train_end_raw], axis=0)
        self.feature_std = np.nanstd(features[:train_end_raw], axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std

        # Replace any remaining inf/nan with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences
        print(f"Creating sequences (length={self.sequence_length}, ~{n_sequences:,} sequences)...")
        X, y_signal, y_price = create_sequences(features, labels, prices, self.sequence_length)
        print(f"Sequences created: {len(X):,}")

        # Compute class weights to handle imbalance
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.array([0, 1, 2]), y=y_signal
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

            if self.model.model is None:
                raise RuntimeError("SignalModel.model is None after build()")
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

            if self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("prepare_data() must be called before training")
            cfg = ModelConfig(
                feature_columns=self.feature_columns,
                feature_mean=self.feature_mean.tolist(),
                feature_std=self.feature_std.tolist(),
                sequence_length=self.sequence_length,
                input_dim=input_dim,
                interval=self.interval,
                training_fetch_date=date.today(),
                holdout_start_date=pd.Timestamp(latest_val_date).date(),
                buy_threshold=self.buy_threshold,
                sell_threshold=self.sell_threshold,
            )
            cfg.save(config_path)
            print(f"Saved model config to {config_path}")

            return history, test_results, config_path

        # Run training with or without MLflow tracking
        if track_with_mlflow:
            run_name = f"train-{'-'.join(tickers[:3])}"
            if len(tickers) > 3:
                run_name += f"-+{len(tickers) - 3}"

            holdout_date_str = pd.Timestamp(latest_val_date).date().isoformat()
            run_tags = {
                "tickers": ",".join(tickers),
                "run_type": "standard",
                "holdout_start_date": holdout_date_str,
            }
            if tags:
                run_tags.update(tags)

            # Keras autolog captures all training metrics automatically.
            # log_every_n_steps=1 matches our existing per-epoch manual logging.
            mlflow.keras.autolog(log_every_n_steps=1, silent=True)

            with training_run(run_name=run_name, tags=run_tags):
                # Build class distribution params for logging
                label_names = {0: "BUY", 1: "HOLD", 2: "SELL"}
                unique, counts = np.unique(labels, return_counts=True)
                class_dist = {
                    f"class_pct_{label_names.get(int(u), str(u)).lower()}": float(c / len(labels))
                    for u, c in zip(unique, counts, strict=False)
                }

                # Log hyperparameters
                log_hyperparameters(
                    {
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
                        **class_dist,
                    }
                )

                history, test_results, config_path = _do_training()

                # Log final test metrics
                log_metrics(
                    {
                        "test_loss": test_results["loss"],
                        "test_signal_accuracy": test_results["signal_accuracy"],
                        "test_price_mae": test_results["price_target_mae"],
                    }
                )

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
            "loaded_tickers": loaded_tickers,
        }
