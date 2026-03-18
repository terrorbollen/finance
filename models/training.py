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
from models.config import FETCH_PERIOD, ModelConfig
from models.mlflow_tracking import (
    log_hyperparameters,
    log_metrics,
    log_model_artifact,
    setup_mlflow,
    training_run,
)
from models.signal_model import SignalModel, create_sequences
from signals.direction import BUY_IDX, HOLD_IDX, SELL_IDX


class ModelTrainer:
    """Handles model training with time-series aware data splitting."""

    def __init__(
        self,
        sequence_length: int = 20,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        prediction_horizons: list[int] | None = None,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
        use_adaptive_thresholds: bool = True,
        holdout_date: date | None = None,
    ):
        """
        Initialize the trainer.

        Args:
            sequence_length: LSTM lookback window in trading days (default 20).
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation (rest is test)
            prediction_horizons: Trading day horizons to predict (default [5, 10, 20]).
                                 Labels use max-return over the window to reduce noise.
            buy_threshold: Min % gain to label as Buy
            sell_threshold: Max % loss to label as Sell
            use_adaptive_thresholds: If True, adjust thresholds based on stock volatility
        """
        self.sequence_length = sequence_length
        self.prediction_horizons = prediction_horizons if prediction_horizons is not None else [5, 10, 20]
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.holdout_date = holdout_date  # If set, only train on data before this date
        self.model: SignalModel | None = None
        self.feature_columns: list[str] = []
        # Set by prepare_data(); declared here so the object's full shape is visible
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        reference_data: dict[str, pd.DataFrame] | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, pd.DatetimeIndex]:
        """
        Prepare features and labels from raw price data.

        Labels use max-return over each horizon window rather than end-of-horizon
        return, which significantly reduces label noise from single-day randomness.

        Args:
            df: DataFrame with OHLCV data
            reference_data: Optional cross-asset reference data (unused, kept for compat).

        Returns:
            Tuple of (features, labels_list, price_changes, date_index)
            - labels_list: one label array per horizon in self.prediction_horizons
        """
        engineer = FeatureEngineer(df, reference_data=reference_data)
        df_features = engineer.add_all_features()
        self.feature_columns = engineer.get_feature_columns()

        # Determine base thresholds (adaptive scales WITH volatility — higher vol → higher thresh)
        if self.use_adaptive_thresholds:
            volatility = df_features["close"].pct_change().rolling(20).std().median()
            ref_vol = 0.01  # 1% daily vol reference (typical Swedish large-cap)
            scale = max(volatility / ref_vol, 0.5)
            base_buy_thresh = self.buy_threshold * scale
            base_sell_thresh = self.sell_threshold * scale
        else:
            base_buy_thresh = self.buy_threshold
            base_sell_thresh = self.sell_threshold

        # Max-return labeling: for each horizon h, label Buy if the best price
        # achievable over the next h days exceeds buy_thresh; label Sell if the
        # worst price falls below sell_thresh (and upside wasn't reached first).
        close_vals = np.asarray(df_features["close"].values, dtype=float)
        n = len(close_vals)

        # Future end-return at the middle horizon for the price-target regression head
        mid_h = self.prediction_horizons[len(self.prediction_horizons) // 2]
        df_features["future_return"] = (
            df_features["close"].shift(-mid_h) / df_features["close"] - 1
        )

        from numpy.lib.stride_tricks import sliding_window_view  # noqa: PLC0415

        base_h = self.prediction_horizons[0]  # shortest horizon is the reference
        for h in self.prediction_horizons:
            # Scale threshold by sqrt(h) so each horizon requires a proportional move
            # (a 20d window needs 2x the threshold of a 5d window to maintain equal label rates)
            horizon_scale = np.sqrt(h / base_h)
            buy_thresh = base_buy_thresh * horizon_scale
            sell_thresh = base_sell_thresh * horizon_scale

            max_ret = np.full(n, np.nan)
            min_ret = np.full(n, np.nan)
            if n > h:
                windows = sliding_window_view(close_vals[1:], window_shape=h)  # (n-h, h)
                max_ret[: n - h] = windows.max(axis=1) / close_vals[: n - h] - 1
                min_ret[: n - h] = windows.min(axis=1) / close_vals[: n - h] - 1

            # Store labels as columns so dropna() aligns them automatically
            lbl = np.full(n, float(HOLD_IDX), dtype=float)  # default HOLD (float to allow NaN passthrough)
            lbl[np.isnan(max_ret)] = np.nan

            buy_cond = max_ret > buy_thresh
            sell_cond = min_ret < sell_thresh  # sell_thresh is negative

            # Non-conflicting cases
            lbl[buy_cond & ~sell_cond] = BUY_IDX
            lbl[sell_cond & ~buy_cond] = SELL_IDX

            # Conflict: both thresholds crossed — label by larger magnitude
            conflict = buy_cond & sell_cond & ~np.isnan(max_ret)
            lbl[conflict & (max_ret >= np.abs(min_ret))] = BUY_IDX   # upside wins
            lbl[conflict & (max_ret < np.abs(min_ret))] = SELL_IDX   # downside wins
            df_features[f"_label_{h}d"] = lbl

        # Drop rows with NaN (TA warmup + last max_horizon rows without future prices)
        df_features = df_features.dropna()

        label_cols = [f"_label_{h}d" for h in self.prediction_horizons]
        aligned_labels: list[np.ndarray] = [
            df_features[col].to_numpy().astype(int) for col in label_cols
        ]

        features = df_features[self.feature_columns].values
        price_changes = df_features["future_return"].to_numpy() * 100
        date_index = pd.DatetimeIndex(df_features.index)

        features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

        return features, aligned_labels, price_changes, date_index

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
        # Resolve model path if not explicitly provided
        if model_path is None:
            model_path = ModelConfig.checkpoint_paths()["weights"]

        # Setup MLflow tracking if enabled
        if track_with_mlflow:
            setup_mlflow()

        # Fetch and combine data from all tickers
        fetcher = StockDataFetcher(period=FETCH_PERIOD)
        all_features: list[np.ndarray] = []
        all_labels: list[list[np.ndarray]] = []  # [ticker][horizon_idx] -> array
        all_prices: list[np.ndarray] = []
        all_dates: list[pd.DatetimeIndex] = []

        loaded_tickers: list[str] = []
        for ticker in tickers:
            try:
                df = fetcher.fetch(ticker)
                ref_data = fetcher.fetch_cross_asset_data(pd.DatetimeIndex(df.index))
                features, labels_list, prices, dates = self.prepare_data(df, reference_data=ref_data)
                if self.holdout_date is not None:
                    # Keep only data strictly before holdout_date so the model never
                    # trains on any part of the evaluation window.
                    cutoff = pd.Timestamp(self.holdout_date)
                    mask = dates < cutoff
                    features = features[mask]
                    labels_list = [lbl[mask] for lbl in labels_list]
                    prices = prices[mask]
                    dates = dates[mask]
                if len(features) == 0:
                    print(f"Skipping {ticker}: no usable samples")
                    continue
                all_features.append(features)
                all_labels.append(labels_list)
                all_prices.append(prices)
                all_dates.append(dates)
                loaded_tickers.append(ticker)
                print(f"Loaded {len(features)} samples from {ticker}")
            except Exception as e:
                print(f"Error loading {ticker}: {e}")

        if not all_features:
            raise ValueError("No data loaded for training")

        # Combine all data; concatenate each horizon's labels separately
        print(f"\nCombining data from {len(all_features)} tickers...")
        features = np.vstack(all_features)
        # labels_list_combined[i] = concatenated labels for horizon i across all tickers
        labels_list_combined: list[np.ndarray] = [
            np.concatenate([t[i] for t in all_labels])
            for i in range(len(self.prediction_horizons))
        ]
        prices = np.concatenate(all_prices)
        print(f"Total samples: {len(features):,}  |  Features: {features.shape[1]}")

        # Compute the actual holdout start: the latest calendar date that appears
        # in any ticker's training or validation split.  We look at each ticker's
        # date index independently so that the split index maps cleanly to a date.
        if self.holdout_date is not None:
            # The holdout boundary is explicitly specified — use it directly.
            latest_val_date = pd.Timestamp(self.holdout_date)
        else:
            latest_val_date = max(
                dates[min(int(len(dates) * (self.train_ratio + self.val_ratio)), len(dates) - 1)]
                for dates in all_dates
            )

        # Print class distribution BEFORE training (use first horizon as representative)
        label_names = {BUY_IDX: "BUY", HOLD_IDX: "HOLD", SELL_IDX: "SELL"}
        labels_repr = labels_list_combined[0]
        unique, counts = np.unique(labels_repr, return_counts=True)
        total = len(labels_repr)
        print("\n" + "=" * 50)
        print(f"CLASS DISTRIBUTION — {self.prediction_horizons[0]}d horizon (before sequences):")
        print("=" * 50)
        for lbl, count in zip(unique, counts, strict=False):
            pct = count / total * 100
            print(f"  {label_names.get(lbl, lbl)}: {count:,} samples ({pct:.1f}%)")
        print("=" * 50 + "\n")

        # Normalize features using training data only to avoid look-ahead bias.
        # Compute per-ticker training boundaries so that data from a later-dated
        # ticker doesn't contaminate another ticker's normalization stats (M6 fix).
        print("Normalizing features...")
        training_rows: list[np.ndarray] = []
        for ticker_feats in all_features:
            n_t = len(ticker_feats)
            n_seq_t = n_t - self.sequence_length + 1
            train_end_t = int(n_seq_t * self.train_ratio) + self.sequence_length - 1
            train_end_t = min(train_end_t, n_t)
            training_rows.append(ticker_feats[:train_end_t])
        training_features = np.vstack(training_rows)
        self.feature_mean = np.nanmean(training_features, axis=0)
        self.feature_std = np.nanstd(training_features, axis=0) + 1e-8
        features = (features - self.feature_mean) / self.feature_std

        # Replace any remaining inf/nan with 0
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        # Create sequences
        n_sequences = len(features) - self.sequence_length + 1
        print(f"Creating sequences (length={self.sequence_length}, ~{n_sequences:,} sequences)...")
        X, y_signal_list, y_price = create_sequences(
            features, labels_list_combined, prices, self.sequence_length
        )
        print(f"Sequences created: {len(X):,}")

        # Compute class weights from the first horizon's labels (representative)
        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.array([0, 1, 2]), y=y_signal_list[0]
        )
        class_weight_dict = dict(enumerate(class_weights))
        print("Class weights (to balance training):")
        for lbl, weight in class_weight_dict.items():
            print(f"  {label_names.get(lbl, lbl)}: {weight:.3f}")
        print()

        # Sample weights from first horizon labels
        sample_weights = np.array([class_weight_dict[int(lbl)] for lbl in y_signal_list[0]])

        # Time-series split (no shuffling to preserve temporal order)
        n = len(X)
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))

        X_train = X[:train_end]
        y_signal_train = [ys[:train_end] for ys in y_signal_list]
        y_price_train = y_price[:train_end]
        sw_train = sample_weights[:train_end]

        X_val = X[train_end:val_end]
        y_signal_val = [ys[train_end:val_end] for ys in y_signal_list]
        y_price_val = y_price[train_end:val_end]

        X_test = X[val_end:]
        y_signal_test = [ys[val_end:] for ys in y_signal_list]
        y_price_test = y_price[val_end:]

        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")

        # Build model
        input_dim = X.shape[2]

        def _do_training():
            """Inner function to run training (optionally wrapped in MLflow context)."""
            # Pass class weights to the model so balanced_focal_loss can apply
            # per-class alpha weighting (down-weights the dominant HOLD class).
            # Only meaningful when focal loss is active.
            cw = [float(class_weight_dict[i]) for i in range(3)] if use_focal_loss else None
            self.model = SignalModel(
                input_dim=input_dim,
                sequence_length=self.sequence_length,
                use_focal_loss=use_focal_loss,
                class_weights=cw,
                prediction_horizons=self.prediction_horizons,
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

            # Build named output dicts for multi-horizon training
            y_train_dict = {f"signal_{h}d": ys for h, ys in zip(self.prediction_horizons, y_signal_train, strict=True)}
            y_train_dict["price_target"] = y_price_train
            y_val_dict = {f"signal_{h}d": ys for h, ys in zip(self.prediction_horizons, y_signal_val, strict=True)}
            y_val_dict["price_target"] = y_price_val
            y_test_dict = {f"signal_{h}d": ys for h, ys in zip(self.prediction_horizons, y_signal_test, strict=True)}
            y_test_dict["price_target"] = y_price_test

            fit_kwargs: dict = {
                "x": X_train,
                "y": y_train_dict,
                "validation_data": (X_val, y_val_dict),
                "epochs": epochs,
                "batch_size": batch_size,
                "callbacks": callbacks,
                "verbose": 1,
            }

            if not use_focal_loss:
                fit_kwargs["sample_weight"] = {
                    f"signal_{h}d": sw_train for h in self.prediction_horizons
                } | {"price_target": np.ones_like(sw_train)}

            if self.model.model is None:
                raise RuntimeError("SignalModel.model is None after build()")
            history = self.model.model.fit(**fit_kwargs)

            # Evaluate on test set
            test_results: dict[str, float] = self.model.model.evaluate(  # type: ignore[assignment]
                X_test,
                y_test_dict,
                verbose=0,
                return_dict=True,
            )

            print("\nTest Results:")
            for h in self.prediction_horizons:
                acc = test_results.get(f"signal_{h}d_accuracy", float("nan"))
                print(f"  Signal Accuracy ({h}d): {acc:.4f}")
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
                training_fetch_date=date.today(),
                holdout_start_date=pd.Timestamp(latest_val_date).date(),
                buy_threshold=self.buy_threshold,
                sell_threshold=self.sell_threshold,
                prediction_horizons=self.prediction_horizons,
            )
            cfg.save(config_path)
            print(f"Saved model config to {config_path}")

            return history, test_results, config_path

        # Run training with or without MLflow tracking
        run_id: str | None = None
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
                # Build class distribution params from first horizon (representative)
                lbl_repr = labels_list_combined[0]
                unique, counts = np.unique(lbl_repr, return_counts=True)
                class_dist = {
                    f"class_pct_{label_names.get(int(u), str(u)).lower()}": float(c / len(lbl_repr))
                    for u, c in zip(unique, counts, strict=False)
                }

                # Log hyperparameters
                log_hyperparameters(
                    {
                        "sequence_length": self.sequence_length,
                        "train_ratio": self.train_ratio,
                        "val_ratio": self.val_ratio,
                        "prediction_horizons": str(self.prediction_horizons),
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

                # Log final test metrics (one accuracy per horizon + price MAE)
                final_metrics: dict[str, float] = {
                    "test_loss": float(test_results["loss"]),
                    "test_price_mae": float(test_results["price_target_mae"]),
                }
                for h in self.prediction_horizons:
                    key = f"signal_{h}d_accuracy"
                    if key in test_results:
                        final_metrics[f"test_signal_accuracy_{h}d"] = float(test_results[key])
                # Use first-horizon accuracy as the canonical test_signal_accuracy for history view
                first_key = f"signal_{self.prediction_horizons[0]}d_accuracy"
                final_metrics["test_signal_accuracy"] = float(
                    test_results.get(first_key, float("nan"))
                )
                log_metrics(final_metrics)

                # Log model artifacts
                log_model_artifact(model_path, artifact_path="model")
                log_model_artifact(config_path, artifact_path="model")

                # Register model in MLflow Model Registry for version tracking.
                # Wrapped in try/except because older MLflow server versions may not
                # support the registry API — training must not fail over this.
                active = mlflow.active_run()
                if active:
                    run_id = active.info.run_id
                    try:
                        mlflow.register_model(
                            model_uri=f"runs:/{run_id}/model",
                            name="trading-signals",
                            tags={"holdout_start_date": holdout_date_str},
                        )
                    except Exception as e:
                        print(f"Warning: MLflow model registry unavailable ({e}). "
                              "Run ID {run_id} still logged.")
        else:
            history, test_results, _ = _do_training()

        first_acc_key = f"signal_{self.prediction_horizons[0]}d_accuracy"
        return {
            "history": history.history,
            "test_loss": float(test_results["loss"]),
            "test_signal_accuracy": float(test_results.get(first_acc_key, float("nan"))),
            "test_price_mae": float(test_results["price_target_mae"]),
            "loaded_tickers": loaded_tickers,
            "mlflow_run_id": run_id,
        }
