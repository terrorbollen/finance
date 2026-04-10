"""Core backtesting engine."""

import os
from datetime import date

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from backtesting.metrics import MetricsCalculator
from backtesting.results import (
    BacktestResult,
    DailyPrediction,
    HorizonPrediction,
    Signal,
)
from data.features import FeatureEngineer
from data.fetcher import StockDataFetcher
from models.config import FETCH_PERIOD, ModelConfig
from models.signal_model import SignalModel


def _bh_correction(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction for a list of p-values."""
    n = len(pvalues)
    if n == 0:
        return []
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    corrected = [1.0] * n
    for rank, (orig_idx, pval) in enumerate(indexed, 1):
        corrected[orig_idx] = min(1.0, pval * n / rank)
    # Enforce monotonicity (corrected p-values must be non-decreasing in rank order)
    for i in range(len(indexed) - 2, -1, -1):
        corrected[indexed[i][0]] = min(corrected[indexed[i][0]], corrected[indexed[i + 1][0]])
    return corrected


class Backtester:
    """
    Backtesting engine that runs the model on historical data.

    Simulates making predictions day-by-day using only data
    available up to that point.
    """

    def __init__(
        self,
        commission_pct: float,
        slippage_factor: float,
        leverage: float,
        model_path: str | None = None,
        strict_holdout: bool = True,
        enforce_position_cooldown: bool = True,
        retrain_every: int | None = None,
        retrain_epochs: int = 20,
        retrain_batch_size: int = 32,
    ):
        """
        Initialize the backtester.

        Args:
            commission_pct: One-way commission as decimal — must match the training config.
            slippage_factor: Scaling constant for volume-based slippage model.
                             slippage_pct = slippage_factor / sqrt(relative_volume),
                             capped at 0.5% per trade. Set to 0 to disable slippage.
            leverage: Leverage multiplier applied to each trade.
            model_path: Path to trained model weights. Defaults to checkpoint.
            strict_holdout: If True, enforce that backtest start is on or after the
                            holdout start date saved in the model config.
            enforce_position_cooldown: If True, after a non-HOLD trade skip the next
                                       horizon predictions to avoid overlapping positions.
            retrain_every: If set, retrain the model every N trading days during the
                           backtest. None disables retraining.
            retrain_epochs: Number of training epochs per retrain (default 20).
            retrain_batch_size: Batch size used during periodic retrains. Should match
                                the batch_size from the training config (default 32).

        Note: sequence_length, buy_threshold, sell_threshold, and prediction_horizons
        are not constructor parameters — they are always read from the saved model config
        (signal_model_config.json) by _load_config(). The model config is the single
        source of truth for these values.
        """
        self.model_path = model_path or ModelConfig.checkpoint_paths()["weights"]
        self.strict_holdout = strict_holdout
        self.enforce_position_cooldown = enforce_position_cooldown
        self.retrain_every = retrain_every
        self.retrain_epochs = retrain_epochs
        self.retrain_batch_size = retrain_batch_size

        # Stored so _load_config() can create MetricsCalculator with the correct values
        self._commission_pct = commission_pct
        self._slippage_factor = slippage_factor
        self._leverage = leverage

        self.model: SignalModel | None = None
        self.feature_columns: list[str] | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.input_dim: int | None = None
        self.holdout_start_date: date | None = None
        self.prediction_horizons: list[int] | None = None
        # These three are set by _load_config() from the saved model JSON.
        # Placeholders here; _load_config() raises if the config file is missing.
        self.sequence_length: int = 0
        self.buy_threshold: float = 0.0
        self.sell_threshold: float = 0.0
        self.metrics_calculator = MetricsCalculator(
            buy_threshold=0.0,
            sell_threshold=0.0,
            commission_pct=commission_pct,
            slippage_factor=slippage_factor,
            leverage=leverage,
            enforce_position_cooldown=enforce_position_cooldown,
        )

        self._load_config()

    def _load_config(self) -> None:
        """Load training configuration from the saved model config JSON.

        This is the single source of truth for sequence_length, buy_threshold,
        sell_threshold, and prediction_horizons. Raises if the file is missing —
        a Backtester without a model config cannot produce meaningful results.
        """
        config_path = self.model_path.replace(".weights.h5", "_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model config not found at '{config_path}'. "
                "Train a model first with: uv run python main.py train --config configs/indexes.json"
            )
        cfg = ModelConfig.load(config_path)
        self.feature_columns = cfg.feature_columns
        self.feature_mean = cfg.feature_mean_array
        self.feature_std = cfg.feature_std_array
        self.sequence_length = cfg.sequence_length
        self.input_dim = cfg.input_dim
        self.holdout_start_date = cfg.holdout_start_date
        self.buy_threshold = cfg.buy_threshold
        self.sell_threshold = cfg.sell_threshold
        self.prediction_horizons = cfg.prediction_horizons
        self.metrics_calculator = MetricsCalculator(
            buy_threshold=cfg.buy_threshold,
            sell_threshold=cfg.sell_threshold,
            commission_pct=self._commission_pct,
            slippage_factor=self._slippage_factor,
            leverage=self._leverage,
            enforce_position_cooldown=self.enforce_position_cooldown,
        )

    def _load_model(self, input_dim: int) -> None:
        """Load the trained model and warm up TF graph compilation."""
        self.model = SignalModel(
            input_dim=input_dim,
            sequence_length=self.sequence_length,
            prediction_horizons=self.prediction_horizons,
        )
        try:
            self.model.load(self.model_path)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using untrained model (predictions will be random)")
        # Warm up: run one dummy prediction to trigger TF graph compilation now,
        # so all subsequent predict calls use the cached compiled graph.
        dummy = np.zeros((1, self.sequence_length, input_dim), dtype=np.float32)
        self.model.predict(dummy)

    def run(
        self,
        ticker: str,
        start_date: date | None = None,
        end_date: date | None = None,
        horizons: list[int] | None = None,
    ) -> BacktestResult:
        """
        Run backtest on historical data.

        Args:
            ticker: Stock ticker symbol
            start_date: Start date for backtest (default: 1 year ago)
            end_date: End date for backtest (default: today)
            horizons: List of prediction horizons in days

        Returns:
            BacktestResult with all predictions and metrics
        """
        # Fetch full history
        fetcher = StockDataFetcher(period=FETCH_PERIOD)
        df_raw = fetcher.fetch(ticker)

        # Add features
        engineer = FeatureEngineer(df_raw)
        df = engineer.add_all_features()

        # Get feature columns
        if self.feature_columns is not None:
            missing = sorted(set(self.feature_columns) - set(df.columns))
            if missing:
                raise ValueError(
                    f"Feature mismatch: {len(missing)} column(s) present in training config "
                    f"but missing from data — retrain or fix the data pipeline. "
                    f"Missing: {missing}"
                )
            feature_cols = self.feature_columns
        else:
            feature_cols = engineer.get_feature_columns()

        # Set default date range
        if start_date is None:
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).date()
        if end_date is None:
            end_date = pd.Timestamp.now().date()

        # Enforce strict holdout: backtest must not overlap with training/validation data.
        if self.strict_holdout and self.holdout_start_date is not None:
            if start_date < self.holdout_start_date:
                print(
                    f"WARNING: Requested start {start_date} overlaps with training data "
                    f"(holdout begins {self.holdout_start_date}). "
                    f"Adjusting start to {self.holdout_start_date} to avoid look-ahead bias."
                )
                start_date = self.holdout_start_date
        elif self.holdout_start_date is None and self.strict_holdout:
            print(
                "NOTE: Model config does not contain a holdout date "
                "(model was trained before this feature was added). "
                "Cannot enforce strict holdout. Re-train the model to enable this check."
            )

        # Initialize model
        if self.model is None:
            input_dim = self.input_dim if self.input_dim else len(feature_cols)
            self._load_model(input_dim)
        if self.model is None:
            raise RuntimeError("Failed to initialize model")

        # Filter to backtest period (but keep earlier data for features)
        df_dates = pd.DatetimeIndex(df.index).date
        backtest_mask = (df_dates >= start_date) & (df_dates <= end_date)
        all_indices = np.where(backtest_mask)[0]

        if len(all_indices) == 0:
            raise ValueError(f"No data in backtest period {start_date} to {end_date}")

        # Need enough history for sequence
        min_start_idx = self.sequence_length
        backtest_indices = [int(i) for i in all_indices if i >= min_start_idx]

        print(f"Running backtest on {ticker} from {start_date} to {end_date}", flush=True)
        print(f"Processing {len(backtest_indices)} trading days...", flush=True)
        print(
            "Note: confidence scores in this backtest are raw model output (pre-calibration). "
            "Live signals via `generate` apply isotonic calibration, so their confidence "
            "values will differ. The calibration table in the summary reflects raw scores.",
            flush=True,
        )

        if horizons is None:
            horizons = self.prediction_horizons or [5, 10, 20]

        df_index = pd.DatetimeIndex(df.index)

        # Build sequences and run predictions.  When retrain_every is set the
        # backtest indices are split into chunks; the model is retrained at the
        # start of each chunk (except the first) using all data up to that date.
        feature_array = df[feature_cols].values

        def _normalize(arr: np.ndarray) -> np.ndarray:
            if self.feature_mean is not None and len(self.feature_mean) == len(feature_cols):
                return (arr - self.feature_mean) / self.feature_std  # type: ignore[operator, no-any-return]
            mean = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-8
            return (arr - mean) / std  # type: ignore[no-any-return]

        if self.model is None:
            raise RuntimeError("Model not initialized — call _load_model() first")

        if self.retrain_every is None:
            # Fast path: single batch over the full period.
            feature_array_norm = _normalize(feature_array)
            X_all = np.stack(
                [
                    feature_array_norm[idx - self.sequence_length + 1 : idx + 1]
                    for idx in backtest_indices
                ]
            )
            horizon_probs_list, horizon_classes_list, price_targets_all = (
                self.model.predict_per_horizon(X_all)
            )
        else:
            # Chunked path: retrain at the boundary of every chunk.
            n = self.retrain_every
            chunks = [backtest_indices[i : i + n] for i in range(0, len(backtest_indices), n)]
            all_horizon_probs_parts: list[list[np.ndarray]] = []
            all_horizon_classes_parts: list[list[np.ndarray]] = []
            all_prices_parts: list[np.ndarray] = []

            for chunk_idx, chunk in enumerate(chunks):
                if chunk_idx > 0:
                    cutoff = df_index[chunk[0]].date()
                    print(
                        f"\nRetraining model at {cutoff} (chunk {chunk_idx + 1}/{len(chunks)})...",
                        flush=True,
                    )
                    self._retrain_model(df_raw, cutoff)

                # Recompute normalised features with current stats after any retrain.
                feature_array_norm = _normalize(feature_array)
                X_chunk = np.stack(
                    [feature_array_norm[idx - self.sequence_length + 1 : idx + 1] for idx in chunk]
                )
                if self.model is None:
                    raise RuntimeError("Model lost after retrain")
                h_probs, h_classes, prices = self.model.predict_per_horizon(X_chunk)
                all_horizon_probs_parts.append(h_probs)
                all_horizon_classes_parts.append(h_classes)
                all_prices_parts.append(prices)

            n_heads = len(self.model.prediction_horizons)
            horizon_probs_list = [
                np.vstack([parts[h] for parts in all_horizon_probs_parts])
                for h in range(n_heads)
            ]
            horizon_classes_list = [
                np.concatenate([parts[h] for parts in all_horizon_classes_parts])
                for h in range(n_heads)
            ]
            price_targets_all = np.concatenate(all_prices_parts)

        # Average probs across heads for fallback (horizons not in the trained set)
        avg_probs_all = np.mean(np.stack(horizon_probs_list, axis=0), axis=0)  # (n, 3)

        # Map trained horizon values → head index for O(1) lookup in inner loop
        model_horizons = self.model.prediction_horizons
        horizon_head_idx: dict[int, int] = {h: i for i, h in enumerate(model_horizons)}

        # Precompute relative volumes for all indices
        rel_vols: np.ndarray | None = None
        if "volume" in df.columns:
            vol_series = df["volume"]
            rolling_avg = vol_series.rolling(window=20, min_periods=1).mean()
            with np.errstate(divide="ignore", invalid="ignore"):
                rel_vols = np.where(rolling_avg > 0, vol_series / rolling_avg, np.nan)

        # Precompute ADX(14) values for regime filtering
        adx_vals: np.ndarray | None = None
        if "adx_14" in df.columns:
            adx_vals = df["adx_14"].to_numpy(dtype=float)

        # Process each day using pre-computed predictions
        daily_predictions = []
        for i, idx in enumerate(backtest_indices):
            pred_date = df_index[idx].date()
            current_price = float(df["close"].iloc[idx])

            predicted_change = float(price_targets_all[i])
            rv = float(rel_vols[idx]) if rel_vols is not None else float("nan")
            relative_volume = None if (rel_vols is None or np.isnan(rv)) else rv
            adx_raw = float(adx_vals[idx]) if adx_vals is not None else float("nan")
            adx = None if np.isnan(adx_raw) else adx_raw

            daily_pred = DailyPrediction(date=pred_date, current_price=current_price)
            for horizon in horizons:
                head_idx = horizon_head_idx.get(horizon)
                if head_idx is not None:
                    probs_row = horizon_probs_list[head_idx][i]
                    direction_idx = int(horizon_classes_list[head_idx][i])
                else:
                    # Horizon not trained — fall back to average across all heads
                    probs_row = avg_probs_all[i]
                    direction_idx = int(np.argmax(probs_row))
                predicted_signal = [Signal.BUY, Signal.HOLD, Signal.SELL][direction_idx]
                confidence = float(probs_row[direction_idx])
                all_probs = (float(probs_row[0]), float(probs_row[1]), float(probs_row[2]))
                daily_pred.add_prediction(
                    HorizonPrediction(
                        prediction_date=pred_date,
                        horizon_days=horizon,
                        predicted_signal=predicted_signal,
                        confidence=confidence,
                        predicted_price_change=predicted_change,
                        relative_volume=relative_volume,
                        adx=adx,
                        all_probs=all_probs,
                    )
                )
            daily_predictions.append(daily_pred)

        # Fill in actual outcomes
        self._fill_actual_outcomes(df, daily_predictions, horizons)

        # Calculate buy & hold return
        first_price = df["close"].iloc[backtest_indices[0]]
        last_price = df["close"].iloc[backtest_indices[-1]]
        buy_hold_return = ((last_price / first_price) - 1) * 100

        # Calculate OMXS30 benchmark return over the same period.
        benchmark_return = self._compute_benchmark_return(start_date, end_date)

        # Calculate metrics per horizon
        horizon_metrics = {}
        for horizon in horizons:
            predictions = [
                dp.predictions[horizon] for dp in daily_predictions if horizon in dp.predictions
            ]
            horizon_metrics[horizon] = self.metrics_calculator.calculate_horizon_metrics(
                predictions, horizon
            )

        # Apply Benjamini-Hochberg FDR correction across horizons
        sorted_horizons = sorted(horizon_metrics.keys())
        raw_pvalues = [horizon_metrics[h].win_rate_pvalue for h in sorted_horizons]
        corrected = _bh_correction(raw_pvalues)
        for h, bh_pval in zip(sorted_horizons, corrected, strict=False):
            horizon_metrics[h].bh_corrected_pvalue = bh_pval

        return BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            trading_days=len(backtest_indices),
            buy_hold_return=buy_hold_return,
            benchmark_return=benchmark_return,
            leverage=self.metrics_calculator.leverage,
            daily_predictions=daily_predictions,
            horizon_metrics=horizon_metrics,
        )

    def _retrain_model(
        self,
        df_raw: pd.DataFrame,
        cutoff_date: date,
    ) -> None:
        """Retrain the model on all raw data strictly before cutoff_date.

        Updates self.model, self.feature_mean, self.feature_std, and
        self.feature_columns in-place.  Normalization stats are computed
        from the filtered training data only — no look-ahead.

        Args:
            df_raw: Full raw OHLCV DataFrame for the ticker.
            cutoff_date: Only rows strictly before this date are used.
        """
        from tensorflow import keras  # noqa: PLC0415

        from models.signal_model import create_sequences  # noqa: PLC0415
        from models.walk_forward import WalkForwardTrainer  # noqa: PLC0415

        mask = pd.DatetimeIndex(df_raw.index).date < cutoff_date
        df_cut = df_raw[mask]

        min_rows = self.sequence_length + max(self.prediction_horizons or [20]) + 50
        if len(df_cut) < min_rows:
            print(
                f"  Skipping retrain: only {len(df_cut)} rows before {cutoff_date} (need {min_rows})"
            )
            return

        horizons = self.prediction_horizons or [5, 10, 20]
        wf = WalkForwardTrainer(
            sequence_length=self.sequence_length,
            prediction_horizons=horizons,
            buy_threshold=self.buy_threshold,
            sell_threshold=self.sell_threshold,
        )

        features, labels_list, price_changes = wf.prepare_data(df_cut)

        # Normalize using this window's data only (invariant: no look-ahead)
        feature_mean = np.nanmean(features, axis=0)
        feature_std = np.nanstd(features, axis=0) + 1e-8
        features_norm = (features - feature_mean) / feature_std
        features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)

        X, y_signal_list, y_price = create_sequences(
            features_norm, labels_list, price_changes, self.sequence_length
        )

        if len(X) < 50:
            print(f"  Skipping retrain: only {len(X)} sequences (need 50)")
            return

        input_dim = X.shape[2]
        model = SignalModel(
            input_dim=input_dim,
            sequence_length=self.sequence_length,
            prediction_horizons=horizons,
            use_focal_loss=True,
        )

        y_dict: dict[str, np.ndarray] = {
            f"signal_{h}d": ys for h, ys in zip(horizons, y_signal_list, strict=False)
        }
        y_dict["price_target"] = y_price

        assert model.model is not None
        model.model.fit(  # type: ignore[call-arg]
            X,
            y_dict,
            epochs=self.retrain_epochs,
            batch_size=self.retrain_batch_size,
            callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
            verbose=0,
        )

        self.model = model
        self.feature_columns = wf.feature_columns
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.input_dim = input_dim
        # NOTE (B13): backtesting/ must not import signals/ (INVARIANTS.md), so the
        # backtester never loads a ConfidenceCalibrator.  Confidence scores are always
        # raw softmax outputs.  After a retrain the newly trained model's probability
        # distribution changes anyway, so any pre-retrain calibration would be
        # misleading — raw scores are the honest choice here.
        print(f"  Retrain complete — {len(X)} sequences, input_dim={input_dim}", flush=True)

    def _compute_benchmark_return(
        self,
        start_date: date,
        end_date: date,
        omxs30_df: pd.DataFrame | None = None,
    ) -> float | None:
        """Return OMXS30 return over the backtest period as a passive benchmark.

        Uses ``omxs30_df`` if provided (already fetched during feature engineering)
        to avoid a redundant yfinance request. Falls back to a live fetch if not
        supplied. Returns None if data is unavailable.
        """
        try:
            if omxs30_df is None:
                omxs30_df = StockDataFetcher(period=FETCH_PERIOD).fetch("^OMX")
            bench_dates = pd.DatetimeIndex(omxs30_df.index).date
            mask = (bench_dates >= start_date) & (bench_dates <= end_date)
            df_period = omxs30_df[mask]
            if len(df_period) < 2:
                return None
            first = float(df_period["close"].iloc[0])
            last = float(df_period["close"].iloc[-1])
            return ((last / first) - 1) * 100
        except (ValueError, KeyError, IndexError, OSError) as e:
            print(f"Warning: Could not compute OMXS30 benchmark return: {e}")
            return None

    def _predict_for_date(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        idx: int,
        horizon: int,
    ) -> HorizonPrediction:
        """
        Generate prediction using data available at a specific date.

        Args:
            df: Full dataframe with features
            feature_cols: Feature column names
            idx: Index of the prediction date in df
            horizon: Number of days to predict ahead

        Returns:
            HorizonPrediction with model outputs
        """
        pred_date = df.index[idx].date()

        # Extract features up to this point
        features = df[feature_cols].iloc[idx - self.sequence_length + 1 : idx + 1].values

        # Normalize
        if self.feature_mean is not None and len(self.feature_mean) == len(feature_cols):
            features_norm = (features - self.feature_mean) / self.feature_std
        else:
            print(
                "Warning: Training normalization stats not available — falling back to "
                "current-window statistics. Predictions may be unreliable. Re-train the model."
            )
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
            features_norm = (features - mean) / std

        # Create sequence
        X = features_norm.reshape(1, self.sequence_length, -1)

        # Get prediction
        if self.model is None:
            raise RuntimeError("Model not initialized — call _load_model() first")
        signal_probs, signal_class, price_target = self.model.predict(X)

        direction_idx = signal_class[0]
        predicted_signal = [Signal.BUY, Signal.HOLD, Signal.SELL][direction_idx]
        confidence = float(signal_probs[0][direction_idx])
        predicted_change = float(price_target[0])

        # Compute relative volume for slippage modeling
        relative_volume: float | None = None
        if "volume" in df.columns:
            try:
                vol_series = df["volume"].iloc[: idx + 1]
                rolling_avg = vol_series.rolling(window=20, min_periods=1).mean().iloc[-1]
                today_vol = float(vol_series.iloc[-1])
                if rolling_avg > 0:
                    relative_volume = today_vol / float(rolling_avg)
            except (IndexError, ValueError, TypeError) as e:
                print(f"Warning: Could not compute relative volume at index {idx}: {e}")
                relative_volume = None

        return HorizonPrediction(
            prediction_date=pred_date,
            horizon_days=horizon,
            predicted_signal=predicted_signal,
            confidence=confidence,
            predicted_price_change=predicted_change,
            relative_volume=relative_volume,
        )

    def _fill_actual_outcomes(
        self,
        df: pd.DataFrame,
        daily_predictions: list[DailyPrediction],
        horizons: list[int],
    ):
        """
        Fill in actual outcomes for all predictions.

        Args:
            df: Full dataframe with price data
            daily_predictions: List of daily predictions to update
            horizons: List of prediction horizons
        """
        # Create date to index mapping
        df_index = pd.DatetimeIndex(df.index)
        date_to_idx = {d.date(): i for i, d in enumerate(df_index)}

        for daily_pred in daily_predictions:
            for horizon in horizons:
                if horizon not in daily_pred.predictions:
                    continue

                pred = daily_pred.predictions[horizon]

                pred_idx = date_to_idx.get(pred.prediction_date)
                if pred_idx is None:
                    continue

                # Use business-day date arithmetic to find the target date.
                # Row-offset arithmetic (pred_idx + horizon) is wrong when
                # yfinance drops sessions: a single gap shifts every target
                # price by one day, silently misstating all P&L.
                # Instead, compute the exact calendar business day that is
                # `horizon` BDays ahead, then look it up by date.  Scan
                # forward up to 3 extra business days to absorb Swedish
                # public holidays that yfinance omits.
                target_ts = df_index[pred_idx] + BDay(horizon)
                target_idx: int | None = None
                for extra in range(4):
                    candidate = (target_ts + BDay(extra)).date()
                    if candidate in date_to_idx:
                        target_idx = date_to_idx[candidate]
                        break

                if target_idx is None:
                    # Target date not yet in data (near end of available history)
                    continue

                # Get actual price change
                pred_price = df["close"].iloc[pred_idx]
                target_price = df["close"].iloc[target_idx]
                actual_change = ((target_price / pred_price) - 1) * 100  # percentage

                # Determine actual signal
                actual_signal = self.metrics_calculator.determine_actual_signal(
                    actual_change / 100  # convert to decimal
                )

                # Update prediction
                pred.actual_price_change = actual_change
                pred.actual_signal = actual_signal
                pred.target_date = df_index[target_idx].date()
