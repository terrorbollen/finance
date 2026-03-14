"""Core backtesting engine."""

import json
import os
from datetime import date

import numpy as np
import pandas as pd

from backtesting.metrics import MetricsCalculator
from backtesting.results import (
    BacktestResult,
    DailyPrediction,
    HorizonPrediction,
    Signal,
)
from data.features import FeatureEngineer
from data.fetcher import StockDataFetcher
from models.signal_model import SignalModel


class Backtester:
    """
    Backtesting engine that runs the model on historical data.

    Simulates making predictions day-by-day using only data
    available up to that point.
    """

    def __init__(
        self,
        model_path: str = "checkpoints/signal_model.weights.h5",
        sequence_length: int = 20,
        buy_threshold: float = 0.02,
        sell_threshold: float = -0.02,
        commission_pct: float = 0.001,
    ):
        """
        Initialize the backtester.

        Args:
            model_path: Path to trained model weights
            sequence_length: Sequence length used during training
            buy_threshold: Price change threshold for BUY signal
            sell_threshold: Price change threshold for SELL signal
            commission_pct: One-way commission as decimal (default 0.1%)
        """
        self.model_path = model_path
        self.sequence_length = sequence_length
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

        self.model: SignalModel | None = None
        self.feature_columns: list[str] | None = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None
        self.input_dim: int | None = None

        self.metrics_calculator = MetricsCalculator(
            buy_threshold=buy_threshold,
            sell_threshold=sell_threshold,
            commission_pct=commission_pct,
        )

        self._load_config()

    def _load_config(self):
        """Load training configuration from file."""
        config_path = self.model_path.replace(".weights.h5", "_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            self.feature_columns = config.get("feature_columns")
            self.feature_mean = np.array(config.get("feature_mean"))
            self.feature_std = np.array(config.get("feature_std"))
            self.sequence_length = config.get("sequence_length", self.sequence_length)
            self.input_dim = config.get("input_dim")

    def _load_model(self, input_dim: int):
        """Load the trained model."""
        self.model = SignalModel(input_dim=input_dim, sequence_length=self.sequence_length)
        try:
            self.model.load(self.model_path)
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using untrained model (predictions will be random)")

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
        fetcher = StockDataFetcher(period="5y")
        df_raw = fetcher.fetch(ticker)

        # Add features
        engineer = FeatureEngineer(df_raw)
        df = engineer.add_all_features()

        # Get feature columns
        if self.feature_columns is not None:
            available_cols = [c for c in self.feature_columns if c in df.columns]
            feature_cols = available_cols
        else:
            feature_cols = engineer.get_feature_columns()

        # Set default date range
        if start_date is None:
            start_date = (pd.Timestamp.now() - pd.DateOffset(years=1)).date()
        if end_date is None:
            end_date = pd.Timestamp.now().date()

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

        print(f"Running backtest on {ticker} from {start_date} to {end_date}")
        print(f"Processing {len(backtest_indices)} trading days...")

        if horizons is None:
            horizons = [1, 2, 3, 4, 5, 6, 7]

        # Process each day
        daily_predictions = []

        for idx in backtest_indices:
            pred_date = pd.DatetimeIndex(df.index)[idx].date()
            current_price = float(df["close"].iloc[idx])

            # Make predictions for all horizons
            daily_pred = DailyPrediction(
                date=pred_date,
                current_price=current_price,
            )

            for horizon in horizons:
                prediction = self._predict_for_date(
                    df=df,
                    feature_cols=feature_cols,
                    idx=idx,
                    horizon=horizon,
                )
                daily_pred.add_prediction(prediction)

            daily_predictions.append(daily_pred)

        # Fill in actual outcomes
        self._fill_actual_outcomes(df, daily_predictions, horizons)

        # Calculate buy & hold return
        first_price = df["close"].iloc[backtest_indices[0]]
        last_price = df["close"].iloc[backtest_indices[-1]]
        buy_hold_return = ((last_price / first_price) - 1) * 100

        # Calculate metrics per horizon
        horizon_metrics = {}
        for horizon in horizons:
            predictions = [
                dp.predictions[horizon] for dp in daily_predictions if horizon in dp.predictions
            ]
            horizon_metrics[horizon] = self.metrics_calculator.calculate_horizon_metrics(
                predictions, horizon
            )

        return BacktestResult(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            trading_days=len(backtest_indices),
            buy_hold_return=buy_hold_return,
            daily_predictions=daily_predictions,
            horizon_metrics=horizon_metrics,
        )

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
            mean = features.mean(axis=0)
            std = features.std(axis=0) + 1e-8
            features_norm = (features - mean) / std

        # Create sequence
        X = features_norm.reshape(1, self.sequence_length, -1)

        # Get prediction
        assert self.model is not None
        signal_probs, signal_class, price_target = self.model.predict(X)

        direction_idx = signal_class[0]
        predicted_signal = Signal(direction_idx)
        confidence = float(signal_probs[0][direction_idx])
        predicted_change = float(price_target[0])

        return HorizonPrediction(
            prediction_date=pred_date,
            horizon_days=horizon,
            predicted_signal=predicted_signal,
            confidence=confidence,
            predicted_price_change=predicted_change,
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
        date_to_idx = {d.date(): i for i, d in enumerate(df.index)}

        for daily_pred in daily_predictions:
            for horizon in horizons:
                if horizon not in daily_pred.predictions:
                    continue

                pred = daily_pred.predictions[horizon]

                # Find target date index
                # Need to find the trading day that is 'horizon' days ahead
                pred_idx = date_to_idx.get(pred.prediction_date)
                if pred_idx is None:
                    continue

                target_idx = pred_idx + horizon

                if target_idx >= len(df):
                    # Future date not available yet
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
                pred.target_date = df.index[target_idx].date()
