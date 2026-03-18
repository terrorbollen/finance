# Architecture & Data Flow

This document describes how data moves through the system from raw market data to a trading signal or a backtest result. Read this before touching any module that sits in the middle of the pipeline.

---

## Pipeline Overview

There are two separate pipelines that share the same data preparation steps but diverge at prediction time.

```
                        ┌─────────────────┐
                        │ StockDataFetcher │  yfinance → OHLCV DataFrame
                        └────────┬────────┘
                                 │  DataFrame: open, high, low, close, volume
                                 ▼
                        ┌─────────────────┐
                        │ FeatureEngineer │  ~37 technical + cross-asset + calendar features
                        └────────┬────────┘
                                 │  DataFrame: OHLCV + feature columns (all scale-independent)
                                 │
               ┌─────────────────┴──────────────────┐
               │                                     │
               ▼                                     ▼
     TRAINING PIPELINE                      INFERENCE / BACKTEST PIPELINE
               │                                     │
               ▼                                     ▼
      ModelTrainer.prepare_data()          Normalize with TRAINING stats
      - Label future price changes          from signal_model_config.json
      - Normalize (compute mean/std         (NOT current data statistics)
        on training split ONLY)                      │
      - create_sequences()                           ▼
      - Time split 70/15/15               X[1, seq_len, n_features]
               │                                     │
               ▼                                     ▼
      X[n, seq_len, n_features]           SignalModel.predict()
      y_signal[n]  (0=BUY,1=HOLD,2=SELL) → probs[1,3], class_idx[1], price_change[1]
      y_price[n]   (% change)                        │
               │                         ┌───────────┴───────────┐
               ▼                         │                       │
      SignalModel.fit()          DirectionalCalibrator    Backtester
               │                  (per-direction           (day-by-day,
               ▼                   isotonic regression)     no lookahead)
      checkpoints/                        │                       │
      signal_model.weights.h5             ▼                       ▼
      signal_model_config.json      Signal                 BacktestResult
                                    (direction,             (HorizonMetrics,
                                     confidence,             equity curve,
                                     stop_loss,              MLflow log)
                                     target_price,
                                     position_size)
```

---

## Stage-by-stage Details

### 1. Data Fetching — `data/fetcher.py`

`StockDataFetcher.fetch(ticker)` returns a DataFrame with a `DatetimeIndex` and columns:
`open`, `high`, `low`, `close`, `volume`

`fetch_cross_asset_data(index)` returns a dict of reference DataFrames:
`{"omxs30": df, "usdsek": df, "eursek": df}`
Each must share the same DatetimeIndex as the stock DataFrame.

### 2. Feature Engineering — `data/features.py`

`FeatureEngineer(df, reference_data).add_all_features()` appends ~37 columns to the DataFrame and drops NaN rows (from rolling window warm-up).

**Important:** All features are scale-independent — ratios, percentages, or bounded indicators. The model never sees raw prices. This is intentional: it makes the model generalisable across stocks with very different price levels.

**ATR is expressed as % of price** (not absolute SEK). Code that uses `atr` must treat it as a fraction/percentage, not a point value. Stop-loss calculations in `signals/generator.py` divide by 100.

### 3. The Config File — `checkpoints/signal_model_config.json`

This JSON is the contract between training and inference. It is written by `ModelTrainer` and read by `SignalGenerator` and `Backtester`.

| Field | Type | Description |
|---|---|---|
| `feature_columns` | `list[str]` | Ordered list of feature column names the model was trained on |
| `feature_mean` | `list[float]` | Per-feature mean computed on the **training split only** |
| `feature_std` | `list[float]` | Per-feature std computed on the **training split only** |
| `sequence_length` | `int` | LSTM lookback window in bars (default 20) |
| `input_dim` | `int` | Number of features (= `len(feature_columns)`) |
| `interval` | `str` | Data interval the model was trained on (always `"1d"`) |
| `training_fetch_date` | `str` (ISO date) | Date training data was fetched |
| `holdout_start_date` | `str` (ISO date) | Earliest date the backtester is allowed to evaluate |

**Adding a new feature requires retraining** — the config will be out of date and `input_dim` will not match the model weights.

### 4. Sequence Creation — `models/signal_model.py:create_sequences`

Converts the normalized feature matrix into overlapping windows:
- Input: `features[n_rows, n_features]`
- Output: `X[n_samples, sequence_length, n_features]`, `y_signals[n_horizons, n_samples]`, `y_price[n_samples]`

Labels are computed by looking forward `h` days for each horizon `h` in `prediction_horizons`. The last `max(prediction_horizons)` rows have no label and are excluded.

**Data is never shuffled.** Temporal ordering is preserved throughout.

### 5. Model — `models/signal_model.py`

Two-output LSTM:
- **Signal head**: softmax over 3 classes → `[P(BUY), P(HOLD), P(SELL)]`
- **Price head**: linear regression → predicted % price change

Class encoding: `BUY=0, HOLD=1, SELL=2`. This ordering is hardcoded in `SignalModel`, `SignalGenerator`, and `Backtester`. Do not change it without updating all three.

### 6. Normalization at Inference Time

`SignalGenerator` and `Backtester` both apply:
```
features_norm = (features - feature_mean) / feature_std
```
where `feature_mean` and `feature_std` come from the config, **not** from the current data window. Using current-data statistics would introduce data leakage and degrade calibration.

### 7. Calibration — `signals/calibration.py`

Two calibrator types:
- `ConfidenceCalibrator` — global isotonic regression (one calibrator for all directions)
- `DirectionalCalibrator` — separate calibrator per direction (BUY/SELL/HOLD); preferred

`SignalGenerator` loads `checkpoints/calibration_directional.json` first, then falls back to `checkpoints/calibration.json`, then to raw model probabilities.

Recalibrate after every retraining run (`uv run python main.py calibrate`).

### 8. Signal Generation — `signals/generator.py`

Combines model output into an actionable `Signal`:
- Stop-loss: `current_price ± (atr_pct/100) × atr_multiplier`
- Take-profit: `current_price ± (atr_pct/100) × take_profit_atr_multiplier`
- Position size: half-Kelly, capped at `max_position_size` (default 25%)

### 9. Backtesting — `backtesting/`

`Backtester` runs day-by-day over the holdout window. For each day it uses only data available up to that date. Predictions are stored as `HorizonPrediction` objects and outcomes are filled in once the horizon passes.

`MetricsCalculator` then computes `HorizonMetrics` (Sharpe, Sortino, drawdown, Calmar, win rate with p-value) for each prediction horizon.

See [`backtesting/STRATEGY.md`](backtesting/STRATEGY.md) for how to interpret the output.

---

## Module Dependency Graph

```
main.py
  ├── data/fetcher.py
  ├── data/features.py
  ├── models/signal_model.py
  ├── models/training.py          → data/, models/signal_model.py, models/losses.py
  ├── models/walk_forward.py      → models/training.py
  ├── models/mlflow_tracking.py
  ├── signals/generator.py        → data/, models/signal_model.py, signals/calibration.py
  ├── signals/calibration.py      (no internal deps)
  ├── backtesting/backtester.py   → data/, models/signal_model.py, backtesting/metrics.py
  ├── backtesting/metrics.py      → backtesting/results.py
  └── backtesting/results.py      (no internal deps)
```

`signals/` and `backtesting/` are independent of each other. Neither should import from the other.
