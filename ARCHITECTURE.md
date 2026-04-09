# Architecture & Data Flow

This document describes how data moves through the system from raw market data to a trading signal or a backtest result. Read this before touching any module that sits in the middle of the pipeline.

---

## Pipeline Overview

There are two separate pipelines that share the same data preparation steps but diverge at prediction time.

```
                        ┌─────────────────┐
                        │ StockDataFetcher │  yfinance → OHLCV DataFrame (10y history)
                        └────────┬────────┘
                                 │  + fetch_cross_asset_data() [concurrent]
                                 │  OMXS30, USD/SEK, EUR/SEK, VIX, VSTOXX, Brent, 10Y rates
                                 ▼
                        ┌─────────────────┐
                        │ FeatureEngineer │  17 technical + cross-asset + calendar features
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
      - Max-return labeling per horizon     from signal_model_config.json
      - Adaptive thresholds (vol-scaled)    (NOT current data statistics)
      - Normalize (training split only,              │
        per-ticker boundary)                         ▼
      - create_sequences()               X[1, seq_len, n_features]
      - Time split 70/15/15                          │
               │                                     ▼
               ▼                          SignalModel / EnsembleSignalModel
      X[n, seq_len, n_features]           .predict_per_horizon()
      y_signal[n_horizons, n]             → per-horizon softmax probs + price_target
      y_price[n]                                     │
               │                         majority-vote consensus → signal_class
               ▼                                     │
      SignalModel / EnsembleSignalModel  ┌────────────┴───────────┐
      .fit() — one head per horizon      │                        │
               │                DirectionalCalibrator      Backtester / PortfolioBacktester
               ▼                  (per-direction             (day-by-day,
      checkpoints/                 isotonic regression)       no lookahead)
      signal_model.weights.h5             │                        │
      signal_model_config.json            ▼                        ▼
                                    Signal                  BacktestResult / PortfolioResult
                                    (direction,              (HorizonMetrics,
                                     confidence,              equity curve,
                                     stop_loss,               MLflow log)
                                     target_price,
                                     position_size)
```

---

## Stage-by-stage Details

### 1. Data Fetching — `data/fetcher.py`

`StockDataFetcher(period="10y", interval="1d")` fetches 10 years of daily OHLCV data by default.

`StockDataFetcher.fetch(ticker)` returns a DataFrame with a `DatetimeIndex` and columns:
`open`, `high`, `low`, `close`, `volume`

`fetch_cross_asset_data(align_to)` fetches 7 reference series **concurrently** using a `ThreadPoolExecutor`:

| Key | Ticker | Purpose |
|---|---|---|
| `omxs30` | `^OMX` | Swedish benchmark index |
| `usdsek` | `USDSEK=X` | USD/SEK FX rate |
| `eursek` | `EURSEK=X` | EUR/SEK FX rate |
| `vix` | `^VIX` | Global fear gauge |
| `vstoxx` | `^V2TX` | European volatility index |
| `oil` | `BZ=F` | Brent crude oil |
| `rates` | `^TNX` | US 10Y Treasury yield |

All reference series are forward-filled, back-filled, and zero-padded to match the primary stock's `DatetimeIndex`. Missing data never propagates NaNs downstream.

### 2. Feature Engineering — `data/features.py`

`FeatureEngineer(df, reference_data).add_all_features()` appends 17 columns to the DataFrame and drops NaN rows (from rolling window warm-up).

**Important:** All features are scale-independent — ratios, percentages, or bounded indicators. The model never sees raw prices. This makes the model generalizable across stocks with very different price levels.

**ATR is expressed as % of price** (not absolute SEK). Code that uses `atr` must treat it as a fraction/percentage, not a point value. Stop-loss calculations in `signals/generator.py` divide by 100.

| Feature | Method | Description |
|---|---|---|
| `rsi` | `_add_rsi(14)` | RSI(14), range 0–100 |
| `macd_histogram` | `_add_macd_histogram(12,26,9)` | MACD histogram normalised by price |
| `momentum_10` | `_add_momentum(10)` | 10-day % price change |
| `returns` | `_add_returns()` | 1-day % return |
| `atr` | `_add_atr(14)` | ATR(14) as % of price |
| `bb_position` | `_add_bb_position(20)` | Position within Bollinger Bands, 0–1 |
| `volume_ratio` | `_add_volume_ratio(20)` | Volume / 20-day avg, clipped at 10x |
| `adx_14` | `_add_adx(14)` | ADX(14) trend strength, 0–100 (Wilder smoothing) |
| `vix_level` | `_add_volatility_features()` | VIX normalised by 252-day rolling mean |
| `vix_1d_change` | `_add_volatility_features()` | Daily % change in VIX |
| `vix_stock_corr` | `_add_volatility_features()` | 20-day rolling corr: stock returns vs VIX changes |
| `vstoxx_level` | `_add_volatility_features()` | VSTOXX normalised by 252-day rolling mean |
| `vstoxx_1d_change` | `_add_volatility_features()` | Daily % change in VSTOXX |
| `oil_level` | `_add_macro_features()` | Brent normalised by 252-day rolling mean |
| `oil_1d_change` | `_add_macro_features()` | Daily % change in Brent crude |
| `rate_level` | `_add_macro_features()` | US 10Y yield in % (e.g. 4.5 = 4.5%) |
| `rate_1d_change` | `_add_macro_features()` | Daily change in 10Y yield (pp) |

All cross-asset features fall back gracefully to neutral values (0 or 1) when data is unavailable so `dropna()` never removes rows due to missing reference data.

### 3. The Config File — `checkpoints/signal_model_config.json`

`ModelConfig` (a Pydantic model in `models/config.py`) is the contract between training and inference. It is written by `ModelTrainer` and read by `SignalGenerator` and `Backtester`. Pydantic validation raises a clear error at load time if any field is malformed or inconsistent.

Named model checkpoints live under `checkpoints/<name>/` (via `ModelConfig.checkpoint_paths(name)`). The default (unnamed) checkpoint lives under `checkpoints/`.

| Field | Type | Description |
|---|---|---|
| `feature_columns` | `list[str]` | Ordered list of feature column names the model was trained on |
| `feature_mean` | `list[float]` | Per-feature mean computed on the **training split only** |
| `feature_std` | `list[float]` | Per-feature std computed on the **training split only** |
| `sequence_length` | `int` | LSTM/GRU lookback window in bars (default 20) |
| `input_dim` | `int` | Number of features (= `len(feature_columns)`) |
| `interval` | `str` | Data interval (always `"1d"`) |
| `training_fetch_date` | `date` | Date training data was fetched |
| `holdout_start_date` | `date` | Earliest date the backtester is allowed to evaluate |
| `buy_threshold` | `float` | Min % gain to label Buy (default 0.015) |
| `sell_threshold` | `float` | Max % loss to label Sell (default -0.015) |
| `prediction_horizons` | `list[int]` | Prediction horizons in trading days (default [5, 10, 20]) |

**Adding a new feature requires retraining** — the config will be stale and `input_dim` will not match the model weights.

Consistency is validated at load time: `len(feature_columns) == len(feature_mean) == len(feature_std) == input_dim`.

### 4. Labeling — `models/training.py:ModelTrainer.prepare_data`

Labels use **max-return over each horizon window**, not end-of-window return, to reduce label noise from single-day randomness.

For each horizon `h`:
- **Buy** if the best price over the next `h` days exceeds `buy_threshold × sqrt(h / base_h)`
- **Sell** if the worst price falls below `sell_threshold × sqrt(h / base_h)` (and upside wasn't reached first)
- **Hold** otherwise; when both thresholds are crossed, the larger magnitude wins

Thresholds are optionally scaled by the stock's realized volatility relative to a 1% daily reference (`use_adaptive_thresholds=True` by default).

The price target regression head uses end-of-window return at the **middle horizon** (e.g. `h=10` for horizons `[5,10,20]`).

### 5. Sequence Creation — `models/signal_model.py:create_sequences`

Converts the normalized feature matrix into overlapping windows:
- Input: `features[n_rows, n_features]`
- Output: `X[n_samples, sequence_length, n_features]`, `y_signals[n_horizons, n_samples]`, `y_price[n_samples]`

**Data is never shuffled.** Temporal ordering is preserved throughout.

### 6. Model — `models/signal_model.py`

**`SignalModel`** — a multi-output Keras model with configurable backbone:

- **LSTM backbone** (default): LSTM → 4-head MultiHeadAttention → GlobalAveragePooling1D → Dropout → Dense → BatchNorm → Dropout
- **GRU backbone**: GRU → GlobalAveragePooling1D → Dropout → Dense → BatchNorm → Dropout (complementary to LSTM in an ensemble)

Outputs:
- **Signal heads**: one `softmax(3)` head per prediction horizon (named `signal_5d`, `signal_10d`, `signal_20d`)
- **Price head**: one `linear(1)` head predicting % price change at the middle horizon (`price_target`)

Loss: balanced focal loss (when `use_focal_loss=True`) with per-class alpha weights computed from training class distribution. Price head loss weight is 0.1.

**`EnsembleSignalModel`** — wraps one LSTM and one GRU `SignalModel`. Predictions are formed by averaging per-horizon softmax probabilities from both backbones before applying the majority-vote consensus. Public interface is identical to `SignalModel`.

**Consensus at inference**: `predict()` calls `predict_per_horizon()` then applies majority vote — a BUY or SELL signal requires ≥ `ceil(n_horizons/2 + 0.5)` horizons to agree; otherwise HOLD is emitted.

Class encoding: `BUY=0, HOLD=1, SELL=2` (defined in `models/direction.py`). This mapping is hardcoded at training time. Reference `BUY_IDX`, `HOLD_IDX`, `SELL_IDX` from `models/direction.py` — never hardcode 0/1/2.

### 7. Normalization at Inference Time

`SignalGenerator` and `Backtester` both apply:
```
features_norm = (features - feature_mean) / feature_std
```
where `feature_mean` and `feature_std` come from the config, **not** from the current data window. Using current-data statistics would introduce data leakage and degrade calibration.

`SignalGenerator` also performs an **OOD check**: if any feature in the most recent bar exceeds ±3σ of the training distribution, a warning is printed to stderr. Signal generation continues unaffected — the warning is advisory.

### 8. Calibration — `signals/calibration.py`

Two calibrator types:
- `ConfidenceCalibrator` — global isotonic regression (one calibrator for all directions)
- `DirectionalCalibrator` — separate calibrator per direction (BUY/SELL/HOLD); **preferred**

`SignalGenerator` loads `checkpoints/calibration_directional.json` first, then falls back to `checkpoints/calibration.json`, then to raw model probabilities.

Calibrators expose a `staleness_warning(max_days=30)` method used by `main.py` to warn when calibration is more than 30 days old.

Recalibrate after every retraining run (`uv run python main.py calibrate`).

### 9. Signal Generation — `signals/generator.py`

`SignalGenerator.generate(ticker)` combines model output into a `Signal` Pydantic model (validated at construction):

- **Stop-loss**: `current_price ± (atr_pct/100) × atr_multiplier` (default: 2.0× ATR)
- **Take-profit**: `current_price ± (atr_pct/100) × take_profit_atr_multiplier` (default: 3.0× ATR)
- **Position size**: half-Kelly capped at `max_position_size` (default 25%)
  - Kelly formula: `f* = (p×b − q) / b` where `p=confidence`, `b=gain/loss ratio`
  - Negative Kelly → position size 0 (no edge)
- **Confidence filtering**: signals below `min_confidence` (default 55%) are forced to HOLD
- **Portfolio risk limits**: suppresses trades when `portfolio_drawdown >= max_drawdown_pct` or `open_position_count >= max_positions`

`SignalGenerator.scan(tickers)` generates signals for multiple tickers and returns them sorted by confidence descending.

### 10. Backtesting — `backtesting/`

**`Backtester`** runs day-by-day over the holdout window. For each day it uses only data available up to that date. It calls `SignalModel.predict_per_horizon()` directly (not the consensus `predict()`) to capture per-horizon signal quality independently.

Key features:
- **Strict holdout enforcement**: backtest start must be ≥ `holdout_start_date` from config (prevents evaluating on training data)
- **Volume-based slippage**: `slippage_pct = slippage_factor / sqrt(relative_volume)`, capped at 0.5%
- **Commission**: one-way, default 0.1%
- **Leverage**: multiplier applied per trade
- **Position cooldown**: optionally skips next-horizon predictions after a non-HOLD trade
- **Walk-forward retraining**: optionally retrains on all data up to the current date every N trading days

**`PortfolioBacktester`** (`backtesting/portfolio.py`) runs the model across multiple tickers simultaneously with a shared capital pool:
- Fixed allocation per ticker: `1 / max_positions` of total capital
- One position per ticker at a time (no pyramiding)
- Commission applied on entry and exit

**`MetricsCalculator`** computes `HorizonMetrics` for each prediction horizon:
- Signal accuracy, per-class precision/recall/F1 (via `ClassMetrics`)
- Sharpe ratio, Sortino ratio, max drawdown, Calmar ratio
- Win rate with binomial p-value and Benjamini-Hochberg FDR correction
- Calibration analysis (Brier score, ECE, ROC AUC)
- Temporal and regime breakdowns (ADX filtering)
- Bootstrap confidence intervals
- Monte Carlo simulation (trade permutation — tests whether results are lucky orderings)

**Data structures** (`backtesting/results.py`):
- `HorizonPrediction` — one prediction for one day at one horizon; stores prediction and actual outcome
- `DailyPrediction` — all horizon predictions made on a single day
- `HorizonMetrics` — computed statistics for one horizon
- `ClassMetrics` — precision, recall, F1, support per direction class
- `MonteCarloResult` — distribution of total return, drawdown, and Sharpe across shuffled trade orderings
- `BacktestResult` — top-level container; exports CSV, JSON, equity curve

See [`backtesting/STRATEGY.md`](backtesting/STRATEGY.md) for how to interpret the output and what thresholds make a result trustworthy.

### 11. Training Pipeline — `models/training.py`

`ModelTrainer.train(tickers, epochs, ...)` orchestrates the full training run:

1. Fetch OHLCV + cross-asset data for each ticker (10y history)
2. `prepare_data()` per ticker → features, labels, price changes, date index
3. Apply holdout cutoff (discard data ≥ `holdout_date`)
4. Normalize: compute mean/std on each ticker's training split only, then pool — avoids cross-ticker leakage (M6 fix)
5. `create_sequences()` → X, y_signals (per horizon), y_price
6. 70/15/15 time-series split (no shuffling)
7. Build `SignalModel` with balanced focal loss; optionally also build GRU backbone for `EnsembleSignalModel`
8. Train with `ModelCheckpoint` + `EarlyStopping(patience=10)` + `ReduceLROnPlateau`
9. Save weights + `ModelConfig` JSON
10. Log everything to MLflow (Keras autolog + manual metrics)

`HyperparameterTuner` in the same file runs random search over `sequence_length`, `buy_threshold`, `batch_size`, and `use_focal_loss`, with each trial logged as a nested MLflow run.

`WalkForwardTrainer` (`models/walk_forward.py`) implements expanding-window cross-validation with configurable purge and embargo gaps.

---

## Module Dependency Graph

```
main.py
  ├── data/fetcher.py
  ├── data/features.py
  ├── models/config.py              (Pydantic ModelConfig; no internal deps)
  ├── models/direction.py           (Direction enum, BUY/HOLD/SELL_IDX; no internal deps)
  ├── models/signal_model.py        → models/direction.py, models/losses.py
  ├── models/losses.py              (no internal deps)
  ├── models/training.py            → data/, models/signal_model.py, models/config.py,
  │                                    models/direction.py, models/mlflow_tracking.py
  ├── models/walk_forward.py        → models/training.py
  ├── models/mlflow_tracking.py     (no internal deps beyond mlflow)
  ├── signals/generator.py          → data/, models/config.py, models/direction.py,
  │                                    models/signal_model.py, signals/calibration.py
  ├── signals/calibration.py        (no internal deps)
  ├── backtesting/backtester.py     → data/, models/config.py, models/signal_model.py,
  │                                    backtesting/metrics.py, backtesting/results.py
  ├── backtesting/portfolio.py      → backtesting/backtester.py, backtesting/results.py,
  │                                    models/config.py
  ├── backtesting/metrics.py        → backtesting/results.py
  ├── backtesting/results.py        → models/direction.py
  └── backtesting/plot.py           → backtesting/results.py
```

`signals/` and `backtesting/` are independent of each other. `backtesting/` must not import from `signals/`.

---

## Checkpoint File Layout

```
checkpoints/                         # default (unnamed) model
  signal_model.weights.h5            # LSTM weights
  signal_model_gru.weights.h5        # GRU weights (ensemble only)
  signal_model_config.json           # ModelConfig (training → inference contract)
  calibration.json                   # Global ConfidenceCalibrator
  calibration_directional.json       # DirectionalCalibrator (preferred)

checkpoints/<name>/                  # named model (--name flag)
  signal_model.weights.h5
  ...
```
