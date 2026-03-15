# Changelog

Format: one entry per meaningful task completion. Add to the top. Each entry should explain what changed and why — not just what files were touched.

---

## 2026-03-16

### Auto-calibration after training
Confidence calibration now runs automatically at the end of `train` (standard mode only, not walk-forward). After weights are saved, the trainer backtests all training tickers over the holdout period at horizon 5d and fits a `ConfidenceCalibrator`. Previously calibration was a separate manual step that was easy to forget, meaning signals were used with uncalibrated confidence scores after every retrain. Use `--no-calibrate` to skip. The core calibration logic was extracted into `_run_calibration()` so both `train` and `calibrate` share the same implementation.

### Leverage simulation in backtester
Added `leverage: float = 1.0` parameter to `Backtester` and `MetricsCalculator`. Returns and transaction costs scale with the leveraged notional. New CLI flags on `backtest`: `--leverage 2.0` for a fixed multiplier, `--compare-leverage` to run 1x/2x/3x in one shot and print a side-by-side comparison table of net return, Sharpe, max drawdown, and win rate per horizon. Motivation: leverage is a key risk/return dial for live trading — the comparison table makes the optimal leverage/risk tradeoff immediately visible without multiple manual runs.

### Training progress logging
Added three print statements to `ModelTrainer.train()` to surface the previously silent data preparation phase: combining ticker data, normalizing features, and sequence creation. Sequence creation (~40k sequences for the 1h model) was taking 10+ minutes with no output, making it impossible to distinguish a running job from a hung one.

### Skip empty tickers in training
`ModelTrainer.train()` now skips tickers that return 0 usable samples (e.g. `^OMX` on hourly interval) instead of crashing with an index error when computing `latest_val_date`. Prints a clear `"Skipping {ticker}: no usable samples"` message.

---

## 2026-03-15

### Pydantic validation for ModelConfig and Signal
`ModelConfig` (`models/config.py`) replaces the hand-rolled `json.load()` dict pattern used by `SignalGenerator`, `Backtester`, and `ModelTrainer`. It validates field presence, types, and invariants (positive std, consistent dimensions) at load time — a malformed checkpoint now raises a clear error immediately rather than crashing deep in prediction code. `ModelConfig.load()` and `.save()` replace all three callers' ad-hoc JSON handling.

`Signal` in `signals/generator.py` is now a Pydantic `BaseModel` instead of a `@dataclass`. Validators enforce confidence in 0–100, position_size in 0–1, positive prices, and directionally-consistent stop-loss/target-price (BUY stop below entry, target above; SELL the reverse). This catches internal logic errors in `generate()` immediately rather than producing silently wrong signals.

### Expanded MLflow logging
Three additions to MLflow tracking, all keeping the existing logging alongside:

- **Per-class precision/recall in backtest runs** — `h{N}.buy_precision`, `h{N}.buy_recall`, `h{N}.sell_precision`, `h{N}.sell_recall`, `h{N}.hold_precision`, `h{N}.hold_recall`, plus `h{N}.trade_count` and `h{N}.win_rate_pvalue`. Previously only overall accuracy and win rate were queryable in MLflow; now class-level regressions are detectable across runs without re-running the backtest.

- **Training data class distribution** — `class_pct_buy`, `class_pct_sell`, `class_pct_hold` logged as params on every training run. Makes it possible to correlate model behaviour changes with shifts in the training data balance.

- **`holdout_start_date` tag on training runs** — logged as a tag so every training run records exactly what period was kept out. Makes it easy to filter in MLflow UI and verify that backtests respect the boundary.

- **Keras autologging** — `mlflow.keras.autolog()` enabled before training. Captures all Keras metrics automatically including system metrics. The existing manual `log_training_history` call is kept alongside for backwards compatibility.

### Multi-interval model support
Added `--interval` flag to `train`, `backtest`, and `calibrate` commands. Each interval (`1d`, `1h`) trains a completely independent model with its own checkpoint, config, and calibration file. Sequence length and prediction horizon scale automatically with the interval — 1h defaults to 140-bar lookback and 35-bar prediction horizon (~5 trading days in hourly bars). Motivation: daily data captures medium-term momentum well; hourly data may improve short-term entry timing. The two models can be used independently or combined for signal confirmation.

### Strict holdout fix
The `holdout_start_date` saved in the model config was previously computed from a hardcoded formula (today minus 15% of 5 years), which drifted from the actual data split. It now records the true last calendar date seen in the training+validation split across all tickers. The backtester enforces this boundary automatically and prints a clear warning when it had to adjust the start date.

### Cross-asset features wired into training pipeline (F2)
`omxs30_corr`, `usdsek_ret`, and `eursek_ret` were implemented in `FeatureEngineer` but always computed as zero because no reference data was passed at training, backtesting, or signal generation time. Fixed by fetching OMXS30, USD/SEK, and EUR/SEK via `StockDataFetcher.fetch_cross_asset_data()` and passing the result to `FeatureEngineer` in all three callers. The model now actually trains on real cross-asset signal.

### Evaluation across four tickers (March 2026 holdout)
Backtested on ERIC-B, HM-B, VOLV-B, and SEB-A over the clean holdout period (2025-09-17 to 2026-03-14, 122 days). SEB-A (+0.7% buy & hold → +128% net at 7d) is the most convincing result. Consistent finding: 5-7 day horizons are statistically significant (p<0.01) across all tickers; 1-day is unreliable after costs. Results documented in `backtesting/STRATEGY.md`.

### Ticker scope decision (M2)
Confirmed the model is ticker-specific — it predicts pure HOLD on unseen stocks. Accepted as an intentional design constraint rather than a bug. To add a ticker to the universe, include it in training and retrain. Documented in `backtesting/STRATEGY.md`.

### Type annotation improvements
Closed the most agent-impactful typing gaps across the codebase. All changes pass mypy and ruff clean.

- **`backtesting/metrics.py`**: Added `TradingMetrics(TypedDict)` covering all 11 trading metric fields. `_calculate_trading_metrics` now returns `TradingMetrics` instead of a bare `dict` — agents consuming this no longer have to read the implementation to know what keys exist.
- **`backtesting/results.py`**: Tightened `equity_curve: list[tuple]` → `list[tuple[date, float]]` to match the TypedDict and make the pair structure explicit.
- **`models/losses.py`**: Both factory functions now declare `-> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]`. Inner loss functions annotated with TF tensor types. `list | None` → `list[float] | None`. Agents adding new loss functions now have a typed pattern to follow.
- **`models/training.py`**: Added `feature_mean: np.ndarray | None` and `feature_std: np.ndarray | None` to `__init__` so the full object shape is visible without reading `prepare_data()`. Added asserts before `.tolist()` calls — surfaced a latent bug where calling `train()` without `prepare_data()` first would crash with an unhelpful `NoneType` error.
- **`models/walk_forward.py`**: `class_distribution: dict` → `dict[int, int]`.
- **`signals/generator.py`, `backtesting/backtester.py`**: All private loader methods (`_load_config`, `_load_calibrator`, `_load_model`) annotated `-> None`.

### Project documentation (ARCHITECTURE.md, INVARIANTS.md, module docstrings)
Added three deliverables to improve agent orientation in the codebase:
- **`ARCHITECTURE.md`**: Full pipeline diagram from fetcher to signal/backtest result, stage-by-stage data shapes, config file format table, and module dependency graph.
- **`INVARIANTS.md`**: Nine named rules that must always hold — normalization stats, ATR encoding, class ordering, no-shuffle, holdout discipline, calibration lifecycle, focal loss + sample weights conflict, module isolation, and reference data alignment. Each rule includes why it exists and where it's enforced.
- **Module `__init__.py` docstrings**: All four modules (`data/`, `models/`, `signals/`, `backtesting/`) now have docstrings covering ownership, public API, boundary rules, and cross-references to ARCHITECTURE.md and INVARIANTS.md.

### Cross-reference audit (markdown files)
Fixed four gaps in how the project's markdown files referenced each other:
- AGENTS.md now links to CONTRIBUTIONS.md and STRATEGY.md at the top, and its finish protocol includes pytest, backtest validation, and CHANGELOG steps.
- CLAUDE.md's duplicated backtest output guide (copy of STRATEGY.md) replaced with a single pointer — one source of truth.
- README.md Contributing section now references both AGENTS.md and CONTRIBUTIONS.md.
- CLAUDE.md key-documents table added so agents see the full map in one place.

## 2026-03-14

### Per-direction calibration (R5)
Added `DirectionalCalibrator` to `signals/calibration.py` — holds a separate `ConfidenceCalibrator` per direction (BUY/SELL/HOLD), each fitted and serialised independently. Updated `SignalGenerator` to load and prefer the directional calibrator over the global one; falls back to the global calibrator if a direction has no data, and falls back to raw model output if neither is available. The global calibrator was treating all directions as homogeneous, which masks the fact that the model can be systematically over-confident on BUY signals but well-calibrated on SELL. Findings during work: the `calibrate` CLI command still only fits the global calibrator (R7 added), and `DirectionalCalibrator` has no diagnostic display method yet (R8 added).

### Equity curve export
Added `equity_curve: list[tuple[date, float]]` to `HorizonMetrics` — a chronologically sorted series of cumulative net returns per trade. `BacktestResult.export_equity_curve_csv(path, horizon)` writes it to a two-column CSV for plotting. Previously there was no way to see whether returns were evenly distributed over time or front-loaded into early data.

### Unit test suite (backtesting + walk-forward)
Added 52 unit tests across four new test files covering `MetricsCalculator` (accuracy, precision/recall, calibration, MAE/RMSE, Sharpe formula, drawdown, Calmar, Sortino, trading P&L, commission, slippage), `Backtester` (holdout clamping, outcome filling, relative volume), volume-based slippage (cap enforcement, low-volume penalty), and `WalkForwardTrainer` (purge/embargo gap logic, no-overlap invariants). Raises coverage on the backtesting module from ~12% to well above 80%.

### Strict holdout split (B3)
Enforced that the backtest period never overlaps training data. Training is now capped at the earliest date in the backtest window, checked in both the standard pipeline and walk-forward. Without this, backtest metrics were inflated because the model had already seen the evaluation data.

### Slippage modeling (B4)
Added volume-based slippage to the backtester. Execution price is now adjusted by a fraction of the bid-ask spread estimated from daily volume, making net return figures more realistic for illiquid Swedish stocks.

### Statistical significance on win rate
Win rate in backtest output now includes a one-sided binomial p-value (`**` p<0.01, `*` p<0.05). Results with fewer than 30 trades are flagged with `⚠`. This makes it easier to distinguish genuine edge from noise in short backtest windows.

### Market regime indicators (F1)
Added ADX and volatility regime features to the feature pipeline. These give the model signal about whether the market is trending or ranging, which matters for whether momentum-based indicators are reliable.

### Cross-asset features (F2)
Added OMXS30 index correlation, USD/SEK, and EUR/SEK as features. Swedish stocks are significantly exposed to currency moves and broad index direction.

### Calendar effects (F3)
Added day-of-week and month-of-year features. Captures known seasonality effects (e.g. Monday effect, year-end flows) without requiring the model to infer them from price alone.

### Purged cross-validation (M1)
Added a gap between training and validation folds in walk-forward cross-validation to prevent label leakage from overlapping sequences. Without the gap, adjacent windows share data points and validation accuracy is optimistic.

### MLflow backtest logging + history command (I2)
Backtest results are logged to MLflow automatically. The `history` CLI command queries MLflow and renders a trend table across runs, making regressions immediately visible.

### MLflow experiment tracking consolidation (I1)
All runs land in a single `trading-signals` experiment, tagged by `run_type`. CLI args are logged as `cli.*` tags so every run is fully reproducible from its MLflow record.

---

## Earlier

### ATR stop-loss, Kelly sizing, take-profit targets (R1–R3)
Stop-loss levels, take-profit targets, and position sizes are computed at signal generation time using ATR multiples and Kelly criterion scaled by confidence.

### Confidence calibration (signals/calibration.py)
Added isotonic regression calibration so raw softmax probabilities map to real accuracy rates. Raw model outputs were not well-calibrated — a 70% confidence signal was not winning 70% of the time.

### Walk-forward training (models/walk_forward.py)
Added walk-forward training that slides a fixed window forward through time, retraining at each step. Each window is a nested MLflow child run. More realistic than a single train/test split.

### Focal loss (models/losses.py)
Replaced standard cross-entropy with focal loss as the default. Addresses class imbalance — most days are HOLD, which biased the model toward predicting HOLD on ambiguous inputs.

### Backtesting metrics (B2)
Added Sharpe, Sortino, max drawdown, and Calmar ratio to backtest output.

### Transaction costs (B1)
Added configurable commission (default 0.1% per trade). Net return is reported after costs.
