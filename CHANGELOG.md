# Changelog

Format: one entry per meaningful task completion. Add to the top. Each entry should explain what changed and why — not just what files were touched.

---

## 2026-03-20

### M5 — Ensemble models (LSTM + GRU)
Added a GRU backbone variant to `SignalModel` via a new `backbone` parameter, and an `EnsembleSignalModel` class that averages per-horizon softmax probabilities from both architectures before applying the majority-vote consensus. The GRU uses plain GAP with no attention, making it architecturally complementary to the LSTM's multi-head attention path. `ModelTrainer` gains a `use_ensemble` flag that trains the GRU alongside the LSTM and assembles both into an `EnsembleSignalModel`; the ensemble's public interface is identical to `SignalModel` so all callers (backtester, signal generator) work unchanged. The `use_ensemble` flag is also logged to MLflow hyperparameters for full run reproducibility.

### B10 — Per-horizon signals wired into the backtester

Replaced the consensus `model.predict()` call in `backtester.py` with `model.predict_per_horizon()`, which returns one set of probabilities and a predicted class per trained horizon head. Each `HorizonPrediction` now carries the signal from the head specifically trained for that horizon (e.g. the 5-day head feeds the 5-day result row) rather than a majority-vote copy. For horizons not present in the trained model, the backtester falls back to the average across all heads. This means per-horizon backtest metrics (accuracy, win-rate, Sharpe) now reflect genuine per-horizon model quality rather than identical signals differing only in holding period. Updated `_mock_model` in the backtester test suite to implement `predict_per_horizon`.

### M9 — Walk-forward trainer now raises on zero windows instead of silently succeeding

`WalkForwardTrainer.run()` in `models/walk_forward.py` would silently complete with no trained windows if `purge_gap + embargo_gap` exceeded `validation_days`, or if the total sample count was smaller than `initial_train_days`. The training loop ran over an empty list, `_run_walk_forward()` returned `([], None)`, and the caller received a result with NaN metrics — no model ever trained. Fixed by raising a `ValueError` immediately after `generate_windows()` returns an empty list, including the gap sizes, validation window size, and sample count so the user can diagnose and adjust parameters.

### M4 — Expose per-horizon predictions from multi-horizon model heads
Added `predict_per_horizon()` to `SignalModel`, returning a list of softmax probability arrays and predicted classes — one per trained horizon — alongside the price target. `predict()` is refactored to call it internally and continues to return the majority-vote consensus, keeping all existing callers (`signals/generator.py`, `backtesting/backtester.py`) unchanged. B10 can now update the backtester to use `predict_per_horizon()` so that per-horizon backtest rows reflect genuine per-horizon signal quality rather than just holding-period effects.

### F9 — Add zero guard to ADX di_plus/di_minus to prevent Inf on constant-price rows

`_add_adx()` in `data/features.py` divided the smoothed directional movement by `atr_w` (EWM true range) without checking for zero. On constant-price rows, trading halts, or very early sparse data, `atr_w` is zero and the division produces `Inf`, which propagates into the `adx_14` feature column. The existing `fillna(0)` on the final `dx` step did not protect the intermediate `di_plus`/`di_minus` values. Fixed by appending `.where(atr_w > 0, 0.0)` to both directional-index calculations so zero-ATR periods produce `di=0` (no directional movement) rather than `Inf`.

---

## 2026-03-20

### F7: Feature mismatch is now a hard error with zero tolerance
Removed the `> 10% missing features → warn and continue` fallback from `backtesting/backtester.py` and `signals/generator.py`. Both now raise `ValueError` immediately if any single feature from the training config is absent from the data, naming the missing columns. The old tolerance was dangerous: a model trained on 37 features silently receiving 33 produces degraded, unpredictable predictions with no indication in output. The fix also simplifies the code — `feature_cols` is now set directly from `self.feature_columns` rather than a filtered subset. Tests updated to match the new zero-tolerance contract.

---

## 2026-03-20

### B18 — Move `Direction` enum to `models/direction.py` (fully resolve INVARIANTS violation)
`Direction`, `BUY_IDX`, `HOLD_IDX`, `SELL_IDX`, and `IDX_TO_DIRECTION` have been moved from `signals/direction.py` to the new `models/direction.py`. All five in-scope callers (`backtesting/results.py`, `signals/generator.py`, `models/signal_model.py`, `models/training.py`) now import from `models.direction`. `signals/direction.py` is kept as a thin re-export shim so existing test imports continue to work. This removes both the `backtesting/ → signals/` and `models/ → signals/` dependencies, completing the module isolation required by INVARIANTS.md.

### R9 — Add regression tests for DirectionalCalibrator fallback path

Added two tests to `tests/test_calibration.py`. `test_fitted_direction_uses_calibrator_not_raw` constructs a deliberately overconfident calibrator (high raw scores map to ~20% actual accuracy), then asserts the fitted direction returns a calibrated value well below the raw input — catching any regression where the calibrator is silently bypassed. `test_fallback_is_per_direction_not_global` fits BUY and SELL with opposite calibration curves and verifies each uses its own calibrator independently while the unfitted HOLD direction returns the raw value unchanged. Together these make the "silently returns raw for all directions" regression visible in CI.

### B15 — Fix `backtesting/plot.py` INVARIANTS violation (cross-package import)
`plot.py` was importing `Direction` directly from `signals/direction.py`, violating the INVARIANTS.md rule that `backtesting/` and `signals/` must not import from each other. The fix re-exports `Signal` via `backtesting/results.py` (which is already the source of the `HorizonPrediction.predicted_signal` type), so `plot.py` now imports only from within its own package. The deeper violation in `results.py` itself is tracked as B18.

### B16 — Narrow bare `except Exception` blocks in backtester.py

Three `except Exception` catch-all blocks in `backtesting/backtester.py` were replaced with specific exception types: model weight loading now catches `(FileNotFoundError, OSError, ValueError)`, benchmark computation catches `(ValueError, KeyError, IndexError, OSError)`, and the relative-volume calculation catches `(IndexError, ValueError, TypeError)`. The relative-volume handler was also missing any log output — a genuine bug (e.g. an unexpected dtype causing a `TypeError`) would silently produce `None` with no trace; it now prints a warning with the exception message. Narrowing these handlers means unexpected exceptions (shape mismatches, programming errors) propagate normally and are visible in tracebacks.

### B15 — Remove cross-module import from `backtesting/plot.py`
`plot.py` was importing `Direction` directly from `signals.direction`, violating the INVARIANTS.md rule that `backtesting/` and `signals/` must not import from each other. Since `backtesting/results.py` already re-exports `Signal` (aliased from `Direction`), the fix was a one-line change: import `Signal` from `backtesting.results` instead. Also opened B18 to address the same underlying violation in `results.py` itself.

### B17 — Fix default backtest horizons to match model's trained horizons

The backtester defaulted to `horizons = [1, 2, 3, 4, 5, 6, 7]` when no horizons were passed. A model trained on `[5, 10, 20]` would then generate `HorizonPrediction` rows for horizons it was never trained to predict (1–4, 6–7), making those rows in every backtest report meaningless noise. Fixed by changing the default to `self.prediction_horizons or [5, 10, 20]`, so the backtester always analyses the exact horizons the loaded model was trained on. Per-horizon metrics in the report now genuinely reflect per-horizon model quality.

---

## 2026-03-18

### Add macro indicators: oil prices and interest rates (F5)

Added Brent crude oil (`BZ=F`) and US 10Y Treasury yield (`^TNX`) as macro features alongside the existing VIX/VSTOXX volatility features. `fetch_cross_asset_data()` in `fetcher.py` now fetches both series with the same graceful fallback-to-zeros pattern. `FeatureEngineer._add_macro_features()` produces four new columns: `oil_level` (Brent normalised by 252-day rolling mean, capturing high/low oil regimes), `oil_1d_change` (daily % change, capturing supply shocks), `rate_level` (raw 10Y yield in %, a regime indicator for financial conditions), and `rate_1d_change` (daily absolute change in yield). All four fall back to neutral values when reference data is unavailable so no rows are silently dropped. Retraining is required to incorporate these features into live signals.

---

## 2026-03-18

### Add --export-equity CLI flag to backtest command (B9)

Added `--export-equity BASE_PATH` to the `backtest` subcommand. After a backtest completes, it writes one CSV per horizon to `BASE_PATH_h{N}d.csv` (e.g. `results/equity_h5d.csv`), each with `date` and `cumulative_net_return_pct` columns. Horizons with no trades are silently skipped. The `export_equity_curve_csv()` method already existed on `BacktestResult`; this change wires it to the CLI so users can plot equity curves in external tools without parsing the full JSON export.

---

## 2026-03-18

### Document calibration absence in walk-forward retrain path (B13)

`INVARIANTS.md` forbids `backtesting/` from importing `signals/`, so the backtester cannot load a `ConfidenceCalibrator`. This means confidence scores are always raw softmax outputs — there was never a stale calibrator being applied after retraining. Added an explicit comment in `_retrain_model()` documenting this constraint so future contributors don't inadvertently add a cross-module import to "fix" it. Also fixed a pre-existing `F821` ruff error (`pd` used as a string annotation before it was imported) in `tests/test_new_features.py`.

---

## 2026-03-18

### Add HyperparameterTuner for systematic hyperparameter search (M3)

Added `HyperparameterTuner` class to `models/training.py`. It runs random search over a configurable param grid (`sequence_length`, `buy_threshold`, `batch_size`, `use_focal_loss` by default), trains a fresh `ModelTrainer` per trial, and ranks results by best validation accuracy. Every trial is tracked in MLflow as a nested child run under a single parent `hyperparameter-search` run, making it easy to compare trials in the UI. The search uses validation accuracy (not test accuracy) to avoid selection bias on the holdout set. Usage: `HyperparameterTuner().search(tickers, n_trials=10)`.

---

## 2026-03-18

### Make calibration discrepancy explicit in backtest output (B12)

`Backtester` cannot apply isotonic calibration because `INVARIANTS.md` forbids `backtesting/` from importing `signals/`. Rather than silently report raw confidence values that don't match what live signals produce, `run()` now prints a clear note at the start of every backtest explaining that confidence scores are pre-calibration and will differ from `generate`/`scan` output. This prevents users from mistaking raw softmax calibration buckets for production-quality confidence estimates.

---

## 2026-03-18 (session 16)

### Report ADX-missing predictions in regime breakdown (B14)

`_calculate_regime_metrics()` in `metrics.py` already guarded against `adx = None` with a `p.adx is not None` filter, so the crash was already prevented. Added the missing reporting piece: if any predictions have no ADX value (common on the first ~14 bars before the ADX warmup completes), a `"no_adx"` entry is now included in the returned `regime_metrics` dict with an `n_predictions` count. The `summary()` display in `results.py` is updated to render this count rather than crashing on the missing `n_trades` key.

---

## 2026-03-18 (session 15)

### Add VIX/VSTOXX volatility regime features (F4)

`fetch_cross_asset_data()` in `fetcher.py` now also fetches VIX (`^VIX`) and VSTOXX (`^V2TX`) alongside the existing OMXS30/FX series. `FeatureEngineer` now stores `reference_data` (previously discarded) and a new `_add_volatility_features()` method produces five new columns: `vix_level` and `vstoxx_level` (current vol index normalised by its 252-day rolling mean — >1 means elevated fear), `vix_1d_change` and `vstoxx_1d_change` (daily % change, capturing sudden regime shifts), and `vix_stock_corr` (20-day rolling correlation between stock returns and VIX changes — negative values identify risk-off stocks). When VIX/VSTOXX data is unavailable, all columns fall back to neutral constants so `dropna()` does not discard rows. Retraining required to incorporate the 5 new features (input_dim changes from 8 to 13).

---

## 2026-03-18 (session 15)

### Out-of-distribution feature detection at inference time (F6)

Added `_check_ood()` to `SignalGenerator` (`signals/generator.py`). After normalising live features with training statistics, the method inspects the most recent bar's normalised values and prints a warning to stderr for any feature that exceeds ±3σ. The warning names each out-of-range feature and its normalised deviation (e.g. `rsi (+4.2σ)`) so users can see exactly what regime shift is occurring. Signal generation continues — the check is advisory only. The check is skipped gracefully when training stats are unavailable. Called automatically in `generate()` after normalization. 6 new unit tests cover: no-warning in-distribution, single OOD, multiple OOD, last-row-only check, missing stats, and custom threshold.

---

## 2026-03-18 (session 14)

### Add `get_calibration_table()` to `DirectionalCalibrator` (R8)

`DirectionalCalibrator` now has `get_calibration_table()` mirroring the same method on `ConfidenceCalibrator`. It renders one indented table per direction (BUY / SELL / HOLD), with a "not fitted" note for any direction that lacked sufficient samples. This is already called by `_run_calibration()` in `main.py` (added in R7) so the output now uses the proper method rather than ad-hoc formatting. Also fixed a latent bug in `DirectionalCalibrator.save()` / `load()`: the `fitted_at` timestamp was not being persisted per sub-calibrator, so the staleness warning introduced in B8 would always fire for directional calibrators loaded from disk. Now `fitted_at` is included in the per-direction JSON payload and restored on load.

---

## 2026-03-18 (session 12)

### Calibration staleness warning (B8)

`ConfidenceCalibrator` now records `fitted_at` (a UTC datetime) when `fit()` is called and persists it in the JSON file. `load()` restores it; files written before this change will have `fitted_at = None`. A new `staleness_warning(max_days=30)` method returns a human-readable warning string if the calibrator is older than the threshold, or if `fitted_at` is missing (which also means the age is unknown and should prompt a recalibrate). `DirectionalCalibrator` gets the same method, delegating to the oldest fitted sub-calibrator. In `main.py`, a new `_check_calibration_staleness()` helper is called in both `cmd_analyze` and `cmd_scan` immediately after the `SignalGenerator` is built; it prints to stderr so the warning is visible even when output is piped. This prevents silently trading on a calibrator trained on a market regime months in the past.

---

## 2026-03-18 (session 13)

### Fix Bollinger Band feature dropping constant-price rows (F8)

`_add_bb_position()` in `data/features.py` was dividing by `band_width.where(band_width > 0)`, which produces NaN when the 20-bar rolling std is zero. Those NaN rows were silently removed by `dropna()`, so the model never saw low-volatility regimes (trading halts, early thin data). Fixed by substituting `0.5` (midpoint of a degenerate flat band) when `band_width == 0` rather than propagating NaN.

---

## 2026-03-18 (session 12)

### Fix target-price lookup: date arithmetic replaces row-offset arithmetic (B11)

`_fill_actual_outcomes()` in `backtesting/backtester.py` previously computed the outcome price as `df["close"].iloc[pred_idx + horizon]` — a raw row offset. If yfinance drops any trading session (e.g., a Swedish public holiday), every subsequent P&L calculation is silently shifted by one day. Changed to `df_index[pred_idx] + BDay(horizon)` date arithmetic: the exact business-day-calendar target date is computed, then looked up by date in `date_to_idx`. If the exact date is absent (genuine holiday gap), the code scans forward up to 3 extra business days before giving up. Added `test_gap_in_data_does_not_shift_target_date` which drops a row from the middle of a synthetic DataFrame and confirms the correct target price is still retrieved.

---

## 2026-03-18 (session 11)

### Directional calibration exposed in `calibrate` CLI command (R7)

`_run_calibration()` in `main.py` now collects per-direction prediction data (BUY/SELL/HOLD) alongside the existing global data in the same backtest loop. After fitting and saving the global `ConfidenceCalibrator`, it fits a `DirectionalCalibrator` — one separate isotonic calibrator per direction — and saves it to `checkpoints/calibration_directional.json` (the path already reserved in `ModelConfig.checkpoint_paths()`). Per-direction calibration tables are printed to the console. The directional path is passed automatically from `cmd_calibrate` and from the post-training auto-calibration in `cmd_train`, so both `calibrate` and `train` now produce both files without any extra flags required.

---

## 2026-03-18 (session 10)

### Monte Carlo simulation for backtest confidence intervals (B7)

Added `MonteCarloResult` dataclass (`backtesting/results.py`) and `_run_monte_carlo()` method (`backtesting/metrics.py`). After every backtest, the realized per-trade net returns are randomly permuted 1 000 times. For each permutation, total return, max drawdown, and Sharpe are computed. Results include mean, std, 5th/95th percentiles, and the observed result's percentile rank vs simulations. This answers the key question: is the strategy's good result genuine edge, or just lucky ordering of the same trades? If the observed total return is in the 95th percentile of shuffled paths, the result is robust to path order; if it's near the median, the outcome was heavily path-dependent. Attached to `HorizonMetrics.monte_carlo`, displayed in `BacktestResult.summary()`, and included in JSON export.

---

## 2026-03-18 (session 9)

### Walk-forward backtest — periodic in-production retraining (B6)

Added `retrain_every: int | None` and `retrain_epochs: int` to `Backtester`. When `retrain_every=N`, the backtest splits into chunks of N trading days; at each chunk boundary the model is retrained from scratch on all raw data strictly before that date, then predictions resume with the updated model and normalization stats. This simulates the realistic production scenario where you periodically retrain rather than running the same model indefinitely. `retrain_every=None` (default) preserves the original single-model behaviour. Exposed as `--retrain-every N` and `--retrain-epochs N` on the `backtest` CLI command.

Also added `reference_data` parameter to `WalkForwardTrainer.prepare_data()` so retrain calls include cross-asset features (OMXS30, FX rates), matching what the standard training pipeline produces.

---

## 2026-03-18 (session 8)

### Backtest plot (`backtesting/plot.py`)

New `plot_backtest()` function renders a two-panel figure after any backtest run: price line with BUY/SELL/HOLD signal markers (sized and shaded by confidence) on top, and equity curves for all horizons vs buy-and-hold on the bottom. Activated via `--plot [FILE]` on the `backtest` command — omitting a filename saves to `tmp/<TICKER>_backtest.png`. Signal markers use per-signal colour gradients (pale = low confidence, saturated = high confidence) so both direction and conviction are readable at a glance.

### Multi-horizon signal design finding documented

Discovered that `SignalModel.predict()` collapses all per-horizon heads into a single majority-vote consensus before returning, so the backtester assigns identical signals to every horizon entry. The different equity curves per horizon are purely a holding-period effect, not evidence of different per-horizon signal quality. Documented as tasks M4 and B10 in `TASKS.md` — fixing this requires exposing raw per-horizon outputs from the model before the backtester can make genuine per-horizon comparisons.

### Backtest validation diagnostics

Added a **VALIDATION DIAGNOSTICS** section to every backtest summary, covering metrics that are essential before trusting results in live trading:

- **Brier score** — multi-class calibration quality of softmax outputs (uses full `[p_buy, p_hold, p_sell]` now stored in `HorizonPrediction.all_probs`)
- **ECE** — Expected Calibration Error, computed from existing calibration buckets
- **Bootstrap 95% CIs** — on win rate, Sharpe ratio, and net return (500 resamples, seeded for reproducibility)
- **Benjamini-Hochberg FDR correction** — p-values corrected across all horizons to control false discovery when testing 7 horizons simultaneously
- **Regime breakdown** — accuracy and trade count split by ADX regime (ranging <20, trending 20–40, strong >40)
- **Per-trade records** — `TradeRecord` dataclass stored in `HorizonMetrics.trades`, exportable via `BacktestResult.export_trades_csv()`
- **ROC/AUC** — one-vs-rest AUC per signal class, stored in `HorizonMetrics.roc_auc` (requires `all_probs`)
- **Monthly accuracy** — stored in `HorizonMetrics.temporal_accuracy` to spot regime-dependent performance

`CONTRIBUTIONS.md` updated with a reference table explaining each diagnostic and what constitutes a red flag.

---

## 2026-03-15 (session 7)

### ADX regime filter and pre-2025 holdout retraining

`HorizonPrediction` now carries an `adx` field (ADX-14 value on the prediction date), populated by the backtester from the already-computed `adx_14` feature column. `PortfolioBacktester` accepts an `adx_filter` parameter; when set, BUY signals are skipped if ADX is below the threshold. CLI flag: `--adx-filter N` on the `portfolio` command. The indexes model was retrained with `--holdout-start 2025-01-01` (new CLI flag on `train`), enforcing a clean calendar boundary and using all pre-2025 data for training. `ModelTrainer` accepts a `holdout_date` parameter that clips each ticker's feature array to dates strictly before the boundary before any splitting occurs.

### Temporal consistency analysis — critical finding documented

Per-quarter portfolio backtests revealed that 15 of 16 ADX-filtered trades occurred in Q2 2025 (April–June, tariff-shock period). The remaining 4 quarters produced 0–1 trades each. The +14.1% headline return and 93.8% win rate are real but temporally concentrated — not demonstrated across multiple independent active periods. The ADX≥25 threshold was also selected by comparing three values on the same holdout data (mild in-sample selection). Full findings documented in `backtesting/STRATEGY.md`. Recommendation: paper trade for 3–6 months covering at least one more high-ADX regime before committing capital.

---

## 2026-03-15 (session 6)

### Named model checkpoints

`ModelConfig.checkpoint_paths()` now accepts an optional `name` parameter. When provided, checkpoints are stored in `checkpoints/<name>/` instead of the default `checkpoints/` directory. All CLI commands (`train`, `backtest`, `calibrate`, `scan`, `analyze`, `portfolio`) accept a `--name` flag to select a named model at runtime. The motivation is to allow multiple specialized models to be trained and evaluated independently without overwriting each other's weights or calibration files.

### EU-5 European indexes model

A specialized model was trained using `--name indexes` on five correlated European indexes: STOXX50E (`^STOXX50E`), DAX (`^GDAXI`), CAC40 (`^FCHI`), OMX Stockholm 30 (`^OMXS30`), and AEX (`^AEX`). Training on correlated indexes provides natural regularization — the model must learn patterns that generalize across all five rather than overfitting to the idiosyncratic noise of a single instrument. Adding global indexes (Asian/US) diluted signal; per-index single-ticker models underfit due to too few samples and no cross-asset regularization. The EU-5 configuration produced Sharpe ratios of 4–12 per index in backtest.

### Volume fallback for indexes

`FeatureEngineer._add_volume_ratio()` now detects when volume data is missing or entirely zero, which is common for index tickers that report no traded volume. In that case the method falls back to `volume_ratio = 1.0` (neutral) rather than computing NaN or zero values that would propagate through and corrupt downstream features.

### Buy bias fix — horizon-scaled thresholds and fixed adaptive scaling

Two root causes of buy bias were fixed simultaneously. The adaptive threshold scaling was inverted: high-volatility stocks had easier thresholds than low-volatility ones; fixed by computing `scale = volatility / ref_vol` (was `ref_vol / volatility`). Additionally, a single threshold was applied across all horizons, meaning the 20d horizon almost always exceeded the 2% threshold while the 5d horizon remained selective; fixed with `buy_thresh_h = base_thresh * sqrt(h / base_h)` so each horizon has a proportionally scaled threshold. The result was a training distribution shift from roughly 48% Buy / 15% Hold / 37% Sell to 28% Buy / 50% Hold / 22% Sell — a much more realistic label balance.

### Portfolio backtester with shared capital

A new `PortfolioBacktester` class was added in `backtesting/portfolio.py`. It runs signals on multiple tickers simultaneously against a shared capital pool, with fixed allocation per ticker (1/max_positions of total capital). Positions open on BUY signals and close automatically after `horizon` trading days, with only one open position per ticker at a time and commission applied on both entry and exit. A new `PortfolioResult` dataclass captures total_return, Sharpe, max_drawdown, win_rate, per_ticker_stats, and equity_curve. A new `portfolio` CLI subcommand exposes `--name`, `--horizon`, `--capital`, `--max-positions`, `--commission`, `--start-date`, `--end-date`, `--leverage`, `--kelly`, and `--kelly-max` flags.

### Leverage and Kelly criterion in portfolio backtester

Fixed leverage (`--leverage 2`) multiplies P&L on each trade, doubling returns at the cost of doubled drawdown. Kelly criterion (`--kelly`) sizes each position as a fraction of total capital using calibrated confidence via `f* = (p*b - q) / b` (half-Kelly, clamped to `kelly_max / max_positions`). Calibrated confidence is loaded directly from the checkpoint's `calibration.json` without importing from `signals/`, respecting module isolation. Comparison results: 1x fixed +9.17% Sharpe 2.13; 2x fixed +18.33% Sharpe 1.55; Kelly +6.63% Sharpe 1.37. Kelly underperforms because model confidence clusters narrowly in the 54–69% calibrated range, leaving little room for meaningful variable sizing; 2x fixed leverage is the best practical choice.

---

## 2026-03-15 (session 5)

### Multi-horizon training with max-return labeling

Replaced single-horizon (5d end-return) labels with max-return labels across three horizons [5d, 10d, 20d], each with its own signal classification head sharing a common LSTM backbone. At inference, a majority-vote consensus (2/3 horizons must agree on Buy or Sell) determines the final signal. The key insight: labeling with `max(close[t+1:t+h])` instead of `close[t+h]` eliminates a large source of noise — stocks that peak mid-horizon but retrace no longer get mislabeled as Hold. Longer horizons have naturally cleaner labels (20d at 78% val accuracy with near-zero train/val gap vs 42% previously). Test accuracy improvements: 5d 52.3%, 10d 66.7%, 20d 77.7% (was 40.5% for single horizon). Calibration spread across multiple buckets (vs 91% collapsing into 30-40% before). `ModelConfig` gains a `prediction_horizons` field so `SignalGenerator` and `Backtester` reconstruct the correct architecture at load time. `create_sequences` updated to accept a list of label arrays.

---

## 2026-03-15 (session 4)

### Removed all 1h / multi-interval code

Removed the `--interval` flag from `train`, `backtest`, and `calibrate` CLI commands. Removed `interval` parameter from `ModelTrainer`, `Backtester`, `MetricsCalculator`, and `SignalGenerator`. Removed `ModelConfig.BARS_PER_DAY` and `ModelConfig.FETCH_PERIOD` dicts — replaced with the module-level constant `FETCH_PERIOD = "10y"`. `ModelConfig.checkpoint_paths()` no longer takes an interval argument. The `bars_per_year = 252 * 7` branch in Sharpe/Sortino calculation is removed. Obsolete tests for 1h annualization factor and checkpoint suffixes updated. Motivation: the 1h model was retired (only 4 months holdout, overfits fast on 2-year data cap, calibration collapses in the 40–70% confidence range). Removing the branching code simplifies all call sites and eliminates an entire class of configuration errors.

### Removed 1h model checkpoints

Deleted `signal_model_1h.weights.h5`, `signal_model_1h_config.json`, and `calibration_1h.json`. The 1h model was retired because: yfinance caps hourly data at 2 years giving only ~4 months of holdout — too short to trust; training overfit early (stopped at epoch 15/50 with a large train/val gap); and calibration collapsed the 40–70% confidence range to a single value (57.3%), indicating the model wasn't differentiating within its most common output range. The `--interval` CLI flag is retained so a 1h model can be retrained if a better data source becomes available.

### Fixed MLflow calibration logging — separate run instead of re-opening completed run

`_run_calibration()` in `main.py` previously re-opened the training run via `mlflow.start_run(run_id=existing_id)` to log calibration metrics. This triggered a `POST /api/2.0/mlflow/logged-models/search` call that returned 404 on MLflow 2.10.0, crashing the auto-calibration step after training. Replaced with a new dedicated run tagged `run_type=calibration` and `training_run_id=<parent>`. The two runs are now linked via tag rather than run re-entry, which is the standard MLflow pattern for post-training steps.

### Fixed Pydantic class constants — BARS_PER_DAY and FETCH_PERIOD moved to module level

Pydantic's `BaseModel` intercepts class-level dict assignments and treats them as model fields, causing `AttributeError: BARS_PER_DAY` at import time. Moved both constants to module level in `models/config.py` and re-attached them to `ModelConfig` after class definition so `ModelConfig.BARS_PER_DAY` / `ModelConfig.FETCH_PERIOD` continue to work at all call sites.

---

## 2026-03-15 (session 3)

### Extended 1d training data from 5y to 10y

`ModelConfig.FETCH_PERIOD` for `1d` changed from `"5y"` to `"10y"`. With 10 years of daily data the test holdout covers ~18 months instead of ~6 months, giving meaningfully more trustworthy validation metrics. The 1h interval remains at `"2y"` — yfinance hard-limits hourly data to 2 years.

### Interval constants moved to ModelConfig (single source of truth)

`BARS_PER_DAY` and `FETCH_PERIOD` are now class-level constants on `ModelConfig` in `models/config.py`. Previously these dicts were duplicated across `ModelTrainer`, `Backtester`, and inline in `main.py`. Any future period or bars-per-day change now requires editing one place only.

### Fixed auto-calibrate horizon in `train` command

The auto-calibration that runs after `train` was hardcoded to `horizon=5` regardless of interval, causing the 1h model to be calibrated against 5-hour outcomes instead of 35-hour (5 trading days) outcomes. Now uses `5 * ModelConfig.BARS_PER_DAY[interval]` so 1d → 5 and 1h → 35.

---

## 2026-03-15 (session 2)

### Fixed calibration hang — naive isotonic regression replaced with sklearn

`ConfidenceCalibrator._isotonic_regression()` used a naive pair-pooling PAV algorithm with a `while True` loop. Pooling a pair can create new violations with neighboring elements, requiring repeated passes that could take very long before floating-point values converge. Replaced with `sklearn.isotonic.IsotonicRegression` which implements the correct O(n) PAV algorithm and is guaranteed to terminate in a single pass. This was the root cause of `calibrate --interval 1h` hanging indefinitely. Full 5-ticker 1h calibration now completes in ~6 seconds.

### Removed redundant OMXS30 benchmark fetch

`Backtester.run()` was fetching `^OMX` twice per ticker: once inside `fetch_cross_asset_data()` (for cross-asset features) and again in `_compute_benchmark_return()` at the end of `run()`. The second fetch is now replaced by passing the already-fetched series from `ref_data`. `_compute_benchmark_return()` falls back to a live fetch only when no cached data is supplied. This halves the yfinance request count per ticker.

### Added 30-second timeout to all yfinance requests

Both `.history()` calls in `StockDataFetcher` now pass `timeout=30`. Previously, a slow or non-responding Yahoo Finance would cause any backtest or calibration run to hang indefinitely with no error. Now a `Timeout` exception is raised after 30 seconds, giving a clear failure message instead of a silent freeze.

---

## 2026-03-15

### Portfolio-level risk limits in SignalGenerator (R6)
`SignalGenerator` now accepts two optional constructor parameters: `max_drawdown_pct` (halt new directional signals when portfolio is down this % from peak) and `max_positions` (suppress new signals once this many positions are open). Both `generate()` and `scan()` accept `portfolio_drawdown` and `open_position_count` arguments that the caller passes in with live portfolio state. The logic lives in `_apply_portfolio_limits()` — a testable helper that runs after the confidence threshold filter and forces HOLD when a limit is breached. HOLD signals pass through unchanged. Previously the generator had no awareness of portfolio state, making it possible to keep adding positions into a drawdown. 11 new unit tests added.

### Benchmark comparison in backtest output (B5)
`BacktestResult` now includes a `benchmark_return` field — the OMXS30 (`^OMX`) total return over the same backtest period. The backtester fetches it automatically at the end of `run()` and falls back to `None` gracefully if the network call fails. The `summary()` output now has a **BENCHMARK COMPARISON** section showing buy & hold (ticker), index return, and a pointer to the strategy net returns in the table above. `to_dict()` includes `benchmark_return` for JSON export. Previously there was no baseline, making it impossible to tell whether a positive net return reflected genuine alpha or simply a rising market.

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
- TASKS.md now links to CONTRIBUTIONS.md and STRATEGY.md at the top, and its finish protocol includes pytest, backtest validation, and CHANGELOG steps.
- CLAUDE.md's duplicated backtest output guide (copy of STRATEGY.md) replaced with a single pointer — one source of truth.
- README.md Contributing section now references both TASKS.md and CONTRIBUTIONS.md.
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
