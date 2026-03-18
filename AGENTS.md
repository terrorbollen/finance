# Agent Coordination

Multi-agent task board and improvement log. Every agent reads this before starting and updates it when finished — or whenever they spot something worth doing.

> Full contribution guidelines (coding standards, testing rules, scope discipline): [`CONTRIBUTIONS.md`](../CONTRIBUTIONS.md)
> Backtest validation rules (what makes a result trustworthy, horizon guide, known limitations): [`backtesting/STRATEGY.md`](STRATEGY.md)

## Protocol

### Before you start
1. Read this file, `CLAUDE.md`, and `CONTRIBUTIONS.md`.
2. Run `ls .agent-locks/` to see which tasks are currently claimed.
3. Pick the **top open task** that has no lockfile.
4. **Claim it atomically:** `mkdir .agent-locks/TASK_ID` — if this fails, someone else just claimed it; pick another task.
5. Update **Claimed by** and **Status** to `in_progress` in the task table below.

### While you work
6. Touch only files listed in your task's **Scope**.
7. If you notice a bug or improvement **outside your scope**, add it as a new `open` task — don't fix it now.

### Before you finish
8. Run `uv run ruff check .` and `uv run mypy .` — both must pass clean.
9. Run `uv run pytest` — all tests pass.
10. If your task touches `models/`, `data/features.py`, or `signals/`, run a backtest before and after.
11. Set **Status** to `done` and clear **Claimed by** in the task table.
12. Release the lock: `rmdir .agent-locks/TASK_ID`
13. Add an entry to `CHANGELOG.md` — what changed and why, not just which files.

> **Priority:** Tasks are listed top-to-bottom within each section from highest to lowest priority.

---

## Task Board

### Backtesting (`backtesting/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| B1 | Add transaction costs (~0.05-0.1% per trade) | `backtesting/backtester.py`, `backtesting/metrics.py` | — | done |
| B2 | Add Sharpe, Sortino, max drawdown, Calmar ratio metrics | `backtesting/metrics.py`, `backtesting/results.py` | — | done |
| B3 | Strict holdout split: enforce backtest period never overlaps training data | `backtesting/backtester.py`, `main.py`, `models/training.py` | Agent-Main | done |
| B4 | Add slippage modeling (volume-based) | `backtesting/backtester.py` | — | done |
| B5 | Add benchmark comparison (index vs strategy) | `backtesting/results.py`, `backtesting/backtester.py` | — | done |
| B6 | Walk-forward backtest: retrain model on expanding window during backtest | `backtesting/backtester.py`, `models/walk_forward.py` | — | done |
| B7 | Monte Carlo simulation for backtest confidence intervals | `backtesting/metrics.py`, `backtesting/results.py` | — | done |
| B8 | Calibration staleness warning: warn if calibrator was fitted >N days ago so live signals aren't silently based on stale calibration | `signals/calibration.py`, `main.py` | — | done |
| B9 | Equity curve CSV export via CLI flag (`--export-equity`): backtester builds equity curves internally but there's no way to get them out | `backtesting/results.py`, `main.py` | Claude | in_progress |
| B10 | **Wire per-horizon signals into the backtester (depends on M4).** Currently `backtester.py` takes a single `(signal, confidence, price_change)` tuple from `model.predict()` and copies it to all horizon entries. Once M4 is done, update the backtester to extract the horizon-specific head output for each `HorizonPrediction` instead. This makes per-horizon backtest results reflect genuine per-horizon model quality rather than just holding-period effects. | `backtesting/backtester.py` | — | open |
| B11 | **Fix target-price lookup: row-index arithmetic breaks on data gaps.** In `backtester.py`, outcome prices are fetched with `features_df.iloc[idx + horizon]` — a row offset. If there are any missing trading days in the fetched data (yfinance occasionally drops sessions), `idx + horizon` lands on the wrong date. For a 5-day horizon, a single gap shifts every outcome price by one day, silently misstating every trade's P&L. Fix by storing the prediction's target date as `features_df.index[idx] + BDay(horizon)` at prediction time and looking up the close price by date, not row offset. | — | done |
| B12 | **Calibration inconsistency: backtest reports raw probabilities, live signals use calibrated ones.** `Backtester` always calls `model.predict()` and uses the raw softmax confidence directly — the calibrator is never loaded in the backtest path. `SignalGenerator` applies isotonic calibration. This means win-rate-by-confidence buckets shown in backtest output are computed against uncalibrated scores, so they don't reflect what you'd actually trade on in production. Either apply calibration in the backtester (preferred) or make the discrepancy explicit in the output so users know the confidence column is pre-calibration. | `backtesting/backtester.py` | — | open |
| B13 | **Retrain in walk-forward backtester doesn't refresh calibration.** When `retrain_every` is set, `_retrain_model()` updates `self.model`, `self.feature_mean`, and `self.feature_std` but never reloads or clears `self.calibrator`. The old calibrator (fitted on pre-cutoff data) keeps being applied to the newly retrained model, so confidence scores become increasingly miscalibrated across retraining windows. Fix: after every retrain, either refit the calibrator on recent backtest predictions or mark confidence as uncalibrated for the new window. | `backtesting/backtester.py` | — | open |
| B14 | **ADX regime breakdown crashes when `adx` is None.** The regime breakdown calls `condition(p.adx)` where `condition = lambda v: v < 20`. If any `HorizonPrediction` has `adx = None` (possible on early rows before the ADX warmup period completes), this raises `TypeError`. The entire regime metrics section is lost for that backtest run. Fix: guard with `p.adx is not None` before classifying, and count/report predictions with missing ADX separately. | `backtesting/metrics.py` | — | done |

### Risk Management (`signals/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| R1 | Dynamic stop-loss based on ATR | `signals/generator.py` | — | done |
| R2 | Position sizing based on confidence (Kelly criterion) | `signals/generator.py` | — | done |
| R3 | Dynamic take-profit targets using ATR multiples | `signals/generator.py` | — | done |
| R4 | Confidence threshold filtering (only trade above X%) | `signals/generator.py` | — | done |
| R5 | Per-direction calibration (separate BUY/SELL/HOLD calibrators) | `signals/calibration.py`, `signals/generator.py` | — | done |
| R6 | Portfolio-level risk limits (max drawdown cap, max position count) | `signals/generator.py` | — | done |
| R7 | Expose directional calibration in `calibrate` CLI command (currently only fits the global calibrator) | `main.py` | — | done |
| R8 | Add `get_calibration_table()` equivalent to `DirectionalCalibrator` for diagnostic display | `signals/calibration.py` | — | done |

### Feature Engineering (`data/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| F1 | Market regime indicators (ADX, volatility regime) | `data/features.py` | — | done |
| F2 | Cross-asset features (OMXS30 correlation, USD/SEK, EUR/SEK) | `data/features.py`, `data/fetcher.py` | — | done |
| F3 | Calendar effects (day of week, month, earnings season) | `data/features.py` | — | done |
| F8 | **Bollinger Band feature silently drops constant-price rows.** `_add_bb_position()` divides by `band_width.where(band_width > 0)`, producing NaN when the 20-bar rolling std is zero (constant price periods, trading halts, early data). Those rows are silently removed by `dropna()`, so the model never trains on low-volatility regimes and has no signal for them in live trading. Fix: when `band_width == 0`, set `bb_position = 0.5` (price at the midpoint of a flat band) rather than NaN. | `data/features.py` | — | done |
| F4 | Volatility features (VIX/VSTOXX correlation) | `data/features.py`, `data/fetcher.py` | — | done |
| F5 | Macro indicators (oil prices, interest rates) | `data/features.py`, `data/fetcher.py` | Claude | in_progress |
| F6 | **Out-of-distribution feature detection.** At inference time, if live features fall outside the training distribution (e.g. mean ± 3 std for any feature), warn the user. The training stats are already saved in `signal_model_config.json` (`feature_mean`, `feature_std`), so the check is cheap. Without this, a regime shift — e.g. extreme volatility, currency crisis, post-earnings gap — silently feeds the model inputs it has never seen, producing confident-looking signals that are actually extrapolation. The warning doesn't need to block the signal, just flag which features are out-of-range. | `signals/generator.py` | Claude | in_progress |
| F7 | **Feature mismatch should be a hard error, not a silent tolerance.** Both `backtester.py` and `generator.py` have a `> 10% missing features → raise, otherwise warn and continue` pattern. This is the wrong tradeoff: a model trained on 37 features receiving 33 produces unpredictably degraded predictions with no indication in the output. Change the threshold to zero tolerance: if any feature from `feature_columns` is missing, raise immediately and name the missing columns. The only acceptable exception is during development when intentionally testing a partial feature set — which should require an explicit flag, not a silent fallback. | `backtesting/backtester.py`, `signals/generator.py` | — | open |

### Model Improvements (`models/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| M1 | Purged cross-validation (gap between train/val folds to prevent leakage) | `models/walk_forward.py` | — | done |
| M2 | Holdout ticker validation (exclude one ticker from training, test generalisation) | `models/training.py`, `backtesting/backtester.py` | — | done |
| M3 | Hyperparameter tuning (systematic search) | `models/training.py` | — | open |
| M4 | **Expose per-horizon signals from multi-horizon heads.** The model already trains one classification head per horizon (`signal_1d` … `signal_7d`), but `SignalModel.predict()` collapses them into a single majority-vote consensus before returning. This means the backtester copies the same signal to every horizon entry, so per-horizon backtests differ only in holding period — not in signal quality. A proper multi-horizon model should return each head's own prediction so that e.g. the 1-day head (trained on short-term momentum) can disagree with the 7-day head (trained on medium-term trend). Change `predict()` to return per-horizon probs/classes and update the backtester (see B10) to consume them. Keep the consensus signal as an optional aggregation for the live `generate` command. | `models/signal_model.py`, `models/training.py` | — | open |
| M5 | Ensemble models (LSTM + GRU) — average predictions from both architectures | `models/signal_model.py`, `models/training.py` | — | open |
| M6 | **Multi-ticker normalization may leak across ticker boundaries.** In `training.py`, all tickers are concatenated into a single feature array before the train/val/test split. The normalization mean/std is then computed on rows `0 : train_end_raw`. If tickers have different date ranges (likely — some have longer history than others), the rows from later-dated tickers may land in the "training" portion alongside early rows from other tickers, or vice versa. This isn't a pure temporal split. Fix by computing normalization stats per ticker on that ticker's training portion only, then applying them before concatenation. This preserves the invariant that no future data informs normalization. | `models/training.py` | — | open |
| M7 | **Off-by-one in max-return label creation.** In `training.py`, the sliding window for max-return labels is created from `close_vals[1:]` (one bar shifted forward), but returns are divided by `close_vals[:n-h]` (starting from bar 0). This means the max return over window [t+1 … t+h] is divided by the price at t−1 instead of t, misstating every label by one day. Needs a unit test to confirm, then align indices so numerator window and denominator price both reference bar t. | `models/training.py` | — | open |
| M8 | **No feature-dimension check before stacking multi-ticker arrays.** After `prepare_data()` is called per ticker, the feature arrays are appended to `all_features` without asserting `features.shape[1]` is consistent across tickers. If the feature engineer drops different columns for one ticker (e.g. missing volume on an index), `np.vstack(all_features)` crashes with a cryptic shape error after potentially minutes of data fetching. Fix: assert `features.shape[1] == expected_n_features` immediately after each `prepare_data()` call, naming the mismatching ticker. | `models/training.py` | — | open |
| M9 | **Walk-forward trainer silently produces zero windows when gaps exceed fold size.** `WalkForwardTrainer` sets default `purge_gap = max(prediction_horizons)` and `embargo_gap = sequence_length`, but never validates that their sum is smaller than the validation window. On a short holdout (e.g. 60 days with gaps of 20+20), zero windows are generated. The training loop completes "successfully" with an empty list, returning a model trained on nothing. Fix: assert `len(windows) > 0` after computing windows and raise a clear error naming the gap sizes and fold size that caused the failure. | `models/walk_forward.py` | — | open |

### Signal Logic (`signals/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| S4 | **Signal timestamp uses wall-clock time instead of data date.** `Signal.timestamp` is set to `pd.Timestamp.now()` at generation time regardless of which date's data produced the signal. When generating signals for historical data (backtest replay, debugging, calibration), every signal is stamped with today's date. This breaks audit trails and makes it impossible to reconstruct when a signal was valid. Fix: pass the last date from the feature DataFrame and use it as the timestamp; fall back to `now()` only when no date context is available. | `signals/generator.py` | — | open |
| S1 | Signal filtering (skip earnings, low-volume, low-confidence) | `signals/generator.py` | — | open |
| S2 | Market regime awareness (different thresholds for trend/range) | `signals/generator.py` | — | open |
| S3 | Multi-timeframe confirmation (daily confirmed by weekly) | `signals/generator.py`, `data/fetcher.py` | — | open |

### Execution Strategy (`signals/`, `backtesting/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| E1 | Time-based exits (exit if signal stale after N days) | `backtesting/backtester.py` | — | open |
| E2 | Trailing stops (based on recent highs/lows) | `signals/generator.py` | — | open |
| E3 | Partial exits (scale out as price target approaches) | `signals/generator.py`, `backtesting/backtester.py` | — | open |

### Infrastructure (`main.py`, new files)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| I1 | MLflow experiment tracking: auto-connect via env var, consolidate experiment names, log CLI args as tags | `models/mlflow_tracking.py`, `models/training.py`, `models/walk_forward.py`, `main.py` | — | done |
| I2 | Log backtest results to MLflow; add `history` CLI command showing trend across runs | `models/mlflow_tracking.py`, `main.py` | — | done |
| I3 | Automated retraining: schedule periodic model retraining with new data | `models/training.py`, `main.py` | — | open |
| I4 | Real-time alerts: notify on high-confidence signals above threshold | `main.py`, `signals/generator.py` | — | open |
| I5 | Add type annotations to all CLI command handlers in main.py (`args: argparse.Namespace`, `-> None`) | `main.py` | — | open |
| I6 | Enable `--check-untyped-defs` in mypy config (currently inner function bodies like `_do_training` in training.py are not type-checked at all) | `pyproject.toml` or `mypy.ini` | — | open |
| I7 | Tighten return types in mlflow_tracking.py: `get_recent_runs` returns `list[dict]`, `get_best_run` returns `dict \| None` — both should use `dict[str, Any]` | `models/mlflow_tracking.py` | — | open |
| I8 | Print `holdout_start_date` prominently after training so users immediately see where the holdout boundary is before running a backtest | `main.py` | — | open |
| I9 | **Remove fallback normalization — it introduces data leakage.** When `signal_model_config.json` is missing or can't be loaded, both `backtester.py` and `generator.py` fall back to computing normalization mean/std from the current data window. This is data leakage: for a prediction on day T, the window includes days T-seq_len through T, so the normalization is influenced by the most recent prices — prices the model is supposed to be predicting without knowing. INVARIANTS.md already forbids this fallback in production, but there's no enforcement. Replace the fallback with a hard `raise RuntimeError` and a clear message telling the user to retrain and save the config. The fallback may also hide the case where someone forgets to run `train` before `backtest`. | `backtesting/backtester.py`, `signals/generator.py` | — | open |
| I10 | Backfill `holdout_start_date` in old model configs or add a CLI flag (`--holdout-start`) to pass it at backtest time so holdout enforcement works on pre-existing checkpoints | `main.py`, `backtesting/backtester.py` | — | open |
| I11 | Log `leverage` to MLflow in backtest runs: `cmd_backtest` logs ticker/dates/commission but omits `leverage`, making it impossible to distinguish or compare leveraged runs in history | `main.py` | — | open |

### Portfolio (`backtesting/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| P1 | **Kelly position sizing uses initial capital instead of current capital.** `position_capital = min(capital, self.initial_capital * kelly_f)` — the cap is current capital but the sizing basis is `initial_capital`. After a drawdown, a 50% Kelly fraction can still attempt a $5k position even if only $2k cash remains, potentially allocating more than 100% of available capital. Fix: use `current_capital * kelly_f` as the sizing basis. | `backtesting/portfolio.py` | — | open |
| P2 | **Portfolio commission double-counts leverage.** Commission is computed as `entry_val * commission_pct * 2 * eff_lev`. If `entry_val` is the base cash committed (pre-leverage), multiplying by `eff_lev` is correct. But if it's already the leveraged notional, this inflates costs by `eff_lev`. Audit which convention `entry_val` follows and make commission calculation consistent with it. Incorrect commission math makes leveraged backtests report unrealistically bad net returns. | `backtesting/portfolio.py` | — | open |

---

> Active locks: run `ls .agent-locks/` to see which tasks are currently in progress.
