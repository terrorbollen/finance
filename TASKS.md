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
| B19 | **Monte Carlo simulation is broken — all permutations produce identical total return and Sharpe.** `_run_monte_carlo()` in `backtesting/metrics.py` computes total return as `np.sum(shuffled)` and Sharpe as `mean(shuffled) / std(shuffled)`. Both are order-independent statistics — shuffling trade order has no effect on them, so all 1 000 permutations always yield the same values. The output (`mean == p5 == p95`) makes the simulation useless as a path-dependency test. Fix: replace `np.sum(shuffled)` with compounded return `(np.prod(1 + shuffled / 100) - 1) * 100`; replace the per-trade mean/std Sharpe with a path-based Sharpe computed over the cumulative equity curve of each permutation (so ordering actually affects the result). Add a unit test asserting that p5 < mean < p95 when at least two distinct return values exist. | `backtesting/metrics.py`, `tests/test_metrics.py` | — | done |

### Risk Management (`signals/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|

### Feature Engineering (`data/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|

### Model Improvements (`models/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| M6 | **Multi-ticker normalization may leak across ticker boundaries.** In `training.py`, all tickers are concatenated into a single feature array before the train/val/test split. The normalization mean/std is then computed on rows `0 : train_end_raw`. If tickers have different date ranges (likely — some have longer history than others), the rows from later-dated tickers may land in the "training" portion alongside early rows from other tickers, or vice versa. This isn't a pure temporal split. Fix by computing normalization stats per ticker on that ticker's training portion only, then applying them before concatenation. This preserves the invariant that no future data informs normalization. | `models/training.py` | — | done |
| M7 | **Off-by-one in max-return label creation.** In `training.py`, the sliding window for max-return labels is created from `close_vals[1:]` (one bar shifted forward), but returns are divided by `close_vals[:n-h]` (starting from bar 0). This means the max return over window [t+1 … t+h] is divided by the price at t−1 instead of t, misstating every label by one day. Needs a unit test to confirm, then align indices so numerator window and denominator price both reference bar t. | — | done |
| M8 | **No feature-dimension check before stacking multi-ticker arrays.** After `prepare_data()` is called per ticker, the feature arrays are appended to `all_features` without asserting `features.shape[1]` is consistent across tickers. If the feature engineer drops different columns for one ticker (e.g. missing volume on an index), `np.vstack(all_features)` crashes with a cryptic shape error after potentially minutes of data fetching. Fix: assert `features.shape[1] == expected_n_features` immediately after each `prepare_data()` call, naming the mismatching ticker. | `models/training.py` | — | done |

### Signal Logic (`signals/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| S4 | **Signal timestamp uses wall-clock time instead of data date.** `Signal.timestamp` is set to `pd.Timestamp.now()` at generation time regardless of which date's data produced the signal. When generating signals for historical data (backtest replay, debugging, calibration), every signal is stamped with today's date. This breaks audit trails and makes it impossible to reconstruct when a signal was valid. Fix: pass the last date from the feature DataFrame and use it as the timestamp; fall back to `now()` only when no date context is available. | `signals/generator.py` | — | done |
| S1 | Signal filtering (skip earnings, low-volume, low-confidence) | `signals/generator.py` | — | done |
| S2 | Market regime awareness (different thresholds for trend/range) | `signals/generator.py` | — | open |
| S3 | Multi-timeframe confirmation (daily confirmed by weekly) | `signals/generator.py`, `data/fetcher.py` | — | done |

### Execution Strategy (`signals/`, `backtesting/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| E1 | Time-based exits (exit if signal stale after N days) | `backtesting/backtester.py` | — | open |
| E2 | Trailing stops (based on recent highs/lows) | `signals/generator.py` | — | open |
| E3 | Partial exits (scale out as price target approaches) | `signals/generator.py`, `backtesting/backtester.py` | — | open |

### Infrastructure (`main.py`, new files)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| I3 | Automated retraining: schedule periodic model retraining with new data | `models/training.py`, `main.py` | — | open |
| I4 | Real-time alerts: notify on high-confidence signals above threshold | `main.py`, `signals/generator.py` | — | open |
| I5 | Add type annotations to all CLI command handlers in main.py (`args: argparse.Namespace`, `-> None`) | `main.py` | — | open |
| I6 | Enable `--check-untyped-defs` in mypy config (currently inner function bodies like `_do_training` in training.py are not type-checked at all) | `pyproject.toml` or `mypy.ini` | — | open |
| I7 | Tighten return types in mlflow_tracking.py: `get_recent_runs` returns `list[dict]`, `get_best_run` returns `dict \| None` — both should use `dict[str, Any]` | `models/mlflow_tracking.py` | — | open |
| I8 | Print `holdout_start_date` prominently after training so users immediately see where the holdout boundary is before running a backtest | `main.py` | — | open |
| I9 | **Remove fallback normalization — it introduces data leakage.** When `signal_model_config.json` is missing or can't be loaded, both `backtester.py` and `generator.py` fall back to computing normalization mean/std from the current data window. This is data leakage: for a prediction on day T, the window includes days T-seq_len through T, so the normalization is influenced by the most recent prices — prices the model is supposed to be predicting without knowing. INVARIANTS.md already forbids this fallback in production, but there's no enforcement. Replace the fallback with a hard `raise RuntimeError` and a clear message telling the user to retrain and save the config. The fallback may also hide the case where someone forgets to run `train` before `backtest`. | `backtesting/backtester.py`, `signals/generator.py` | — | open |
| I12 | **No tests for `portfolio.py`, `data/fetcher.py`, or `models/losses.py`.** These three modules have zero test coverage: `portfolio.py` contains the Kelly sizing bug (P1) and commission double-counting (P2) that need regression tests before fixing; `fetcher.py` has the cross-asset reference data fallback that is a known invariant concern; `losses.py` has focal-loss implementations used in every training run. Add `tests/test_portfolio.py` (Kelly sizing, commission, P1/P2 regression), `tests/test_fetcher.py` (fallback behaviour when yfinance returns no data), and `tests/test_losses.py` (focal loss output shape, class-weight exclusivity). | `tests/` | — | done |
| I10 | Backfill `holdout_start_date` in old model configs or add a CLI flag (`--holdout-start`) to pass it at backtest time so holdout enforcement works on pre-existing checkpoints | `main.py`, `backtesting/backtester.py` | — | open |
| I11 | Log `leverage` to MLflow in backtest runs: `cmd_backtest` logs ticker/dates/commission but omits `leverage`, making it impossible to distinguish or compare leveraged runs in history | `main.py` | — | open |
| I13 | **`models/signal_model.py` is 24% covered — `predict()` is entirely untested.** `SignalModel.predict()` returns a per-horizon dict of `(signal, confidence, price_change)` tuples and a consensus signal. None of this is exercised in the test suite. A shape mismatch, wrong horizon key, or swapped BUY/SELL index would corrupt every signal and backtest silently. Add `tests/test_signal_model.py` covering: (1) `predict()` returns one key per trained horizon with correct shape; (2) confidence is in [0, 1]; (3) consensus signal matches the majority of per-horizon signals on a synthetic input; (4) `EnsembleSignalModel.predict()` averages correctly when the two backbones disagree. Use a tiny model (1 layer, small dims) built in-process — do not load weights from disk. | `tests/test_signal_model.py` | — | open |
| I14 | **`signals/generator.py` stop-loss, Kelly sizing, and confidence-filter paths are untested (57% coverage).** The ATR stop-loss calculation (lines 138–170), Kelly position sizing (lines 174–215), and confidence threshold filtering (lines 238–270) are not exercised. These paths directly determine the entry price, stop-loss level, position size, and whether a signal is suppressed — all values shown to the user. Add tests to `tests/test_signal_generation.py`: (1) ATR stop-loss distance equals `atr * multiplier / 100 * price`; (2) Kelly fraction is capped at `max_position`; (3) a signal below `min_confidence` returns `None`; (4) OOD feature warning fires when a feature exceeds 3 std. Use a `MockModel` that returns fixed per-horizon outputs. | `tests/test_signal_generation.py` | — | open |
| I15 | **`models/walk_forward.py` (30%) and `backtesting/results.py` (52%) have large untested paths.** In `walk_forward.py`, the window-generation logic (purge/embargo gap, fold count) and the assertion that raises on zero windows are never exercised by tests. In `backtesting/results.py`, the CSV export, JSON summary, and benchmark return columns (lines 278–426) have no coverage. Add: (1) `tests/test_walk_forward.py` — verify correct window count for given train/val sizes, purge, and embargo; assert zero-window error is raised with the right message; (2) extend `tests/test_metrics.py` or add `tests/test_results.py` — round-trip a `BacktestResults` object through CSV export and re-import, verify equity curve length and benchmark column presence. | `tests/test_walk_forward.py`, `tests/test_results.py` | — | open |
| I16 | **`tests/test_portfolio.py` line 216 has a mypy type error that will crash at runtime.** A variable typed as `int \| None` is used as the right-hand operand of `/` without a `None`-guard. `mypy` reports `Unsupported operand types for / ("float" and "None")`. If the `None` branch is reached, Python raises `TypeError` and the test (and any code path it exercises) crashes silently in CI. Fix: add a `assert variable is not None` or an `if variable is None` guard before the division, or narrow the type at the point of assignment. | `tests/test_portfolio.py` | — | open |

### Portfolio (`backtesting/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| P1 | **Kelly position sizing uses initial capital instead of current capital.** `position_capital = min(capital, self.initial_capital * kelly_f)` — the cap is current capital but the sizing basis is `initial_capital`. After a drawdown, a 50% Kelly fraction can still attempt a $5k position even if only $2k cash remains, potentially allocating more than 100% of available capital. Fix: use `current_capital * kelly_f` as the sizing basis. | `backtesting/portfolio.py` | — | open |
| P2 | **Portfolio commission double-counts leverage.** Commission is computed as `entry_val * commission_pct * 2 * eff_lev`. If `entry_val` is the base cash committed (pre-leverage), multiplying by `eff_lev` is correct. But if it's already the leveraged notional, this inflates costs by `eff_lev`. Audit which convention `entry_val` follows and make commission calculation consistent with it. Incorrect commission math makes leveraged backtests report unrealistically bad net returns. | `backtesting/portfolio.py` | — | open |

---

> Active locks: run `ls .agent-locks/` to see which tasks are currently in progress.
