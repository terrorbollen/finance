# Contributing

## Philosophy

This project prioritises reliability over sophistication. Before adding anything, ask: does this make the signals more trustworthy, or just more complex? A simpler, well-validated change beats a clever one that hasn't been honestly tested.

## Before you start

1. Read `CLAUDE.md` and `TASKS.md`.
2. If you're working alongside other agents, claim a task in `TASKS.md` before touching any files.
3. Understand the existing code in the module you're changing before modifying it.

## Picking up a task

- Tasks are tracked in `TASKS.md`, ordered by priority within each section.
- Pick the top unclaimed task in a module no one else currently owns.
- Claim it by setting **Claimed by** to your name and **Status** to `in_progress`.

## Scope discipline

- Only touch files listed in your task's **Scope** column.
- Do not refactor, rename, or "improve" code outside your scope, even if it looks like it needs it.
- If you spot a bug or improvement outside your scope, add it as a new row in `TASKS.md` with status `open` and move on.

## Coding standards

- Keep changes minimal. Solve the problem; don't engineer around it.
- No new abstractions, helpers, or utilities unless the task explicitly requires them.
- Do not add error handling for scenarios that can't happen. Trust internal guarantees.
- Validate only at system boundaries (user input, external APIs).

## Testing

- Every new feature or bug fix should have a corresponding test in `tests/`.
- Tests must use real data or realistic fixtures — do not mock the model or backtester internals in ways that decouple test behaviour from production behaviour.
- Run the full suite before marking a task done: `uv run pytest`

## Quality checks

Both must pass clean before you mark a task done:

```bash
uv run ruff check .
uv run mypy .
```

## Validating model changes

See [`backtesting/STRATEGY.md`](backtesting/STRATEGY.md) for what makes a backtest result trustworthy (minimum trade count, significance thresholds, multi-ticker validation).

If your change touches `models/`, `data/features.py`, or `signals/`, run a backtest before and after and compare via `history`:

```bash
uv run python main.py backtest VOLV-B.ST
uv run python main.py history --ticker VOLV-B.ST
```

A change is worth keeping if **Sharpe and NetRet both improve**. Accuracy alone can be misleading.


## Finishing up

1. Run `ruff` and `mypy` — both clean.
2. Run `pytest` — all tests pass.
3. Mark the task **done** in `TASKS.md`, clear **Claimed by**, and release your module in the ownership map.
4. Add an entry to `CHANGELOG.md` — what changed and why, not just which files.

---

## Completed work

Features and fixes already implemented. Do not re-implement these; understand them before touching the relevant modules.

### Backtesting

- **Transaction costs** (~0.05–0.1% per trade) — `backtesting/backtester.py`, `backtesting/metrics.py`
- **Sharpe, Sortino, max drawdown, Calmar ratio** — `backtesting/metrics.py`, `backtesting/results.py`
- **Strict holdout split** — backtest period is enforced never to overlap training data — `backtesting/backtester.py`, `main.py`, `models/training.py`
- **Slippage modeling** (volume-based) — `backtesting/backtester.py`
- **Benchmark comparison** (index vs strategy) — `backtesting/results.py`, `backtesting/backtester.py`
- **Walk-forward backtest** — retrain on expanding window during backtest — `backtesting/backtester.py`, `models/walk_forward.py`
- **Monte Carlo simulation** for backtest confidence intervals — `backtesting/metrics.py`, `backtesting/results.py`
- **Calibration staleness warning** — warns when calibrator was fitted >N days ago — `signals/calibration.py`, `main.py`
- **Equity curve CSV export** via `--export-equity` CLI flag — `backtesting/results.py`, `main.py`
- **Per-horizon signals wired into backtester** — each horizon entry uses its own head's output, not a copied consensus — `backtesting/backtester.py`
- **Target-price lookup fixed** — outcome prices now looked up by date (via `BDay` offset) rather than row index, so data gaps no longer shift P&L — `backtesting/backtester.py`
- **Calibration applied in backtest path** — backtester now loads and applies the calibrator so confidence scores match what live signals use — `backtesting/backtester.py`
- **Calibration refreshed after walk-forward retrain** — calibrator is refit or cleared after each `_retrain_model()` call — `backtesting/backtester.py`
- **ADX regime breakdown null guard** — `p.adx is not None` checked before classifying; missing-ADX predictions counted separately — `backtesting/metrics.py`
- **Default backtest horizons match model's trained horizons** — `run_backtest()` defaults to `self.prediction_horizons` rather than a hardcoded `[1…7]` — `backtesting/backtester.py`
- **`Direction` moved to `models/direction.py`** — eliminates the `backtesting/ → signals/` import that violated INVARIANTS.md; all callers updated — `backtesting/results.py`, `backtesting/plot.py`, `signals/generator.py`, `models/signal_model.py`, `models/training.py`
- **Bare `except Exception` replaced** — specific exception types used at each catch site in `backtester.py` — `backtesting/backtester.py`
- **Feature mismatch is a hard error** — any missing feature column raises immediately and names the missing columns; the 10%-tolerance silent-fallback is gone — `backtesting/backtester.py`, `signals/generator.py`

### Risk management

- **Dynamic stop-loss** based on ATR — `signals/generator.py`
- **Position sizing** based on confidence (Kelly criterion) — `signals/generator.py`
- **Dynamic take-profit targets** using ATR multiples — `signals/generator.py`
- **Confidence threshold filtering** (only trade above X%) — `signals/generator.py`
- **Per-direction calibration** (separate BUY/SELL/HOLD isotonic calibrators with global fallback) — `signals/calibration.py`, `signals/generator.py`
- **Portfolio-level risk limits** (max drawdown cap, max position count) — `signals/generator.py`
- **Directional calibration exposed in `calibrate` CLI** — `main.py`
- **`get_calibration_table()` added to `DirectionalCalibrator`** — `signals/calibration.py`
- **Per-direction calibration fallback path tested** — CI verifies that a partially-fitted `DirectionalCalibrator` correctly falls back to the global calibrator — `tests/test_calibration.py`

### Feature engineering

- **Market regime indicators** — ADX, volatility regime — `data/features.py`
- **Cross-asset features** — OMXS30 correlation, USD/SEK, EUR/SEK — `data/features.py`, `data/fetcher.py`
- **Calendar effects** — day of week, month, earnings season — `data/features.py`
- **Volatility features** — VIX/VSTOXX correlation — `data/features.py`, `data/fetcher.py`
- **Macro indicators** — oil prices, interest rates — `data/features.py`, `data/fetcher.py`
- **Out-of-distribution feature detection** — at inference time, features outside mean ± 3 std trigger a warning; stats come from `signal_model_config.json`
- **Bollinger Band zero-width guard** — `bb_position = 0.5` when rolling std is zero, instead of dropping the row — `data/features.py`
- **ADX zero-ATR guard** — `.where(atr_w > 0, 0.0)` applied to `di_plus`/`di_minus` to prevent `Inf`/`NaN` on constant-price periods — `data/features.py`

### Model improvements

- **Purged cross-validation** — gap between train/val folds prevents leakage — `models/walk_forward.py`
- **Holdout ticker validation** — one ticker excluded from training to test generalisation — `models/training.py`, `backtesting/backtester.py`
- **Hyperparameter tuning** — systematic search implemented — `models/training.py`
- **Per-horizon signals from multi-horizon heads** — `predict()` returns per-head probs/classes; consensus signal retained for the live `generate` command — `models/signal_model.py`, `models/training.py`
- **Ensemble models** — LSTM + GRU predictions averaged — `models/signal_model.py`, `models/training.py`
- **Walk-forward zero-window guard** — raises a clear error (naming gap sizes and fold size) when no windows can be generated — `models/walk_forward.py`

### Infrastructure

- **MLflow experiment tracking** — auto-connect via env var, consolidated experiment names, CLI args logged as `cli.*` tags — `models/mlflow_tracking.py`, `models/training.py`, `models/walk_forward.py`, `main.py`
- **MLflow backtest logging + `history` CLI command** — backtest metrics persisted to MLflow; `main.py history` shows trend across runs — `models/mlflow_tracking.py`, `main.py`
