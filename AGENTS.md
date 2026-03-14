# Agent Coordination

Multi-agent task board and improvement log. Every agent reads this before starting and updates it when finished — or whenever they spot something worth doing.

## Protocol

### Before you start
1. Read this file and `CLAUDE.md`.
2. Pick the **top unclaimed task** in a category whose module no other agent currently owns.
3. Edit **Claimed by** to your agent name and set **Status** to `in_progress`.
4. Set yourself as **Current owner** in the Module Ownership Map.

### While you work
5. Touch only files listed in your task's **Scope**.
6. If you notice a bug, design issue, missing test, or potential improvement **outside your current scope**, add it as a new task at the bottom of the relevant section — don't fix it now, just log it with status `open`.

### Before you finish
7. Run `uv run ruff check .` and `uv run mypy .` — both must pass clean.
8. Mark the task **done**, clear **Claimed by**, and release your module in the ownership map.

> **Priority:** Tasks are listed top-to-bottom within each section from highest to lowest priority. Always pick the top unclaimed one you can handle.

> **Conflicts:** If two agents accidentally claim the same module, the later one backs off and picks a different task.

---

## Task Board

### Backtesting (`backtesting/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| B1 | Add transaction costs (~0.05-0.1% per trade) | `backtesting/backtester.py`, `backtesting/metrics.py` | — | done |
| B2 | Add Sharpe, Sortino, max drawdown, Calmar ratio metrics | `backtesting/metrics.py`, `backtesting/results.py` | — | done |
| B3 | Strict holdout split: enforce backtest period never overlaps training data | `backtesting/backtester.py`, `main.py`, `models/training.py` | Agent-Main | done |
| B4 | Add slippage modeling (volume-based) | `backtesting/backtester.py` | Agent-B4 | in_progress |
| B5 | Add benchmark comparison (index vs strategy) | `backtesting/results.py`, `backtesting/backtester.py` | — | open |
| B6 | Walk-forward backtest: retrain model on expanding window during backtest | `backtesting/backtester.py`, `models/walk_forward.py` | — | open |
| B7 | Monte Carlo simulation for backtest confidence intervals | `backtesting/metrics.py`, `backtesting/results.py` | — | open |

### Risk Management (`signals/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| R1 | Dynamic stop-loss based on ATR | `signals/generator.py` | — | done |
| R2 | Position sizing based on confidence (Kelly criterion) | `signals/generator.py` | — | done |
| R3 | Dynamic take-profit targets using ATR multiples | `signals/generator.py` | — | done |
| R4 | Confidence threshold filtering (only trade above X%) | `signals/generator.py` | — | done |
| R5 | Per-direction calibration (separate BUY/SELL/HOLD calibrators) | `signals/calibration.py` | — | open |
| R6 | Portfolio-level risk limits (max drawdown cap, max position count) | `signals/generator.py` | — | open |

### Feature Engineering (`data/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| F1 | Market regime indicators (ADX, volatility regime) | `data/features.py` | Agent-F | in_progress |
| F2 | Cross-asset features (OMXS30 correlation, USD/SEK, EUR/SEK) | `data/features.py`, `data/fetcher.py` | Agent-F | in_progress |
| F3 | Calendar effects (day of week, month, earnings season) | `data/features.py` | Agent-F | in_progress |
| F4 | Volatility features (VIX/VSTOXX correlation) | `data/features.py`, `data/fetcher.py` | — | open |
| F5 | Macro indicators (oil prices, interest rates) | `data/features.py`, `data/fetcher.py` | — | open |

### Model Improvements (`models/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| M1 | Purged cross-validation (gap between train/val folds to prevent leakage) | `models/walk_forward.py` | Agent-M1 | in_progress |
| M2 | Holdout ticker validation (exclude one ticker from training, test generalisation) | `models/training.py`, `backtesting/backtester.py` | — | open |
| M3 | Hyperparameter tuning (systematic search) | `models/training.py` | — | open |
| M4 | Multi-horizon output heads (1-7 day predictions) | `models/signal_model.py`, `models/training.py` | — | open |
| M5 | Ensemble models (LSTM + GRU) — average predictions from both architectures | `models/signal_model.py`, `models/training.py` | — | open |

### Signal Logic (`signals/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
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

---

## Module Ownership Map

| Module | Files | Current owner |
|--------|-------|---------------|
| `data/` | `data/fetcher.py`, `data/features.py` | Agent-F |
| `models/` | `models/signal_model.py`, `models/training.py`, `models/walk_forward.py`, `models/mlflow_tracking.py` | Agent-M1 |
| `signals/` | `signals/generator.py`, `signals/calibration.py` | — |
| `backtesting/` | `backtesting/backtester.py`, `backtesting/metrics.py`, `backtesting/results.py` | Agent-B4 |
| `main.py` | `main.py` | — |
