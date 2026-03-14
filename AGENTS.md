# Agent Coordination

Multi-agent task board. Before starting work, an agent must claim a task by adding its name to the **Claimed by** column. Pick only unclaimed tasks. Work only within the listed module(s) for your task.

## Protocol

1. Read this file.
2. Pick an unclaimed task that matches the module you want to touch.
3. Edit **Claimed by** to your agent name (e.g. `Agent-1`).
4. Do the work. Touch only the files listed under **Scope**.
5. Mark the task `done` when finished.

> Pick the highest-priority unclaimed task in a category you can handle.

---

## Task Board

### Backtesting (`backtesting/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| B1 | Add transaction costs (~0.05-0.1% per trade) | `backtesting/backtester.py`, `backtesting/metrics.py` | — | done |
| B2 | Add Sharpe, Sortino, max drawdown, Calmar ratio metrics | `backtesting/metrics.py`, `backtesting/results.py` | — | done |
| B3 | Add slippage modeling (volume-based) | `backtesting/backtester.py` | — | open |
| B4 | Add benchmark comparison (index vs strategy) | `backtesting/results.py`, `backtesting/backtester.py` | — | open |
| B5 | Monte Carlo simulation for backtest confidence intervals | `backtesting/metrics.py`, `backtesting/results.py` | — | open |
| B6 | Walk-forward backtest: retrain model on expanding window during backtest | `backtesting/backtester.py`, `models/walk_forward.py` | — | open |
| B7 | Strict holdout split: enforce backtest period never overlaps training data | `backtesting/backtester.py`, `main.py` | — | open |

### Risk Management (`signals/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| R1 | Dynamic stop-loss based on ATR | `signals/generator.py` | — | done |
| R2 | Position sizing based on confidence (Kelly criterion) | `signals/generator.py` | — | done |
| R3 | Dynamic take-profit targets using ATR multiples | `signals/generator.py` | Agent-Main | done |
| R4 | Confidence threshold filtering (only trade above X%) | `signals/generator.py` | — | done |
| R5 | Per-direction calibration (separate BUY/SELL/HOLD calibrators) | `signals/calibration.py` | — | open |
| R6 | Portfolio-level risk limits (max drawdown cap, max position count) | `signals/generator.py` | — | open |

### Feature Engineering (`data/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| F1 | Market regime indicators (ADX, volatility regime) | `data/features.py` | — | open |
| F2 | Calendar effects (day of week, month, earnings season) | `data/features.py` | — | open |
| F3 | Cross-asset features (OMXS30 correlation, USD/SEK, EUR/SEK) | `data/features.py`, `data/fetcher.py` | — | open |
| F4 | Volatility features (VIX/VSTOXX correlation) | `data/features.py`, `data/fetcher.py` | — | open |
| F5 | Macro indicators (oil prices, interest rates) | `data/features.py`, `data/fetcher.py` | — | open |
| F6 | Sentiment features (financial news, insider trading from FI) | `data/fetcher.py`, `data/features.py` | — | open |

### Model Improvements (`models/`)

| # | Task | Scope | Claimed by | Status |
|---|------|-------|------------|--------|
| M1 | Multi-horizon output heads (1-7 day predictions) | `models/signal_model.py`, `models/training.py` | — | open |
| M2 | Ensemble models (LSTM + GRU + Transformer) | `models/signal_model.py`, `models/training.py` | — | open |
| M3 | Hyperparameter tuning (systematic search) | `models/training.py` | — | open |
| M4 | Purged cross-validation (gap between train/val folds) | `models/walk_forward.py` | — | open |
| M5 | Holdout ticker validation (exclude one ticker from training, test generalisation) | `models/training.py`, `backtesting/backtester.py` | — | open |

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
| I1 | MLflow experiment tracking: auto-connect via env var, consolidate experiment names, log CLI args as tags | `models/mlflow_tracking.py`, `models/training.py`, `models/walk_forward.py`, `main.py` | Claude | done |
| I2 | Log backtest results to MLflow; add `history` CLI command showing trend across runs | `models/mlflow_tracking.py`, `main.py` | Claude | done |
| I3 | Automated retraining: schedule periodic model retraining with new data | `models/training.py`, `main.py` | — | open |
| I4 | Real-time alerts: notify on high-confidence signals above threshold | `main.py`, `signals/generator.py` | — | open |

---

## Module Ownership Map

To avoid all conflicts, at most one agent should own a module at a time:

| Module | Files | Current owner |
|--------|-------|---------------|
| `data/` | `data/fetcher.py`, `data/features.py` | — |
| `models/` | `models/signal_model.py`, `models/training.py`, `models/walk_forward.py`, `models/mlflow_tracking.py` | — |
| `signals/` | `signals/generator.py`, `signals/calibration.py` | — |
| `backtesting/` | `backtesting/backtester.py`, `backtesting/metrics.py`, `backtesting/results.py` | — |
| `main.py` | `main.py` | — |
