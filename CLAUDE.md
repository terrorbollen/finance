# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Multi-Agent Coordination

When running as one of several parallel agents, follow this protocol before doing any work:

1. **Read `AGENTS.md`** — it is the task board. Read `CONTRIBUTIONS.md` for the full coding and workflow guidelines.
2. **Check active locks** — run `ls .agent-locks/` to see which tasks are currently claimed.
3. **Claim a task atomically** — pick the highest-priority `open` task whose lock does not exist, then run:
   ```bash
   mkdir .agent-locks/TASK_ID   # e.g. mkdir .agent-locks/B12
   ```
   `mkdir` is atomic — if another agent claimed the same task first, this command fails and you pick a different task. Then update **Claimed by** and **Status** in `AGENTS.md`.
4. **Stay in scope** — only edit files listed in your task's **Scope** column.
5. **Mark done and release** — set **Status** to `done` and clear **Claimed by** in `AGENTS.md`, then remove the lockfile:
   ```bash
   rmdir .agent-locks/TASK_ID
   ```
6. **Update CHANGELOG.md** — add an entry describing what changed and why.

## Key documents

| Document | Purpose |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Task board and module ownership map |
| [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) | Full coding standards, testing rules, workflow |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Data flow, stage-by-stage details, module dependency graph |
| [`INVARIANTS.md`](INVARIANTS.md) | Rules that must always hold — read before touching the pipeline |
| [`backtesting/STRATEGY.md`](backtesting/STRATEGY.md) | Holdout discipline, output interpretation, horizon guide |
| [`CHANGELOG.md`](CHANGELOG.md) | History of completed changes |

## Philosophy

We are building a finance tool that generates **reliable, actionable trading signals** for Swedish stocks. The goal is not to build an impressive ML system — it is to make good decisions in the market consistently over time.

Reliability comes before everything else. A signal that is wrong 40% of the time but tells you honestly when it is confident is more valuable than a complex model that looks good on paper but fails in live trading. Every improvement should make the system more trustworthy, not just more sophisticated.

When in doubt, do less. A simpler model that is well-validated beats a clever one that has never been honestly tested.

## Project Overview

A CLI-based trading signal generator for Swedish stocks and indexes (e.g., OMX Stockholm 30). Uses TensorFlow for ML-based predictions and yfinance for market data.

### Signal Output
- **Direction**: Buy / Sell / Hold
- **Confidence score**: Probability/confidence level (0-100%)
- **Price targets**: Entry and exit price predictions

## Architecture

```
finance/
├── main.py                     # CLI entry point
├── data/
│   ├── fetcher.py              # yfinance data retrieval
│   └── features.py             # Technical indicators (MA, RSI, MACD, ATR, OBV...)
├── models/
│   ├── signal_model.py         # LSTM model definition
│   ├── training.py             # Standard training pipeline
│   ├── walk_forward.py         # Walk-forward training with nested MLflow runs
│   ├── losses.py               # Focal loss implementations
│   └── mlflow_tracking.py      # MLflow utilities (setup, logging, querying)
├── signals/
│   ├── generator.py            # Signal generation (ATR stop-loss, Kelly sizing)
│   └── calibration.py          # Confidence calibration (isotonic regression)
└── backtesting/
    ├── backtester.py           # Day-by-day historical simulation
    ├── metrics.py              # Sharpe, Sortino, drawdown, win rate
    └── results.py              # Result dataclasses and CSV/JSON export
```

## Development Environment

- Python 3.13 (managed via `.python-version`)
- Package manager: uv (see `uv.lock`)
- Virtual environment: `.venv/`

## Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies
uv sync --all-extras

# Run the application
uv run python main.py

# Add a new dependency
uv add <package>

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=. --cov-report=html

# Lint with ruff
uv run ruff check .

# Format with ruff
uv run ruff format .

# Type check with mypy
uv run mypy .

# Start MLflow tracking server (Docker)
docker-compose up -d

# View MLflow UI
# Open http://localhost:5001 in browser
```

## MLflow Integration

All training and backtest runs are tracked in a single MLflow experiment named **`trading-signals`**. Runs are differentiated by the `run_type` tag.

### Server setup
```bash
docker-compose up -d   # starts MLflow at http://localhost:5001
```
The code auto-connects via the `MLFLOW_TRACKING_URI` env var (set in `docker-compose.yml`). Without Docker, results fall back to `./mlruns/` (file store).

### What is logged per run type

| `run_type` | Logged by | Params | Metrics | Tags |
|---|---|---|---|---|
| `standard` | `train` | sequence_length, thresholds, epochs, batch_size, num_tickers, class_pct_buy/sell/hold | test_signal_accuracy, test_loss, test_price_mae (+ per-epoch via Keras autolog) | run_type, tickers, holdout_start_date, cli.* |
| `walk-forward` | `train --walk-forward` | window sizes, thresholds, epochs, purge_gap, embargo_gap | mean/std/best/worst val_accuracy; nested child run per window | run_type, tickers, holdout_start_date, cli.* |
| `backtest` | `backtest` | ticker, date range, commission, trading_days, buy_hold_return | h{N}.accuracy, h{N}.win_rate, h{N}.net_return, h{N}.sharpe, h{N}.sortino, h{N}.max_drawdown, h{N}.calmar, h{N}.price_mae, h{N}.trade_count, h{N}.win_rate_pvalue, h{N}.buy/sell/hold_precision/recall/support | run_type, ticker |

CLI args are always logged as `cli.*` tags (e.g. `cli.epochs`, `cli.tickers`) so every run is fully reproducible from its MLflow record.

**Keras autologging** is enabled for standard and walk-forward training runs (`mlflow.keras.autolog()`). It automatically captures all Keras training metrics per epoch, model summary, and system metrics. The manual `log_training_history` call is kept alongside for backwards compatibility.

### Querying runs
`models/mlflow_tracking.py` exposes:
- `setup_mlflow()` — call before any logging; handles URI resolution
- `training_run()` — context manager wrapping `mlflow.start_run`
- `get_recent_runs(run_type, ticker, max_results)` — used by `main.py history`
- `get_best_run(experiment_name, metric)` — returns the run with the best value for a given metric

### history command
```bash
uv run python main.py history                          # last 20 backtests
uv run python main.py history --ticker VOLV-B.ST       # filter by ticker
uv run python main.py history --type standard          # training runs
```

## Backtest Output Guide

See [`backtesting/STRATEGY.md`](backtesting/STRATEGY.md) for the full guide: column definitions, statistical significance rules, equity curve export, horizon recommendations, and what makes a result trustworthy before acting on it.

## Dependencies

- **tensorflow**: Machine learning framework
- **yfinance**: Yahoo Finance market data downloader
- **mlflow**: Experiment tracking and model registry
- **scikit-learn**: Confidence calibration (isotonic regression) and class weighting

### Dev Dependencies

- **pytest**: Testing framework
- **pytest-cov**: Code coverage
- **ruff**: Linting and formatting
- **mypy**: Type checking
- **pandas-stubs**: Type stubs for pandas
