# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Multi-Agent Coordination

When running as one of several parallel agents, follow this protocol before doing any work:

1. **Read `AGENTS.md`** — it is the task board and module ownership map.
2. **Claim a task** — edit the **Claimed by** cell for your chosen task to your agent name (e.g. `Agent-1`). Pick the highest-priority unclaimed task in a module no other agent currently owns.
3. **Stay in scope** — only edit files listed in your task's **Scope** column. Do not touch `main.py` or other modules unless your task explicitly lists them.
4. **Mark done** — update the task **Status** to `done` when finished.
5. **Release the module** — clear the **Current owner** entry in the Module Ownership Map when done.

> If two agents accidentally pick the same module, the one that claimed it later should back off and choose a different task.

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
# Open http://localhost:5000 in browser
```

## MLflow Integration

All training and backtest runs are tracked in a single MLflow experiment named **`trading-signals`**. Runs are differentiated by the `run_type` tag.

### Server setup
```bash
docker-compose up -d   # starts MLflow at http://localhost:5000
```
The code auto-connects via the `MLFLOW_TRACKING_URI` env var (set in `docker-compose.yml`). Without Docker, results fall back to `./mlruns/` (file store).

### What is logged per run type

| `run_type` | Logged by | Params | Metrics |
|---|---|---|---|
| `standard` | `train` | sequence_length, thresholds, epochs, batch_size, num_tickers | test_signal_accuracy, test_loss, test_price_mae (+ per-epoch) |
| `walk-forward` | `train --walk-forward` | window sizes, thresholds, epochs | mean/std/best/worst val_accuracy; nested child run per window |
| `backtest` | `backtest` | ticker, date range, commission | h{N}.accuracy, h{N}.win_rate, h{N}.net_return, h{N}.sharpe, h{N}.max_drawdown per horizon |

CLI args are always logged as `cli.*` tags (e.g. `cli.epochs`, `cli.tickers`) so every run is fully reproducible from its MLflow record.

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
