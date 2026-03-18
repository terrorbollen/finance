# Trading Signal Generator

A CLI-based ML trading signal generator for Swedish stocks and the OMX Stockholm 30 index. Uses a TensorFlow LSTM model trained on technical indicators to predict Buy / Hold / Sell signals with confidence scores and price targets.

---

## Setup

```bash
# Install dependencies
uv sync --all-extras

# Start the MLflow tracking server (optional but recommended)
docker-compose up -d
```

---

## Quick Start

```bash
# Analyze a single stock
uv run python main.py analyze VOLV-B.ST

# Scan top Swedish stocks for signals
uv run python main.py scan

# Train the model
uv run python main.py train

# Backtest on historical data
uv run python main.py backtest VOLV-B.ST

# View history of backtest and training results
uv run python main.py history
```

---

## CLI Reference

### `analyze <ticker>`
Generate a signal for a single stock.
```bash
uv run python main.py analyze VOLV-B.ST
uv run python main.py analyze VOLV-B.ST --min-confidence 60
```

### `scan [tickers...]`
Scan multiple tickers and rank by confidence.
```bash
uv run python main.py scan
uv run python main.py scan ERIC-B.ST HM-B.ST VOLV-B.ST --min-confidence 55
```

### `train [tickers...]`
Train the LSTM model. MLflow tracking is on by default.
```bash
uv run python main.py train
uv run python main.py train --epochs 100 --batch-size 64
uv run python main.py train --walk-forward              # Walk-forward validation
uv run python main.py train --no-focal-loss             # Use standard cross-entropy
uv run python main.py train --no-mlflow                 # Disable MLflow tracking
```

### `backtest <ticker>`
Run the model on historical data day-by-day. Results are logged to MLflow automatically.
```bash
uv run python main.py backtest VOLV-B.ST
uv run python main.py backtest VOLV-B.ST --horizons 1 3 5
uv run python main.py backtest VOLV-B.ST --start-date 2024-01-01 --end-date 2024-12-31
uv run python main.py backtest VOLV-B.ST --output results.csv
uv run python main.py backtest VOLV-B.ST --no-mlflow              # Skip MLflow logging
uv run python main.py backtest VOLV-B.ST --leverage 2.0           # 2x leverage
uv run python main.py backtest VOLV-B.ST --position-cooldown      # No overlapping positions
uv run python main.py backtest VOLV-B.ST --name production        # Use named model checkpoint

# Periodic retraining during the backtest (simulates production retraining cadence)
uv run python main.py backtest VOLV-B.ST --retrain-every 60       # Retrain every 60 trading days
uv run python main.py backtest VOLV-B.ST --retrain-every 30 --retrain-epochs 30
```

`--retrain-every N` splits the backtest into chunks of N trading days. At each boundary the model is retrained on all data available up to that date, then predictions resume with the updated model. This simulates the realistic scenario of periodic retraining in production. Without this flag, the same initial model is used for the entire period.

### `calibrate [tickers...]`
Fit a confidence calibrator from backtest data so raw model probabilities map to real accuracy rates.
```bash
uv run python main.py calibrate
uv run python main.py calibrate --horizon 3
```

### `history`
Query MLflow and display a trend table of past runs — the primary tool for judging whether the model is improving.
```bash
# Last 20 backtests (all tickers, horizon 5)
uv run python main.py history

# Filter by ticker
uv run python main.py history --ticker VOLV-B.ST

# Different horizon
uv run python main.py history --horizon 1

# Training runs instead
uv run python main.py history --type standard
uv run python main.py history --type walk-forward

# More runs
uv run python main.py history --runs 50
```

Output example:
```
Date                   Ticker         Acc  WinRate  NetRet%  Sharpe   MaxDD%
---------------------------------------------------------------------------
2026-03-14 10:22       VOLV-B.ST    0.521    0.548    +4.21   +0.83    -8.20
2026-03-10 09:15       VOLV-B.ST    0.498    0.512    +1.90   +0.41   -12.50
```

---

## MLflow Workflow

### Starting the server
```bash
docker-compose up -d
# UI available at http://localhost:5001
```
When the server is running, all training and backtest runs are automatically stored there. Without Docker, results are saved locally to `./mlruns/` and are not visible in the UI (but `history` still works).

### Experiment structure
All runs land in the single `trading-signals` experiment. Run type is identified by the `run_type` tag:

| `run_type` | Created by | Key metrics logged |
|---|---|---|
| `standard` | `train` | `test_signal_accuracy`, `test_loss`, `test_price_mae` |
| `walk-forward` | `train --walk-forward` | `mean_val_accuracy`, `best_window_accuracy`, per-epoch metrics |
| `backtest` | `backtest` | `h{N}.accuracy`, `h{N}.win_rate`, `h{N}.net_return`, `h{N}.sharpe`, `h{N}.max_drawdown` |

Walk-forward training also creates nested child runs (one per window) under the parent run.

---

## Fine-Tuning Workflow

> See [`backtesting/STRATEGY.md`](backtesting/STRATEGY.md) for the full evaluation guide — holdout discipline, what makes a result trustworthy, horizon recommendations, and known limitations.

The typical loop for improving the model:

### 1. Establish a baseline
```bash
uv run python main.py backtest VOLV-B.ST
uv run python main.py history --ticker VOLV-B.ST
```
Note the current `h5.accuracy`, `h5.sharpe`, and `h5.net_return`.

### 2. Retrain with a change
```bash
# e.g. more epochs, different loss, different tickers
uv run python main.py train --epochs 100
```

### 3. Backtest again and compare
```bash
uv run python main.py backtest VOLV-B.ST
uv run python main.py history --ticker VOLV-B.ST
```
The `history` table now shows both runs side by side so you can see whether the change helped.

### 4. What to look for

| Metric | What it means | Target direction |
|---|---|---|
| `Acc` | Signal direction accuracy | Higher |
| `WinRate` | Fraction of trades that were profitable | Higher |
| `NetRet%` | Cumulative net return after commission | Higher |
| `Sharpe` | Return per unit of risk | Higher (>1.0 is good) |
| `MaxDD%` | Worst peak-to-trough loss | Less negative |

A change is worth keeping if **Sharpe and NetRet both improve** — accuracy alone can be misleading if the model is making more trades with lower quality.

### 5. Inspect a specific run in the UI
Open `http://localhost:5001`, find the run in the `trading-signals` experiment, and inspect per-epoch loss curves and all logged params to understand exactly what configuration produced the result.

---

## Architecture

```
finance/
├── main.py                     # CLI entry point
├── ARCHITECTURE.md             # Data flow and module dependency graph
├── INVARIANTS.md               # Rules that must always hold in the pipeline
├── data/
│   ├── fetcher.py              # yfinance data retrieval
│   └── features.py             # Technical indicators (MA, RSI, MACD, ATR, OBV...)
├── models/
│   ├── signal_model.py         # LSTM model definition
│   ├── training.py             # Standard training pipeline
│   ├── walk_forward.py         # Walk-forward training
│   ├── losses.py               # Focal loss implementations
│   └── mlflow_tracking.py      # MLflow utilities
├── signals/
│   ├── generator.py            # Signal generation (ATR stop-loss, Kelly sizing)
│   └── calibration.py          # Confidence calibration (isotonic regression)
└── backtesting/
    ├── backtester.py           # Day-by-day historical simulation
    ├── metrics.py              # Sharpe, Sortino, drawdown, win rate
    ├── results.py              # Result dataclasses and export
    └── STRATEGY.md             # Evaluation guide, holdout discipline, findings
```

---

## Contributing / Adding Tasks

Work on this project is coordinated through [`AGENTS.md`](AGENTS.md) (task board and module ownership) and [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md) (coding standards, testing rules, and the full workflow). Before modifying the pipeline, read [`ARCHITECTURE.md`](ARCHITECTURE.md) for data flow and [`INVARIANTS.md`](INVARIANTS.md) for rules that must not be broken.

**If you're working on a task and notice something worth fixing or adding** — a bug, a missing guard, a potential improvement — add it as a new row in the relevant section of `AGENTS.md` with status `open` before moving on. Don't fix it inline; keep your scope focused and let the board track it.

**If you want to pick up a task**, read `AGENTS.md` and `CONTRIBUTIONS.md` first, claim the top unclaimed item in a module no one else owns, and follow the protocol there.

---

## Key Design Decisions

- **Focal loss** is on by default to handle the class imbalance (most days are HOLD)
- **Confidence calibration** maps raw softmax probabilities to real accuracy rates — use `calibrate` before relying on confidence scores
- **Walk-forward training** is the more realistic option: it simulates how the model would actually be retrained in production
- **ATR-based stop-loss** and **Kelly position sizing** are computed at signal generation time, not during backtesting
- MLflow uses the `MLFLOW_TRACKING_URI` env var for server discovery — Docker sets this automatically
