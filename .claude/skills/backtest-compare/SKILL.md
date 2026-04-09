---
name: backtest-compare
description: Run a backtest and compare results against the previous run for the same ticker. Prints a before/after table of Sharpe, Sortino, net return, win rate, max drawdown, and trade count. Use after any change to models/, data/features.py, or signals/ to verify the change is an improvement.
allowed-tools: Read, Grep, Glob, Bash
---

# Backtest Compare

Run a new backtest for the specified ticker, fetch the most recent previous run from MLflow history, and produce a structured comparison table.

**Default ticker:** `VOLV-B.ST` (use whatever ticker the user specifies, or this default if none given).

---

## Step 1 — Capture the previous run

Fetch the last backtest result for the ticker from MLflow history before running a new one:

```bash
uv run python main.py history --ticker <TICKER> --type backtest
```

Record the most recent run's key metrics: Sharpe, Sortino, net return, win rate, max drawdown, trade count, and the horizon used. Note the run ID and date. If no previous run exists, note that this will be the baseline.

---

## Step 2 — Run the new backtest

```bash
uv run python main.py backtest <TICKER>
```

Capture all output. Note: the backtester enforces the holdout date from the model config — do not pass `--no-strict-holdout`.

---

## Step 3 — Fetch the new run from history

```bash
uv run python main.py history --ticker <TICKER> --type backtest
```

The most recent entry is the run you just completed.

---

## Step 4 — Print the comparison table

Print a table in this format for each horizon that appears in both runs:

```
=== Backtest Comparison: <TICKER> ===
Previous run: <date / run_id>
Current run:  <date / run_id>

Horizon  Metric          Before      After       Delta
-------  -----------     --------    --------    -------
5d       Sharpe          X.XX        X.XX        +/-X.XX
5d       Sortino         X.XX        X.XX        +/-X.XX
5d       Net Return      +XX.X%      +XX.X%      +/-XX.X%
5d       Win Rate        XX.X%       XX.X%       +/-X.X%
5d       Max Drawdown    -X.X%       -X.X%       +/-X.X%
5d       Trades          N           N           +/-N
...      (repeat for each horizon)
```

Use colour/emoji if supported: green/+ for improvement, red/- for regression.

---

## Step 5 — Verdict

Apply the trust criteria from `backtesting/STRATEGY.md`:

| Criterion | Threshold |
|---|---|
| Trade count | ≥ 30 trades |
| Win rate significance | p < 0.05 |
| Net return | > 0 after costs |

Print one of:

- **IMPROVEMENT** — Sharpe and Net Return both increased, no trust criteria violated
- **REGRESSION** — Sharpe or Net Return decreased; describe which and by how much
- **INCONCLUSIVE** — trade count below threshold or results are mixed; describe why
- **BASELINE** — no previous run to compare against; this is the new baseline

If result is REGRESSION, recommend reverting the change unless there is a specific reason to accept the regression (e.g. trade count was unrealistically high before, or drawdown decreased substantially).

---

## Multi-ticker validation (optional)

If the user asks for a multi-ticker comparison, or if the initial result is INCONCLUSIVE, repeat steps 1–4 for a second ticker (e.g. `SEB-A.ST` for a flat-market check or `ERIC-B.ST` for a trending-market check). Per `STRATEGY.md`, results should be consistent across ≥ 3 tickers before a change is considered validated.
