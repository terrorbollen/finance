# Contributing

## Philosophy

This project prioritises reliability over sophistication. Before adding anything, ask: does this make the signals more trustworthy, or just more complex? A simpler, well-validated change beats a clever one that hasn't been honestly tested.

## Before you start

1. Read `CLAUDE.md` and `AGENTS.md`.
2. If you're working alongside other agents, claim a task in `AGENTS.md` before touching any files.
3. Understand the existing code in the module you're changing before modifying it.

## Picking up a task

- Tasks are tracked in `AGENTS.md`, ordered by priority within each section.
- Pick the top unclaimed task in a module no one else currently owns.
- Claim it by setting **Claimed by** to your name and **Status** to `in_progress`.

## Scope discipline

- Only touch files listed in your task's **Scope** column.
- Do not refactor, rename, or "improve" code outside your scope, even if it looks like it needs it.
- If you spot a bug or improvement outside your scope, add it as a new row in `AGENTS.md` with status `open` and move on.

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
3. Mark the task **done** in `AGENTS.md`, clear **Claimed by**, and release your module in the ownership map.
4. Add an entry to `CHANGELOG.md` — what changed and why, not just which files.
