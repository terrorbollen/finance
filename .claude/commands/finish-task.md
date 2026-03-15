You are completing a task. Run through every step below in order. Do not skip any step, even if it seems unlikely to have issues.

## Step 1 — Lint
Run: `uv run ruff check .`

If there are errors, fix them, then re-run until clean. Report what you fixed.

## Step 2 — Type check
Run: `uv run mypy .`

If there are errors, fix them, then re-run until clean. Report what you fixed.

If a mypy error is in a file outside your task's scope, add it as a new open task in `AGENTS.md` (do not fix it) and note it in your report.

## Step 3 — Tests
Run: `uv run pytest`

If tests fail, fix the failures in files within your scope. If a failure is outside your scope, report it to the user and stop — do not proceed until the user decides how to handle it.

## Step 4 — Backtest check (if applicable)
If your changes touched any of: `models/`, `data/features.py`, `signals/generator.py`, `signals/calibration.py`

Run a before/after comparison is not possible at this point, but at minimum run:
```
uv run python main.py backtest VOLV-B.ST
```
Check that Sharpe and NetRet are not significantly worse than the last run shown by:
```
uv run python main.py history --ticker VOLV-B.ST
```
Report the comparison to the user.

If your changes did not touch those files, skip this step and say so.

## Step 5 — Update AGENTS.md
Edit `AGENTS.md`:
- Set the task **Status** to `done`
- Clear **Claimed by** (set to empty)
- Remove yourself from the **Current owner** column in the Module Ownership Map

If you spotted any bugs or improvements while working that were out of scope, add them as new `open` rows now.

## Step 6 — Update CHANGELOG.md
Add an entry at the top of `CHANGELOG.md` (below the header, above the previous entry).

Format:
```
## YYYY-MM-DD

### <Task title or short description>
<2–5 sentences: what changed, why it matters, what problem it solves. Not a file list.>
```

Use today's date. Focus on the *why*, not the *what*.

## Step 7 — Final report
Tell the user:
- All checks passed / what you fixed
- Whether a backtest was run and what the result was
- What was added to CHANGELOG.md
- Any new tasks you opened in AGENTS.md
