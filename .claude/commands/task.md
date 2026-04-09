You are picking up and completing a task from the TASKS.md task board. Work through every phase in order.

---

## Phase 1 — Pick

**1.1** Read `TASKS.md` in full. Run `ls .agent-locks/` to see which tasks are already claimed.

**1.2** Pick the **highest-priority open task** (top-to-bottom within each section) that has no lockfile.
If all open tasks are locked, tell the user and stop.

**1.3** Read every file listed in the task's **Scope** column before claiming anything.

**1.4** Claim atomically:
```bash
mkdir .agent-locks/TASK_ID
```
If `mkdir` fails, another agent just claimed it — go back to 1.2 and pick the next task.

Edit `TASKS.md`: set **Claimed by** = `Claude`, **Status** = `in_progress`.

**1.5** Run `/health-check` to verify the environment is ready before writing any code.
If health-check reports any FAIL items, stop and resolve them first. WARN items can be noted but do not block work.

**1.6** Tell the user:
- Task ID and title
- Scope files
- What needs to be done (based on reading the files)
- Relevant invariants from `INVARIANTS.md`
- Health check result summary

Ask: **"Shall I proceed?"** — do not write any code until the user confirms.

---

## Phase 2 — Design tests first

Before writing any implementation code, write the tests that will verify the fix or feature.

**What makes a good test here:**
- It catches the exact bug described in the task, or verifies the exact behaviour the feature introduces
- It would have caught this bug in production if it had existed before
- It uses real data or realistic fixtures — do not mock model internals or the backtester in ways that decouple the test from production behaviour
- It fails before your fix and passes after — if it passes before you touch anything, it's not testing what you think

**What to skip:**
- Do not write tests just to increase coverage. Coverage is not the goal.
- Do not test things that can't break (e.g. that a variable equals itself, that a constructor runs without error)
- Do not test framework behaviour (TensorFlow, pandas, yfinance) — only test our code

Add tests to the appropriate file under `tests/`. Run them to confirm they fail:
```bash
uv run pytest tests/<your_test_file>.py -v
```
They should fail at this point. If they pass before the fix, rethink what you're testing.

---

## Phase 3 — Implement

Do the work. Stay within the files listed in the task's **Scope**.

Run your tests after each meaningful change:
```bash
uv run pytest tests/<your_test_file>.py -v
```

If you spot a bug or improvement outside your scope, use `/add-task` to add it to `TASKS.md` — do not fix it now.

---

## Phase 4 — Finish

**4.1 Lint**
```bash
uv run ruff check .
```
Fix any errors in your scope files, re-run until clean.

**4.2 Type check**
```bash
uv run mypy .
```
Fix errors in your scope files. If an error is outside your scope, use `/add-task` to log it in `TASKS.md` instead of fixing it.

**4.3 Tests**
```bash
uv run pytest
```
Fix failures in your scope files. If a failure is outside your scope, stop and report it to the user.

**4.4 Backtest comparison (if applicable)**
If your changes touched `models/`, `data/features.py`, `signals/generator.py`, or `signals/calibration.py`, run `/backtest-compare` on `VOLV-B.ST`.

A change is worth keeping only if **Sharpe and Net Return both improve or hold**. If the result is REGRESSION, do not mark the task done — investigate and fix or revert. If not touched, skip this step.

**4.5 Release lock and close task — MANDATORY, do this before anything else in this step**

> Skipping this step leaves a stale lock that blocks future agents from picking up the task and makes the task board unreliable. Always do this, even if the task turned out to be trivial or already done.

```bash
rmdir .agent-locks/TASK_ID   # e.g. rmdir .agent-locks/M6
```

Then update `TASKS.md`:
- **Status** → `done`
- **Claimed by** → `—`

**4.6 Update CHANGELOG.md**
Add an entry at the top (below the header):
```
## YYYY-MM-DD

### <Task title>
<2–5 sentences: what changed, why it matters, what problem it solves.>
```

**4.7 Report to user**
- Checks passed / what you fixed
- Whether a backtest was run and result
- What was added to CHANGELOG.md
- Any new tasks opened in TASKS.md
