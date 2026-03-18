You are picking up and completing a task from the AGENTS.md task board. Work through every phase in order.

---

## Phase 1 — Pick

**1.1** Read `AGENTS.md` in full. Run `ls .agent-locks/` to see which tasks are already claimed.

**1.2** Pick the **highest-priority open task** (top-to-bottom within each section) that has no lockfile.
If all open tasks are locked, tell the user and stop.

**1.3** Read every file listed in the task's **Scope** column before claiming anything.

**1.4** Claim atomically:
```bash
mkdir .agent-locks/TASK_ID
```
If `mkdir` fails, another agent just claimed it — go back to 1.2 and pick the next task.

Edit `AGENTS.md`: set **Claimed by** = `Claude`, **Status** = `in_progress`.

**1.5** Tell the user:
- Task ID and title
- Scope files
- What needs to be done (based on reading the files)
- Relevant invariants from `INVARIANTS.md`

Ask: **"Shall I proceed?"** — do not write any code until the user confirms.

---

## Phase 2 — Implement

Do the work. Stay within the files listed in the task's **Scope**.

If you spot a bug or improvement outside your scope, note it — add it as a new `open` task in `AGENTS.md` at the end, but do not fix it now.

---

## Phase 3 — Finish

**3.1 Lint**
```bash
uv run ruff check .
```
Fix any errors in your scope files, re-run until clean.

**3.2 Type check**
```bash
uv run mypy .
```
Fix errors in your scope files. If an error is outside your scope, add it as a new open task in `AGENTS.md` instead of fixing it.

**3.3 Tests**
```bash
uv run pytest
```
Fix failures in your scope files. If a failure is outside your scope, stop and report it to the user.

**3.4 Backtest (if applicable)**
If your changes touched `models/`, `data/features.py`, `signals/generator.py`, or `signals/calibration.py`, run:
```bash
uv run python main.py backtest VOLV-B.ST
uv run python main.py history --ticker VOLV-B.ST
```
Report whether Sharpe and NetRet are stable vs the previous run. If not touched, skip this step.

**3.5 Update AGENTS.md and release lock**
- Set **Status** = `done`, **Claimed by** = `—`
- `rmdir .agent-locks/TASK_ID`

**3.6 Update CHANGELOG.md**
Add an entry at the top (below the header):
```
## YYYY-MM-DD

### <Task title>
<2–5 sentences: what changed, why it matters, what problem it solves.>
```

**3.7 Report to user**
- Checks passed / what you fixed
- Whether a backtest was run and result
- What was added to CHANGELOG.md
- Any new tasks opened in AGENTS.md
