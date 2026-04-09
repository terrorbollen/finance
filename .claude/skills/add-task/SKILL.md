---
name: add-task
description: Add a well-formed task entry to TASKS.md. Asks structured questions to extract the root cause, scope, and fix, then writes an entry that matches the quality bar of existing tasks. Use whenever you spot a bug or improvement that is out of scope for the current task.
allowed-tools: Read, Edit, Bash
---

# Add Task

Add a new, well-formed task to `TASKS.md`. The goal is to write an entry at the quality bar of the existing tasks — specific enough that any agent can pick it up without asking questions.

---

## Step 1 — Gather information

If the user has already described the task in sufficient detail, extract the following from their description. If anything is missing, ask:

1. **What section does this belong to?**
   - `Backtesting (backtesting/)`, `Risk Management (signals/)`, `Feature Engineering (data/)`, `Model Improvements (models/)`, `Signal Logic (signals/)`, `Execution Strategy`, `Infrastructure`, `Portfolio`

2. **What is the symptom?** (what breaks, what behaves wrong, what is missing)

3. **What is the root cause?** (the specific code path, line, or design decision that causes it)

4. **What files are in scope?** (only list files that need to change — be precise)

5. **What is the fix?** (concrete enough that an agent can implement it without guessing)

6. **What is the impact if left unfixed?** (silent wrong results? crash? misleading metrics?)

---

## Step 2 — Determine the next task ID

Read `TASKS.md` and find the highest existing ID in the relevant section:
- Backtesting: `B` prefix
- Risk/Signal: `S` prefix
- Feature Engineering: `F` prefix
- Model: `M` prefix
- Execution: `E` prefix
- Infrastructure: `I` prefix
- Portfolio: `P` prefix

Assign the next sequential ID (e.g. if highest is `M8`, new task is `M9`).

---

## Step 3 — Write the task description

Format the task description following the existing high-quality entries in TASKS.md. The description in the **Task** column should:

- Start with a bold one-line title summarising the problem: `**Short title here.**`
- Follow with 2–4 sentences explaining:
  - The specific code path or variable that is wrong
  - Why it produces incorrect results (the mechanism, not just "it's wrong")
  - What the fix is (concrete — name the variable, function, or logic to change)
- End with a unit test requirement if the fix is non-trivial: `Needs a unit test to confirm.`

Look at tasks M6, M7, I9, P1, P2 in `TASKS.md` as quality benchmarks.

---

## Step 4 — Add to TASKS.md

Insert the new row into the correct section table in `TASKS.md`:

```
| <ID> | <full description> | <scope files, comma separated> | — | open |
```

Place it at the **bottom** of the relevant section (lowest priority until triaged).

---

## Step 5 — Confirm

Print the final task entry and confirm it was added. If this task was discovered while working on another task, remind the agent not to fix it now — stay in scope.
