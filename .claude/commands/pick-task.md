You are picking up a task from the AGENTS.md task board. Follow these steps exactly.

## Step 1 — Read the task board
Read `AGENTS.md` in full. Identify:
- All tasks with **Status** = `open`
- Which modules are currently owned (Module Ownership Map section)

## Step 2 — Choose a task
Pick the **highest-priority open task** whose module is not currently owned by another agent.
Priority order within AGENTS.md is top-to-bottom within each section; sections are ordered by importance in the file.

If all open tasks are in owned modules, report this to the user and stop.

## Step 3 — Read the task's scope files
Before claiming anything, read every file listed in the task's **Scope** column so you understand what you're walking into. Do not propose an approach yet.

## Step 4 — Claim the task
Edit `AGENTS.md`:
- Set **Claimed by** to `Claude`
- Set **Status** to `in_progress`
- Add your agent name to the **Current owner** column in the Module Ownership Map for the relevant module

## Step 5 — Summarise and confirm
Tell the user:
- Which task you claimed (ID and title)
- What the scope is
- Your understanding of what needs to be done based on reading the files
- Any invariants from `INVARIANTS.md` that are relevant to this task

Then ask: "Shall I proceed with the implementation?"

Do not start writing code until the user confirms.
