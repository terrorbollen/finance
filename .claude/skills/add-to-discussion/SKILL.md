---
name: add-to-discussion
description: Add an idea, architectural question, or deferred decision to DISCUSSION.md. Use when something worth tracking comes up in conversation but isn't ready to be a TASKS.md ticket — e.g. "should we switch to TCN?", "is this feature worth adding?", open research questions.
allowed-tools: Read, Edit, Write, Bash
---

# Add to Discussion

Capture an idea or open question in `DISCUSSION.md` so it isn't lost between sessions.

---

## Step 1 — Understand the entry

Extract the following from the conversation. Ask if anything is unclear:

1. **Title** — one short phrase (e.g. "TCN vs LSTM architecture")
2. **Context** — what prompted this? (a performance problem, a question, a trade-off noticed)
3. **Key points** — what was said? Include concrete facts, numbers, or code references if relevant
4. **Status** — pick one:
   - `open` — no decision yet, worth revisiting
   - `deferred` — good idea, not the right time
   - `decided: do it` — agreed to proceed, needs a TASKS.md ticket
   - `decided: skip` — consciously chose not to do it, with a reason

---

## Step 2 — Read DISCUSSION.md

Read `DISCUSSION.md` to check if this topic already has an entry. If it does, update the existing entry rather than adding a duplicate.

---

## Step 3 — Write the entry

Append to `DISCUSSION.md` using this format:

```markdown
## <Title>
_<date> · status: <status>_

**Context:** <1-2 sentences on what prompted this>

**Key points:**
- <point>
- <point>

**Conclusion:** <current thinking, or "no decision yet">
```

---

## Step 4 — Confirm

Print the entry that was added and confirm it is in `DISCUSSION.md`.
