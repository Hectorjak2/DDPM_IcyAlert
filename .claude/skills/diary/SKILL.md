---
name: diary
description: "Record the current session's theory discussions, design decisions, and code-design notes as a short dated entry in docs/diaries. USER-INVOKED ONLY: use this skill exclusively when the user explicitly types /diary or directly asks for a diary entry. Never invoke it proactively, never at the end of a session, and never because the work seems worth recording."
---

# Write a Session Diary Entry

Capture the reasoning from this session as a dated entry in `docs/diaries/`.

## ⛔ The user invokes this skill. You never do.

Write a diary entry **only** when the user explicitly asks for one — by typing `/diary`, or by directly requesting a diary entry in their own words.

Do **not** run it because:

- the session is ending or feels finished
- a lot of ground was covered and it seems worth recording
- a design decision was just made
- the user asked you to update `docs/architecture.md` or any other file
- you are verifying that this skill works

If you think an entry would be valuable, **say so in one line and stop.** Deciding when the session gets written down is the user's call, not yours.

## What this is for

This project keeps two kinds of written knowledge, and they must not blur together:

| Document | Holds | Tense |
|---|---|---|
| `docs/architecture.md` | **Stable facts** — what the code *is*, and the rationale that still applies | Timeless |
| `docs/diaries/DD-MM-YY_HHMM.md` | **The reasoning that produced it** — theory worked through, alternatives rejected, decisions deferred | Dated snapshot |

CLAUDE.md notes that commit messages are not a reliable source of "why". The diary is where the "why" that hasn't yet hardened into architecture goes: the argument, not the conclusion alone.

**This skill writes only to `docs/diaries/`. It never edits `docs/architecture.md`** — it lists promotion candidates instead, and the user decides.

## Steps

### 1. Get the timestamp

One call gives both forms you need:

```bash
date "+file=%d-%m-%y_%H%M label=%d-%m-%y:%H:%M"
```

- `file=` → the filename stem, e.g. `07-09-25_1432` → `docs/diaries/07-09-25_1432.md`
- `label=` → the H1 inside the file, e.g. `# 07-09-25:14:32`

The filename deliberately has no colon (Finder renders `:` as `/` on macOS and it breaks some tooling); the colon form lives inside the file where it is read.

If that path already exists — two runs in the same minute — append `-2`, then `-3`, and so on.

### 2. Ensure the directory

```bash
mkdir -p docs/diaries
```

### 3. Select the essentials

Review the **whole session**, then cut hard. This is the step that determines whether the entry is worth anything.

Apply this test to every candidate item:

> Would a future session make a **worse decision** without this?

If no, drop it. Most of a session fails this test — that is expected and correct.

Things that reliably pass: a piece of theory that took real work to settle; a design choice with a live alternative; a number that was the *evidence* for a conclusion; something deliberately deferred and the condition for revisiting it; a trap that cost time and would cost it again.

Things that reliably fail: what was tried in what order; anything readable from the code or the diff; restatements of `architecture.md`; ideas mentioned but not pursued.

### 4. Write the entry

Use the Write tool with this template. Replace the bracketed guidance with content; keep the section headings verbatim.

```markdown
# DD-MM-YY:HH:MM

**Focus:** [one line — what this session was about]

## Theory
[Substantive technical or mathematical discussion. State the question, the answer,
and the evidence that settled it. Keep the numbers that were the evidence.]

## Design decisions
[What was decided and why. Alternatives rejected, with the reason they lost.
Decisions deliberately deferred, and what should trigger revisiting them.]

## Code design
[Structure: where things now live and why; interfaces or signatures introduced;
invariants a future change must preserve; rough edges knowingly left in place.]

## Open threads
[Unresolved or explicitly deferred. "None." is a valid entry.]

## Promotion candidates
[Items now stable enough for docs/architecture.md, each naming a target section.
Do not edit architecture.md — just list them.]
```

### 5. Report

Print the path written, then list the promotion candidates in the chat so the user can act on them without opening the file.

Entries are **not** gitignored, so the new file will appear in `git status`. Tell the user to commit it. Never run git yourself (see Constraints).

## Brevity rules

Brevity is a hard requirement, not a preference. A long diary entry does not get read, which defeats the point.

- **Max ~5 bullets per section.** A bullet is 1–3 sentences. Whole entry under ~80 lines.
- **Conclusions, not narration.** Record what was established, not the sequence of attempts that got there.
- **Never restate the code.** The repo already says what it does. Record *why*, which it cannot.
- **Keep a number only when the number was the evidence.** Good: "the 0.0075 loss was just the t≳500 bucket; t=5 sat at 0.45." Bad: a table of every measurement taken.
- **Empty sections say `None this session.`** Never pad a section to look complete.
- **Name files as links** — `[models/ddpm.py](../../models/ddpm.py)` — so the entry is navigable from `docs/diaries/`.

## Constraints

- **Never invoke this skill on your own initiative.** See the top of this file. The user calls `/diary`; you do not.
- **Never run git commands.** Project CLAUDE.md forbids it and `.claude/settings.json` denies `Bash(git *)`. If the entry should be committed, say so and stop.
- **Never edit `docs/architecture.md` from this skill.** Promotion is the user's call — list candidates and leave it.
- **Do not invent content.** Only record what actually happened in the session. If a section has no basis in the conversation, it gets `None this session.`
