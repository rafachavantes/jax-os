# Status & Report Contract

Format reference only — role duties live in the role contracts.

## PURPOSE

- `report.md` = the evidence of a run. Machine-validated header + human-read body.
- `status.md` = the project card on the Jax OS dashboard. Advisory only.

## REPORT FILE (report.md)

**Path:** `<repo>/.local/reports/<run_id>.md` — one file per run. Builder runs write it
themselves; reviewer runs emit it as their final message and the wrapper writes it.
Once the wrapper finalizes the file it is immutable; a correction requires a new run.

**Frontmatter grammar (the wrapper validates it in two stages):**

- Stage 1 (strict): a `---`-bounded block, UTF-8, one field per line, flat `key: value`
  split on the FIRST `": "`. A duplicate key, an unknown key, a malformed line, or a key
  set that is not exactly `run_id`/`project`/`role`/`phase`/(`result` or
  `verdict`)/`summary` rejects this candidate.
- Stage 2 (loose): entered whenever stage 1 does not yield exactly one fully valid
  candidate. No `---` block is required. The wrapper scans the whole report text, line
  by line, for its OWN role's label (`result:` for a builder, `verdict:` for a
  reviewer) and for a `summary:` label — markup optional (`**result**:`, `- result:`,
  `### result:` all resolve to the plain label), case-insensitive, a fenced or
  blockquoted line never counted. `run_id`/`project`/`role`/`phase` are NOT checked at
  this stage — jaxflow already knows all four from the run that produced the report's
  path. An unknown line is prose, never validated.
- Either stage: the label's value must be one of the unchanged enums below; `summary`
  must be non-empty, ≤300 characters, and free of control characters.
- Process exit 0 with a missing or invalid report never makes `contract_status` "ok" —
  that describes the report file only; a builder run's own `result` may still be
  resolved separately from jaxflow's own verify evidence (builder runs only; reviewers
  never get this).

A builder's report carries `result`, never `verdict`; a reviewer's carries `verdict`,
never `result` — see `builder-contract.md`/`reviewer-contract.md` for each role's own
concrete frontmatter example. This file states the grammar both roles share; it names
neither key.

`summary` is the only report content that machines forward (it is what reaches
the owner's Telegram). Make it carry the decision-relevant fact: outcome + the one thing
that matters. Detail belongs in the body.

**Body (free markdown, read by the tech lead and the owner):**

- Builder: what was done (per handoff task), deviations from the plan, resources
  created and their teardown status, and — when the run is blocked — the concrete
  questions that stopped the run. `<repo>/.local/reports/<run_id>.tests.txt`
  format: for each non-`none` command field, test then build (each field is one
  shell command string): a line `COMMAND: <the command, secrets redacted>`, the
  captured stdout+stderr after mandatory redaction (may be empty), a line `EXIT:
  <code>`; every credential/token/secret/signed-URL value in the command line,
  stream, and report excerpt is replaced by `[REDACTED]` — persisted evidence is
  never an unredacted raw stream; a `none` field is not a command and is omitted;
  if both fields are `none`, the file contains exactly the marker sentence `No
  test/build command was required by this handoff.`. The report body carries, per
  command: the redacted command, the exit code, and a bounded excerpt (tail ≤50 lines).
- Reviewer: the reviewed commit range `<base-sha>..<head-sha>`, then findings numbered
  `F1, F2, …`, each with severity `HIGH | MEDIUM | LOW`, exact location, and a concrete
  fix. Verdict rule: any HIGH ⇒ `reject`; only LOW/MEDIUM ⇒ `approve-with-changes`;
  nothing ⇒ `approve`.

## STATUS FILE (status.md)

**Path:** `<repo>/.jax-os/status.md` (gitignored). Existing dashboard schema, unchanged:

```markdown
---
project: <display name>
stage: spec | build | review | test | ship
gate: awaiting-approval | blocked        # OPTIONAL — omit when nothing waits on a human
builder: <free text>
branch: <branch>
tmux: <session name, optional>
flag: <badge, optional>
updated: <ISO-8601 with offset>
---

## Now
First paragraph after "## Now" surfaces on the card.
```

**`gate` is a commitment, not an activity.** Write it ONLY when a human decision is pending:
`awaiting-approval` (someone owes an approval) or `blocked` (needs a human now). Omit the field
otherwise — absence is the normal case. Live activity is observed from agent events, never
declared here.

Do not write `state:`. If a file still has it from before, `gate` wins when both are present.

**Role of this file:** advisory, best-effort, dashboard-only. A stale file makes only
the dashboard card stale — nothing in the workflow blocks on it. It is NEVER workflow
state; the run's truth is `report.md` plus the wrapper's events.
