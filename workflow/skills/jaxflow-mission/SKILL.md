---
name: jaxflow-mission
description: Use when the owner asks to start, update, or finish a mission -- their own declared multi-phase objective with milestones, separate from any single jaxflow run -- or says "/jaxflow-mission", "start a mission", "mission status", "finish the mission". Tech-lead-only; never run by a builder or reviewer harness.
---

# jaxflow-mission

A mission is the owner's own declared objective (e.g. "ship these 6 phases tonight"): a
name, a one-sentence goal, a milestone checklist, and a free-text status line the tech
lead keeps current. Opt-in only -- never inferred, never started by a build or review on
its own (`.local/docs/specs/2026-09-19-jaxflow-mission-spec.md` §3). Tech-lead-only: never
invoke this skill, or any `jaxflow mission` command, from inside a builder or reviewer run.

## Step 1 -- the four fields (one question per message)

Use what the owner's prompt already states; ask only for what is missing, one field per
message, in this order:

| # | Field | Rule | Becomes |
|---|---|---|---|
| 1 | Final result | ONE verifiable sentence of what must be true when the mission ends, at most 280 chars. Example: "MOA-506 in Done: five phases each with a PR open and a diff review approved." | `--goal` |
| 2 | Source of truth | A Linear issue id and/or a spec/plan path. | Mission log header; no CLI field |
| 3 | Milestones | The intermediate deliverables, 1 to 12; a title is never a bare number (numbers are `mark`'s index shorthand) and no two titles match (case/whitespace-insensitive). | `--milestone` (repeatable) |
| 4 | Out of scope | What the agent must not touch even when it looks useful. | Mission log header; no CLI field |

Also ask the **name** (short, `--name`); it is not one of the four fields.

| Situation | Behaviour |
|---|---|
| Field 2 empty or unreachable | REFUSE to start: say "a mission without an issue or spec is just a long chat". Run no command. |
| Field 1 has no observable end state | Ask once for a rewrite; never invent one. |
| Field 3 empty | Ask again (the CLI refuses `malformed milestone` anyway). |
| Field 4 empty | Accept "nothing specific" and record it as exactly that. |

The final result is the criterion the closing report checks against; mark a milestone only
when it advances that sentence. After the four fields ask: **"normal or autonomous?"**
Default is normal (today's behaviour exactly). Normal goes to Step 3; autonomous goes to
Step 2.

## Step 2 -- autonomous dialogue (only if the owner picked autonomous)

Show sections 2.1-2.7 below IN THIS ORDER, in the owner's language, then ask for ONE
explicit "ok". Nothing is written before that ok. Ask the owner for the max-rounds number
(default 6) while showing 2.3. Tell the owner plainly that in v1 the rules are SOFT: you
obey them, jaxflow does not enforce them, and `--checks` runs arbitrary shell, so the
"never" list is a promise, not a sandbox.

**2.1 What it will do alone**

| Action | Condition |
|---|---|
| Open PRs (`jaxflow pr open`) | After a diff review passes. |
| Merge to the DELIVERY target via `jaxflow merge` | Diff review verdict is approve, or approve-with-changes with no HIGH left; checks green; the SHA merged is the reviewed head, or that head plus commits mapped exclusively to accepted LOW/MEDIUM findings. The owner's "ok" here replaces the per-branch merge question (`merge-contract.md`, autonomous mission exception). |
| Apply LOW/MEDIUM findings | Through the short path (`~/.claude/CLAUDE.md` Workflow rule). |
| Run cold reviews and builds | Only through `jaxflow review` / `jaxflow build`. |

**2.2 What it never does**

| Never | Note |
|---|---|
| Land anything on a production target (`dual-branch-pr` production `main`, `release/*` heads, `jaxflow release`) | `merge-contract.md` already needs a distinct agreement; autonomy never supplies it. |
| Deploy | Includes `bubble-buildprint` LIVE. |
| Edit shared config: `~/.jax-os/agent-settings.json`, hooks, `~/.claude/settings.json` | |
| Change the acceptance criteria of a plan or spec | Rewording a finding's fix is fine; moving the bar is not. |
| Choose new work outside the milestones | |
| Delete unmerged branches | |

**2.3 Hard stops.** The phase is marked "waits on you": write the reason and the decision
needed to the mission log, put a digest in the status line. Continue on OTHER milestones
only if they are independent of the stopped one.

| Stop | Definition |
|---|---|
| Max rounds reached | Default 6 (the number agreed in this dialogue). Counting key: milestone + review kind (spec, plan, diff): up to three independent counters per milestone. EVERY `jaxflow review` dispatch of that kind for that milestone consumes one: correction rounds (`--since`), reruns after a crash or malformed report, reviews of a renamed/replaced document or rebuilt branch. The counter follows the milestone, never the file path or branch name, and is counted from the mission log's round list (jaxflow's own chain is keyed by path/branch). Example: P2 has spec rounds 1-2 and plan rounds 1-3; the plan is replaced by a split plan: its first review is plan round 4. A 6th review that approves still closes the phase; the limit stops dispatching a 7th of that kind. |
| Criteria-only fix | A finding that can only be fixed by changing the acceptance criteria. |
| Verify twice | A build fails `--verify` twice on the same plan. |
| Unlisted refusal | Any jaxflow refusal outside the routine table in 2.7. |
| Production | Anything touching a production target. |

**2.4 Repetition brake (observational in v1).** The tally is the line `tally: N findings
— R repeated (F2≈prev F3 0.34), K new` that `jaxflow result <review_run_id>` prints for a
review whose previous ok round exists (`scripts/jaxflow_cli.py:520-522`; it comes from
`payload["tally"]` of the reviewer's run-finished event, built by
`scripts/jaxflow_workerkit.py:98` `build_review_tally`). Read it right after each review
round, when you write that run id into the mission log's round counter. Copy every pair
with score >= 0.60 into ONE brake note in the mission log, with both run ids (this round
and the previous): "brake would have fired here: <finding ids, score, run ids>". The tally
can be absent (no previous round, no findings, classifier off, more than 100 pairs):
absent = no note. The tally changes no behaviour; continue exactly as without it.
Replanning (changing the approach while the baseline acceptance criteria stay locked) is
your ordinary judgment inside the hard stops. Max rounds is the only deterministic brake.

**2.5 Closing report.** Written in full to the mission log BEFORE `mission done`/`cancel`,
copied as a Linear comment on the source issue when there is one; the last `mission
status` carries a one-line digest.

| Block | Content |
|---|---|
| Delivered | PRs and merges with numbers. |
| Waits on you | Each paused phase, the reason, the decision needed. |
| Stopped and why | Which brake fired (2.3 row or 2.4 "would have"). |
| Not verified | Only when a source of truth was unreachable; names it. |

The status line caps at 280 chars: it carries digests only, never the report.

**2.6 Mission log.** One Markdown file per mission: `.local/missions/<mission-id>.md` in
the control repo, where `<mission-id>` is the integer `jaxflow mission start` prints
(`mission <id>`). Untracked (never commit `.local/`). The Header is created right after a
successful `mission start` in BOTH modes (Step 3); every other section below is
autonomous-only and appended as you go. It is yours alone; jaxflow and the hub never read it.

| Section | Written when | Content |
|---|---|---|
| Header | start (both modes) | Final result, source of truth, milestones, out of scope, mode (normal/autonomous), max rounds (autonomous only). |
| Baseline per phase | phase start | A VERBATIM copy of the phase's acceptance-criteria section (spec acceptance criteria or the plan's per-task expected results) plus the file path and its `sha256sum`. Later reviews and any replan are judged against this copy, never the current file. |
| Merge triage per delivery | before each `jaxflow merge` | The full `<full-sha> -> F#` list and every `F# -> rationale` dismissal (as `merge-contract.md` requires in the merge question), plus the pinned head. |
| Round count per phase | each dispatch | Review run ids in order, per milestone and kind. |
| Pauses | hard stop | Reason, decision needed, affected milestone, "would have" notes. |
| Closing report | end | The blocks of 2.5. |

**2.7 Routine refusals (the only retries allowed without a hard stop)**

| Refusal | Permitted response |
|---|---|
| `checks-failed` | Read the output, fix via the short path, ONE retry; a second failure is a hard stop. |
| `sha-mismatch` / head moved after the pinned review | Re-pin to the reviewed head (or that head plus commits mapped to accepted LOW/MEDIUM findings); if a new review is needed it counts as a round. |
| `github-unreachable`, `mission-hub-unreachable` | One retry after a minute; then hard stop. |
| Mission refusals (`mission-active`, `unknown-milestone`, ...) | Never retried silently; report to the owner. |
| Anything else | Hard stop (2.3 "Unlisted refusal"). |

## Step 3 -- confirm, then start

Show the assembled command and wait for the owner's "ok" before running anything (same
"nothing written before the owner says ok" gate as `jax-init` Step 3). In autonomous mode
the ok to Step 2 is that gate; do not ask a second time.

```
jaxflow mission start --name "<name>" --goal "<final result>" \
  --milestone "<title 1>" --milestone "<title 2>" ... [--autonomous]
```

On success it prints one line, `mission <id>`. In BOTH modes, create
`.local/missions/<id>.md` with the Header section NOW (final result, source of truth,
milestones, out of scope, mode, max rounds if autonomous); fields 2 and 4 live nowhere
else. Only the autonomous-only sections of 2.6 (merge triage, round counters, pauses,
brake notes) are conditional on autonomous mode. A refusal (`mission-active` -- one
is already running, finish or cancel it first; `malformed name`/`malformed goal`/
`malformed milestone`/`too-many-milestones`/`malformed autonomous`; `mission-hub-unreachable`
-- the cockpit is not reachable at `127.0.0.1:3100`; `mission-hub-rejected: <status>`)
prints its code to stderr and exits 2: read it back to the owner plainly, never retry
silently with different values.

## Step 4 -- the two-command rule at every milestone

**At every milestone, run BOTH commands -- never just one:**

```
jaxflow mission mark <milestone> done
jaxflow mission status "<what's true right now>"
```

`mark` only flips that ONE milestone's state; it never touches the status line. Skipping
`status` leaves that line stale, so both run together, every time a milestone lands.
`<milestone>` is the 1-based position shown by `jaxflow mission show` or the exact title.
To mark a milestone started but not done use `mark <milestone> in-progress`; `status`
still runs. Inside an autonomous mission the status line is a digest (<= 280 chars) and the
detail goes to the mission log.

## Step 5 -- finishing

When every milestone is done (or the objective is abandoned), first write the closing
report (autonomous missions: 2.5), then run exactly one of:

```
jaxflow mission done
jaxflow mission cancel
```

This is the LAST command of the mission -- no further `mark`/`status` after it (there is no
active mission left). `mission cancel` is also how the owner revokes an autonomous grant.

## Checking in

`jaxflow mission show` prints the mission's name and goal, `id: <n>`, `mode:
normal|autonomous`, the status line and the milestone checklist, or `no active mission`.
It is read-only but CAN refuse: `mission-hub-unreachable` on a transport failure or a
non-ok hub reply (`scripts/jaxflow_cli.py:380-387`). Use it to confirm what the owner sees
on the Mission Control block before reporting progress. Before relying on the autonomous
merge exception you must have a successful `show` that printed `mode: autonomous`; if it
refuses, fall back to the normal merge question (pinned SHA, owner's agreement).

## Never

- Never call `jaxflow mission start` speculatively -- only on the owner's explicit request.
- Never invoke this skill, or any `jaxflow mission` command, from inside a builder or
  reviewer run -- tech-lead-only, like `jax-init`.
- Never attach a build/review run to a mission automatically; the only link is the
  `mission_id` jaxflow adds to `merge-approved` during an autonomous mission, and the
  hand-kept mission log.
- Never use `--autonomous` unless the owner chose autonomous and gave the single "ok".
