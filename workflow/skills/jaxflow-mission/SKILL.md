---
name: jaxflow-mission
description: Use when the owner asks to start, update, or finish a mission -- their own declared multi-phase objective with milestones, separate from any single jaxflow run -- or says "/jaxflow-mission", "start a mission", "mission status", "finish the mission". Tech-lead-only; never run by a builder or reviewer harness.
---

# jaxflow-mission

A mission is the owner's own declared objective (e.g. "ship these 6 phases tonight"): a
name, a one-sentence goal, a milestone checklist, and a free-text status line the tech
lead keeps current. Opt-in only -- never inferred, never started by a build or review
on its own (`.local/docs/specs/2026-09-19-jaxflow-mission-spec.md` §3).

## Step 1 -- gather what's missing

If the owner already gave the name, goal, and milestones, use them as stated; ask only for
what's missing. Otherwise ask, in this order:

1. **Name** -- short.
2. **Goal** -- one sentence.
3. **Milestones** -- a list, at least one, at most 12; a title can't be a bare number
   (numbers are reserved for `mark`'s own index shorthand, spec §6 Decision 5) and no
   two titles may be the same (case/whitespace-insensitive).

## Step 2 -- confirm, then start

Show the assembled command and wait for the owner's ok before running anything -- same
"nothing written before the owner says ok" gate `jax-init`'s own Step 3 uses:

```
jaxflow mission start --name "<name>" --goal "<one-sentence goal>" \
  --milestone "<title 1>" --milestone "<title 2>" ...
```

Exit 0 with no output means it started. A refusal (`mission-active` -- one is already
running, finish or cancel it first; `malformed name`/`malformed goal`/`malformed
milestone`/`too-many-milestones`; `mission-hub-unreachable` -- the cockpit isn't
reachable at `127.0.0.1:3100`; `mission-hub-rejected: <status>` -- the hub answered and
refused) prints its code to stderr and exits 2 -- read it back to
the owner plainly, don't retry silently with different values.

## Step 3 -- the two-command rule at every milestone

**At every milestone, run BOTH commands below -- never just one:**

```
jaxflow mission mark <milestone> done
jaxflow mission status "<what's true right now>"
```

`mark` only flips that ONE milestone's own state; it never touches the free-text
status line (spec §6 Decision 5). Skipping `status` leaves that line stale, so both
commands run together, every time a milestone lands. `<milestone>` is either the
1-based position shown by `jaxflow mission show` or the exact title.

To mark a milestone as started but not yet done, use `mark <milestone> in-progress`
instead of `done` -- `status` still runs either way.

## Step 4 -- finishing

When every milestone is done (or the objective is abandoned), run exactly one of:

```
jaxflow mission done
jaxflow mission cancel
```

This is the LAST command of the mission -- no further `mark`/`status` calls after it
(there is no active mission left to update).

## Checking in

`jaxflow mission show` prints the active mission's name, goal, status line, and
milestone checklist, or `no active mission` -- it never refuses and never changes
anything. Use it any time to confirm what the owner is seeing on the Mission Control block
before reporting progress.

## Never

- Never call `jaxflow mission start` speculatively "just in case" -- only on the owner's
  explicit request.
- Never invoke this skill, or any `jaxflow mission` command, from inside a builder or
  reviewer run (spec §11 item 5) -- tech-lead-only, like `jax-init`.
- Never attach a build/review run to a mission automatically -- the owner rejected that
  explicitly (spec §3 Non-goals); the mission and jaxflow's own run ledger share no
  connection beyond the tech lead updating both by hand.
