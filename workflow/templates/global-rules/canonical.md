## Workflow (global rule)

Tech lead: brainstorm → spec → cold review loop → plan → cold review loop →
`jaxflow build` → tech-lead verification → `jaxflow review --diff` → merge with the
owner's approval. Cross-model review: whoever wrote the spec/plan/diff never reviews
it. The owner's spec gate applies only when the change alters what the user sees or
does; merge approval is always required. Short path for small changes: no new
UI/behavior, ≤50 lines/≤3 files, no migration/dependency/route, no
security/disk-write/allowlist code — skip spec and cold review, ship on a branch with
a test and the owner's "merge". The tech lead executes it — inline when the touched
files are already in context, via a subagent when exploration or a file sweep is
needed — never a builder; this covers both a new small change and a correction (a
post-build verification finding, or a LOW/MEDIUM review finding) that stays within the
same thresholds. TDD is the default. Read the repo's AGENTS.md for the
project-specific way to run this (deploy preset, verification policy).

Review loop. Severity per the reviewer contract. HIGH = shipped as-is it breaks:
product stops, data lost/corrupted, security hole, or the spec/plan cannot be
implemented as written. Only HIGH forces another cold review. The tech lead may
downgrade a HIGH that does not meet that bar, writing the reason in the triage.
- No HIGH left: a subagent applies the accepted findings, then a second subagent reads
  the updated spec/plan end to end for contradictions and fixes them inline. Done, no
  new cold review.
- HIGH: same two subagent steps, then a new cold review.
- Diff review: same rule. Fixes go through the short path; only a HIGH triggers
  another `--diff` review.
- A finding returning for the 3rd time = change the design, not another round.

Builders and reviewers use the saved settings in `/settings` → Agents (runtime,
model, effort). Reviewers use the opposite runtime from whoever wrote the document
under review. If the opposite runtime is not installed, the review runs on the same
runtime in a fresh session with no prior context (still cold).

## Merge approval (global rule)

Tech lead: before merging, ask the owner "May I merge `<branch>` into `<destination>`?".
Any clear agreement is the approval; the owner never repeats the SHA. Pin the branch
head SHA yourself when asking; if the branch moves afterwards, ask again. Only the
tech lead executes the merge.

## Never commit (global rule)

`.local/`, `.jax-os/`, `AGENTS.md`, `CLAUDE.md`, secrets or `.env*` files — except a
committed template such as `.env.example`.

## Project status file (global rule)

`.jax-os/status.md` is written by `jaxflow` as a by-product of every run (`build`,
`review`, `merge`) — never hand-edit it outside a jaxflow run. Keep `.jax-os/`
gitignored (`jax-init` does it). Schema: `workflow/contracts/status-report-contract.md`.

## Delegation and parallelism (global rule)

Mechanical work (schema extraction, log parsing, file sweeps, bulk edits) always goes
to a subagent, never inline. Independent work dispatches as concurrent subagents in
one round; serial execution only when a real dependency forces it.

## Commits (global rule)

Imperative mood, `feat:`/`fix:`/`docs:` prefixes.
