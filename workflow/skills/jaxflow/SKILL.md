---
name: jaxflow
description: Use when the tech lead needs to dispatch a cold review (spec, plan or a build's diff), dispatch a builder, merge an approved branch, or check on/read the result of/cancel a run already going. Trigger any time you would otherwise ask a human to run one of those by hand.
---

# jaxflow

One command dispatches every review, build and merge, and reads the outcome back. Never
call a reviewer or builder runtime (`codex`, `claude`, `opencode`) directly — always go
through `jaxflow`.

## Applies to every dispatch

`review`, `build` and `merge` share the rules below. `status`, `result` and `cancel` read
the run ledger only — they need no git repo, no caller identity, and touch no status file.

- **Run it from inside a git repo.** Anywhere else refuses `not-a-git-toplevel`.
- **Caller identity.** jaxflow reads `CLAUDECODE` / `CODEX_THREAD_ID` from the environment
  to know which tech lead is calling. `--from claude|codex` is REQUIRED whenever BOTH are
  set (the nested case — a Codex session dispatched from inside Claude Code inherits both
  markers) or NEITHER is; otherwise it refuses `caller-unknown`. The named side also needs
  its session-id variable set (`CLAUDE_CODE_SESSION_ID` for claude, `CODEX_THREAD_ID` for
  codex) or it refuses `caller-session-missing`. `CLAUDECODE=1` only marks the environment;
  `CLAUDE_CODE_SESSION_ID` carries the id recorded as the run's `caller_session`. On the
  Codex side `CODEX_THREAD_ID` does both jobs.
- **`--phase` on `review`/`build` is a token**, not a title: `[A-Za-z0-9._-]`, 1–64 chars,
  no spaces. Anything else refuses `malformed phase`. (`merge --phase` is the opposite — a
  free-text commit subject; see that section.)
- **The callback line.** Codex callers receive it through native `codex queue` on the
  captured canonical `caller_session`, whether or not a pane was recorded. Claude callers
  receive it through a native `asyncRewake` hook — no tmux pane needed — which wakes the
  originating session and delivers the line on the hook's stderr, wrapped in Claude
  Code's own "Stop hook feedback/blocking error" text; delivery needs the one-time
  `claude-callback` hook installed in `~/.claude/settings.json` (a missing hook prints
  one dispatch-time warning, never blocks dispatch) and the originating session still
  running from dispatch until the run finishes — exiting the session kills its watcher
  immediately, so that notice is lost; a run longer than ~4h gets a "watcher expired"
  wake instead. `status`/`result` is the fallback either way. `--no-callback`
  suppresses both routes. Queue needs an installed CLI that supports that command and an
  accessible existing session; acceptance does not prove receipt. Delivery failure is
  best-effort (worker stderr, then `status`/`result`) with no retry or fallback.
- **`.jax-os/status.md`** is rewritten as a by-product of a finished run and of `merge`,
  always in the CONTROL repo — a run dispatched from inside a linked worktree updates the
  control repo's card, never the worktree's (`build`, `review` and `merge` all share this).
  Best-effort: a repo with no such file, or one whose frontmatter does not parse, is
  skipped with a note, and a newer run's write is never overwritten by an older one.
  Never hand-edit it, and never instruct a builder to touch one — the wrapper is the
  only writer.
- **Runtime capability coverage** (which signals each runtime actually delivers — run
  started/finished, turn-stopped, questions, callbacks, last-tool-used, subagent
  start/stop) is a committed reference table, not a guess:
  `workflow/contracts/runtime-capability-matrix.md`.

## Dispatch a review

```
jaxflow review --spec <path>   # or --plan <path> — exactly one of --spec/--plan/--diff
```

Runs immediately in the background (a new tmux session) and prints a `run_id`. The reviewer
runtime is fixed cross-runtime: Claude callers use Codex, and Codex callers use Claude.
If that opposite agent is switched off in `/settings` -> General, the review runs on the other reviewer runtime (a fresh sealed session, so still cold) and says so: dispatch prints `reviewer_runtime: <runtime> (fallback: <agent> off)`, `jaxflow result <id>` prints it first (before any diagnostics, tally or report), and the finish callback ends with `(fallback: <agent> off)`. A malformed or unreadable `settings.json` refuses `review` and `build` (`settings-malformed` / `settings-unreadable`).
Model and effort default to the saved reviewer settings from `/settings` Agents; `--model` and
`--effort` remain per-call overrides. Other optional flags are `--focus "<text>"` (extra
reviewer instructions), `--phase <token>` (defaults to the target file's stem),
`--no-callback` (suppress the `[JAXFLOW] ...` line), and `--from claude|codex`.

Every reviewer handoff also carries one `threat-model:` line, injected automatically from
the project's own `AGENTS.md` `## Threat model` section (`mode: internal-single-user` or
`mode: public-app`) — no flag needed. `--focus` stays for per-review extras on top of it.

```
jaxflow review --spec <path> --from claude   # dispatching as the Claude Code tech lead
jaxflow review --plan <path> --from codex    # dispatching as the Codex tech lead
```

Spec and plan reviews pass the canonical document path, not the document contents,
to either reviewer runtime. The reviewer reads the document and named evidence
from disk. Claude reviewers receive the directory access they need via `--add-dir`
— deduplicated: the canonical control repo (added when the review or the build under
review originates in a linked worktree), the build worktree for diffs, the ROOT of a
related linked worktree when the reviewed document lives inside one, and a validated
external document's parent only when the document lives outside every granted root.
Never a sibling-repo grant. This does not change the Codex reviewer's or builders' argv.
Directory grants do not permit reviewing unnamed inputs; the reviewer contract still
defines the evidence scope.

Refusals specific to `review --spec|--plan`:

- `dirty-tracked-tree` — **the caller's OWN repo** has a staged or modified tracked file.
  Untracked files are fine. Commit or stash before dispatching.
- `detached-head` — the caller's repo is not on a branch.
- `secret-detected: <path>` — the target resolved to a `.env*`/key/credential-shaped path;
  jaxflow refuses to read it, ever.
- `path-outside-allowlist` — the target is outside `~/repos`.
- `no-reviewer-agent` — neither Claude Code nor Codex is switched on (OpenCode alone cannot review).

On finish (unless `--no-callback`) the caller receives, with `<kind>` = `spec` or
`plan` (and `diff` for `review --diff`), via `codex queue` for Codex or the native
`asyncRewake` hook for Claude (no tmux pane needed). The outcome of a review is the
report's own `verdict` —
`approve` / `approve-with-changes` / `reject` — not a build's `result`.
A fallback review appends ` (fallback: <agent> off)` as the last segment.

```
[JAXFLOW] <kind> <run_id> finished — <verdict> — <summary> — <report_path>
```

## Dispatch a build

```
jaxflow build --plan <path> --phase <token> --branch <name> \
    --whitelist <p1,p2,...> --verify "<cmd>" [--build "<cmd>"] [--base <ref>] \
    [--fallback] [--from claude|codex] [--no-callback]

jaxflow build --resume <run_id> [--fallback] [--from claude|codex] [--no-callback]
```

Reserves a fresh worktree at `~/repos/<project>-<branch-slug>` and dispatches a builder
there. The plan — and the spec it declares — are read by the builder from their ORIGINAL
locations in the validated read scope (control repo, worktree, or an explicitly
validated external document parent); no plan/spec copies are made, and AGENTS.md stays
the only file jaxflow brings into the worktree. Prints a `run_id`, same as `review`.
**Never create that worktree yourself** (by hand or through a worktree skill) — `build`
owns the reservation.

Before the worktree is reserved, jaxflow validates the plan's structure and refuses
`plan-invalid` rather than burning a reservation on a defective plan. There is no
required heading shape any more — no `**Goal:**`/H1 line, no `### Task N:` heading is
checked — the builder reads the whole plan regardless. The one check that remains: a
`**Spec:**` line (`**Spec:**`, `Spec:`, or `spec:`, backticks optional, case-
insensitive), if the plan has one, must resolve to a real, non-secret, in-allowlist file
— including any `#<heading>` fragment it names (see the `**Spec:**` line below).

A plan's `**Spec:**` line may carry a `#<heading>` fragment — `**Spec:** <path>#<heading
text>` — naming the ONE section builders and reviewers are meant to read instead of the
whole document (matches the first heading, any level, whose trimmed text equals the
fragment, case-sensitive). A fragment naming no heading in the resolved document is a
`plan-invalid` defect. No fragment means the whole document, exactly as before.

- `--branch` names a NEW branch — refuses `branch-exists` if it (or its worktree path)
  already exists.
- `--base <ref>` is the ref the new branch starts from; the default is the repo's own
  default branch. Use it to STACK work: a part-2 plan that has to build on part 1's branch
  is `--base feat/thing-p1 --branch feat/thing-p2`. The ref is resolved to a commit SHA
  before anything is created, so a typo costs nothing — an unknown or malformed ref refuses
  `base-invalid`.
- `--whitelist` is a comma-separated list of paths the builder may touch.
- `--verify` is the TEST command: ONE shell command string (pipelines/`&&` chains count as
  one) jaxflow itself re-runs after the builder finishes. Required on a fresh build, and it
  must not be blank; `--resume` reuses the prior command.
- `--build` is the optional BUILD command. Pass it rather than chaining a build into
  `--verify`: both commands always run, each gets its own frame in
  `<worktree>/.local/reports/<run_id>.tests.txt` (redacted), and a failure of EITHER folds
  into the run's `result`. Chaining with `&&` means a failing test skips the build, so you
  never learn whether the tree still compiles — and the builder's handoff would say
  `commands.build: none`, which would be false.
- **Builder profiles.** Save profiles in `/settings` → Agents; jaxflow reads
  `~/.jax-os/agent-settings.json`. Builds use `opencode-builder` with saved Default, or
  Fallback with `--fallback`; provider/model/effort/routing come from the selected profile.
  Native credentials stay in OpenCode, and an `env` credential injects only the named key
  from `$JAXOS_HOME/.env`. Stale native sources refuse
  `agent-settings-source-changed`; alias/model/variant/routing mismatches refuse
  `agent-profile-conflict` — no overlay repair.
- `--resume <run_id>` retries a managed builder that finished `failure` or `blocked` and
  still has its registered worktree and checkpoint. Reuses the recorded branch/worktree;
  plan/phase/whitelist/verify/build/base come from the prior attempt — passing any of
  those fresh-build options is refused. No `--force`. Without
  `--fallback`, the prior profile NAME is kept and its **current** saved config is
  resolved at child start; with `--fallback`, current Fallback. New run id, report and
  callback; previous evidence is not rewritten. Partial tracked/untracked work is kept;
  a checkpoint mismatch refuses instead of resetting. Verify/build always rerun. The
  diff base stays the original base, not the partial HEAD. Cleanup never deletes a
  worktree this attempt did not reserve. `status`/`result` label the persisted
  selection as **resolved selection**, never "launched": the record is observation,
  not proof the child started.

Other refusals and result values:

- `secret-detected: <path>` — `--plan` (or a `**Spec:**` file the plan itself names) resolved
  to a `.env*`/key/credential-shaped path.
- `builder-on-default-branch` — the freshly reserved worktree's branch collided with the
  repo's own default (`main`/`master`/its real default); practically unreachable through
  `build` itself (the branch it just created is never one of those names), but still the
  refusal `jr.preflight()` raises if it ever is.
- A finished run's own `result` (in the callback line, or `jaxflow result <run_id>`) is one
  of `success`/`failure`/`blocked`/omitted (report missing or invalid) — `failure` covers
  BOTH the builder's own claim of failure AND a `success` report whose verification commands
  themselves failed (jaxflow's own verify overrides only this ledger field, never the builder's
  report text).
- `agent-disabled: opencode` — OpenCode is switched off in `/settings` -> General (applies to `--resume` too; nothing is reserved).

On finish (unless `--no-callback`), Codex callers receive a line through `codex queue`;
Claude callers receive it through the native `asyncRewake` hook (no tmux pane needed):

```
[JAXFLOW] build <run_id> finished — <result> — <summary> — <report_path>
```

```
jaxflow build --plan .local/docs/plans/2026-09-06-my-plan.md --phase my-phase \
    --branch feat/my-phase --whitelist src/foo.ts,src/foo.test.ts \
    --verify "pnpm exec vitest run src" --build "pnpm build"
```

Tests only — `--build` is optional, and the handoff then honestly says
`commands.build: none`:

```
jaxflow build --plan .local/docs/plans/2026-09-06-my-plan.md --phase my-phase \
    --branch feat/my-phase --whitelist src/foo.ts,src/foo.test.ts \
    --verify "pnpm exec vitest run src"
```

Part 2 of a split plan, stacked on part 1's branch:

```
jaxflow build --plan .local/docs/plans/2026-09-06-my-plan-p2.md --phase my-phase-p2 \
    --base feat/my-phase --branch feat/my-phase-p2 \
    --whitelist src/foo.ts --verify "pnpm exec vitest run src"
```

From inside a Claude Code session, dispatching its own nested build (naming the caller
explicitly whenever the environment can't infer it on its own):

```
jaxflow build --plan .local/docs/plans/my-plan.md --phase my-phase --branch feat/my-phase \
    --whitelist src/foo.ts --verify "pnpm test" --from claude
```

From inside a Codex session, the same shape, naming the other side:

```
jaxflow build --plan .local/docs/plans/my-plan.md --phase my-phase --branch feat/my-phase \
    --whitelist src/foo.ts --verify "pnpm test" --from codex
```

The worktree, and everything under it (including the builder's own `report.md`), lives until
`jaxflow merge` removes it — `build` itself never cleans up a SUCCESSFUL run's worktree.

## Review a build's diff

```
jaxflow review --diff <builder_run_id> [--focus "<text>"] [--model <id>] [--effort <level>] \
    [--phase <token>] [--from claude|codex] [--no-callback] \
    [--since <prior_diff_review_run_id> | --full "<reason>"] [--reverify]
```

A new stacked or correction build's first `--diff` is still `<merge-base>..HEAD`
(inherited commits included). `--since` only narrows a later correction round on
an already-reviewed branch; a later dispatch base is not permission to shrink
that first review.

Looks up the finished `build` run by its `run_id`. `run_id` must name a **finished** `build` run —
finished, not successful: a build whose `result` is `failure` can still be diff-reviewed (its
verification commands have to pass at review time, which is what `verify-failed` enforces), and a
build whose REPORT was invalid or missing is still reviewable as long as it finished with a
`head_sha` recorded. `unknown-run` now means only: no run with that id, the id names a reviewer
run (spec/plan/diff) instead of a build, the build hasn't finished yet, or a finished build has no
`head_sha`. A `run_id` whose worktree was already removed refuses `worktree-missing` instead. The
handoff also names the build's own plan and the spec it declares (the plan itself when it declares
none) — the same required spec/plan evidence a builder's own handoff names, at their ORIGINAL paths;
a missing or unusable build manifest refuses `unknown-run` with a hint naming the manifest path.
`--phase` defaults to the builder run's own phase (there is no filename to take a stem from, unlike
`review --spec|--plan`). `--no-callback` suppresses the `[JAXFLOW] diff <run_id> finished — ...` line
normally delivered on completion (`codex queue` for Codex, the native `asyncRewake` hook
for Claude).

**Terminal-verdict re-review guard.** A second `review --diff` dispatch against a branch whose most
recent terminal diff-review verdict (`approve`/`approve-with-changes`) is still current refuses
`prior-review-accepted`, naming the prior run id, its verdict and its `head_sha` — no point paying for
a fresh review of a branch already accepted. A diff review still running on that branch always refuses
`review-running` first, regardless of any flag below. A prior `reject` verdict never blocks — dispatch
proceeds as a full review, with an advisory hint suggesting `--since`. Every refusal from either lock
is appended as one line to `<control repo>/.local/runs/refusals.jsonl`.

- `--full "<reason>"` bypasses the terminal-verdict guard and runs a full review anyway. The reason is
  required (blank refuses `full-reason-blank`) and recorded verbatim in the manifest.
- `--since <prior_diff_review_run_id>` bypasses the same guard as a CORRECTION ROUND: it reviews only
  `<prior run's head_sha>..HEAD` instead of `<merge-base>..HEAD`, on top of the same fresh verify. The
  prior run must be a finished `diff` review with a terminal verdict on the SAME branch (its linked
  build must target the same branch too) whose `head_sha` is an ancestor of current `HEAD`, and the
  whole `--since` chain back to its full-range root must check out — every node re-checked, not only
  the one named. Refusal codes (all mean: run a full review instead):
  - `since-invalid` — the `--since` value isn't shaped like a run id.
  - `since-chain-missing` — a node's manifest is missing or unreadable while walking the chain.
  - `since-chain-cycle` / `since-chain-depth` — the chain revisits a run id, or exceeds the bounded walk depth.
  - `since-chain-broken` — the `--since` target itself has no recorded `head_sha`.
  - `since-not-ancestor` — the target's `head_sha` isn't an ancestor of current `HEAD`.
  - `since-lineage-mismatch` — some node's review, or its linked build, targets a different branch.
  - `since-not-terminal` — some node in the chain has no terminal verdict.
  - `since-ancestry-broken` — some node's own `base_sha` isn't an ancestor of its own `head_sha`.
  - `since-root-stale` — the chain's full-range root `base_sha` no longer matches the current merge-base.
  - `since-continuity-broken` — a non-root node's `base_sha` doesn't match its predecessor's `head_sha`.
  - `since-full-conflict` — `--since` and `--full` given together.
  - `full-reason-blank` — `--full` with an empty or whitespace-only reason.
- **Verify reuse.** When current `HEAD`, a clean worktree, the verify/build command strings, and the
  builder's own `.tests.txt` evidence all still match the linked build's finished run, jaxflow skips
  re-running verify/build entirely and the manifest records `verify: {mode: reused, ...}` (printed as
  "verification reused from build `<id>` at `<sha>`"). Any mismatch — a moved `HEAD`, a dirty tree, a
  changed command, missing/malformed `.tests.txt`, or a non-zero exit recorded in it — reruns verify
  exactly as before (`mode: rerun`, reason printed) and a failure of either command still refuses
  `verify-failed` before any reviewer is dispatched. `--reverify` forces a full rerun unconditionally.

A branch that was NOT built through `jaxflow build` has no `run_id`, and therefore no `--diff` path
at all. That is a bypass case: report it, propose the alternative, and wait for confirmation before
reviewing it another way.

```
jaxflow review --diff a1b2c3d4e5f6
jaxflow review --diff a1b2c3d4e5f6 --full "branch changed materially since approval"
jaxflow review --diff a1b2c3d4e5f6 --since <prior_diff_review_run_id>
```

`status`/`result`/`cancel` work identically on a `diff` review's `run_id` as on any other run.

## Open a PR (PR presets)

```
jaxflow pr open <branch> --sha <full-40-sha> --target <base> --title "<text>" \
    [--body-file <path>] [--from claude|codex]
```

Run right after a diff review passes — no separate approval from the owner to open one (opening delivers
nothing but a Vercel-style preview; the merge question is the ONE gate). `--target` must equal
the Delivery target for a normal branch, or the Production target for a `release/*` branch
(from `jaxflow release`) — anything else refuses `target-mismatch` before any mutation.
Re-running it is safe: the same sha reuses the existing PR (prints its URL, no mutation); a
descendant sha fast-forward-pushes and refreshes the record (a NEW approval is then required
for that sha); a diverged remote head refuses `pr-remote-diverged`; a closed-and-unmerged PR is
reported, never reopened (`pr-closed`).

Refusals: `sha-mismatch`, `branch-invalid`, `target-invalid`, `target-mismatch`,
`production-target-unconfigured` (the project's `AGENTS.md` has no `Production target:` line; always the case for `single-branch-pr`),
`preset-not-pr` (the repo's Preset is not `single-branch-pr` or `dual-branch-pr`; refused for ANY head, `release/*`
included, before any push or `gh` call),
`pr-remote-diverged`, `pr-closed`, `pr-ambiguous` (an un-recorded PR and more than one
`gh pr list` match), `push-failed`, `github-unreachable`, `hub-unreachable` (the PR exists on
GitHub, re-run the same command to record it), `hub-rejected: <status> <hub error>` (the hub
answered and refused; re-running will not fix it, read the error). A `<field>-too-long (n > cap)`
refusal happens before any side effect; for `verify`/`build`, move long commands into a
package.json script.

## Cut a release (dual-branch-pr only)

```
jaxflow release [--from claude|codex]
```

Snapshots the current Delivery target, names `release/<date>-staging-promotion` (a numeric
suffix if that name is already taken today), and opens its PR against the Production target
through the same routine as `pr open`. An existing open release PR is reused and reported —
never silently advanced to newer staging work. Prints the PR URL, the snapshot sha, and the
exact `jaxflow merge` command to run once the owner gives a SEPARATE production-promotion approval
(`merge-contract.md`'s own gate, distinct from the per-feature merge question).
Refuses `preset-no-release` on every preset except `dual-branch-pr`, before any ledger read, fetch or ref write.

## Merge an approved branch

ONLY after the owner agrees to your question, phrased exactly "Posso mergear `<branch>` em `<destination>`?" or "May I merge `<branch>` into `<destination>`?" (backticks required, no SHA in the text, or the dashboard's Approve merge button stays off) —
any clear yes is enough. Resolve the branch head SHA when you ask and pass it as `--sha`; if
the branch moves afterwards, ask again. The detector tolerates the verb in either language, a backticked branch and target, an optional (`sha`) and any trailing words before the `?`; the legacy gate fallback applies only to rows stored before this change. Run it from the control repo:

    jaxflow merge <branch> --sha <full-40-sha> --phase "<phase title>" \
        --checks "<cmd && cmd>" --target <approved-destination>

Here `--phase` is a free-text commit subject (spaces welcome), NOT the token the other verbs
take. `--checks` must not be blank — an empty string would exit 0 without checking anything,
and is refused at the CLI boundary before any git state is touched. `--target` is required:
it is the approved destination, asserted against the Deploy policy target — it cannot
override policy and is never defaulted from current policy. A legacy retry missing
`--target` must add its originally approved destination; do not guess it from policy,
audit rows, or ref membership. Add `--recheck` to force the `--checks` run even when the reuse rule would skip it; it changes no other gate and does nothing when the delivery is already merged.

    jaxflow merge <branch> --sha <full-40-sha> --phase "<phase title>" \
        --checks "<cmd && cmd>" --target <approved-destination> --from claude

`--checks` runs inside the branch's registered worktree at `~/repos/<project>-<branch-slug>`
(the slug is the branch name with `/` turned into `-`), never in the control repo. A build
leaves that worktree behind; a short-path branch has no build, so create it yourself before
asking for the merge — `git worktree add ~/repos/<project>-<branch-slug> <branch>`, then the
project setup from its `AGENTS.md` (e.g. `pnpm install --frozen-lockfile`). Without it the
merge refuses `checks-failed` with `hint: no registered build worktree`. The short path skips
spec and plan, not the worktree. `jaxflow merge` removes the worktree afterwards as usual.

It runs in the foreground and prints a pass/fail — there is no run id, no tmux session, and
no `status`/`result` for a merge. It resolves the delivery target from the project's
`AGENTS.md` Deploy policy, refuses `target-mismatch` if `--target` differs, runs the
merge-contract sequence on the merged tree, posts the `merge-approved` audit event, pushes
when an `origin` remote exists, copies the build worktree's reports into the control repo,
removes that worktree and its branch, and writes `status.md` with `stage: ship`.

The last two lines of its output are the outcome to relay to the owner:

    merged <merge-sha> pushed origin/<target>
    checks: <command> exit 0

or, with no remote configured, `merged <merge-sha> — no remote delivery configured`.

`--checks` is still required, but it only RUNS on code nobody has tested yet. When the SHA to be
tested is the `verify.head_sha` of the branch's latest diff review, `--checks` equals the verify
command that build recorded, every frame of the build worktree's `tests.txt` is `EXIT: 0` and the
worktree is clean, the run is skipped and the second output line reads:

    checks: reused from review <review-run-id> at <sha:12>

On a local preset this only applies when the target has not moved since the branch was cut (a
fast-forward-shaped merge). A branch with a commit after its last diff review, a moved target, an
EQUAL tip (`--sha` already IS the target's tip: it fails at `git commit` with nothing staged,
`merge-failed`, as before) or a `release/*` head runs the full checks. A lookup problem (no
review, unreadable manifest, missing build worktree) never refuses: the checks simply run. The
`merge-approved` audit carries `checks: reused|run|resumed`; a retry of an already-merged delivery
(`resumed`) prints no `checks:` line.

Refusals: `sha-mismatch` (HEAD is not the approved SHA, or `--sha` is not full 40-hex),
`phase-invalid` (`--phase` is empty, untrimmed, over 200 UTF-16 code units, or carries a
tab, newline, BOM or unpaired surrogate), `branch-invalid` (`--branch` is empty, over 512
characters, or contains whitespace), `target-invalid` (`--target` missing on a direct call,
empty, whitespace, leading dash, over 512 UTF-16 code units, unencodable, or rejected by
`git check-ref-format refs/heads/<name>`), `dirty-tracked-tree` (untracked files are fine),
`target-mismatch` (git switch failed, or approved `--target` differs from the policy
target), `merge-failed`,
`checks-failed`, `checks-dirtied-tree` (the checks modified a tracked file),
`push-failed`, `preset-unknown` (two preset blocks, one that does not name a delivery
target, or a name outside the five valid presets — the hint lists them), `hub-unreachable` (the audit POST failed after the commit), `hub-rejected: <status> <hub error>`
(the hub answered and refused; re-running will not fix it, read the error). A `<field>-too-long (n > cap)`
refusal happens before any side effect; for `verify`/`build`, move long commands into a
package.json script.

A PR preset (`single-branch-pr`, `dual-branch-pr`) takes a DIFFERENT path from here: `merge` looks up the PR `pr open` recorded
for `<branch>`/`--sha` (refuses `pr-not-found` if none exists and none is found on GitHub
either), re-verifies its identity against the approval, runs the full checks on the pinned SHA, unless the reuse rule above applies
(in the build worktree for a feature branch, in a temporary detached checkout for a
`release/*` branch), then merges it on GitHub — never a local `git merge` at all. Refusals
specific to this path: `pr-not-found`, `pr-identity-mismatch`, `pr-head-moved`,
`pr-closed`, `pr-ambiguous`, `github-merge-refused` (GitHub itself refused the merge — a
required check missing, pending, or failing; its own message plus the informational
`gh pr checks <n>` output are printed), `merge-queue-required`, `mergeability-unknown`.

A `hub-unreachable` or `push-failed` refusal KEEPS the merge commit. Re-run the exact same
command, including the originally approved `--target`, to resume: jaxflow detects that the
target already carries the merge and skips straight to the audit and push. If policy
changed, restore the approved policy or obtain a NEW approval naming the new target.
A `hub-rejected: <status> <hub error>` on the merge audit also KEEPS the commit. For a 5xx
(a hub-side fault) the resume advice above still applies; for a 4xx / `{ok:false}` the hub
answered and refused, so re-running will not fix it — read the error.
A `<field>-too-long (n > cap)` refusal happens before any side effect; for `verify`/`build`,
move long commands into a package.json script.

## Check on a run

```
jaxflow status <run_id>   # unknown | running | dead | finished(ok|failed|cancelled)
jaxflow result <run_id>   # prints the report path + content once finished; a cancelled
                          # run instead prints `cancelled — <summary>` (no report)
jaxflow cancel <run_id>   # kills the session and posts a cancelled terminal row
```

`result` refuses `report-path-mismatch` when the path the ledger recorded is not the exact
path this run's own dispatch would have written (a resolved-but-different string included),
and `report-missing` when that path is right but no file is there any more — the report is
gone for good, so re-running `result` will not bring it back (`git worktree remove` deletes
the worktree's gitignored `.local/` with it). `cancel` refuses `already-finished` on a run that has already terminated.

Every run also writes `<control repo>/.local/runs/<run_id>/child.log`: the child
runtime's own interleaved stdout+stderr (OpenCode, Codex) or stderr alone (Claude —
its stdout is the report itself), redacted once on close, capped at 40 MiB head + 10
MiB tail with a `[jaxflow: N bytes omitted]` marker in between for anything longer.
When a `run-finished` row's `contract_status` is `missing` or `invalid`, `status`
folds a classified `reason` — `provider-limit`, `crash`, `no-report`, or `unknown` —
into its one-line summary, and `result` prints that same `reason` plus the
`exit_code`, a `tail` excerpt, and the `child.log` path for EVERY such row, read
straight from the ledger, never by re-opening `child.log` — then behaves as always: a
`missing` row still ends in a `report-missing` refusal (its report.md never existed),
an `invalid` row's report.md contents follow the block. A run killed by `cancel`, or
one whose worker crashed before closing the file, leaves `child.log` partial and
**UNREDACTED** — treat it as sensitive before sharing or pasting its contents anywhere.

For any `diff` review (spec/plan reviews and builds never print this), both `status` and
`result` print a `chain:` block before anything else: one line per node from the full-range
root to this run, oldest first — `<run_id> <base_sha[:12]>..<head_sha[:12]> <verdict|running>
[since <predecessor>]` (`[since ...]` only on non-root lines). A `--diff` run with no
`--since` still prints a chain — just a single line, itself only. A broken or unreadable link
prints `chain: broken at <run_id> (<reason>)` instead of the block; a cycle prints
`chain: cycle at <run_id>`.

A finished run also delivers `[JAXFLOW] <kind> <run_id> finished — ... ` to this session
(`codex queue` for Codex, the native `asyncRewake` hook for Claude), so `status`/`result`
are for checking BEFORE that line arrives, or when delivery is skipped.
A non-zero exit means refused —
the reason is usually a single kebab-case code on stderr (e.g. `path-outside-allowlist`), but
a few refusals inherited from the underlying engine are printed as plain English instead
(e.g. `reviewer handoff required`, `run path collision`).

## Never

- Never dispatch a review with `--spec`/`--plan` pointing outside `~/repos`.
- Never create a build's worktree yourself — `jaxflow build` reserves its own.
- Never poll `status` in a tight loop — dispatch, then wait for the callback line.
- Never invoke `scripts/jaxflow.py --run-worker` directly; it is `jaxflow`'s own internal
  entry point for the background tmux session, not a user-facing verb.
