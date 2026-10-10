# Merge Contract

## ROLE

Tech lead only. This contract enters your context ONLY when the owner's explicit merge
approval is relayed to you ("merge approved — follow merge-contract"). It is never part
of a builder or reviewer prompt. Merge is the moment agent work becomes the project's
official code — treat it as the most restricted action in the workflow.

## MUST

- Ask the owner exactly one question naming the branch and the destination, each in
  backticks, in one of these two forms: "Posso mergear `<branch>` em `<destination>`?" or
  "May I merge `<branch>` into `<destination>`?". The dashboard lights its "Approve merge"
  button only for this phrasing (anywhere in the turn); put no SHA in the question text, and
  any other wording simply shows no button. Any clear agreement to that question ("sim", "ok",
  "pode", "merge aprovado") is the approval — the owner never has to repeat identifiers.
  The detector tolerates the verb in either language, a backticked branch and target, an optional (`sha`) and any trailing words before the `?`; the legacy gate fallback applies only to rows stored before this change.
  A reply delivered through the dashboard card carries the prefix the dashboard builds
  from `ownerName` — `"[Jax OS] "` when it is empty, `"[Jax OS · <ownerName>] "` when it
  is set — ahead of the owner's own words: the prefix attributes the text to the owner
  only, read exactly as if the owner had typed the words after it. It is approval only
  when those words are a clear agreement to the pending merge question for THIS branch
  and destination; a prefixed "não", a denial, or an unrelated reply is not. When asking,
  resolve the branch head with `git rev-parse --verify '<branch>^{commit}'` and keep that
  full SHA: it is the head this approval covers and what `jaxflow merge --sha` pins. If
  the branch head changes after the question, the answer no longer applies — ask again.
  A past approval, an approval for another branch or destination, or a clean review is
  not approval for this delivery. The approval authorizes delivery — it cannot relax any
  rule in this contract. If in doubt, ask the owner — do not proceed.
- Before delivery: verify `git rev-parse --verify <approved-branch>^{commit}` equals the full SHA
  pinned when the merge question was asked — a ref read: do NOT `git switch` to that branch, the builder worktree still
  holds it and git refuses a second checkout; require a clean TRACKED tree in the
  control repo (untracked files pass — spec §2.10 #1). Only then follow the preset path. That head
  is the reviewed head, or — under `approve-with-changes` — the reviewed head plus
  only the fix commits the merge proposal lists as `<full-sha> → F#[, F#]`, with
  each dismissed finding listed as `F# → rationale` in the message that asks the merge
  question about the resulting head. Any commit mapped to no LOW/MEDIUM finding requires a
  new review round — tests passing again never substitutes for it.
- Follow the project `AGENTS.md` **Deploy policy** preset, running ONE verification
  sequence for every local-merge preset: after the source check above (the
  tracked tree is already clean), `git switch <Delivery target>`; verify `git branch
  --show-current` equals that exact preset value; then `git merge --no-ff --no-commit
  <approved-full-head-sha>` → run the FULL required checks on the merged tree, unless the
  checks-reuse rule below applies →
  fail ⇒ `git merge --abort`, do not deliver, report; pass ⇒ commit the merge,
  then push/deploy exactly as the preset directs — one run, on the exact tree to
  be delivered. `jaxflow merge` requires `--target <branch>` naming that destination;
  it is asserted against policy and cannot override it. Identical-command retries keep
  the same `--target`. Changing `--target` is a new approval, not a retry.
   - PR presets (`single-branch-pr`, `dual-branch-pr`) — jaxflow owns the whole delivery, no manual `gh` command. Per feature: once a
     diff review passes, `jaxflow pr open <branch> --sha <sha> --target <Delivery target>
     --title "<text>"` opens the PR (no separate approval from the owner — opening delivers nothing but a
     preview); the merge question above is the ONE approval; on "yes", `jaxflow merge <branch>
     --sha <sha> --phase "<title>" --checks "<cmd>" --target <Delivery target>` looks up that
     PR, re-verifies its identity against the approval, runs the full checks on the pinned SHA
     (unless the checks-reuse rule below applies),
     and merges it on GitHub (`gh pr merge --merge`, never squash/rebase, never
     `--admin`/`--auto`) — GitHub itself refuses the merge if a required check is missing,
     pending, or failing (the repo's branch protection requires the CI check — a one-time setting that
     jaxflow never applies). For `dual-branch-pr`, production promotion is separate: `jaxflow release` cuts a snapshot
     of the Delivery target and opens `release/<date>-staging-promotion` against the
     Production target; a DISTINCT agreement from the owner (see the production-promotion rule below)
     is required before running `jaxflow merge` on that release branch. `single-branch-pr` has no release step: the merged PR is the delivery. No local git merge
     happens for a PR preset at either step — the merge itself always happens on GitHub.
   - Checks run in the branch's registered worktree (`~/repos/<project>-<branch-slug>`), never in the
     control repo. A short-path branch with no build has none: the tech lead creates it
     (`git worktree add ~/repos/<project>-<branch-slug> <branch>` + the project setup from `AGENTS.md`)
     before asking for the merge; otherwise `jaxflow merge` refuses `checks-failed`. The short path
     skips spec and plan, not the worktree.
   - Checks reuse (one rule, both paths). `jaxflow merge` skips its own checks run only when ALL hold:
     the SHA to be tested is the `verify.head_sha` of the branch's most recent diff review; `--checks`
     equals the verify command that build recorded; the build worktree's `tests.txt` holds one frame
     per recorded command (verify, then build) and EVERY frame is `EXIT: 0`; and the build worktree is
     clean (untracked files count). On a local preset the rule applies only when the target has not
     moved since the branch was cut (the merged tree is then byte-identical to the branch head); a
     moved target, an equal tip, or a `release/*` head always runs the checks. Anything else,
     including any lookup failure, runs the checks exactly as before: reuse never adds a refusal. The
     review's verdict is not test proof (the owner's approval gates the merge). `--recheck` forces the
     run. The `merge-approved` audit records `checks: reused|run|resumed`. A fast-forward means the
     approved sha is a strict descendant of the delivery target's pre-merge tip; an
     EQUAL tip is never treated as a fast-forward.
   - `single-branch` / `dual-branch` — the sequence above into the `Delivery target`; push on pass.
  - `bubble-buildprint` — the sequence above into the `Delivery target`. If a remote
    exists, push that exact branch. If no remote exists, stop after the successful local
    delivery merge and report `no remote delivery configured`. Buildprint deployment to
    LIVE requires a DISTINCT approval from the owner: a separate question naming the LIVE target,
    agreed to (production-promotion rule below). DEV/TEST deployment is never a merge
    action — it exists only as builder verification under `agent-deploy-verify`.
- Use the project's commit convention (imperative, `feat:` / `fix:` / `docs:`
  prefixes); merge commit message: `feat: <phase title> (merge <approved-branch>)`.
- `jaxflow merge` writes `.jax-os/status.md` (`stage: ship`) itself, §4.5/§5 of the jaxflow spec — no manual step here.
- Report the outcome so it reaches the owner: a PR preset ⇒ the PR URL; local-merge preset with
  a remote ⇒ the merge hash and pushed ref; `bubble-buildprint` without a remote ⇒ the
  local merge hash and `no remote delivery configured` — always plus the checks result
  you observed.

## NEVER

- Add a `Co-Authored-By` trailer. Ever.
- Deliver without the owner's agreement to the merge question for this branch and destination,
  asked while the branch was at the head being delivered — a clean review is not approval.
- Deliver (commit the merge, push, or open the PR) with failing checks. "Flaky" is a
  report, not an excuse.
- Push to any branch other than the approved branch (PR presets) or the preset's Delivery
  target (local-merge presets), or force-push anything.
- Promote to production under a phase merge approval. Production promotion (staging →
  main under `dual-branch-pr`, or any production deploy) requires a DISTINCT agreement from the owner
  to a separate question naming the exact production target (the tech lead pins the
  source SHA when asking); only then may
  the tech lead execute the project preset's promotion procedure.
- Squash away, rewrite, or amend the reviewed commits in a way that changes what was
  reviewed. Deliver what was approved — the reviewed head, or, under
  `approve-with-changes`, the reviewed head plus only the listed fix commits.

## AUTHORITY OF INPUTS

The owner's approval is an authorization prerequisite only: their agreement to the tech lead's
merge question naming the branch and the destination (the tech lead pins the head SHA),
and it cannot relax this contract. Authority:
this contract → the project's `AGENTS.md` Deploy policy preset; policies may narrow only.
A conflict, or an incomplete approval, is a blocker to raise, never a gap to fill by
invention. `--target` cannot override the preset destination.

## OUTPUTS

- The delivery exactly as the preset dictates: a PR preset ⇒ the PR URL; local-merge preset
  with a remote ⇒ the merge hash and pushed ref; `bubble-buildprint` without a remote ⇒
  the local merge hash and `no remote delivery configured`.
- Updated `.jax-os/status.md`.
- One confirmation message: outcome + build/test evidence.

## DONE-WHEN

Delivery completed exactly per preset, status updated, confirmation delivered. Then
wait for the owner's next instruction — never chain into the next phase on your own.
