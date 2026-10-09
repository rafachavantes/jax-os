# AGENTS.md — <PROJECT NAME>

<!-- TEMPLATE. Copy to <repo>/AGENTS.md, fill every <PLACEHOLDER>, delete the presets you
     did not choose and every HTML comment. Keep it lean: rules, not prose. English only.
     CLAUDE.md contains exactly one line: `@AGENTS.md`. Both files, `.local/`, and
     `.jax-os/` are gitignored. -->

Single source of project context for AI agents. Universal rules only — role-specific
rules (builder / reviewer / merge) live in the workflow contracts. Builder/reviewer
contracts are injected per run; the merge contract is injected only after the owner's
explicit merge approval.

## What this is

<ONE PARAGRAPH: what the product does, who it is for, current stage.>

## Stack

- <Framework + version, language, package manager>
- <Backend / DB / auth>
- <Hosting / deploy target>
- <Key libs worth naming>

## Commands

- Install: `<cmd>`
- Dev: `<cmd>`
- Test: `<cmd>`
- Build: `<cmd>`
- Lint/typecheck: `<cmd>`

## Layout

- `src/` — <what lives where, one line per top-level dir that matters>
- `.local/` — ALL non-code project memory, gitignored: `docs/specs/`, `docs/plans/`,
  `reports/` (workflow run reports), `scratch/` (per-run temp, auto-deleted), `handoff.md`
  (session summary — last session only, always overwritten; history lives in Honcho).
- `.jax-os/status.md` — Jax OS dashboard card (gitignored). Schema: see
  `workflow/contracts/status-report-contract.md` in jax-os.

## Deploy policy  <!-- CHOOSE ONE PRESET, DELETE THE OTHERS -->

**Preset: `single-branch`** — trunk-based development: only `main`, local merge and push.
- Base branch: `main` (default). `main` is the only PERSISTENT branch.
- Implementation runs use temporary `feat/<phase>` branches from the base.
- Delivery target: `main`, merged and pushed per `merge-contract.md` after the owner's merge
  approval.
- Create `staging` when the project first ships to real users; then switch preset to
  `dual-branch` or `dual-branch-pr`.
- Verification deployment target: none.

**Preset: `single-branch-pr`** — GitHub Flow: only `main`, feature PR with CI required.
- Base branch: `main` (default). `main` is the only PERSISTENT branch.
- Delivery target: `main` (feature-branch PR against `main`); executor and gate per
  `merge-contract.md`.
- Implementation runs use temporary `feat/<phase>` branches from the base.
- Never commit directly to `main`.
- Delivery per feature: `jaxflow pr open` right after a diff review passes, then the owner's merge
  approval, then `jaxflow merge` — merges on GitHub, no local merge, no manual `gh` command. No
  release step: the merged PR is the delivery.
- "CI required" is a ONE-TIME per-repo GitHub setting (branch protection); jaxflow neither sets nor
  checks it. Run once with `<target>` = `main`; `<ci-check>` is the CI job id (the check context)
  and must be renamed together with the job:
  ```
  gh api -X PUT repos/<owner>/<repo>/branches/<target>/protection --input - <<'JSON'
  {"required_status_checks":{"strict":false,"contexts":["<ci-check>"]},
   "enforce_admins":true,"required_pull_request_reviews":null,"restrictions":null}
  JSON
  ```
- Verification deployment target: none.

**Preset: `dual-branch`** — environment branches: `staging` + `main`, local merge into `staging`.
- Base branch: `staging` (default). Delivery target: `staging`, merged and pushed per
  `merge-contract.md` after the owner's merge approval.
- Implementation runs use temporary `feat/<phase>` branches from the base.
- `staging → main` promotion only under a distinct, explicit promotion approval from the owner,
  executed by the tech lead — never implied by a phase merge approval.
- Verification deployment target: none.

**Preset: `dual-branch-pr`** — GitLab Flow with PRs: `staging` + `main`, feature PR into `staging`.
- Base branch: `staging` (default). Delivery target: feature-branch PR against
  `staging`; executor and gate per `merge-contract.md`.
- Production target: `main`.
- Implementation runs use temporary `feat/<phase>` branches from the base.
- Never commit directly to `staging` or `main`.
- Delivery per feature: `jaxflow pr open` right after a diff review passes, then the owner's merge
  approval, then `jaxflow merge` — merges on GitHub, no local merge, no manual `gh` command.
- `staging → main` promotion: `jaxflow release` cuts and opens the promotion PR; a SEPARATE,
  explicit approval from the owner gates its own `jaxflow merge`. Any other production deploy:
  the owner only, separate gate.
- "CI required" is a ONE-TIME per-repo GitHub setting (branch protection); jaxflow neither sets nor
  checks it. Run it twice, once with `<target>` = `staging` and once with `<target>` = `main`;
  `<ci-check>` is the CI job id (the check context) and must be renamed together with the job:
  ```
  gh api -X PUT repos/<owner>/<repo>/branches/<target>/protection --input - <<'JSON'
  {"required_status_checks":{"strict":false,"contexts":["<ci-check>"]},
   "enforce_admins":true,"required_pull_request_reviews":null,"restrictions":null}
  JSON
  ```
- Verification deployment target: none.

**Preset: `bubble-buildprint`** — Bubble via Buildprint: Bubble.io app built through Buildprint.
- Base branch: `main` (default). Delivery target: `main`. Local git REQUIRED (the
  workflow wrapper refuses non-git repos); private GitHub remote optional (backup). To
  move the base later, create the new branch from `main` first (policy migration, not
  bootstrap).
- Implementation runs use temporary `feat/<phase>` branches from the base.
- Source of truth: local Bubble JSON exports managed by Buildprint.
- Verification deployment target: `<Bubble DEV/TEST environment>` — a builder may deploy
  there ONLY under a handoff `verification: agent-deploy-verify: target=…; check=…` line
  naming this exact target; never as delivery.
- Deploy-to-LIVE is delivery: NEVER by a builder. Tech lead only, after the owner's gate, per
  `merge-contract.md`.

## Threat model  <!-- CHOOSE ONE MODE -->
mode: internal-single-user
<!-- internal-single-user: a single user on a VPS behind Tailscale. Only the owner or the
     agents he dispatched write files and the DB. A hostile local process is OUT of scope.
     Path traversal, symlink escape outside the allowlist and secret leaks stay IN scope.
     public-app: anything reached by users you do not control (app with sign-ups, public
     API). Full scope: the whole OWASP surface. -->

## Verification policy  <!-- CHOOSE ONE DEFAULT; the phase handoff may override -->

- `rafa-verifies` — after completing the handoff's named test/build commands, the
  builder runs no additional verification command; the owner checks by hand.
  — `rafa-verifies` is this workflow's fixed name for "the owner verifies by hand,"
  independent of who the owner is.
- `agent-verify: <Playwright | agent browser | curl smoke | …>` — the builder runs the
  named check and pastes evidence in the report.
- `agent-deploy-verify: target=<Verification deployment target>; check=<check>` — ONLY
  when the selected preset's Verification deployment target is a concrete, non-`none`
  DEV/TEST target (e.g. `bubble-buildprint`); `target=none` is invalid and never deploys;
  the builder deploys there for verification only, then runs the named check.

## Rules (hard)

1. <PROJECT-SPECIFIC HARD RULE — e.g. "never bind 0.0.0.0", "RLS on every table">
2. <…>

## Persistent tech-lead session end

Does not apply to one-shot builder or reviewer runs. REWRITE `.local/handoff.md`:
what closed, what is open, the exact next step. Last session only — overwrite, do
not append.

## Do NOT

- <top 3–5 project-specific traps, one line each>
