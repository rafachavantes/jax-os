---
name: jax-init
description: Use when the user says "jax-init", "init project", "novo projeto", "bootstrap project", or wants to open a new project or adopt an existing repo under ~/repos. Runs the bootstrap questionnaire, then the jax-init wrapper, then fills AGENTS.md.
---

# jax-init

Bootstrap a new project, or adopt an existing repo, with minimal manual work.
Tech-lead-only: a one-shot builder or reviewer harness never runs this.
Adoption is non-destructive: it preserves tracked contents, the index, HEAD, refs,
existing remotes and hook routing.

## Step 1 -- read first

Ask the target path (default `~/repos/<slug>`, `<slug>` derived from the name once
known). If `<path>/.local/docs/` exists and has files, read them first -- when they
total more than ~400 lines, delegate a subagent to summarize into candidate answers
(name, preset, stack, verification, hard rules, traps) instead of reading them all
inline. Pre-fill whatever the docs already answer.

Also read, without writing:

- An existing `<path>/AGENTS.md` and any repository contribution instructions.
- The project's existing branches and remotes (`git branch -a`, `git remote -v`), so an
  adopted repo's base/delivery/production mappings can be confirmed rather than invented.
- Its hook configuration WITHOUT executing any hook: `git config --get core.hooksPath`,
  `git rev-parse --git-path hooks`, the files there, and a `.githooks/` directory if one
  exists. Note which hooks are non-sample.

GitHub method capabilities may be shown as optional candidates; unavailable GitHub access
never blocks local setup and never authorizes a guess about company style.

## Step 2 -- questions, one per message

Ask only what the docs didn't already answer; a pre-filled answer needs just a
one-word confirmation. Found conventions are CANDIDATES requiring confirmation, never
silently adopted. Fixed order:

1. **Name** -- from context or asked directly.
2. **What it is** -- one paragraph: what the product does, for whom, current stage.
   Fills the template's `## What this is`. Usually answered by the docs; never
   synthesized by the skill.
3. **Preset** -- one line each, with its market name:
   - `single-branch` = trunk-based development: only `main`, local merge + push.
   - `single-branch-pr` = GitHub Flow: only `main`, feature PR with CI required, merged on GitHub.
   - `dual-branch` = environment branches: `staging` + `main`, local merge into `staging` + push.
   - `dual-branch-pr` = GitLab Flow with PRs: `staging` + `main`, feature PR into `staging`,
     `jaxflow release` promotes to `main` under a separate approval.
   - `bubble-buildprint` = Bubble via Buildprint, no code deploy.
4. **Stack** -- `nextjs` / `python` / `bubble` / `other`; show these defaults for
   confirmation:

   | Stack | Install | Test | Build |
   |---|---|---|---|
   | nextjs | `pnpm install` | `pnpm test` | `pnpm build` |
   | python | `pip install -r requirements.txt` | `pytest` | none |
   | bubble | none | none | none |
   | other | ask the user for all three commands | | |

5. **Commit convention** -- the owner's format and a concrete example. Present the Jax
   default: imperative subjects with `feat:`/`fix:`/`docs:` prefixes.
6. **PR title convention** -- PR presets only. Ask for the feature-PR format and example,
   or `No additional rule`. Present the Jax default. State that production promotions
   keep the fixed `Release: <target> promotion <date>` format.
7. **Implementation branch convention** -- the owner's format and example. Present the
   default `feat/<phase>`; `release/` stays reserved for the existing promotion procedure.
8. **PR merge method** -- PR presets only: `merge` (default), `squash`, `rebase`, or
   `auto`. `auto` prefers merge, then squash, then rebase. Do not ask for a local-only
   repo; a local preset must not configure one.
9. **Merge subject** -- ask only the choices that apply: `default` (Jax default
   everywhere), `pr-title` (PR presets; not with rebase), or `phase` (not with
   `auto`). Explain the outcomes:
   - PR: `default` = generated subject for merge, the current PR title for squash, no
     override for rebase. `pr-title` = the current title for merge/squash; refuses if
     `auto` resolves to rebase. `phase` = the exact validated `--phase` text.
   - Local: `default` = `feat: <phase> (merge <branch>)`; `phase` = the validated phase
     text plus ` (merge <branch>)` appended by jaxflow.
10. **Threat model** -- ask which mode applies; do not default it silently:

    | Mode | Meaning |
    |---|---|
    | `internal-single-user` | Hostile local writer out of scope; traversal, symlink escape and secrets stay in scope. Suggest this default for personal tools. |
    | `public-app` | Untrusted users reach this app; full OWASP scope. |

    Passed to the script as `--threat-model <mode>`.
11. **Who verifies** -- one of `rafa-verifies` / `agent-verify: <check>` /
    `agent-deploy-verify: target=<...>; check=<...>`. `rafa-verifies` is this workflow's
    fixed name for "the owner verifies by hand," independent of who the owner is.
12. **Project-specific hard rules and "Do NOT" traps** -- open text; `"none"` is valid.
13. **Remote** -- yes/no; default yes for every preset (a PR preset cannot work
    without a GitHub remote; the others still get one as the shared origin). Ask whether
    to CREATE one only for a fresh repo without an origin. An adopted repo with history
    preserves its existing origin and never creates or pushes one, even if `--remote` is
    passed. An adopted repo without an origin: a PR preset stops and asks the owner to
    connect the intended company remote, then rerun; a local preset may proceed without
    one after an explicit choice.

### Adoption branch mappings

For a repository with history, discover and confirm the ACTUAL base/delivery/production
branches for the selected preset, and map them into AGENTS.md's existing policy fields.
Never create, rename, or enforce protection on company branches. A missing required
mapping stops before writes with directions -- do not invent a branch.

### Bootstrap subject

For a repository without commits, show the bootstrap commit subject. Keep the default
`chore: bootstrap` for normal Jax projects; when the chosen company convention needs a
different subject, ask for the COMPLETE subject and never invent an issue identifier.
Pass it through `--initial-commit-subject`; omission keeps the default.

### Before nondefault conventions -- amendment preflight

Defaults require no amendment migration. Before writing NONDEFAULT conventions, verify
the exact amendment text is present in the current tech lead's effective global
instructions and every ENABLED Jax-managed instruction destination:

```text
Project conventions may override only commit subjects, PR titles, implementation branch naming, PR merge method, and generated delivery commit subject; omitted conventions retain Jax defaults and all other guarantees remain binding.
```

- **Host identity first.** The host is the harness executing this skill (Claude or Codex),
  not its provider or model. Only if that is unavailable, use a single unambiguous
  `CLAUDECODE` versus `CODEX_THREAD_ID` marker; both or neither stops with the identity
  conflict. Do not guess and do not add a flag.
- Read `integrations.agents` from existing settings and check `enabled` managed
  destinations plus the actual current harness:
  `$CLAUDE_CONFIG_DIR/CLAUDE.md` else `~/.claude/CLAUDE.md`;
  `$CODEX_HOME/AGENTS.md` else `~/.codex/AGENTS.md`;
  `~/.config/opencode/AGENTS.md`. Never inspect a disabled runtime merely because its
  file exists (the current tech lead is always checked).
- Normalize line endings and paragraph wrapping only when matching; wording must match,
  not merely its meaning.
- An extra nondefault `OPENCODE_CONFIG` (the existing `_open_code_config_is_extra`
  semantics), any unknown instruction-source override, a missing/unreadable required
  file, or a missing exact amendment stops NONDEFAULT setup BEFORE writes, names the
  conflict, and points to `/settings` -> Jax Rules (apply the amendment or resolve the
  source) before rerunning. This skill never edits those files or the installer's DB.

## Step 3 -- summary + hard gate

Show a summary table of every answer plus the exact baseline that will be created, and
whether `<path>/AGENTS.md` already exists (check now, before anything runs).
**Nothing is written before the owner says ok.** The summary lists:

- The exact proposed AGENTS.md patch/contents, preserving company instructions.
- Exact artifacts and local metadata: baseline directories, `CLAUDE.md`,
  `.jax-os/status.md`, and the exact local ignore entries. Adoption uses the shared
  `info/exclude` and never edits `.gitignore`; fresh setup keeps the normal `.gitignore`
  behavior.
- Hook disposition: installed vs preserved, the exact path/configuration change, and
  whether any approved Git operation will execute hooks. For a fresh target without Git
  metadata, describe installation as conditional on the post-init revalidation and the
  absence of template-supplied hooks; inherited routing, when already known, is reported
  as preserved. State that preserved company hooks do NOT gain Jax `Co-Authored-By`
  trailer stripping; agents stay bound by the no-attribution rule.
- Bootstrap subject (no-commit repos only), exact refs created/pushed, and for a fresh
  `--remote` the resolved GitHub account/repo destination.
- For adoption: the preserved origin and the fact that no commit, branch, `.gitignore`,
  hosted repo, or push will be created.

If AGENTS.md is TRACKED and the proposed answers require changing it, show the proposed
patch and STOP before ANY bootstrap mutation: the owner integrates it through the
company's own documentation workflow, then reruns Init. If its committed contents
already contain the chosen policy, continue without editing it.

## Step 4 -- execute

Run:

```
jax-init --path <path> --name "<name>" \
  --preset <preset> --stack <stack> --threat-model <mode> \
  [--remote] [--hooks auto|preserve] \
  [--initial-commit-subject "<complete subject>"] [--dry-run]
```

`--hooks auto` (default) installs the Jax hook only when no existing hook setup would be
displaced; otherwise it preserves and reports. `--hooks preserve` installs nothing and
changes no routing. Fresh and adopted repositories both install into Git's resolved
default common-metadata hooks directory and never set `core.hooksPath` or create
`.githooks/`. A fresh target without metadata makes `git init` the first write; inherited
routing and the native destination are revalidated before the hook write, and any hook
supplied by a Git template is preserved.

Relay its JSON summary and stderr lines verbatim. A refusal (exit 2) stops the flow
and shows the one-line reason -- no partial `AGENTS.md` write, no invented retry loop.

On success, fill `workflow/templates/AGENTS-template.md` with the answers and write
`<path>/AGENTS.md`: resolve every `<PLACEHOLDER>`. In a PR preset block, replace
`<owner>/<repo>` with the repo slug when the remote is known and `<ci-check>` with
the CI job id; leave them as written when unknown. Keep only the chosen preset and
verification policy (delete the other four presets and two policies), set the
`## Threat model` block's `mode:` line to the chosen `--threat-model` value, remove
every HTML comment. `## Layout`: fresh repo -> "Not yet -- fill after the first
implementation phase."; adopted repo -> list the top-level directories the skill can
see, one line each, no invented purpose.

If `<path>/AGENTS.md` already exists and is UNTRACKED, do NOT replace it wholesale:
apply ONLY the explicitly approved patch, preserving its company instructions and
unrelated contents. A TRACKED AGENTS.md follows the proposal-and-stop gate above.
Never agent-commit or untrack it.

If the existing document, tracked state, or hook setup changed since the summary, stop
and present the revised proposal before applying affected changes.

## Step 5 -- close

Final message: what was created, the `AGENTS.md` path, the hook disposition, and the
next action ("open the tech-lead session in the repo; the first spec is next"). Never
runs a build, never pushes unless `--remote` was chosen for a FRESH repo, never touches
a global rule file.

Show branch-protection setup ONLY for fresh projects. When `single-branch-pr` or
`dual-branch-pr` was chosen fresh, show the owner the branch-protection command from the
AGENTS block and say it is a ONE-TIME repo setting that jaxflow neither sets nor checks.
Targets: `main` for `single-branch-pr`; `staging` and then `main` for `dual-branch-pr`.
`<ci-check>` is the CI job id (the check context). Never run it from this skill, and
never suggest replacing an adopted repo's existing governance.

    gh api -X PUT repos/<owner>/<repo>/branches/<target>/protection --input - <<'JSON'
    {"required_status_checks":{"strict":false,"contexts":["<ci-check>"]},
     "enforce_admins":true,"required_pull_request_reviews":null,"restrictions":null}
    JSON

## Never

- Never overwrite an existing `<path>/AGENTS.md`; apply only the approved patch to an
  untracked file, and stop on a tracked one that needs changes.
- Never create or modify `.gitignore`, a commit, a branch, a hosted repo, or a push when
  adopting a repo with history.
- Never change, replace, chain, copy, or directly execute a company hook.
- Never run an install or build command.
- Never edit `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, or
  `~/.config/opencode/AGENTS.md` -- that is Jax Rules' job only.
- Never push without an explicit "yes" to the Remote question.
- Never invent a retry loop after a refusal -- show the reason and stop.
