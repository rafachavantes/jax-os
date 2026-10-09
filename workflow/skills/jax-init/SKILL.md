---
name: jax-init
description: Use when the user says "jax-init", "init project", "novo projeto", "bootstrap project", or wants to open a new project or adopt an existing repo under ~/repos. Runs the bootstrap questionnaire, then the jax-init wrapper, then fills AGENTS.md.
---

# jax-init

Bootstrap a new project, or adopt an existing repo, with minimal manual work.
Tech-lead-only: a one-shot builder or reviewer harness never runs this.

## Step 1 -- read first

Ask the target path (default `~/repos/<slug>`, `<slug>` derived from the name once
known). If `<path>/.local/docs/` exists and has files, read them first -- when they
total more than ~400 lines, delegate a subagent to summarize into candidate answers
(name, preset, stack, verification, hard rules, traps) instead of reading them all
inline. Pre-fill whatever the docs already answer.

## Step 2 -- questions, one per message

Ask only what the docs didn't already answer; a pre-filled answer needs just a
one-word confirmation. Fixed order:

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

5. **Threat model** -- ask which mode applies; do not default it silently:

   | Mode | Meaning |
   |---|---|
   | `internal-single-user` | Hostile local writer out of scope; traversal, symlink escape and secrets stay in scope. Suggest this default for personal tools. |
   | `public-app` | Untrusted users reach this app; full OWASP scope. |

   Passed to the script as `--threat-model <mode>`.
6. **Who verifies** -- one of `rafa-verifies` / `agent-verify: <check>` /
   `agent-deploy-verify: target=<...>; check=<...>`. `rafa-verifies` is this workflow's
   fixed name for "the owner verifies by hand," independent of who the owner is.
7. **Project-specific hard rules and "Do NOT" traps** -- open text; `"none"` is valid.
8. **Remote** -- yes/no; default yes for `single-branch-pr`, `dual-branch` and
   `dual-branch-pr`, no otherwise (a PR preset needs a GitHub remote).

## Step 3 -- summary + hard gate

Show a summary table of every answer plus the baseline that will be created, and
whether `<path>/AGENTS.md` already exists (check now, before anything runs).
**Nothing is written before the owner says ok.**

## Step 4 -- execute

Run:

```
jax-init --path <path> --name "<name>" \
  --preset <preset> --stack <stack> --threat-model <mode> [--remote]
```

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
see, one line each, no invented purpose. If `<path>/AGENTS.md` already exists, do NOT
overwrite it -- show a diff proposal instead and stop.

## Step 5 -- close

Final message: what was created, the `AGENTS.md` path, and the next action ("open
the tech-lead session in the repo; the first spec is next"). Never runs a build,
never pushes unless `--remote` was chosen, never touches a global rule file.

When `single-branch-pr` or `dual-branch-pr` was chosen, also show the owner the branch-protection
command from the AGENTS block and say it is a ONE-TIME repo setting that jaxflow neither sets nor
checks. Targets: `main` for `single-branch-pr`; `staging` and then `main` for `dual-branch-pr`.
`<ci-check>` is the CI job id (the check context). Never run it from this skill.

    gh api -X PUT repos/<owner>/<repo>/branches/<target>/protection --input - <<'JSON'
    {"required_status_checks":{"strict":false,"contexts":["<ci-check>"]},
     "enforce_admins":true,"required_pull_request_reviews":null,"restrictions":null}
    JSON

## Never

- Never overwrite an existing `<path>/AGENTS.md`.
- Never run an install or build command.
- Never edit `~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, or
  `~/.config/opencode/AGENTS.md` -- that is Jax Rules' job only.
- Never push without an explicit "yes" to the Remote question.
- Never invent a retry loop after a refusal -- show the reason and stop.
