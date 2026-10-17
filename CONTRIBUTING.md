# Contributing

Thanks for wanting to contribute. Jax OS is a personal, single-owner project;
this file tells you how a change gets through review without guessing.

## Local checks

CI is deliberately lean: it type-checks, runs the static repo checks (locale key
parity, no raw hex outside the design tokens, the OSS docs check) and builds. Before
opening a PR, run the full suite locally too — it includes integration tests that need
the agent CLIs and a real home directory, which a clean CI runner does not have:

```bash
pnpm exec tsc --noEmit
pnpm exec vitest run --dir src
python3 -m pytest scripts -q
pnpm build
```

A few Python tests need the inventory helper's site-packages on `PYTHONPATH`;
[workflow/contracts/tools-inventory-setup.md](workflow/contracts/tools-inventory-setup.md)
("Development and tests") has the one-line recipe.

## Commit hook

`.githooks/commit-msg` rejects `Co-Authored-By:` trailers. It only runs when
`core.hooksPath` points at `.githooks` — local git config a fresh clone never
inherits, so do not count on the hook firing for you. The real enforcement
point for a pull request is CI's own no-`Co-Authored-By` step in
`.github/workflows/ci.yml`, which checks every commit the PR adds.

## Opening a PR

An ordinary fork PR is enough. No `jaxflow` run, and no paid model call, is
ever a prerequisite for contributing: CI plus a human review is the whole
path.

## How an external PR is reviewed

`jaxflow review --diff` takes a build `run_id` — `python3 scripts/jaxflow.py
review --help` says it plainly: "RUN ID of a finished 'jaxflow build' whose
diff to review -- NOT a path and NOT a branch name". A fork PR was never built
through `jaxflow`, so there is no `run_id` for it. The maintainer reviews it as
an ordinary human code review, or — at the maintainer's own choice — runs a
manual companion-script pass per the project's bypass rule. Do not expect
`--diff` to work on a bare branch name.

## `jaxflow.py`'s shape

`scripts/jaxflow.py` is a single Python 3 stdlib CLI that drives spec/plan
cold review, build dispatch, and merge. Run `python3 scripts/jaxflow.py
--help`, or the installed `jaxflow` wrapper once `INSTALL.md`'s PATH-wrappers
step is done, for the authoritative command list — read it there instead of
trusting a copy that can drift.
