# Inventory helper setup and operation recovery

The Tools → Inventory workspace controls native per-executor capabilities
through one finite helper: `scripts/jaxflow_inventory_io.py`. This document is
the one-time deployment note.

## One-time setup (production host)

```sh
bash scripts/setup-inventory-helper.sh ~/.jax-os/inventory-venv
```

- The script takes ONE explicit absolute venv directory. It refuses a symlink
  target, an existing non-directory, an unowned directory, and an existing
  directory that is not a virtualenv. Re-running it is idempotent.
- It creates the venv with `python3 -m venv` and installs the pinned
  `scripts/requirements-inventory.txt` into THAT venv only. No sudo, no global
  pip, no environment-file edits, no upload/command API, no auto-install from
  an HTTP handler or a startup hook.
- The pinned dependency is `tomlkit==0.13.2` (the owner's approved exception). It
  is used ONLY for Codex plugin/skill TOML edits, imported lazily.

## Where each piece runs

- Read-only `snapshot`/`preview` use the system `python3` and the stdlib
  (`tomllib`); merely listing tools never needs the helper venv or a running
  Codex.
- Codex write actions use the fixed interpreter
  `~/.jax-os/inventory-venv/bin/python`. The collector derives that path;
  clients cannot choose an executable.
- Missing helper: every capability that needs a write is shown unavailable with
  `inventory-setup-required`. The app never crashes and read-only listing keeps
  working.

## Development and tests

- Development uses a worktree-local, git-ignored `.local/inventory-venv`:
  `bash scripts/setup-inventory-helper.sh "$PWD/.local/inventory-venv"`.
- Tests run the existing pytest with the helper's site-packages on a
  command-scoped `PYTHONPATH` so pytest is reused, not added as a dependency:
  `PYTHONPATH="$(.local/inventory-venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" python3 -m pytest ...`
- Tests inject temporary roots (`roots` key) and never touch production paths.

## Native scopes and safety

- Fixed managed scope only; every target is derived server-side. No
  client-supplied path, command or raw config.
- Writes hold a configuration claim (`~/.jax-os/inventory.lock` for
  Claude/Codex, the existing agent-settings lock for OpenCode), re-read the
  source, compare the preview revision, stage a candidate, create a
  protected owner-only backup, atomically rename and read back.
- Configuration edits preserve unrelated keys, comments and restrictive file
  modes. A form the writer cannot round-trip is refused rather than rewritten.
- OpenCode edits republish the agent-settings source hashes through the shared
  publisher, so a subsequent `jaxflow` launch is not broken by the changed
  native source.

## Recovery limitations

- `Remover configuração` removes only the exact registration; installation
  files remain.
- Delete-installation is a quarantine: the exact owned, unshared target is
  moved to `~/.jax-os/removed-tools/<operationId>/`. It is recoverable there,
  not permanently deleted, and the confirmation says so.
- A successful native publish whose settings publication fails is reported as
  `activation-pending`, not as a silent success. Recheck reads the actual
  native state; the separately confirmed Reconcile is the only repair for that
  recorded partial-publication state.
- Timeouts are unconfirmed. Use Recheck by the original client operation ID;
  never resubmit the effect blindly.
