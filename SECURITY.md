# Security

## Trust model

Jax OS is built for a single owner on a single trusted host. It ships no
authentication of its own, binds every listener to `127.0.0.1`, and is not
designed for multi-tenant or public deployment. Everything reachable from
that host — the agent CLIs, the files inside the allowlist, the secrets in
`$JAXOS_HOME/.env` — is assumed to belong to the owner and to be trusted.

## What's in scope

- **Path traversal / symlink escape** outside the Files allowlist
  (`reposRoot` and `vaultPath` from `/settings`). The server canonicalizes
  every path (resolving symlinks and `..`) before checking containment.
- **Secret leaks** — a credential reaching the client bundle, a log, or a
  report.
- **Credential handling** for the integrations that read secrets (Linear,
  OpenCode GO, TypeSafe, the notification webhook, `gh`).

## What's out of scope

- **A hostile process running as the same local user.** It can read the same
  files and secrets the app can; nothing the app does can stop it.
- **Multi-user exposure** — separate accounts, per-user data, tenant
  isolation.
- **Running this behind a public or untrusted network without an auth layer
  of the owner's own choosing.**

Loopback bind plus Fetch Metadata is **not** authentication once a proxy sits
in front of the app: whatever reaches the tunnel reaches the service behind
it. `INSTALL.md`'s ttyd section states the same fact for a concrete listener.

## Reporting

Report a suspected vulnerability privately through GitHub: [Report a vulnerability](https://github.com/rafachavantes/jax-os/security/advisories/new), with a description, reproduction steps, and the version or commit you tested.

There is no bug bounty and no disclosure SLA — this is a personal project,
and fixes happen on the owner's own schedule.
