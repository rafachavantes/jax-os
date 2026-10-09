# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-10-09

First public release.

- Cockpit pages: Mission Control, Tokens, Health, Files, Kanban, tmux viewer and Settings, with a dark and a light theme and a mobile layout.
- `jaxflow` CLI: `review` (spec, plan or diff, always by the opposite runtime), `build` (own worktree, `--verify` re-run), `merge`, `status`, `result`, `cancel` and `doctor`.
- Agent hooks and skills for Claude Code and Codex; OpenCode as the builder runtime (`workflow/`).
- Every integration is off until switched on in Settings; a fresh install needs no external account.
- Loopback-only: every listener binds `127.0.0.1`; state lives in one local SQLite file; the UI polls, no websockets.
- Documentation: `README.md`, `INSTALL.md`, `docs/ARCHITECTURE.md`, `SECURITY.md`, `CONTRIBUTING.md`.
- Self-hosted Orbitron display font and the Relay/Axis brand assets.

[Unreleased]: https://github.com/rafachavantes/jax-os/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rafachavantes/jax-os/releases/tag/v0.1.0
