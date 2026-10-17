# Architecture

What the shipped system does and the rules it follows. Each rule has a
canonical home elsewhere; this file states the rule and points at that home
instead of copying it.

## Collectors boundary

All system access lives in `src/server/collectors/` as pure, typed functions.
Parse functions take the raw input (command output, file text) as an injected
string, so a collector is testable without a live system, and API routes are
thin wrappers: read, call the collector, return its result. No route or
component talks to the system directly.

## Error contract

Every data source returns exactly one of two shapes, and the UI never
confuses them:

| Result | Shape | UI |
|---|---|---|
| Source unavailable | `200 {ok:false, error}` | visible warning |
| Empty data | `200 {ok:true, data:[]}` | quiet empty state |

Never conflate the two, and no page may crash because a data source failed.

## Integrations on/off

Every integration starts disabled on a fresh install. The integrations are
Linear, ttyd, Hermes tokens, GitHub, vault, classifier, and the
notification webhook; each is enabled in `/settings`, and an integration that
is off performs zero reads and zero calls. The per-integration toggle
mechanics (what each one unlocks, which secret it expects) live in the code
and in `/settings` itself, not in this file. Agents (Claude Code, Codex, OpenCode) are the exception: each has its own switch, seeded once at first start from the CLIs found on PATH; an agent that is off drops out of review routing (the other reviewer runtime is used), `jaxflow build`, `jaxflow doctor`, Jax Rules and the hub, and an agent that is on shows its usage meter in the Topbar and on the Tokens page when a credential exists.

## DS tokens

Every color and type choice comes from the vendored design-system tokens
through Tailwind utilities: `bg-surface`/`-2`/`-3`, `bg-base`, `text-ink`,
`text-body-ink`, `text-muted`, `border-line`/`-strong`, `bg-brand`(`-soft`),
`bg-accent`(`-soft`), the semantic `success`/`warning`/`danger`/`info`, and
`font-display`/`font-sans`/`font-mono`. Never a raw hex value, never a
hardcoded string.

## i18n

Every UI string lives in both `messages/pt-BR.json` and `messages/en-US.json`
— both locales, always. The locale is a cookie (no URL routing), pt-BR by
default. Components never embed a user-facing string of their own.

## Mutations audit

Reads are the default; the cockpit performs an external mutation only on an
owner trigger. Every mutation it performs — Linear writeback, file operations,
the approval webhook, tmux control, rule writes — is recorded in the
`mutations` table of `$JAXOS_HOME/jaxos.db` (default `~/.jax-os/jaxos.db`).

## Loopback bind

Every listener binds `127.0.0.1`, in dev and production alike; remote access
goes through a private tunnel (`tailscale serve`), never a `0.0.0.0` bind.
[`SECURITY.md`](../SECURITY.md) states what that protects and what it does
not.

## Config surfaces

Three files, each with one writer:

- **`$JAXOS_HOME/settings.json`** — general settings (`ownerName`, `locale`,
  `reposRoot`, `vaultPath`, `monitoredUnits`, `integrations.agents`), written by the app when the
  owner saves `/settings`. The schema, reader, and writer live in
  `src/server/settings.ts`.
- **`$JAXOS_HOME/.env`** — the owner's secrets (`LINEAR_API_KEY`,
  `OPENCODE_GO_API_KEY`, `NOTIFICATION_WEBHOOK_URL`,
  `NOTIFICATION_WEBHOOK_SECRET`, `TYPESAFE_API`), filled in by hand with mode
  `0600`. [`INSTALL.md`](../INSTALL.md)'s "Secrets & runtime settings" step
  covers where each value comes from.
- **`$JAXOS_HOME/agent-settings.json`** — native harness configuration (which
  model and profile each agent CLI runs with), published only by the Tools UI
  under `/settings` Agentes. Its schema is defined in
  [`workflow/contracts/agent-settings-contract.md`](../workflow/contracts/agent-settings-contract.md).
