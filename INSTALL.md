# Installing Jax OS

Agent-executable end to end: every step is a command or a clearly marked
manual edit, and the last step (`jaxflow doctor`) tells you whether the result
is wired up. No installer script.

## Prerequisites

- Linux with systemd user services (units and timers below run as user units).
- Python ≥ 3.11 — `.python-version` pins the version the project is tested with.
- Node and pnpm per `.nvmrc`, `package.json`'s `engines`, and
  `packageManager` — use those pins, not a version from an old blog post.
- `tmux`, `git`, and `rg` on `PATH`.
- Agent CLIs: at least one of `claude`/`codex`/`opencode` for the dashboard. `jaxflow build` needs `opencode`; `jaxflow review` needs `claude` or `codex`; the full matrix in the README takes all three.

## Clone

```bash
export JAXOS_REPO="$HOME/repos/jax-os"   # default clone path; change it if you like
mkdir -p "$(dirname "$JAXOS_REPO")"
git clone https://github.com/rafachavantes/jax-os.git "$JAXOS_REPO"
cd "$JAXOS_REPO"
pnpm install --frozen-lockfile
```

`pnpm install` runs the `postinstall` script
(`scripts/copy-monaco.mjs`), which vendors the editor bundle and its license
files into `public/`.

Every later step that needs the checkout refers to `$JAXOS_REPO` — keep it
exported in the same shell, or re-export the same value.

## Secrets & runtime settings

`$JAXOS_HOME` is where the app keeps its own state; it defaults to `~/.jax-os`
and is only overridden if you export it.

**Secrets (6)** — create `$JAXOS_HOME/.env` (default `~/.jax-os/.env`), fill
in what you actually use, and lock it down:

```bash
mkdir -p ~/.jax-os
touch ~/.jax-os/.env
chmod 600 ~/.jax-os/.env
```

If you exported a custom `$JAXOS_HOME`, use that directory instead of
`~/.jax-os` in these commands (and in the table's `$JAXOS_HOME/.env`).

| Key | Purpose | Gates in `/settings` |
|---|---|---|
| `LINEAR_API_KEY` | Linear reads and writeback | `linear` |
| `OPENCODE_GO_API_KEY` | OpenCode GO usage token | `agents.opencode` |
| `NOTIFICATION_WEBHOOK_URL` | HMAC-signed notification target | `webhook` |
| `NOTIFICATION_WEBHOOK_SECRET` | Signing secret for that webhook | `webhook` |
| `TYPESAFE_API` | TypeSafe (Jev) key for the message classifier | `classifier` |
| `OPENROUTER_API_KEY_BUILDER` | OpenRouter key for the classifier's fallback model | `classifier` |

**Runtime settings (3)** — not secrets, and never put them in
`$JAXOS_HOME/.env`:

- `PORT` — the port `next start` binds. Set it as the systemd unit's
  `Environment=` line for a persistent install, or export it in the shell for
  the Run step below.
- `TTYD_RO_URL` / `TTYD_RW_URL` — the ttyd listener addresses for the tmux
  viewer. Put these in `$JAXOS_REPO/.env.local` (Next's own env file, read
  directly as `process.env`, per `src/app/api/tmux/config/route.ts`).

The complete key set, for a quick copy:

```text
<!-- env-keys:start -->
LINEAR_API_KEY
OPENCODE_GO_API_KEY
NOTIFICATION_WEBHOOK_URL
NOTIFICATION_WEBHOOK_SECRET
TYPESAFE_API
OPENROUTER_API_KEY_BUILDER
TTYD_RO_URL
TTYD_RW_URL
PORT
<!-- env-keys:end -->
```

### Data egress

Every integration is off by default; **off means zero reads and zero calls**.
One row per integration that ever leaves the box:

| Integration | Egress |
|---|---|
| `classifier` | Message tails / review-finding text / build-failure excerpts → TypeSafe (Jev); on a Jev failure → OpenRouter/DeepSeek |
| `webhook` | The configured URL, HMAC-signed payload, wherever the owner points it |
| `agents.*` | Read-only native OAuth files; no outbound call beyond the provider's own usage endpoint |
| `linear` / `github` | The named provider's own API, credentials from `$JAXOS_HOME/.env` or `gh auth` |

### ttyd, two worked examples

1. **Local browser.** ttyd bound to `127.0.0.1`,
   `TTYD_RO_URL=http://127.0.0.1:7681` — browser and server are the same
   machine, so loopback is the whole protection.
2. **Trusted remote.** `tailscale serve` (or an equivalent private tunnel)
   exposing ttyd to a browser on another device. Loopback bind plus Fetch
   Metadata is **not** authentication once a proxy is in front of it: whatever
   reaches the tunnel reaches ttyd's read-write endpoint too. The tunnel
   itself — Tailscale ACLs, a reverse-proxy auth layer — is the only real
   gate, and BOTH the RO and RW listeners need it, not just the one you
   remember to check. See [`SECURITY.md`](SECURITY.md) for the trust model.

## Run (quick start)

```bash
pnpm build
PORT=3100 pnpm start
```

The server binds `127.0.0.1` only — never `0.0.0.0`. Open
`http://127.0.0.1:3100/settings`.

This foreground `pnpm start` is the disposable way to reach `/settings` for
the next step; the systemd units below are the persistent alternative once
configuration is done.

To background it instead (for example to run `jaxflow doctor` in the same
shell), run the `next` binary directly rather than `pnpm start &` — pnpm
forks a child process, so `$!` captures pnpm's own PID and `kill "$PID"`
leaves the real `next-server` listening:

```bash
./node_modules/.bin/next start -H 127.0.0.1 -p "${PORT:-3100}" &
PID=$!
# ... run jaxflow doctor, curl, etc ...
kill "$PID"
```

## Settings

With the app running, open `/settings` → Geral and set `ownerName`, `locale`,
`reposRoot`, `vaultPath`, and `monitoredUnits`. Leave every integration off
until the secret it needs exists in `$JAXOS_HOME/.env`.

## Global rules

Open `/settings` → Jax Rules. On first run the canonical slot is seeded from
`workflow/templates/global-rules/canonical.md` plus, if you already had a
`CLAUDE.md`/`AGENTS.md` at `$CLAUDE_CONFIG_DIR/CLAUDE.md` /
`$CODEX_HOME/AGENTS.md` (or `~/.claude/CLAUDE.md` / `~/.codex/AGENTS.md` when
those variables are unset), an "imported" block holding that content
as-is, apart from the same whitespace normalization every rule edit gets. The Claude/Codex/opencode exception slots are seeded from the
matching starter file in the same directory. Nothing is written to
`~/.claude/CLAUDE.md`, `~/.codex/AGENTS.md`, or
`~/.config/opencode/AGENTS.md` until you review each slot in the tab and
click Apply.

## systemd

```bash
mkdir -p ~/.config/systemd/user
cp "$JAXOS_REPO"/systemd/*.example ~/.config/systemd/user/
for f in ~/.config/systemd/user/*.example; do mv "$f" "${f%.example}"; done
systemctl --user daemon-reload
systemctl --user enable --now jaxos.service
```

The units use `%h`, which resolves to your home directory — no path
substitution needed for the default clone path. If you cloned `$JAXOS_REPO`
somewhere else, edit the paths inside the copied units before enabling them.
The metrics timer and the workflow poll timer (`jaxos-metrics.timer`,
`jaxos-workflow-poll.timer`) are optional extras; enable them with
`systemctl --user enable --now <unit>` the same way.

This is the persistent alternative to the Run step's foreground `pnpm start`.

## Hooks

Merge — never overwrite — the blocks from
[`workflow/hooks/`](workflow/hooks/):

- `claude-hooks.example.json` → the `hooks` key of `~/.claude/settings.json`.
- `codex-hooks.example.json` → `~/.codex/hooks.json`.

Keep any existing entries alongside the new ones. To remove them later, delete
only the matching `jaxflow-hook <event>` command entries.

On a fresh harness install the settings file may not exist yet — create the
parent directory and an empty `{}` file first, then back up and merge with
`jq`. This keeps every other key in the file and concatenates hook arrays per
event instead of overwriting them:

```bash
mkdir -p ~/.claude
[ -f ~/.claude/settings.json ] || echo '{}' > ~/.claude/settings.json
cp ~/.claude/settings.json ~/.claude/settings.json.bak-$(date +%s)
jq -s '.[0] as $old | .[1].hooks as $new | $old * {hooks: (($old.hooks // {}) as $oh |
  reduce ($new | keys[]) as $k ($oh; .[$k] = (($oh[$k] // []) + $new[$k])))}' \
  ~/.claude/settings.json "$JAXOS_REPO/workflow/hooks/claude-hooks.example.json" \
  > /tmp/settings.merged.json && mv /tmp/settings.merged.json ~/.claude/settings.json
```

The same pattern applies to `~/.codex/hooks.json` with
`codex-hooks.example.json`:

```bash
mkdir -p ~/.codex
[ -f ~/.codex/hooks.json ] || echo '{}' > ~/.codex/hooks.json
cp ~/.codex/hooks.json ~/.codex/hooks.json.bak-$(date +%s)
jq -s '.[0] as $old | .[1].hooks as $new | $old * {hooks: (($old.hooks // {}) as $oh |
  reduce ($new | keys[]) as $k ($oh; .[$k] = (($oh[$k] // []) + $new[$k])))}' \
  ~/.codex/hooks.json "$JAXOS_REPO/workflow/hooks/codex-hooks.example.json" \
  > /tmp/hooks.merged.json && mv /tmp/hooks.merged.json ~/.codex/hooks.json
```

## Skills

Symlink the workflow skills into whichever harnesses you have installed (a
fresh `$HOME` has none of these directories yet):

```bash
mkdir -p ~/.claude/skills ~/.codex/skills ~/.agents/skills
for d in ~/.claude/skills ~/.codex/skills ~/.agents/skills; do
  for s in jaxflow jax-init jaxflow-mission; do
    ln -sfn "$JAXOS_REPO/workflow/skills/$s" "$d/$s"
  done
done
```

## PATH wrappers

```bash
mkdir -p ~/.local/bin
ln -sf "$JAXOS_REPO/bin/jaxflow" "$JAXOS_REPO/bin/jaxflow-hook" "$JAXOS_REPO/bin/jax-init" ~/.local/bin/
```

The wrappers resolve their own location at runtime, so the symlink is enough.
`~/.local/bin` must be on `PATH` for the wrapper names to resolve anywhere
else in this document: if `command -v jaxflow` fails after linking, add
`export PATH="$HOME/.local/bin:$PATH"` to your shell rc file and re-source it.

## Inventory helper

One-time setup for the Tools → Inventory workspace, exactly as
[`workflow/contracts/tools-inventory-setup.md`](workflow/contracts/tools-inventory-setup.md)
specifies (that file is the authority on what the script does and refuses):

```bash
bash "$JAXOS_REPO/scripts/setup-inventory-helper.sh" ~/.jax-os/inventory-venv
```

## Verification

With the app running (Run or systemd):

```bash
jaxflow doctor
```

It is read-only, prints one line per check, and exits 0 iff every required
check passes. An integration-gated check (for example `cli:gh`) is required
only while that integration is on, so a fresh install with every integration
off is expected to pass with those checks showing as not required. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the local check commands, and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces fit
together.
