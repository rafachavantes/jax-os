# Review context — read before reviewing

Jax OS is a **personal, single-user** work cockpit. One owner, one machine. Not a SaaS, not multi-tenant.

## Deployment and threat model

- Runs on a single VPS owned by the user, as a systemd user service.
- The Next.js server binds **127.0.0.1 only**. It is never exposed on a public interface.
- Remote access happens **only through Tailscale** (`tailscale serve` on the user's private tailnet). There is no public route to the app.
- SSH is tailnet-only. The host firewall opens only 80/443 publicly, for unrelated services behind a reverse proxy; Jax OS is not routed through it.
- Every process on the host belongs to the same user. There is no untrusted local user.

**Out of scope — do not flag:**
- Missing authentication, sessions, CSRF tokens or rate limiting on API routes (loopback + tailnet is the auth boundary).
- "A local attacker could…" scenarios — there is no other local user.
- Missing CORS hardening, security headers, or multi-tenant isolation.

**Still in scope — do flag:**
- Anything that binds a non-loopback interface (`0.0.0.0`, `::`) or exposes a port beyond loopback.
- File access escaping the allowlist (`/home/rafa/repos`, `/home/rafa/obsidian-vault`): paths must be canonicalized (symlinks, `..`) server-side before the check.
- Secrets reaching the browser, API responses or logs.
- Shell/command injection where a value from a request reaches a subprocess.
- Data loss or corruption in the app's own SQLite DB (`~/.jax-os/jaxos.db`).

## Hard rules (deviations are bugs)

1. **Read by default.** External mutations happen only on an explicit user action and are recorded in the mutations table. Any autonomous action on an external system is a bug.
2. All system access lives in `src/server/collectors/` as pure typed functions; API routes are thin wrappers.
3. Error contract: source unavailable → `200 {ok:false, error}` (visible warning); empty data → `{ok:true, data:[]}` (quiet). Conflating them, or a page crashing from a data source, is a bug.
4. Polling (TanStack Query), never websockets — by design.
5. Every UI string comes from `messages/{pt-BR,en-US}.json`, in both locales. Colors come from design-system tokens only, never raw hex.

## Intentional choices — not bugs

- Polling instead of push.
- No e2e tests; tests cover parsers, validators and DB helpers.
- `/tokens` reads unofficial subscription endpoints on purpose.
- Metrics aggregation runs once per hour by design; disk sizes are decimal GB.

## What we want from this review

This is a whole-codebase review, not a change review. Treat every file as fair game. The project values the
minimum code that works (KISS, YAGNI): the best finding is code we can delete.

Report, in this order of priority:

1. **Simplification**
   - Dead code: unused exports, functions, files, branches, flags, messages keys.
   - Reinvented stdlib, platform or already-installed-dependency features (Node, Next.js, React, TanStack Query,
     better-sqlite3, Python stdlib in `scripts/`).
   - Needless abstraction: an interface or wrapper with one implementation, a factory for one product, config
     for a value that never changes, indirection that only forwards.
   - Duplication: the same logic written in two or more places that should share one helper.
   - Over-long functions or modules that would read better split, or tiny modules that would read better merged.
2. **Improvements**
   - Clearer names, simpler control flow, fewer special cases.
   - Inconsistent patterns: the same problem solved different ways across the codebase — pick the best one.
   - Performance waste with a visible cost (redundant polling, repeated I/O, N+1 queries, re-renders).
   - Tests that assert nothing meaningful, or core logic with no test at all.
3. **Correctness bugs** — logic errors, unhandled edge cases, race conditions, the in-scope security items above.

For each finding: file and line, what to change, what replaces it, and the estimated lines removed or saved.
Do not propose new features, new dependencies, or a rewrite of the architecture.

## Scope of this PR

This PR covers the UI: Next.js routes and API route handlers (`src/app/`), React components (`src/components/`), i18n (`src/i18n/`, `messages/`) and app styles.
Test files are excluded. The rest of the repo exists and is reviewed in sibling PRs: do not flag imports of modules outside this PR as missing.
