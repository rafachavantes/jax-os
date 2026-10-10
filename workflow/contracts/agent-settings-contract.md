# Agent Settings Contract

Cross-language v1 format for `~/.jax-os/agent-settings.json`. Python and TypeScript
must accept and reject the same bytes. Part 2's UI writer is the only production
publisher; this runtime batch is a consumer.

## File

- Production path: `~/.jax-os/agent-settings.json`. Readers may take a path for tests;
  there is no public `--settings` flag.
- Cap 128 KiB. Valid UTF-8. Regular file, owning uid, mode `0600`.
- No symlink on the file or any parent. Reads never create the file.
- Only ENOENT is uninitialized (legacy dispatch continues). Permissions, symlinks,
  non-files, oversized input, malformed JSON, unknown versions, and schema defects
  are named refusals — never silent defaults.

## Refusal codes

| Code | When |
| --- | --- |
| `agent-settings-uninitialized` | Managed builder flags before the file exists |
| `agent-settings-malformed` | JSON/schema/UTF-8/duplicate-key/non-finite/unknown-version defects |
| `agent-settings-too-large` | More than 128 KiB |
| `agent-settings-symlink` | File or parent is a symlink |
| `agent-settings-permissions` | Wrong owner or mode other than `0600` |
| `agent-settings-not-file` | Exists but is not a regular file |
| `agent-settings-override` | Builder name/model/effort override after activation |
| `agent-settings-source-changed` | Native `opencode.json`/`opencode.jsonc` bytes no longer match `source_revisions` |
| `agent-profile-conflict` | Saved profile disagrees with effective native alias/model/variant/routing |
| `reviewer-effort-invalid` | Reviewer CLI effort not in the resolved executor's set |
| `reviewer-model-invalid` | Reviewer CLI model fails the model grammar |

Errors are these codes only. Never interpolate raw settings, env values, or SDK options.

## Document

Snake_case keys in both languages. Exact root keys, no extras:

```
schema_version, revision, reviewers, builders, source_revisions
```

- `schema_version` is the integer `1` (reject bool/float/`true`).
- `revision` is 32 lowercase hex, generated anew by the writer.
- Duplicate JSON keys, `NaN`, `Infinity`, and `-Infinity` are malformed.

### Reviewers

Keyed by reviewer **executor** (`claude`, `codex`), never the caller. Both executors
are required. Each value is exactly `{model, effort}`.

- Codex caller → runtime `claude` → `reviewers.claude`
- Claude caller → runtime `codex` → `reviewers.codex`

CLI `--model`/`--effort` replace only that field; the other still comes from the
resolved executor. With the file absent, preserve caller-keyed `REVIEWER_DEFAULTS`.

Reviewer effort cannot be null.

| Executor | Effort |
| --- | --- |
| Codex | `minimal`, `low`, `medium`, `high`, `xhigh` |
| Claude | `low`, `medium`, `high`, `xhigh`, `max` |

### Builders

Both `default` and `fallback` are required. Each profile is exactly:

```
connection, model, effort, credential, routing
```

No `adapter`, `base_url`, `model_config`, or other copies of native config.

- `connection`: `[A-Za-z0-9][A-Za-z0-9_-]{0,63}`
- `model`: nonempty, ≤256 UTF-8 bytes, no whitespace/control, no leading dash; non-ASCII allowed
- `effort`: `[A-Za-z0-9._-]{1,64}` or `null` for a non-reasoning model
- After activation, selection is always runtime `opencode-builder` plus profile
  `default` or `fallback`. `--fallback` selects Fallback. Legacy builder names and
  builder `--model`/`--effort` refuse `agent-settings-override`.

### Credential

Exactly `{"kind":"native"}` or `{"kind":"env","env":"..."}` (MOA-498 — no remote secret
id).

`env` is `JAX_PROVIDER_[A-Z0-9_]+_API_KEY` or a verified native provider API-key
name (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `XAI_API_KEY`,
`DEEPSEEK_API_KEY`, `GOOGLE_GENERATIVE_AI_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`,
`MISTRAL_API_KEY`, `TOGETHER_API_KEY`, `CEREBRAS_API_KEY`, `COHERE_API_KEY`,
`FIREWORKS_API_KEY`, `PERPLEXITY_API_KEY`). Never `PATH`, `HOME`, webhook tokens, or
arbitrary selectors. Native credentials stay native; do not read
them into this document. The named env var's actual value lives in `$JAXOS_HOME/.env`,
never in this file.

### Routing

`null` (non-OpenRouter, checked against effective config at launch) or an object
whose required keys are `sort` (`price`) and `allow_fallbacks` (boolean). Optional:
`preferred_min_throughput`, `preferred_max_latency`, `max_price`, `only`, `ignore`,
`quantizations`. Reject `order`, `models`, and any other key. Empty optional controls
are omitted.

- A percentile object has exactly one of `p50`/`p75`/`p90`/`p99` → positive finite number
- `max_price` uses only `prompt`/`completion` → nonnegative finite USD-per-million
- `only`/`ignore`: ≤128 unique IDs, each ≤256 chars, no whitespace/control, no overlap
- `quantizations`: a non-empty array of unique strings, each one of `int4`, `int8`,
  `fp4`, `mxfp4`, `nvfp4`, `fp6`, `fp8`, `mxfp8`, `fp16`, `bf16`, `fp32`, `unknown`
  (OpenRouter's `provider.quantizations` wire order). Only providers reporting one of
  the allowed levels are eligible; absent means no filter.

### Source revisions

Exact keys `opencode.json` and `opencode.jsonc` (filenames, never paths). Each value
is `absent` or a 64-hex SHA-256 of the complete source bytes. Stale-source token, not
a credential hash. Never expose source bytes.

## Writer activation (Part 2)

The Tools UI (`/settings` Agentes) is the only production publisher. Hold
`~/.jax-os/agent-settings.lock` (`LOCK_EX|LOCK_NB`, no-follow, owner-only) only
while revalidating and committing. Stage/validate first, write the two native aliases
(`jaxflow-builder-default`, `jaxflow-builder-fallback`), publish this document last.
If settings and projection revisions disagree, refuse launch until reconciliation.

Never bootstrap this file from a test, migration, probe, page read, or convenience
command; fixtures use throwaway roots. Provider/model edits before the first reviewed
Agentes save leave the file absent. Native-credential saves need no host provisioning.
A `{"kind":"env","env":<NAME>}` binding reads its value from `$JAXOS_HOME/.env`;
replacing a shared secret affects every connection that uses it, including consumers
outside Jax.

Partial first activation (`activation-pending`) leaves managed aliases without
authority — reconcile or repair; projected native state is not published. Stale
editor/source revisions refuse `agent-settings-source-changed`; re-read and
explicitly save or reconcile. No automatic cross-profile fallback, historical
replay, or silent transport downgrade.

Until this file exists, `--fallback` and `--resume` refuse
`agent-settings-uninitialized` and legacy `--builder`/`--model` continue. After it
exists, Default/Fallback are the authority. Runtime never creates the file.

### Supported transports

Writer configuration coverage (argv + written `opencode.json`, `test_writer_produced_aliases_capture_supported_transports`) for the three adapters below; the real captured-request transport smoke is three spawns, one per adapter: OpenRouter (`test_saved_profiles_forward_model_reasoning_and_full_routing`), xAI (`test_native_xai_captured_request_uses_own_wire_format`) and OpenAI-compatible (the `openai-compatible` case of the writer test):

| Adapter | npm | Effort on the wire | Routing |
| --- | --- | --- | --- |
| OpenRouter | `@openrouter/ai-sdk-provider` | nested `reasoning.effort` | price-only object |
| xAI | `@ai-sdk/xai` | `reasoningEffort` | `null` |
| OpenAI-compatible | `@ai-sdk/openai-compatible` | `reasoningEffort`, or none (`null`) | `null` |

Unknown adapters/effort templates stay visible as `unsupported-transport` and cannot
activate. Endpoint comparison uses a 30-minute observation window; missing metrics
stay null and never loosen a saved routing draft.

### OpenRouter endpoint comparison

The endpoint route (`/api/opencode-providers/endpoints`) only compares an
`openrouter` connection whose model is configured on that connection. It resolves
the selected connection's own credential — native through the private
`read-credential` reader (never a guessed vendor env name), or an
`{"kind":"env","env":<NAME>}` binding through the same protected path — and sends
no key to any other fetch. OAuth-only connections and connections without a usable
credential are `unavailable`, never a probe of a foreign key. The cache is keyed by
connection, model and non-secret credential identity/revision (the source revision
for native, the hash of the env value for an env binding); it is invalidated on the
next credential registration/replacement, so a rotated key does not reuse stale
endpoint data. Metadata reads stay bounded and
never write a credential.

The Agentes preview renders the returned before/after profile bindings and affected
source files, not just a success flag. Before the first settings activation, an
explicit preview acknowledgement unlocks the activation save; an unchanged draft on
uninitialized settings can then be activated, while an unchanged initialized
settings save remains a no-op.

## Launch, credentials, resume

Fresh managed builds use Default unless `--fallback`. Resume is
`jaxflow build --resume <run_id> [--fallback]`: only the profile NAME is reused;
model/effort/routing/credential resolve again at child start from current settings
and current native OpenCode config. No historical replay, per-run alias, or automatic
cross-profile retry.

`{"kind":"native"}` leaves credentials in OpenCode; jaxflow does not look them up.
An `{"kind":"env","env":<NAME>}` binding injects only the named key, read from
`$JAXOS_HOME/.env`, into the sealed child env. Parent env is unchanged. Keys never
appear in argv, logs, manifests, reports, or callbacks.

Conflicting effective config is refused, not repaired. Resume keeps retained
tracked/untracked work, prior reports, and the original diff base; a checkpoint
mismatch refuses rather than reset/clean/stash. New run id, report, and callback.

`status`/`result` label persisted `launch_selection` as **resolved selection**
(real model; effort or `n/a`). That record is observation only: it is not proof
`Popen` started a child, and it is never the next attempt's runtime input.
