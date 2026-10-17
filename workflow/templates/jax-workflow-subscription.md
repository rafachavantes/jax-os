# `jax-workflow` — Hermes webhook subscription (Phase B/C)

**Optional** — needs a running Hermes instance; jax-os's own generic webhook,
`integrations.webhook`, needs none of this.

Version-controlled next to the contracts (spec §4.2). **Propose-only:** this
subscription never triggers runs, merges, or any action — Hermes relays, the
owner decides, and Hermes acts only on the owner's explicit reply. The live
subscription stays `--deliver-only`. Phase C correlates answers from the first
line of a forwarded question (`Workflow question #<event_id>`) without editing
the subscription. The topic prompt in
`workflow/templates/jax-workflow-answer-channel-prompt.md` POSTs the explicit
Telegram reply to `/api/workflow/answer`. Activating a `~/.hermes/config.yaml`
edit (topic prompt + topic name) is operator-gated and requires the owner to
restart the Hermes adapter.

## Forward body (what Jax OS POSTs, HMAC-signed)

Fixed key set, always all present (`null` where N/A):

```json
{"event_id": 12, "ts": "2026-08-18T12:00:00.000Z", "type": "run-finished", "project": "jax-os",
 "run_id": "r-001", "role": "builder", "source": "deterministic", "emitter": "wrapper",
 "message": "Project jax-os (phase B): builder run finished, exit 0 — all green. Report at <repo>/.local/reports/r-001.md.",
 "payload": {"phase": "B", "exit_code": 0, "contract_status": "ok", "report_path": "…", "summary": "all green", "head_sha": "<40-hex>"}}
```

`message` is the deterministic one-liner Jax OS already rendered from the event
matrix; Hermes relays it verbatim. `payload` carries the structured fields
(question ids and options for Phase C's answer route). Only `run-finished`,
`question`, `attention-needed` (excerpt stripped) and `turn-stopped` are ever
forwarded; the rest are local.

## Subscription prompt (the `--prompt` value)

The live subscription runs in **deliver-only** mode: the `--prompt` IS the
Telegram message (no agent, zero LLM cost), so it must be the rendered line
plus a short footer, exactly as in the setup step below:

```text
{message}

— Jax OS · {type} #{event_id}
```

The block that follows is the **agent-mode** alternative — only for a setup
that drops `--deliver-only` and lets the agent compose the relay. Do NOT paste
it into a deliver-only subscription: its instructions would be sent verbatim
to Telegram.

```text
Jax OS workflow event {type} #{event_id} for project {project} (source: {source}, emitter: {emitter}).
Message: {message}
Details: {payload}

Relay this to the owner on Telegram as ONE message: send the Message text as-is.
- question: it already starts with `Workflow question #<event_id>` and ends with the indexed-reply instruction — keep both; the owner's one-message reply is POSTed by the topic prompt, not this subscription.
- run-finished: if the tech lead follows up with a proposal, relay that proposal when it arrives.
- attention-needed / turn-stopped: relay the one-liner and viewer pointer, nothing else.
NEVER start a run, merge, deploy, or any other action because of this event. Propose only; act only on the owner's explicit reply.
Do not read the report file; the Message is all you need.
```

Placeholders used: `type`, `event_id`, `project`, `source`, `emitter`,
`message`, `payload` — all always present in the body. Verify Hermes's
placeholder syntax against the installed version (`hermes webhook subscribe
--help`, `gateway/platforms/webhook.py`) before creating; the approvals
subscription uses the same `{key}` form.

## Fallback poll (runs from day one, cursorless)

Executed by a **systemd user timer on the host**, not by a Hermes job:
`scripts/workflow_poll.py` + `systemd/jaxos-workflow-poll.{service,timer}` (every 2 min).
It re-forwards each pending event to this same webhook with the same HMAC
recipe and acks only what returned 200 — so retries cost zero LLM tokens and
Jax OS keeps no retry machinery of its own. Equivalent manual recipe:

```bash
curl -s "http://127.0.0.1:3100/api/workflow/events?pending=1&limit=20"
# → {"ok":true,"data":[<forward bodies>]}: relay each `message` to the owner (same rules as above), then
curl -s -X POST http://127.0.0.1:3100/api/workflow/events/ack -H 'Content-Type: application/json' -d '{"ids":[<exactly the event_ids you relayed>]}'
```

**Deduplicate by `event_id`.** The synchronous push and this poll can both hand
you the same event (the poll may pick up an event whose push is still in
flight, or whose push got a 200 that Jax OS never recorded). Keep the set of
`event_id`s you have already relayed and never send the same one to Telegram
twice. Keep a SET, not a high-water mark: ids are assigned at insert time but
delivered out of order (an event whose push failed stays pending and reappears
on a later poll, after higher ids were already pushed), so "everything below N
is done" would silently drop it.

Anything not acked reappears next poll — nothing is skipped or stranded; a
poison event stays visibly pending on the dashboard (accepted v0). Never
"catch up" by acking ids you did not relay.

## One-time setup (after Hermes is migrated to this VPS)

1. Backup: `cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-jax-workflow-$(date +%Y%m%d%H%M%S)`.
2. Confirm the webhook platform is enabled on `127.0.0.1:8644` (`ss -ltnp | grep 8644` shows 127.0.0.1 only; `hermes webhook list` shows `jax-os-approvals`).
3. **Let Hermes generate the secret** — `--secret` is optional and auto-generated when omitted, so it never has to appear in a command line. (Caveat found in practice: the CLI **prints the generated secret to stdout**. If an agent runs this command, that secret lands in its transcript — rotate afterwards, see step 5.)
4. Subscribe. `--deliver-only` renders the message Jax OS already composed and skips the agent entirely (zero LLM cost per event); note the `=` form for `--deliver-chat-id`, since a leading `-` in a group id is otherwise parsed as a flag:

   ```bash
   hermes webhook subscribe jax-workflow \
     --description "Jax OS workflow events (Phase B) — propose-only relay to the owner" \
     --prompt '{message}

   — Jax OS · {type} #{event_id}' \
     --deliver telegram --deliver-chat-id=<chat-id> --deliver-only
   ```

   **Forum topics:** the CLI stores whatever you pass as `deliver_extra.chat_id`, but the delivery code reads `chat_id` and `thread_id` as SEPARATE keys (`gateway/platforms/webhook.py`), so a `<chat>:<thread>` string silently lands in the group's General topic. Fix it directly in `~/.hermes/webhook_subscriptions.json`:

   ```json
   "deliver_extra": { "chat_id": "-100XXXXXXXXXX", "thread_id": "26" }
   ```

   The gateway hot-reloads `webhook_subscriptions.json` on every request (mtime-gated). Do not restart `hermes-gateway` for a subscription-file change. A `~/.hermes/config.yaml` topic-prompt or topic-name change is different: it stays inactive until the owner performs the operator-gated adapter restart.
5. **Rotate the secret.** Generate a fresh value locally (never printed), write it into `~/.hermes/webhook_subscriptions.json` (chmod `600` — the file ships as `664` and holds every subscription's HMAC secret), and store the same value as `NOTIFICATION_WEBHOOK_SECRET` in `$JAXOS_HOME/.env` (mode `0600`) — or in your own secret manager, syncing that one line into the file. Jax OS reads the value per call, so a rotation takes effect without a restart. **Never put it in `.env.local`:** that file wins over `$JAXOS_HOME/.env`, so a stale copy there would silently survive a rotation.
6. Set `NOTIFICATION_WEBHOOK_URL=http://127.0.0.1:8644/webhooks/jax-workflow` in the same `$JAXOS_HOME/.env` (the approvals URL/secret stay untouched), and turn the `webhook` integration on in `/settings`. Restart Jax OS (`systemctl --user restart jaxos`) and POST one `run-finished` smoke event with `"project":"smoke-b"` — expect `forwarded:true` and one Telegram line in the target topic. `workflow_events` is append-only by design: leave the smoke rows; removing them is a deliberate manual `DELETE FROM workflow_events WHERE project='smoke-b';` against `~/.jax-os/jaxos.db`, never routine.
7. Install and enable the fallback poll timer: `install -m 644 systemd/jaxos-workflow-poll.* ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now jaxos-workflow-poll.timer`. Run one tick (`systemctl --user start jaxos-workflow-poll.service`) and confirm `journalctl --user -u jaxos-workflow-poll` logs `polled N pending, delivered N, acked N`.
