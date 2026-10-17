import type Database from "better-sqlite3";
import { CAPSULE_STATUSES, CODEX_EMITTERS, isCodexEmitter, paneKey, type CapsuleRule, type CapsuleStatus, type Source, type WorkflowEventInput } from "../../lib/workflow";
import { readGeneralSettings } from "../settings";

export type PaneRef = { pane: string; tmuxIncarnation: string };

// Pane-derived evidence is Claude-only (spec §3): a Codex event's optional pane metadata is never
// authority for Claude pane correlation, and `jaxos` rows are synthetic audit, not agent activity.
// Every pane-keyed predicate in this file and workflows.ts filters through this fragment (aliased
// with the table's own prefix), so a stored pane-carrying codex row can never open/close a Claude
// turn, resolve its question, or alter its capsule.
const PANE_AGENT_EMITTERS_SQL = ["jaxos", ...CODEX_EMITTERS].map((e) => `'${e}'`).join(", ");
export const paneAgentFilter = (alias = ""): string => `${alias}emitter NOT IN (${PANE_AGENT_EMITTERS_SQL})`;

/**
 * MOVED VERBATIM from workflows.ts (it was at :49-74). Not rewritten — the delivery decision in
 * insertEvent and rung 2 of the ladder must answer "does Rafa still owe an answer?" identically,
 * and two definitions would eventually disagree. workflows.ts re-exports it, so every existing
 * importer and its tests keep working untouched.
 */
export function hasPendingQuestion(db: Database.Database, ref: PaneRef): boolean {
  return hasOpenQuestion(db, ref, true);
}

/**
 * "Does Rafa still owe an answer on this pane?"
 *
 * `deliveredOnly` is the ONE difference between the two callers, and it is a scope, not a
 * second definition of "unresolved":
 *  - the forwarding/suppression decision (insertEvent) cares only about questions that were
 *    actually SENT to Rafa, so it passes true;
 *  - rung 2 of the ladder cares whether an answer is owed AT ALL. A question asked while AFK
 *    is off is stored `delivery: "local"` and never forwarded, but it is still sitting on
 *    Rafa's screen unanswered. Passing true here would make rung 2 nearly dead in normal use.
 */
export function hasOpenQuestion(db: Database.Database, ref: PaneRef, deliveredOnly: boolean): boolean {
  const key = paneKey(ref.pane, ref.tmuxIncarnation);
  const row = db.prepare(
    `SELECT 1 FROM workflow_events q
     WHERE q.pane_key = ? AND q.type = 'question'
       AND (? = 0 OR q.delivery IN ('pending', 'delivered'))
       AND NOT EXISTS (
         SELECT 1 FROM workflow_events r
         WHERE r.type = 'question-resolved' AND r.pane_key = q.pane_key
           AND json_extract(r.payload, '$.tool_use_id') = json_extract(q.payload, '$.tool_use_id'))
       AND NOT EXISTS (
         SELECT 1 FROM workflow_answer_claims c WHERE c.source_event_id = q.id)
       AND NOT EXISTS (
         SELECT 1 FROM workflow_events t
         WHERE t.type = 'turn-stopped' AND t.pane_key = q.pane_key AND t.id > q.id
           AND ${paneAgentFilter("t.")})
     LIMIT 1`,
  ).get(key, deliveredOnly ? 1 : 0);
  return row !== undefined;
}

export type Derived = { capsule_status: CapsuleStatus; capsule_rule: CapsuleRule; source: Source };

const det = (capsule_status: CapsuleStatus, capsule_rule: CapsuleRule): Derived =>
  ({ capsule_status, capsule_rule, source: "deterministic" });

const isCapsuleStatus = (v: unknown): v is CapsuleStatus =>
  typeof v === "string" && (CAPSULE_STATUSES as readonly string[]).includes(v);

/**
 * The stop-state ladder (spec §3.2). Applied in order, first match wins. MUST be called before
 * the event's own row is inserted: rung 2's predicate excludes questions that already have a
 * later turn-stopped, so a post-insert call could never fire it.
 */
export function deriveCapsule(
  db: Database.Database,
  ev: WorkflowEventInput,
  readSettings: typeof readGeneralSettings = readGeneralSettings,
): Derived {
  const p = ev.payload as { capsule_status?: string; capsule_rule?: string; message_tail?: string };

  // Rung 1b - the hook saw the canonical merge question (merge question contract): needs_input by
  // construction. Ingress validates the whole shape; the rule is what keeps the ladder (and Jev)
  // from rewriting the hook's verdict.
  if (p.capsule_rule === "merge-question") return det("needs_input", "merge-question");

  // Rung 1 — the hook matched the capsule tag. An agent that tags itself is believed, but the
  // value is re-checked here rather than cast: ingress is the guard, and a derivation that
  // trusts a cast would store `undefined` the day that guard is loosened.
  if (p.capsule_rule === "tag" && isCapsuleStatus(p.capsule_status)) {
    return { capsule_status: p.capsule_status, capsule_rule: "tag", source: "deterministic" };
  }

  // A Codex event's optional pane is NOT authority (spec §3): native input never consumes a Claude
  // question, attention or turn-start on that pane, so rungs 2/3 stay Claude-pane-only.
  const key = ev.pane && ev.tmux_incarnation && !isCodexEmitter(ev.emitter)
    ? paneKey(ev.pane, ev.tmux_incarnation)
    : null;
  if (key) {
    // Rung 2 — Rafa owes an answer. Same "unresolved" definition the delivery decision uses,
    // WITHOUT its delivery scope: an unanswered question counts whether or not it was forwarded.
    if (hasOpenQuestion(db, { pane: ev.pane!, tmuxIncarnation: ev.tmux_incarnation! }, false)) {
      return det("needs_input", "question_open");
    }

    // Rung 3 — a permission request the current turn has not moved past. The turn-started
    // bound is optional: with none on record the attention still counts (spec §3.3). Both reads
    // are agent-sourced AND pane-Claude-only: a codex attention/start on the same pane is not
    // evidence for a Claude stop.
    const attention = db.prepare(
      `SELECT id FROM workflow_events
       WHERE pane_key = ? AND type = 'attention-needed' AND ${paneAgentFilter()} ORDER BY id DESC LIMIT 1`,
    ).get(key) as { id: number } | undefined;
    if (attention) {
      const started = db.prepare(
        `SELECT id FROM workflow_events
         WHERE pane_key = ? AND type = 'turn-started' AND ${paneAgentFilter()} ORDER BY id DESC LIMIT 1`,
      ).get(key) as { id: number } | undefined;
      if (!started || attention.id > started.id) return det("blocked", "attention");
    }
  }

  // Rungs 4 and 5 — pair this turn with the jaxflow run its own session dispatched.
  // Identity proven by Task 0 (MOA-469): codex-stop pairs too, but Codex runs are constrained to
  // caller=codex and the same project — a legacy run with a missing caller is never proof. Claude
  // keeps its legacy pairing (missing caller still counts) but never pairs with a codex-dispatched
  // run: a UUID-shaped harness_session on a Claude emitter must not collide with a Codex identity.
  if (ev.harness_session && (ev.emitter === "claude-stop" || ev.emitter === "codex-stop")) {
    const callerClause = ev.emitter === "codex-stop"
      ? "AND json_extract(s.payload, '$.caller') = 'codex' AND s.project = ?"
      : "AND (json_extract(s.payload, '$.caller') IS NULL OR json_extract(s.payload, '$.caller') = 'claude')";
    const params = ev.emitter === "codex-stop" ? [ev.harness_session, ev.project] : [ev.harness_session];
    // Rung 4 — ANY run from this session still in flight, not just the newest one. Taking only
    // the newest `run-started` let a finished newer run mask an older one that is still going,
    // reporting `done` while work was running.
    const inFlight = db.prepare(
      `SELECT s.id FROM workflow_events s
       WHERE s.type = 'run-started' AND s.harness_session = ? ${callerClause}
         AND NOT EXISTS (SELECT 1 FROM workflow_events f
                         WHERE f.type = 'run-finished' AND f.run_id = s.run_id)
       LIMIT 1`,
    ).get(...params);
    if (inFlight) return det("waiting", "run_in_flight");

    // Rung 5 — a run that finished SINCE this turn began. The turn-started bound is what makes
    // it "just finished" rather than "finished at some point" (spec §3.2); without it a run
    // from three days ago would still colour today's turn `done`. With no turn-started on
    // record there is no new turn to protect, so the finish still counts (spec §3.3).
    // Runs are paired to their session through run_id and `run-started`, because only
    // `run-started` carries caller_session — `run-finished`'s payload has no such key.
    const finished = db.prepare(
      `SELECT f.id FROM workflow_events f
       JOIN workflow_events s ON s.run_id = f.run_id AND s.type = 'run-started'
       WHERE f.type = 'run-finished' AND s.harness_session = ? ${callerClause}
       ORDER BY f.id DESC LIMIT 1`,
    ).get(...params) as { id: number } | undefined;
    if (finished) {
      // The turn bound is per-identity: a native Codex turn is bounded by its own session's
      // codex-userprompt (never by a pane it may carry), a Claude turn by its pane's Claude
      // turn-started. Using the pane for native input let a newer Claude turn hide a real finish.
      const started = ev.emitter === "codex-stop"
        ? (db.prepare(
            `SELECT id FROM workflow_events
             WHERE harness_session = ? AND type = 'turn-started' AND emitter = 'codex-userprompt'
             ORDER BY id DESC LIMIT 1`,
          ).get(ev.harness_session) as { id: number } | undefined)
        : key
          ? (db.prepare(
              `SELECT id FROM workflow_events
               WHERE pane_key = ? AND type = 'turn-started' AND ${paneAgentFilter()}
               ORDER BY id DESC LIMIT 1`,
            ).get(key) as { id: number } | undefined)
          : undefined;
      if (!started || finished.id > started.id) return det("done", "run_finished");
    }
  }

  // No text to judge: neither rung 6 nor rung 7 is applicable, so settle now rather than queue.
  // `abandoned` already means "unknown and we stopped trying", which is exactly true here;
  // deferring would spend five paid attempts on an empty prompt.
  const messageTail = typeof p.message_tail === "string" ? p.message_tail : "";
  const lines = messageTail.split("\n").map((l) => l.trim()).filter((l) => l !== "");
  if (lines.length === 0) return { capsule_status: "unknown", capsule_rule: "abandoned", source: "behavioral" };

  // Rung 6 is gone (native-answer-delivery spec D1): a synchronous "?"-ending guess intercepted
  // rows before Jev ever ran on them. Every non-empty tail now falls straight through to rung 7,
  // where Jev judges capsule_status ONLY (merge intent is the hook's deterministic merge-question rule).

  // Rung 7 — the leftover guess, queued for the poller, UNLESS integrations.classifier is off:
  // a live, uncached read (no consumer here is high-volume enough to need caching) — {ok:false}
  // (missing/malformed settings.json) takes the same off path, the safe no-egress default
  // (spec MOA-502 Decision 1; the malformed-settings warning itself is MOA-496's own row).
  const settings = readSettings();
  const classifierOn = settings.ok && settings.data.integrations.classifier;
  if (!classifierOn) {
    return { capsule_status: "unknown", capsule_rule: "classifier-off", source: "behavioral" };
  }
  return { capsule_status: "unknown", capsule_rule: "deferred", source: "behavioral" };
}
