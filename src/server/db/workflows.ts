import type Database from "better-sqlite3";
import { isAbsolute, join } from "node:path";
import {
  CONTRACT_STATUSES, CONTROL_CHARS_RE, DEFAULT_FORWARD_TYPES, EVENT_MATRIX, HOOK_TELEMETRY_RETENTION_MS, HOOK_TELEMETRY_TYPES_SQL, LIMITS, RUN_KINDS, RUNTIMES, STAGES, paneKey, parseIndexedReply,
  type AnswerRequest, type CapsuleStatus, type ContractStatus, type Delivery, type EventType, type Role, type Stage,
  type WorkflowEventInput, type WorkflowEventRow, type WorkflowQuestionAnswer, type WorkflowQuestionShape,
} from "../../lib/workflow";
import { REPOS_ROOT } from "../collectors/projects";
import { childLogPath, lastStreamTextLine } from "../collectors/child-log";
import { redactSecrets } from "../collectors/redact";
import { projectPayload } from "../collectors/workflow-events";
import { isCodexCommand } from "../collectors/workflow-tmux";
import { isAnswerWatcherAlive } from "../collectors/workflow-callbacks";
import type { CodexSnapshot, CodexThread } from "../collectors/workflow-codex";
import type { LedgerRow } from "../collectors/worktrees";
import { deriveCapsule, hasPendingQuestion, paneAgentFilter, type PaneRef } from "./derive";
import { finishMutation, insertMutation } from "./mutations";
import { readGeneralSettings } from "../settings";

export { hasPendingQuestion, type PaneRef } from "./derive";

export const FINALIZATION_ERROR = "answer finalization failed";
export const RECOVERY_ERROR = "answer injection abandoned after restart";
export const RUN_ALREADY_FINISHED = "run already finished";

export type AnswerClaim = {
  question_event_id: number;
  tool_use_id: string | null;
  kind: "structured" | "freeform";
  project: string;
  pane: string | null;
  tmux_incarnation: string | null;
  emitter: string;
  harnessSession: string | null;
  role: Role;
  mutation_id: number;
  question_shapes: WorkflowQuestionShape[] | null;
  answers: WorkflowQuestionAnswer[] | null;
  text: string | null;
};

export type ClaimAnswerResult =
  | { ok: true; claim: AnswerClaim }
  | { ok: false; reason: "not-pending" | "invalid-answer" | "invalid-target" };

// Workflow hub persistence (spec §3.2 tables, §4.2 semantics). Every SQL
// statement for workflow_events / workflow_sessions lives here; db is
// injected so tests run on ":memory:". Append-only events; no runs table.

type RawRow = Omit<WorkflowEventRow, "payload"> & { payload: string | null };
const COLS = "id, ts, run_id, project, role, type, source, emitter, payload, delivery, forwarded_at, pane, tmux_incarnation, harness_session";

function parseStoredPayload(raw: string | null): Record<string, unknown> {
  try {
    const value: unknown = raw ? JSON.parse(raw) : {};
    if (value && typeof value === "object" && !Array.isArray(value)) {
      return value as Record<string, unknown>;
    }
  } catch {
    // Corrupt stored payloads degrade to empty data, never crash a hub poll.
  }
  return {};
}

function rowOf(r: RawRow): WorkflowEventRow {
  return { ...r, payload: parseStoredPayload(r.payload) };
}

// AFK switch singleton (§4.5 step 3). No seed row — absent IS the off default.
export function getAfkEnabled(db: Database.Database): boolean {
  const row = db.prepare("SELECT enabled FROM workflow_afk WHERE id = 1").get() as { enabled: number } | undefined;
  return row ? row.enabled === 1 : false;
}

export function setAfk(db: Database.Database, enabled: boolean, now = new Date()): void {
  db.transaction(() => {
    const previous = getAfkEnabled(db); // read before the upsert, inside this same transaction (Finding 2)
    db.prepare(
      "INSERT INTO workflow_afk (id, enabled, updated_at) VALUES (1, ?, ?) " +
      "ON CONFLICT(id) DO UPDATE SET enabled = excluded.enabled, updated_at = excluded.updated_at",
    ).run(enabled ? 1 : 0, now.toISOString());
    // NOTE (plan-vs-reality): flat fields on the mutation entry, matching every other
    // insertMutation call site in this file (extractMutation JSON.stringifies the whole entry
    // AS mutations.payload — there is no separate nested `payload` key convention here).
    insertMutation(db, { ts: now.toISOString(), kind: "workflow-afk-toggle", enabled, previous });
  }).immediate();
}

// Forward-types allowlist singleton (§5 Decision 2/6). Same no-seed-row/read-fallback shape as
// getAfkEnabled: an absent row falls back to DEFAULT_FORWARD_TYPES (NOT NULL DEFAULT backfills
// every existing row on migration, so a present row always has a valid JSON string).
export function getForwardTypes(db: Database.Database): EventType[] {
  const row = db.prepare("SELECT forward_types FROM workflow_afk WHERE id = 1").get() as { forward_types: string } | undefined;
  if (!row) return [...DEFAULT_FORWARD_TYPES];
  return JSON.parse(row.forward_types) as EventType[];
}

export function setForwardTypes(db: Database.Database, types: EventType[], now = new Date()): void {
  db.transaction(() => {
    const previous = getForwardTypes(db); // read before the upsert, inside this same transaction
    db.prepare(
      "INSERT INTO workflow_afk (id, enabled, updated_at, forward_types) VALUES (1, 0, ?, ?) " +
      "ON CONFLICT(id) DO UPDATE SET forward_types = excluded.forward_types, updated_at = excluded.updated_at",
    ).run(now.toISOString(), JSON.stringify(types));
    insertMutation(db, { ts: now.toISOString(), kind: "workflow-forward-types", types, previous });
  }).immediate();
}

// ---- Phase 3: hide/pin preferences + auto-hide threshold (spec §10, Decisions 15/16/20) ----------

export type ProjectPrefs = { pinned: boolean; hiddenAt: string | null };

// EVERY row, no membership window (spec round 3 F1): a status-only project keeps its pin/hide even
// with no live pane and no event in 30 days. Keyed by project so buildModel joins it per card.
export function getProjectPrefs(db: Database.Database): Record<string, ProjectPrefs> {
  const rows = db.prepare("SELECT project, pinned, hidden_at AS hiddenAt FROM project_prefs").all() as
    { project: string; pinned: number; hiddenAt: string | null }[];
  const prefs: Record<string, ProjectPrefs> = Object.create(null); // same "__proto__" hazard as byProject
  for (const r of rows) prefs[r.project] = { pinned: r.pinned === 1, hiddenAt: r.hiddenAt };
  return prefs;
}

// Boolean-shaped patch (spec §10): `hidden: true` stamps hidden_at = now, `hidden: false` clears it;
// `pinned` never touches hidden_at (F6 — pinning must not reset the hidden baseline). One audited
// `mission-prefs` mutation carrying the previous row (AGENTS.md rule 8), inside one immediate
// transaction like setAfk.
export function setProjectPrefs(
  db: Database.Database,
  project: string,
  patch: { pinned?: boolean; hidden?: boolean },
  now = new Date(),
): ProjectPrefs {
  return db.transaction((): ProjectPrefs => {
    const row = db.prepare("SELECT pinned, hidden_at AS hiddenAt FROM project_prefs WHERE project = ?").get(project) as
      { pinned: number; hiddenAt: string | null } | undefined;
    const previous: ProjectPrefs = row ? { pinned: row.pinned === 1, hiddenAt: row.hiddenAt } : { pinned: false, hiddenAt: null };
    const next: ProjectPrefs = {
      pinned: patch.pinned ?? previous.pinned,
      hiddenAt: patch.hidden === undefined ? previous.hiddenAt : patch.hidden ? now.toISOString() : null,
    };
    db.prepare(
      "INSERT INTO project_prefs (project, pinned, hidden_at) VALUES (?, ?, ?) " +
      "ON CONFLICT(project) DO UPDATE SET pinned = excluded.pinned, hidden_at = excluded.hidden_at",
    ).run(project, next.pinned ? 1 : 0, next.hiddenAt);
    insertMutation(db, { ts: now.toISOString(), kind: "mission-prefs", project, ...patch, previous });
    return next;
  }).immediate();
}

export const ARCHIVE_AFTER_DAYS_DEFAULT = 14;
export const ARCHIVE_AFTER_DAYS_MIN = 1;
export const ARCHIVE_AFTER_DAYS_MAX = 30; // F7: never beyond Decision 6's 30-day retention window

// Same no-seed-row/read-fallback pattern as getAfkEnabled: an absent row IS the default 14.
export function getArchiveAfterDays(db: Database.Database): number {
  const row = db.prepare("SELECT archive_after_days AS days FROM workflow_afk WHERE id = 1").get() as { days: number } | undefined;
  return row ? row.days : ARCHIVE_AFTER_DAYS_DEFAULT;
}

export function setArchiveAfterDays(db: Database.Database, days: number, now = new Date()): void {
  if (!Number.isInteger(days) || days < ARCHIVE_AFTER_DAYS_MIN || days > ARCHIVE_AFTER_DAYS_MAX) {
    throw new Error(`archive_after_days must be an integer ${ARCHIVE_AFTER_DAYS_MIN}-${ARCHIVE_AFTER_DAYS_MAX}`);
  }
  db.transaction(() => {
    const previous = getArchiveAfterDays(db);
    // `enabled = 0` only seeds a brand-new row; ON CONFLICT leaves an existing `enabled` untouched.
    db.prepare(
      "INSERT INTO workflow_afk (id, enabled, updated_at, archive_after_days) VALUES (1, 0, ?, ?) " +
      "ON CONFLICT(id) DO UPDATE SET archive_after_days = excluded.archive_after_days, updated_at = excluded.updated_at",
    ).run(now.toISOString(), days);
    insertMutation(db, { ts: now.toISOString(), kind: "mission-archive-after-days", days, previous });
  }).immediate();
}

export type WorkflowSessionInput = { project: string; session: string; pane: string; role: "lead" | "adhoc"; tmux_incarnation: string };

// §3.2: this table is written on every mapped hook fire (Spec B's watchtower registry) but
// read by NONE of Spec A's own mechanics — every consumer here derives from workflow_events.pane
// directly. getSession is deleted, not kept unused.
export function upsertSession(db: Database.Database, input: WorkflowSessionInput, now = new Date()): void {
  db.prepare(
    `INSERT INTO workflow_sessions (session, pane, tmux_incarnation, project, role, registered_at)
     VALUES (?, ?, ?, ?, ?, ?)
     ON CONFLICT(pane, tmux_incarnation) DO UPDATE SET
       session = excluded.session, project = excluded.project, role = excluded.role, registered_at = excluded.registered_at`,
  ).run(input.session, input.pane, input.tmux_incarnation, input.project, input.role, now.toISOString());
}

// Insert-first (spec §4.2/§4.5). §3.3: a `question` with no pane can never be claimed or
// injected post-re-key — it is forced terminal `local` here, unconditionally, before any other
// delivery rule runs, with a workflow-question-expired audit row in the same transaction.
// immediate: this transaction reads before writing — a deferred one can hit
// SQLITE_BUSY_SNAPSHOT under WAL with a second writer (cold review 2026-08-19)
// Phase 3 (spec §4.3, Decision 6): the lazy telemetry prune runs at most once per minute per
// process. ponytail: a plain mutable module object, not a class — tests reset `lastMs` to force
// a prune; the ceiling is "one process", which is all jaxos.service ever is.
export const pruneState = { lastMs: 0 };
const PRUNE_INTERVAL_MS = 60_000;

// MOA-498 D3: a forward-policy event is only ever queued while the webhook integration is
// on — fail closed when settings are unreadable, exactly like an explicit off.
function webhookIntegrationEnabled(): boolean {
  const settings = readGeneralSettings();
  return settings.ok && settings.data.integrations.webhook === true;
}

export function insertEvent(db: Database.Database, ev: WorkflowEventInput, now = new Date()): WorkflowEventRow {
  const rule = EVENT_MATRIX[ev.type];
  const pane = ev.pane ?? null;
  const tmux_incarnation = ev.tmux_incarnation ?? null;
  // Two producers, ONE column (spec §5). The hooks send `harness_session` at the top level;
  // `jaxflow` sends the same identifier as `caller_session` inside the `run-started` payload
  // and is NOT modified. Normalizing here keeps one writer and one meaning for the column, and
  // lets rungs 4/5 use its index instead of a json_extract scan. Only `run-started` carries the
  // key — `run-finished` has no such field, and is paired to its session through run_id.
  let harness_session = ev.harness_session ?? null;
  if (!harness_session && ev.type === "run-started") {
    const caller = (ev.payload as { caller_session?: unknown }).caller_session;
    if (typeof caller === "string" && caller) harness_session = caller;
  }
  return db.transaction((): WorkflowEventRow => {
    // ONE DELETE inside this same transaction (Decision 6): only HOOK_TELEMETRY_TYPES rows, only
    // strictly older than the 30-day window — never any other kind. No scheduler, no new table.
    if (now.getTime() - pruneState.lastMs >= PRUNE_INTERVAL_MS) {
      db.prepare(`DELETE FROM workflow_events WHERE type IN (${HOOK_TELEMETRY_TYPES_SQL}) AND ts < ?`)
        .run(new Date(now.getTime() - HOOK_TELEMETRY_RETENTION_MS).toISOString());
      pruneState.lastMs = now.getTime();
    }
    // Terminal claim (spec §2.4/§2.6 cold review G1): a second run-finished for the
    // same run_id must never land — cancel/worker races on this same INSERT path,
    // and the loser must fail here, before any row is written, not after.
    if (ev.type === "run-finished") {
      const existing = db
        .prepare("SELECT 1 FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1")
        .get(ev.run_id);
      if (existing) throw new Error(RUN_ALREADY_FINISHED);
    }
    // Idempotent audit (spec §2.6, MOA-453). `jaxflow merge` POSTs this BEFORE it pushes, so
    // §5.3's recovery re-runs the identical command and re-POSTs. Unlike the terminal claim
    // above, this SUPPRESSES instead of throwing: a 409 would make the retry's post() raise,
    // refuse `hub-unreachable`, and strand a merge that is already committed.
    //
    // Key: (project, payload.target, payload.merge_sha). `target` is in the key because a git
    // commit does not encode the ref it was delivered to and the target is re-resolved from a
    // MUTABLE Deploy policy — two refs can point at the same merge commit, and that second
    // delivery deserves its own audit row.
    //
    // First row wins, and that is a SERVER contract, not a restatement of what the producer
    // happens to emit: a same-key event whose non-key fields differ is suppressed too, with the
    // first payload preserved. Returning the STORED row (not a fresh one) is what keeps the
    // route's `delivery` rule deciding forwarding, with no forwarding logic added here.
    //
    // No index: measured 2026-09-08 at 7 rows with a full scan. Spec §2.6 names ~1000
    // `merge-approved` rows as the threshold to revisit.
    if (ev.type === "merge-approved") {
      // `?? null` keeps the binding valid for a payload missing either key — the ingress
      // validator requires both, and `json_extract(...) = NULL` is never true, so such an
      // event simply never suppresses.
      const { target, merge_sha } = ev.payload as { target?: string; merge_sha?: string };
      const existing = db
        .prepare(
          `SELECT ${COLS} FROM workflow_events
           WHERE type = 'merge-approved' AND project = ?
             AND json_extract(payload, '$.target') = ?
             AND json_extract(payload, '$.merge_sha') = ?
           ORDER BY id LIMIT 1`,
        )
        .get(ev.project, target ?? null, merge_sha ?? null) as RawRow | undefined;
      if (existing) return rowOf(existing);
    }
    // MOA-465 Ledger & card: same idempotent-audit shape as merge-approved above, keyed on
    // (project, branch, sha) instead of (project, target, merge_sha) — a re-run of `pr open`
    // at the SAME sha (already-open reuse, decision 8) must never double-post; a fast-forward
    // refresh posts a NEW sha and therefore a NEW row on purpose.
    if (ev.type === "pr-opened") {
      const { branch, sha } = ev.payload as { branch?: string; sha?: string };
      const existing = db
        .prepare(
          `SELECT ${COLS} FROM workflow_events
           WHERE type = 'pr-opened' AND project = ?
             AND json_extract(payload, '$.branch') = ?
             AND json_extract(payload, '$.sha') = ?
           ORDER BY id LIMIT 1`,
        )
        .get(ev.project, branch ?? null, sha ?? null) as RawRow | undefined;
      if (existing) return rowOf(existing);
    }
    let payload = ev.payload;
    let source = ev.source;
    if (ev.type === "turn-stopped") {
      const d = deriveCapsule(db, ev);
      source = d.source;
      payload = { ...ev.payload, capsule_status: d.capsule_status, capsule_rule: d.capsule_rule };
      // Both rung-7 outcomes decided at ingress carry the counter: `deferred` because the
      // poller increments it, `abandoned` because §5's lifecycle table says a row settled
      // here has 0 — an absent counter would read as "written before phase 4".
      if (d.capsule_rule === "deferred" || d.capsule_rule === "abandoned") payload.capsule_attempts = 0;
    }
    const paneLessQuestion = ev.type === "question" && !pane;
    let delivery: Delivery;
    if (paneLessQuestion) {
      delivery = "local"; // step 1 (§4.5, Task A1)
    } else if (rule.policy === "local") {
      delivery = "local"; // step 2 (§4.5, Task A1)
    } else if (!getAfkEnabled(db)) {
      delivery = "local"; // step 3 (§4.5, Task B4) — uniform AFK gate, no type carve-out
    } else if (!getForwardTypes(db).includes(ev.type)) {
      delivery = "local"; // step 4 (MOA-487 §5 Decision 3) — a forward-policy type not opted in
    } else if (!webhookIntegrationEnabled()) {
      // MOA-498 D3: an event created while the integration is off is never queued for future
      // delivery — enabling it later delivers only future events, never a backlog.
      delivery = "local"; // step 4.5
    } else if (rule.suppressWhileQuestionPending && pane && tmux_incarnation && hasPendingQuestion(db, { pane, tmuxIncarnation: tmux_incarnation })) {
      delivery = "suppressed"; // step 5
    } else {
      delivery = "pending"; // step 6
    }
    const ts = now.toISOString();
    const r = db
      .prepare(
        `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery, pane, tmux_incarnation, harness_session)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      )
      .run(ts, ev.run_id, ev.project, ev.role, ev.type, source, ev.emitter, JSON.stringify(payload), delivery, pane, tmux_incarnation, harness_session);
    const id = Number(r.lastInsertRowid);
    if (paneLessQuestion) {
      // NOTE (plan-vs-reality): insertMutation/extractMutation stringify the whole entry AS the
      // mutations.payload column (see claimAnswer's call below) — there is no separate nested
      // `payload` key. Fields go flat on the entry, matching every other insertMutation call site.
      insertMutation(db, {
        ts, kind: "workflow-question-expired", ok: true, error: null,
        question_event_id: id, reason: "question asked with no live pane target; cannot be answered, must be re-asked",
      });
    }
    return { ...ev, payload, source, id, ts, delivery, forwarded_at: null, pane, tmux_incarnation, harness_session };
  }).immediate();
}

// Cursorless poll: the server-side pending filter IS the cursor (spec §4.2). turn-stopped is local
// telemetry (MOA-469 correction) — excluded here so a legacy pending stop row can never leak through
// the fallback poll or crowd out an eligible run notification; it is not marked delivered or deleted.
export function listPending(db: Database.Database, limit: number): WorkflowEventRow[] {
  return (
    db.prepare(`SELECT ${COLS} FROM workflow_events WHERE delivery = 'pending' AND type != 'turn-stopped' ORDER BY id LIMIT ?`).all(limit) as RawRow[]
  ).map(rowOf);
}

export function listDeferred(db: Database.Database, limit: number): { id: number; message_tail: string }[] {
  return db.prepare(
    `SELECT id, json_extract(payload, '$.message_tail') AS message_tail
     FROM workflow_events
     WHERE type = 'turn-stopped'
       AND json_extract(payload, '$.capsule_rule') = 'deferred'
       AND json_extract(payload, '$.message_tail') IS NOT NULL
     ORDER BY id LIMIT ?`,
  ).all(limit) as { id: number; message_tail: string }[];
}

// Ack (sync 200 or poll ack): only ever moves pending → delivered, and
// forwarded_at is only ever set here — the §3.2 invariant lives in this statement.
export function markDelivered(db: Database.Database, ids: number[], now = new Date()): number {
  if (ids.length === 0) return 0;
  const marks = ids.map(() => "?").join(",");
  return db
    .prepare(`UPDATE workflow_events SET delivery = 'delivered', forwarded_at = ? WHERE delivery = 'pending' AND id IN (${marks})`)
    .run(now.toISOString(), ...ids).changes;
}

/**
 * The rung-7 transitions, and the only writer of them. Returns false when the row was not a
 * deferred turn-stopped — a replayed or hostile request can never rewrite a settled verdict.
 * Every branch is ONE statement: an increment that must also decide abandonment cannot be
 * split, or a crash between the halves leaves a row that retries for ever. `indeterminate`
 * reaches the same `unknown`/`abandoned` state the `failed` branch reaches after five
 * retries, in one call — an honest "the model couldn't decide" answer is not a transport
 * failure and must not spend the retry budget (rung-7 indeterminate, 2026-09-22).
 * A merge-question row is never `deferred`, so this never touches it (the hook's `merge_ask` survives).
 */
export function classifyDeferred(
  db: Database.Database,
  id: number,
  outcome: { status: CapsuleStatus } | { failed: true } | { indeterminate: true },
): boolean {
  const sql = "status" in outcome
    ? `UPDATE workflow_events
       SET payload = json_set(payload, '$.capsule_status', ?, '$.capsule_rule', 'classified')
       WHERE id = ? AND type = 'turn-stopped' AND json_extract(payload, '$.capsule_rule') = 'deferred'`
    : "indeterminate" in outcome
    ? `UPDATE workflow_events
       SET payload = json_set(payload, '$.capsule_status', 'unknown', '$.capsule_rule', 'abandoned')
       WHERE id = ? AND type = 'turn-stopped' AND json_extract(payload, '$.capsule_rule') = 'deferred'`
    : `UPDATE workflow_events
       SET payload = json_set(payload,
             '$.capsule_attempts', COALESCE(json_extract(payload, '$.capsule_attempts'), 0) + 1,
             '$.capsule_rule',
             CASE WHEN COALESCE(json_extract(payload, '$.capsule_attempts'), 0) + 1 >= 5
                  THEN 'abandoned' ELSE 'deferred' END)
       WHERE id = ? AND type = 'turn-stopped' AND json_extract(payload, '$.capsule_rule') = 'deferred'`;
  const params = "status" in outcome ? [outcome.status, id] : [id];
  return db.prepare(sql).run(...params).changes === 1;
}

function questionShapes(questions: unknown): WorkflowQuestionShape[] | null {
  if (!Array.isArray(questions) || questions.length === 0) return null;
  return questions.map((q) => {
    const obj = q && typeof q === "object" && !Array.isArray(q) ? q as Record<string, unknown> : {};
    return { multiSelect: obj.multiSelect === true, option_count: Array.isArray(obj.options) ? obj.options.length : 0 };
  });
}

// NOTE (plan-vs-reality): migration 4 moves the race-guard UNIQUE constraint from
// tool_use_id (old PK) to source_event_id (new UNIQUE) — a concurrent double-claim now
// violates that column, not tool_use_id. Updated in lockstep with the rename below;
// not called out as a separate step in the plan.
function isToolUseIdConstraint(e: unknown): boolean {
  return e instanceof Error && e.message.includes("workflow_answer_claims.source_event_id");
}

export function claimAnswer(db: Database.Database, request: AnswerRequest, now = new Date()): ClaimAnswerResult {
  try {
    return db.transaction((): ClaimAnswerResult => {
      const row = db.prepare(
        "SELECT id, ts, project, role, payload, pane, tmux_incarnation FROM workflow_events WHERE id = ? AND type = 'question'",
      ).get(request.question_event_id) as {
        id: number; ts: string; project: string; role: Role; payload: string | null;
        pane: string | null; tmux_incarnation: string | null;
      } | undefined;
      if (!row) return { ok: false, reason: "not-pending" };
      if (!row.pane || !row.tmux_incarnation) return { ok: false, reason: "invalid-target" };
      const ref: PaneRef = { pane: row.pane, tmuxIncarnation: row.tmux_incarnation };
      const key = paneKey(ref.pane, ref.tmuxIncarnation);
      let payload: Record<string, unknown> = {};
      try {
        const v: unknown = row.payload ? JSON.parse(row.payload) : {};
        if (v && typeof v === "object" && !Array.isArray(v)) payload = v as Record<string, unknown>;
      } catch {
        return { ok: false, reason: "not-pending" };
      }
      const tool_use_id = payload.tool_use_id;
      if (typeof tool_use_id !== "string" || !tool_use_id) return { ok: false, reason: "not-pending" };
      // Spec B §4.1 (migration 5): every per-pane guard keys on the generated pane_key column
      // (pane || ':' || tmux_incarnation), never on pane alone — a pane id is reused after a
      // tmux server restart, and a pane-only predicate would let a pre-restart event
      // resolve/close/inject-block a post-restart one (round 3 finding).
      const resolved = db.prepare(
        `SELECT 1 FROM workflow_events
         WHERE type = 'question-resolved' AND pane_key = ?
           AND json_extract(payload, '$.tool_use_id') = ?`,
      ).get(key, tool_use_id);
      const existing = db.prepare(
        "SELECT 1 FROM workflow_answer_claims WHERE source_event_id = ?",
      ).get(row.id);
      const laterStop = db.prepare(
        `SELECT 1 FROM workflow_events WHERE type = 'turn-stopped' AND pane_key = ? AND ${paneAgentFilter()} AND id > ?`,
      ).get(key, row.id);
      const injecting = db.prepare(
        `SELECT 1 FROM workflow_answer_claims c
         JOIN workflow_events q ON q.id = c.source_event_id
         WHERE q.pane_key = ? AND c.status = 'injecting'`,
      ).get(key);
      if (resolved || existing || laterStop || injecting) return { ok: false, reason: "not-pending" };
      const shapes = questionShapes(payload.questions);
      if (!shapes) return { ok: false, reason: "invalid-answer" };
      const parsed = parseIndexedReply(request.reply, shapes);
      if (!parsed.ok) return { ok: false, reason: "invalid-answer" };
      const mutation_id = insertMutation(db, {
        ts: now.toISOString(),
        kind: "workflow-answer",
        project: row.project,
        target: ref.pane,
        question_event_id: row.id,
        tool_use_id,
        answer_count: parsed.answers.length,
        outcome: "pending",
      });
      db.prepare(
        `INSERT INTO workflow_answer_claims
         (source_event_id, kind, tool_use_id, mutation_id, status, claimed_at)
         VALUES (?, 'structured', ?, ?, 'injecting', ?)`,
      ).run(row.id, tool_use_id, mutation_id, now.toISOString());
      return {
        ok: true,
        claim: {
          question_event_id: row.id,
          tool_use_id,
          kind: "structured",
          project: row.project,
          pane: ref.pane,
          tmux_incarnation: ref.tmuxIncarnation,
          emitter: "claude-pretool",
          harnessSession: null,
          role: row.role,
          mutation_id,
          question_shapes: shapes,
          answers: parsed.answers,
          text: null,
        },
      };
    }).immediate();
  } catch (e) {
    if (isToolUseIdConstraint(e)) return { ok: false, reason: "not-pending" };
    throw e;
  }
}

export type FreeformClaimResult =
  | { ok: true; claim: AnswerClaim }
  | { ok: false; reason: "not-answerable" | "not-idle" | "not-pending" | "reply must not start with /" | "length" | "newline" | "control characters" };

export function claimFreeformAnswer(db: Database.Database, request: AnswerRequest, now = new Date()): FreeformClaimResult {
  const text = request.reply;
  if (text.length > LIMITS.freeformReply) return { ok: false, reason: "length" };
  if (text.trimStart().startsWith("/")) return { ok: false, reason: "reply must not start with /" };
  if (text.includes("\n") || text.includes("\r")) return { ok: false, reason: "newline" };
  if (CONTROL_CHARS_RE.test(text)) return { ok: false, reason: "control characters" }; // shared guard, Finding 5
  try {
    return db.transaction((): FreeformClaimResult => {
      const row = db.prepare(
        "SELECT id, project, role, pane, tmux_incarnation, emitter, harness_session FROM workflow_events WHERE id = ? AND type = 'turn-stopped'",
      ).get(request.question_event_id) as {
        id: number; project: string; role: Role; pane: string | null; tmux_incarnation: string | null;
        emitter: string; harness_session: string | null;
      } | undefined;
      // D3: harness_session — not pane/tmux_incarnation — is the freeform claim's identity now.
      // It works identically for a tmux pane or a pane-less native session; a codex-stop row's
      // optional pane metadata (if any) is not consulted here at all.
      if (!row || !row.harness_session) return { ok: false, reason: "not-answerable" };
      const laterStart = db.prepare(
        `SELECT 1 FROM workflow_events WHERE type = 'turn-started' AND harness_session = ? AND id > ?`,
      ).get(row.harness_session, row.id);
      if (laterStart) return { ok: false, reason: "not-idle" };
      const existing = db.prepare("SELECT 1 FROM workflow_answer_claims WHERE source_event_id = ?").get(row.id);
      const injecting = db.prepare(
        `SELECT 1 FROM workflow_answer_claims c
         JOIN workflow_events q ON q.id = c.source_event_id
         WHERE q.harness_session = ? AND c.status = 'injecting'`,
      ).get(row.harness_session);
      if (existing || injecting) return { ok: false, reason: "not-pending" };
      const mutation_id = insertMutation(db, {
        ts: now.toISOString(), kind: "workflow-answer", project: row.project,
        target: row.pane ?? row.harness_session, // D6: a pane label when one exists, the session/thread UUID otherwise
        question_event_id: row.id, tool_use_id: null, answer_count: 1, outcome: "pending",
      });
      db.prepare(
        `INSERT INTO workflow_answer_claims (source_event_id, kind, tool_use_id, mutation_id, status, claimed_at)
         VALUES (?, 'freeform', NULL, ?, 'injecting', ?)`,
      ).run(row.id, mutation_id, now.toISOString());
      return {
        ok: true,
        claim: {
          question_event_id: row.id, tool_use_id: null, kind: "freeform", project: row.project,
          pane: row.pane, tmux_incarnation: row.tmux_incarnation, emitter: row.emitter, harnessSession: row.harness_session,
          role: row.role, mutation_id, question_shapes: null, answers: null, text,
        },
      };
    }).immediate();
  } catch (e) {
    if (isToolUseIdConstraint(e)) return { ok: false, reason: "not-pending" };
    throw e;
  }
}

function boundError(error: string | null): string | null {
  if (error === null) return null;
  return error.length > LIMITS.error ? error.slice(0, LIMITS.error) : error;
}

export function finishAnswer(
  db: Database.Database,
  claim: AnswerClaim,
  ok: boolean,
  error: string | null,
  now = new Date(),
): void {
  const bounded = boundError(error);
  db.transaction(() => {
    finishMutation(db, claim.mutation_id, ok, bounded, ok ? "done" : "failed");
    db.prepare(
      "UPDATE workflow_answer_claims SET status = ? WHERE source_event_id = ? AND status = 'injecting'",
    ).run(ok ? "done" : "failed", claim.question_event_id);
    const payload: Record<string, unknown> = {
      question_event_id: claim.question_event_id,
      tool_use_id: claim.tool_use_id,
      kind: claim.kind,
      mutation_id: claim.mutation_id,
      ok,
    };
    if (bounded) payload.error = bounded;
    db.prepare(
      `INSERT INTO workflow_events
       (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES (?, NULL, ?, ?, 'question-answered', 'deterministic', 'jaxos', ?, 'local')`,
    ).run(now.toISOString(), claim.project, claim.role, JSON.stringify(payload));
  }).immediate();
}

export function failAnswerFinalization(db: Database.Database, claim: AnswerClaim): boolean {
  return db.transaction(() => {
    const row = db.prepare(
      "SELECT status FROM workflow_answer_claims WHERE source_event_id = ? AND status = 'injecting'",
    ).get(claim.question_event_id);
    if (!row) return false;
    finishMutation(db, claim.mutation_id, false, FINALIZATION_ERROR, "failed");
    db.prepare(
      "UPDATE workflow_answer_claims SET status = 'failed' WHERE source_event_id = ? AND status = 'injecting'",
    ).run(claim.question_event_id);
    return true;
  }).immediate();
}

export type StalenessCandidate =
  | {
      transport: "tmux"; pane: string; tmuxIncarnation: string; project: string; role: Role;
      reason: "possibly-stuck" | "no-signal"; referenceEventId: number; payload: Record<string, unknown>;
    }
  | {
      transport: "codex"; threadId: string; project: string; role: Role;
      reason: "possibly-stuck" | "no-signal"; referenceEventId: number; payload: Record<string, unknown>;
    };

export type StalenessResult = {
  candidates: StalenessCandidate[];
  nullIncarnation: number;
  checked: number;
  codexSourceOk: boolean;
  tmuxSourceOk: boolean;
};

export function evaluateStaleness(
  db: Database.Database, now: number, livePaneIds: Set<string>, liveIncarnation: string | null,
  codex: CodexSnapshot | null = null, tmuxSourceOk = true,
): StalenessResult {
  const candidates: StalenessCandidate[] = [];
  let nullIncarnation = 0;
  let checked = 0;
  if (liveIncarnation !== null) {
    const panes = db.prepare(
      `SELECT DISTINCT pane FROM workflow_events WHERE pane IS NOT NULL AND ${paneAgentFilter()}`,
    ).all() as { pane: string }[];

    for (const { pane } of panes) {
      if (!livePaneIds.has(pane)) continue; // dead — Finding 19

      const latestAgent = db.prepare(
        `SELECT tmux_incarnation, project, role FROM workflow_events WHERE pane = ? AND ${paneAgentFilter()} ORDER BY id DESC LIMIT 1`,
      ).get(pane) as { tmux_incarnation: string | null; project: string; role: Role } | undefined;
      if (!latestAgent) continue;
      if (latestAgent.tmux_incarnation === null) { nullIncarnation += 1; continue; } // Finding 28 — no membership-only fallback
      if (latestAgent.tmux_incarnation !== liveIncarnation) continue; // Finding 3 — reissued pane, wrong incarnation

      // Findings 9/30 — checked counts every live, incarnation-confirmed pane, whether or not
      // it goes on to produce a candidate below (a healthy pane is still "checked").
      checked += 1;

      const { project, role } = latestAgent;
      const key = paneKey(pane, liveIncarnation);
      // Rule A (§4.4) + Spec B §4.1 (migration 5): both reference-state lookups are agent-sourced
      // AND keyed on this exact pane_key — a same-pane row from an EXPIRED incarnation must never
      // supersede or invalidate the live agent's own state (Finding 1; round 3 finding: this pair
      // used to be pane-only, which could pair a live incarnation with a stale turn-stopped).
      const ts = db.prepare(
        `SELECT id, ts, payload FROM workflow_events WHERE pane_key = ? AND type = 'turn-stopped' AND ${paneAgentFilter()} ORDER BY id DESC LIMIT 1`,
      ).get(key) as { id: number; ts: string; payload: string } | undefined;
      const tst = db.prepare(
        `SELECT id, ts FROM workflow_events WHERE pane_key = ? AND type = 'turn-started' AND ${paneAgentFilter()} ORDER BY id DESC LIMIT 1`,
      ).get(key) as { id: number; ts: string } | undefined;

      const supersededByLaterStart = tst && (!ts || tst.id > ts.id);
      if (!supersededByLaterStart && ts) {
        const capsule = JSON.parse(ts.payload) as { capsule_status: string; capsule_minutes?: number };
        if (capsule.capsule_status === "done" || capsule.capsule_status === "needs_input" || capsule.capsule_status === "blocked") {
          continue; // no synthetic alert, ever
        }
        if (capsule.capsule_status === "waiting" && typeof capsule.capsule_minutes === "number") {
          const declaredAtMs = Date.parse(ts.ts);
          const n = capsule.capsule_minutes;
          const deadlineMs = declaredAtMs + (n * 60 + Math.max(120, n * 60 * 0.2)) * 1000;
          if (now > deadlineMs) {
            candidates.push({
              transport: "tmux",
              pane, tmuxIncarnation: liveIncarnation, project, role, reason: "possibly-stuck", referenceEventId: ts.id,
              payload: { reason: "possibly-stuck", capsule_minutes: n, declared_at: ts.ts, overdue_by_s: Math.round((now - deadlineMs) / 1000), reference_event_id: ts.id },
            });
          }
          continue;
        }
        // capsule_status === "unknown" — falls through to the no-signal fallback below
      }

      // No-signal fallback: last agent-sourced row for this pane_key.
      const last = db.prepare(
        `SELECT id, ts FROM workflow_events WHERE pane_key = ? AND ${paneAgentFilter()} ORDER BY id DESC LIMIT 1`,
      ).get(key) as { id: number; ts: string };
      const quietForMs = now - Date.parse(last.ts);
      if (quietForMs > 120 * 60 * 1000) {
        candidates.push({
          transport: "tmux",
          pane, tmuxIncarnation: liveIncarnation, project, role, reason: "no-signal", referenceEventId: last.id,
          payload: { reason: "no-signal", since: last.ts, quiet_for_s: Math.round(quietForMs / 1000), reference_event_id: last.id },
        });
      }
    }
  }

  // Native Codex staleness: exact-session identity against the live snapshot (MOA-469 §6). An
  // unavailable source skips ONLY this source. An unloaded/systemError/unknown thread, or a failed
  // runtime query, can never become a "possibly stuck" alert; a working (active) thread is not idle.
  const codexSourceOk = codex?.ok === true;
  if (codexSourceOk) {
    const byId = new Map(codex.threads.map((t) => [t.threadId, t]));
    const sessions = db.prepare(
      `SELECT harness_session AS session, project, role, MAX(ts) AS lastTs
       FROM workflow_events
       WHERE emitter IN ('codex-stop','codex-permission','codex-userprompt') AND harness_session IS NOT NULL
       GROUP BY harness_session, project ORDER BY lastTs DESC`,
    ).all() as { session: string; project: string; role: Role; lastTs: string }[];
    for (const row of sessions) {
      const thread = byId.get(row.session);
      if (!thread || thread.status !== "idle") continue; // unloaded/busy/failed — never stuck
      const { pendingAttention, runInFlight } = codexCurrentCorrelation(db, row.project, row.session);
      if (pendingAttention || runInFlight) continue; // current-session evidence outranks the capsule
      const ts = db.prepare(
        "SELECT id, ts, payload FROM workflow_events WHERE harness_session = ? AND type = 'turn-stopped' AND emitter = 'codex-stop' ORDER BY id DESC LIMIT 1",
      ).get(row.session) as { id: number; ts: string; payload: string } | undefined;
      const tst = db.prepare(
        "SELECT id FROM workflow_events WHERE harness_session = ? AND type = 'turn-started' AND emitter = 'codex-userprompt' ORDER BY id DESC LIMIT 1",
      ).get(row.session) as { id: number } | undefined;
      const supersededByLaterStart = tst && (!ts || tst.id > ts.id);
      if (!supersededByLaterStart && ts) {
        const capsule = JSON.parse(ts.payload) as { capsule_status: string; capsule_minutes?: number };
        if (capsule.capsule_status === "done" || capsule.capsule_status === "needs_input" || capsule.capsule_status === "blocked") continue;
        if (capsule.capsule_status === "waiting" && typeof capsule.capsule_minutes === "number") {
          const declaredAtMs = Date.parse(ts.ts);
          const n = capsule.capsule_minutes;
          const deadlineMs = declaredAtMs + (n * 60 + Math.max(120, n * 60 * 0.2)) * 1000;
          if (now > deadlineMs) {
            candidates.push({
              transport: "codex", threadId: row.session, project: row.project, role: row.role,
              reason: "possibly-stuck", referenceEventId: ts.id,
              payload: { reason: "possibly-stuck", capsule_minutes: n, declared_at: ts.ts, overdue_by_s: Math.round((now - deadlineMs) / 1000), reference_event_id: ts.id },
            });
          }
          continue;
        }
      }
      const last = db.prepare(
        `SELECT id, ts FROM workflow_events
         WHERE harness_session = ? AND emitter IN ('codex-stop','codex-permission','codex-userprompt')
         ORDER BY id DESC LIMIT 1`,
      ).get(row.session) as { id: number; ts: string };
      const quietForMs = now - Date.parse(last.ts);
      if (quietForMs > 120 * 60 * 1000) {
        candidates.push({
          transport: "codex", threadId: row.session, project: row.project, role: row.role,
          reason: "no-signal", referenceEventId: last.id,
          payload: { reason: "no-signal", since: last.ts, quiet_for_s: Math.round(quietForMs / 1000), reference_event_id: last.id },
        });
      }
    }
  }
  return { candidates, nullIncarnation, checked, codexSourceOk, tmuxSourceOk };
}

export function maybeInsertStalenessAlert(db: Database.Database, c: StalenessCandidate, now: number): boolean {
  return db.transaction(() => {
    if (!getAfkEnabled(db)) return false; // condition 1

    if (c.transport === "codex") {
      // Native: exact thread identity — a different session's alert or reference can never
      // suppress or revalidate this one. (No pane-scoped pending-question suppression applies:
      // native Codex has no structured AskUserQuestion events.)
      if (c.reason === "no-signal") {
        const stillLast = db.prepare(
          `SELECT id FROM workflow_events
           WHERE harness_session = ? AND emitter IN ('codex-stop','codex-permission','codex-userprompt')
           ORDER BY id DESC LIMIT 1`,
        ).get(c.threadId) as { id: number } | undefined;
        if (!stillLast || stillLast.id !== c.referenceEventId) return false;
      } else {
        const stillTs = db.prepare(
          "SELECT id FROM workflow_events WHERE harness_session = ? AND type = 'turn-stopped' AND emitter = 'codex-stop' ORDER BY id DESC LIMIT 1",
        ).get(c.threadId) as { id: number } | undefined;
        if (!stillTs || stillTs.id !== c.referenceEventId) return false;
        const laterStart = db.prepare(
          "SELECT 1 FROM workflow_events WHERE harness_session = ? AND type = 'turn-started' AND emitter = 'codex-userprompt' AND id > ? LIMIT 1",
        ).get(c.threadId, c.referenceEventId);
        if (laterStart) return false;
      }
      const already = db.prepare(
        `SELECT payload FROM workflow_events WHERE harness_session = ? AND emitter = 'jaxos' AND type = 'attention-needed'
           AND json_extract(payload, '$.reason') = ?
         ORDER BY id DESC LIMIT 1`,
      ).get(c.threadId, c.reason) as { payload: string } | undefined;
      if (already) {
        const p = JSON.parse(already.payload) as { reference_event_id?: number };
        if (p.reference_event_id === c.referenceEventId) return false;
      }
      insertEvent(db, {
        run_id: null, type: "attention-needed", emitter: "jaxos", source: "deterministic",
        role: c.role, project: c.project, harness_session: c.threadId,
        payload: c.payload,
      });
      return true;
    }

    // tmux (legacy): pane_key-scoped conditions and dedup, unchanged.
    const ref: PaneRef = { pane: c.pane, tmuxIncarnation: c.tmuxIncarnation };
    if (hasPendingQuestion(db, ref)) return false; // condition 2 (§4.5's widened definition)
    const key = paneKey(ref.pane, ref.tmuxIncarnation);

    // condition 3 — revalidate referenceEventId inside this transaction (Finding 46), keyed on
    // pane_key per Spec B §4.1, migration 5 (round 3 finding: this used to be pane-only).
    if (c.reason === "no-signal") {
      const stillLast = db.prepare(
        `SELECT id FROM workflow_events WHERE pane_key = ? AND ${paneAgentFilter()} ORDER BY id DESC LIMIT 1`,
      ).get(key) as { id: number } | undefined;
      if (!stillLast || stillLast.id !== c.referenceEventId) return false;
    } else {
      // Rule A (§4.4): revalidation reads only agent-sourced rows too — same invariant as
      // evaluateStaleness above, and for the same reason (Finding 1).
      const stillTs = db.prepare(
        `SELECT id FROM workflow_events WHERE pane_key = ? AND type = 'turn-stopped' AND ${paneAgentFilter()} ORDER BY id DESC LIMIT 1`,
      ).get(key) as { id: number } | undefined;
      if (!stillTs || stillTs.id !== c.referenceEventId) return false;
      const laterStart = db.prepare(
        `SELECT 1 FROM workflow_events WHERE pane_key = ? AND type = 'turn-started' AND ${paneAgentFilter()} AND id > ? LIMIT 1`,
      ).get(key, c.referenceEventId);
      if (laterStart) return false;
    }

    // condition 4 — dedup against the single most recent jaxos-sourced attention-needed row for
    // this exact pane_key and reason (spec 4.4.3 condition 4; Spec B §4.1 adds the incarnation
    // scoping — an old incarnation's dedup row must never suppress a new one's).
    const already = db.prepare(
      `SELECT payload FROM workflow_events WHERE pane_key = ? AND emitter = 'jaxos' AND type = 'attention-needed'
         AND json_extract(payload, '$.reason') = ?
       ORDER BY id DESC LIMIT 1`,
    ).get(key, c.reason) as { payload: string } | undefined;
    if (already) {
      const p = JSON.parse(already.payload) as { reference_event_id?: number };
      if (p.reference_event_id === c.referenceEventId) return false;
    }

    insertEvent(db, {
      run_id: null, type: "attention-needed", emitter: "jaxos", source: "deterministic",
      role: c.role, project: c.project, pane: c.pane, tmux_incarnation: c.tmuxIncarnation,
      payload: c.payload,
    });
    return true;
  }).immediate();
}

export function abandonInjectingAnswers(db: Database.Database): number {
  return db.transaction(() => {
    const rows = db.prepare(
      "SELECT source_event_id, mutation_id FROM workflow_answer_claims WHERE status = 'injecting'",
    ).all() as { source_event_id: number; mutation_id: number }[];
    for (const row of rows) {
      finishMutation(db, row.mutation_id, false, RECOVERY_ERROR, "abandoned");
      db.prepare("UPDATE workflow_answer_claims SET status = 'abandoned' WHERE source_event_id = ?").run(row.source_event_id);
    }
    return rows.length;
  }).immediate();
}

// ── Mission Control v2 (Spec B) hub reads — §7.1b/§4.1/§4.4/§7.4 ──────────────────────────
//
// Every query below that needs one specific pane's identity binds the generated pane_key column
// (migration 5, Task 2) via paneKey(pane, tmuxIncarnation) — never pane alone, and never pane +
// tmux_incarnation as two separate ANDed predicates either, for consistency with every other
// per-pane query in this file (see Task 2's design note and spec §4.1).
//
// Every PANE-derived read filters with paneAgentFilter(): jaxos audit rows are never activity, and
// a Codex event's optional pane metadata is never authority for a Claude pane (spec §3) — so a
// stored pane-carrying codex row cannot open/close a Claude turn, change its capsule or ownership,
// or resolve its question. Project-level reads (countFreshEvents, getHubMembership, activeRunFor,
// newestEventTs) stay jaxos-only: a native Codex session must still keep its project visible.
// getProjectTimeline filters nothing, so synthetic rows render once a project is already a member
// on its own merits (§7.1b).

export type LiveSnapshot = { paneIds: Set<string>; incarnation: string | null; paneCommands: Map<string, string> };

export type Capsule = { status: string; minutes: number | null; declaredAt: string; eventId: number; mergeAsk: number | null; question: string | null; answerable?: boolean };

export type PendingQuestion = {
  eventId: number;
  pane: string;
  tmuxIncarnation: string;
  payload: Record<string, unknown>;
};

// Session-runtime-label fix: the pane's own harness, for display — never the project-level
// `builder` field from status.md, which names the project's assigned BUILD runtime and can
// legitimately differ from whoever's sitting in this tmux pane (e.g. a Claude tech-lead pane
// on a project whose builder is codex). Prefix-based rather than a hardcoded "claude" literal:
// paneAgentFilter already restricts pane-carrying rows to claude-* emitters (spec §3), so this
// self-corrects if that invariant ever changes, and reads "unknown" for a session-registration-
// only pane with no event row at all (getProjectPanes' own sessionRows fallback).
export type PaneRuntime = "claude" | "codex" | "unknown";

export function runtimeFromEmitter(emitter: string | undefined): PaneRuntime {
  if (emitter?.startsWith("claude-")) return "claude";
  if (emitter?.startsWith("codex-")) return "codex";
  return "unknown";
}

export type HubPane = {
  pane: string;
  tmuxIncarnation: string;
  session: string | null;
  role: "lead" | "adhoc" | null;
  live: boolean;
  working: boolean;
  capsule: Capsule | null;
  pendingQuestion: PendingQuestion | null;
  lastEventTs: string;
  subagentCount: number;
  runtime: PaneRuntime;
};

// A native Codex session row (MOA-469 §2) — NEVER a fabricated pane or incarnation. Membership is
// derived from validated Codex agent events plus the live runtime snapshot; the runtime is the
// liveness source. `working`/`attention` are live-runtime facts; `pendingAttention`/`runInFlight`
// are the current-session correlation used when the runtime is idle. `sourceOk` records whether
// the runtime snapshot was available at all — a missing source is a warning, never an empty
// healthy hub.
export type HubCodexSession = {
  threadId: string;
  status: string;
  activeFlags: string[];
  sourceOk: boolean;
  working: boolean;
  attention: boolean;
  pendingAttention: boolean;
  runInFlight: boolean;
  capsule: Capsule | null;
  lastEventTs: string | null;
  subagentCount: number;
};

export type HubCodexSource = "ok" | "failed" | "truncated" | "absent";
// tmux is a separate liveness source from Codex: a genuine listLivePanes failure is a visible
// source warning, never a quiet healthy zero-agents hub (C4).
export type HubTmuxSource = "ok" | "failed";

export type HubTimelineEvent = {
  id: number;
  ts: string;
  type: string;
  pane: string | null;
  role: string | null;
  emitter: string;
  payload: unknown;
};

export type HubActiveRun = {
  runId: string;
  startedAt: string | null;
  kind: (typeof RUN_KINDS)[number] | null;
  runtime: (typeof RUNTIMES)[number] | null;
  target: string | null;
  count: number;
  lastStep: string | null;
};

// A light per-run summary for the Sessions card — every in-flight run as its own line, unlike
// HubActiveRun (the single newest one, with `count`/`lastStep` for that one summary line only).
export type HubActiveRunSummary = {
  runId: string;
  role: Role;
  kind: (typeof RUN_KINDS)[number] | null;
  runtime: (typeof RUNTIMES)[number] | null;
  target: string | null;
  startedAt: string | null;
};

export type HubLastRun = {
  runId: string;
  role: Role;
  kind: (typeof RUN_KINDS)[number] | null;
  runtime: (typeof RUNTIMES)[number] | null;
  target: string | null;
  finishedAt: string | null;
  outcome: string | null;
  contractStatus: ContractStatus;
  stage: Stage | null;
  diagnostic: string | null;
  headSha: string | null;
  // Round-3 F1: re-validated defensively (own inline 0..999 bound) since a stored row
  // can predate this feature or bypass ingress (raw insertEvent in tests) -- same
  // defensive posture contractStatus/stage already use.
  findings: { high: number; medium: number; low: number } | null;
  runtimeModel: string | null;
  profileName: string | null; // builder-only (Decision 17)
  reportRel: string | null;
  targetRel: string | null;
  // Phase 2 (spec Decision 9, round-3 F3): the build's own --verify/--build, read straight off
  // its run-started payload (already ingress-validated, LIMITS.verify/build), builder + kind
  // build only — the merge form's --checks prefill. Never a guess.
  verifyCommand: string | null;
  buildCommand: string | null;
};

export type HubProject = {
  panes: HubPane[];
  codexSessions: HubCodexSession[];
  codexUntracked: boolean;
  pendingQuestion: PendingQuestion | null;
  freshnessCount: number;
  newestEventTs: string | null;
  timeline: HubTimelineEvent[];
  activeRun?: HubActiveRun | null;
  activeRuns: HubActiveRunSummary[];
  lastRun?: HubLastRun | null;
  loopSummary?: LoopSummary | null;
  heartbeat: number[];
  lastAction: { tool: string; ts: string } | null;
};

export type HubEnvelopeData = {
  byProject: Record<string, HubProject>; historicalTruncated: boolean; codexSource: HubCodexSource; tmuxSource: HubTmuxSource;
  prefs: Record<string, ProjectPrefs>;
  settings: { archiveAfterDays: number };
};

const HUB_WINDOW_MS = 30 * 24 * 60 * 60 * 1000; // 30 days, §7.1b
const HISTORICAL_CAP = 50; // §7.1b

// An "open turn" is a turn-started with no later turn-stopped for the same pane IDENTITY
// (pane, tmux_incarnation) — agent-sourced rows only. This alone is NOT "working": a dead pane
// with an open turn is idle, not working (§4.1, test 17) — see isWorking below.
export function hasOpenTurn(db: Database.Database, ref: PaneRef): boolean {
  const key = paneKey(ref.pane, ref.tmuxIncarnation);
  const started = db
    .prepare(
      `SELECT id FROM workflow_events
       WHERE pane_key = ? AND type = 'turn-started' AND ${paneAgentFilter()}
       ORDER BY id DESC LIMIT 1`,
    )
    .get(key) as { id: number } | undefined;
  if (!started) return false;
  const stopped = db
    .prepare(
      `SELECT 1 FROM workflow_events
       WHERE pane_key = ? AND type = 'turn-stopped' AND ${paneAgentFilter()} AND id > ?
       LIMIT 1`,
    )
    .get(key, started.id);
  return !stopped;
}

// working = an open turn AND the pane is live in the CURRENT tmux snapshot with a matching
// incarnation (Rule A, §4.1). A pane reused under a new incarnation, or absent from the
// snapshot, is never "working" — it resolves to idle.
//
// MOA-469 §4: an old Claude open turn on a pane whose CURRENT process is positively Codex must
// not keep the project "working" indefinitely — that turn belongs to the previous occupant. A
// generic shell/unknown command is NOT proof of a runtime switch: uncertainty is flagged (the
// open turn keeps its working claim) rather than assigning a different session by cwd.
export function isWorking(db: Database.Database, ref: PaneRef, live: LiveSnapshot): boolean {
  if (!live.paneIds.has(ref.pane) || live.incarnation !== ref.tmuxIncarnation) return false;
  const started = db
    .prepare(
      `SELECT id, emitter FROM workflow_events
       WHERE pane_key = ? AND type = 'turn-started' AND ${paneAgentFilter()}
       ORDER BY id DESC LIMIT 1`,
    )
    .get(paneKey(ref.pane, ref.tmuxIncarnation)) as { id: number; emitter: string } | undefined;
  if (!started) return false;
  const stopped = db
    .prepare(
      `SELECT 1 FROM workflow_events
       WHERE pane_key = ? AND type = 'turn-stopped' AND ${paneAgentFilter()} AND id > ?
       LIMIT 1`,
    )
    .get(paneKey(ref.pane, ref.tmuxIncarnation), started.id);
  if (stopped) return false;
  const paneCmd = live.paneCommands.get(ref.pane) ?? "";
  if (started.emitter === "claude-userprompt" && isCodexCommand(paneCmd)) return false;
  return true;
}

// Events for `project` newer than its status.md mtime, excluding emitter='jaxos' (§4.4). The
// caller supplies the mtime — this module never touches the filesystem (house rule 2).
export function countFreshEvents(db: Database.Database, project: string, statusMtime: string): number {
  const row = db
    .prepare("SELECT COUNT(*) AS c FROM workflow_events WHERE project = ? AND emitter != 'jaxos' AND ts > ?")
    .get(project, statusMtime) as { c: number };
  return row.c;
}

// merge-ask spec §4: the last non-empty trimmed line that CONTAINS a "?" anywhere, searched
// from the end — not "ends with", which is rung 6's own narrower heuristic (a trailing_question
// row's last line ends in "?" by construction, so it trivially also "contains" one; a rung-7
// row's often doesn't, which is exactly why it fell through to Jev). Falls back to the last
// non-empty trimmed line verbatim when no line contains "?". Capped at LIMITS.capsuleExcerpt,
// the existing bound for this same kind of on-card excerpt.
function deriveQuestion(messageTail: string | undefined): string | null {
  if (!messageTail) return null;
  const lines = messageTail.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  if (lines.length === 0) return null;
  let line: string | undefined;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].includes("?")) { line = lines[i]; break; }
  }
  line ??= lines[lines.length - 1];
  return line.length > LIMITS.capsuleExcerpt ? line.slice(0, LIMITS.capsuleExcerpt) : line;
}

// A successfully answered ask is closed: the card must stop offering its buttons before the next
// turn-started lands (a second click would only race the poll and get not-idle). A failed answer
// (ok:false) leaves the ask open.
function isAnswered(db: Database.Database, eventId: number): boolean {
  return db.prepare(
    `SELECT 1 FROM workflow_events WHERE type = 'question-answered'
       AND json_extract(payload, '$.ok') = 1 AND json_extract(payload, '$.question_event_id') = ? LIMIT 1`,
  ).get(eventId) !== undefined;
}

// project-scoped (Finding 1): a pane_key persists across a project switch, so a pane_key-only
// query here would surface the OTHER project's latest turn-stopped capsule on this project's row
// once the pane moves on. `project` is always the project currently being enumerated by
// getProjectPanes, never "whatever this pane_key's current project is".
function lastCapsule(
  db: Database.Database, ref: PaneRef, project: string, isCurrentProject: boolean,
  isAlive: typeof isAnswerWatcherAlive = isAnswerWatcherAlive,
): Capsule | null {
  const key = paneKey(ref.pane, ref.tmuxIncarnation);
  const row = db
    .prepare(
      `SELECT id, ts, emitter, harness_session, payload FROM workflow_events
       WHERE pane_key = ? AND project = ? AND type = 'turn-stopped' AND ${paneAgentFilter()}
       ORDER BY id DESC LIMIT 1`,
    )
    .get(key, project) as
      { id: number; ts: string; emitter: string; harness_session: string | null; payload: string | null } | undefined;
  if (!row || !row.payload || isAnswered(db, row.id)) return null;
  try {
    const p = JSON.parse(row.payload) as {
      capsule_status?: string; capsule_minutes?: number; capsule_rule?: string;
      merge_ask?: number; message_tail?: string;
    };
    if (!p.capsule_status) return null;
    if ((p.capsule_status === "needs_input" || p.capsule_status === "blocked") &&
        (p.capsule_rule === "classified" || p.capsule_rule === "trailing_question")) {
      const progressedTurn = db.prepare(`
        SELECT 1 FROM workflow_events
        WHERE project = ? AND pane_key = ? AND type = 'turn-started'
          AND ${paneAgentFilter()} AND id > ? LIMIT 1
      `).get(project, key, row.id);
      if (progressedTurn) return null;

      // A pane capsule is Claude-only now (codex rows are excluded above), so the run pairing is
      // the Claude caller — a UUID-shaped Claude session must never pair with a codex-dispatched run.
      const caller = row.emitter === "claude-stop" ? "claude" : null;
      if (isCurrentProject && row.harness_session && caller) {
        const run = db.prepare(`
          SELECT r.ts, EXISTS (
            SELECT 1 FROM workflow_events f
            WHERE f.type = 'run-finished' AND f.run_id = r.run_id
          ) AS finished
          FROM workflow_events r
          WHERE r.project = ? AND r.harness_session = ?
            AND r.type = 'run-started' AND r.run_id IS NOT NULL AND r.id > ?
            AND json_extract(r.payload, '$.caller') = ?
          ORDER BY finished ASC, r.id DESC LIMIT 1
        `).get(project, row.harness_session, row.id, caller) as
          { ts: string; finished: number } | undefined;
        if (run) return run.finished ? null : {
          status: "waiting", minutes: null, declaredAt: run.ts, eventId: row.id, mergeAsk: null, question: null, answerable: true,
        };
      }
    }
    // D4: liveness, not a capsule_rule branch — every needs_input/blocked status gets it,
    // whichever rung set it. done/waiting (above) and codex-stop (lastCodexCapsule) never call it.
    const answerable = (p.capsule_status === "needs_input" || p.capsule_status === "blocked")
      ? row.harness_session !== null && isAlive(row.harness_session, row.id)
      : true;
    return {
      status: p.capsule_status,
      minutes: typeof p.capsule_minutes === "number" ? p.capsule_minutes : null,
      declaredAt: row.ts,
      eventId: row.id, // Phase 2 Decision 2: the free-text answer's event_id
      mergeAsk: typeof p.merge_ask === "number" ? p.merge_ask : null,
      question: deriveQuestion(p.message_tail),
      answerable,
    };
  } catch {
    return null;
  }
}

// Selects the pending question ROW itself — not "is any question pending, then take the newest
// question" (Finding 5 of the cold review: those are different questions when an older question
// is still pending and a newer one has already resolved; the two-query form returned the newer,
// non-pending one). Same predicates as hasPendingQuestion (pending/delivered delivery, no
// matching question-resolved, no answer claim, no later turn-stopped), applied in one query and
// scoped to pane_key throughout — so this function no longer calls hasPendingQuestion at all.
// project-scoped throughout, including both correlated subqueries (Finding 1) — same reasoning
// as lastCapsule above: a pane_key-only lookup would surface another project's pending question
// (or let another project's turn-stopped/question-resolved wrongly close out this project's own
// question) once the pane has moved on to it.
function pendingQuestionFor(db: Database.Database, ref: PaneRef, project: string): PendingQuestion | null {
  const row = db
    .prepare(
      `SELECT q.id, q.payload FROM workflow_events q
       WHERE q.pane_key = ? AND q.project = ? AND q.type = 'question' AND q.delivery IN ('pending', 'delivered')
         AND NOT EXISTS (
           SELECT 1 FROM workflow_events r
           WHERE r.type = 'question-resolved' AND r.pane_key = q.pane_key AND r.project = q.project
             AND json_extract(r.payload, '$.tool_use_id') = json_extract(q.payload, '$.tool_use_id'))
         AND NOT EXISTS (
           SELECT 1 FROM workflow_answer_claims c WHERE c.source_event_id = q.id)
         AND NOT EXISTS (
           SELECT 1 FROM workflow_events t
           WHERE t.type = 'turn-stopped' AND t.pane_key = q.pane_key AND t.project = q.project AND t.id > q.id
             AND ${paneAgentFilter("t.")})
       ORDER BY q.id DESC LIMIT 1`,
    )
    .get(paneKey(ref.pane, ref.tmuxIncarnation), project) as { id: number; payload: string | null } | undefined;
  if (!row) return null;
  let payload: Record<string, unknown> = {};
  if (row.payload) {
    try {
      const v: unknown = JSON.parse(row.payload);
      if (v && typeof v === "object" && !Array.isArray(v)) payload = v as Record<string, unknown>;
    } catch {
      // malformed payload — surface the question with an empty body rather than dropping it
    }
  }
  return { eventId: row.id, pane: ref.pane, tmuxIncarnation: ref.tmuxIncarnation, payload };
}

// Resolves the CURRENT project for each live pane_key. The workflow_sessions upsert
// (ON CONFLICT(pane, tmux_incarnation) DO UPDATE SET ... project = excluded.project) keeps
// exactly one row per pane_key holding its latest project, which is what "current" means here —
// unlike a plain `SELECT DISTINCT project FROM workflow_events WHERE pane_key IN (...)`, which
// admits EVERY project that pane_key ever emitted an event for. The hook derives project from
// cwd per event (scripts/jaxflow_hook.py:317-327), so the same pane identity can legitimately
// switch projects without a tmux restart; once it does, the project it left behind must stop
// reading live (cold review Finding 2). Falls back to the latest agent-sourced event's project
// only when no session row exists yet — the hook posts the session registration BEFORE the event
// (scripts/jaxflow_hook.py:317-327), so a successful session POST followed by a failed event POST
// can leave a live pane with no session row at all (Finding 4, still handled here as the
// fallback path).
//
// Reconciled by recency (Finding 2 of the branch review): session registration is itself
// best-effort, so a pane can emit a successful project-B event while an earlier project-A session
// row is still sitting there un-updated. Whichever of the two is NEWER (session's registered_at
// vs. the latest agent event's ts) wins; the session row wins only the tie / no-event case, same
// as before.
function getCurrentProjectByPaneKey(db: Database.Database, live: LiveSnapshot): Map<string, string> {
  const map = new Map<string, string>();
  if (live.incarnation === null || live.paneIds.size === 0) return map;
  const incarnation = live.incarnation; // narrow once outside the closures below
  // A dynamic pane_key IN (...) list, one bound param per live pane — there is no `pane`
  // column reference here at all, only the generated pane_key column (spec §4.1).
  const keys = [...live.paneIds].map((pane) => paneKey(pane, incarnation));
  const placeholders = keys.map(() => "?").join(",");
  const sessionRows = db
    .prepare(`SELECT pane_key AS paneKey, project, registered_at AS registeredAt FROM workflow_sessions WHERE pane_key IN (${placeholders})`)
    .all(...keys) as { paneKey: string; project: string; registeredAt: string }[];
  const sessionByKey = new Map(sessionRows.map((r) => [r.paneKey, r]));
  // ponytail: a plain per-key loop, not a window function — bounded by the handful of live panes,
  // same style as every other "latest row for this pane_key" query in this file (hasOpenTurn,
  // lastCapsule, pendingQuestionFor).
  for (const key of keys) {
    const session = sessionByKey.get(key);
    const eventRow = db
      .prepare(
        `SELECT project, ts FROM workflow_events
         WHERE pane_key = ? AND ${paneAgentFilter()}
         ORDER BY id DESC LIMIT 1`,
      )
      .get(key) as { project: string; ts: string } | undefined;
    if (session && eventRow) {
      // ISO-8601 timestamps (both from now.toISOString()) sort correctly under string comparison.
      map.set(key, eventRow.ts > session.registeredAt ? eventRow.project : session.project);
    } else if (session) {
      map.set(key, session.project);
    } else if (eventRow) {
      map.set(key, eventRow.project);
    }
  }
  return map;
}

// Membership (§7.1b): every project with a live pane, OR an agent-sourced event in the last 30
// days. Live projects are NEVER capped. Non-live ("historical") projects cap at 50, newest
// activity first, with `historicalTruncated` reported rather than implied completeness.
export function getHubMembership(
  db: Database.Database,
  live: LiveSnapshot,
  now: number,
): { projects: string[]; historicalTruncated: boolean } {
  const liveProjects = new Set(getCurrentProjectByPaneKey(db, live).values());

  const windowStart = new Date(now - HUB_WINDOW_MS).toISOString();
  const historicalRows = db
    .prepare(
      `SELECT project, MAX(ts) AS newest FROM workflow_events
       WHERE emitter != 'jaxos' AND ts > ?
       GROUP BY project ORDER BY newest DESC`,
    )
    .all(windowStart) as { project: string; newest: string }[];

  const historical = historicalRows.filter((r) => !liveProjects.has(r.project));
  const historicalTruncated = historical.length > HISTORICAL_CAP;
  const projects = [...liveProjects, ...historical.slice(0, HISTORICAL_CAP).map((r) => r.project)];
  return { projects, historicalTruncated };
}

// Every (pane, tmux_incarnation) pair that has produced an agent-sourced event for `project`,
// newest activity first, joined against workflow_sessions for session/role and against the
// current live snapshot for liveness/working.
export function getProjectPanes(
  db: Database.Database, project: string, live: LiveSnapshot, nowMs: number = Date.now(),
  isAlive: typeof isAnswerWatcherAlive = isAnswerWatcherAlive,
): HubPane[] {
  // `emitter` rides along as a bare column next to MAX(ts): SQLite's documented min()/max()
  // extension resolves it from the SAME row as each group's max ts, so this is the pane's
  // LATEST emitter, not an arbitrary one from the group (verified against better-sqlite3).
  const rows = db
    .prepare(
      `SELECT pane, tmux_incarnation AS tmuxIncarnation, MAX(ts) AS lastEventTs, emitter
       FROM workflow_events
       WHERE project = ? AND pane IS NOT NULL AND tmux_incarnation IS NOT NULL AND ${paneAgentFilter()}
       GROUP BY pane, tmux_incarnation
       ORDER BY lastEventTs DESC`,
    )
    .all(project) as { pane: string; tmuxIncarnation: string; lastEventTs: string; emitter: string }[];

  // Finding 4 (cold review): a session can be registered with no event row yet — the hook posts
  // the session before the event, and the event POST can fail (scripts/jaxflow_hook.py:317-327).
  // Union session registrations for this project too, so that pane is still enumerated — but
  // ONLY while it is still the CURRENT live pane under the CURRENT live incarnation (Finding 6,
  // round-2 cold review): with no liveness/incarnation filter here, a session row for a pane that
  // registered once and then died would render as a dead pane forever, since nothing ever cleans
  // up workflow_sessions. tmux is already the source of truth for liveness — there is no TTL to
  // invent, just the same live-snapshot check every other liveness read in this file uses. A
  // session-only pane that passes the filter has no event `ts` to report; its fallback
  // lastEventTs is its own registered_at, the only timestamp this module has for it.
  const sessionRows = (
    db
      .prepare(
        `SELECT pane, tmux_incarnation AS tmuxIncarnation, registered_at AS registeredAt
         FROM workflow_sessions WHERE project = ?`,
      )
      .all(project) as { pane: string; tmuxIncarnation: string; registeredAt: string }[]
  ).filter((row) => row.tmuxIncarnation === live.incarnation && live.paneIds.has(row.pane));

  const byKey = new Map<string, { pane: string; tmuxIncarnation: string; lastEventTs: string; emitter?: string }>();
  for (const row of rows) byKey.set(paneKey(row.pane, row.tmuxIncarnation), row);
  for (const row of sessionRows) {
    const key = paneKey(row.pane, row.tmuxIncarnation);
    if (!byKey.has(key)) {
      // No event row at all yet — nothing to derive a runtime from (emitter stays undefined).
      byKey.set(key, { pane: row.pane, tmuxIncarnation: row.tmuxIncarnation, lastEventTs: row.registeredAt });
    }
  }
  const merged = [...byKey.values()].sort((a, b) => b.lastEventTs.localeCompare(a.lastEventTs));

  // A pane row belongs to THIS project's live state only if `project` is still the pane_key's
  // CURRENT project (cold review Finding 2) — a pane that switched to another project without a
  // tmux restart keeps this project's historical row (it genuinely happened), but that row must
  // read live:false/working:false here, not "live" for a project the pane already left.
  const currentProjectByPaneKey = getCurrentProjectByPaneKey(db, live);

  return merged.map((row) => {
    const ref: PaneRef = { pane: row.pane, tmuxIncarnation: row.tmuxIncarnation };
    const sessionRow = db
      .prepare("SELECT session, role FROM workflow_sessions WHERE pane_key = ?")
      .get(paneKey(ref.pane, ref.tmuxIncarnation)) as { session: string; role: "lead" | "adhoc" } | undefined;
    const isCurrentProject = currentProjectByPaneKey.get(paneKey(ref.pane, ref.tmuxIncarnation)) === project;
    return {
      pane: ref.pane,
      tmuxIncarnation: ref.tmuxIncarnation,
      session: sessionRow?.session ?? null,
      role: sessionRow?.role ?? null,
      live: isCurrentProject,
      working: isCurrentProject && isWorking(db, ref, live),
      capsule: lastCapsule(db, ref, project, isCurrentProject, isAlive),
      pendingQuestion: pendingQuestionFor(db, ref, project),
      runtime: runtimeFromEmitter(row.emitter),
      subagentCount: openSubagentCount(db, project, { paneKey: paneKey(ref.pane, ref.tmuxIncarnation) }, nowMs),
      lastEventTs: row.lastEventTs,
    };
  });
}

// Most recent `limit` events for a project, newest first (§7.4 caps this at 20). Unlike every
// query above, this one does NOT filter emitter — synthetic jaxos rows render in the timeline
// once the project is a member on its own merits (§7.1b).
export function getProjectTimeline(db: Database.Database, project: string, limit = 20): HubTimelineEvent[] {
  const rows = db
    .prepare(
      `SELECT id, ts, type, pane, role, emitter, payload
       FROM workflow_events WHERE project = ? AND type NOT IN (${HOOK_TELEMETRY_TYPES_SQL}) ORDER BY id DESC LIMIT ?`,
    )
    .all(project, limit) as {
    id: number;
    ts: string;
    type: string;
    pane: string | null;
    role: string | null;
    emitter: string;
    payload: string | null;
  }[];
  return rows.map((r) => ({
    id: r.id,
    ts: r.ts,
    type: r.type,
    pane: r.pane,
    role: r.role,
    emitter: r.emitter,
    payload: r.payload
      ? projectPayload(r.type as EventType, parseStoredPayload(r.payload))
      : null,
  }));
}

// Phase 3 (spec §8): the child.log tail read is filesystem I/O — injected, never done by this
// module directly (AGENTS.md rule 2). Returns null on any failure or when no reader is injected.
export type ChildLogTailReader = (path: string) => string | null;
// Phase 3 (spec §11, round-1 plan review F2): the inverse-canonicalization read, injected the same
// way. PROJECT-root relative (`/home/rafa/repos/<project>`, e.g. ".local/docs/reports/x.md", never
// "jax-os/..."). Files URL grammar on main: `filesUrl.ts:14-20,31-44` pass `rel` through as-is;
// `files.ts:14,270` resolve `resolveWithin(ROOTS.repos, rel)` — so the Files page's own href is
// `repos:<project>/<rel>`; Part 2's `LastRunLine` prepends `${project}/` once. Never imports files.ts.
export type RelToRepos = (project: string, absolutePath: string) => string | null;

// spec §6, Decision 5: pure zero-fill fold over one indexed query's rows — 60 one-minute buckets,
// oldest first, current (possibly partial) minute last. `rows` keys are "YYYY-MM-DDTHH:MM" —
// exactly `Date.prototype.toISOString().slice(0, 16)`'s shape, which is also what SQLite's
// strftime('%Y-%m-%dT%H:%M', ts) produces for an ISO `ts` column.
export function heartbeatFromRows(rows: { minute: string; n: number }[], nowMs: number): number[] {
  const buckets = new Map(rows.map((r) => [r.minute, r.n]));
  const out: number[] = [];
  for (let i = 59; i >= 0; i--) {
    const minute = new Date(nowMs - i * 60_000).toISOString().slice(0, 16);
    out.push(buckets.get(minute) ?? 0);
  }
  return out;
}

// cold review F1: the oldest bucket heartbeatFromRows emits is i=59 (nowMs - 59min), not -60min —
// align the SQL cutoff to that bucket's minute start, or an event in the dropped 60th-to-59th-minute
// gap is queried here and then silently discarded by the fold.
function heartbeatFor(db: Database.Database, project: string, nowMs: number): number[] {
  const oldestBucketMinute = new Date(nowMs - 59 * 60_000).toISOString().slice(0, 16);
  const windowStart = `${oldestBucketMinute}:00.000Z`;
  const rows = db.prepare(
    `SELECT strftime('%Y-%m-%dT%H:%M', ts) AS minute, COUNT(*) AS n
     FROM workflow_events WHERE project = ? AND ts >= ? AND emitter != 'jaxos'
     GROUP BY minute`,
  ).all(project, windowStart) as { minute: string; n: number }[];
  return heartbeatFromRows(rows, nowMs);
}

// spec §6, round 2 F6: the latest tool-used row, `null` when there is none or it is > 24h old
// (exactly 24h still counts — the boundary is inclusive: `ageMs > 24h` excludes, not `>=`).
const LAST_ACTION_WINDOW_MS = 24 * 60 * 60 * 1000;

function lastActionFor(db: Database.Database, project: string, nowMs: number): { tool: string; ts: string } | null {
  // cold review F2: filter by the cutoff and order by ts before limiting, or a late-inserted older
  // row (higher id, older ts) hides a newer qualifying row that was inserted first.
  const cutoff = new Date(nowMs - LAST_ACTION_WINDOW_MS).toISOString();
  const row = db.prepare(
    `SELECT ts, json_extract(payload, '$.tool') AS tool FROM workflow_events
     WHERE project = ? AND type = 'tool-used' AND ts >= ? ORDER BY ts DESC, id DESC LIMIT 1`,
  ).get(project, cutoff) as { ts: string; tool: string | null } | undefined;
  if (!row || !row.tool) return null;
  const ageMs = nowMs - Date.parse(row.ts);
  if (!Number.isFinite(ageMs) || ageMs > LAST_ACTION_WINDOW_MS) return null;
  return { tool: row.tool, ts: row.ts };
}

// spec §7, Decision 7 / cold review F11: an unmatched subagent-started (no LATER subagent-stopped
// sharing its agent_id) counts, unless it is STRICTLY older than 30 minutes (exactly 30 still
// counts) — scoped by project PLUS pane_key (Claude) or harness_session (Codex), never pane_key
// alone, which persists across a project switch.
const SUBAGENT_ORPHAN_MS = 30 * 60_000;

function openSubagentCount(
  db: Database.Database,
  project: string,
  identity: { paneKey: string } | { harnessSession: string },
  nowMs: number,
): number {
  const scopeCol = "paneKey" in identity ? "pane_key" : "harness_session";
  const scopeVal = "paneKey" in identity ? identity.paneKey : identity.harnessSession;
  const starts = db.prepare(
    `SELECT id, ts, json_extract(payload, '$.agent_id') AS agentId FROM workflow_events
     WHERE project = ? AND ${scopeCol} = ? AND type = 'subagent-started'`,
  ).all(project, scopeVal) as { id: number; ts: string; agentId: string | null }[];
  let count = 0;
  for (const s of starts) {
    if (!s.agentId) continue;
    const stopped = db.prepare(
      `SELECT 1 FROM workflow_events WHERE project = ? AND ${scopeCol} = ? AND type = 'subagent-stopped'
       AND json_extract(payload, '$.agent_id') = ? AND id > ? LIMIT 1`,
    ).get(project, scopeVal, s.agentId, s.id);
    if (stopped) continue;
    const ageMs = nowMs - Date.parse(s.ts);
    if (Number.isFinite(ageMs) && ageMs > SUBAGENT_ORPHAN_MS) continue; // strictly over 30 min: orphan
    count += 1;
  }
  return count;
}

// Every unfinished run-started row for the project (same WHERE both activeRunFor and
// activeRunsFor read): no run-finished row yet for that run_id, and no NEWER run-started row
// for that same run_id (duplicate starts keep only the latest). Newest start id first.
function unfinishedRunStartedRows(db: Database.Database, project: string): RawRow[] {
  return db.prepare(
    `SELECT s.* FROM workflow_events s
     WHERE s.project = ? AND s.type = 'run-started'
       AND s.emitter != 'jaxos' AND s.run_id IS NOT NULL
       AND NOT EXISTS (
         SELECT 1 FROM workflow_events f
         WHERE f.project = s.project AND f.run_id = s.run_id
           AND f.type = 'run-finished'
       )
       AND NOT EXISTS (
         SELECT 1 FROM workflow_events newer
         WHERE newer.project = s.project AND newer.run_id = s.run_id
           AND newer.type = 'run-started' AND newer.emitter != 'jaxos'
           AND newer.id > s.id
       )
     ORDER BY s.id DESC`,
  ).all(project) as RawRow[];
}

// The kind/runtime/target/startedAt fields both activeRunFor and activeRunsFor project off a
// run-started row's own payload — factored out so the two never drift (activeRunsFor Task 1).
function activeRunFields(row: WorkflowEventRow): {
  runId: string; startedAt: string | null; kind: HubActiveRun["kind"]; runtime: HubActiveRun["runtime"]; target: string | null;
} {
  const p = projectPayload("run-started", row.payload);
  const kind = typeof p.kind === "string" &&
    (RUN_KINDS as readonly string[]).includes(p.kind)
    ? p.kind as HubActiveRun["kind"] : null;
  const runtime = typeof p.runtime === "string" &&
    (RUNTIMES as readonly string[]).includes(p.runtime)
    ? p.runtime as HubActiveRun["runtime"] : null;
  const rawTarget = typeof p.target === "string" && p.target.length > 0 &&
    p.target.length <= LIMITS.target && !/[\u0000-\u001f\u007f]/.test(p.target)
    ? p.target : null;
  const target = !kind || !rawTarget ? null
    : kind === "spec" || kind === "plan"
      ? rawTarget.split("/").at(-1) || null : rawTarget;
  const startedAt = Number.isFinite(Date.parse(row.ts)) ? row.ts : null;
  return { runId: row.run_id as string, startedAt, kind, runtime, target };
}

function activeRunFor(db: Database.Database, project: string, readChildLogTail: ChildLogTailReader | null = null): HubActiveRun | null {
  const rows = unfinishedRunStartedRows(db, project);
  if (rows.length === 0) return null;
  const row = rowOf(rows[0]);
  const fields = activeRunFields(row);
  // spec §8, Decision 10/11, cold review F1: the CONTROL repo comes from THIS run's own
  // run-started payload — never a worktree formula. Best-effort only: a missing repo, a
  // non-build kind, or no injected reader all degrade to null.
  const p = projectPayload("run-started", row.payload);
  const rawRepo = typeof p.repo === "string" && p.repo.length > 0 ? p.repo : null;
  let lastStep: string | null = null;
  if (readChildLogTail && fields.kind === "build" && rawRepo) {
    const logPath = childLogPath(rawRepo, row.run_id as string); // null on invalid repo/run_id (round 2 F1)
    const tail = logPath ? readChildLogTail(logPath) : null;
    if (tail !== null) lastStep = lastStreamTextLine(redactSecrets(tail)); // redact BEFORE parsing (§4.4)
  }
  return { ...fields, count: rows.length, lastStep };
}

// Every in-flight run as its own line (MOA session-list), not just the newest — the Sessions
// card needs to show several parallel headless runs (e.g. concurrent Codex reviews) at once.
// Same rows as activeRunFor (unfinished, latest start per run_id), no LIMIT, no child-log read.
export function activeRunsFor(db: Database.Database, project: string): HubActiveRunSummary[] {
  return unfinishedRunStartedRows(db, project).map((raw) => {
    const row = rowOf(raw);
    return { ...activeRunFields(row), role: row.role };
  });
}

// Mirrors activeRunFor's own query shape (§12.4): the most recent run-finished row for the
// project with no LATER run-started row — hidden the moment a newer run starts, even before
// that new run has its own run-finished row (D4). Joined against the matching run-started
// row (same run_id) for kind/runtime/target, the same trimming activeRunFor already applies.
export function lastRunFor(db: Database.Database, project: string, relToRepos: RelToRepos | null = null): HubLastRun | null {
  const rows = db.prepare(
    `SELECT f.* FROM workflow_events f
     WHERE f.project = ? AND f.type = 'run-finished' AND f.run_id IS NOT NULL
       AND NOT EXISTS (
         SELECT 1 FROM workflow_events s
         WHERE s.project = f.project AND s.type = 'run-started' AND s.emitter != 'jaxos'
           AND s.id > f.id
       )
     ORDER BY f.id DESC LIMIT 1`,
  ).all(project) as RawRow[];
  if (rows.length === 0) return null;
  const row = rowOf(rows[0]);
  const p = projectPayload("run-finished", row.payload);
  const role = row.role;

  const startedRow = db
    .prepare(
      `SELECT payload FROM workflow_events
       WHERE project = ? AND run_id = ? AND type = 'run-started' AND emitter != 'jaxos'
       ORDER BY id DESC LIMIT 1`,
    )
    .get(project, row.run_id) as { payload: string | null } | undefined;
  const sp = startedRow ? projectPayload("run-started", parseStoredPayload(startedRow.payload)) : {};

  const kind = typeof sp.kind === "string" && (RUN_KINDS as readonly string[]).includes(sp.kind)
    ? sp.kind as HubLastRun["kind"] : null;
  const runtime = typeof sp.runtime === "string" && (RUNTIMES as readonly string[]).includes(sp.runtime)
    ? sp.runtime as HubLastRun["runtime"] : null;
  const rawTarget = typeof sp.target === "string" && sp.target.length > 0 &&
    sp.target.length <= LIMITS.target && !/[\x00-\x1f\x7f]/.test(sp.target)
    ? sp.target : null;
  const target = !kind || !rawTarget ? null
    : kind === "spec" || kind === "plan"
      ? rawTarget.split("/").at(-1) || null : rawTarget;

  const contractStatus = (typeof p.contract_status === "string" &&
    (CONTRACT_STATUSES as readonly string[]).includes(p.contract_status)
    ? p.contract_status : "ok") as ContractStatus;
  const stage = typeof p.stage === "string" && (STAGES as readonly string[]).includes(p.stage)
    ? p.stage as Stage : null;
  const diagnostic = typeof p.diagnostic === "string" ? p.diagnostic : null;
  // A reviewer verdict counts only on an ok report (spec §7): never surface a label parsed
  // out of a missing/invalid/interrupted row, whatever the payload carries.
  const outcome = role === "reviewer"
    ? (contractStatus === "ok" && typeof p.verdict === "string" ? p.verdict : null)
    : (typeof p.result === "string" ? p.result : null);
  const headSha = role === "builder" && typeof p.head_sha === "string" ? p.head_sha : null;
  const finishedAt = Number.isFinite(Date.parse(row.ts)) ? row.ts : null;

  // Phase 2 (Decision 9): the recorded --verify/--build, builder + kind build only. Same inline
  // defensive bound lastRunFor's other fields already apply — a stored row can predate ingress.
  const command = (v: unknown, cap: number): string | null =>
    typeof v === "string" && v.length > 0 && v.length <= cap && !/[\x00-\x1f\x7f]/.test(v) ? v : null;
  const isBuild = role === "builder" && kind === "build";
  const verifyCommand = isBuild ? command(sp.verify, LIMITS.verify) : null;
  const buildCommand = isBuild ? command(sp.build, LIMITS.build) : null;

  // Round-3 F1 / round-1 plan review F4: same 0..999 inclusive-integer bound as the
  // ingress validator (FINDINGS_COUNT_MAX), applied defensively to whatever is actually
  // stored -- AND gated on role/contractStatus, mirroring the outcome/verdict gate just
  // above, so a legacy builder row or an invalid-reviewer row can never project findings
  // even when the stored payload happens to carry a well-shaped one (e.g. a raw
  // insertEvent bypassing ingress in a test).
  const rawFindings = p.findings;
  const boundedFindingsCount = (v: unknown): boolean => Number.isInteger(v) && (v as number) >= 0 && (v as number) <= 999;
  const findingsValid = role === "reviewer" && contractStatus === "ok"
    && typeof rawFindings === "object" && rawFindings !== null && !Array.isArray(rawFindings)
    && boundedFindingsCount((rawFindings as Record<string, unknown>).high)
    && boundedFindingsCount((rawFindings as Record<string, unknown>).medium)
    && boundedFindingsCount((rawFindings as Record<string, unknown>).low);
  const findings = findingsValid ? (rawFindings as { high: number; medium: number; low: number }) : null;

  // spec §11, Decision 17/18, cold review F2/F3: runtimeModel/profileName/targetRel read the RAW
  // run-started payload (`sp`); reportRel reads the RAW run-finished payload (`p`) — report_path
  // validates only there (workflow-events.ts), never on run-started. Both *Rel fields degrade to
  // null with no injected relToRepos, an unresolved path, or (targetRel) a non-spec/plan kind. F2:
  // relToRepos is PROJECT-root relative, always called with THIS function's `project` param, never
  // `sp.repo`/`p.repo` (control-repo path, can be a worktree — `project` is the row's own field).
  const runtimeModel = typeof sp.model === "string" && sp.model.length > 0 ? sp.model : null;
  const profileName = role === "builder" && typeof sp.requested_profile === "string" && sp.requested_profile.length > 0
    ? sp.requested_profile : null;
  const rawReportPath = typeof p.report_path === "string" && p.report_path.length > 0 ? p.report_path : null;
  const reportRel = rawReportPath && relToRepos ? relToRepos(project, rawReportPath) : null;
  // F3: sp.target is a non-empty bounded string only, never required absolute (e.g. the fixture's
  // own ".local/docs/specs/x-spec.md"). Normalize against the run's OWN project root FIRST — a
  // relative target must never resolve against the process cwd — only THEN call relToRepos.
  const rawTargetForRel = typeof sp.target === "string" && sp.target.length > 0 ? sp.target : null;
  const absTargetForRel = rawTargetForRel === null ? null : isAbsolute(rawTargetForRel) ? rawTargetForRel : join(REPOS_ROOT, project, rawTargetForRel);
  const targetRel = (kind === "spec" || kind === "plan") && absTargetForRel && relToRepos ? relToRepos(project, absTargetForRel) : null;

  return { runId: row.run_id as string, role, kind, runtime, target, finishedAt, outcome, contractStatus, stage, diagnostic, headSha, findings, runtimeModel, profileName, reportRel, targetRel, verifyCommand, buildCommand };
}

export function getWorktreeLedgerRows(db: Database.Database, project: string): LedgerRow[] {
  const rows = db.prepare(
    `SELECT s.id AS id, s.run_id AS runId, s.payload AS startedPayload, s.ts AS startedAt,
            f.payload AS finishedPayload, f.ts AS finishedAt
     FROM workflow_events s
     LEFT JOIN workflow_events f ON f.run_id = s.run_id AND f.type = 'run-finished'
     WHERE s.project = ? AND s.type = 'run-started' AND s.role = 'builder'
     ORDER BY s.id`,
  ).all(project) as { id: number; runId: string; startedPayload: string | null; startedAt: string; finishedPayload: string | null; finishedAt: string | null }[];
  return rows.map((r) => {
    const sp = parseStoredPayload(r.startedPayload);
    return { id: r.id, runId: r.runId, kind: "build" as const, branch: String(sp.target ?? ""), startedAt: r.startedAt,
      status: r.finishedPayload ? ("finished" as const) : ("running" as const), finishedAt: r.finishedAt };
  });
}

type LoopRunRow = { id: number; runId: string; ts: string; kind: string; phase: string; branch: string; builderRunId: string | null };

function normalizeLoopPhase(s: string): string {
  return s.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function loopRunRows(db: Database.Database, project: string): LoopRunRow[] {
  const rows = db.prepare(
    `SELECT id, run_id, ts, payload FROM workflow_events
     WHERE project = ? AND type = 'run-started' AND role IN ('builder', 'reviewer')
     ORDER BY ts, id`, // F5 (round 4): event timestamp, `id` only breaks an exact tie --
                        // insertion order alone can diverge from `ts` on a backfilled row.
  ).all(project) as { id: number; run_id: string; ts: string; payload: string | null }[];
  return rows
    .map((r) => {
      const p = parseStoredPayload(r.payload);
      return {
        id: r.id, runId: r.run_id, ts: r.ts, kind: String(p.kind ?? ""), phase: String(p.phase ?? ""),
        branch: String(p.target ?? ""), builderRunId: typeof p.builder_run_id === "string" ? p.builder_run_id : null,
      };
    })
    .filter((r) => ["spec", "plan", "diff", "build"].includes(r.kind));
}

function loopRunFinished(db: Database.Database, runId: string): { contractStatus: string; verdict: string | null } | null {
  const row = db.prepare(`SELECT payload FROM workflow_events WHERE run_id = ? AND type = 'run-finished' LIMIT 1`).get(runId) as
    { payload: string | null } | undefined;
  if (!row) return null;
  const p = parseStoredPayload(row.payload);
  return { contractStatus: String(p.contract_status ?? ""), verdict: typeof p.verdict === "string" ? p.verdict : null };
}

export type LoopSummary = {
  specCount: number; planCount: number; buildCount: number; diffCount: number;
  rounds: number; approved: boolean; wallDays: number | null;
};

// Auto-derived from the latest run-started row's ANCHOR BUILD (L1/L6, round 3 F4) --
// never the latest row's own raw phase (a diff's longer suffix is never a useful anchor).
// F4 (round 4): ports the CLI's own full anchor/fallback grouping (`cmd_loop`,
// `_anchor_builds_for_prefix`/`_direct_phase_match`, `scripts/jaxflow.py:5285-5440`) --
// once a prefix is resolved, the WHOLE matching family is counted (every build whose own
// normalized phase contains it -- a `--resume` attempt included -- and, with no anchor
// build yet, every spec/plan row directly matching the seed's own phase), never just the
// one row the seed happened to point at.
export function getLoopSummary(db: Database.Database, project: string): LoopSummary | null {
  const rows = loopRunRows(db, project);
  if (rows.length === 0) return null;
  const seed = rows[rows.length - 1];
  const builds = rows.filter((r) => r.kind === "build");

  // The seed's own single anchor build (L1's per-row rule), before the full family (F4)
  // is pulled in below.
  const seedAnchor: LoopRunRow | null = seed.kind === "build" ? seed
    : seed.kind === "diff"
      ? (builds.find((b) => b.runId === seed.builderRunId)
         ?? [...builds].reverse().find((b) => b.branch === seed.branch) ?? null)
      // F4 (round 5): a trailing spec/plan review dispatched AFTER a build already
      // exists must still resolve to that build (same containment join as the CLI's
      // `_anchor_builds_for_prefix`, scripts/jaxflow.py:5289 -- a spec/plan's own
      // normalized phase is naturally a substring of the real build's phase). Without
      // this, seedAnchor stayed null and the no-build fallback below dropped the
      // whole existing build/diff/merge family.
      : [...builds].reverse().find((b) => normalizeLoopPhase(b.phase).includes(normalizeLoopPhase(seed.phase))) ?? null;

  let anchors: LoopRunRow[];
  let specs: LoopRunRow[];
  let plans: LoopRunRow[];
  if (seedAnchor) {
    const prefixNorm = normalizeLoopPhase(seedAnchor.phase);
    anchors = builds.filter((b) => normalizeLoopPhase(b.phase).includes(prefixNorm));
    const buildPhasesNorm = anchors.map((b) => normalizeLoopPhase(b.phase));
    specs = rows.filter((r) => r.kind === "spec" && buildPhasesNorm.some((bp) => bp.startsWith(normalizeLoopPhase(r.phase))));
    plans = rows.filter((r) => r.kind === "plan" && buildPhasesNorm.some((bp) => bp.startsWith(normalizeLoopPhase(r.phase))));
  } else {
    // L1 fallback: no build dispatched yet -- the seed (a spec/plan review) anchors
    // itself directly; every spec/plan row whose own normalized phase CONTAINS the
    // seed's joins the same family.
    const prefixNorm = normalizeLoopPhase(seed.phase);
    anchors = [];
    specs = rows.filter((r) => r.kind === "spec" && normalizeLoopPhase(r.phase).includes(prefixNorm));
    plans = rows.filter((r) => r.kind === "plan" && normalizeLoopPhase(r.phase).includes(prefixNorm));
  }

  const anchorRunIds = new Set(anchors.map((b) => b.runId));
  const anchorBranches = [...new Set(anchors.map((b) => b.branch))];
  const diffs = rows.filter((r) => r.kind === "diff" &&
    (r.builderRunId !== null ? anchorRunIds.has(r.builderRunId) : anchorBranches.includes(r.branch)));

  let rounds = 0;
  let approved = false;
  for (const d of diffs) {
    rounds += 1;
    const finished = loopRunFinished(db, d.runId);
    if (finished?.contractStatus === "ok" && (finished.verdict === "approve" || finished.verdict === "approve-with-changes")) {
      approved = true;
      break;
    }
  }

  let wallDays: number | null = null;
  if (anchors.length > 0) {
    const merges = db.prepare(
      `SELECT ts, payload FROM workflow_events WHERE project = ? AND type = 'merge-approved' ORDER BY ts`,
    ).all(project) as { ts: string; payload: string | null }[];
    // F4 (round 2, MEDIUM): windowed by TIMESTAMPS per branch, matching Python's
    // `_wall_time_days` -- the window's end is a LATER build dispatched on the same
    // branch (any build, not only an anchor), if any.
    // F5 (round 4): that later build must be the EARLIEST-ts one, found by explicit
    // min-reduce -- an array `.find()` only happens to return the earliest match when
    // the array is itself ts-ordered, which insertion order alone does not guarantee.
    let mergeTs: string | null = null;
    for (const branch of anchorBranches) {
      const windowStart = anchors.filter((b) => b.branch === branch).reduce((a, b) => (a.ts < b.ts ? a : b)).ts;
      const laterBuildOnBranch = builds
        .filter((b) => b.branch === branch && b.ts > windowStart)
        .reduce<LoopRunRow | null>((min, b) => (min === null || b.ts < min.ts ? b : min), null);
      for (const m of merges) {
        if (String(parseStoredPayload(m.payload).branch ?? "") !== branch) continue;
        if (m.ts < windowStart) continue;
        if (laterBuildOnBranch && m.ts >= laterBuildOnBranch.ts) continue;
        if (mergeTs === null || m.ts < mergeTs) mergeTs = m.ts;
        break; // merges is ts-ordered ascending -- first in-window match is earliest
      }
    }
    if (mergeTs !== null) {
      const startRows = specs.length ? specs : plans.length ? plans : anchors;
      const startTs = startRows.reduce((a, b) => (a.ts < b.ts ? a : b)).ts;
      // F1 (diff review 79cbcce4, MEDIUM): a trailing spec/plan review dispatched AFTER
      // the merge (e.g. a re-review) must not push startTs past mergeTs -- that would
      // render as a negative wall time. null instead.
      if (startTs <= mergeTs) {
        wallDays = (Date.parse(mergeTs) - Date.parse(startTs)) / 86_400_000;
      }
    }
  }
  return { specCount: specs.length, planCount: plans.length, buildCount: anchors.length, diffCount: diffs.length, rounds, approved, wallDays };
}

// ---- Native Codex sessions (MOA-469 §2) -------------------------------------------------------

// Last codex-stop capsule for one exact session (harness_session), project-scoped. Unlike the
// pane variant, this never consults pane_key — the thread UUID is the identity.
function lastCodexCapsule(db: Database.Database, project: string, threadId: string): Capsule | null {
  const row = db
    .prepare(
      `SELECT id, ts, payload FROM workflow_events
       WHERE project = ? AND harness_session = ? AND type = 'turn-stopped' AND emitter = 'codex-stop'
       ORDER BY id DESC LIMIT 1`,
    )
    .get(project, threadId) as { id: number; ts: string; payload: string | null } | undefined;
  if (!row || !row.payload || isAnswered(db, row.id)) return null;
  try {
    const p = JSON.parse(row.payload) as { capsule_status?: string; capsule_minutes?: number; merge_ask?: number; message_tail?: string };
    if (typeof p.capsule_status !== "string" || !p.capsule_status) return null;
    return {
      status: p.capsule_status,
      minutes: typeof p.capsule_minutes === "number" ? p.capsule_minutes : null,
      declaredAt: row.ts,
      eventId: row.id,
      // D2: the same payload keys lastCapsule reads, and the same deriveQuestion helper — a
      // codex-stop row now reaches Jev (D1), so its classified merge_ask/message_tail are real
      // signal here too. No rung-6 pass exists on this path, only the poller's.
      mergeAsk: typeof p.merge_ask === "number" ? p.merge_ask : null,
      question: deriveQuestion(p.message_tail),
      answerable: true, // D4: no proactive Codex liveness check — a dead session just drops out
      // of card.sessions entirely (collectCodexSnapshot); nothing left here to disable.
    };
  } catch {
    return null;
  }
}

// Current-session correlation: a codex-permission attention with no later turn-started in THIS
// session, and a caller=codex run with no finish for THIS session. Legacy missing-caller runs are
// never proof (spec §3.3).
function codexCurrentCorrelation(
  db: Database.Database,
  project: string,
  threadId: string,
): { pendingAttention: boolean; runInFlight: boolean } {
  const pendingAttention = db
    .prepare(
      `SELECT 1 FROM workflow_events a
       WHERE a.project = ? AND a.harness_session = ? AND a.type = 'attention-needed' AND a.emitter = 'codex-permission'
         AND NOT EXISTS (
           SELECT 1 FROM workflow_events t
           WHERE t.project = ? AND t.harness_session = ? AND t.type = 'turn-started' AND t.id > a.id)
       LIMIT 1`,
    )
    .get(project, threadId, project, threadId);
  const runInFlight = db
    .prepare(
      `SELECT 1 FROM workflow_events s
       WHERE s.project = ? AND s.type = 'run-started' AND s.harness_session = ?
         AND json_extract(s.payload, '$.caller') = 'codex'
         AND NOT EXISTS (
           SELECT 1 FROM workflow_events f WHERE f.type = 'run-finished' AND f.run_id = s.run_id)
       LIMIT 1`,
    )
    .get(project, threadId);
  return { pendingAttention: pendingAttention !== undefined, runInFlight: runInFlight !== undefined };
}

function codexSessionFrom(
  db: Database.Database,
  project: string,
  threadId: string,
  thread: CodexThread | null,
  lastEventTs: string,
  sourceOk: boolean,
): Omit<HubCodexSession, "subagentCount"> {
  const capsule = lastCodexCapsule(db, project, threadId);
  const { pendingAttention, runInFlight } = codexCurrentCorrelation(db, project, threadId);
  if (!thread) {
    // A stored thread with no live runtime row is NOT active — a stale turn-started alone is not
    // liveness (spec §4). `working`/`attention` stay false; the row is visible as unavailable.
    return {
      threadId, status: "unknown", activeFlags: [], sourceOk,
      working: false, attention: false, pendingAttention, runInFlight, capsule, lastEventTs,
    };
  }
  const attention = thread.status === "active" && thread.activeFlags.some(
    (f) => f === "waitingOnApproval" || f === "waitingOnUserInput",
  );
  const working = thread.status === "active" && !attention;
  return {
    threadId, status: thread.status, activeFlags: thread.activeFlags, sourceOk,
    working, attention, pendingAttention, runInFlight, capsule, lastEventTs,
  };
}

// Native session membership for one project: distinct validated Codex harness sessions from its
// own events, each joined against the live snapshot. sourceOk is false when the snapshot was not
// available — the caller surfaces that as a warning, never as an empty successful hub.
export function getCodexSessions(
  db: Database.Database,
  project: string,
  snapshot: CodexSnapshot | null,
  nowMs: number = Date.now(),
): { sessions: HubCodexSession[]; sourceOk: boolean; untracked: boolean } {
  const sourceOk = snapshot?.ok === true;
  const byId = new Map<string, CodexThread>();
  if (snapshot?.ok) {
    for (const t of snapshot.threads) byId.set(t.threadId, t);
  }
  const rows = db
    .prepare(
      `SELECT harness_session AS session, MAX(ts) AS lastTs
       FROM workflow_events
       WHERE project = ? AND emitter IN ('codex-stop','codex-permission','codex-userprompt')
         AND harness_session IS NOT NULL
       GROUP BY harness_session ORDER BY lastTs DESC`,
    )
    .all(project) as { session: string; lastTs: string }[];
  const sessions = rows.map((r) => ({
    ...codexSessionFrom(db, project, r.session, byId.get(r.session) ?? null, r.lastTs, sourceOk),
    subagentCount: openSubagentCount(db, project, { harnessSession: r.session }, nowMs),
  }));
  // Missing-telemetry warning only: a loaded thread whose CANONICAL owner is a KNOWN scanned
  // project but has produced no validated events here must never become fabricated work/history —
  // it is only evidence that telemetry may be missing. Warn even when another root in the SAME
  // project already has events (C3): ownership comes from the collector's Git resolution, never a
  // basename guess, and is not gated on an empty session list.
  let untracked = false;
  if (sourceOk) {
    const tracked = new Set(rows.map((r) => r.session));
    untracked = (snapshot as Extract<CodexSnapshot, { ok: true }>).threads.some(
      (t) => t.owner === project && !tracked.has(t.threadId),
    );
  }
  return { sessions, sourceOk, untracked };
}


export function getHubData(
  db: Database.Database,
  live: LiveSnapshot,
  statusMtimes: Record<string, string | undefined>,
  codex: CodexSnapshot | null = null,
  now: number = Date.now(),
  tmuxSourceOk = true,
  readChildLogTail: ChildLogTailReader | null = null,
  relToRepos: RelToRepos | null = null,
  codexAbsent = false,
): HubEnvelopeData {
  const { projects, historicalTruncated } = getHubMembership(db, live, now);
  const codexSource: HubCodexSource =
    codexAbsent ? "absent" :
    codex === null ? "failed" : codex.ok ? (codex.truncated ? "truncated" : "ok") : "failed";
  // Object.create(null): `project` is externally sourced (derived from repo dir names) — a plain
  // {} would let a project literally named "__proto__" silently change the object's prototype
  // instead of becoming an enumerable entry, dropping it from the JSON envelope (Finding 6).
  const byProject: Record<string, HubProject> = Object.create(null);
  for (const project of projects) {
    const panes = getProjectPanes(db, project, live, now);
    const pendingQuestion = panes.find((p) => p.pendingQuestion)?.pendingQuestion ?? null;
    const { sessions: codexSessions, untracked: codexUntracked } = getCodexSessions(db, project, codex, now);
    // hasOwnProperty guard: `statusMtimes` is caller-built from external project dir names too
    // (same Finding 6 hazard) — a plain `statusMtimes["__proto__"]` read returns Object.prototype
    // itself (truthy, not a string) instead of undefined, which would crash countFreshEvents below.
    const mtime = Object.prototype.hasOwnProperty.call(statusMtimes, project) ? statusMtimes[project] : undefined;
    const freshnessCount = mtime ? countFreshEvents(db, project, mtime) : 0;
    const newestRow = db
      .prepare("SELECT MAX(ts) AS ts FROM workflow_events WHERE project = ? AND emitter != 'jaxos'")
      .get(project) as { ts: string | null };
    byProject[project] = {
      panes,
      codexSessions,
      codexUntracked,
      pendingQuestion,
      freshnessCount,
      newestEventTs: newestRow.ts,
      timeline: getProjectTimeline(db, project, 20),
      activeRun: activeRunFor(db, project, readChildLogTail),
      activeRuns: activeRunsFor(db, project),
      lastRun: lastRunFor(db, project, relToRepos),
      loopSummary: getLoopSummary(db, project),
      heartbeat: heartbeatFor(db, project, now),
      lastAction: lastActionFor(db, project, now),
    };
  }
  // spec §10, round 3 F1: a STANDALONE scan, no membership filter — a pinned/hidden project with
  // zero workflow_events rows and no live pane still gets its row, independent of `byProject`.
  const prefs = getProjectPrefs(db);
  const settings = { archiveAfterDays: getArchiveAfterDays(db) };
  return { byProject, historicalTruncated, codexSource, tmuxSource: tmuxSourceOk ? "ok" : "failed", prefs, settings };
}
