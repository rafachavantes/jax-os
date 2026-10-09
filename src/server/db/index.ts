import { mkdirSync } from "node:fs";
import { join } from "node:path";
import Database from "better-sqlite3";
import { runImportIfNeeded } from "./import";
import { abandonInjectingAnswers } from "./workflows";
import { readSeedSources } from "../collectors/rules";
import { seedRuleDocsIfNeeded } from "./rules";
import { jaxosHome } from "../env";

// Ordered migrations; PRAGMA user_version tracks how many have run. Phase B
// appended the workflow tables the same way. Each runs in its own
// transaction and bumps the version — a failed migration leaves the db at
// the last good version.
export const MIGRATIONS: readonly string[] = [
  `CREATE TABLE mutations (
    id      INTEGER PRIMARY KEY,
    ts      TEXT NOT NULL,
    kind    TEXT NOT NULL,
    ok      INTEGER,
    error   TEXT,
    payload TEXT NOT NULL
  );
  CREATE INDEX mutations_ts ON mutations (ts);
  CREATE INDEX mutations_kind ON mutations (kind);`,
  `CREATE TABLE metrics (
    id            INTEGER PRIMARY KEY,
    ts            TEXT NOT NULL,
    cpu_pct       REAL,
    mem_used_mb   INTEGER,
    mem_total_mb  INTEGER,
    swap_used_mb  INTEGER,
    disk_used_gb  REAL,
    disk_total_gb REAL,
    load1         REAL,
    load5         REAL,
    load15        REAL,
    uptime_s      INTEGER,
    services      TEXT NOT NULL
  );
  CREATE INDEX metrics_ts ON metrics (ts);
  CREATE TABLE metrics_hourly (
    hour              TEXT PRIMARY KEY,
    cpu_pct_avg       REAL,
    cpu_pct_max       REAL,
    mem_used_mb_avg   INTEGER,
    mem_used_mb_max   INTEGER,
    load5_max         REAL,
    disk_used_gb_last REAL,
    samples           INTEGER NOT NULL
  );`,
  `CREATE TABLE workflow_events (
    id           INTEGER PRIMARY KEY,
    ts           TEXT NOT NULL,
    run_id       TEXT,
    project      TEXT NOT NULL,
    role         TEXT,
    type         TEXT NOT NULL,
    source       TEXT NOT NULL,
    emitter      TEXT NOT NULL,
    payload      TEXT,
    delivery     TEXT NOT NULL,
    forwarded_at TEXT
  );
  CREATE INDEX workflow_events_ts ON workflow_events (ts);
  CREATE TABLE workflow_sessions (
    project       TEXT PRIMARY KEY,
    tmux_target   TEXT NOT NULL,
    registered_at TEXT NOT NULL
  );
  CREATE TABLE workflow_answer_claims (
    tool_use_id       TEXT PRIMARY KEY,
    question_event_id INTEGER NOT NULL,
    claimed_at        TEXT NOT NULL,
    status            TEXT NOT NULL,
    mutation_id       INTEGER NOT NULL
  );`,
  `-- (a) workflow_events gains pane + tmux_incarnation (spec §3.1a)
  ALTER TABLE workflow_events ADD COLUMN pane TEXT;
  ALTER TABLE workflow_events ADD COLUMN tmux_incarnation TEXT;
  CREATE INDEX workflow_events_pane ON workflow_events (pane, id);

  -- (b) workflow_sessions re-keyed: PK synthetic id, UNIQUE(pane, tmux_incarnation) (spec §3.1b)
  CREATE TABLE workflow_sessions_v2 (
    id                INTEGER PRIMARY KEY,
    session           TEXT NOT NULL,
    pane              TEXT NOT NULL,
    tmux_incarnation  TEXT NOT NULL,
    project           TEXT NOT NULL,
    role              TEXT NOT NULL CHECK (role IN ('lead', 'adhoc')),
    registered_at     TEXT NOT NULL,
    UNIQUE (pane, tmux_incarnation)
  );
  DROP TABLE workflow_sessions;
  ALTER TABLE workflow_sessions_v2 RENAME TO workflow_sessions;
  CREATE INDEX workflow_sessions_session ON workflow_sessions (session);
  CREATE INDEX workflow_sessions_project ON workflow_sessions (project);

  -- (c) workflow_answer_claims generalized: PK synthetic id, UNIQUE(source_event_id),
  -- deduped on copy to the most-recently-claimed row per question_event_id (finding 7)
  CREATE TABLE workflow_answer_claims_v2 (
    id              INTEGER PRIMARY KEY,
    source_event_id INTEGER NOT NULL,
    kind            TEXT NOT NULL,
    tool_use_id     TEXT,
    claimed_at      TEXT NOT NULL,
    status          TEXT NOT NULL,
    mutation_id     INTEGER NOT NULL,
    UNIQUE (source_event_id)
  );
  INSERT INTO workflow_answer_claims_v2 (source_event_id, kind, tool_use_id, claimed_at, status, mutation_id)
    SELECT c.question_event_id, 'structured', c.tool_use_id, c.claimed_at, c.status, c.mutation_id
    FROM workflow_answer_claims c
    WHERE c.rowid = (
      SELECT c2.rowid FROM workflow_answer_claims c2
      WHERE c2.question_event_id = c.question_event_id
      ORDER BY c2.claimed_at DESC, c2.rowid DESC
      LIMIT 1
    );
  DROP TABLE workflow_answer_claims;
  ALTER TABLE workflow_answer_claims_v2 RENAME TO workflow_answer_claims;

  -- (d) a pre-existing unanswered question has no pane; expire it, don't fabricate a claim (finding 23/27/33/42)
  UPDATE workflow_events
    SET delivery = 'local'
    WHERE type = 'question'
      AND NOT EXISTS (SELECT 1 FROM workflow_answer_claims c WHERE c.source_event_id = workflow_events.id)
      AND NOT EXISTS (
        SELECT 1 FROM workflow_events r
        WHERE r.type = 'question-resolved' AND r.project = workflow_events.project
          AND (CASE WHEN json_valid(r.payload) THEN json_extract(r.payload, '$.tool_use_id') END)
              = (CASE WHEN json_valid(workflow_events.payload) THEN json_extract(workflow_events.payload, '$.tool_use_id') END)
      );

  INSERT INTO mutations (ts, kind, ok, error, payload)
    SELECT strftime('%Y-%m-%dT%H:%M:%fZ','now'), 'workflow-question-expired', 1, NULL,
      json_object('question_event_id', e.id,
                   'reason', 'pre-migration question has no pane; cannot be answered post-migration, must be re-asked')
    FROM workflow_events e
    WHERE e.type = 'question'
      AND NOT EXISTS (SELECT 1 FROM workflow_answer_claims c WHERE c.source_event_id = e.id)
      AND NOT EXISTS (
        SELECT 1 FROM workflow_events r
        WHERE r.type = 'question-resolved' AND r.project = e.project
          AND (CASE WHEN json_valid(r.payload) THEN json_extract(r.payload, '$.tool_use_id') END)
              = (CASE WHEN json_valid(e.payload) THEN json_extract(e.payload, '$.tool_use_id') END)
      );

  -- (e) AFK switch singleton — no seed row; "absent" IS the off default
  CREATE TABLE workflow_afk (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    enabled    INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
  );`,
  `-- Spec B §4.1 (migration 5): a generated pane_key column replaces the bare "pane = ?" SQL
  -- predicate as the enforcement point for per-pane identity — see the recurrence ledger
  -- (.local/docs/plans/2026-08-27-spec-b-recurrence-ledger.md, R1: seven prior occurrences of
  -- this exact defect class). A pane id is reused after a tmux server restart; deriving pane_key
  -- from BOTH columns makes the correct predicate the SHORTEST one to write, which removes the
  -- incentive to write an unscoped query instead of policing it after the fact. NULL
  -- tmux_incarnation yields NULL pane_key (SQLite's || operator propagates NULL), so a row that
  -- never had an incarnation can never be matched by accident.
  ALTER TABLE workflow_events   ADD COLUMN pane_key TEXT
    GENERATED ALWAYS AS (pane || ':' || tmux_incarnation) VIRTUAL;
  ALTER TABLE workflow_sessions ADD COLUMN pane_key TEXT
    GENERATED ALWAYS AS (pane || ':' || tmux_incarnation) VIRTUAL;
  CREATE INDEX workflow_events_pane_key ON workflow_events (pane_key, id);`,
  `CREATE TABLE rule_docs (
    slot       TEXT PRIMARY KEY,
    content    TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );`,
  `-- Phase 4 (migration 7): the harness's own session id, carried by both Claude Code and Codex
  -- on every hook event. Rules 4 and 5 of the stop-state ladder pair a turn-stopped with the
  -- jaxflow run dispatched from the same session, so this is a lookup key, not payload data —
  -- a json_extract predicate here would scan the table on every turn.
  ALTER TABLE workflow_events ADD COLUMN harness_session TEXT;
  CREATE INDEX workflow_events_harness_session ON workflow_events (harness_session, id);`,
  `CREATE UNIQUE INDEX mutations_provider_credential_operation ON mutations (json_extract(payload, '$.operation_id'))
    WHERE kind = 'provider-credential' AND json_valid(payload) AND json_extract(payload, '$.operation_id') IS NOT NULL;
  CREATE UNIQUE INDEX mutations_provider_credential_active ON mutations (json_extract(payload, '$.provider_id'))
    WHERE kind = 'provider-credential' AND json_valid(payload)
      AND json_extract(payload, '$.provider_id') IS NOT NULL
      AND json_extract(payload, '$.stage') NOT IN ('done', 'not-applied');`,
  `-- Phase 3 (migration 9): manual hide/pin per project (spec §10, Decision 15). hidden_at is a
  -- TIMESTAMP, not a boolean — NULL = not hidden; a value is WHEN it was hidden, the only thing the
  -- un-hide-on-activity rule compares against, so toggling pinned can never disturb it (F6). The
  -- auto-hide threshold is a column on the EXISTING workflow_afk singleton, not a second settings
  -- table (F9). The (project, ts) index backs the per-minute heartbeat query (spec §6, Decision 5)
  -- so a 10s poll stays a range scan.
  CREATE TABLE project_prefs (
    project   TEXT PRIMARY KEY,
    pinned    INTEGER NOT NULL DEFAULT 0,
    hidden_at TEXT NULL
  );
  ALTER TABLE workflow_afk ADD COLUMN archive_after_days INTEGER NOT NULL DEFAULT 14;
  CREATE INDEX workflow_events_project_ts ON workflow_events (project, ts);`,
  `-- MOA-487 (migration 10): per-type Telegram-forward options (spec §5 Decision 1). One new
  -- column on the existing workflow_afk singleton — this literal MUST equal lib/workflow.ts's
  -- DEFAULT_FORWARD_TYPES exactly; index.test.ts's migration-10 test asserts it.
  ALTER TABLE workflow_afk ADD COLUMN forward_types TEXT NOT NULL DEFAULT '["question","attention-needed","mission-finished"]';`,
  `-- MOA-488 (migration 11): jaxflow mission tracking (spec §6). ONE table, no join --
  -- 'milestones' is a whole JSON array (Decision 1/2): at up to 12 items a join table buys
  -- nothing and forces mark <title> to run a second query. The partial unique index is the
  -- DB-level backstop for "exactly one active mission" (Decision 3) -- the route's own
  -- pre-check (Task 5) is the primary refusal path; this index only catches a race.
  CREATE TABLE missions (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    goal        TEXT NOT NULL,
    status_line TEXT NOT NULL,
    state       TEXT NOT NULL CHECK (state IN ('active', 'done', 'cancelled')),
    started_at  TEXT NOT NULL,
    ended_at    TEXT NULL,
    milestones  TEXT NOT NULL
  );
  CREATE UNIQUE INDEX missions_one_active ON missions (state) WHERE state = 'active';`,
];

export function migrate(db: Database.Database): void {
  let v = db.pragma("user_version", { simple: true }) as number;
  while (v < MIGRATIONS.length) {
    const sql = MIGRATIONS[v];
    const next = v + 1;
    db.transaction(() => {
      db.exec(sql);
      db.pragma(`user_version = ${next}`);
    })();
    v = next;
  }
}

// Open + configure + migrate. Path is injectable so tests run on ":memory:".
export function openDb(path: string): Database.Database {
  const db = new Database(path);
  db.pragma("journal_mode = WAL"); // two writer processes by SP2 (Next + metrics collector)
  db.pragma("busy_timeout = 5000");
  migrate(db);
  return db;
}

const DB_DIR = jaxosHome();
export const DB_PATH = join(DB_DIR, "jaxos.db");
const LOG_PATH = join(DB_DIR, "mutations.log");

// Next dev hot-reload re-evaluates modules; a module-level variable would
// leak one connection per reload. globalThis survives reloads.
const g = globalThis as typeof globalThis & { __jaxDb?: Database.Database };

export function getDb(): Database.Database {
  if (!g.__jaxDb) {
    mkdirSync(DB_DIR, { recursive: true });
    const db = openDb(DB_PATH);
    runImportIfNeeded(db, LOG_PATH);
    abandonInjectingAnswers(db);
    seedRuleDocsIfNeeded(db, readSeedSources);
    g.__jaxDb = db;
  }
  return g.__jaxDb;
}
