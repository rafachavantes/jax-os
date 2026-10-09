import { describe, expect, it } from "vitest";
import Database from "better-sqlite3";
import { migrate, MIGRATIONS, openDb } from "./index";
import { DEFAULT_FORWARD_TYPES } from "../../lib/workflow";

describe("db migrations", () => {
  it("brings a fresh db to user_version 3 with the mutations + metrics + workflow schema", () => {
    // ponytail/finding: this test predates migration 4 and asserts the v0 shape; openDb() now
    // always runs every migration, so it must apply only the first 3 (openV0Fixture, defined
    // below — hoisted) to keep exercising the shape it was written to check.
    const db = openV0Fixture();
    expect(db.pragma("user_version", { simple: true })).toBe(3);
    const cols = (db.prepare("PRAGMA table_info(mutations)").all() as { name: string }[]).map((c) => c.name);
    expect(cols).toEqual(["id", "ts", "kind", "ok", "error", "payload"]);
    const idx = (db.prepare("PRAGMA index_list(mutations)").all() as { name: string }[]).map((i) => i.name).sort();
    expect(idx).toEqual(["mutations_kind", "mutations_ts"]);
    const mcols = (db.prepare("PRAGMA table_info(metrics)").all() as { name: string }[]).map((c) => c.name);
    expect(mcols).toEqual([
      "id", "ts", "cpu_pct", "mem_used_mb", "mem_total_mb", "swap_used_mb",
      "disk_used_gb", "disk_total_gb", "load1", "load5", "load15", "uptime_s", "services",
    ]);
    const hcols = (db.prepare("PRAGMA table_info(metrics_hourly)").all() as { name: string }[]).map((c) => c.name);
    expect(hcols).toEqual([
      "hour", "cpu_pct_avg", "cpu_pct_max", "mem_used_mb_avg", "mem_used_mb_max",
      "load5_max", "disk_used_gb_last", "samples",
    ]);
    const midx = (db.prepare("PRAGMA index_list(metrics)").all() as { name: string }[]).map((i) => i.name);
    expect(midx).toContain("metrics_ts");
    const wcols = (db.prepare("PRAGMA table_info(workflow_events)").all() as { name: string }[]).map((c) => c.name);
    expect(wcols).toEqual([
      "id", "ts", "run_id", "project", "role", "type", "source", "emitter", "payload", "delivery", "forwarded_at",
    ]);
    const widx = (db.prepare("PRAGMA index_list(workflow_events)").all() as { name: string }[]).map((i) => i.name);
    expect(widx).toContain("workflow_events_ts");
    const scols = (db.prepare("PRAGMA table_info(workflow_sessions)").all() as { name: string; pk: number }[]);
    expect(scols.map((c) => c.name)).toEqual(["project", "tmux_target", "registered_at"]);
    expect(scols.find((c) => c.name === "project")?.pk).toBe(1);
    const ccols = (db.prepare("PRAGMA table_info(workflow_answer_claims)").all() as { name: string; pk: number }[]);
    expect(ccols.map((c) => c.name)).toEqual(["tool_use_id", "question_event_id", "claimed_at", "status", "mutation_id"]);
    expect(ccols.find((c) => c.name === "tool_use_id")?.pk).toBe(1);
    db.close();
  });

  it("re-running migrate is a no-op", () => {
    const db = new Database(":memory:");
    migrate(db);
    migrate(db); // would throw "table mutations already exists" if not guarded by user_version
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    db.close();
  });

  it("migration 4 brings the re-keyed workflow schema (openDb also runs migrations 5–6 — see the dedicated tests)", () => {
    const db = openDb(":memory:");
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    // Finding 1 (cold review round 3): these two exact-list assertions stay on table_info and stay
    // WITHOUT pane_key — table_info omits generated columns entirely (it never contained pane_key
    // to begin with, migration 5 or not), and the ordinary columns here are unchanged by migration
    // 5. Asserting pane_key's existence is the dedicated table_xinfo test above; duplicating it
    // here via table_info would just fail, since the pragma structurally cannot show it.
    const ecols = (db.prepare("PRAGMA table_info(workflow_events)").all() as { name: string }[]).map((c) => c.name);
    expect(ecols).toEqual([
      "id", "ts", "run_id", "project", "role", "type", "source", "emitter",
      "payload", "delivery", "forwarded_at", "pane", "tmux_incarnation", "harness_session",
    ]);
    const eidx = (db.prepare("PRAGMA index_list(workflow_events)").all() as { name: string }[]).map((i) => i.name);
    expect(eidx).toContain("workflow_events_pane");
    expect(eidx).toContain("workflow_events_harness_session");
    const scols = (db.prepare("PRAGMA table_info(workflow_sessions)").all() as { name: string; pk: number }[]);
    expect(scols.map((c) => c.name)).toEqual(["id", "session", "pane", "tmux_incarnation", "project", "role", "registered_at"]);
    expect(scols.find((c) => c.name === "id")?.pk).toBe(1);
    const sidx = (db.prepare("PRAGMA index_list(workflow_sessions)").all() as { name: string }[]).map((i) => i.name).sort();
    expect(sidx).toEqual(["sqlite_autoindex_workflow_sessions_1", "workflow_sessions_project", "workflow_sessions_session"]);
    const ccols = (db.prepare("PRAGMA table_info(workflow_answer_claims)").all() as { name: string; pk: number }[]);
    expect(ccols.map((c) => c.name)).toEqual(["id", "source_event_id", "kind", "tool_use_id", "claimed_at", "status", "mutation_id"]);
    expect(ccols.find((c) => c.name === "id")?.pk).toBe(1);
    const acols = (db.prepare("PRAGMA table_info(workflow_afk)").all() as { name: string; pk: number }[]);
    expect(acols.map((c) => c.name)).toEqual(["id", "enabled", "updated_at", "archive_after_days", "forward_types"]);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_afk").get()).toEqual({ n: 0 }); // no seed row — empty IS the off default
    db.close();
  });

  it("migration 5 adds generated pane_key columns, derived from pane + tmux_incarnation", () => {
    const db = openDb(":memory:");
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    // PRAGMA table_info OMITS generated columns entirely — verified on the real stack (SQLite
    // 3.53.2 via better-sqlite3): table_info(workflow_events) here still returns exactly
    // ["pane","tmux_incarnation"] as its last two names, with no pane_key at all. Only
    // table_xinfo surfaces a generated column, marked by hidden: 2 (GENERATED ALWAYS ... VIRTUAL).
    // This is why the exact-list assertions below stay on table_info unchanged, and pane_key gets
    // its own table_xinfo assertion here instead (cold review round 3, Finding 1).
    const exinfo = db.prepare("PRAGMA table_xinfo(workflow_events)").all() as { name: string; hidden: number }[];
    const eKeyCol = exinfo.find((c) => c.name === "pane_key");
    expect(eKeyCol).toBeDefined();
    expect(eKeyCol?.hidden).toBe(2);
    const eidx = (db.prepare("PRAGMA index_list(workflow_events)").all() as { name: string }[]).map((i) => i.name);
    expect(eidx).toContain("workflow_events_pane_key");
    expect(eidx).toContain("workflow_events_pane"); // migration 4's (pane, id) index survives — the discovery query still needs it
    const sxinfo = db.prepare("PRAGMA table_xinfo(workflow_sessions)").all() as { name: string; hidden: number }[];
    const sKeyCol = sxinfo.find((c) => c.name === "pane_key");
    expect(sKeyCol).toBeDefined();
    expect(sKeyCol?.hidden).toBe(2);

    // F3: pane_key is behaviorally derived from BOTH session identity fields, not just present.
    db.prepare(
      `INSERT INTO workflow_sessions (session, pane, tmux_incarnation, project, role, registered_at)
       VALUES ('jax-p1-lead', '%1', '100:1', 'p1', 'lead', '2026-08-28T00:00:00.000Z')`,
    ).run();
    const session1 = db.prepare("SELECT pane_key FROM workflow_sessions WHERE pane = '%1'").get() as { pane_key: string };
    expect(session1.pane_key).toBe("%1:100:1");

    db.prepare(
      `INSERT INTO workflow_sessions (session, pane, tmux_incarnation, project, role, registered_at)
       VALUES ('jax-p1-lead', '%1', '200:2', 'p1', 'lead', '2026-08-28T00:01:00.000Z')`,
    ).run();
    const session2 = db.prepare("SELECT pane_key FROM workflow_sessions WHERE tmux_incarnation = '200:2'").get() as { pane_key: string };
    expect(session2.pane_key).toBe("%1:200:2");
    expect(session2.pane_key).not.toBe(session1.pane_key); // a second incarnation of the same pane is a distinct key

    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery, pane, tmux_incarnation)
       VALUES ('2026-08-28T00:00:00.000Z', NULL, 'p1', 'lead', 'turn-started', 'deterministic', 'claude-userprompt', '{}', 'local', '%1', '100:1')`,
    ).run();
    const withIncarnation = db.prepare("SELECT pane_key FROM workflow_events WHERE pane = '%1'").get() as { pane_key: string };
    expect(withIncarnation.pane_key).toBe("%1:100:1");

    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery, pane, tmux_incarnation)
       VALUES ('2026-08-28T00:00:01.000Z', NULL, 'p1', 'builder', 'run-started', 'deterministic', 'wrapper', '{}', 'local', NULL, NULL)`,
    ).run();
    const noIncarnation = db.prepare("SELECT pane_key FROM workflow_events WHERE pane IS NULL").get() as { pane_key: string | null };
    expect(noIncarnation.pane_key).toBeNull(); // a row that never had an incarnation can never be matched by accident
    db.close();
  });

  it("migration 6 creates rule_docs with slot as PRIMARY KEY, empty until seeded", () => {
    const db = openDb(":memory:");
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    expect(db.prepare("SELECT COUNT(*) AS c FROM rule_docs").get()).toEqual({ c: 0 });
    db.prepare("INSERT INTO rule_docs (slot, content, updated_at) VALUES ('global', 'a', 't')").run();
    expect(() =>
      db.prepare("INSERT INTO rule_docs (slot, content, updated_at) VALUES ('global', 'b', 't')").run(),
    ).toThrow(); // PK violation — one row per slot
    db.close();
  });

  it("migration 9 adds project_prefs (hidden_at is a timestamp, not a boolean), archive_after_days on workflow_afk, and the (project, ts) heartbeat index", () => {
    const db = openDb(":memory:");
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    const pcols = db.prepare("PRAGMA table_info(project_prefs)").all() as { name: string; type: string; pk: number; dflt_value: string | null }[];
    expect(pcols.map((c) => c.name)).toEqual(["project", "pinned", "hidden_at"]);
    expect(pcols.find((c) => c.name === "project")?.pk).toBe(1);
    expect(pcols.find((c) => c.name === "hidden_at")?.type).toBe("TEXT"); // F6: a timestamp, never a boolean
    expect(pcols.find((c) => c.name === "pinned")?.dflt_value).toBe("0");
    const acols = db.prepare("PRAGMA table_info(workflow_afk)").all() as { name: string; dflt_value: string | null }[];
    expect(acols.find((c) => c.name === "archive_after_days")?.dflt_value).toBe("14");
    expect(db.prepare("SELECT COUNT(*) AS n FROM project_prefs").get()).toEqual({ n: 0 });
    // F9: ONE new table — no mission_settings.
    expect(db.prepare("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'mission_settings'").get()).toBeUndefined();
    const eidx = (db.prepare("PRAGMA index_list(workflow_events)").all() as { name: string }[]).map((i) => i.name);
    expect(eidx).toContain("workflow_events_project_ts");
    db.close();
  });
});

describe("migration 10 adds forward_types (MOA-487 §5 Decision 1)", () => {
  it("backfills the exact DEFAULT_FORWARD_TYPES JSON onto an existing row, preserves enabled/archive_after_days, reaches user_version 10", () => {
    const db = openV9Fixture();
    db.prepare(
      "INSERT INTO workflow_afk (id, enabled, updated_at, archive_after_days) VALUES (1, 1, '2026-09-19T00:00:00.000Z', 21)",
    ).run();
    migrate(db);
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    const row = db.prepare("SELECT enabled, archive_after_days, forward_types FROM workflow_afk WHERE id = 1").get() as
      { enabled: number; archive_after_days: number; forward_types: string };
    expect(row.enabled).toBe(1);
    expect(row.archive_after_days).toBe(21);
    expect(JSON.parse(row.forward_types)).toEqual(DEFAULT_FORWARD_TYPES);
    db.close();
  });

  it("a fresh install lands at user_version 10 with no seed row", () => {
    const db = openDb(":memory:");
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_afk").get()).toEqual({ n: 0 });
    db.close();
  });
});

// ---- migration-4 fixture helper: seed a v0-shape (user_version 3) db, then run migration 4 alone ----

function openV0Fixture(): Database.Database {
  const db = new Database(":memory:");
  db.pragma("journal_mode = WAL");
  for (let i = 0; i < 3; i += 1) {
    db.transaction(() => {
      db.exec(MIGRATIONS[i]);
      db.pragma(`user_version = ${i + 1}`);
    })();
  }
  return db;
}

// F2: a v3 fixture plus migration 4 applied manually, landing at user_version 4 — the actual
// shape openDb() must upgrade from in the field, distinct from openDb() running all 5 at once.
function openV4Fixture(): Database.Database {
  const db = openV0Fixture();
  db.transaction(() => {
    db.exec(MIGRATIONS[3]);
    db.pragma("user_version = 4");
  })();
  return db;
}

function openV9Fixture(): Database.Database {
  const db = new Database(":memory:");
  db.pragma("journal_mode = WAL");
  for (let i = 0; i < 9; i += 1) {
    db.transaction(() => { db.exec(MIGRATIONS[i]); db.pragma(`user_version = ${i + 1}`); })();
  }
  return db;
}

describe("migration 5 upgrade path from a real v4 db", () => {
  it("applies only migration 5, preserving existing rows and deriving pane_key", () => {
    const db = openV4Fixture();
    expect(db.pragma("user_version", { simple: true })).toBe(4);

    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery, pane, tmux_incarnation)
       VALUES ('2026-08-28T00:00:00.000Z', NULL, 'p1', 'lead', 'turn-started', 'deterministic', 'claude-userprompt', '{}', 'local', '%1', '100:1')`,
    ).run();
    db.prepare(
      `INSERT INTO workflow_sessions (session, pane, tmux_incarnation, project, role, registered_at)
       VALUES ('jax-p1-lead', '%1', '100:1', 'p1', 'lead', '2026-08-28T00:00:00.000Z')`,
    ).run();

    migrate(db);
    expect(db.pragma("user_version", { simple: true })).toBe(11);

    const ev = db.prepare("SELECT project, pane_key FROM workflow_events WHERE pane = '%1'").get() as {
      project: string;
      pane_key: string;
    };
    expect(ev.project).toBe("p1"); // pre-existing row preserved, not dropped by the migration
    expect(ev.pane_key).toBe("%1:100:1");
    const sess = db.prepare("SELECT session, pane_key FROM workflow_sessions WHERE pane = '%1'").get() as {
      session: string;
      pane_key: string;
    };
    expect(sess.session).toBe("jax-p1-lead");
    expect(sess.pane_key).toBe("%1:100:1");

    const eidx = (db.prepare("PRAGMA index_list(workflow_events)").all() as { name: string }[]).map((i) => i.name);
    expect(eidx).toContain("workflow_events_pane_key");
    db.close();
  });
});

describe("migration 4 fixtures", () => {
  it("clean v0-shape db migrates without loss: old session/claim rows dropped, not corrupted", () => {
    const db = openV0Fixture();
    db.prepare("INSERT INTO workflow_sessions (project, tmux_target, registered_at) VALUES (?, ?, ?)").run(
      "p1", "jax-p1-lead:1.1", "2026-08-01T00:00:00.000Z",
    );
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES (?, NULL, 'p1', 'lead', 'question', 'deterministic', 'claude-pretool', ?, 'delivered')`,
    ).run("2026-08-01T00:00:00.000Z", JSON.stringify({ tool_use_id: "tu_1", questions: [] }));
    const qid = db.prepare("SELECT id FROM workflow_events WHERE type='question'").get() as { id: number };
    db.prepare(
      `INSERT INTO workflow_answer_claims (tool_use_id, question_event_id, claimed_at, status, mutation_id)
       VALUES ('tu_1', ?, '2026-08-01T00:01:00.000Z', 'done', 1)`,
    ).run(qid.id);
    migrate(db);
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_sessions").get()).toEqual({ n: 0 }); // dropped, not migrated (§3.1b)
    const claim = db.prepare("SELECT * FROM workflow_answer_claims").get() as Record<string, unknown>;
    expect(claim.source_event_id).toBe(qid.id);
    expect(claim.kind).toBe("structured");
    const ev = db.prepare("SELECT pane, tmux_incarnation FROM workflow_events WHERE id = ?").get(qid.id);
    expect(ev).toEqual({ pane: null, tmux_incarnation: null });
    db.close();
  });

  it("malformed question-resolved payload does not abort the migration (cold review finding 42)", () => {
    const db = openV0Fixture();
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:00:00.000Z', NULL, 'p1', 'lead', 'question', 'deterministic', 'claude-pretool', ?, 'delivered')`,
    ).run(JSON.stringify({ tool_use_id: "tu_1", questions: [] }));
    const qid = db.prepare("SELECT id FROM workflow_events WHERE type='question'").get() as { id: number };
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:02:00.000Z', NULL, 'p1', 'lead', 'question-resolved', 'deterministic', 'claude-posttool', '{not-json}', 'local')`,
    ).run();
    // sibling: well-formed matching resolution stays unaffected (finding 33's already-fixed case)
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:00:00.000Z', NULL, 'p1', 'lead', 'question', 'deterministic', 'claude-pretool', ?, 'delivered')`,
    ).run(JSON.stringify({ tool_use_id: "tu_2", questions: [] }));
    const qid2 = db.prepare("SELECT id FROM workflow_events WHERE type='question' AND id != ?").get(qid.id) as { id: number };
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:02:00.000Z', NULL, 'p1', 'lead', 'question-resolved', 'deterministic', 'claude-posttool', ?, 'local')`,
    ).run(JSON.stringify({ tool_use_id: "tu_2" }));
    expect(() => migrate(db)).not.toThrow();
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    const row1 = db.prepare("SELECT delivery FROM workflow_events WHERE id = ?").get(qid.id) as { delivery: string };
    expect(row1.delivery).toBe("local"); // malformed resolution treated as non-matching → expired
    const audit1 = db.prepare(
      "SELECT COUNT(*) AS n FROM mutations WHERE kind='workflow-question-expired' AND json_extract(payload,'$.question_event_id') = ?",
    ).get(qid.id) as { n: number };
    expect(audit1.n).toBe(1);
    const row2 = db.prepare("SELECT delivery FROM workflow_events WHERE id = ?").get(qid2.id) as { delivery: string };
    expect(row2.delivery).toBe("delivered"); // well-formed resolution: untouched
    const audit2 = db.prepare(
      "SELECT COUNT(*) AS n FROM mutations WHERE kind='workflow-question-expired' AND json_extract(payload,'$.question_event_id') = ?",
    ).get(qid2.id) as { n: number };
    expect(audit2.n).toBe(0);
    db.close();
  });

  it("duplicate claim rows for one question_event_id collapse to the most-recently-claimed (cold review finding 7)", () => {
    const db = openV0Fixture();
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:00:00.000Z', NULL, 'p1', 'lead', 'question', 'deterministic', 'claude-pretool', ?, 'delivered')`,
    ).run(JSON.stringify({ tool_use_id: "tu_1", questions: [] }));
    const qid = db.prepare("SELECT id FROM workflow_events WHERE type='question'").get() as { id: number };
    db.prepare(
      `INSERT INTO workflow_answer_claims (tool_use_id, question_event_id, claimed_at, status, mutation_id)
       VALUES ('tu_1_abandoned', ?, '2026-08-01T00:01:00.000Z', 'abandoned', 1)`,
    ).run(qid.id);
    db.prepare(
      `INSERT INTO workflow_answer_claims (tool_use_id, question_event_id, claimed_at, status, mutation_id)
       VALUES ('tu_1_retry', ?, '2026-08-01T00:02:00.000Z', 'done', 2)`,
    ).run(qid.id);
    expect(() => migrate(db)).not.toThrow();
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    const rows = db.prepare("SELECT tool_use_id FROM workflow_answer_claims WHERE source_event_id = ?").all(qid.id);
    expect(rows).toEqual([{ tool_use_id: "tu_1_retry" }]); // later claimed_at wins

    // sibling: same claimed_at, tie-break by rowid (later-inserted wins)
    const db2 = openV0Fixture();
    db2.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:00:00.000Z', NULL, 'p1', 'lead', 'question', 'deterministic', 'claude-pretool', ?, 'delivered')`,
    ).run(JSON.stringify({ tool_use_id: "tu_2", questions: [] }));
    const qid2 = db2.prepare("SELECT id FROM workflow_events WHERE type='question'").get() as { id: number };
    db2.prepare(
      `INSERT INTO workflow_answer_claims (tool_use_id, question_event_id, claimed_at, status, mutation_id)
       VALUES ('tu_2_first', ?, '2026-08-01T00:01:00.000Z', 'abandoned', 1)`,
    ).run(qid2.id);
    db2.prepare(
      `INSERT INTO workflow_answer_claims (tool_use_id, question_event_id, claimed_at, status, mutation_id)
       VALUES ('tu_2_second', ?, '2026-08-01T00:01:00.000Z', 'done', 2)`,
    ).run(qid2.id);
    migrate(db2);
    const rows2 = db2.prepare("SELECT tool_use_id FROM workflow_answer_claims WHERE source_event_id = ?").all(qid2.id);
    expect(rows2).toEqual([{ tool_use_id: "tu_2_second" }]); // later rowid (insertion order) wins the tie
    db.close();
    db2.close();
  });

  it("a pre-existing unanswered question is expired-and-audited, never fabricated a claim (finding 23, extended by 27/33)", () => {
    const db = openV0Fixture();
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:00:00.000Z', NULL, 'Acme.AI', 'lead', 'question', 'deterministic', 'claude-pretool', ?, 'delivered')`,
    ).run(JSON.stringify({ tool_use_id: "tu_orphan", questions: [] }));
    const qid = db.prepare("SELECT id FROM workflow_events WHERE type='question'").get() as { id: number };
    migrate(db);
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    const ev = db.prepare("SELECT delivery, pane, tmux_incarnation FROM workflow_events WHERE id = ?").get(qid.id);
    expect(ev).toEqual({ delivery: "local", pane: null, tmux_incarnation: null }); // finding 27: delivery actually moved
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_answer_claims WHERE source_event_id = ?").get(qid.id)).toEqual({ n: 0 });
    const audit = db.prepare(
      "SELECT payload FROM mutations WHERE kind='workflow-question-expired' AND json_extract(payload,'$.question_event_id') = ?",
    ).get(qid.id) as { payload: string } | undefined;
    expect(audit).toBeDefined();
    expect(JSON.parse(audit!.payload).question_event_id).toBe(qid.id);
    // a subsequent listPending() sees nothing for it — finding 27's actual bug proven closed, not just documented
    const pending = db.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE delivery = 'pending' AND id = ?").get(qid.id);
    expect(pending).toEqual({ n: 0 });

    // finding 33 sibling: a question ALREADY resolved locally (no claim needed) is left untouched, not expired
    const db2 = openV0Fixture();
    db2.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:00:00.000Z', NULL, 'p1', 'lead', 'question', 'deterministic', 'claude-pretool', ?, 'delivered')`,
    ).run(JSON.stringify({ tool_use_id: "tu_resolved", questions: [] }));
    const qid2 = db2.prepare("SELECT id FROM workflow_events WHERE type='question'").get() as { id: number };
    db2.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-08-01T00:02:00.000Z', NULL, 'p1', 'lead', 'question-resolved', 'deterministic', 'claude-posttool', ?, 'local')`,
    ).run(JSON.stringify({ tool_use_id: "tu_resolved" }));
    migrate(db2);
    const ev2 = db2.prepare("SELECT delivery FROM workflow_events WHERE id = ?").get(qid2.id) as { delivery: string };
    expect(ev2.delivery).toBe("delivered"); // untouched — hasPendingQuestion's own resolution semantics already treat it as resolved
    expect(db2.prepare(
      "SELECT COUNT(*) AS n FROM mutations WHERE kind='workflow-question-expired' AND json_extract(payload,'$.question_event_id') = ?",
    ).get(qid2.id)).toEqual({ n: 0 });
    db.close();
    db2.close();
  });
});

describe("migration 8 provider-credential indexes", () => {
  const ts = "2026-09-13T00:00:00.000Z";
  function payload(fields: Record<string, unknown>) {
    return JSON.stringify({ ts, kind: "provider-credential", ...fields });
  }
  function insert(db: Database.Database, fields: Record<string, unknown>) {
    db.prepare("INSERT INTO mutations (ts, kind, ok, error, payload) VALUES (?, 'provider-credential', NULL, NULL, ?)").run(
      ts,
      payload(fields),
    );
  }

  it("lands at user_version 8 with two typed partial unique indexes", () => {
    const db = openDb(":memory:");
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    const idx = (db.prepare("PRAGMA index_list(mutations)").all() as { name: string }[]).map((i) => i.name).sort();
    expect(idx).toEqual([
      "mutations_kind",
      "mutations_provider_credential_active",
      "mutations_provider_credential_operation",
      "mutations_ts",
    ]);
    db.close();
  });

  it("rejects a duplicate operation_id and a second nonterminal provider_id", () => {
    const db = openDb(":memory:");
    insert(db, { operation_id: "op-1", provider_id: "openrouter", stage: "reserved" });
    expect(() => insert(db, { operation_id: "op-1", provider_id: "xai", stage: "done" })).toThrow();
    expect(() => insert(db, { operation_id: "op-2", provider_id: "openrouter", stage: "submitted" })).toThrow();
    insert(db, { operation_id: "op-other", provider_id: "xai", stage: "done" });
    insert(db, { operation_id: "op-next", provider_id: "xai", stage: "reserved" });
    db.prepare("INSERT INTO mutations (ts, kind, ok, error, payload) VALUES (?, 'file-edit', NULL, NULL, ?)").run(
      ts,
      payload({ operation_id: "op-1", provider_id: "openrouter", stage: "reserved" }),
    );
    db.close();
  });
});

describe("migration 11 creates missions (spec §6 Decisions 1-3, 6)", () => {
  it("lands at user_version 11 with an empty table and the one-active partial unique index", () => {
    const db = openDb(":memory:");
    expect(db.pragma("user_version", { simple: true })).toBe(11);
    const cols = (db.prepare("PRAGMA table_info(missions)").all() as { name: string }[]).map((c) => c.name);
    expect(cols).toEqual(["id", "name", "goal", "status_line", "state", "started_at", "ended_at", "milestones"]);
    expect(db.prepare("SELECT COUNT(*) AS n FROM missions").get()).toEqual({ n: 0 });
    const idx = (db.prepare("PRAGMA index_list(missions)").all() as { name: string }[]).map((i) => i.name);
    expect(idx).toEqual(["missions_one_active"]);
    db.close();
  });

  it("the partial unique index refuses a second concurrent active row, allows any number of non-active ones (spec §14 Risk 2)", () => {
    const db = openDb(":memory:");
    const insert = (state: string) => db.prepare(
      "INSERT INTO missions (name, goal, status_line, state, started_at, ended_at, milestones) VALUES (?, ?, '', ?, ?, NULL, '[]')",
    ).run("m", "g", state, "2026-09-19T00:00:00.000Z");
    insert("active");
    expect(() => insert("active")).toThrow();
    expect(() => insert("done")).not.toThrow();
    expect(() => insert("cancelled")).not.toThrow();
    db.close();
  });

  it("the state CHECK constraint refuses any value outside active/done/cancelled", () => {
    const db = openDb(":memory:");
    expect(() => db.prepare(
      "INSERT INTO missions (name, goal, status_line, state, started_at, ended_at, milestones) VALUES ('m', 'g', '', 'bogus', '2026-09-19T00:00:00.000Z', NULL, '[]')",
    ).run()).toThrow();
    db.close();
  });
});
