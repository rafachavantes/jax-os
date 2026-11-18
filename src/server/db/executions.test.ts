import { describe, expect, it } from "vitest";
import type Database from "better-sqlite3";
import { openDb } from "./index";
import { buildExecutionQuery, getCurrentRunStates, queryExecutions } from "./workflows";
import {
  decodeExecutionCursor, encodeExecutionCursor, executionFilterIdentity,
  type ExecutionFilters,
} from "../../lib/executions";

const NOW = Date.parse("2026-11-18T12:00:00.000Z");
const DEST = "jax-os";
const RUN_A = "aaaaaaaaaaaa";
const RUN_B = "bbbbbbbbbbbb";
const RUN_C = "cccccccccccc";

function filters(overrides: Partial<ExecutionFilters> = {}): ExecutionFilters {
  return { project: null, kind: "all", state: "all", period: "all", run: null, limit: 20, ...overrides };
}

function ev(
  db: Database.Database,
  o: {
    project?: string; runId: string; type: "run-started" | "run-finished";
    role?: string | null; emitter?: string; ts: string; payload?: unknown; rawPayload?: string;
  },
): number {
  const info = db.prepare(
    `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
     VALUES (?, ?, ?, ?, ?, 'deterministic', ?, ?, 'local')`,
  ).run(
    o.ts, o.runId, o.project ?? DEST, o.role === undefined ? "builder" : o.role, o.type,
    o.emitter ?? "wrapper",
    o.rawPayload !== undefined ? o.rawPayload : JSON.stringify(o.payload ?? {}),
  );
  return Number(info.lastInsertRowid);
}

function start(db: Database.Database, runId: string, ts: string, payload: Record<string, unknown> = {}, extra: Partial<Parameters<typeof ev>[1]> = {}): number {
  return ev(db, { runId, type: "run-started", ts, payload: { kind: "build", runtime: "codex", caller: "claude", target: "feat/x", ...payload }, ...extra });
}

function finish(db: Database.Database, runId: string, ts: string, payload: Record<string, unknown> = {}, extra: Partial<Parameters<typeof ev>[1]> = {}): number {
  return ev(db, { runId, type: "run-finished", ts, payload: { contract_status: "ok", result: "success", ...payload }, ...extra });
}

function run(db: Database.Database, runId: string, p: { start: string; finish?: string; project?: string }): void {
  start(db, runId, p.start, {}, { project: p.project });
  if (p.finish) finish(db, runId, p.finish, {}, { project: p.project });
}

describe("queryExecutions — durable projection", () => {
  it("groups by destination+run, newest relevant event first, and keeps the same run ID in two destinations", () => {
    const db = openDb(":memory:");
    run(db, RUN_A, { start: "2026-11-17T10:00:00.000Z", finish: "2026-11-17T10:05:00.000Z" });
    start(db, RUN_A, "2026-11-18T09:00:00.000Z", { kind: "diff" }, { project: "other", role: "reviewer" });
    finish(db, RUN_A, "2026-11-18T09:10:00.000Z", { verdict: "approve" }, { project: "other", role: "reviewer" });
    run(db, RUN_B, { start: "2026-11-18T10:00:00.000Z", finish: "2026-11-18T10:01:00.000Z" });

    const page = queryExecutions(db, filters(), null, NOW);
    expect(page.rows.map((r) => [r.destination, r.runId, r.outcome])).toEqual([
      [DEST, RUN_B, "success"],
      ["other", RUN_A, "approve"],
      [DEST, RUN_A, "success"],
    ]);
  });

  it("excludes known merge-only runs and ignores non-lifecycle events", () => {
    const db = openDb(":memory:");
    start(db, RUN_A, "2026-11-17T10:00:00.000Z", { kind: "merge" });
    finish(db, RUN_A, "2026-11-17T10:01:00.000Z", { result: "success" });
    ev(db, { runId: RUN_B, type: "run-started", ts: "2026-11-17T10:00:00.000Z", payload: {} });
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-11-17T10:00:00.000Z', ?, ?, NULL, 'tool-used', 'deterministic', 'wrapper', '{}', 'local')`,
    ).run(RUN_B, DEST);
    expect(queryExecutions(db, filters(), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_B]);
  });

  it("keeps orphan finishes by recorded role and accepts jaxos finishes", () => {
    const db = openDb(":memory:");
    finish(db, RUN_A, "2026-11-17T10:00:00.000Z", { result: "failure" }, { emitter: "jaxos" });
    start(db, RUN_B, "2026-11-17T09:00:00.000Z");
    finish(db, RUN_B, "2026-11-17T09:05:00.000Z", { contract_status: "cancelled" }, { emitter: "jaxos" });
    ev(db, { runId: RUN_C, type: "run-started", ts: "2026-11-17T08:00:00.000Z", emitter: "jaxos", payload: { kind: "build" } });
    finish(db, RUN_C, "2026-11-17T08:05:00.000Z");

    const rows = queryExecutions(db, filters(), null, NOW).rows;
    const a = rows.find((r) => r.runId === RUN_A)!;
    expect(a.outcome).toBe("failure");
    expect(a.role).toBe("builder");
    expect(a.warnings).toContain("orphan");
    expect(rows.find((r) => r.runId === RUN_B)!.outcome).toBe("cancelled");
    // a jaxos start is not admissible: only its finish remains, so it is an orphan (never a guessed start)
    expect(rows.find((r) => r.runId === RUN_C)!.startEventId).toBeNull();
  });

  it("latest finish by event ID wins on conflicting duplicates; a later start reopens", () => {
    const db = openDb(":memory:");
    start(db, RUN_A, "2026-11-17T10:00:00.000Z");
    finish(db, RUN_A, "2026-11-17T10:05:00.000Z", { contract_status: "ok", result: "failure" });
    finish(db, RUN_A, "2026-11-17T10:06:00.000Z", { contract_status: "ok", result: "success" });
    expect(queryExecutions(db, filters(), null, NOW).rows[0].outcome).toBe("success");

    start(db, RUN_A, "2026-11-17T10:07:00.000Z", { kind: "build" });
    const reopened = queryExecutions(db, filters(), null, NOW).rows[0];
    expect(reopened.outcome).toBe("open");
    expect(reopened.terminal).toBe(false);
  });

  it("a later malformed start reopens the run without discarding its event", () => {
    const db = openDb(":memory:");
    start(db, RUN_A, "2026-11-17T10:00:00.000Z");
    finish(db, RUN_A, "2026-11-17T10:05:00.000Z");
    ev(db, { runId: RUN_A, type: "run-started", ts: "2026-11-17T10:10:00.000Z", rawPayload: "{not-json" });
    const row = queryExecutions(db, filters(), null, NOW).rows[0];
    expect(row.outcome).toBe("open");
    expect(row.kind).toBeNull();
    expect(row.startEventId).not.toBeNull();
  });

  it("role/kind conflicts appear only in All and warn", () => {
    const db = openDb(":memory:");
    start(db, RUN_A, "2026-11-17T10:00:00.000Z", { kind: "diff" }, { role: "builder" });
    finish(db, RUN_A, "2026-11-17T10:05:00.000Z");
    const all = queryExecutions(db, filters(), null, NOW).rows;
    expect(all).toHaveLength(1);
    expect(all[0].warnings).toContain("kind-role-conflict");
    expect(queryExecutions(db, filters({ kind: "build" }), null, NOW).rows).toHaveLength(0);
    expect(queryExecutions(db, filters({ kind: "review" }), null, NOW).rows).toHaveLength(0);
    expect(queryExecutions(db, filters({ kind: "diff" }), null, NOW).rows).toHaveLength(0);
  });

  it("applies every filter before the page limit", () => {
    const db = openDb(":memory:");
    run(db, RUN_A, { start: "2026-11-17T10:00:00.000Z", finish: "2026-11-17T10:01:00.000Z", project: "alpha" });
    start(db, RUN_B, "2026-11-18T10:00:00.000Z", { kind: "diff" }, { role: "reviewer", project: "beta" });
    finish(db, RUN_B, "2026-11-18T10:01:00.000Z", { verdict: "reject" }, { role: "reviewer", project: "beta" });
    start(db, RUN_C, "2026-10-01T10:00:00.000Z", { kind: "spec" }, { role: "reviewer", project: "alpha" });

    expect(queryExecutions(db, filters({ project: "alpha", limit: 1 }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_C]);
    expect(queryExecutions(db, filters({ kind: "review" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_C, RUN_B]);
    expect(queryExecutions(db, filters({ kind: "build" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_A]);
    expect(queryExecutions(db, filters({ kind: "spec" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_C]);
    expect(queryExecutions(db, filters({ state: "open" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_C]);
    expect(queryExecutions(db, filters({ state: "finished" }), null, NOW).rows).toHaveLength(2);
    expect(queryExecutions(db, filters({ state: "success" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_A]);
    expect(queryExecutions(db, filters({ state: "reject" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_B]);
    expect(queryExecutions(db, filters({ period: "7d" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_B, RUN_A]);
    expect(queryExecutions(db, filters({ period: "30d" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_B, RUN_A]);
    expect(queryExecutions(db, filters({ period: "all" }), null, NOW).rows.map((r) => r.runId)).toEqual([RUN_C, RUN_B, RUN_A]);
  });

  it("an unknown period anchor stays available only in All", () => {
    const db = openDb(":memory:");
    start(db, RUN_A, "not-a-date", { kind: "build" });
    expect(queryExecutions(db, filters({ period: "7d" }), null, NOW).rows).toHaveLength(0);
    const all = queryExecutions(db, filters({ period: "all" }), null, NOW).rows;
    expect(all).toHaveLength(1);
    expect(all[0].firstDispatchedAt).toBeNull();
    expect(all[0].warnings).toContain("period-unknown");
  });

  it("falls back to the latest finite finish for the period only", () => {
    const db = openDb(":memory:");
    start(db, RUN_A, "not-a-date");
    finish(db, RUN_A, "2026-11-17T10:00:00.000Z");
    const rows = queryExecutions(db, filters({ period: "7d" }), null, NOW).rows;
    expect(rows).toHaveLength(1);
    expect(rows[0].firstDispatchedAt).toBeNull();
    expect(rows[0].finishedAt).toBe("2026-11-17T10:00:00.000Z");
  });

  it("keeps event IDs authoritative when timestamps are out of order", () => {
    const db = openDb(":memory:");
    start(db, RUN_A, "2026-11-17T11:00:00.000Z");
    finish(db, RUN_A, "2026-11-17T09:00:00.000Z");
    const row = queryExecutions(db, filters(), null, NOW).rows[0];
    expect(row.outcome).toBe("success");
    expect(row.durationMs).toBeNull(); // finish before start: never negative
    expect(row.warnings).toContain("duration-unknown");
  });

  it("honours an exact run context and ignores every other scope", () => {
    const db = openDb(":memory:");
    run(db, RUN_A, { start: "2026-11-17T10:00:00.000Z", finish: "2026-11-17T10:01:00.000Z" });
    run(db, RUN_B, { start: "2026-11-18T10:00:00.000Z", finish: "2026-11-18T10:01:00.000Z" });
    const rows = queryExecutions(db, filters({ run: { project: DEST, runId: RUN_A }, period: "all" }), null, NOW).rows;
    expect(rows.map((r) => r.runId)).toEqual([RUN_A]);
  });
});

describe("queryExecutions — cursor", () => {
  it("freezes the fence and ordering: appends never duplicate or skip a prior page", () => {
    const db = openDb(":memory:");
    for (let i = 0; i < 5; i++) {
      const id = String(i).padStart(12, "0");
      run(db, id, { start: `2026-11-17T1${i}:00:00.000Z`, finish: `2026-11-17T1${i}:01:00.000Z` });
    }
    const first = queryExecutions(db, filters({ limit: 2 }), null, NOW);
    expect(first.rows).toHaveLength(2);
    expect(first.nextCursor).not.toBeNull();

    // a new run arrives between pages
    run(db, "ffffffffffff", { start: "2026-11-18T11:00:00.000Z", finish: "2026-11-18T11:01:00.000Z" });

    const cursor = decodeExecutionCursor(first.nextCursor!)!;
    const second = queryExecutions(db, filters({ limit: 2 }), cursor, NOW);
    expect(second.rows.map((r) => r.runId)).toEqual(["000000000002", "000000000001"]);
    const third = queryExecutions(db, filters({ limit: 2 }), decodeExecutionCursor(second.nextCursor!)!, NOW);
    expect(third.rows.map((r) => r.runId)).toEqual(["000000000000"]);
    expect(third.nextCursor).toBeNull();
    // the first page after an explicit refresh (no cursor) sees the append
    expect(queryExecutions(db, filters({ limit: 2 }), null, NOW).rows[0].runId).toBe("ffffffffffff");
  });

  it("rejects an identity mismatch, an impossible ordering and a future fence", () => {
    const db = openDb(":memory:");
    run(db, RUN_A, { start: "2026-11-17T10:00:00.000Z", finish: "2026-11-17T10:01:00.000Z" });
    const identity = executionFilterIdentity(filters());
    const base = { v: 1 as const, fence: 2, lastId: 2, project: DEST, runId: RUN_A, cutoff: null, identity };
    expect(() => queryExecutions(db, filters({ kind: "build" }), base, NOW)).toThrow("invalid cursor");
    expect(() => queryExecutions(db, filters(), { ...base, lastId: 3 }, NOW)).toThrow("invalid cursor");
    expect(() => queryExecutions(db, filters(), { ...base, fence: 999 }, NOW)).toThrow("invalid cursor");
    expect(() => queryExecutions(db, filters({ period: "7d" }), { ...base, cutoff: null }, NOW)).toThrow("invalid cursor");
  });

  it("binds the captured period cutoff across pages", () => {
    const db = openDb(":memory:");
    run(db, RUN_A, { start: "2026-11-17T10:00:00.000Z", finish: "2026-11-17T10:01:00.000Z" });
    run(db, RUN_B, { start: "2026-11-16T10:00:00.000Z", finish: "2026-11-16T10:01:00.000Z" });
    const first = queryExecutions(db, filters({ period: "7d", limit: 1 }), null, NOW);
    expect(first.nextCursor).not.toBeNull();
    // an old run outside the captured cutoff cannot appear on a later page even when the clock
    // is advanced past the point where it would qualify
    run(db, RUN_C, { start: "2026-10-01T10:00:00.000Z", finish: "2026-10-01T10:01:00.000Z" });
    const later = NOW + 30 * 86_400_000;
    const second = queryExecutions(db, filters({ period: "7d", limit: 1 }), decodeExecutionCursor(first.nextCursor!)!, later);
    expect(second.rows.map((r) => r.runId)).toEqual([RUN_A]);
    expect(second.nextCursor).toBeNull();
  });
});

describe("execution history query plan", () => {
  it("uses the additive run index on a synthetic 100k-event / 2k-run ledger", () => {
    const db = openDb(":memory:");
    const insert = db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES (?, ?, ?, 'builder', ?, 'deterministic', 'wrapper', '{}', 'local')`,
    );
    const noise = db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES (?, NULL, 'noise', 'lead', 'tool-used', 'deterministic', 'wrapper', '{}', 'local')`,
    );
    const stmt = queryExecutions;
    db.transaction(() => {
      for (let r = 0; r < 2000; r++) {
        const runId = r.toString(16).padStart(12, "0");
        const project = r % 2 === 0 ? "alpha" : "beta";
        for (let i = 0; i < 40; i++) {
          insert.run(`2026-11-${String(1 + (i % 17)).padStart(2, "0")}T10:00:00.000Z`, runId, project, i % 2 === 0 ? "run-started" : "run-finished");
        }
      }
      for (let i = 0; i < 20_000; i++) noise.run("2026-11-17T10:00:00.000Z");
    })();

    const started = performance.now();
    const page = stmt(db, filters(), null, NOW); // also registers the connection's ts function
    const elapsed = performance.now() - started;
    const built = buildExecutionQuery(filters({ project: "alpha" }), null, Number.MAX_SAFE_INTEGER, null);
    const plan = db.prepare(`EXPLAIN QUERY PLAN ${built.sql}`).all(...built.params) as { detail: string }[];
    const detail = plan.map((p) => p.detail).join("\n");
    expect(detail).toContain("workflow_events_executions");
    expect(page.rows).toHaveLength(20);
    expect(elapsed).toBeLessThan(2_000); // evidence bound, not a universal SLA
    // all-project scope also stays on the index
    const allBuilt = buildExecutionQuery(filters(), null, Number.MAX_SAFE_INTEGER, null);
    const allPlan = db.prepare(`EXPLAIN QUERY PLAN ${allBuilt.sql}`).all(...allBuilt.params) as { detail: string }[];
    expect(allPlan.map((p) => p.detail).join("\n")).toContain("workflow_events_executions");
  }, 60_000);
});

describe("getCurrentRunStates", () => {
  it("re-resolves a bounded selector set through the own DB in one pass", () => {
    const db = openDb(":memory:");
    const startId = start(db, RUN_A, "2026-11-18T11:59:30.000Z");
    const finishId = finish(db, RUN_A, "2026-11-18T11:59:40.000Z");
    const states = getCurrentRunStates(db, [
      { destination: DEST, runId: RUN_A },
      { destination: DEST, runId: RUN_B },
      { destination: "../etc", runId: RUN_C },
    ]);
    expect(states[0]).toMatchObject({
      destination: DEST, runId: RUN_A, startEventId: startId, finishEventId: finishId,
      attemptStartedAt: "2026-11-18T11:59:30.000Z",
    });
    expect(states[0].row?.outcome).toBe("success");
    expect(states[1].row).toBeNull();
    expect(states[2].row).toBeNull(); // illegal destination never reaches a query
  });
});
