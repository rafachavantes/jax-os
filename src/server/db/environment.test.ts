import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";
import {
  createOwnDbReader, createOwnDbWorkerRunner, ownPaneKey, OWN_DB_TIMEOUT_MS, PANE_IDENTITY_MAX,
  systemOwnDbWorkerFactory, validateOwnDbMessage,
  type OwnDbWorkerData, type OwnDbWorkerLike,
} from "./environment";
import { openDb } from "./index";

const tempDirs: string[] = [];

async function tempOwnDb(): Promise<{ dir: string; path: string; db: Database.Database }> {
  const dir = await mkdtemp(join(tmpdir(), "jax-m5-db-"));
  tempDirs.push(dir);
  const path = join(dir, "jaxos.db");
  const db = openDb(path);
  return { dir, path, db };
}

afterEach(async () => {
  await Promise.all(tempDirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
});

type FakeWorker = OwnDbWorkerLike & {
  emit(event: "message" | "error" | "exit", value?: unknown): void;
  terminated: boolean;
};

function fakeWorker(over: Partial<OwnDbWorkerLike> = {}): FakeWorker {
  const listeners = new Map<string, ((value?: unknown) => void)[]>();
  const worker: FakeWorker = {
    terminated: false,
    once(event, listener) {
      const list = listeners.get(event) ?? [];
      list.push(listener);
      listeners.set(event, list);
      return worker;
    },
    terminate: vi.fn(async () => {
      worker.terminated = true;
      return 0;
    }),
    emit(event, value) {
      for (const listener of listeners.get(event) ?? []) listener(value);
    },
    ...over,
  };
  return worker;
}

describe("own-DB readonly worker (M5 Task 1)", () => {
  it("opens the fixed own DB path readonly and validates the identity cap", async () => {
    const run = vi.fn(async (_data: OwnDbWorkerData) => ({
      ok: true as const, schemaVersion: 15, latestMetricTs: null, paneContext: null,
    }));
    const read = createOwnDbReader(run);
    await read();
    const data = run.mock.calls[0][0];
    expect(data.path.endsWith("/jaxos.db")).toBe(true);
    expect(data.paneKey).toBeNull();
    expect(data.paneSql).toBeNull();

    await read({ pane: "%1", incarnation: "10:20" });
    const withPane = run.mock.calls[1][0];
    expect(withPane.paneKey).toBe("%1:10:20");
    expect(withPane.paneSql).toContain("pane_key = ?");
    expect(withPane.paneSql).toContain("emitter NOT IN");

    expect(ownPaneKey("%1", "x".repeat(PANE_IDENTITY_MAX))).toBeNull();
    expect(ownPaneKey("not-a-pane", "10:20")).toBeNull();
    expect(ownPaneKey("%1", null)).toBeNull();
  });

  it("reads schema, latest metric and a bounded pane context from a synthetic own DB", async () => {
    const { path, db } = await tempOwnDb();
    const insert = db.prepare(
      `INSERT INTO workflow_events (ts, project, type, source, emitter, delivery, pane, tmux_incarnation, role, harness_session)
       VALUES (?, ?, 'turn-stopped', 'hook', ?, 'local', '%1', '9:20', ?, ?)`,
    );
    insert.run("2026-11-19T00:00:00.000Z", "jax-os", "claude", "builder", "s1");
    insert.run("2026-11-19T00:01:00.000Z", "other", "codex", null, null);
    db.prepare("INSERT INTO metrics (ts, services) VALUES (?, ?)").run("2026-11-19T00:02:00.000Z", "{}");
    db.close();

    const run = createOwnDbWorkerRunner(systemOwnDbWorkerFactory, 5_000);
    const snapshot = await run({
      path,
      paneKey: "%1:9:20",
      paneSql: `SELECT project, role, harness_session, ts FROM workflow_events
        WHERE pane_key = ? AND pane IS NOT NULL AND emitter NOT IN ('jaxos', 'codex')
        ORDER BY id DESC LIMIT 1`,
    });
    expect(snapshot).toMatchObject({
      ok: true,
      schemaVersion: 15,
      latestMetricTs: "2026-11-19T00:02:00.000Z",
      paneContext: { project: "jax-os", role: "builder", session: "s1", observedAt: "2026-11-19T00:00:00.000Z" },
    });
  });

  it("resolves unavailable for a missing file, invalid message, worker error or missing message", async () => {
    const { path } = await tempOwnDb();
    const run = createOwnDbWorkerRunner(systemOwnDbWorkerFactory, 5_000);
    expect(await run({ path: join(path, "missing", "jaxos.db"), paneKey: null, paneSql: null })).toEqual({
      ok: false, error: "database-unavailable",
    });
    expect(validateOwnDbMessage({ ok: true, schemaVersion: "x" })).toMatchObject({ ok: true, schemaVersion: null });
    expect(validateOwnDbMessage({ nope: true })).toEqual({ ok: false, error: "database-unavailable" });

    let errorWorker: FakeWorker | undefined;
    const errored = createOwnDbWorkerRunner(() => {
      errorWorker = fakeWorker();
      return errorWorker;
    }, 50);
    const pending = errored({ path, paneKey: null, paneSql: null });
    errorWorker?.emit("error", new Error("boom"));
    errorWorker?.emit("exit", 0);
    expect(await pending).toEqual({ ok: false, error: "database-unavailable" });
  });

  it("terminates on timeout, keeps the operation pending until the worker exits and never returns a late value", async () => {
    let worker: FakeWorker | undefined;
    const run = createOwnDbWorkerRunner((data) => {
      expect(data.path).toBeTruthy();
      worker = fakeWorker();
      return worker;
    }, 5);
    let settled = false;
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null }).then((value) => {
      settled = true;
      return value;
    });
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(worker?.terminated).toBe(true);
    expect(settled).toBe(false);
    worker?.emit("message", { ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null });
    expect(settled).toBe(false);
    worker?.emit("exit", 0);
    expect(await pending).toEqual({ ok: false, error: "database-unavailable" });
  });

  it("publishes a validated message only after the worker exits normally", async () => {
    const worker = fakeWorker();
    const run = createOwnDbWorkerRunner(() => worker, 500);
    let settled = false;
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null }).then((value) => {
      settled = true;
      return value;
    });
    worker.emit("message", { ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null });
    expect(settled).toBe(false);
    expect(worker.terminated).toBe(false);
    worker.emit("exit", 0);
    expect(await pending).toMatchObject({ ok: true, schemaVersion: 15 });
    expect(worker.terminated).toBe(false);
  });

  it("keeps ownership after an error event instead of resolving before the actual exit", async () => {
    const worker = fakeWorker();
    const run = createOwnDbWorkerRunner(() => worker, 500);
    let settled = false;
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null }).then((value) => {
      settled = true;
      return value;
    });
    worker.emit("message", { ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null });
    worker.emit("error", new Error("boom"));
    expect(settled).toBe(false);
    expect(worker.terminated).toBe(true);
    worker.emit("exit", 0);
    expect(await pending).toEqual({ ok: false, error: "database-unavailable" });
  });

  it("rejects a normal exit that happens after the original allowance", async () => {
    const worker = fakeWorker();
    let clock = 0;
    const run = createOwnDbWorkerRunner(() => worker, 200, () => clock);
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null });
    worker.emit("message", { ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null });
    clock = 201;
    worker.emit("exit", 0);
    await expect(pending).resolves.toEqual({ ok: false, error: "database-unavailable" });
  });

  it("rejects a valid message followed by an abnormal exit without an error event", async () => {
    const worker = fakeWorker();
    const run = createOwnDbWorkerRunner(() => worker, 500, () => 0);
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null });
    worker.emit("message", { ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null });
    worker.emit("exit", 1);
    await expect(pending).resolves.toEqual({ ok: false, error: "database-unavailable" });
  });

  it("accepts only an actual code-0 exit before the deadline", async () => {
    const worker = fakeWorker();
    let clock = 0;
    const run = createOwnDbWorkerRunner(() => worker, 200, () => clock);
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null });
    worker.emit("message", { ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null });
    clock = 199;
    worker.emit("exit", 0);
    await expect(pending).resolves.toMatchObject({ ok: true, schemaVersion: 15 });
  });

  it("treats an undefined exit code as no proof of a normal exit", async () => {
    const worker = fakeWorker();
    const run = createOwnDbWorkerRunner(() => worker, 500, () => 0);
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null });
    worker.emit("message", { ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null });
    worker.emit("exit", undefined);
    await expect(pending).resolves.toEqual({ ok: false, error: "database-unavailable" });
  });

  it("counts startup time spent before control returned against the original deadline", async () => {
    let clock = 0;
    const worker = fakeWorker();
    const run = createOwnDbWorkerRunner(() => {
      clock += 250; // synchronous startup already exceeded the 200 ms allowance
      return worker;
    }, OWN_DB_TIMEOUT_MS, () => clock);
    let settled = false;
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null }).then((value) => {
      settled = true;
      return value;
    });
    expect(worker.terminated).toBe(true);
    expect(settled).toBe(false);
    worker.emit("message", { ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null });
    worker.emit("exit", 0);
    await expect(pending).resolves.toEqual({ ok: false, error: "database-unavailable" });
  });

  it("handles an asynchronous terminate rejection and still resolves at exit", async () => {
    let worker: FakeWorker | undefined;
    const run = createOwnDbWorkerRunner(() => {
      worker = fakeWorker({ terminate: () => Promise.reject(new Error("terminate failed")) });
      return worker;
    }, 5);
    const pending = run({ path: "/x/jaxos.db", paneKey: null, paneSql: null });
    await new Promise((resolve) => setTimeout(resolve, 20));
    worker?.emit("exit", 0);
    await expect(pending).resolves.toEqual({ ok: false, error: "database-unavailable" });
  });

  it("uses a 200 ms default deadline including worker startup", () => {
    expect(OWN_DB_TIMEOUT_MS).toBe(200);
  });
});
