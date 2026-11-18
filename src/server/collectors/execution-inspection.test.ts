import { afterEach, describe, expect, it } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { openDb } from "../db";
import { inspectExecutionRows } from "./execution-inspection";
import type { ProcReader } from "./execution-worker";

const NOW = Date.parse("2026-11-18T12:00:00.000Z");
const RUN = "0123456789ab";
const DEST = "alpha";
const roots: string[] = [];

afterEach(() => {
  while (roots.length) rmSync(roots.pop()!, { recursive: true, force: true });
});

function fixture(): { root: string; abs: string; db: ReturnType<typeof openDb>; startId: number } {
  const root = mkdtempSync(join(tmpdir(), "jaxos-inspection-"));
  roots.push(root);
  const abs = join(root, DEST);
  mkdirSync(join(abs, ".git"), { recursive: true });
  mkdirSync(join(abs, ".local", "runs", RUN), { recursive: true });
  writeFileSync(join(abs, ".local", "runs", RUN, "manifest.json"), JSON.stringify({
    run_id: RUN, project: DEST, repo: abs, kind: "build", role: "builder", branch: "feat/x", worktree: join(root, "alpha-wt"),
  }));
  mkdirSync(join(abs, ".local", "reports"), { recursive: true });
  writeFileSync(join(abs, ".local", "reports", `${RUN}.md`), "report");
  const db = openDb(":memory:");
  const insert = db.prepare(
    `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
     VALUES (?, ?, ?, 'builder', ?, 'deterministic', 'wrapper', ?, 'local')`,
  );
  const startId = Number(insert.run("2026-11-18T11:59:30.000Z", RUN, DEST, "run-started", JSON.stringify({
    kind: "build", runtime: "codex", caller: "claude", session: "jax-alpha-build", repo: abs,
  })).lastInsertRowid);
  return { root, abs, db, startId };
}

function proc(abs: string): ProcReader {
  const argv = ["/usr/bin/python3.14", "/opt/jax-os/scripts/jaxflow.py", "--run-worker", join(abs, ".local", "runs", RUN, "manifest.json")];
  const stat = (pid: number) => `${pid} (node) S 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1000 0 0`;
  return {
    async stat(pid) { return pid === 100 ? stat(pid) : null; },
    async children() { return []; },
    async cmdline(pid) { return pid === 100 ? argv : null; },
    async cwd(pid) { return pid === 100 ? abs : null; },
    async exe(pid) { return pid === 100 ? "/usr/bin/python3.14" : null; },
  };
}

describe("inspectExecutionRows", () => {
  it("returns durable artifacts and a positively identified worker for a matching open row", async () => {
    const { root, abs, db, startId } = fixture();
    const rows = await inspectExecutionRows(db, [{ destination: DEST, runId: RUN, startEventId: startId }], {
      reposRoot: root,
      now: () => NOW,
      tmux: async () => `%1 100 jax-alpha-build\n`,
      proc: proc(abs),
      realpath: async (p: string) => p,
      launcherPath: "/opt/jax-os/scripts/jaxflow.py",
      lastGood: new Map(),
    });
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      destination: DEST, runId: RUN, branch: "feat/x", branchSource: "manifest",
      report: { rel: `alpha/.local/reports/${RUN}.md` },
      refreshNeeded: false,
      observation: { status: "observed", pid: 100 },
      current: { outcome: "open", terminal: false },
    });
  });

  it("overlays a terminal recorded after the durable page read without a process scan", async () => {
    const { root, abs, db, startId } = fixture();
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-11-18T11:59:40.000Z', ?, ?, 'builder', 'run-finished', 'deterministic', 'wrapper', ?, 'local')`,
    ).run(RUN, DEST, JSON.stringify({ contract_status: "ok", result: "failure" }));
    const rows = await inspectExecutionRows(db, [{ destination: DEST, runId: RUN, startEventId: startId }], {
      reposRoot: root, now: () => NOW, proc: proc(abs), realpath: async (p: string) => p, lastGood: new Map(),
    });
    expect(rows[0].current).toMatchObject({ terminal: true, outcome: "failure" });
    expect(rows[0].observation).toEqual({ status: "not-checked" });
  });

  it("marks a newer start refresh-needed and suppresses the stale worker label", async () => {
    const { root, db, startId } = fixture();
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-11-18T11:59:50.000Z', ?, ?, 'builder', 'run-started', 'deterministic', 'wrapper', ?, 'local')`,
    ).run(RUN, DEST, JSON.stringify({ kind: "build", session: "jax-alpha-build" }));
    const rows = await inspectExecutionRows(db, [{ destination: DEST, runId: RUN, startEventId: startId }], {
      reposRoot: root, now: () => NOW, lastGood: new Map(), tmux: async () => "",
    });
    expect(rows[0].refreshNeeded).toBe(true);
    expect(rows[0].observation).toEqual({ status: "not-checked" });
  });

  it("keeps known rows readable when artifact discovery is unavailable", async () => {
    const { db, startId } = fixture();
    const rows = await inspectExecutionRows(db, [{ destination: DEST, runId: RUN, startEventId: startId }], {
      reposRoot: join(tmpdir(), "jaxos-does-not-exist-xyz"),
      now: () => NOW,
      lastGood: new Map(),
      tmux: async () => "",
    });
    expect(rows[0].branch).toBeNull();
    expect(rows[0].warnings).toContain("root-unavailable");
    expect(rows[0].current?.outcome).toBe("open");
  });

  it("never resolves an unknown selector into a row", async () => {
    const { root, db } = fixture();
    const rows = await inspectExecutionRows(db, [{ destination: DEST, runId: "ffffffffffff", startEventId: 1 }], {
      reposRoot: root, now: () => NOW, lastGood: new Map(),
    });
    expect(rows[0].current).toBeNull();
    expect(rows[0].observation).toEqual({ status: "not-checked" });
  });
});

describe("inspectExecutionRows — fresh observation and single flight", () => {
  function selector(startId: number) {
    return { destination: DEST, runId: RUN, startEventId: startId };
  }

  function insertFinish(db: ReturnType<typeof openDb>): void {
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-11-18T11:59:40.000Z', ?, ?, 'builder', 'run-finished', 'deterministic', 'wrapper', ?, 'local')`,
    ).run(RUN, DEST, JSON.stringify({ result: "failure" }));
  }

  function insertStart(db: ReturnType<typeof openDb>): void {
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES ('2026-11-18T11:59:50.000Z', ?, ?, 'builder', 'run-started', 'deterministic', 'wrapper', ?, 'local')`,
    ).run(RUN, DEST, JSON.stringify({ kind: "build", session: "jax-alpha-build" }));
  }

  it("lets a finish inserted during the process snapshot win", async () => {
    const { root, db, startId } = fixture();
    const rows = await inspectExecutionRows(db, [selector(startId)], {
      reposRoot: root, now: () => NOW, lastGood: new Map(),
      tmux: async () => { insertFinish(db); return `%1 100 jax-alpha-build\n`; },
    });
    expect(rows[0].current).toMatchObject({ terminal: true, outcome: "failure", contractStatus: "invalid" });
    expect(rows[0].current?.warnings).toContain("contract-invalid");
    expect(rows[0].observation).toEqual({ status: "not-checked" });
  });

  it("marks a newer start inserted during the snapshot refresh-needed", async () => {
    const { root, db, startId } = fixture();
    const rows = await inspectExecutionRows(db, [selector(startId)], {
      reposRoot: root, now: () => NOW, lastGood: new Map(),
      tmux: async () => { insertStart(db); return `%1 100 jax-alpha-build\n`; },
    });
    expect(rows[0].refreshNeeded).toBe(true);
    expect(rows[0].observation).toEqual({ status: "not-checked" });
  });

  it("shares one native pass and starts a new one only after settlement", async () => {
    const { root, db, startId } = fixture();
    let tmuxCalls = 0;
    let release: () => void = () => {};
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const deps = {
      reposRoot: join(root, "missing"),
      now: () => NOW,
      lastGood: new Map(),
      tmux: async () => { tmuxCalls++; await gate; return ""; },
    };
    const first = inspectExecutionRows(db, [selector(startId)], deps);
    while (tmuxCalls === 0) await new Promise((resolve) => setTimeout(resolve, 1));
    const same = inspectExecutionRows(db, [selector(startId)], deps);
    const differing = await inspectExecutionRows(db, [selector(startId + 1)], deps);
    expect(differing[0]).toMatchObject({ observation: { status: "not-checked" }, refreshNeeded: true });
    expect(differing[0].current).toMatchObject({ outcome: "open", terminal: false });
    expect(tmuxCalls).toBe(1);

    release();
    const firstRows = await first;
    expect(await same).toBe(firstRows);
    await new Promise((resolve) => setTimeout(resolve, 10));

    const next = await inspectExecutionRows(db, [selector(startId)], deps);
    expect(tmuxCalls).toBe(2);
    expect(next[0].observation).not.toEqual({ status: "not-checked" });
  });

  it("does not scan tmux when the resolved rows have no open attempt", async () => {
    const { root, db, startId } = fixture();
    insertFinish(db);
    let calls = 0;
    const rows = await inspectExecutionRows(db, [selector(startId)], {
      reposRoot: root, now: () => NOW, lastGood: new Map(),
      tmux: async () => { calls++; return ""; },
    });
    expect(rows[0].current?.terminal).toBe(true);
    expect(calls).toBe(0);
  });
});
