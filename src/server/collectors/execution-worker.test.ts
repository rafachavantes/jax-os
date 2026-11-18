import { describe, expect, it } from "vitest";
import {
  applyCurrentOverlay, inspectWorkers, parsePaneSnapshot, parseProcStat, workerKey,
  type ProcReader, type WorkerTarget,
} from "./execution-worker";
import { BudgetError, InspectionBudget } from "./execution-artifacts";
import type { ExecutionRow } from "../../lib/executions";
import type { CurrentExecutionState } from "../db/workflows";

const NOW = Date.parse("2026-11-18T12:00:00.000Z");
const DEST = "alpha";
const RUN = "0123456789ab";
const MANIFEST = `/repos/${DEST}/.local/runs/${RUN}/manifest.json`;
const REPO = `/repos/${DEST}`;

type Node = { ppid?: number; start: string; argv: string[]; cwd?: string; exe?: string; children?: number[] };

function statLine(pid: number, comm: string, ppid: number, start: string): string {
  const rest = ["S", String(ppid), "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "0", start, "0", "0"];
  return `${pid} (${comm}) ${rest.join(" ")}`;
}

function procTree(nodes: Record<number, Node>, unstable?: (pid: number) => boolean): ProcReader {
  const seen = new Map<number, number>();
  return {
    async stat(pid) {
      const node = nodes[pid];
      if (!node) return null;
      const count = (seen.get(pid) ?? 0) + 1;
      seen.set(pid, count);
      const start = unstable?.(pid) && count > 1 ? String(99_999 + count) : node.start;
      return statLine(pid, "proc with (parens)", node.ppid ?? 1, start);
    },
    async children(pid) {
      return nodes[pid]?.children ?? null;
    },
    async cmdline(pid) {
      return nodes[pid]?.argv ?? null;
    },
    async cwd(pid) {
      return nodes[pid]?.cwd ?? null;
    },
    async exe(pid) {
      return nodes[pid]?.exe ?? null;
    },
  };
}

function wrapperNode(o: { start?: string; cwd?: string; argv?: string[]; exe?: string } = {}): Node {
  return {
    start: o.start ?? "1000",
    argv: o.argv ?? ["/usr/bin/python3.14", "/opt/jax-os/scripts/jaxflow.py", "--run-worker", MANIFEST],
    cwd: o.cwd ?? REPO,
    exe: o.exe ?? "/usr/bin/python3.14",
    children: [],
  };
}

function target(overrides: Partial<WorkerTarget> = {}): WorkerTarget {
  return {
    destination: DEST, runId: RUN, startEventId: 1, startTs: "2026-11-18T11:59:30.000Z",
    session: `jax-${DEST}-build`, manifestPath: MANIFEST, repo: REPO, ...overrides,
  };
}

function deps(nodes: Record<number, Node>, extra: Partial<Parameters<typeof inspectWorkers>[1]> & { unstable?: (pid: number) => boolean } = {}) {
  return {
    tmux: async () => `%1 100 jax-${DEST}-build\n%2 200 other-session\n`,
    proc: procTree(nodes, extra.unstable),
    realpath: async (p: string) => p,
    now: () => NOW,
    budget: new InspectionBudget(4, 1_000, () => NOW),
    cache: new Map<string, { at: number; observedAt: string }>(),
    ...extra,
  };
}

function observe(nodes: Record<number, Node>, t: WorkerTarget = target(), extra: Parameters<typeof deps>[1] = {}) {
  return inspectWorkers([t], deps(nodes, extra));
}

describe("parsers", () => {
  it("parses pane snapshots and keeps session names with spaces intact", () => {
    expect(parsePaneSnapshot("%12 3456 my session name\n%13 99 x\nbogus\n")).toEqual([
      { paneId: "%12", panePid: 3456, session: "my session name" },
      { paneId: "%13", panePid: 99, session: "x" },
    ]);
  });

  it("parses /proc stat with spaces and parentheses in comm", () => {
    expect(parseProcStat(statLine(321, "my (odd) proc", 7, "12345"))).toEqual({ pid: 321, ppid: 7, startTicks: "12345" });
    expect(parseProcStat("garbage")).toBeNull();
    expect(parseProcStat("1 (x) S 1")).toBeNull();
  });
});

describe("inspectWorkers — positive identity", () => {
  it("accepts a pane_pid that is itself the exact launcher tuple", async () => {
    const result = await observe({ 100: wrapperNode() });
    expect(result.get(workerKey(target()))).toMatchObject({ status: "observed", pid: 100 });
  });

  it("accepts a bounded shell-root child chain (depth 2)", async () => {
    const result = await observe({
      100: { start: "1", argv: ["/bin/bash"], children: [101] },
      101: { ppid: 100, start: "2", argv: ["/bin/sh"], children: [102] },
      102: { ppid: 101, ...wrapperNode({ start: "3" }) },
    });
    expect(result.get(workerKey(target()))).toMatchObject({ status: "observed", pid: 102 });
  });

  it("rejects a wrong manifest, cwd, interpreter or executable", async () => {
    const cases: Record<number, Node>[] = [
      { 100: wrapperNode({ argv: ["/usr/bin/python3.14", "/opt/jax-os/scripts/jaxflow.py", "--run-worker", "/other/manifest.json"] }) },
      { 100: wrapperNode({ cwd: "/repos/alpha-wt" }) },
      { 100: wrapperNode({ argv: ["/usr/bin/node", "/opt/jax-os/scripts/jaxflow.py", "--run-worker", MANIFEST] }) },
      { 100: wrapperNode({ argv: ["/usr/bin/python3.14", "/opt/jax-os/scripts/other.py", "--run-worker", MANIFEST] }) },
      { 100: wrapperNode({ exe: "/usr/bin/other-python" }) },
    ];
    for (const nodes of cases) {
      const result = await observe(nodes);
      expect(result.get(workerKey(target()))?.status).not.toBe("observed");
    }
  });

  it("rejects PID replacement between identity reads", async () => {
    const result = await observe({ 100: wrapperNode() }, target(), { unstable: (pid) => pid === 100 });
    expect(result.get(workerKey(target()))?.status).not.toBe("observed");
  });

  it("treats multiple matching candidates as ambiguous", async () => {
    const result = await observe({
      100: { start: "1", argv: ["/bin/bash"], children: [101, 102] },
      101: { ppid: 100, ...wrapperNode({ start: "2" }) },
      102: { ppid: 100, ...wrapperNode({ start: "3" }) },
    });
    expect(result.get(workerKey(target()))).toMatchObject({ status: "unknown", warning: "observation-ambiguous" });
  });

  it("returns not-checked beyond the 12-run observation budget", async () => {
    const targets = Array.from({ length: 13 }, (_, i) => target({ runId: i.toString(16).padStart(12, "0") }));
    const result = await inspectWorkers(targets, deps({ 100: wrapperNode() }));
    expect(result.get(workerKey(targets[12]))).toEqual({ status: "not-checked" });
  });
});

describe("inspectWorkers — absence, failure and cache", () => {
  it("an absent pane on a fresh recent start awaits confirmation", async () => {
    const result = await inspectWorkers([target()], { ...deps({}), tmux: async () => "%2 200 other\n" });
    expect(result.get(workerKey(target()))).toEqual({ status: "awaiting-confirmation" });
  });

  it("an absent pane on an old or invalid start never claims awaiting", async () => {
    const old = await inspectWorkers([target({ startTs: "2026-11-18T10:00:00.000Z" })], { ...deps({}), tmux: async () => "" });
    expect(old.get(workerKey(target({ startTs: "2026-11-18T10:00:00.000Z" })))).toEqual({ status: "no-worker-observed" });
    const invalid = await inspectWorkers([target({ startTs: "nope" })], { ...deps({}), tmux: async () => "" });
    expect(invalid.get(workerKey(target({ startTs: "nope" })))).toEqual({ status: "unknown", warning: "timestamp-unknown" });
  });

  it("a prior positive then a fresh absence is no-worker-observed even under 60s", async () => {
    const t = target();
    const cache = new Map<string, { at: number; observedAt: string }>();
    expect((await observe({ 100: wrapperNode() }, t, { cache })).get(workerKey(t))?.status).toBe("observed");
    const absent = await inspectWorkers([t], { ...deps({}), cache, tmux: async () => "%2 200 other\n" });
    expect(absent.get(workerKey(t))).toMatchObject({ status: "no-worker-observed", lastGoodAt: new Date(NOW).toISOString() });
  });

  it("a tmux source failure is unknown, never a healthy absence", async () => {
    const result = await inspectWorkers([target()], { ...deps({}), tmux: async () => { throw new Error("permission denied"); } });
    expect(result.get(workerKey(target()))).toEqual({ status: "unknown", warning: "tmux-unavailable" });
  });

  it("a missing tmux server is a healthy empty snapshot", async () => {
    const result = await inspectWorkers([target()], { ...deps({}), tmux: async () => { throw new Error("no server running on /tmp/tmux-1000/default"); } });
    expect(result.get(workerKey(target()))).toEqual({ status: "awaiting-confirmation" });
  });

  it("a timeout produces unknown with a last-good timestamp", async () => {
    const t = target();
    const cache = new Map<string, { at: number; observedAt: string }>();
    await observe({ 100: wrapperNode() }, t, { cache });
    const timedOut: ProcReader = {
      ...procTree({ 100: wrapperNode() }),
      async stat() { throw new BudgetError("timeout"); },
    };
    const result = await inspectWorkers([t], { ...deps({}), cache, proc: timedOut, tmux: async () => `%1 100 ${t.session}\n` });
    expect(result.get(workerKey(t))).toMatchObject({ status: "unknown", warning: "observation-timeout" });
  });

  it("marks a run without a recorded session or manifest unverified", async () => {
    const noSession = await observe({ 100: wrapperNode() }, target({ session: null }));
    expect(noSession.get(workerKey(target({ session: null })))).toEqual({ status: "unknown", warning: "session-unknown" });
    const noManifest = await observe({ 100: wrapperNode() }, target({ manifestPath: null }));
    expect(noManifest.get(workerKey(target({ manifestPath: null })))).toEqual({ status: "unknown", warning: "manifest-unverified" });
  });
});

describe("applyCurrentOverlay", () => {
  function row(overrides: Partial<ExecutionRow> = {}): ExecutionRow {
    return {
      destination: DEST, runId: RUN, latestEventId: 2, startEventId: 1, finishEventId: null,
      role: "builder", kind: "build", runtime: "codex", model: null, caller: null, profile: null,
      phase: null, target: null, firstDispatchedAt: "2026-11-18T10:00:00.000Z",
      attemptStartedAt: "2026-11-18T10:00:00.000Z", finishedAt: null, outcome: "open", terminal: false,
      contractStatus: null, durationMs: null, warnings: [], ...overrides,
    };
  }

  it("overlays a terminal observed after the frozen page without moving the row", () => {
    const current: CurrentExecutionState = {
      destination: DEST, runId: RUN, startEventId: 1, finishEventId: 9,
      attemptStartedAt: "2026-11-18T10:00:00.000Z", attemptSession: null,
      row: row({ terminal: true, outcome: "failure", contractStatus: "ok", finishEventId: 9, finishedAt: "2026-11-18T10:05:00.000Z", durationMs: 300_000 }),
    };
    const applied = applyCurrentOverlay(row(), current);
    expect(applied.row.outcome).toBe("failure");
    expect(applied.row.finishEventId).toBe(9);
    expect(applied.workerSuppressed).toBe(true);
    expect(applied.refreshNeeded).toBe(false);
  });

  it("a newer start makes the displayed attempt refresh-needed and suppresses worker labels", () => {
    const current: CurrentExecutionState = {
      destination: DEST, runId: RUN, startEventId: 5, finishEventId: null,
      attemptStartedAt: "2026-11-18T11:00:00.000Z", attemptSession: null,
      row: row({ startEventId: 5, latestEventId: 5 }),
    };
    const applied = applyCurrentOverlay(row(), current);
    expect(applied.refreshNeeded).toBe(true);
    expect(applied.workerSuppressed).toBe(true);
    expect(applied.row).toStrictEqual(row());
  });

  it("leaves a known terminal row alone and ignores a stale open observation", () => {
    const terminal = row({ terminal: true, outcome: "success", finishEventId: 3 });
    const current: CurrentExecutionState = {
      destination: DEST, runId: RUN, startEventId: 1, finishEventId: 3, attemptStartedAt: null, attemptSession: null,
      row: row({ terminal: true, outcome: "success", finishEventId: 3 }),
    };
    expect(applyCurrentOverlay(terminal, current)).toMatchObject({ refreshNeeded: false, workerSuppressed: false });
  });
});
