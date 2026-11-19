import { describe, expect, it, vi } from "vitest";
import { createEnvironmentRuntime, type EnvironmentDeps } from "./environment";
import {
  attributeOwners, buildWorkloadGroups, collectWorkloads, createIdentityReader, executableFamily,
  identityConcurrency, IDENTITY_BUDGET, orderIdentityPids, parseProcStat, parsePs, readIdentityPhase,
  MAX_GROUP_DETAILS, MAX_GROUPS, MAX_PHASE_READS,
} from "./workloads";

function statText(pid: number, ppid: number, start: number, comm = "proc"): string {
  return `${pid} (${comm}) S ${ppid} ${"0 ".repeat(17)}${start}`;
}

type ChainEntry = { ppid: number; start: number; comm?: string };

function chain(entries: Record<number, ChainEntry>, afterStart: Record<number, number> = {}) {
  const calls = new Map<number, number>();
  const read = vi.fn(async (pid: number) => {
    const entry = entries[pid];
    if (!entry) throw new Error("ENOENT");
    const n = (calls.get(pid) ?? 0) + 1;
    calls.set(pid, n);
    const start = n > 1 && afterStart[pid] !== undefined ? afterStart[pid] : entry.start;
    return statText(pid, entry.ppid, start, entry.comm);
  });
  return read;
}

const PANE = new Map([["%1", 20]]);
const LIVE = {
  paneIds: new Set(["%1"]), incarnation: "10:20", panePids: new Map([["%1", 20]]),
  paneDead: new Map([["%1", false]]),
};

const PS_LINES = [
  "  1 0 1000 Ss 100 100 init",
  " 10 1 1000 Ss 100 100 systemd",
  " 20 10 1000 Sl 100 100 tmux: server",
  " 30 20 1000 S+ 10 1024 codex --full",
  " 31 30 1000 Z 10 0 codex --full",
  " 40 1 1000 Sl 100 512 chrome",
  " 41 40 1000 Sl 100 512 Google Chrome Helper (Renderer)",
].join("\n");

const STABLE: Record<number, ChainEntry> = {
  1: { ppid: 0, start: 1, comm: "init" },
  10: { ppid: 1, start: 2, comm: "systemd" },
  20: { ppid: 10, start: 3, comm: "tmux: server" },
  30: { ppid: 20, start: 4, comm: "codex --full" },
  31: { ppid: 30, start: 5, comm: "codex --full" },
  40: { ppid: 1, start: 6, comm: "chrome" },
  41: { ppid: 40, start: 7, comm: "Google Chrome Helper (Renderer)" },
};

function environmentDeps(over: Partial<EnvironmentDeps> = {}): EnvironmentDeps {
  const at = new Date(1).toISOString();
  const child = { ok: true as const, observedAt: at, lastGoodAt: at };
  return {
    now: () => 1,
    sleep: () => new Promise<void>(() => {}),
    readSettingsRaw: () => ({ ok: true, raw: undefined, file: "absent" }),
    ownDb: async () => ({ ok: true, schemaVersion: 15, latestMetricTs: null, paneContext: null }),
    checkProjects: async () => ({ ok: true }),
    listPanes: async () => ({
      paneIds: new Set(), incarnation: null, paneCommands: new Map(), panePids: new Map(), paneDead: new Map(),
    }),
    ttydTransport: () => ({ ro: null, rw: null }),
    httpOk: async () => true,
    cli: async () => "missing",
    linearCredential: () => false,
    envFile: () => "ok",
    readMountInfo: async () => "22 1 8:1 / / rw - ext4 /dev/sda1 rw",
    statfs: async () => ({ bsize: 4096, blocks: 100, bfree: 25, bavail: 20 }),
    realpath: () => "/tmp",
    packages: async () => ({
      rows: [], holds: [], simulation: [], nativeArch: "amd64", state: "ok",
      observedAt: at, lastGoodAt: at, stale: false, error: null, warnings: [],
      children: { candidates: child, holds: child, simulation: child, architecture: child },
    }),
    ps: async () => "",
    readProc: async () => statText(1, 0, 1),
    ...over,
  };
}

describe("parsePs (F9)", () => {
  it("takes the command as the remainder after six fields, preserving spaces", () => {
    const snap = parsePs("  30 20 1000 S+ 10 512 Google Chrome Helper (Renderer)\n");
    expect(snap.processes).toEqual([{
      pid: 30, ppid: 20, uid: 1000, stat: "S+", ageSeconds: 10, rssKb: 512,
      comm: "Google Chrome Helper (Renderer)",
    }]);
    expect(snap.total).toBe(1);
    expect(snap.partial).toBe(false);
  });

  it("marks malformed output partial instead of dropping it silently", () => {
    const snap = parsePs("PID PPID UID STAT ETIMES RSS COMMAND\nnot-a-row\n30 20 1000 S+ 10 512 node\n");
    expect(snap.processes.map((p) => p.pid)).toEqual([30]);
    expect(snap.partial).toBe(true);
  });

  it("keeps an invalid RSS explicitly non-finite rather than inventing zero", () => {
    const snap = parsePs("  30 20 1000 S+ 10 - node\n");
    expect(snap.processes[0].rssKb).toBeNaN();
  });
});

describe("parseProcStat (F9)", () => {
  it("uses the final ')' as the command boundary, even with spaces and parentheses", () => {
    expect(parseProcStat(statText(42, 7, 1000, "chrome (renderer) helper"))).toEqual({
      pid: 42, comm: "chrome (renderer) helper", state: "S", ppid: 7, startTime: 1000,
    });
  });

  it("rejects malformed content", () => {
    expect(parseProcStat("no parens")).toBeNull();
    expect(parseProcStat("42 (x) S 7")).toBeNull();
  });
});

describe("proven ancestry (AC6)", () => {
  it("proves a complete stable chain against the fresh pane incarnation", async () => {
    const readProc = chain(STABLE);
    const result = await collectWorkloads({
      nowMs: 1, psOutput: PS_LINES, panePids: PANE, readProc,
      paneContext: () => null, incarnation: "10:20", paneDead: new Map([["%1", false]]),
      verifyPanes: async () => LIVE, monotonic: () => 0,
    });
    const codex = result.groups.find((g) => g.family === "codex");
    expect(codex?.owner).toEqual({ kind: "pane", pane: "%1" });
    expect(codex).toMatchObject({ count: 2, running: 1, zombies: 1 });
    expect(result.groups.find((g) => g.family === "chrome")?.owner.kind).toBe("unknown");
    expect(result.coverage.ownershipKnown).toBe(3); // pane itself + codex + its zombie
    expect(result.state).toBe("ok");
  });

  it("requires an explicit alive fact in both observations before proving an owner", async () => {
    const snapshot = parsePs(PS_LINES);
    const pids = orderIdentityPids(snapshot, PANE);
    const before = await readIdentityPhase(pids, createIdentityReader(chain(STABLE), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
    const after = await readIdentityPhase(pids, createIdentityReader(chain(STABLE), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
    const withoutInitial = attributeOwners({
      snapshot, panePids: PANE, paneIncarnation: "10:20", before, after, live: LIVE, partial: false,
    });
    expect(withoutInitial.ownerByPid.has(30)).toBe(false);
    expect(withoutInitial.reasonByPid.get(30)).toBe("race");

    const withBoth = attributeOwners({
      snapshot, panePids: PANE, paneIncarnation: "10:20", before, after, live: LIVE,
      paneDead: new Map([["%1", false]]), partial: false,
    });
    expect(withBoth.ownerByPid.get(30)).toBe("%1");
  });

  it("changes the initial leaf identity between phases and drops the owner", async () => {
    const snapshot = parsePs(PS_LINES);
    const before = await readIdentityPhase(orderIdentityPids(snapshot, PANE), createIdentityReader(chain(STABLE), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
    const after = await readIdentityPhase(
      orderIdentityPids(snapshot, PANE),
      createIdentityReader(chain({ ...STABLE, 30: { ...STABLE[30], start: 999 } }), { now: () => 0 }).read,
      { deadline: 1e9, now: () => 0 },
    );
    const attribution = attributeOwners({
      snapshot, panePids: PANE, paneIncarnation: "10:20", before, after, live: LIVE, partial: false,
    });
    expect(attribution.ownerByPid.has(30)).toBe(false);
    expect(attribution.reasonByPid.get(30)).toBe("race");
    expect(attribution.partial).toBe(true);
  });

  it("treats a changed ancestor start tick or PPID as a race", async () => {
    const snapshot = parsePs(PS_LINES);
    const pids = orderIdentityPids(snapshot, PANE);
    const before = await readIdentityPhase(pids, createIdentityReader(chain(STABLE), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
    const after = await readIdentityPhase(
      pids, createIdentityReader(chain({ ...STABLE, 20: { ...STABLE[20], start: 404 } }), { now: () => 0 }).read,
      { deadline: 1e9, now: () => 0 },
    );
    const attribution = attributeOwners({
      snapshot, panePids: PANE, paneIncarnation: "10:20", before, after, live: LIVE, partial: false,
    });
    expect(attribution.ownerByPid.has(30)).toBe(false);
    expect(attribution.ownerByPid.has(20)).toBe(false);
  });

  it("never inherits an owner from a missing node, a loop or a cross-UID chain", async () => {
    const missing = { ...STABLE };
    delete (missing as Record<number, ChainEntry>)[20];
    const missingSnapshot = parsePs(PS_LINES);
    const missingPids = orderIdentityPids(missingSnapshot, PANE);
    const missingResult = await collectWorkloads({
      nowMs: 1, psOutput: PS_LINES, panePids: PANE, readProc: chain(missing),
      paneContext: () => null, incarnation: "10:20", verifyPanes: async () => LIVE, monotonic: () => 0,
    });
    expect(missingResult.groups.find((g) => g.family === "codex")?.owner.kind).toBe("unknown");
    expect(missingResult.coverage.partial).toBe(true);

    const loopSnapshot = parsePs([" 50 60 1000 S 1 1 a", " 60 50 1000 S 1 1 b", " 20 1 1000 S 1 1 pane"].join("\n"));
    const loopPids = orderIdentityPids(loopSnapshot, new Map([["%1", 20]]));
    const loopBefore = await readIdentityPhase(loopPids, createIdentityReader(chain({
      20: { ppid: 1, start: 1 }, 50: { ppid: 60, start: 2 }, 60: { ppid: 50, start: 3 },
    }), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
    const loopAfter = await readIdentityPhase(loopPids, createIdentityReader(chain({
      20: { ppid: 1, start: 1 }, 50: { ppid: 60, start: 2 }, 60: { ppid: 50, start: 3 },
    }), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
    const loop = attributeOwners({
      snapshot: loopSnapshot, panePids: new Map([["%1", 20]]), paneIncarnation: "10:20",
      before: loopBefore, after: loopAfter, live: LIVE, partial: false,
    });
    expect(loop.ownerByPid.has(50)).toBe(false);
    expect(loop.reasonByPid.get(50)).toBe("cycle");

    const crossUid = parsePs([" 50 20 0 S 1 1 sudo", " 20 1 1000 S 1 1 pane"].join("\n"));
    const crossStable = { 20: { ppid: 1, start: 1 }, 50: { ppid: 20, start: 2 } };
    const crossPids = orderIdentityPids(crossUid, new Map([["%1", 20]]));
    const crossBefore = await readIdentityPhase(crossPids, createIdentityReader(chain(crossStable), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
    const crossAfter = await readIdentityPhase(crossPids, createIdentityReader(chain(crossStable), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
    const cross = attributeOwners({
      snapshot: crossUid, panePids: new Map([["%1", 20]]), paneIncarnation: "10:20",
      before: crossBefore, after: crossAfter, live: LIVE, partial: false,
    });
    expect(cross.ownerByPid.has(50)).toBe(false);
    expect(cross.reasonByPid.get(50)).toBe("cross-uid");
  });

  it("an explicit dead pane cannot establish a live owner even with a stable recycled PID", async () => {
    const result = await collectWorkloads({
      nowMs: 1, psOutput: PS_LINES, panePids: PANE, readProc: chain(STABLE),
      paneContext: () => null, incarnation: "10:20",
      verifyPanes: async () => ({ ...LIVE, paneDead: new Map([["%1", true]]) }), monotonic: () => 0,
    });
    expect(result.groups.find((g) => g.family === "codex")?.owner.kind).toBe("unknown");
    expect(result.coverage.partial).toBe(true);
  });

  it("a changed/closed pane incarnation yields unknown, never another owner", async () => {
    const readProc = chain(STABLE);
    const result = await collectWorkloads({
      nowMs: 1, psOutput: PS_LINES, panePids: PANE, readProc,
      paneContext: () => null, incarnation: "10:20",
      verifyPanes: async () => ({ ...LIVE, incarnation: "10:21" }), monotonic: () => 0,
    });
    expect(result.groups.find((g) => g.family === "codex")?.owner.kind).toBe("unknown");
    expect(result.coverage.partial).toBe(true);
  });

  it("keeps ps data with unknown owners and a tmux warning when tmux fails", async () => {
    const result = await collectWorkloads({
      nowMs: 1, psOutput: PS_LINES, panePids: new Map(), readProc: chain(STABLE),
      paneContext: () => null, incarnation: null, tmuxWarning: "tmux-unavailable",
      verifyPanes: async () => { throw new Error("no tmux"); }, monotonic: () => 0,
    });
    expect(result.groups.length).toBeGreaterThan(0);
    expect(result.groups.every((g) => g.owner.kind === "unknown")).toBe(true);
    expect(result.warning).toBe("tmux-unavailable");
    expect(result.state).toBe("partial");
  });
});

describe("one before/after identity budget (M5 Task 3)", () => {
  it("caps total reads at 2048, phase reads at 1024 and concurrency at 8", async () => {
    const lines = ["  1 0 1000 Ss 1 1 init"];
    for (let i = 0; i < 1_100; i += 1) lines.push(` ${100 + i} 1 1000 S 1 1 family${i}`);
    const snapshot = parsePs(lines.join("\n"));
    const ordered = orderIdentityPids(snapshot, new Map());
    expect(ordered).toHaveLength(MAX_PHASE_READS);

    let active = 0;
    let maxActive = 0;
    let total = 0;
    const read = vi.fn(async (pid: number) => {
      total += 1;
      active += 1;
      maxActive = Math.max(maxActive, active);
      await new Promise((resolve) => setTimeout(resolve, 0));
      active -= 1;
      return statText(pid, 1, 1);
    });
    const result = await collectWorkloads({
      nowMs: 1, psOutput: lines.join("\n"), panePids: new Map(), readProc: read,
      paneContext: () => null, incarnation: null, monotonic: () => 0,
    });
    expect(total).toBeLessThanOrEqual(2_048);
    expect(total).toBe(2_048);
    expect(maxActive).toBeLessThanOrEqual(8);
    expect(result.coverage.partial).toBe(true);
    expect(result.coverage.reasons.budget).toBeGreaterThan(0);
  });

  it("stops new identity scheduling at the shared five-second deadline", async () => {
    const lines = [" 20 1 1000 S 1 1 pane", " 30 20 1000 S 1 1 codex"];
    for (let i = 0; i < 50; i += 1) lines.push(` ${100 + i} 1 1000 S 1 1 other${i}`);
    let clock = 0;
    const read = async (pid: number) => {
      clock += 2_000;
      return statText(pid, 1, 1);
    };
    const result = await collectWorkloads({
      nowMs: 1, psOutput: lines.join("\n"), panePids: new Map([["%1", 20]]), readProc: read,
      paneContext: () => null, incarnation: "10:20", monotonic: () => clock,
    });
    expect(result.coverage.partial).toBe(true);
    expect(result.coverage.reasons.budget).toBeGreaterThan(0);
  });
});

describe("raw read ownership across caller timeouts (observation completion Task 3)", () => {
  it("derives the libuv scheduling cap from the startup pool size, reserving one slot", () => {
    expect(identityConcurrency({})).toBe(2);
    expect(identityConcurrency({ UV_THREADPOOL_SIZE: "4" })).toBe(2);
    expect(identityConcurrency({ UV_THREADPOOL_SIZE: "2" })).toBe(1);
    expect(identityConcurrency({ UV_THREADPOOL_SIZE: "1" })).toBe(0);
    expect(identityConcurrency({ UV_THREADPOOL_SIZE: "0" })).toBe(0);
    expect(identityConcurrency({ UV_THREADPOOL_SIZE: "many" })).toBe(0);
  });

  it("admits no identity read at all with a zero/one/invalid startup pool", async () => {
    const read = vi.fn(async () => statText(1, 0, 1));
    const reader = createIdentityReader(read, {
      concurrency: identityConcurrency({ UV_THREADPOOL_SIZE: "1" }), now: () => 0,
    });
    await expect(reader.read(1, 1e9)).resolves.toBe(IDENTITY_BUDGET);
    expect(read).not.toHaveBeenCalled();
  });

  it("applies the real startup pool size in the default factory", async () => {
    vi.stubEnv("UV_THREADPOOL_SIZE", "1");
    try {
      let starts = 0;
      const reader = createIdentityReader(async () => { starts += 1; return ""; }, { now: () => 0 });
      await expect(reader.read(101, 5_000)).resolves.toBe(IDENTITY_BUDGET);
      expect(starts).toBe(0);
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it("carries budget facts from the runtime default factory without a proc read at pool 1", async () => {
    vi.stubEnv("UV_THREADPOOL_SIZE", "1");
    try {
      const readProc = vi.fn(async () => statText(1, 0, 1));
      const runtime = createEnvironmentRuntime(environmentDeps({
        ps: async () => " 10 1 1000 S 1 1 tmux\n 100 10 1000 R 1 1 codex",
        readProc,
      }));
      const result = await runtime.workloads();
      expect(readProc).not.toHaveBeenCalled();
      expect(result.coverage.reasons.budget).toBe(2);
      expect(result.groups.every((g) => g.owner.kind === "unknown" && g.owner.reason === "budget")).toBe(true);
    } finally {
      vi.unstubAllEnvs();
    }
  });

  it("defaults to two originals absent the pool variable and honors a configured pool", async () => {
    vi.useFakeTimers();
    const startsWith = async (pool: string | undefined): Promise<number> => {
      vi.stubEnv("UV_THREADPOOL_SIZE", pool);
      try {
        let starts = 0;
        const read = vi.fn(() => {
          starts += 1;
          return new Promise<string>(() => {});
        });
        const reader = createIdentityReader(read, { now: () => 0, perReadMs: 100 });
        const pending = [reader.read(1, 5_000), reader.read(2, 5_000), reader.read(3, 5_000)];
        await vi.advanceTimersByTimeAsync(100);
        const outcomes = await Promise.all(pending);
        expect(outcomes.every((outcome) => outcome === IDENTITY_BUDGET)).toBe(true);
        return starts;
      } finally {
        vi.unstubAllEnvs();
      }
    };
    try {
      expect(await startsWith(undefined)).toBe(2);
      expect(await startsWith("2")).toBe(1);
      expect(await startsWith("0")).toBe(0);
      expect(await startsWith("many")).toBe(0);
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps settled malformed, empty or rejected identities missing instead of budget", async () => {
    const cases: (() => Promise<string>)[] = [
      async () => "malformed",
      async () => "",
      async () => { throw new Error("ENOENT"); },
    ];
    for (const read of cases) {
      const reader = createIdentityReader(read, { concurrency: 1, now: () => 0 });
      await expect(reader.read(101, 5_000)).resolves.toBeNull();
    }
  });

  it("reuses an expired original without a new attempt and reports its budget", async () => {
    vi.useFakeTimers();
    let clock = 0;
    let release!: (text: string) => void;
    const read = vi.fn(() => new Promise<string>((resolve) => { release = resolve; }));
    try {
      const reader = createIdentityReader(read, { concurrency: 1, now: () => clock, perReadMs: 100 });
      const first = reader.read(101, 5_000);
      await vi.advanceTimersByTimeAsync(100);
      await expect(first).resolves.toBe(IDENTITY_BUDGET);
      expect(read).toHaveBeenCalledTimes(1);
      clock = 100;
      await expect(reader.read(101, 5_000)).resolves.toBe(IDENTITY_BUDGET);
      expect(read).toHaveBeenCalledTimes(1);
      const phase = await readIdentityPhase([101], reader.read, { deadline: 10_000, now: () => clock });
      expect(phase.attempted.has(101)).toBe(false);
      expect(phase.stats.size).toBe(0);
      release(statText(101, 1, 1));
      await vi.advanceTimersByTimeAsync(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("keeps one original read per pending PID and never starts another after the deadline", async () => {
    vi.useFakeTimers();
    let clock = 0;
    let starts = 0;
    let active = 0;
    let maxActive = 0;
    let hanging = true;
    let timedOut = false;
    const startsByPid = new Map<number, number>();
    const deferred: (() => void)[] = [];
    let first: Promise<unknown> | null = null;
    let second: Promise<unknown> | null = null;
    const readProc = async (pid: number) => {
      starts += 1;
      active += 1;
      maxActive = Math.max(maxActive, active);
      startsByPid.set(pid, (startsByPid.get(pid) ?? 0) + 1);
      if (!hanging) {
        active -= 1;
        return statText(pid, pid === 10 ? 1 : 10, 1);
      }
      return new Promise<string>((resolve) => {
        deferred.push(() => {
          active -= 1;
          resolve(statText(pid, pid === 10 ? 1 : 10, 1));
        });
      });
    };
    try {
      const psOutput = ["10 1 1000 S 1 1 tmux", ...Array.from({ length: 24 }, (_, i) => `${1000 + i} 10 1000 R 1 1 helper`)].join("\n");
      const identityReader = createIdentityReader(readProc, { now: () => clock });
      const input = {
        nowMs: 0, psOutput, panePids: new Map([["%1", 10]]), paneContext: () => null,
        incarnation: "10:20",
        verifyPanes: async () => ({ paneIds: new Set(["%1"]), incarnation: "10:20", panePids: new Map([["%1", 10]]) }),
        monotonic: () => clock, readProc, identityReader,
      };
      first = collectWorkloads(input);
      await Promise.resolve();
      await Promise.resolve();
      expect(maxActive).toBeLessThanOrEqual(2); // default libuv scheduling cap
      expect(starts).toBe(2);
      expect(startsByPid.get(10)).toBe(1);

      second = collectWorkloads(input); // overlapping poll reuses the pending originals
      await Promise.resolve();
      expect(startsByPid.get(10)).toBe(1);

      clock += 2_500;
      await vi.advanceTimersByTimeAsync(2_500);
      timedOut = true;
      const afterTimeouts = starts;
      expect(afterTimeouts).toBe(2); // a public timeout does not release the raw permit
      clock += 2_600; // now past the shared five-second scheduling deadline
      await vi.advanceTimersByTimeAsync(2_600);
      expect(starts).toBe(afterTimeouts); // no new original read after the deadline
      expect(starts).toBeLessThanOrEqual(8); // hard ceiling of unsettled originals
    } finally {
      hanging = false;
      for (const release of deferred.splice(0)) release();
      if (first) await first.catch(() => {});
      if (second) await second.catch(() => {});
      vi.useRealTimers();
    }
    expect(timedOut).toBe(true);
    expect(maxActive).toBeLessThanOrEqual(8);
    expect(active).toBe(0);
    const result = await first as Awaited<ReturnType<typeof collectWorkloads>>;
    expect(result.coverage.partial).toBe(true);
    expect(result.coverage.reasons.budget).toBeGreaterThan(0);
  });
});

describe("grouping and lifecycle (AC5)", () => {
  it("maps fixed browser/helper families, keeps zombies and last-known context", async () => {
    const snapshot = parsePs(PS_LINES);
    const attribution = await (async () => {
      const pids = orderIdentityPids(snapshot, PANE);
      const before = await readIdentityPhase(pids, createIdentityReader(chain(STABLE), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
      const after = await readIdentityPhase(pids, createIdentityReader(chain(STABLE), { now: () => 0 }).read, { deadline: 1e9, now: () => 0 });
      return attributeOwners({
        snapshot, panePids: PANE, paneIncarnation: "10:20", before, after, live: LIVE,
        paneDead: new Map([["%1", false]]), partial: false,
      });
    })();
    const result = buildWorkloadGroups({
      nowMs: Date.parse("2026-11-19T00:00:00.000Z"),
      snapshot,
      attribution,
      contexts: new Map([["%1", { project: "jax-os", role: "lead", session: "s1", observedAt: "2026-11-18T23:00:00.000Z" }]]),
    });
    const codex = result.groups.find((g) => g.family === "codex");
    expect(codex).toMatchObject({
      count: 2, running: 1, zombies: 1, project: "jax-os", session: "s1",
      contextObservedAt: "2026-11-18T23:00:00.000Z", purpose: "tmux-pane",
      owner: { kind: "pane", pane: "%1" },
    });
    const zombie = codex?.pids.find((p) => p.state.startsWith("Z"));
    expect(zombie).toMatchObject({ ppid: 30, parentCommand: "codex --full" });
    const chrome = result.groups.filter((g) => g.family === "chrome");
    expect(chrome.length).toBeGreaterThan(0);
    expect(chrome.every((g) => g.zombies === 0)).toBe(true);
    expect(chrome.every((g) => g.project === null && g.session === null)).toBe(true);
    expect(executableFamily("Google Chrome Helper (Renderer)")).toBe("chrome");
    expect(executableFamily("very-long-family-name".repeat(8)).length).toBeLessThanOrEqual(64);
  });

  it("keeps unknown origin explicit with a reason and an approximate RSS only where valid", async () => {
    const snapshot = parsePs([" 40 1 1000 Sl 1 - chrome", " 41 40 1000 Sl 1 2048 chrome"].join("\n"));
    const attribution = { ownerByPid: new Map(), rootByPid: new Map([[40, 1], [41, 1]]), reasonByPid: new Map<number, "no-pane">([[40, "no-pane"], [41, "no-pane"]]), partial: false };
    const result = buildWorkloadGroups({ nowMs: 1, snapshot, attribution, contexts: new Map() });
    expect(result.groups.every((g) => g.owner.kind === "unknown" && g.purpose === null)).toBe(true);
    expect(result.groups[0].rssMb).toBe(2);
    expect(result.groups[0].pids.find((p) => p.pid === 40)?.rssMb).toBeNull();
    expect(result.coverage.reasons["no-pane"]).toBe(2);
  });

  it("caps groups/details deterministically and reports truncation", () => {
    const lines = ["  1 0 1000 Ss 1 1 init"];
    for (let i = 1; i <= MAX_GROUPS + 5; i += 1) {
      lines.push(` ${100 + i} 1 1000 S 1 ${i} family${i}`);
    }
    lines.push(" 999 1 1000 S 1 1 big");
    for (let i = 0; i < MAX_GROUP_DETAILS + 2; i += 1) lines.push(` ${1000 + i} 999 1000 S 1 100000 big`);
    const snap = parsePs(lines.join("\n"));
    const roots = new Map(snap.processes.map((p) => [p.pid, 1]));
    const result = buildWorkloadGroups({
      nowMs: 1, snapshot: snap,
      attribution: { ownerByPid: new Map(), rootByPid: roots, reasonByPid: new Map(), partial: false },
      contexts: new Map(),
    });
    expect(result.groups.length).toBe(MAX_GROUPS);
    expect(result.truncated).toBe(true);
    expect(result.groups[0].family).toBe("big");
    expect(result.groups[0].pids).toHaveLength(MAX_GROUP_DETAILS);
    expect(result.groups[0].detailsTruncated).toBe(true);
  });
});
