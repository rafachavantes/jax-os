import { describe, expect, it, vi } from "vitest";
import {
  attributeOwners, buildWorkloadGroups, collectWorkloads, executableFamily, parseProcStat, parsePs,
  readSnapshotStats, stabilizeOwners, MAX_GROUP_DETAILS, MAX_GROUPS,
} from "./workloads";

function statText(pid: number, ppid: number, start: number, comm = "proc"): string {
  return `${pid} (${comm}) S ${ppid} ${"0 ".repeat(17)}${start}`;
}

const read = (entries: Record<number, string>) => vi.fn(async (pid: number) => {
  const text = entries[pid];
  if (text === undefined) throw new Error("ENOENT");
  return text;
});

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
  const snapshot = parsePs([
    "  1 0 1000 Ss 100 100 init",
    " 10 1 1000 Ss 100 100 systemd",
    " 20 10 1000 Sl 100 100 tmux: server",
    " 30 20 1000 S+ 100 100 codex",
    " 40 1 1000 S 100 100 chrome",
    " 99 0 0 Z 100 0 zombie",
  ].join("\n"));

  it("attributes a process to a pane only when the observed chain reaches its PID", async () => {
    const { stats } = await readSnapshotStats(snapshot.processes, read({
      1: statText(1, 0, 1), 10: statText(10, 1, 2), 20: statText(20, 10, 3),
      30: statText(30, 20, 4), 40: statText(40, 1, 5), 99: statText(99, 0, 6),
    }));
    const attr = attributeOwners(snapshot, new Map([["%1", 20]]), stats, false);
    expect(attr.ownerByPid.get(30)).toBe("%1");
    expect(attr.ownerByPid.has(40)).toBe(false);
    expect(attr.ownerByPid.has(20)).toBe(true); // the pane process itself is owned by its pane
  });

  it("never inherits an owner from a missing node, a loop or a cross-UID chain", async () => {
    const { stats } = await readSnapshotStats(snapshot.processes, read({
      1: statText(1, 0, 1), 10: statText(10, 1, 2), 20: statText(20, 10, 3),
      30: statText(30, 20, 4), 99: statText(99, 0, 6),
      // 40 is missing entirely
    }));
    const attr = attributeOwners(snapshot, new Map([["%1", 20]]), stats, true);
    expect(attr.ownerByPid.get(30)).toBe("%1");
    expect(attr.ownerByPid.has(40)).toBe(false);
    expect(attr.partial).toBe(true);

    const crossUid = parsePs([" 50 20 0 S 1 1 sudo", " 20 1 1000 S 1 1 pane"].join("\n"));
    const cross = await readSnapshotStats(crossUid.processes, read({
      20: statText(20, 1, 1), 50: statText(50, 20, 2),
    }));
    expect(attributeOwners(crossUid, new Map([["%1", 20]]), cross.stats, false).ownerByPid.has(50)).toBe(false);
  });

  it("a changed PPID or a recycled start tick drops the owner on the stabilizing re-read", async () => {
    const { stats } = await readSnapshotStats(snapshot.processes, read({
      1: statText(1, 0, 1), 10: statText(10, 1, 2), 20: statText(20, 10, 3), 30: statText(30, 20, 4),
    }));
    const attr = attributeOwners(snapshot, new Map([["%1", 20]]), stats, false);
    expect(attr.ownerByPid.get(30)).toBe("%1");

    const seen = new Map<number, number>();
    const recycled = vi.fn(async (pid: number) => {
      const n = (seen.get(pid) ?? 0) + 1;
      seen.set(pid, n);
      // first pass (before) keeps start=4, second pass (after) sees a recycled PID
      return pid === 30 && n > 1 ? statText(30, 20, 999) : statText(pid, pid === 20 ? 10 : 20, pid === 20 ? 3 : 4);
    });
    await stabilizeOwners(attr, recycled);
    expect(attr.ownerByPid.has(30)).toBe(false);
  });
});

describe("grouping and lifecycle (AC5)", () => {
  const snapshot = parsePs([
    "  1 0 1000 Ss 100 100 init",
    " 20 1 1000 Sl 100 100 tmux: server",
    " 30 20 1000 S+ 10 1024 codex --full",
    " 31 30 1000 Z 10 0 codex --full",
    " 40 1 1000 Sl 100 512 chrome",
    " 41 40 1000 Sl 100 512 Google Chrome Helper (Renderer)",
  ].join("\n"));

  it("separates zombies from a running browser and keeps a proven pane's last-known context", async () => {
    const { stats } = await readSnapshotStats(snapshot.processes, read({
      1: statText(1, 0, 1), 20: statText(20, 1, 2), 30: statText(30, 20, 3),
      31: statText(31, 30, 4), 40: statText(40, 1, 5), 41: statText(41, 40, 6),
    }));
    const attr = attributeOwners(snapshot, new Map([["%1", 20]]), stats, false);
    const result = buildWorkloadGroups({
      nowMs: Date.parse("2026-11-19T00:00:00.000Z"),
      snapshot,
      attribution: attr,
      contexts: new Map([["%1", { project: "jax-os", role: "lead", session: "s1", observedAt: "2026-11-18T23:00:00.000Z" }]]),
    });
    const codex = result.groups.find((g) => g.family === "codex");
    expect(codex).toMatchObject({
      count: 2, running: 1, zombies: 1, project: "jax-os", session: "s1", purpose: "tmux-pane",
      owner: { kind: "pane", pane: "%1" },
    });
    const chrome = result.groups.filter((g) => g.family === "chrome" || g.family === "Google");
    expect(chrome.length).toBeGreaterThan(0);
    expect(chrome.every((g) => g.zombies === 0)).toBe(true);
    expect(chrome.every((g) => g.project === null && g.session === null)).toBe(true);
    expect(result.coverage.ownershipKnown).toBe(3); // init, pane, codex + zombie (same chain)
  });

  it("an unowned process keeps an explicit unknown origin — never a repo/purpose guess", async () => {
    const { stats } = await readSnapshotStats(snapshot.processes, read({
      1: statText(1, 0, 1), 20: statText(20, 1, 2), 30: statText(30, 20, 3),
      31: statText(31, 30, 4), 40: statText(40, 1, 5), 41: statText(41, 40, 6),
    }));
    const attr = attributeOwners(snapshot, new Map(), stats, false);
    const result = buildWorkloadGroups({ nowMs: 1, snapshot, attribution: attr, contexts: new Map() });
    expect(result.groups.every((g) => g.owner.kind === "unknown" && g.purpose === null)).toBe(true);
    expect(executableFamily("Google Chrome Helper (Renderer)")).toBe("Google");
  });

  it("caps groups/details deterministically and reports truncation", async () => {
    const lines = ["  1 0 1000 Ss 1 1 init"];
    for (let i = 1; i <= MAX_GROUPS + 5; i += 1) {
      lines.push(` ${100 + i} 1 1000 S 1 ${i} family${i}`);
    }
    lines.push(" 999 1 1000 S 1 1 big");
    for (let i = 0; i < MAX_GROUP_DETAILS + 2; i += 1) lines.push(` ${1000 + i} 999 1000 S 1 100000 big`);
    const snap = parsePs(lines.join("\n"));
    const stats = new Map(snap.processes.map((p) => [p.pid, { pid: p.pid, comm: p.comm, state: "S", ppid: p.ppid, startTime: 1 }]));
    const roots = new Map(snap.processes.map((p) => [p.pid, 1]));
    const result = buildWorkloadGroups({ nowMs: 1, snapshot: snap, attribution: { ownerByPid: new Map(), rootByPid: roots, reads: 0, partial: false }, contexts: new Map() });
    expect(result.groups.length).toBe(MAX_GROUPS);
    expect(result.truncated).toBe(true);
    expect(result.groups[0].family).toBe("big");
    expect(result.groups[0].pids).toHaveLength(MAX_GROUP_DETAILS);
    expect(result.groups[0].detailsTruncated).toBe(true);
  });
});

describe("collectWorkloads", () => {
  it("ties the snapshot to the live pane PID and never reads a real /proc", async () => {
    const result = await collectWorkloads({
      nowMs: 1,
      psOutput: [
        "  1 0 1000 Ss 100 100 init",
        " 20 1 1000 Sl 100 100 tmux: server",
        " 30 20 1000 S+ 100 100 codex",
      ].join("\n"),
      panePids: new Map([["%1", 20]]),
      readProc: read({ 1: statText(1, 0, 1), 20: statText(20, 1, 2), 30: statText(30, 20, 3) }),
      paneContext: () => null,
      incarnation: "10:20",
    });
    expect(result.groups.find((g) => g.family === "codex")?.owner).toEqual({ kind: "pane", pane: "%1" });
    expect(result.state).toBe("ok");
  });
});
