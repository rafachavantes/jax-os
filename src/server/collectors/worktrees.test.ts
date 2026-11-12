import { describe, expect, test } from "vitest";
import { parseWorktreePorcelain, collectRetainedWorktrees, retainedWorktreeSessions, type LedgerRow } from "./worktrees";

const PORCELAIN =
  "worktree /home/rafa/repos/demo\nHEAD " + "a".repeat(40) + "\nbranch refs/heads/main\n\n" +
  "worktree /home/rafa/repos/demo-feat-x\nHEAD " + "b".repeat(40) + "\nbranch refs/heads/feat/x\n\n" +
  "worktree /home/rafa/repos/demo-detached\nHEAD " + "c".repeat(40) + "\ndetached\n";

describe("parseWorktreePorcelain", () => {
  test("parses branch, control repo, and detached entries; empty input yields no entries", () => {
    const entries = parseWorktreePorcelain(PORCELAIN);
    expect(entries).toEqual([
      { path: "/home/rafa/repos/demo", branch: "main", detached: false },
      { path: "/home/rafa/repos/demo-feat-x", branch: "feat/x", detached: false },
      { path: "/home/rafa/repos/demo-detached", branch: null, detached: true },
    ]);
    expect(parseWorktreePorcelain("")).toEqual([]);
  });
});

function row(overrides: Partial<LedgerRow>): LedgerRow {
  return { id: 1, runId: "aaaabbbbcccc", kind: "build", branch: "feat/x", startedAt: "2026-09-01T00:00:00Z", status: "finished", finishedAt: "2026-09-01T00:00:00Z", ...overrides };
}

describe("collectRetainedWorktrees", () => {
  const opts = { now: "2026-09-19T00:00:00Z", thresholdDays: 7, controlRepoPath: "/home/rafa/repos/demo" };

  test("counts a registered, finished, old-enough build; excludes control repo, unowned, open, and unregistered branches", () => {
    expect(collectRetainedWorktrees(PORCELAIN, [row({})], opts)).toEqual({ ok: true, count: 1, oldestAgeDays: 18 });
    expect(collectRetainedWorktrees(PORCELAIN, [row({ branch: "main" })], opts)).toEqual({ ok: true, count: 0, oldestAgeDays: 0 });
    expect(collectRetainedWorktrees(PORCELAIN, [], opts)).toEqual({ ok: true, count: 0, oldestAgeDays: 0 });
    const open = collectRetainedWorktrees(PORCELAIN, [row({ status: "running", finishedAt: null })], opts);
    expect(open.ok && open.count).toBe(0);
    const gone = collectRetainedWorktrees(PORCELAIN, [row({ branch: "feat/gone" })], opts);  // already removed on disk
    expect(gone.ok && gone.count).toBe(0);
  });

  test("threshold boundary: exactly at threshold counts, one day under does not", () => {
    const atThreshold = collectRetainedWorktrees(PORCELAIN, [row({ finishedAt: "2026-09-12T00:00:00Z" })], opts);
    expect(atThreshold.ok && atThreshold.count).toBe(1);
    const underThreshold = collectRetainedWorktrees(PORCELAIN, [row({ finishedAt: "2026-09-13T00:00:00Z" })], opts);
    expect(underThreshold.ok && underThreshold.count).toBe(0);
  });

  // F6 (round 4): "latest builder attempt" is by ts, `id` breaking an exact tie -- the
  // same rule as Python's `_latest_builder_attempt` (`scripts/jaxflow_common.py`).
  test("picks the latest build per branch by (startedAt, id), not array order", () => {
    const older = row({ runId: "aaaa", id: 1, startedAt: "2026-09-01T00:00:00Z", finishedAt: "2026-09-01T00:00:00Z" });
    const newerById = row({ runId: "bbbb", id: 2, startedAt: "2026-09-01T00:00:00Z", finishedAt: "2026-09-18T00:00:00Z" }); // same ts, higher id, NOT old enough
    // Both share `startedAt` -- the tie MUST break on `id`, so the newer-by-id row (not
    // old enough) wins and the branch is NOT counted, whichever order they're given in.
    expect(collectRetainedWorktrees(PORCELAIN, [older, newerById], opts)).toEqual({ ok: true, count: 0, oldestAgeDays: 0 });
    expect(collectRetainedWorktrees(PORCELAIN, [newerById, older], opts)).toEqual({ ok: true, count: 0, oldestAgeDays: 0 });
  });
});

describe("retainedWorktreeSessions", () => {
  const opts = { now: "2026-09-19T00:00:00Z", thresholdDays: 7, controlRepoPath: "/home/rafa/repos/demo" };
  const TWO = "worktree /home/rafa/repos/demo\nHEAD " + "a".repeat(40) + "\nbranch refs/heads/main\n\n" +
    "worktree /home/rafa/repos/demo-feat-x\nHEAD " + "b".repeat(40) + "\nbranch refs/heads/feat/x\n\n" +
    "worktree /home/rafa/repos/demo-feat-y\nHEAD " + "c".repeat(40) + "\nbranch refs/heads/feat/y\n";
  const A = "aaaaaaaa-0000-4000-8000-00000000000a";

  test("attributes each retained build to its run's recorded session", () => {
    const rows = [row({ id: 1, runId: "r1", branch: "feat/x" }), row({ id: 2, runId: "r2", branch: "feat/y" })];
    expect(retainedWorktreeSessions(TWO, rows, [{ runId: "r1", session: A }, { runId: "r2", session: null }], opts))
      .toEqual([{ session: A, count: 1 }, { session: null, count: 1 }]);
  });

  test("groups a run id absent from runSessions under null", () => {
    const rows = [row({ id: 1, runId: "r1", branch: "feat/x" }), row({ id: 2, runId: "r2", branch: "feat/y" })];
    expect(retainedWorktreeSessions(TWO, rows, [{ runId: "r1", session: A }], opts))
      .toEqual([{ session: A, count: 1 }, { session: null, count: 1 }]);
  });

  test("counts two retained builds of one session together", () => {
    const rows = [row({ id: 1, runId: "r1", branch: "feat/x" }), row({ id: 2, runId: "r2", branch: "feat/y" })];
    expect(retainedWorktreeSessions(TWO, rows, [{ runId: "r1", session: A }, { runId: "r2", session: A }], opts))
      .toEqual([{ session: A, count: 2 }]);
  });
});
