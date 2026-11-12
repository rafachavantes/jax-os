import { join } from "node:path";
import { describe, expect, test, vi } from "vitest";
import { REPOS_ROOT } from "../../../../server/collectors/projects";
import { handleWorktreesGet, type WorktreesRouteDeps } from "./handler";

describe("handleWorktreesGet", () => {
  // F3 (round 2, MEDIUM): `dir` is relative ("demo") -- join REPOS_ROOT before use, query
  // the ledger by `dir` (mission.ts's own hubMap key), never the display `name`.
  const baseDeps = (probeWorktreePorcelain: WorktreesRouteDeps["probeWorktreePorcelain"]): WorktreesRouteDeps => ({
    getProjects: () => ({ projects: [{ dir: "demo", name: "Demo Display Name" } as never], skipped: 0, reposRoot: "" }),
    probeWorktreePorcelain, getWorktreeLedgerRows: () => [], getDb: () => ({} as never), now: () => "2026-09-19T00:00:00Z",
  });

  test("joins REPOS_ROOT + dir for the git probe, queries the ledger by dir (not name), and keys output by the relative dir", async () => {
    const probe = vi.fn(async () => ({ ok: true as const, stdout: "worktree " + join(REPOS_ROOT, "demo") + "\nHEAD " + "a".repeat(40) + "\nbranch refs/heads/main\n" }));
    const getWorktreeLedgerRows = vi.fn(() => []);
    // Plan deviation: `collectorResponse` returns a NextResponse, so the assertions go
    // through `.json()` exactly like prs/route.test.ts -- the plan's draft treated the
    // return value as a plain {ok,data} envelope.
    const res = await handleWorktreesGet({ ...baseDeps(probe), getWorktreeLedgerRows });
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(probe).toHaveBeenCalledWith(join(REPOS_ROOT, "demo"));
    expect(getWorktreeLedgerRows).toHaveBeenCalledWith(expect.anything(), "demo");
    expect(Object.keys(body.data)).toEqual(["demo"]);
    const failedRes = await handleWorktreesGet(baseDeps(async () => ({ ok: false, error: "spawn failed" })));
    const failedBody = await failedRes.json();
    expect(failedBody.data["demo"]).toEqual({ ok: false, error: "spawn failed" });
  });

  test("attaches the per-session breakdown when the dep is present, and only on success", async () => {
    const ledgerRow = { id: 1, runId: "r1", kind: "build" as const, branch: "feat/x", startedAt: "2026-09-01T00:00:00Z", status: "finished" as const, finishedAt: "2026-09-01T00:00:00Z" };
    const stdout = "worktree " + join(REPOS_ROOT, "demo") + "\nHEAD " + "a".repeat(40) + "\nbranch refs/heads/main\n\n" +
      "worktree " + join(REPOS_ROOT, "demo-feat-x") + "\nHEAD " + "b".repeat(40) + "\nbranch refs/heads/feat/x\n";
    const getBuildRunSessions = vi.fn(() => [{ runId: "r1", session: "S1" }]);
    const body = await (await handleWorktreesGet({
      ...baseDeps(async () => ({ ok: true as const, stdout })),
      getWorktreeLedgerRows: () => [ledgerRow], getBuildRunSessions,
    })).json();
    expect(getBuildRunSessions).toHaveBeenCalledWith(expect.anything(), "demo");
    expect(body.data.demo).toMatchObject({ ok: true, count: 1, sessions: [{ session: "S1", count: 1 }] });

    const withoutDep = await (await handleWorktreesGet({ ...baseDeps(async () => ({ ok: true as const, stdout })), getWorktreeLedgerRows: () => [ledgerRow] })).json();
    expect("sessions" in withoutDep.data.demo).toBe(false);

    const failed = await (await handleWorktreesGet({
      ...baseDeps(async () => ({ ok: false, error: "probe failed" })), getBuildRunSessions,
    })).json();
    expect(failed.data.demo).toEqual({ ok: false, error: "probe failed" });
  });

  test("probes a nested project at its discovered location while keeping the dir identity", async () => {
    const probe = vi.fn(async () => ({ ok: true as const, stdout: "" }));
    const getWorktreeLedgerRows = vi.fn(() => []);
    const deps: WorktreesRouteDeps = {
      getProjects: () => ({
        projects: [{ dir: "nested", name: "Nested", path: "company/nested" } as never],
        skipped: 0, reposRoot: "",
      }),
      probeWorktreePorcelain: probe, getWorktreeLedgerRows, getDb: () => ({} as never),
      now: () => "2026-09-19T00:00:00Z",
    };
    const body = await (await handleWorktreesGet(deps)).json();
    expect(probe).toHaveBeenCalledWith(join(REPOS_ROOT, "company/nested"));
    expect(getWorktreeLedgerRows).toHaveBeenCalledWith(expect.anything(), "nested");
    expect(Object.keys(body.data)).toEqual(["nested"]);
  });
});
