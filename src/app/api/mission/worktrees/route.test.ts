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
});
