import { join } from "node:path";
import { collectorResponse } from "../../../../server/api";
import { getProjects, REPOS_ROOT } from "../../../../server/collectors/projects";
import { collectRetainedWorktrees, probeWorktreePorcelain, type WorktreeResult } from "../../../../server/collectors/worktrees";
import { getDb } from "../../../../server/db";
import { getWorktreeLedgerRows } from "../../../../server/db/workflows";

const OLDER_THAN_DAYS = 7; // mirrors jaxflow gc's own default (spec G3)

export type WorktreesRouteDeps = {
  getProjects: typeof getProjects;
  probeWorktreePorcelain: typeof probeWorktreePorcelain;
  getWorktreeLedgerRows: typeof getWorktreeLedgerRows;
  getDb: typeof getDb;
  now: () => string;
};

const systemDeps: WorktreesRouteDeps = { getProjects, probeWorktreePorcelain, getWorktreeLedgerRows, getDb, now: () => new Date().toISOString() };

// Round 3 F2: one entry per PROJECT-SCAN entry (getProjects()), never gated on
// getHubData's 30-day/live-pane membership -- an idle project still gets its real count.
//
// F3 (round 2, MEDIUM): `dir` is relative ("demo") but porcelain paths are absolute --
// join REPOS_ROOT for `cwd`/controlRepoPath. Ledger key is `p.dir` (mission.ts's hubMap
// key), never `p.name` (status.md's display name). Output stays keyed by relative `p.dir`.
export async function handleWorktreesGet(deps: WorktreesRouteDeps = systemDeps) {
  return collectorResponse<Record<string, WorktreeResult>>(async () => {
    const { projects } = deps.getProjects();
    const db = deps.getDb();
    const now = deps.now();
    const out: Record<string, WorktreeResult> = {};
    for (const p of projects) {
      const absPath = join(REPOS_ROOT, p.dir);
      const probe = await deps.probeWorktreePorcelain(absPath);
      if (!probe.ok) {
        out[p.dir] = { ok: false, error: probe.error };
        continue;
      }
      const ledgerRows = deps.getWorktreeLedgerRows(db, p.dir);
      out[p.dir] = collectRetainedWorktrees(probe.stdout, ledgerRows, { now, thresholdDays: OLDER_THAN_DAYS, controlRepoPath: absPath });
    }
    return out;
  });
}
