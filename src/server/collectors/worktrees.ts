import { execFile } from "node:child_process";
import { promisify } from "node:util";

// Pure parser + collector split, mirroring workflow-codex.ts's gitProbe/collectCodexSnapshot.

const execFileAsync = promisify(execFile);

export type WorktreeEntry = { path: string; branch: string | null; detached: boolean };

export function parseWorktreePorcelain(output: string): WorktreeEntry[] {
  const entries: WorktreeEntry[] = [];
  let current: WorktreeEntry | null = null;
  for (const line of output.split("\n")) {
    if (line.startsWith("worktree ")) {
      current = { path: line.slice("worktree ".length).trim(), branch: null, detached: false };
      entries.push(current);
    } else if (!current) {
      continue;
    } else if (line.startsWith("branch ")) {
      const ref = line.slice("branch ".length).trim();
      current.branch = ref.startsWith("refs/heads/") ? ref.slice("refs/heads/".length) : ref;
    } else if (line === "detached") {
      current.detached = true;
    }
  }
  return entries;
}

export type LedgerRow = { id: number; runId: string; kind: "spec" | "plan" | "diff" | "build" | "merge"; branch: string; startedAt: string; status: "running" | "finished"; finishedAt: string | null };

export type WorktreeResult = { ok: true; count: number; oldestAgeDays: number } | { ok: false; error: string };

// Pure (spec §7): injected porcelain/ledgerRows/opts, no tmux/git calls of its own.
export function collectRetainedWorktrees(
  porcelain: string, ledgerRows: LedgerRow[], opts: { now: string; thresholdDays: number; controlRepoPath: string },
): WorktreeResult {
  const entries = parseWorktreePorcelain(porcelain);
  const registeredBranches = new Set(entries.filter((e) => !e.detached && e.path !== opts.controlRepoPath && e.branch).map((e) => e.branch as string));
  // F6 (round 4): "latest" is by event TIMESTAMP, `id` breaking an exact tie -- the
  // same rule Python's `_latest_builder_attempt` (`scripts/jaxflow_common.py`) uses, so gc
  // and this card-line count never disagree on which attempt is the latest for a
  // backfilled or out-of-order write.
  const latestBuildByBranch = new Map<string, LedgerRow>();
  for (const r of ledgerRows) {
    if (r.kind !== "build") continue;
    const prior = latestBuildByBranch.get(r.branch);
    if (!prior || r.startedAt > prior.startedAt || (r.startedAt === prior.startedAt && r.id > prior.id)) {
      latestBuildByBranch.set(r.branch, r);
    }
  }
  const nowMs = Date.parse(opts.now);
  let count = 0;
  let oldestAgeDays = 0;
  for (const [branch, r] of latestBuildByBranch) {
    if (!registeredBranches.has(branch)) continue;
    if (r.status !== "finished" || !r.finishedAt) continue;
    const ageDays = (nowMs - Date.parse(r.finishedAt)) / 86_400_000;
    if (ageDays < opts.thresholdDays) continue;
    count += 1;
    if (ageDays > oldestAgeDays) oldestAgeDays = ageDays;
  }
  return { ok: true, count, oldestAgeDays: Math.floor(oldestAgeDays) };
}

// Real collector: one `git worktree list --porcelain` call for `repoPath`.
export async function probeWorktreePorcelain(repoPath: string): Promise<{ ok: true; stdout: string } | { ok: false; error: string }> {
  try {
    const { stdout } = await execFileAsync("git", ["worktree", "list", "--porcelain"], { cwd: repoPath, encoding: "utf8" });
    return { ok: true, stdout };
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) };
  }
}
