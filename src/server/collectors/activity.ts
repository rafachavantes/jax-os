import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { getProjects, REPOS_ROOT } from "./projects";
import { LOCAL_OPTS } from "../inputLimits";

export type Commit = {
  repo: string;
  hash: string;
  author: string;
  message: string;
  epoch: number;
};

export function parseGitLog(repo: string, out: string): Commit[] {
  return out
    .split(/\r?\n/)
    .map((l) => l.split("\x1f"))
    .filter((p) => p.length === 4 && /^\d+$/.test(p[3]))
    .map(([hash, author, message, epoch]) => ({
      repo,
      hash,
      author,
      message,
      epoch: Number(epoch),
    }));
}

export function mergeActivity(perRepo: Commit[][], limit = 10): Commit[] {
  return perRepo
    .flat()
    .sort((a, b) => b.epoch - a.epoch)
    .slice(0, limit);
}

// Untested shell glue. Only PARSED projects are asked for git history (bare
// plans/ dirs like random-debugs/ have no .git); a repo failing git log is
// skipped silently per spec — the feed shows the rest.
export type Utf8ExecSync = (
  file: string,
  args: readonly string[],
  options: { encoding: "utf8"; timeout: number; maxBuffer: number },
) => string;

function isLimitError(e: unknown): boolean {
  const code = (e as { code?: unknown }).code;
  return code === "ETIMEDOUT" || code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER";
}

export function getActivity(
  projects?: { dir: string }[],
  reposRoot: string = REPOS_ROOT,
  exec: Utf8ExecSync = execFileSync as Utf8ExecSync,
): Commit[] {
  const list = projects ?? getProjects().projects;
  const perRepo = list.map((p) => {
    try {
      const out = exec(
        "git",
        ["-C", join(reposRoot, p.dir), "log", "-n", "5", "--format=%h%x1f%an%x1f%s%x1f%ct"],
        LOCAL_OPTS,
      );
      return parseGitLog(p.dir, out);
    } catch (e) {
      if (isLimitError(e)) throw e;
      return [];
    }
  });
  return mergeActivity(perRepo);
}
