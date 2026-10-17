// Pure git-status collector for the file tree's per-file tinting (spec §5 item 9, §7). Mirrors
// projects.ts's injected-GitProbe style: parsing is pure and unit-tested without a real subprocess;
// only defaultGitStatusForDir at the bottom wires a real `git` invocation.
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { GitProbe } from "./projects";

const execFileAsync = promisify(execFile);

export type FileGitStatus = "added" | "modified" | "deleted" | "untracked";

// Highest-priority status wins when more than one changed path aggregates under the same immediate
// child. ponytail: a fixed priority is enough for a tint, not a full status log.
const PRIORITY: Record<FileGitStatus, number> = { modified: 3, added: 2, deleted: 1, untracked: 0 };

function classify(code: string): FileGitStatus {
  if (code === "??") return "untracked";
  if (code.includes("A")) return "added";
  if (code.includes("D")) return "deleted";
  return "modified"; // M, R, C, U, or any other porcelain combination
}

// Parses `git status --porcelain=v1 --untracked-files=normal .` stdout (paths relative to the cwd
// the probe ran in, i.e. the LISTED directory) into a map keyed by the IMMEDIATE child name — the
// first path segment. A wholly untracked directory already reports as a single "?? dir/" line (git
// doesn't descend into it); a nested change several levels down still reports the full nested path,
// so taking the first segment tints whichever immediate child contains it either way.
export function parseGitStatusPorcelain(stdout: string): Map<string, FileGitStatus> {
  const result = new Map<string, FileGitStatus>();
  for (const line of stdout.split("\n")) {
    if (line.length < 4) continue;
    const code = line.slice(0, 2);
    let rest = line.slice(3);
    if (code.includes("R") || code.includes("C")) {
      const arrow = rest.indexOf(" -> ");
      if (arrow >= 0) rest = rest.slice(arrow + 4); // rename/copy: tint the NEW path, ignore the old one
    }
    const path = rest.replace(/\/$/, "");
    const segment = path.split("/")[0];
    if (!segment) continue;
    const status = classify(code);
    const existing = result.get(segment);
    if (!existing || PRIORITY[status] > PRIORITY[existing]) result.set(segment, status);
  }
  return result;
}

// Root-scoped: `absDir` may or may not be inside a git repo at all (e.g. `vault`, which is never a
// repo — spec §7). A probe failure (non-repo, timeout, git missing) degrades to null for the WHOLE
// directory, never a thrown error (AGENTS.md error contract: degrade quietly, never block the listing).
// The probe call itself is wrapped too (cold review round 1 F2): a REJECTED probe promise must
// degrade the same way a resolved {ok:false} does, never escape and abort the caller's listing.
export async function gitStatusForDir(absDir: string, probe: GitProbe): Promise<Map<string, FileGitStatus> | null> {
  let result;
  try {
    result = await probe(["status", "--porcelain=v1", "--untracked-files=normal", "."], absDir);
  } catch {
    return null;
  }
  if (!result || !result.ok) return null;
  return parseGitStatusPorcelain(result.stdout);
}

const GIT_STATUS_TIMEOUT_MS = 3000;
const realGitProbe: GitProbe = async (args, cwd) => {
  try {
    const { stdout } = await execFileAsync("git", args, {
      cwd, timeout: GIT_STATUS_TIMEOUT_MS, maxBuffer: 1024 * 1024, encoding: "utf8",
    });
    return { ok: true, stdout };
  } catch {
    return null;
  }
};

// Real resolver used by files.ts's listDirAt default — mirrors workflow-codex.ts's
// resolveProjectOwner pattern (a curried real-probe version of the pure function).
export function defaultGitStatusForDir(absDir: string): Promise<Map<string, FileGitStatus> | null> {
  return gitStatusForDir(absDir, realGitProbe);
}
