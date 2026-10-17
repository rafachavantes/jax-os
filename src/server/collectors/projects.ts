import { closeSync, fstatSync, openSync, readdirSync, readSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { reposRoot } from "../reposRoot";

export type ProjectStage = "spec" | "build" | "review" | "test" | "ship";
// Spec B §2.6: `state` is retired. `gate` is a commitment, not an activity — null is the normal
// case, and it is optional on disk (absence means nothing waits on a human).
export type ProjectGate = "awaiting-approval" | "blocked";

export type Project = {
  dir: string;
  name: string;
  stage: ProjectStage;
  gate: ProjectGate | null;
  // true only when `gate` was read from the deprecated `state:` alias (§2.6) — lets the UI
  // show a muted "legacy field" hint so the migration off `state:` is visible and finite.
  legacyGateField: boolean;
  builder: string;
  branch: string;
  tmux?: string;
  flag?: string;
  updated: string;
  now: string;
  // Parsed from `## Residuals`, one entry per bullet line — accepted risks to revisit (§4.3).
  residuals: string[];
  // status.md's own mtime, ISO string — matches workflow_events.ts's own `ts` column format so
  // Task 4's freshness count can compare them directly in SQL (§4.4).
  statusMtime: string;
  // Phase 3 §10: `archived: true` marks a project as archived; any other/absent value is false,
  // the same one-line hand-parsed pattern as flag/gate.
  archived: boolean;
};

export type ProjectScan = { projects: Project[]; skipped: number; reposRoot: string };

const STAGES = ["spec", "build", "review", "test", "ship"] as const;
const GATES = ["awaiting-approval", "blocked"] as const;

export const REPOS_ROOT = reposRoot();
export const STATUS_CAP = 256 * 1024;

export type StatusFs = {
  openSync: typeof openSync;
  fstatSync: typeof fstatSync;
  readSync: typeof readSync;
  closeSync: typeof closeSync;
};

const defaultStatusFs: StatusFs = { openSync, fstatSync, readSync, closeSync };

export function readStatusBounded(path: string, fs: StatusFs = defaultStatusFs): { text: string; mtime: string } {
  const fd = fs.openSync(path, "r");
  try {
    const stat = fs.fstatSync(fd);
    if (!stat.isFile()) throw new Error("not a regular file");
    if (stat.size > STATUS_CAP) throw new Error("status too large");
    const buffer = Buffer.alloc(STATUS_CAP + 1);
    let used = 0;
    while (used < buffer.length) {
      const n = fs.readSync(fd, buffer, used, buffer.length - used, used);
      if (n === 0) break;
      used += n;
    }
    if (used > STATUS_CAP || fs.fstatSync(fd).size > STATUS_CAP) throw new Error("status too large");
    return { text: buffer.subarray(0, used).toString("utf8"), mtime: new Date(stat.mtimeMs).toISOString() };
  } finally {
    fs.closeSync(fd);
  }
}

// Spec B §2.6: `gate` wins whenever present (valid or not — and the file is never "legacy" then).
// Absent `gate` falls back to the deprecated `state:` key — `idle`/`building`/anything
// unrecognised maps to no gate, `awaiting-approval`/`blocked` carry their meaning across.
function parseGate(fields: Record<string, string>): { gate: ProjectGate | null; legacyGateField: boolean } {
  const gateRaw = fields.gate;
  if (gateRaw !== undefined) {
    const gate = (GATES as readonly string[]).includes(gateRaw) ? (gateRaw as ProjectGate) : null;
    return { gate, legacyGateField: false };
  }
  const stateRaw = fields.state;
  if (stateRaw !== undefined) {
    const gate = (GATES as readonly string[]).includes(stateRaw) ? (stateRaw as ProjectGate) : null;
    return { gate, legacyGateField: true };
  }
  return { gate: null, legacyGateField: false };
}

// ponytail: hand parser for a flat `key: value` schema — a yaml dep is overkill.
// Split on first ": " (colon-space), never any ":", or ISO timestamps break.
export function parseStatusMd(dir: string, raw: string, statusMtime: string): Project | null {
  const m = raw.match(/^---\r?\n([\s\S]*?)\r?\n---/);
  if (!m) return null;
  const fields: Record<string, string> = {};
  for (const line of m[1].split(/\r?\n/)) {
    const i = line.indexOf(": ");
    if (i > 0) fields[line.slice(0, i).trim()] = line.slice(i + 2).trim();
  }
  const stage = fields.stage as ProjectStage;
  // Spec B §2.6: unlike `stage`, an absent or unrecognised gate no longer rejects the file — a
  // typo costs a badge, not a card. `stage` stays the one required field (no hub equivalent).
  if (!(STAGES as readonly string[]).includes(stage)) return null;
  const { gate, legacyGateField } = parseGate(fields);
  const body = raw.slice(m[0].length);
  return {
    dir,
    name: fields.project || dir,
    stage,
    gate,
    legacyGateField,
    builder: fields.builder ?? "",
    branch: fields.branch ?? "",
    tmux: fields.tmux || undefined,
    flag: fields.flag || undefined,
    archived: fields.archived === "true",
    updated: fields.updated ?? "",
    now: extractNow(body),
    residuals: extractResiduals(body),
    statusMtime,
  };
}

export function extractNow(markdown: string): string {
  const lines = markdown.split(/\r?\n/);
  const start = lines.findIndex((l) => /^##\s+Now\s*$/.test(l));
  if (start === -1) return "";
  const para: string[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "" && para.length === 0) continue;
    if (line === "" || line.startsWith("#")) break;
    para.push(line);
  }
  return para.join(" ");
}

// Parsed exactly as extractNow, except each line becomes its own array entry (a bullet list, not
// a paragraph) and a leading "-"/"*" marker is stripped (spec §4.3's Residuals block).
export function extractResiduals(markdown: string): string[] {
  const lines = markdown.split(/\r?\n/);
  const start = lines.findIndex((l) => /^##\s+Residuals\s*$/.test(l));
  if (start === -1) return [];
  const items: string[] = [];
  for (let i = start + 1; i < lines.length; i++) {
    const line = lines[i].trim();
    if (line === "" && items.length === 0) continue;
    if (line === "" || line.startsWith("#")) break;
    items.push(line.replace(/^[-*]\s+/, ""));
  }
  return items;
}

// MOA-467 Task 4 (acceptance fixes): a linked worktree's `.git` is a plain FILE whose
// `gitdir:` line points at the worktree's administrative dir under the control repo's
// common dir; that admin dir carries its own `commondir` file naming the common dir.
// Both metadata reads are pure and testable; only the collector touches the filesystem.
export function parseGitdirRef(content: string): string | null {
  for (const line of content.split(/\r?\n/)) {
    if (line.startsWith("gitdir:")) {
      const ref = line.slice("gitdir:".length).trim();
      return ref || null;
    }
  }
  return null;
}

// The `commondir` file is a single path line, relative to the gitdir dir (git writes
// `../..`) or absolute. An empty/absent value is null.
export function parseCommondir(content: string): string | null {
  for (const line of content.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed) return trimmed;
  }
  return null;
}

// True only for a real linked-worktree administrative relationship, never for a path
// that merely LOOKS like `.git/worktrees/<name>`: git registers every worktree admin dir
// at `<common-dir>/worktrees/<name>`, and that dir's `commondir` file must resolve (it is
// relative to the gitdir itself) back to the admin area's parent, i.e. the common dir.
// A separate-git-dir main repo has no `commondir` in its gitdir and stays independent;
// so does a lookalike path whose commondir is missing or points elsewhere.
export function isLinkedWorktreeGitdir(gitdirRef: string, commondir: string | null): boolean {
  if (!commondir) return false;
  const gitdir = resolve(gitdirRef);
  if (basename(dirname(gitdir)) !== "worktrees") return false;
  return dirname(dirname(gitdir)) === resolve(gitdir, commondir);
}

// MOA-469 C3: canonical owner resolution shared with the collector boundary, mirroring
// scripts/jaxflow_hook.py's `_git_owner` (MOA-467). Pure over an injected probe so the collector
// supplies the real subprocess and tests stay filesystem-free. An ordinary checkout and a
// `--separate-git-dir` main checkout both have gitdir == common dir and resolve to their own
// toplevel; a linked worktree (gitdir != common) resolves to the MAIN entry of `git worktree list`.
// `signal` lets the collector share ONE deadline across every git probe (MOA-469 F2): a probe after
// the collection deadline returns null without spawning, and an in-flight probe is aborted at expiry.
export type GitProbe = (args: string[], cwd: string, signal?: AbortSignal) => Promise<{ ok: boolean; stdout: string } | null>;

export async function canonicalOwner(cwd: string, probe: GitProbe, signal?: AbortSignal): Promise<string | null> {
  if (!cwd) return null;
  const probeResult = await probe(["rev-parse", "--path-format=absolute", "--git-dir", "--git-common-dir"], cwd, signal);
  if (!probeResult || !probeResult.ok) return null;
  const lines = probeResult.stdout.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  if (lines.length < 2) return null;
  const gitdir = resolve(lines[0]);
  const common = resolve(lines[1]);
  if (gitdir === common) {
    const top = await probe(["rev-parse", "--show-toplevel"], cwd, signal);
    if (!top || !top.ok || !top.stdout.trim()) return null;
    return resolve(top.stdout.trim());
  }
  const listed = await probe(["worktree", "list", "--porcelain"], cwd, signal);
  if (!listed || !listed.ok) return null;
  for (const line of listed.stdout.split(/\r?\n/)) {
    if (line.startsWith("worktree ")) {
      const p = line.slice("worktree ".length).trim();
      return p ? resolve(p) : null;
    }
  }
  return null;
}

// Project identity = the basename of the session's canonical repository root, or null whenever the
// repository cannot be resolved (any probe failure, timeout or a non-repository cwd). Never a
// basename-only guess: MOA-469 C3 / spec §4 "Unknown ownership must not be guessed from cwd". The
// Python hook keeps its own pre-existing non-repository fallback as a separate code path.
export async function ownerProjectName(cwd: string, probe: GitProbe, signal?: AbortSignal): Promise<string | null> {
  if (!cwd) return null;
  const owner = await canonicalOwner(cwd, probe, signal);
  return owner ? basename(owner) || null : null;
}

function isLinkedWorktree(root: string, dir: string, fs: StatusFs = defaultStatusFs): boolean {
  // readStatusBounded doubles as the `.git` file probe: a directory `.git` (normal
  // repo), a missing one, an unreadable one, or an oversized one all throw, and an
  // unprovable relationship must never suppress an independent repo's card.
  const checkout = join(root, dir);
  let ref: string | null;
  try {
    ref = parseGitdirRef(readStatusBounded(join(checkout, ".git"), fs).text);
  } catch {
    return false;
  }
  if (ref === null) return false;
  // MOA-467 (acceptance fix): a relative gitdir reference is resolved against the
  // checkout dir that CONTAINS this `.git` file, never the collector's own cwd, before
  // `commondir` is read or the relationship is classified.
  const gitdir = resolve(checkout, ref);
  let commondir: string | null;
  try {
    commondir = parseCommondir(readStatusBounded(join(gitdir, "commondir"), fs).text);
  } catch {
    return false;
  }
  return isLinkedWorktreeGitdir(gitdir, commondir);
}

// Untested fs glue — logic above is the tested surface.
export function getProjects(root: string = REPOS_ROOT): ProjectScan {
  const projects: Project[] = [];
  let skipped = 0;
  for (const entry of readdirSync(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    // MOA-467 Task 4: a linked worktree is another directory's checkout — the CONTROL
    // repo owns its project card. Filter BEFORE any status read, so a worktree's valid
    // status never becomes a duplicate card and never counts as skipped; a positively
    // identified worktree is excluded even when its control repo is outside the scan or
    // has no valid status. Historical worktree status files may remain on disk.
    if (isLinkedWorktree(root, entry.name)) continue;
    const statusPath = join(root, entry.name, ".jax-os", "status.md");
    let raw: string;
    let statusMtime: string;
    try {
      const read = readStatusBounded(statusPath);
      raw = read.text;
      statusMtime = read.mtime;
    } catch (e) {
      // ENOENT/ENOTDIR = no status.md → not a pipeline project, not "skipped".
      // Anything else (EACCES, over-cap…) is a real read failure → count it as skipped.
      const code = (e as NodeJS.ErrnoException).code;
      if (code !== "ENOENT" && code !== "ENOTDIR") skipped++;
      continue;
    }
    const project = parseStatusMd(entry.name, raw, statusMtime);
    if (project) projects.push(project);
    else skipped++; // has a status.md but no valid frontmatter → muted hint in UI
  }
  // ponytail: lexicographic ISO sort — good enough while all repos share -03:00
  projects.sort((a, b) => b.updated.localeCompare(a.updated));
  return { projects, skipped, reposRoot: root };
}
