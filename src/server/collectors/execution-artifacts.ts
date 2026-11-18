import { constants } from "node:fs";
import { lstat, open, opendir, realpath } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { basename, isAbsolute, join, relative, sep } from "node:path";
import { parseStatusMd, canonicalOwner, type GitProbe } from "./projects";
import { reposRoot as configuredReposRoot } from "../reposRoot";
import { RUN_ID_RE, PROJECT_LABEL } from "../../lib/workflow";

// M3a, Task 2 — safe retained artifacts. The only inputs are a validated destination and a
// validated 12-lowercase-hex run ID; every path is derived from those, never from a payload.

export const ARTIFACT_WARNINGS = [
  "root-unavailable",
  "manifest-missing",
  "manifest-invalid",
  "report-missing",
  "tests-missing",
  "worktree-unavailable",
] as const;
export type ArtifactWarning = (typeof ARTIFACT_WARNINGS)[number];

export type ArtifactLink = { rel: string };

export type ExecutionArtifactFacts = {
  destination: string;
  runId: string;
  branch: string | null;
  branchSource: "manifest" | null;
  report: ArtifactLink | null;
  tests: ArtifactLink | null;
  warnings: ArtifactWarning[];
};

export class BudgetError extends Error {
  constructor(readonly reason: "deadline" | "timeout") {
    super(reason);
  }
}

// One shared native-operation gate: a fixed in-flight maximum, individual operation timeouts and
// stop-starting-at-deadline. Timed-out operations keep their slot until the native promise
// actually settles, so a hung filesystem call can never accumulate concurrency across polls.
export class InspectionBudget {
  private active = 0;
  private readonly waiters: (() => void)[] = [];
  private readonly deadline: number;
  constructor(
    readonly limit = 4,
    readonly totalMs = 2_000,
    private readonly clock: () => number = Date.now,
  ) {
    this.deadline = this.clock() + totalMs;
  }

  get expired(): boolean {
    return this.clock() >= this.deadline;
  }

  get remainingMs(): number {
    return Math.max(0, this.deadline - this.clock());
  }

  private acquire(): Promise<void> {
    if (this.expired) return Promise.reject(new BudgetError("deadline"));
    if (this.active < this.limit) {
      this.active++;
      return Promise.resolve();
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new BudgetError("deadline")), Math.max(0, this.deadline - this.clock()));
      this.waiters.push(() => {
        clearTimeout(timer);
        this.active++;
        resolve();
      });
    });
  }

  private release(): void {
    this.active--;
    const next = this.waiters.shift();
    if (next) next();
  }

  async run<T>(fn: () => Promise<T>, timeoutMs: number): Promise<T> {
    await this.acquire();
    const native = fn();
    void native.finally(() => this.release()).catch(() => {});
    return await new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new BudgetError("timeout")), Math.max(0, Math.min(timeoutMs, this.remainingMs)));
      native.then(
        (value) => { clearTimeout(timer); resolve(value); },
        (error) => { clearTimeout(timer); reject(error); },
      );
    });
  }

  // Starts nothing after the deadline for work that is cheaply checkable before starting.
  async check(): Promise<void> {
    if (this.expired) throw new BudgetError("deadline");
  }
}

const execFileAsync = promisify(execFile);
const defaultGitProbe: GitProbe = async (args, cwd) => {
  try {
    const { stdout } = await execFileAsync("git", args, {
      cwd, encoding: "utf8", timeout: 1_000, maxBuffer: 16 * 1024,
    });
    return { ok: true, stdout };
  } catch {
    return { ok: false, stdout: "" };
  }
};

type DiscoveredRoot = { abs: string; rel: string };

const ROOT_CACHE_TTL_MS = 10_000;
const rootCache = new Map<string, { at: number; roots: Map<string, DiscoveredRoot[]> }>();

type ScanLimits = { maxEntries: number; maxCandidates: number; maxDepth: number };
const DEFAULT_SCAN_LIMITS: ScanLimits = { maxEntries: 10_000, maxCandidates: 512, maxDepth: 2 };

export type ArtifactDeps = {
  reposRoot?: string;
  budget?: InspectionBudget;
  now?: () => number;
  gitProbe?: GitProbe;
  discoveryMs?: number;
  fsMs?: number;
  gitMs?: number;
  scanLimits?: Partial<ScanLimits>;
};

function contained(base: string, candidate: string): boolean {
  const rel = relative(base, candidate);
  return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

async function lstatKind(path: string): Promise<"present" | "absent" | "unknown"> {
  try {
    await lstat(path);
    return "present";
  } catch (e) {
    const code = (e as NodeJS.ErrnoException).code;
    return code === "ENOENT" || code === "ENOTDIR" ? "absent" : "unknown";
  }
}

async function readCapped(
  path: string, cap: number, rootCanonical: string, budget: InspectionBudget, timeoutMs: number,
): Promise<{ ok: true; text: string } | { ok: false; reason: "missing" | "unsafe" | "too-large" | "failed" }> {
  try {
    return await budget.run(async () => {
      const fd = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
      try {
        const before = await fd.stat();
        if (!before.isFile()) return { ok: false as const, reason: "unsafe" as const };
        if (before.size > cap) return { ok: false as const, reason: "too-large" as const };
        let canonical: string;
        try {
          canonical = await realpath(path);
        } catch {
          return { ok: false as const, reason: "missing" as const };
        }
        if (!contained(rootCanonical, canonical)) return { ok: false as const, reason: "unsafe" as const };
        const buffer = Buffer.alloc(cap + 1);
        const { bytesRead } = await fd.read(buffer, 0, buffer.length, 0);
        if (bytesRead > cap) return { ok: false as const, reason: "too-large" as const };
        const after = await fd.stat();
        if (after.size !== before.size || after.ino !== before.ino) return { ok: false as const, reason: "failed" as const };
        return { ok: true as const, text: buffer.subarray(0, bytesRead).toString("utf8") };
      } finally {
        await fd.close();
      }
    }, timeoutMs);
  } catch (e) {
    const code = (e as NodeJS.ErrnoException).code;
    if (e instanceof BudgetError) return { ok: false, reason: "failed" };
    return { ok: false, reason: code === "ENOENT" || code === "ENOTDIR" ? "missing" : "failed" };
  }
}

async function regularFileInside(
  path: string, rootCanonical: string, budget: InspectionBudget, timeoutMs: number,
): Promise<boolean> {
  try {
    return await budget.run(async () => {
      let fd;
      try {
        fd = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW);
      } catch {
        return false;
      }
      try {
        const info = await fd.stat();
        if (!info.isFile()) return false;
        const canonical = await realpath(path);
        return contained(rootCanonical, canonical);
      } finally {
        await fd.close();
      }
    }, timeoutMs);
  } catch {
    return false;
  }
}

async function isLinkedWorktreeDir(
  checkout: string, budget: InspectionBudget, timeoutMs: number, limit = 256 * 1024,
): Promise<boolean> {
  const gitFile = await readCapped(join(checkout, ".git"), limit, checkout, budget, timeoutMs);
  if (!gitFile.ok) return false;
  let gitdirRef: string | null = null;
  for (const line of gitFile.text.split(/\r?\n/)) {
    if (line.startsWith("gitdir:")) {
      gitdirRef = line.slice("gitdir:".length).trim() || null;
      break;
    }
  }
  if (gitdirRef === null) return false;
  const gitdir = isAbsolute(gitdirRef) ? gitdirRef : join(checkout, gitdirRef);
  const commondir = await readCapped(join(gitdir, "commondir"), limit, gitdir, budget, timeoutMs);
  if (!commondir.ok) return false;
  const first = commondir.text.split(/\r?\n/).map((l) => l.trim()).find(Boolean) ?? null;
  if (!first) return false;
  const resolved = isAbsolute(first) ? first : join(gitdir, first);
  const adminParent = basename(join(gitdir, ".."));
  if (adminParent !== "worktrees") return false;
  try {
    return (await realpath(join(gitdir, "..", ".."))) === (await realpath(resolved));
  } catch {
    return false;
  }
}

// Completed two-level scans only. Any timeout/budget exhaustion returns no cache and the caller
// marks every destination unavailable — never a guessed unique root.
async function discoverRoots(
  root: string, budget: InspectionBudget, limits: ScanLimits, timeoutMs: number,
): Promise<{ ok: true; roots: Map<string, DiscoveredRoot[]> } | { ok: false }> {
  const clock = budget;
  try {
    const rootCanonical = await budget.run(() => realpath(root), timeoutMs);
    const cached = rootCache.get(rootCanonical);
    if (cached && Date.now() - cached.at < ROOT_CACHE_TTL_MS) return { ok: true, roots: cached.roots };

    let entries = 0;
    const candidates: { abs: string; rel: string; dir: string; depth: number }[] = [];
    const depthOne: { abs: string; rel: string; dir: string }[] = [];

    async function list(dir: string): Promise<{ name: string; isDirectory: boolean; isSymbolicLink: boolean }[]> {
      return budget.run(async () => {
        const handle = await opendir(dir);
        const out: { name: string; isDirectory: boolean; isSymbolicLink: boolean }[] = [];
        try {
          for await (const entry of handle) {
            entries++;
            if (entries > limits.maxEntries) throw new BudgetError("deadline");
            out.push({ name: entry.name, isDirectory: entry.isDirectory(), isSymbolicLink: entry.isSymbolicLink() });
          }
        } finally {
          await handle.close().catch(() => {});
        }
        return out;
      }, timeoutMs);
    }

    for (const entry of await list(root)) {
      if (!entry.isDirectory || entry.isSymbolicLink) continue;
      const abs = join(root, entry.name);
      if (await isLinkedWorktreeDir(abs, clock, timeoutMs)) continue;
      depthOne.push({ abs, rel: entry.name, dir: entry.name });
      candidates.push({ abs, rel: entry.name, dir: entry.name, depth: 1 });
    }

    for (const parent of depthOne) {
      let children: { name: string; isDirectory: boolean; isSymbolicLink: boolean }[];
      try {
        children = await list(parent.abs);
      } catch (e) {
        if (e instanceof BudgetError) throw e;
        continue;
      }
      for (const entry of children) {
        if (!entry.isDirectory || entry.isSymbolicLink) continue;
        const abs = join(parent.abs, entry.name);
        if (await isLinkedWorktreeDir(abs, clock, timeoutMs)) continue;
        candidates.push({ abs, rel: `${parent.rel}/${entry.name}`, dir: entry.name, depth: 2 });
      }
    }
    if (candidates.length > limits.maxCandidates) return { ok: false };

    const ownerOf = new Map<string, { owners: number; uncertain: boolean }>();
    for (const candidate of candidates) {
      const git = await lstatKind(join(candidate.abs, ".git"));
      let owner = git === "present";
      if (git === "unknown") {
        const info = ownerOf.get(candidate.dir) ?? { owners: 0, uncertain: false };
        info.uncertain = true;
        ownerOf.set(candidate.dir, info);
        continue;
      }
      if (!owner) {
        const statusPath = join(candidate.abs, ".jax-os", "status.md");
        const statusExists = await lstatKind(statusPath);
        if (statusExists === "unknown") {
          const info = ownerOf.get(candidate.dir) ?? { owners: 0, uncertain: false };
          info.uncertain = true;
          ownerOf.set(candidate.dir, info);
          continue;
        }
        if (statusExists === "present") {
          const read = await readCapped(statusPath, 256 * 1024, candidate.abs, clock, timeoutMs);
          owner = read.ok && parseStatusMd(candidate.dir, read.text, new Date().toISOString()) !== null;
        }
      }
      if (!owner) continue;
      const info = ownerOf.get(candidate.dir) ?? { owners: 0, uncertain: false };
      info.owners++;
      ownerOf.set(candidate.dir, info);
    }

    const roots = new Map<string, DiscoveredRoot[]>();
    for (const candidate of candidates) {
      const info = ownerOf.get(candidate.dir);
      if (!info || info.owners !== 1 || info.uncertain) continue;
      if (!PROJECT_LABEL.test(candidate.dir)) continue;
      const list = roots.get(candidate.dir) ?? [];
      list.push({ abs: candidate.abs, rel: candidate.rel });
      roots.set(candidate.dir, list);
    }
    rootCache.set(rootCanonical, { at: Date.now(), roots });
    return { ok: true, roots };
  } catch {
    return { ok: false };
  }
}

type Manifest = {
  runId: string;
  project: string;
  repo: string;
  worktree: string | null;
  branch: string | null;
  kind: string;
  role: "builder" | "reviewer";
};

function parseManifest(raw: string): Manifest | null {
  let value: unknown;
  try {
    value = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const m = value as Record<string, unknown>;
  const bounded = (v: unknown, cap: number): string | null =>
    typeof v === "string" && v.length > 0 && v.length <= cap && !/[\u0000-\u001f\u007f]/.test(v) ? v : null;
  const runId = bounded(m.run_id, 64);
  const project = bounded(m.project, 64);
  const repo = bounded(m.repo, 512);
  if (!runId || !project || !repo) return null;
  if (m.kind !== "build" && m.kind !== "spec" && m.kind !== "plan" && m.kind !== "diff" && m.kind !== "merge") return null;
  if (m.role !== "builder" && m.role !== "reviewer") return null;
  return {
    runId, project, repo, kind: m.kind, role: m.role,
    worktree: bounded(m.worktree, 512),
    branch: bounded(m.branch, 512) ?? bounded(m.target, 512),
  };
}

type RootResolution = { ok: true; root: DiscoveredRoot; canonical: string } | { ok: false };

async function resolveRoot(
  destination: string, roots: Map<string, DiscoveredRoot[]>, budget: InspectionBudget, timeoutMs: number,
): Promise<RootResolution> {
  const candidates = roots.get(destination) ?? [];
  if (candidates.length !== 1) return { ok: false };
  const candidate = candidates[0];
  try {
    const canonical = await budget.run(() => realpath(candidate.abs), timeoutMs);
    return { ok: true, root: candidate, canonical };
  } catch {
    return { ok: false };
  }
}

async function inspectOne(
  input: { destination: string; runId: string },
  deps: Required<Pick<ArtifactDeps, "fsMs" | "gitMs" | "gitProbe">> & {
    reposCanonical: string; roots: Map<string, DiscoveredRoot[]>; budget: InspectionBudget;
  },
): Promise<ExecutionArtifactFacts> {
  const warnings: ArtifactWarning[] = [];
  const unavailable = (warning: ArtifactWarning): ExecutionArtifactFacts => ({
    destination: input.destination, runId: input.runId, branch: null, branchSource: null,
    report: null, tests: null, warnings: [warning],
  });
  if (!PROJECT_LABEL.test(input.destination) || !RUN_ID_RE.test(input.runId)) {
    return { ...unavailable("root-unavailable"), warnings: ["root-unavailable"] };
  }
  const resolved = await resolveRoot(input.destination, deps.roots, deps.budget, deps.fsMs);
  if (!resolved.ok) return unavailable("root-unavailable");
  const root = resolved.root;
  const canonical = resolved.canonical;

  const manifestPath = join(root.abs, ".local", "runs", input.runId, "manifest.json");
  const manifestRead = await readCapped(manifestPath, 64 * 1024, canonical, deps.budget, deps.fsMs);
  let manifest: Manifest | null = null;
  if (manifestRead.ok) {
    const parsed = parseManifest(manifestRead.text);
    // Identity: the manifest belongs to this exact destination/run and to this control repo.
    if (parsed && parsed.runId === input.runId && parsed.project === input.destination) {
      try {
        const repoCanonical = await deps.budget.run(() => realpath(parsed.repo), deps.fsMs);
        if (repoCanonical === canonical) manifest = parsed;
        else warnings.push("manifest-invalid");
      } catch {
        warnings.push("manifest-invalid");
      }
    } else {
      warnings.push("manifest-invalid");
    }
  } else {
    warnings.push(manifestRead.reason === "missing" ? "manifest-missing" : "manifest-invalid");
  }

  let worktree: string | null = null;
  if (manifest?.kind === "build" && manifest.worktree) {
    try {
      const worktreeCanonical = await deps.budget.run(() => realpath(manifest.worktree!), deps.fsMs);
      if (!contained(deps.reposCanonical, worktreeCanonical)) {
        warnings.push("worktree-unavailable");
      } else {
        const owner = await deps.budget.run(() => canonicalOwner(worktreeCanonical, deps.gitProbe), deps.gitMs);
        if (owner === canonical) worktree = worktreeCanonical;
        else warnings.push("worktree-unavailable");
      }
    } catch {
      warnings.push("worktree-unavailable");
    }
  }

  const reportName = `${input.runId}.md`;
  const testsName = `${input.runId}.tests.txt`;
  const controlReports = join(root.abs, ".local", "reports");
  let report: ArtifactLink | null = null;
  let tests: ArtifactLink | null = null;

  if (await regularFileInside(join(controlReports, reportName), canonical, deps.budget, deps.fsMs)) {
    report = { rel: `${root.rel}/.local/reports/${reportName}` };
  }
  if (await regularFileInside(join(controlReports, testsName), canonical, deps.budget, deps.fsMs)) {
    tests = { rel: `${root.rel}/.local/reports/${testsName}` };
  }
  if ((report === null || tests === null) && worktree !== null) {
    const worktreeRel = relative(deps.reposCanonical, worktree).split(sep).join("/");
    if (report === null && await regularFileInside(join(worktree, ".local", "reports", reportName), worktree, deps.budget, deps.fsMs)) {
      report = { rel: `${worktreeRel}/.local/reports/${reportName}` };
    }
    if (tests === null && await regularFileInside(join(worktree, ".local", "reports", testsName), worktree, deps.budget, deps.fsMs)) {
      tests = { rel: `${worktreeRel}/.local/reports/${testsName}` };
    }
  }
  if (report === null) warnings.push("report-missing");
  if (tests === null) warnings.push("tests-missing");

  return {
    destination: input.destination,
    runId: input.runId,
    branch: manifest?.branch ?? null,
    branchSource: manifest?.branch ? "manifest" : null,
    report,
    tests,
    warnings,
  };
}

// Async, independently bounded artifact inspection for a bounded row set. A discovery failure
// keeps known rows: every row reports root-unavailable, never a guessed path or a DB change.
export async function inspectArtifacts(
  rows: readonly { destination: string; runId: string }[],
  deps: ArtifactDeps = {},
): Promise<ExecutionArtifactFacts[]> {
  const now = deps.now ?? Date.now;
  const budget = deps.budget ?? new InspectionBudget(4, 2_000, now);
  const repos = deps.reposRoot ?? configuredReposRoot();
  const discoveryMs = deps.discoveryMs ?? 1_000;
  const fsMs = deps.fsMs ?? 250;
  const gitMs = deps.gitMs ?? 1_000;
  const limits = { ...DEFAULT_SCAN_LIMITS, ...deps.scanLimits };
  const gitProbe = deps.gitProbe ?? defaultGitProbe;

  let reposCanonical: string;
  try {
    reposCanonical = await budget.run(() => realpath(repos), discoveryMs);
  } catch {
    return rows.map((row) => ({
      destination: row.destination, runId: row.runId, branch: null, branchSource: null,
      report: null, tests: null, warnings: ["root-unavailable"],
    }));
  }
  const discovered = await discoverRoots(reposCanonical, budget, limits, discoveryMs);
  if (!discovered.ok) {
    return rows.map((row) => ({
      destination: row.destination, runId: row.runId, branch: null, branchSource: null,
      report: null, tests: null, warnings: ["root-unavailable"],
    }));
  }
  return Promise.all(rows.map((row) => inspectOne(row, {
    reposCanonical, roots: discovered.roots, budget, fsMs, gitMs, gitProbe,
  })));
}
