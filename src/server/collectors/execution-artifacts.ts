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

// Server-only facts used to bind worker observation to the exact manifest and control repo.
// Never part of the API payload.
export type ExecutionArtifactInternal = { manifestPath: string | null; repo: string | null };

export function artifactKey(destination: string, runId: string): string {
  return `${destination}/${runId}`;
}

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
  private readonly pending = new Set<Promise<unknown>>();
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
      const waiter = () => {
        clearTimeout(timer);
        // Re-check after wakeup: a queued callback must never start native work after the deadline.
        if (this.expired) {
          reject(new BudgetError("deadline"));
          return;
        }
        this.active++;
        resolve();
      };
      const timer = setTimeout(() => {
        const index = this.waiters.indexOf(waiter);
        if (index >= 0) this.waiters.splice(index, 1);
        reject(new BudgetError("deadline"));
      }, Math.max(0, this.deadline - this.clock()));
      this.waiters.push(waiter);
    });
  }

  private release(): void {
    this.active--;
    const next = this.waiters.shift();
    if (next) next();
  }

  async run<T>(fn: () => Promise<T>, timeoutMs: number): Promise<T> {
    await this.acquire();
    if (this.expired) {
      this.release();
      throw new BudgetError("deadline");
    }
    let native: Promise<T>;
    try {
      native = fn();
    } catch (error) {
      // A synchronous throw must never leak the slot acquired above.
      this.release();
      throw error;
    }
    const done = (): void => {
      this.pending.delete(native);
      this.release();
    };
    this.pending.add(native);
    void native.then(done, done);
    return await new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => reject(new BudgetError("timeout")), Math.max(0, Math.min(timeoutMs, this.remainingMs)));
      native.then(
        (value) => { clearTimeout(timer); resolve(value); },
        (error) => { clearTimeout(timer); reject(error); },
      );
    });
  }

  // The coordinator's cleanup latch: resolves only when every started native operation has
  // actually settled, including ones whose caller already timed out.
  async settled(): Promise<void> {
    while (this.pending.size > 0) await Promise.allSettled([...this.pending]);
  }

  // A view over the same native capacity pool with an earlier subdeadline. It never resets the
  // in-flight maximum, so a composed collector cannot gain extra concurrency.
  scope(totalMs: number): BudgetLike {
    const parent = this;
    const deadline = parent.clock() + totalMs;
    const remaining = (): number => Math.max(0, Math.min(deadline - parent.clock(), parent.remainingMs));
    return {
      get expired() { return parent.clock() >= deadline || parent.expired; },
      get remainingMs() { return remaining(); },
      async run<T>(fn: () => Promise<T>, timeoutMs: number): Promise<T> {
        if (parent.clock() >= deadline) throw new BudgetError("deadline");
        return parent.run(fn, Math.min(timeoutMs, remaining()));
      },
      async check(): Promise<void> {
        if (parent.clock() >= deadline || parent.expired) throw new BudgetError("deadline");
      },
      settled: () => parent.settled(),
    };
  }

  // Starts nothing after the deadline for work that is cheaply checkable before starting.
  async check(): Promise<void> {
    if (this.expired) throw new BudgetError("deadline");
  }
}

export type BudgetLike = {
  readonly expired: boolean;
  readonly remainingMs: number;
  run<T>(fn: () => Promise<T>, timeoutMs: number): Promise<T>;
  check(): Promise<void>;
  settled(): Promise<void>;
};

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
type DirEntry = { name: string; isDirectory: boolean; isSymbolicLink: boolean };
type Canonicalize = (path: string) => Promise<string>;

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
  // Injectable native seams: the coordinator gate/deadline applies around them exactly as it
  // does around the real `node:fs` calls.
  realpath?: Canonicalize;
  listDir?: (dir: string) => Promise<DirEntry[]>;
};

function contained(base: string, candidate: string): boolean {
  const rel = relative(base, candidate);
  return rel === "" || (rel !== ".." && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

async function lstatKind(path: string, budget: InspectionBudget, timeoutMs: number): Promise<"present" | "absent" | "unknown"> {
  try {
    return await budget.run(async () => {
      await lstat(path);
      return "present" as const;
    }, timeoutMs);
  } catch (e) {
    if (e instanceof BudgetError) throw e;
    const code = (e as NodeJS.ErrnoException).code;
    return code === "ENOENT" || code === "ENOTDIR" ? "absent" : "unknown";
  }
}

async function readCapped(
  path: string, cap: number, rootCanonical: string, budget: InspectionBudget, timeoutMs: number,
  canonicalize: Canonicalize,
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
          canonical = await canonicalize(path);
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
    if (e instanceof BudgetError) throw e;
    const code = (e as NodeJS.ErrnoException).code;
    return { ok: false, reason: code === "ENOENT" || code === "ENOTDIR" ? "missing" : "failed" };
  }
}

async function regularFileInside(
  path: string, rootCanonical: string, budget: InspectionBudget, timeoutMs: number, canonicalize: Canonicalize,
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
        const canonical = await canonicalize(path);
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
  checkout: string, budget: InspectionBudget, timeoutMs: number, canonicalize: Canonicalize, limit = 256 * 1024,
): Promise<boolean> {
  const gitFile = await readCapped(join(checkout, ".git"), limit, checkout, budget, timeoutMs, canonicalize);
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
  const commondir = await readCapped(join(gitdir, "commondir"), limit, gitdir, budget, timeoutMs, canonicalize);
  if (!commondir.ok) return false;
  const first = commondir.text.split(/\r?\n/).map((l) => l.trim()).find(Boolean) ?? null;
  if (!first) return false;
  const resolved = isAbsolute(first) ? first : join(gitdir, first);
  const adminParent = basename(join(gitdir, ".."));
  if (adminParent !== "worktrees") return false;
  try {
    return (await budget.run(() => canonicalize(join(gitdir, "..", "..")), timeoutMs))
      === (await budget.run(() => canonicalize(resolved), timeoutMs));
  } catch (e) {
    if (e instanceof BudgetError) throw e;
    return false;
  }
}

// A bounded real directory iterator. The entry limit and deadline are checked between entries,
// so one huge directory cannot keep the native promise running past the scan budget.
function defaultListDir(limits: ScanLimits, budget: InspectionBudget): (dir: string) => Promise<DirEntry[]> {
  let entries = 0;
  return async (dir) => {
    const handle = await opendir(dir);
    const out: DirEntry[] = [];
    try {
      for await (const entry of handle) {
        if (budget.expired) throw new BudgetError("deadline");
        entries++;
        if (entries > limits.maxEntries) throw new BudgetError("deadline");
        out.push({ name: entry.name, isDirectory: entry.isDirectory(), isSymbolicLink: entry.isSymbolicLink() });
      }
    } finally {
      await handle.close().catch(() => {});
    }
    return out;
  };
}

// Completed two-level scans only. Any timeout/budget exhaustion OR an incomplete nested scan
// returns no cache and the caller marks every destination unavailable — never a guessed root.
async function discoverRoots(
  root: string, budget: InspectionBudget, limits: ScanLimits, timeoutMs: number,
  canonicalize: Canonicalize, listDir: (dir: string) => Promise<DirEntry[]>,
): Promise<{ ok: true; roots: Map<string, DiscoveredRoot[]> } | { ok: false }> {
  try {
    const rootCanonical = await budget.run(() => canonicalize(root), timeoutMs);
    const cached = rootCache.get(rootCanonical);
    if (cached && Date.now() - cached.at < ROOT_CACHE_TTL_MS) return { ok: true, roots: cached.roots };

    const candidates: { abs: string; rel: string; dir: string; depth: number }[] = [];
    const depthOne: { abs: string; rel: string; dir: string }[] = [];

    async function list(dir: string): Promise<DirEntry[]> {
      await budget.check();
      return budget.run(() => listDir(dir), timeoutMs);
    }

    for (const entry of await list(root)) {
      await budget.check();
      if (!entry.isDirectory || entry.isSymbolicLink) continue;
      const abs = join(root, entry.name);
      if (await isLinkedWorktreeDir(abs, budget, timeoutMs, canonicalize)) continue;
      depthOne.push({ abs, rel: entry.name, dir: entry.name });
      candidates.push({ abs, rel: entry.name, dir: entry.name, depth: 1 });
    }

    for (const parent of depthOne) {
      await budget.check();
      let children: DirEntry[];
      try {
        children = await list(parent.abs);
      } catch (e) {
        // Any incomplete subtree invalidates the whole discovery pass; never cache a partial map.
        if (e instanceof BudgetError) throw e;
        return { ok: false };
      }
      for (const entry of children) {
        await budget.check();
        if (!entry.isDirectory || entry.isSymbolicLink) continue;
        const abs = join(parent.abs, entry.name);
        if (await isLinkedWorktreeDir(abs, budget, timeoutMs, canonicalize)) continue;
        candidates.push({ abs, rel: `${parent.rel}/${entry.name}`, dir: entry.name, depth: 2 });
      }
    }
    if (candidates.length > limits.maxCandidates) return { ok: false };

    type Evaluated = { candidate: (typeof candidates)[number]; owner: boolean; uncertain: boolean };
    const evaluated: Evaluated[] = [];
    for (const candidate of candidates) {
      await budget.check();
      const git = await lstatKind(join(candidate.abs, ".git"), budget, timeoutMs);
      let owner = git === "present";
      let uncertain = git === "unknown";
      if (!owner && !uncertain) {
        const statusPath = join(candidate.abs, ".jax-os", "status.md");
        const statusExists = await lstatKind(statusPath, budget, timeoutMs);
        if (statusExists === "unknown") {
          uncertain = true;
        } else if (statusExists === "present") {
          const read = await readCapped(statusPath, 256 * 1024, candidate.abs, budget, timeoutMs, canonicalize);
          owner = read.ok && parseStatusMd(candidate.dir, read.text, new Date().toISOString()) !== null;
        }
      }
      evaluated.push({ candidate, owner, uncertain });
    }

    // Only directories that actually own themselves enter the root map. A same-basename nested
    // folder without its own ownership marker stays unresolvable instead of shadowing the real
    // destination; two real owners of the same basename remain ambiguous.
    const byDir = new Map<string, { owned: DiscoveredRoot[]; uncertain: boolean }>();
    for (const result of evaluated) {
      const info = byDir.get(result.candidate.dir) ?? { owned: [], uncertain: false };
      if (result.uncertain) info.uncertain = true;
      else if (result.owner) info.owned.push({ abs: result.candidate.abs, rel: result.candidate.rel });
      byDir.set(result.candidate.dir, info);
    }

    const roots = new Map<string, DiscoveredRoot[]>();
    for (const [dir, info] of byDir) {
      if (info.uncertain || info.owned.length === 0) continue;
      if (!PROJECT_LABEL.test(dir)) continue;
      roots.set(dir, info.owned);
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
  const branch = bounded(m.branch, 512);
  // Only native build/diff manifests use `target` as a branch. A spec/plan target is a document
  // path and must never surface as a branch; a deleted/invalid manifest leaves branch unknown.
  const target = m.kind === "build" || m.kind === "diff" ? bounded(m.target, 512) : null;
  return {
    runId, project, repo, kind: m.kind, role: m.role,
    worktree: bounded(m.worktree, 512),
    branch: branch ?? target,
  };
}

type RootResolution = { ok: true; root: DiscoveredRoot; canonical: string } | { ok: false };

async function resolveRoot(
  destination: string, roots: Map<string, DiscoveredRoot[]>, budget: InspectionBudget, timeoutMs: number,
  canonicalize: Canonicalize,
): Promise<RootResolution> {
  const candidates = roots.get(destination) ?? [];
  if (candidates.length !== 1) return { ok: false };
  const candidate = candidates[0];
  try {
    const canonical = await budget.run(() => canonicalize(candidate.abs), timeoutMs);
    return { ok: true, root: candidate, canonical };
  } catch {
    return { ok: false };
  }
}

async function inspectOne(
  input: { destination: string; runId: string },
  deps: Required<Pick<ArtifactDeps, "fsMs" | "gitMs" | "gitProbe">> & {
    reposCanonical: string; roots: Map<string, DiscoveredRoot[]>; budget: InspectionBudget;
    canonicalize: Canonicalize;
  },
): Promise<{ facts: ExecutionArtifactFacts; internal: ExecutionArtifactInternal }> {
  const warnings: ArtifactWarning[] = [];
  const unavailable = (warning: ArtifactWarning): { facts: ExecutionArtifactFacts; internal: ExecutionArtifactInternal } => ({
    facts: {
      destination: input.destination, runId: input.runId, branch: null, branchSource: null,
      report: null, tests: null, warnings: [warning],
    },
    internal: { manifestPath: null, repo: null },
  });
  if (!PROJECT_LABEL.test(input.destination) || !RUN_ID_RE.test(input.runId)) {
    return unavailable("root-unavailable");
  }
  const resolved = await resolveRoot(input.destination, deps.roots, deps.budget, deps.fsMs, deps.canonicalize);
  if (!resolved.ok) return unavailable("root-unavailable");
  const root = resolved.root;
  const canonical = resolved.canonical;

  const manifestPath = join(root.abs, ".local", "runs", input.runId, "manifest.json");
  let manifestRead: Awaited<ReturnType<typeof readCapped>>;
  try {
    manifestRead = await readCapped(manifestPath, 64 * 1024, canonical, deps.budget, deps.fsMs, deps.canonicalize);
  } catch {
    manifestRead = { ok: false, reason: "failed" };
  }
  let manifest: Manifest | null = null;
  if (manifestRead.ok) {
    const parsed = parseManifest(manifestRead.text);
    // Identity: the manifest belongs to this exact destination/run and to this control repo.
    if (parsed && parsed.runId === input.runId && parsed.project === input.destination) {
      try {
        const repoCanonical = await deps.budget.run(() => deps.canonicalize(parsed.repo), deps.fsMs);
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
      const worktreeCanonical = await deps.budget.run(() => deps.canonicalize(manifest.worktree!), deps.fsMs);
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
  const absoluteReports = join(root.abs, ".local", "reports");
  const expectedReports = join(canonical, ".local", "reports");
  let controlReports: string | null = null;
  try {
    // Anchor on the exact canonical report directory, not merely somewhere inside the repository:
    // a `.local/reports` symlink to another repo directory is not the retained report location.
    const resolvedReports = await deps.budget.run(() => deps.canonicalize(absoluteReports), deps.fsMs);
    if (resolvedReports === expectedReports) controlReports = resolvedReports;
  } catch {
    controlReports = null;
  }
  let report: ArtifactLink | null = null;
  let tests: ArtifactLink | null = null;

  if (controlReports !== null
    && await regularFileInside(join(absoluteReports, reportName), controlReports, deps.budget, deps.fsMs, deps.canonicalize)) {
    report = { rel: `${root.rel}/.local/reports/${reportName}` };
  }
  if (controlReports !== null
    && await regularFileInside(join(absoluteReports, testsName), controlReports, deps.budget, deps.fsMs, deps.canonicalize)) {
    tests = { rel: `${root.rel}/.local/reports/${testsName}` };
  }
  if ((report === null || tests === null) && worktree !== null) {
    const worktreeRel = relative(deps.reposCanonical, worktree).split(sep).join("/");
    const absoluteWorktreeReports = join(worktree, ".local", "reports");
    const expectedWorktreeReports = join(worktree, ".local", "reports");
    let worktreeReports: string | null = null;
    try {
      const resolvedReports = await deps.budget.run(() => deps.canonicalize(absoluteWorktreeReports), deps.fsMs);
      if (resolvedReports === expectedWorktreeReports) worktreeReports = resolvedReports;
    } catch {
      worktreeReports = null;
    }
    if (worktreeReports !== null) {
      if (report === null
        && await regularFileInside(join(absoluteWorktreeReports, reportName), worktreeReports, deps.budget, deps.fsMs, deps.canonicalize)) {
        report = { rel: `${worktreeRel}/.local/reports/${reportName}` };
      }
      if (tests === null
        && await regularFileInside(join(absoluteWorktreeReports, testsName), worktreeReports, deps.budget, deps.fsMs, deps.canonicalize)) {
        tests = { rel: `${worktreeRel}/.local/reports/${testsName}` };
      }
    }
  }
  if (report === null) warnings.push("report-missing");
  if (tests === null) warnings.push("tests-missing");

  return {
    facts: {
      destination: input.destination,
      runId: input.runId,
      branch: manifest?.branch ?? null,
      branchSource: manifest?.branch ? "manifest" : null,
      report,
      tests,
      warnings,
    },
    internal: manifest ? { manifestPath, repo: canonical } : { manifestPath: null, repo: null },
  };
}

// Async, independently bounded artifact inspection for a bounded row set. A discovery failure
// keeps known rows: every row reports root-unavailable, never a guessed path or a DB change.
export async function inspectArtifactsDetailed(
  rows: readonly { destination: string; runId: string }[],
  deps: ArtifactDeps = {},
): Promise<{ facts: ExecutionArtifactFacts[]; internal: Map<string, ExecutionArtifactInternal> }> {
  const now = deps.now ?? Date.now;
  const budget = deps.budget ?? new InspectionBudget(4, 2_000, now);
  const repos = deps.reposRoot ?? configuredReposRoot();
  const discoveryMs = deps.discoveryMs ?? 1_000;
  const fsMs = deps.fsMs ?? 250;
  const gitMs = deps.gitMs ?? 1_000;
  const limits = { ...DEFAULT_SCAN_LIMITS, ...deps.scanLimits };
  const gitProbe = deps.gitProbe ?? defaultGitProbe;
  const canonicalize = deps.realpath ?? realpath;
  const listDir = deps.listDir ?? defaultListDir(limits, budget);

  let reposCanonical: string;
  try {
    reposCanonical = await budget.run(() => canonicalize(repos), discoveryMs);
  } catch {
    return {
      facts: rows.map((row) => ({
        destination: row.destination, runId: row.runId, branch: null, branchSource: null,
        report: null, tests: null, warnings: ["root-unavailable" as const],
      })),
      internal: new Map(rows.map((row) => [artifactKey(row.destination, row.runId), { manifestPath: null, repo: null }])),
    };
  }
  const discovered = await discoverRoots(reposCanonical, budget, limits, discoveryMs, canonicalize, listDir);
  if (!discovered.ok) {
    return {
      facts: rows.map((row) => ({
        destination: row.destination, runId: row.runId, branch: null, branchSource: null,
        report: null, tests: null, warnings: ["root-unavailable" as const],
      })),
      internal: new Map(rows.map((row) => [artifactKey(row.destination, row.runId), { manifestPath: null, repo: null }])),
    };
  }
  const inspected = await Promise.all(rows.map((row) => inspectOne(row, {
    reposCanonical, roots: discovered.roots, budget, fsMs, gitMs, gitProbe, canonicalize,
  })));
  return {
    facts: inspected.map((i) => i.facts),
    internal: new Map(inspected.map((i) => [artifactKey(i.facts.destination, i.facts.runId), i.internal])),
  };
}

export async function inspectArtifacts(
  rows: readonly { destination: string; runId: string }[],
  deps: ArtifactDeps = {},
): Promise<ExecutionArtifactFacts[]> {
  return (await inspectArtifactsDetailed(rows, deps)).facts;
}
