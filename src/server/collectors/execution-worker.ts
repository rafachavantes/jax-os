import { constants } from "node:fs";
import { open, readlink, realpath } from "node:fs/promises";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { basename, dirname } from "node:path";
import { BudgetError, InspectionBudget, type BudgetLike } from "./execution-artifacts";
import { JAXFLOW_SCRIPT } from "./workflow-spawn";
import type { ExecutionRow } from "../../lib/executions";
import type { CurrentExecutionState } from "../db/workflows";

// M3a, Task 3 — current wrapper observation only. A positive result proves exactly one thing:
// the recorded run's fixed jaxflow worker launcher is alive in its recorded tmux pane. It never
// claims provider, child-process or model progress, and it never invents a terminal outcome.

export const WORKER_WARNINGS = [
  "tmux-unavailable",
  "session-unknown",
  "manifest-unverified",
  "launcher-unverified",
  "observation-timeout",
  "observation-ambiguous",
  "timestamp-unknown",
] as const;
export type WorkerWarning = (typeof WORKER_WARNINGS)[number];

export type WorkerStatus =
  | "observed"
  | "awaiting-confirmation"
  | "no-worker-observed"
  | "unknown"
  | "not-checked"
  | "updated-refresh-needed";

export type WorkerObservation = {
  status: WorkerStatus;
  warning?: WorkerWarning;
  lastGoodAt?: string;
  observedAt?: string;
  pid?: number;
};

export type PaneInfo = { session: string; paneId: string; panePid: number };

export function parsePaneSnapshot(text: string): PaneInfo[] {
  const panes: PaneInfo[] = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    const match = /^(%[0-9]{1,10}) ([0-9]{1,10}) (.+)$/.exec(line);
    if (!match) continue;
    const panePid = Number(match[2]);
    if (!Number.isSafeInteger(panePid) || panePid <= 0) continue;
    panes.push({ paneId: match[1], panePid, session: match[3] });
  }
  return panes;
}

export type ProcStat = { pid: number; ppid: number; startTicks: string };

// `/proc/<pid>/stat` — comm may contain spaces AND parentheses, so parse from the LAST ')'.
export function parseProcStat(text: string): ProcStat | null {
  const open = text.indexOf("(");
  const close = text.lastIndexOf(")");
  if (open <= 0 || close <= open) return null;
  const pid = Number(text.slice(0, open).trim());
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  const rest = text.slice(close + 1).trim().split(/\s+/);
  if (rest.length < 20) return null;
  const ppid = Number(rest[1]);
  const startTicks = rest[19];
  if (!Number.isSafeInteger(ppid) || !/^[0-9]{1,20}$/.test(startTicks)) return null;
  return { pid, ppid, startTicks };
}

export type ProcReader = {
  stat(pid: number): Promise<string | null>;
  children(pid: number): Promise<number[] | null>;
  cmdline(pid: number): Promise<string[] | null>;
  cwd(pid: number): Promise<string | null>;
  exe(pid: number): Promise<string | null>;
};

async function readProcCapped(path: string, cap: number): Promise<string | null> {
  let fd;
  try {
    fd = await open(path, constants.O_RDONLY);
  } catch {
    return null;
  }
  try {
    const buffer = Buffer.alloc(cap + 1);
    const { bytesRead } = await fd.read(buffer, 0, buffer.length, 0);
    if (bytesRead > cap) return null;
    return buffer.subarray(0, bytesRead).toString("utf8");
  } catch {
    return null;
  } finally {
    await fd.close().catch(() => {});
  }
}

export const defaultProcReader: ProcReader = {
  async stat(pid) {
    return readProcCapped(`/proc/${pid}/stat`, 4 * 1024);
  },
  async children(pid) {
    const text = await readProcCapped(`/proc/${pid}/task/${pid}/children`, 4 * 1024);
    if (text === null) return null;
    const ids = text.split(/\s+/).filter(Boolean).map(Number);
    return ids.every((id) => Number.isSafeInteger(id) && id > 0) ? ids : null;
  },
  async cmdline(pid) {
    const text = await readProcCapped(`/proc/${pid}/cmdline`, 16 * 1024);
    if (text === null) return null;
    const argv = text.split("\0");
    if (argv.at(-1) === "") argv.pop();
    return argv.length > 0 && argv.every((part) => !part.includes("\0")) ? argv : null;
  },
  async cwd(pid) {
    try {
      return await readlink(`/proc/${pid}/cwd`);
    } catch {
      return null;
    }
  },
  async exe(pid) {
    try {
      return await readlink(`/proc/${pid}/exe`);
    } catch {
      return null;
    }
  },
};

const execFileAsync = promisify(execFile);

export async function defaultTmuxSnapshot(): Promise<string> {
  const { stdout } = await execFileAsync(
    "tmux",
    ["list-panes", "-a", "-F", "#{pane_id} #{pane_pid} #{session_name}"],
    { encoding: "utf8", timeout: 500, maxBuffer: 256 * 1024 },
  );
  return stdout;
}

const NO_SERVER_RE = /no server running/i;
// ENOENT/no-socket is a demonstrated "nothing running" condition. Permission and connection
// failures are real source errors and stay unknown.
const NO_SOCKET_RE = /error connecting to .*(?:no such file or directory|enoent)/i;
const WORKER_DEADLINE_MS = 1_000;

export type WorkerDeps = {
  tmux?: () => Promise<string>;
  proc?: ProcReader;
  realpath?: (path: string) => Promise<string>;
  now?: () => number;
  budget?: InspectionBudget;
  perReadMs?: number;
  maxRuns?: number;
  workerMs?: number;
  launcherPath?: string;
  // Volatile process-memory last-good observations; injectable for tests, bounded regardless.
  cache?: Map<string, { at: number; observedAt: string }>;
};

export type WorkerTarget = {
  destination: string;
  runId: string;
  startEventId: number | null;
  startTs: string | null;
  session: string | null;
  manifestPath: string | null;
  repo: string | null;
};

export function workerKey(target: Pick<WorkerTarget, "destination" | "runId" | "startEventId">): string {
  return `${target.destination}/${target.runId}/${target.startEventId ?? "none"}`;
}

const SHELLS = new Set(["sh", "bash", "dash", "zsh", "fish", "ksh", "ash", "busybox"]);
const MAX_CANDIDATES = 8;
const MAX_DEPTH = 2;
const LAST_GOOD_MAX = 200;
const LAST_GOOD_TTL_MS = 60_000;

type LastGood = { at: number; observedAt: string };
const lastGood = new Map<string, LastGood>();

function remember(store: Map<string, LastGood>, key: string, nowMs: number, observedAt: string): void {
  store.set(key, { at: nowMs, observedAt });
  while (store.size > LAST_GOOD_MAX) {
    const oldest = store.keys().next().value;
    if (oldest === undefined) break;
    store.delete(oldest);
  }
}

function priorObservation(store: Map<string, LastGood>, key: string, nowMs: number): LastGood | null {
  const previous = store.get(key);
  if (!previous) return null;
  if (nowMs - previous.at > LAST_GOOD_TTL_MS) {
    store.delete(key);
    return null;
  }
  return previous;
}

const singleFlight = new WeakMap<() => Promise<string>, Promise<string>>();

async function snapshotTmux(run: () => Promise<string>): Promise<string> {
  const pending = singleFlight.get(run);
  if (pending) return pending;
  const next = run();
  singleFlight.set(run, next);
  try {
    return await next;
  } finally {
    singleFlight.delete(run);
  }
}

type Identity = { pid: number; startTicks: string };

async function readStable(proc: ProcReader, pid: number): Promise<Identity | null> {
  const text = await proc.stat(pid);
  if (text === null) return null;
  const parsed = parseProcStat(text);
  return parsed && parsed.pid === pid ? { pid, startTicks: parsed.startTicks } : null;
}

async function collectCandidates(
  rootPid: number, proc: ProcReader, readTimeout: number, budget: BudgetLike,
): Promise<number[]> {
  const rootStat = await budget.run(() => readStable(proc, rootPid), readTimeout);
  if (!rootStat) return [];
  const rootArgv = await budget.run(() => proc.cmdline(rootPid), readTimeout);
  const rootIsShell = rootArgv !== null && rootArgv.length > 0 && SHELLS.has(basename(rootArgv[0]).toLowerCase());
  if (!rootIsShell) return [rootPid];
  const out = [rootPid];
  let level: number[] = [rootPid];
  for (let depth = 1; depth <= MAX_DEPTH && out.length < MAX_CANDIDATES; depth++) {
    const next: number[] = [];
    for (const pid of level) {
      const children = await budget.run(() => proc.children(pid), readTimeout);
      if (!children) continue;
      for (const child of children) {
        if (out.length >= MAX_CANDIDATES) break;
        out.push(child);
        const childArgv = await budget.run(() => proc.cmdline(child), readTimeout);
        if (childArgv && childArgv.length > 0 && SHELLS.has(basename(childArgv[0]).toLowerCase())) next.push(child);
      }
    }
    level = next;
  }
  return out;
}

async function canonicalOrNull(
  canonicalize: (p: string) => Promise<string>, path: string | null, budget: BudgetLike, timeoutMs: number,
): Promise<string | null> {
  if (path === null) return null;
  try {
    return await budget.run(() => canonicalize(path), timeoutMs);
  } catch (e) {
    if (e instanceof BudgetError) throw e;
    return null;
  }
}

function isPythonInterpreter(path: string): boolean {
  return /^python[0-9]*(?:\.[0-9]+)?$/.test(basename(path));
}

type WrapperIdentity = {
  manifestCanonical: string;
  repoCanonical: string;
  launcherCanonical: string;
};

// The actual four-argument launcher tuple: interpreter, the exact canonical installed control-app
// scripts/jaxflow.py, --run-worker, exact manifest. Command/session names, a same-basename script
// elsewhere, or any other four-argument Python invocation never prove a worker.
async function matchWrapper(
  pid: number, identity: WrapperIdentity, proc: ProcReader, realpath: (p: string) => Promise<string>,
  budget: BudgetLike, readTimeout: number,
): Promise<Identity | "no" | "ambiguous"> {
  const before = await budget.run(() => readStable(proc, pid), readTimeout);
  if (!before) return "no";
  const argv = await budget.run(() => proc.cmdline(pid), readTimeout);
  if (!argv) return "no";
  const [interpreter, script, flag, manifest] = argv;
  if (argv.length !== 4 || flag !== "--run-worker") return "no";
  if (basename(script) !== "jaxflow.py" || basename(dirname(script)) !== "scripts") return "no";
  if (!isPythonInterpreter(interpreter)) return "no";
  const scriptCanonical = await canonicalOrNull(realpath, script, budget, readTimeout);
  if (scriptCanonical === null || scriptCanonical !== identity.launcherCanonical) return "no";
  const manifestCanonical = await canonicalOrNull(realpath, manifest, budget, readTimeout);
  if (manifestCanonical === null || manifestCanonical !== identity.manifestCanonical) return "no";
  const exe = await budget.run(() => proc.exe(pid), readTimeout);
  const exeCanonical = await canonicalOrNull(realpath, exe, budget, readTimeout);
  const interpreterCanonical = await canonicalOrNull(realpath, interpreter, budget, readTimeout);
  if (exeCanonical === null || interpreterCanonical === null || exeCanonical !== interpreterCanonical) return "no";
  const cwd = await budget.run(() => proc.cwd(pid), readTimeout);
  const cwdCanonical = await canonicalOrNull(realpath, cwd, budget, readTimeout);
  if (cwdCanonical === null || cwdCanonical !== identity.repoCanonical) return "no";
  const after = await budget.run(() => readStable(proc, pid), readTimeout);
  if (!after || after.startTicks !== before.startTicks) return "no";
  return before;
}

async function observeTarget(
  target: WorkerTarget, panes: PaneInfo[], deps: Required<Pick<WorkerDeps, "proc" | "realpath" | "now" | "launcherPath">> & {
    budget: BudgetLike; perReadMs: number; cache: Map<string, LastGood>;
  },
): Promise<WorkerObservation> {
  const key = workerKey(target);
  const nowMs = deps.now();
  if (target.session === null) return { status: "unknown", warning: "session-unknown" };
  if (target.manifestPath === null || target.repo === null) return { status: "unknown", warning: "manifest-unverified" };

  try {
    const budget = deps.budget;
    const manifestCanonical = await canonicalOrNull(deps.realpath, target.manifestPath, budget, deps.perReadMs);
    const repoCanonical = await canonicalOrNull(deps.realpath, target.repo, budget, deps.perReadMs);
    const launcherCanonical = await canonicalOrNull(deps.realpath, deps.launcherPath, budget, deps.perReadMs);
    if (launcherCanonical === null) return { status: "unknown", warning: "launcher-unverified" };
    if (manifestCanonical === null || repoCanonical === null) return { status: "unknown", warning: "manifest-unverified" };

    const sessionPanes = panes.filter((p) => p.session === target.session);
    if (sessionPanes.length === 0) {
      const previous = priorObservation(deps.cache, key, nowMs);
      if (previous) return { status: "no-worker-observed", lastGoodAt: previous.observedAt };
      const started = target.startTs === null ? NaN : Date.parse(target.startTs);
      if (!Number.isFinite(started)) return { status: "unknown", warning: "timestamp-unknown" };
      return started >= nowMs - 60_000
        ? { status: "awaiting-confirmation" }
        : { status: "no-worker-observed" };
    }
    if (sessionPanes.length > 1) return { status: "unknown", warning: "observation-ambiguous" };

    const pids = await collectCandidates(sessionPanes[0].panePid, deps.proc, deps.perReadMs, budget);
    if (pids.length === 0) return { status: "unknown", warning: "observation-timeout" };
    const matches: Identity[] = [];
    for (const pid of pids) {
      const result = await matchWrapper(
        pid, { manifestCanonical, repoCanonical, launcherCanonical }, deps.proc, deps.realpath, budget, deps.perReadMs,
      );
      if (result === "ambiguous") return { status: "unknown", warning: "observation-ambiguous" };
      if (result !== "no") matches.push(result);
    }
    if (matches.length > 1) return { status: "unknown", warning: "observation-ambiguous" };
    if (matches.length === 1) {
      const observedAt = new Date(nowMs).toISOString();
      remember(deps.cache, key, nowMs, observedAt);
      return { status: "observed", pid: matches[0].pid, observedAt };
    }
    const previous = priorObservation(deps.cache, key, nowMs);
    if (previous) return { status: "no-worker-observed", lastGoodAt: previous.observedAt };
    const started = target.startTs === null ? NaN : Date.parse(target.startTs);
    if (!Number.isFinite(started)) return { status: "unknown", warning: "timestamp-unknown" };
    return started >= nowMs - 60_000
      ? { status: "awaiting-confirmation" }
      : { status: "no-worker-observed" };
  } catch (e) {
    if (e instanceof BudgetError) {
      const previous = priorObservation(deps.cache, key, nowMs);
      return { status: "unknown", warning: "observation-timeout", ...(previous ? { lastGoodAt: previous.observedAt } : {}) };
    }
    return { status: "unknown", warning: "observation-timeout" };
  }
}

// Bounded pass: at most 12 open displayed runs are checked; the rest are not-checked, never
// fabricated active. One fixed tmux snapshot per pass, single-flight. The worker subdeadline is
// 1s inside the shared 2s pass budget; no second capacity pool is created.
export async function inspectWorkers(
  targets: readonly WorkerTarget[], deps: WorkerDeps = {},
): Promise<Map<string, WorkerObservation>> {
  const now = deps.now ?? Date.now;
  const shared = deps.budget ?? new InspectionBudget(4, WORKER_DEADLINE_MS, now);
  const budget = shared.scope(deps.workerMs ?? WORKER_DEADLINE_MS);
  const proc = deps.proc ?? defaultProcReader;
  const canonical = deps.realpath ?? realpath;
  const tmux = deps.tmux ?? defaultTmuxSnapshot;
  const perReadMs = deps.perReadMs ?? 100;
  const maxRuns = deps.maxRuns ?? 12;
  const cache = deps.cache ?? lastGood;
  const launcherPath = deps.launcherPath ?? JAXFLOW_SCRIPT;
  const checked = targets.slice(0, maxRuns);
  const result = new Map<string, WorkerObservation>();
  for (const target of targets.slice(maxRuns)) result.set(workerKey(target), { status: "not-checked" });
  // No eligible open row means no process scan at all.
  if (checked.length === 0) return result;

  let panes: PaneInfo[];
  try {
    panes = parsePaneSnapshot(await budget.run(() => snapshotTmux(tmux), 500));
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    if (e instanceof BudgetError || !(NO_SERVER_RE.test(message) || NO_SOCKET_RE.test(message))) {
      for (const target of checked) result.set(workerKey(target), { status: "unknown", warning: "tmux-unavailable" });
      return result;
    }
    panes = []; // demonstrated no-server/no-socket: a healthy empty snapshot, never a source failure
  }

  const observations = await Promise.all(checked.map((target) =>
    observeTarget(target, panes, { proc, realpath: canonical, now, budget, perReadMs, cache, launcherPath })));
  checked.forEach((target, index) => result.set(workerKey(target), observations[index]));
  return result;
}

// Frozen-page overlay: a terminal observed after the page read wins over any local label; a
// newer start makes the displayed attempt stale (updated-refresh-needed) and suppresses worker
// classification for the old attempt. Page membership/order stays frozen.
export function applyCurrentOverlay(
  row: ExecutionRow,
  current: CurrentExecutionState | null,
): { row: ExecutionRow; refreshNeeded: boolean; workerSuppressed: boolean } {
  if (current === null || current.row === null) return { row, refreshNeeded: false, workerSuppressed: false };
  const newerStart = current.startEventId !== row.startEventId;
  if (newerStart) return { row, refreshNeeded: true, workerSuppressed: true };
  const live = current.row;
  if (!live.terminal) return { row, refreshNeeded: false, workerSuppressed: false };
  if (row.terminal && row.finishEventId === live.finishEventId) {
    return { row, refreshNeeded: false, workerSuppressed: false };
  }
  return {
    row: {
      ...row,
      outcome: live.outcome,
      terminal: true,
      contractStatus: live.contractStatus,
      finishEventId: live.finishEventId,
      finishedAt: live.finishedAt,
      durationMs: live.durationMs,
      warnings: live.warnings,
    },
    refreshNeeded: false,
    workerSuppressed: true,
  };
}
