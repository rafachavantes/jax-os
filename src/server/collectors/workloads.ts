// M5 Task 2/3: bounded process snapshots, proven pane ancestry and deterministic workload groups.
// Ownership comes ONLY from a complete leaf-to-pane chain validated in BOTH a before and an after
// identity pass against the same freshly rechecked live pane/server incarnation. Logical
// project/session context stays separately last-known.
import { readFile } from "node:fs/promises";
import type { LastKnownPaneContext } from "../db/workflows";

export type ProcRow = {
  pid: number;
  ppid: number;
  uid: number;
  stat: string;
  ageSeconds: number;
  rssKb: number;
  comm: string;
};

export type ProcSnapshot = { processes: ProcRow[]; total: number; truncated: boolean; partial: boolean };

export const MAX_PS_PROCESSES = 10_000;
export const MAX_ANCESTORS = 64;
// One before map + one after map, each capped at 1024 unique identities (2048 total), 8 at a time.
export const MAX_PHASE_READS = 1_024;
export const READ_CONCURRENCY = 8;
export const IDENTITY_READ_TIMEOUT_MS = 2_000;
export const IDENTITY_DEADLINE_MS = 5_000;
export const MAX_GROUPS = 200;
export const MAX_GROUP_DETAILS = 10;

// F9: the command is the remainder after the six numeric/stat fields, so spaces survive.
const PS_RE = /^\s*(\d{1,10})\s+(\d{1,10})\s+(\d{1,10})\s+(\S+)\s+(\d{1,10})\s+(\S+)\s+(.+)$/;

export function parsePs(out: string): ProcSnapshot {
  const processes: ProcRow[] = [];
  let malformed = 0;
  for (const line of out.split(/\r?\n/)) {
    if (!line.trim()) continue;
    if (/^\s*PID\b/.test(line)) continue;
    const m = PS_RE.exec(line);
    if (!m) {
      malformed += 1;
      continue;
    }
    processes.push({
      pid: Number(m[1]), ppid: Number(m[2]), uid: Number(m[3]), stat: m[4],
      ageSeconds: Number(m[5]), rssKb: Number(m[6]), comm: m[7].trim(),
    });
  }
  const truncated = processes.length > MAX_PS_PROCESSES;
  return {
    processes: truncated ? processes.slice(0, MAX_PS_PROCESSES) : processes,
    total: processes.length,
    truncated,
    partial: malformed > 0 || truncated,
  };
}

export type ProcStat = { pid: number; comm: string; state: string; ppid: number; startTime: number };

// F9: the command name may contain spaces and parentheses — the LAST ")" is its boundary.
export function parseProcStat(text: string): ProcStat | null {
  const open = text.indexOf("(");
  const close = text.lastIndexOf(")");
  if (open < 0 || close <= open) return null;
  const pid = Number(text.slice(0, open).trim());
  if (!Number.isSafeInteger(pid) || pid <= 0) return null;
  const comm = text.slice(open + 1, close);
  const rest = text.slice(close + 1).trim().split(/\s+/);
  if (rest.length < 20) return null;
  const ppid = Number(rest[1]);
  const startTime = Number(rest[19]);
  if (!Number.isSafeInteger(ppid) || !Number.isSafeInteger(startTime)) return null;
  return { pid, comm, state: rest[0], ppid, startTime };
}

export const PROC_STAT_CAP = 4 * 1024;

export async function readProcStat(pid: number): Promise<string> {
  const text = await readFile(`/proc/${pid}/stat`, "utf8");
  if (Buffer.byteLength(text, "utf8") > PROC_STAT_CAP) throw new Error("proc-stat-too-large");
  return text;
}

export type OwnerReason = "no-pane" | "missing" | "cycle" | "cross-uid" | "race" | "budget" | "depth" | "tmux";

export type Attribution = {
  ownerByPid: Map<number, string>;
  rootByPid: Map<number, number>;
  reasonByPid: Map<number, OwnerReason>;
  partial: boolean;
};

export type LivePaneSnapshot = {
  paneIds: Set<string>;
  incarnation: string | null;
  panePids?: Map<string, number>;
  // Explicit tmux `pane_dead` fact where the snapshot carries it. Only a literal `true` blocks a
  // positive owner; legacy fixtures without the fact keep their existing behavior.
  paneDead?: Map<string, boolean>;
};

async function mapLimit<T>(items: T[], limit: number, fn: (item: T) => Promise<void>): Promise<void> {
  let index = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    for (;;) {
      const i = index++;
      if (i >= items.length) return;
      await fn(items[i]);
    }
  });
  await Promise.all(workers);
}

// Hard ceiling of unsettled ORIGINAL proc reads per runtime, across overlapping polls. A public
// timeout never frees one of these: only the actual read settlement does.
export const MAX_ORIGINAL_IDENTITY_READS = 8;
// Default libuv pool scheduling cap. A configured startup pool size n reserves one slot:
// min(2, max(0, n - 1)); a pool of one/zero or an invalid value admits no identity read.
export const DEFAULT_IDENTITY_CONCURRENCY = 2;

export function identityConcurrency(env: { UV_THREADPOOL_SIZE?: string | undefined } = process.env as { UV_THREADPOOL_SIZE?: string | undefined }): number {
  const raw = env.UV_THREADPOOL_SIZE;
  if (raw === undefined) return DEFAULT_IDENTITY_CONCURRENCY;
  if (!/^\d+$/.test(raw)) return 0;
  const size = Number(raw);
  if (!Number.isSafeInteger(size) || size < 2) return 0;
  return Math.min(DEFAULT_IDENTITY_CONCURRENCY, size - 1);
}

// A read that never obtained (or never settled inside) its permit is reported as budget, not as
// a missing process, so permit exhaustion is visibly partial instead of a false absence.
export const IDENTITY_BUDGET = "identity-budget" as const;
export type IdentityOutcome = ProcStat | null | typeof IDENTITY_BUDGET;
export type IdentityRead = (pid: number, deadline: number) => Promise<IdentityOutcome>;
export type IdentityReader = { read: IdentityRead };

function readWithinAllowance<T>(promise: Promise<T>, ms: number): Promise<T | null> {
  if (ms <= 0) return Promise.resolve(null);
  return new Promise((resolve) => {
    let settled = false;
    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      resolve(null);
    }, ms);
    timer.unref?.();
    promise.then((value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(value);
    });
  });
}

// Collector-local/persistent runtime closure owning the RAW reads. A pid already pending is
// reused (never a second original); the same pid gets a fresh operation only after its original
// settles, so the before/after identity maps stay independent proofs. Scheduling respects the
// libuv-derived concurrency and the injected five-second deadline; a public per-read timeout
// returns unknown without cancelling or duplicating the original operation.
export function createIdentityReader(
  read: (pid: number) => Promise<string>,
  opts: { concurrency?: number; now?: () => number; perReadMs?: number } = {},
): IdentityReader {
  const concurrency = Math.max(0, Math.min(MAX_ORIGINAL_IDENTITY_READS, opts.concurrency ?? DEFAULT_IDENTITY_CONCURRENCY));
  const now = opts.now ?? (() => performance.now());
  const perReadMs = opts.perReadMs ?? IDENTITY_READ_TIMEOUT_MS;
  type Entry = { startedAt: number; promise: Promise<IdentityOutcome> };
  const pending = new Map<number, Entry>();
  const waiters: (() => void)[] = [];
  let active = 0;

  const release = () => {
    active -= 1;
    waiters.shift()?.();
  };

  const acquire = (entry: Entry, deadline: number): Promise<boolean> => new Promise((resolve) => {
    const attempt = () => {
      if (now() >= deadline || now() - entry.startedAt >= perReadMs) {
        resolve(false);
        return;
      }
      if (active < concurrency) {
        active += 1;
        resolve(true);
        return;
      }
      waiters.push(attempt);
    };
    attempt();
  });

  const start = (pid: number, entry: Entry, deadline: number): Promise<IdentityOutcome> =>
    (async () => {
      const admitted = await acquire(entry, deadline);
      try {
        if (!admitted) return IDENTITY_BUDGET;
        const stat = parseProcStat(await read(pid));
        return stat && stat.pid === pid ? stat : null;
      } catch {
        return null; // a settled missing/malformed identity stays explicitly unknown
      } finally {
        if (admitted) release();
        pending.delete(pid);
      }
    })();

  return {
    async read(pid: number, deadline: number): Promise<IdentityOutcome> {
      if (concurrency === 0) return IDENTITY_BUDGET;
      const t = now();
      if (t >= deadline) return IDENTITY_BUDGET;
      let entry = pending.get(pid);
      if (!entry) {
        if (pending.size >= MAX_ORIGINAL_IDENTITY_READS) return IDENTITY_BUDGET;
        const created = { startedAt: t, promise: Promise.resolve(null) } as Entry;
        pending.set(pid, created);
        created.promise = start(pid, created, deadline);
        entry = created;
      }
      const outcome = await readWithinAllowance(entry.promise, entry.startedAt + perReadMs - now());
      return outcome === null ? IDENTITY_BUDGET : outcome;
    },
  };
}

// Priority: live pane PIDs, then every node on a ps-observed chain that reaches a pane, then the
// remainder. The cap applies BEFORE scheduling, so required ancestry is never pushed out first.
export function orderIdentityPids(snapshot: ProcSnapshot, panePids: Map<string, number>): number[] {
  const byPid = new Map(snapshot.processes.map((proc) => [proc.pid, proc]));
  const panePidSet = new Set([...panePids.values()].filter((pid) => Number.isSafeInteger(pid) && pid > 0));
  const required = new Set<number>();
  for (const proc of snapshot.processes) {
    const seen = new Set<number>();
    let current: ProcRow | undefined = proc;
    for (let depth = 0; depth < MAX_ANCESTORS && current !== undefined; depth += 1) {
      if (seen.has(current.pid)) break;
      seen.add(current.pid);
      if (panePidSet.has(current.pid)) {
        for (const pid of seen) required.add(pid);
        break;
      }
      current = byPid.get(current.ppid);
    }
    if (panePidSet.has(proc.pid)) required.add(proc.pid);
  }
  const paneOrder = [...panePidSet].sort((a, b) => a - b);
  const requiredOrder = [...required].filter((pid) => !panePidSet.has(pid)).sort((a, b) => a - b);
  const rest = snapshot.processes
    .map((proc) => proc.pid)
    .filter((pid) => !panePidSet.has(pid) && !required.has(pid))
    .sort((a, b) => a - b);
  return [...paneOrder, ...requiredOrder, ...rest].slice(0, MAX_PHASE_READS);
}

export type IdentityPhase = {
  stats: Map<number, ProcStat>;
  attempted: Set<number>;
  deadlineHit: boolean;
};

// One before/after identity map over the runtime reader. The shared deadline stops NEW scheduling
// only; the reader owns each original read's allowance and releases capacity only on settlement.
export async function readIdentityPhase(
  pids: number[],
  read: IdentityRead,
  opts: { deadline: number; now: () => number },
): Promise<IdentityPhase> {
  const stats = new Map<number, ProcStat>();
  const attempted = new Set<number>();
  let deadlineHit = false;
  await mapLimit(pids, READ_CONCURRENCY, async (pid) => {
    if (opts.now() >= opts.deadline) { deadlineHit = true; return; }
    const outcome = await read(pid, opts.deadline);
    if (outcome === IDENTITY_BUDGET) return; // permit exhaustion stays an explicit budget reason
    attempted.add(pid);
    if (outcome && outcome.pid === pid) stats.set(pid, outcome);
  });
  return { stats, attempted, deadlineHit };
}

export type AttributeInput = {
  snapshot: ProcSnapshot;
  panePids: Map<string, number>;
  paneIncarnation: string | null;
  before: IdentityPhase;
  after: IdentityPhase;
  live: LivePaneSnapshot | null;
  partial: boolean;
};

function consistent(pid: number, before: Map<number, ProcStat>, after: Map<number, ProcStat>, psPpid: number | undefined): "ok" | "missing" | "race" {
  const first = before.get(pid);
  const second = after.get(pid);
  if (!first || !second || first.pid !== pid || second.pid !== pid) return "missing";
  if (first.startTime !== second.startTime || first.ppid !== second.ppid) return "race";
  if (psPpid !== undefined && psPpid !== first.ppid) return "race";
  return "ok";
}

// A positive owner requires EVERY node from leaf to pane to be present in both maps, agree on
// PID/start ticks/PPID, match the ps parent relation and UID, and reach the freshly rechecked
// live pane/server incarnation. Anything else is an explicit unknown reason.
export function attributeOwners(input: AttributeInput): Attribution {
  const { snapshot, panePids, before, after, live } = input;
  const paneByPid = new Map<number, string>();
  for (const [pane, pid] of panePids) if (Number.isSafeInteger(pid) && pid > 0) paneByPid.set(pid, pane);
  const byPid = new Map(snapshot.processes.map((proc) => [proc.pid, proc]));
  const ownerByPid = new Map<number, string>();
  const rootByPid = new Map<number, number>();
  const reasonByPid = new Map<number, OwnerReason>();
  let partial = input.partial || before.deadlineHit || after.deadlineHit;

  for (const proc of snapshot.processes) {
    const unknown = (reason: OwnerReason, root = proc.pid) => {
      reasonByPid.set(proc.pid, reason);
      rootByPid.set(proc.pid, root);
      if (reason !== "no-pane") partial = true;
    };
    if (!before.attempted.has(proc.pid) && !after.attempted.has(proc.pid)) {
      unknown("budget");
      continue;
    }
    const leafLeaf = consistent(proc.pid, before.stats, after.stats, byPid.get(proc.pid)?.ppid);
    if (leafLeaf === "missing") {
      const attempted = before.attempted.has(proc.pid) || after.attempted.has(proc.pid);
      unknown(attempted ? "missing" : "budget");
      continue;
    }
    if (leafLeaf === "race") { unknown("race"); continue; }

    const seen = new Set<number>();
    let currentPid = proc.pid;
    let root = proc.pid;
    let failed: OwnerReason | null = null;
    let pane: string | null = null;
    let panePid: number | null = null;
    for (let depth = 0; depth < MAX_ANCESTORS; depth += 1) {
      if (seen.has(currentPid)) { failed = "cycle"; break; }
      seen.add(currentPid);
      root = currentPid;
      const paneName = paneByPid.get(currentPid);
      if (paneName !== undefined) { pane = paneName; panePid = currentPid; break; }
      const psRow = byPid.get(currentPid);
      const stat = before.stats.get(currentPid);
      if (!psRow || !stat) { failed = "missing"; break; }
      if (stat.ppid <= 0 || stat.ppid === currentPid) { failed = "no-pane"; break; }
      const parentRow = byPid.get(stat.ppid);
      if (!parentRow) { failed = stat.ppid <= 1 ? "no-pane" : "missing"; break; }
      if (parentRow.uid !== proc.uid) { failed = "cross-uid"; break; }
      const relation = consistent(stat.ppid, before.stats, after.stats, parentRow.ppid);
      if (relation === "missing") { failed = "missing"; break; }
      if (relation === "race") { failed = "race"; break; }
      currentPid = stat.ppid;
    }
    if (pane === null || panePid === null) {
      unknown(failed ?? "depth", root);
      continue;
    }
    const paneStat = consistent(panePid, before.stats, after.stats, byPid.get(panePid)?.ppid);
    if (paneStat !== "ok") { unknown(paneStat === "missing" ? "missing" : "race"); continue; }
    if (live === null
      || !live.paneIds.has(pane)
      || live.paneDead?.get(pane) === true // explicit dead pane never proves a live owner
      || live.incarnation !== input.paneIncarnation
      || live.panePids?.get(pane) !== panePid) {
      unknown("race");
      continue;
    }
    ownerByPid.set(proc.pid, pane);
    rootByPid.set(proc.pid, panePid);
  }
  return { ownerByPid, rootByPid, reasonByPid, partial };
}

export type WorkloadOwner = { kind: "pane"; pane: string } | { kind: "unknown"; reason: OwnerReason };

export type WorkloadPid = {
  pid: number;
  ppid: number;
  state: string;
  ageSeconds: number;
  rssMb: number | null;
  parentCommand: string | null;
};

export type WorkloadGroup = {
  key: string;
  family: string;
  count: number;
  running: number;
  zombies: number;
  rssMb: number | null;
  maxAgeSeconds: number | null;
  owner: WorkloadOwner;
  project: string | null;
  session: string | null;
  contextObservedAt: string | null;
  purpose: "tmux-pane" | null;
  pids: WorkloadPid[];
  detailsTruncated: boolean;
};

export type WorkloadCoverage = {
  ownershipKnown: number;
  ownershipUnknown: number;
  partial: boolean;
  reasons: Partial<Record<OwnerReason, number>>;
};

export type WorkloadsSection = {
  observedAt: string;
  state: "ok" | "partial" | "unavailable";
  groups: WorkloadGroup[];
  totalProcesses: number;
  truncated: boolean;
  coverage: WorkloadCoverage;
  warning?: "tmux-unavailable" | "tmux-unverified" | null;
};

const BROWSER_HELPERS: [RegExp, string][] = [
  [/^(google chrome|chrome|chromium)/i, "chrome"],
  [/^firefox/i, "firefox"],
];

// Bounded executable family: a fixed browser-helper map preserves meaningful names; otherwise the
// leading token is kept (bounded). This is never evidence of purpose or safe-to-kill status.
export function executableFamily(comm: string): string {
  const whole = comm.trim();
  for (const [pattern, family] of BROWSER_HELPERS) if (pattern.test(whole)) return family;
  const first = whole.split(/\s+/)[0] ?? "";
  const base = first.split("/").pop() ?? "";
  return base.slice(0, 64) || "unknown";
}

export type BuildWorkloadsInput = {
  nowMs: number;
  snapshot: ProcSnapshot;
  attribution: Attribution;
  contexts: Map<string, LastKnownPaneContext>;
  warning?: "tmux-unavailable" | "tmux-unverified" | null;
};

export function buildWorkloadGroups(input: BuildWorkloadsInput): WorkloadsSection {
  const { snapshot, attribution, contexts } = input;
  const byPid = new Map(snapshot.processes.map((proc) => [proc.pid, proc]));
  const buckets = new Map<string, { owner: WorkloadOwner; rows: ProcRow[] }>();
  for (const proc of snapshot.processes) {
    const family = executableFamily(proc.comm);
    const pane = attribution.ownerByPid.get(proc.pid);
    const owner: WorkloadOwner = pane !== undefined
      ? { kind: "pane", pane }
      : { kind: "unknown", reason: attribution.reasonByPid.get(proc.pid) ?? "no-pane" };
    const root = attribution.rootByPid.get(proc.pid) ?? proc.pid;
    const key = pane !== undefined
      ? `pane:${pane}:${family}`
      : `unknown:${proc.uid}:${family}:${root}`;
    const bucket = buckets.get(key) ?? { owner, rows: [] };
    bucket.rows.push(proc);
    buckets.set(key, bucket);
  }

  const groups: WorkloadGroup[] = [...buckets.entries()].map(([key, bucket]) => {
    const rows = bucket.rows;
    const live = rows.filter((r) => !r.stat.startsWith("Z"));
    const rss = rows.map((r) => (Number.isFinite(r.rssKb) && r.rssKb >= 0 ? r.rssKb / 1024 : null));
    const rssMb = rss.some((v) => v !== null) ? rss.reduce<number>((sum, v) => sum + (v ?? 0), 0) : null;
    const ageSeconds = rows.map((r) => r.ageSeconds).filter((v) => Number.isFinite(v));
    const owner = bucket.owner;
    const context = owner.kind === "pane" ? contexts.get(owner.pane) : undefined;
    const pids = rows.slice(0, MAX_GROUP_DETAILS).map((r) => ({
      pid: r.pid, ppid: r.ppid, state: r.stat,
      ageSeconds: r.ageSeconds,
      rssMb: Number.isFinite(r.rssKb) && r.rssKb >= 0 ? r.rssKb / 1024 : null,
      parentCommand: r.stat.startsWith("Z") ? byPid.get(r.ppid)?.comm ?? null : null,
    }));
    return {
      key,
      family: executableFamily(rows[0].comm),
      count: rows.length,
      running: live.length,
      zombies: rows.length - live.length,
      rssMb,
      maxAgeSeconds: ageSeconds.length > 0 ? Math.max(...ageSeconds) : null,
      owner,
      // Last-known logical context only when the owner is a validated pane; never an inference.
      project: context?.project ?? null,
      session: context?.session ?? null,
      contextObservedAt: context?.observedAt ?? null,
      purpose: owner.kind === "pane" ? "tmux-pane" : null,
      pids,
      detailsTruncated: rows.length > MAX_GROUP_DETAILS,
    };
  });

  groups.sort((a, b) => (b.rssMb ?? -1) - (a.rssMb ?? -1) || a.key.localeCompare(b.key));
  const truncated = groups.length > MAX_GROUPS;
  const shown = truncated ? groups.slice(0, MAX_GROUPS) : groups;
  const known = snapshot.processes.filter((p) => attribution.ownerByPid.has(p.pid)).length;
  const reasons: Partial<Record<OwnerReason, number>> = {};
  for (const reason of attribution.reasonByPid.values()) {
    reasons[reason] = (reasons[reason] ?? 0) + 1;
  }
  return {
    observedAt: new Date(input.nowMs).toISOString(),
    state: snapshot.partial || attribution.partial || truncated ? "partial" : "ok",
    groups: shown,
    totalProcesses: snapshot.total,
    truncated,
    coverage: {
      ownershipKnown: known,
      ownershipUnknown: snapshot.processes.length - known,
      partial: snapshot.partial || attribution.partial,
      reasons,
    },
    warning: input.warning ?? null,
  };
}

export type CollectWorkloadsInput = {
  nowMs: number;
  psOutput: string;
  panePids: Map<string, number>;
  readProc: (pid: number) => Promise<string>;
  paneContext: (pane: string, incarnation: string | null) => LastKnownPaneContext | null | Promise<LastKnownPaneContext | null>;
  incarnation?: string | null;
  tmuxWarning?: "tmux-unavailable" | "tmux-unverified" | null;
  // A FRESH pane snapshot taken after the identity reads; a changed/closed/respawned pane yields
  // unknown ownership instead of a second owner.
  verifyPanes?: () => Promise<LivePaneSnapshot>;
  monotonic?: () => number;
  // Persistent per-runtime raw-read ownership; a direct caller gets its own bounded instance.
  identityReader?: IdentityReader;
};

export async function collectWorkloads(input: CollectWorkloadsInput): Promise<WorkloadsSection> {
  const snapshot = parsePs(input.psOutput);
  const now = input.monotonic ?? (() => performance.now());
  const deadline = now() + IDENTITY_DEADLINE_MS;
  const pids = orderIdentityPids(snapshot, input.panePids);
  const identity = input.identityReader ?? createIdentityReader(input.readProc, { now });
  const before = await readIdentityPhase(pids, identity.read, { deadline, now });
  const after = await readIdentityPhase(pids, identity.read, { deadline, now });

  let live: LivePaneSnapshot | null = null;
  let warning = input.tmuxWarning ?? null;
  if (input.verifyPanes) {
    try {
      live = await input.verifyPanes();
    } catch {
      live = null;
      warning = "tmux-unavailable";
    }
  } else {
    live = {
      paneIds: new Set(input.panePids.keys()),
      incarnation: input.incarnation ?? null,
      panePids: input.panePids,
    };
  }
  const attribution = attributeOwners({
    snapshot,
    panePids: input.panePids,
    paneIncarnation: input.incarnation ?? null,
    before,
    after,
    live,
    partial: snapshot.partial || warning !== null,
  });
  const contexts = new Map<string, LastKnownPaneContext>();
  for (const pane of new Set(attribution.ownerByPid.values())) {
    const context = await input.paneContext(pane, input.incarnation ?? null);
    if (context) contexts.set(pane, context);
  }
  return buildWorkloadGroups({ nowMs: input.nowMs, snapshot, attribution, contexts, warning });
}
