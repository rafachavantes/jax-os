// M5 Task 2: bounded process snapshots, proven pane ancestry and deterministic workload groups.
// Ownership comes ONLY from an observed ancestor tree reaching a live pane's PID, re-validated
// against /proc start ticks; logical project/session context stays separately last-known.
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
export const MAX_PROC_READS = 2_048;
export const READ_CONCURRENCY = 8;
export const MAX_GROUPS = 200;
export const MAX_GROUP_DETAILS = 10;

// F9: the command is the remainder after the six numeric/stat fields, so spaces survive.
const PS_RE = /^\s*(\d{1,10})\s+(\d{1,10})\s+(\d{1,10})\s+(\S+)\s+(\d{1,10})\s+(\d{1,10})\s+(.+)$/;

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

export type OwnerReason = "no-pane" | "missing" | "cycle" | "race" | "cross-uid" | "budget";

export type Attribution = {
  ownerByPid: Map<number, string>;
  rootByPid: Map<number, number>;
  reads: number;
  partial: boolean;
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

// Read once for every snapshot PID (bounded), then attribute purely from that map.
export async function readSnapshotStats(
  processes: ProcRow[],
  read: (pid: number) => Promise<string>,
): Promise<{ stats: Map<number, ProcStat>; reads: number; partial: boolean }> {
  const stats = new Map<number, ProcStat>();
  const budget = processes.slice(0, MAX_PROC_READS);
  await mapLimit(budget, READ_CONCURRENCY, async (proc) => {
    try {
      const stat = parseProcStat(await read(proc.pid));
      if (stat?.pid === proc.pid) stats.set(proc.pid, stat);
    } catch {}
  });
  return { stats, reads: budget.length, partial: processes.length > MAX_PROC_READS };
}

export function attributeOwners(
  snapshot: ProcSnapshot,
  panePids: Map<string, number>,
  stats: Map<number, ProcStat>,
  partialReads: boolean,
): Attribution {
  const paneByPid = new Map<number, string>();
  for (const [pane, pid] of panePids) if (Number.isSafeInteger(pid) && pid > 0) paneByPid.set(pid, pane);
  const byPid = new Map(snapshot.processes.map((p) => [p.pid, p]));
  const ownerByPid = new Map<number, string>();
  const rootByPid = new Map<number, number>();
  let partial = partialReads;

  for (const proc of snapshot.processes) {
    const leaf = stats.get(proc.pid);
    if (!leaf) continue; // missing process → unknown, never an owner
    const seen = new Set<number>();
    let current = leaf;
    let root = leaf.pid;
    let crossUid = false;
    for (let depth = 0; depth < MAX_ANCESTORS; depth += 1) {
      if (seen.has(current.pid)) { root = current.pid; break; }
      seen.add(current.pid);
      root = current.pid;
      if (paneByPid.has(current.pid)) {
        ownerByPid.set(proc.pid, paneByPid.get(current.pid) as string);
        break;
      }
      if (current.ppid <= 0 || current.ppid === current.pid) break;
      // A parent observed in the snapshot must agree with /proc (no stale PPID race).
      const parentRow = byPid.get(current.ppid);
      if (parentRow && parentRow.uid !== proc.uid) { crossUid = true; break; }
      const parent = stats.get(current.ppid);
      if (!parent) break;
      const expected = byPid.get(current.pid)?.ppid;
      if (expected !== undefined && expected !== current.ppid) break;
      current = parent;
    }
    if (crossUid) continue;
    rootByPid.set(proc.pid, ownerByPid.has(proc.pid) ? (panePids.get(ownerByPid.get(proc.pid) as string) ?? root) : root);
  }
  return { ownerByPid, rootByPid, reads: 0, partial };
}

// Second pass: a PID recycled between the snapshot and the walk must not inherit ownership.
export async function stabilizeOwners(
  attribution: Attribution,
  read: (pid: number) => Promise<string>,
): Promise<Attribution> {
  if (attribution.ownerByPid.size === 0) return attribution;
  const leaves = [...attribution.ownerByPid.keys()].slice(0, Math.floor(MAX_PROC_READS / 2));
  const before = new Map<number, number>();
  const after = new Map<number, number>();
  await mapLimit(leaves, READ_CONCURRENCY, async (pid) => {
    try {
      const stat = parseProcStat(await read(pid));
      if (stat) before.set(pid, stat.startTime);
    } catch {}
  });
  await mapLimit(leaves, READ_CONCURRENCY, async (pid) => {
    try {
      const stat = parseProcStat(await read(pid));
      if (stat) after.set(pid, stat.startTime);
    } catch {}
  });
  for (const pid of leaves) {
    if (!before.has(pid) || before.get(pid) !== after.get(pid)) attribution.ownerByPid.delete(pid);
  }
  return attribution;
}

export type WorkloadOwner = { kind: "pane"; pane: string } | { kind: "unknown"; reason: OwnerReason };

export type WorkloadPid = {
  pid: number;
  ppid: number;
  state: string;
  ageSeconds: number;
  rssMb: number | null;
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
  purpose: "tmux-pane" | null;
  pids: WorkloadPid[];
  detailsTruncated: boolean;
};

export type WorkloadsSection = {
  observedAt: string;
  state: "ok" | "partial" | "unavailable";
  groups: WorkloadGroup[];
  totalProcesses: number;
  truncated: boolean;
  coverage: { ownershipKnown: number; ownershipUnknown: number; partial: boolean };
  warning?: "tmux-unavailable" | "tmux-unverified" | null;
};

export function executableFamily(comm: string): string {
  const first = comm.trim().split(/\s+/)[0] ?? "";
  const base = first.split("/").pop() ?? "";
  return base || "unknown";
}

export type BuildWorkloadsInput = {
  nowMs: number;
  snapshot: ProcSnapshot;
  attribution: Attribution;
  contexts: Map<string, LastKnownPaneContext>;
};

export function buildWorkloadGroups(input: BuildWorkloadsInput): WorkloadsSection {
  const { snapshot, attribution, contexts } = input;
  const buckets = new Map<string, { owner: WorkloadOwner; rows: ProcRow[] }>();
  for (const proc of snapshot.processes) {
    const family = executableFamily(proc.comm);
    const pane = attribution.ownerByPid.get(proc.pid);
    const owner: WorkloadOwner = pane !== undefined ? { kind: "pane", pane } : { kind: "unknown", reason: "no-pane" };
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
    }));
    return {
      key,
      family: executableFamily(rows[0].comm),
      count: rows.length,
      running: live.length,
      zombies: rows.length - live.length,
      rssMb,
      maxAgeSeconds: ageSeconds.length > 0 ? Math.max(...ageSeconds) : null,
      owner: bucket.owner,
      // Last-known logical context only when the owner is a validated pane; never an inference.
      project: context?.project ?? null,
      session: context?.session ?? null,
      purpose: owner.kind === "pane" ? "tmux-pane" : null,
      pids,
      detailsTruncated: rows.length > MAX_GROUP_DETAILS,
    };
  });

  groups.sort((a, b) => (b.rssMb ?? -1) - (a.rssMb ?? -1) || a.key.localeCompare(b.key));
  const truncated = groups.length > MAX_GROUPS;
  const shown = truncated ? groups.slice(0, MAX_GROUPS) : groups;
  const known = snapshot.processes.filter((p) => attribution.ownerByPid.has(p.pid)).length;
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
    },
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
};

export async function collectWorkloads(input: CollectWorkloadsInput): Promise<WorkloadsSection> {
  const snapshot = parsePs(input.psOutput);
  const { stats, partial } = await readSnapshotStats(snapshot.processes, input.readProc);
  const attribution = await stabilizeOwners(
    attributeOwners(snapshot, input.panePids, stats, partial),
    input.readProc,
  );
  const contexts = new Map<string, LastKnownPaneContext>();
  for (const pane of new Set(attribution.ownerByPid.values())) {
    const context = await input.paneContext(pane, input.incarnation ?? null);
    if (context) contexts.set(pane, context);
  }
  return {
    ...buildWorkloadGroups({ nowMs: input.nowMs, snapshot, attribution, contexts }),
    warning: input.tmuxWarning ?? null,
  };
}
