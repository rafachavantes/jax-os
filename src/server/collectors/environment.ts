// M5 Task 1: Environment diagnostics — typed readers for components, /tmp capacity and package
// candidates. Every external fact comes through an injected async seam; pure parsers carry the
// fixture tests. Nothing here mutates host state: no cleanup, no package execution, no config.
import { execFile } from "node:child_process";
import { readFile, statfs } from "node:fs/promises";
import { realpathSync } from "node:fs";
import { promisify } from "node:util";
import type Database from "better-sqlite3";
import { envFileStatus, type EnvFileState } from "../envFile";
import { getDb } from "../db";
import { checkOwnDatabase, type OwnDbCheck } from "../db/environment";
import { lastKnownPaneContext, type LastKnownPaneContext } from "../db/workflows";
import {
  parseSettings, readRawSettings, settingsKeyPresence, type KeyPresence, type RawSettingsRead, type SettingsResult,
} from "../settings";
import { isCliPresent } from "./cli-presence";
import { getProjects } from "./projects";
import {
  packageKey, parseHoldRows, parseSimulationRows,
  type PackageObservation,
} from "./alerts";
import { listLivePanes } from "./workflow-tmux";
import { collectWorkloads, readProcStat, type WorkloadsSection } from "./workloads";

export const COMPONENT_IDS = [
  "database", "projects", "tmux", "ttyd", "github", "linear",
  "claude", "codex", "opencode", "metrics", "env-file",
] as const;
export type ComponentId = (typeof COMPONENT_IDS)[number];

export type ComponentState =
  | "healthy" | "needs-configuration" | "failed" | "disabled-default"
  | "disabled-explicit" | "unavailable" | "unknown" | "not-applicable";

export type ComponentResult = {
  id: ComponentId;
  state: ComponentState;
  // Bounded machine code: what was actually observed. Never prose, never a secret.
  detail: string | null;
  observedAt: string;
  lastGoodAt: string | null;
};

export type ComponentsSection = { observedAt: string; components: ComponentResult[] };

export type MountInfo = {
  mountId: string;
  parentId: string;
  majorMinor: string;
  root: string;
  mountPoint: string;
  fsType: string;
};

// mountinfo escapes space/tab/newline/backslash as octal; decode only those documented escapes.
export function decodeMountField(value: string): string {
  return value.replace(/\\([0-7]{3})/g, (_m, oct: string) => String.fromCharCode(parseInt(oct, 8)));
}

export function parseMountInfo(text: string): MountInfo[] {
  const out: MountInfo[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const split = line.indexOf(" - ");
    if (split < 0) continue;
    const left = line.slice(0, split).split(" ");
    const right = line.slice(split + 3).split(" ");
    if (left.length < 6 || right.length < 1) continue;
    out.push({
      mountId: left[0], parentId: left[1], majorMinor: left[2],
      root: decodeMountField(left[3]), mountPoint: decodeMountField(left[4]), fsType: right[0],
    });
  }
  return out;
}

function normalizedMountPoint(mountPoint: string): string {
  if (mountPoint === "/") return "/";
  return mountPoint.replace(/\/+$/, "");
}

// F4: longest matching mount prefix; among equal prefixes the LAST (overmount) wins.
export function selectMount(mounts: MountInfo[], path: string): MountInfo | null {
  let best: MountInfo | null = null;
  let bestLen = -1;
  for (const mount of mounts) {
    const mp = normalizedMountPoint(mount.mountPoint);
    const matches = path === mp || path.startsWith(mp === "/" ? "/" : `${mp}/`);
    if (!matches) continue;
    if (mp.length >= bestLen) {
      best = mount;
      bestLen = mp.length;
    }
  }
  return best;
}

export type StatfsFacts = { bsize: number; blocks: number; bfree: number; bavail: number };

export type TmpCapacity = {
  path: string;
  mountPath: string;
  fsType: string;
  tmpfs: boolean;
  sharedRoot: boolean;
  usedGb: number;
  totalGb: number;
  availableGb: number;
};

export function capacityFromStatfs(facts: StatfsFacts): Pick<TmpCapacity, "usedGb" | "totalGb" | "availableGb"> | null {
  const values = [facts.bsize, facts.blocks, facts.bfree, facts.bavail];
  if (values.some((v) => typeof v !== "number" || !Number.isFinite(v) || v < 0)) return null;
  const total = facts.blocks * facts.bsize;
  if (total <= 0) return null;
  const used = Math.max(0, (facts.blocks - facts.bfree) * facts.bsize);
  return { usedGb: used / 1e9, totalGb: total / 1e9, availableGb: (facts.bavail * facts.bsize) / 1e9 };
}

export type ResourcesSection = {
  observedAt: string;
  lastGoodAt: string | null;
  state: "ok" | "unavailable";
  stale: boolean;
  tmp: TmpCapacity | null;
  warning: "mount-facts" | "statfs-failed" | null;
};

export type PackageCandidate = {
  key: string;
  name: string;
  arch: string;
  fromVersion: string;
  toVersion: string;
  security: boolean;
  held: boolean;
  selectedNow: boolean;
  reason: "held" | "selected-now" | "deferred-unknown" | "unknown";
};

export type PackagesSection = {
  observedAt: string;
  lastGoodAt: string | null;
  state: "ok" | "partial" | "unavailable";
  candidates: PackageCandidate[];
  counts: { total: number; selectedNow: number; held: number; deferred: number } | null;
  truncated: boolean;
  warning: "hold-unavailable" | "simulation-unavailable" | "package-query-failed" | null;
};

export type EnvironmentDeps = {
  now: () => number;
  readSettingsRaw: () => RawSettingsRead;
  checkDatabase: () => OwnDbCheck;
  checkProjects: (root: string) => { ok: boolean };
  listPanes: typeof listLivePanes;
  ttydBase: () => string | null;
  httpOk: (url: string) => Promise<boolean>;
  cli: (name: string) => boolean;
  linearCredential: () => boolean;
  metricsLatest: () => string | null;
  envFile: () => EnvFileState;
  readMountInfo: () => Promise<string>;
  statfs: (path: string) => Promise<StatfsFacts>;
  realpath: (path: string) => string;
  packages: () => Promise<PackageObservation>;
  exec: (file: string, args: readonly string[], opts: { timeout: number; maxBuffer: number }) => Promise<string>;
  ps: () => Promise<string>;
  readProc: (pid: number) => Promise<string>;
  paneContext: (pane: string, incarnation: string | null) => LastKnownPaneContext | null;
};

export type ComponentFacts = {
  nowMs: number;
  settings: SettingsResult;
  presence: KeyPresence | null;
  db: OwnDbCheck;
  projects: { ok: boolean };
  tmux: { ok: boolean };
  ttyd: { configured: boolean; reachable: boolean | null };
  clis: { claude: boolean; codex: boolean; opencode: boolean };
  githubCli: boolean;
  linearCredential: boolean;
  metricsLatest: string | null;
  envFile: EnvFileState;
};

export const METRICS_STALE_S = 180;

function result(
  id: ComponentId, state: ComponentState, detail: string | null, nowMs: number, lastGoodAt: string | null = null,
): ComponentResult {
  return { id, state, detail, observedAt: new Date(nowMs).toISOString(), lastGoodAt };
}

function disabledState(enabled: boolean, explicit: boolean): ComponentState {
  if (enabled) return "healthy";
  return explicit ? "disabled-explicit" : "disabled-default";
}

// Pure matrix projection: every state and detail is fixture-testable without touching the host.
export function buildComponents(facts: ComponentFacts): ComponentResult[] {
  const t = facts.nowMs;
  const settingsOk = facts.settings.ok;
  const integrations = settingsOk ? facts.settings.data.integrations : null;
  const explicit = facts.presence;

  const integration = (
    id: ComponentId,
    enabled: boolean,
    explicitKey: boolean,
    whenEnabled: () => ComponentResult,
  ): ComponentResult => {
    if (!settingsOk || explicit === null) return result(id, "unknown", "settings-unreadable", t);
    if (!enabled) return result(id, disabledState(enabled, explicitKey), null, t);
    return whenEnabled();
  };

  const metrics = ((): ComponentResult => {
    if (facts.metricsLatest === null) return result("metrics", "unavailable", "no-sample", t);
    const ageS = Math.max(0, Math.round((t - Date.parse(facts.metricsLatest)) / 1000));
    if (!Number.isFinite(ageS)) return result("metrics", "unavailable", "no-sample", t);
    return ageS <= METRICS_STALE_S
      ? result("metrics", "healthy", "fresh", t)
      : result("metrics", "unavailable", "stale", t);
  })();

  const envFile = ((): ComponentResult => {
    if (facts.envFile === "ok") return result("env-file", "healthy", "secure", t);
    if (facts.envFile === "missing") return result("env-file", "disabled-default", "missing", t);
    return result("env-file", "needs-configuration", facts.envFile, t);
  })();

  return [
    facts.db.ok
      ? result("database", "healthy", "select-1", t)
      : result("database", "failed", facts.db.error ?? "database-unavailable", t),
    !settingsOk
      ? result("projects", "unknown", "settings-unreadable", t)
      : facts.projects.ok
        ? result("projects", "healthy", "root-readable", t)
        : result("projects", "failed", "root-unreadable", t),
    facts.tmux.ok
      ? result("tmux", "healthy", "snapshot", t)
      : result("tmux", "unavailable", "tmux-unavailable", t),
    integration("ttyd", integrations?.ttyd ?? false, explicit?.ttyd ?? false, () =>
      !facts.ttyd.configured
        ? result("ttyd", "needs-configuration", "urls-missing", t)
        : facts.ttyd.reachable === true
          ? result("ttyd", "healthy", "http-ok", t)
          : facts.ttyd.reachable === false
            ? result("ttyd", "failed", "http-unreachable", t)
            : result("ttyd", "unknown", "http-unverified", t)),
    integration("github", integrations?.github ?? false, explicit?.github ?? false, () =>
      facts.githubCli
        ? result("github", "unknown", "configured-unverified", t)
        : result("github", "needs-configuration", "cli-missing", t)),
    integration("linear", integrations?.linear ?? false, explicit?.linear ?? false, () =>
      facts.linearCredential
        ? result("linear", "unknown", "credential-present", t)
        : result("linear", "needs-configuration", "credential-missing", t)),
    integration("claude", integrations?.agents.claude ?? false, explicit?.agents.claude ?? false, () =>
      facts.clis.claude
        ? result("claude", "healthy", "cli-present", t)
        : result("claude", "needs-configuration", "cli-missing", t)),
    integration("codex", integrations?.agents.codex ?? false, explicit?.agents.codex ?? false, () =>
      facts.clis.codex
        ? result("codex", "healthy", "cli-present", t)
        : result("codex", "needs-configuration", "cli-missing", t)),
    integration("opencode", integrations?.agents.opencode ?? false, explicit?.agents.opencode ?? false, () =>
      facts.clis.opencode
        ? result("opencode", "healthy", "cli-present", t)
        : result("opencode", "needs-configuration", "cli-missing", t)),
    metrics,
    envFile,
  ];
}

export const MAX_PACKAGE_ROWS = 500;

function normalizeKeys(rows: { name: string; arch: string | null }[], nativeArch: string | null): Set<string> {
  return new Set(rows.map((row) => packageKey(row.name, row.arch, nativeArch)));
}

// Fixed read-only package queries. Simulation is labelled selection by the latest simulation,
// never guaranteed installability; no apt update/install/upgrade/autoremove ever runs here.
export async function buildPackageSection(
  observation: PackageObservation,
  deps: Pick<EnvironmentDeps, "exec" | "now">,
): Promise<PackagesSection> {
  const nowMs = deps.now();
  const observedAt = new Date(nowMs).toISOString();
  if (observation.state === "failed" && observation.rows.length === 0) {
    return {
      observedAt, lastGoodAt: observation.lastGoodAt, state: "unavailable",
      candidates: [], counts: null, truncated: false, warning: "package-query-failed",
    };
  }
  let nativeArch: string | null = null;
  try {
    const out = (await deps.exec("dpkg", ["--print-architecture"], { timeout: 2_000, maxBuffer: 4 * 1024 })).trim();
    nativeArch = /^[a-z0-9-]{1,32}$/.test(out) ? out : null;
  } catch {}
  let holdSet: Set<string> | null = null;
  try {
    holdSet = normalizeKeys(parseHoldRows(await deps.exec("apt-mark", ["showhold"], { timeout: 5_000, maxBuffer: 1024 * 1024 })), nativeArch);
  } catch {}
  let simSet: Set<string> | null = null;
  try {
    simSet = normalizeKeys(parseSimulationRows(await deps.exec("apt-get", ["--simulate", "upgrade"], { timeout: 10_000, maxBuffer: 2 * 1024 * 1024 })), nativeArch);
  } catch {}

  const truncated = observation.rows.length > MAX_PACKAGE_ROWS;
  const rows = observation.rows.slice(0, MAX_PACKAGE_ROWS);
  const candidates: PackageCandidate[] = rows.map((row) => {
    const key = packageKey(row.name, row.arch, nativeArch);
    const held = holdSet?.has(key) ?? false;
    const selectedNow = !held && (simSet?.has(key) ?? false);
    const reason: PackageCandidate["reason"] =
      holdSet === null || simSet === null ? "unknown"
        : held ? "held"
          : selectedNow ? "selected-now"
            : "deferred-unknown";
    return {
      key, name: row.name, arch: row.arch || nativeArch || "unknown",
      fromVersion: row.fromVersion, toVersion: row.toVersion, security: row.security,
      held, selectedNow, reason,
    };
  }).sort((a, b) => a.key.localeCompare(b.key));

  const counts = holdSet === null || simSet === null ? null : {
    total: candidates.length,
    selectedNow: candidates.filter((c) => c.reason === "selected-now").length,
    held: candidates.filter((c) => c.held).length,
    deferred: candidates.filter((c) => c.reason === "deferred-unknown").length,
  };
  const warning = holdSet === null ? "hold-unavailable"
    : simSet === null ? "simulation-unavailable"
      : observation.state === "failed" ? "package-query-failed" : null;
  return {
    observedAt,
    lastGoodAt: observation.lastGoodAt,
    state: observation.state === "ok" && warning === null ? (truncated ? "partial" : "ok") : "partial",
    candidates, counts, truncated, warning,
  };
}

export type EnvironmentRuntime = {
  components: () => Promise<ComponentsSection>;
  resources: () => Promise<ResourcesSection>;
  packages: () => Promise<PackagesSection>;
  workloads: () => Promise<WorkloadsSection>;
};

type Cache<T> = { at: number; value: T } | null;

function cached<T>(holder: { cache: Cache<T> }, ttlMs: number, nowMs: number, load: () => Promise<T>): Promise<T> {
  if (holder.cache && nowMs - holder.cache.at < ttlMs) return Promise.resolve(holder.cache.value);
  return load().then((value) => {
    holder.cache = { at: nowMs, value };
    return value;
  });
}

export function createEnvironmentRuntime(deps: EnvironmentDeps): EnvironmentRuntime {
  const panesCache: { cache: Cache<{ ok: boolean }> } = { cache: null };
  const ttydCache: { cache: Cache<boolean | null> } = { cache: null };
  const dbCache: { cache: Cache<OwnDbCheck> } = { cache: null };
  const mountCache: { cache: Cache<MountInfo[]> } = { cache: null };
  const statfsCache: { cache: Cache<StatfsFacts> } = { cache: null };
  let lastTmp: TmpCapacity | null = null;
  let lastTmpAt: string | null = null;

  async function components(): Promise<ComponentsSection> {
    const nowMs = deps.now();
    const raw = deps.readSettingsRaw();
    const settings: SettingsResult = raw.ok ? parseSettings(raw.raw) : raw;
    const presence = raw.ok && settings.ok ? settingsKeyPresence(raw.raw) : null;
    const panes = await cached(panesCache, 5_000, nowMs, async () => {
      try {
        await deps.listPanes();
        return { ok: true as const };
      } catch {
        return { ok: false as const };
      }
    });
    const db = await cached(dbCache, 5 * 60_000, nowMs, async () => deps.checkDatabase());
    const ttydEnabled = settings.ok && settings.data.integrations.ttyd;
    const base = ttydEnabled ? deps.ttydBase() : null;
    const reachable = base === null ? null : await cached(ttydCache, 30_000, nowMs, () => deps.httpOk(base));
    const root = settings.ok ? settings.data.reposRoot : null;
    let projects = { ok: false };
    if (root !== null) {
      try { projects = deps.checkProjects(root); } catch { projects = { ok: false }; }
    }
    const facts: ComponentFacts = {
      nowMs,
      settings,
      presence,
      db,
      projects,
      tmux: panes,
      ttyd: { configured: base !== null, reachable },
      clis: { claude: deps.cli("claude"), codex: deps.cli("codex"), opencode: deps.cli("opencode") },
      githubCli: deps.cli("gh"),
      linearCredential: deps.linearCredential(),
      metricsLatest: deps.metricsLatest(),
      envFile: deps.envFile(),
    };
    return { observedAt: new Date(nowMs).toISOString(), components: buildComponents(facts) };
  }

  async function resources(): Promise<ResourcesSection> {
    const nowMs = deps.now();
    try {
      const path = deps.realpath("/tmp");
      const mounts = await cached(mountCache, 5_000, nowMs, async () => parseMountInfo(await deps.readMountInfo()));
      const mount = selectMount(mounts, path);
      if (!mount) throw new Error("mount-facts");
      const capacity = capacityFromStatfs(await cached(statfsCache, 5_000, nowMs, () => deps.statfs(path)));
      if (!capacity) throw new Error("statfs-failed");
      const root = selectMount(mounts, "/");
      const tmp: TmpCapacity = {
        path,
        mountPath: mount.mountPoint,
        fsType: mount.fsType,
        tmpfs: mount.fsType === "tmpfs",
        sharedRoot: root !== null && root.majorMinor === mount.majorMinor,
        ...capacity,
      };
      lastTmp = tmp;
      lastTmpAt = new Date(nowMs).toISOString();
      return { observedAt: lastTmpAt, lastGoodAt: lastTmpAt, state: "ok", stale: false, tmp, warning: null };
    } catch (err) {
      const warning = err instanceof Error && err.message === "mount-facts" ? "mount-facts" : "statfs-failed";
      return {
        observedAt: new Date(nowMs).toISOString(), lastGoodAt: lastTmpAt,
        state: "unavailable", stale: lastTmp !== null, tmp: lastTmp, warning,
      };
    }
  }

  async function packages(): Promise<PackagesSection> {
    const observation = await deps.packages();
    return buildPackageSection(observation, deps);
  }

  async function workloads(): Promise<WorkloadsSection> {
    const panes = await deps.listPanes();
    const panePids = panes.panePids ?? new Map<string, number>();
    try {
      return await collectWorkloads({
        nowMs: deps.now(),
        psOutput: await deps.ps(),
        panePids,
        readProc: deps.readProc,
        paneContext: deps.paneContext,
        incarnation: panes.incarnation,
      });
    } catch {
      const nowMs = deps.now();
      return {
        observedAt: new Date(nowMs).toISOString(), state: "unavailable", groups: [],
        totalProcesses: 0, truncated: false,
        coverage: { ownershipKnown: 0, ownershipUnknown: 0, partial: true },
      };
    }
  }

  return { components, resources, packages, workloads };
}

// ── system wiring (collectors own every system read) ──────────────────────────────────────────

const execFileAsync = promisify(execFile);

const systemExec: EnvironmentDeps["exec"] = async (file, args, opts) => {
  const { stdout } = await execFileAsync(file, args, { encoding: "utf8", ...opts });
  return stdout;
};

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "[::1]", "::1"]);

// Ttyd is GET-only, only after a validated loopback base, with no redirects.
export function loopbackBase(value: string | undefined): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    if (url.protocol !== "http:" && url.protocol !== "https:") return null;
    if (!LOOPBACK_HOSTS.has(url.hostname)) return null;
    return url.origin;
  } catch {
    return null;
  }
}

async function boundedHttpOk(url: string): Promise<boolean> {
  try {
    const res = await fetch(url, { redirect: "error", signal: AbortSignal.timeout(2_000) });
    const reader = res.body?.getReader();
    if (reader) {
      let seen = 0;
      while (seen < 16 * 1024) {
        const { done, value } = await reader.read();
        if (done) break;
        seen += value.byteLength;
      }
      await reader.cancel().catch(() => {});
    }
    return res.ok;
  } catch {
    return false;
  }
}

function latestMetricTs(): string | null {
  try {
    const row = getDb().prepare("SELECT ts FROM metrics ORDER BY ts DESC, id DESC LIMIT 1").get() as { ts?: string } | undefined;
    return typeof row?.ts === "string" ? row.ts : null;
  } catch {
    return null;
  }
}

export function systemEnvironmentDeps(): EnvironmentDeps {
  return {
    now: Date.now,
    readSettingsRaw: () => readRawSettings(),
    checkDatabase: () => checkOwnDatabase(getDb()),
    checkProjects: (root) => {
      getProjects(root);
      return { ok: true };
    },
    listPanes: () => listLivePanes(),
    ttydBase: () => loopbackBase(process.env.TTYD_RO_URL),
    httpOk: boundedHttpOk,
    cli: (name) => isCliPresent(name),
    linearCredential: () => Boolean(process.env.LINEAR_API_KEY),
    metricsLatest: latestMetricTs,
    envFile: () => envFileStatus().state,
    readMountInfo: async () => {
      const text = await readFile("/proc/self/mountinfo", "utf8");
      if (Buffer.byteLength(text, "utf8") > 256 * 1024) throw new Error("mount-facts");
      return text;
    },
    statfs: async (path) => {
      const stats = await statfs(path);
      return { bsize: stats.bsize, blocks: stats.blocks, bfree: stats.bfree, bavail: stats.bavail };
    },
    realpath: (path) => realpathSync(path),
    packages: async () => {
      const { getPackageObservation } = await import("./alerts");
      return getPackageObservation();
    },
    exec: systemExec,
    ps: () => systemExec("ps", ["-eo", "pid=,ppid=,uid=,stat=,etimes=,rss=,comm="], { timeout: 5_000, maxBuffer: 2 * 1024 * 1024 }),
    readProc: readProcStat,
    paneContext: (pane, incarnation) => {
      try { return lastKnownPaneContext(getDb(), pane, incarnation); } catch { return null; }
    },
  };
}
