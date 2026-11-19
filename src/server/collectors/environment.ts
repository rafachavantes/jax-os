// M5 Task 1: Environment diagnostics — typed readers for components, /tmp capacity and package
// candidates. Every external fact comes through an injected async seam; pure parsers carry the
// fixture tests. Nothing here mutates host state: no cleanup, no package execution, no config.
import { execFile } from "node:child_process";
import { access, readFile, stat, statfs } from "node:fs/promises";
import { constants as fsConstants, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, join } from "node:path";
import { promisify } from "node:util";
import { envFileStatus, type EnvFileState } from "../envFile";
import {
  createOwnDbReader, ownPaneKey, readOwnDb,
  type OwnDbRequest, type OwnDbSnapshot, type OwnDbWorkerRunner,
} from "../db/environment";
import type { LastKnownPaneContext } from "../db/workflows";
import {
  parseSettings, readRawSettings, settingsKeyPresence, type KeyPresence, type RawSettingsRead, type SettingsResult,
} from "../settings";
import {
  getPackageObservation, packageKey, validPackageArch, validPackageName,
  type PackageObservation, type PackageRow, type PackageWarning,
} from "./alerts";
import { listLivePanes } from "./workflow-tmux";
import { collectWorkloads, readProcStat, type WorkloadsSection } from "./workloads";

export const COMPONENT_IDS = [
  "database", "projects", "tmux", "ttyd", "github", "linear",
  "claude", "codex", "opencode", "metrics", "host-probes", "env-file",
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
  if (values.some((v) => typeof v !== "number" || !Number.isFinite(v))) return null;
  // Internally consistent counters only: an invented negative/clamped value is never shown.
  if (!(facts.bsize > 0) || !(facts.blocks > 0)) return null;
  if (facts.bavail < 0 || facts.bfree < 0 || facts.bavail > facts.bfree || facts.bfree > facts.blocks) return null;
  const total = facts.blocks * facts.bsize;
  const used = (facts.blocks - facts.bfree) * facts.bsize;
  const available = facts.bavail * facts.bsize;
  if (!Number.isFinite(total) || !Number.isFinite(used) || !Number.isFinite(available)) return null;
  if (total <= 0) return null;
  return { usedGb: used / 1e9, totalGb: total / 1e9, availableGb: available / 1e9 };
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
  conflict: boolean;
  reason: "held" | "selected-now" | "deferred-unknown" | "unknown";
};

export type PackageCounts = {
  total: number;
  shown: number;
  remainder: number;
  selectedNow: number | null;
  held: number | null;
  deferred: number | null;
  unknown: number | null;
};

export type PackagesSection = {
  observedAt: string;
  lastGoodAt: string | null;
  state: "ok" | "partial" | "unavailable";
  candidates: PackageCandidate[];
  counts: PackageCounts | null;
  truncated: boolean;
  warnings: PackageWarning[];
};

export type CliState = "present" | "missing" | "unknown";
export type ProbeState = "ok" | "failed" | "unknown";
export type DbFacts =
  | { state: "ok"; schemaVersion: number | null }
  | { state: "failed"; error: string }
  | { state: "unknown" };
export type MetricsFacts =
  | { state: "ok"; latest: string | null }
  | { state: "failed" }
  | { state: "unknown" };

export type EnvironmentDeps = {
  now: () => number;
  // Caller-visible deadline seam; tests can supply a controllable sleep.
  sleep?: (ms: number) => Promise<void>;
  readSettingsRaw: () => RawSettingsRead;
  ownDb: (req?: OwnDbRequest) => Promise<OwnDbSnapshot>;
  checkProjects: (root: string) => Promise<{ ok: boolean }>;
  listPanes: typeof listLivePanes;
  ttydTransport: () => { ro: string | null; rw: string | null };
  httpOk: (url: string) => Promise<boolean>;
  cli: (name: string) => Promise<CliState>;
  linearCredential: () => boolean;
  envFile: () => EnvFileState;
  readMountInfo: () => Promise<string>;
  statfs: (path: string) => Promise<StatfsFacts>;
  realpath: (path: string) => string;
  packages: () => Promise<PackageObservation>;
  exec: (file: string, args: readonly string[], opts: { timeout: number; maxBuffer: number }) => Promise<string>;
  ps: () => Promise<string>;
  readProc: (pid: number) => Promise<string>;
};

export type ComponentFacts = {
  nowMs: number;
  settings: SettingsResult;
  presence: KeyPresence | null;
  db: DbFacts;
  projects: ProbeState;
  tmux: ProbeState;
  ttyd: { roConfigured: boolean; rwConfigured: boolean; probeBase: string | null; reachable: boolean | null };
  clis: { claude: CliState; codex: CliState; opencode: CliState };
  githubCli: CliState;
  linearCredential: boolean;
  metrics: MetricsFacts;
  envFile: EnvFileState;
  hostProbes: { resources: boolean | null; workloads: boolean | null; packages: boolean | null };
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

function ttydSide(facts: ComponentFacts["ttyd"]): "ro" | "rw" | "both" | null {
  if (facts.roConfigured && facts.rwConfigured) return "both";
  if (facts.roConfigured) return "ro";
  if (facts.rwConfigured) return "rw";
  return null;
}

function hostProbesResult(probes: ComponentFacts["hostProbes"], t: number): ComponentResult {
  const values = [probes.resources, probes.workloads, probes.packages];
  const observed = values.filter((v): v is boolean => v !== null);
  if (observed.length === 0) return result("host-probes", "unknown", "not-checked", t);
  if (observed.some((v) => !v)) return result("host-probes", "unavailable", "sections-partial", t);
  if (observed.length < values.length) return result("host-probes", "unknown", "sections-partial", t);
  return result("host-probes", "healthy", "sections-observed", t);
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

  const cliResult = (id: ComponentId, state: CliState, whenPresentDetail: string): ComponentResult =>
    state === "present"
      ? result(id, "healthy", whenPresentDetail, t)
      : state === "missing"
        ? result(id, "needs-configuration", "cli-missing", t)
        : result(id, "unknown", "cli-unverified", t);

  const metrics = ((): ComponentResult => {
    if (facts.metrics.state === "unknown") return result("metrics", "unknown", "database-unverified", t);
    if (facts.metrics.state === "failed") return result("metrics", "unavailable", "database-unavailable", t);
    if (facts.metrics.latest === null) return result("metrics", "unavailable", "no-sample", t);
    const ageS = Math.max(0, Math.round((t - Date.parse(facts.metrics.latest)) / 1000));
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

  const database = ((): ComponentResult => {
    if (facts.db.state === "ok") return result("database", "healthy", "select-1", t);
    if (facts.db.state === "failed") return result("database", "failed", facts.db.error, t);
    return result("database", "unknown", "database-unverified", t);
  })();

  const projects = ((): ComponentResult => {
    if (!settingsOk) return result("projects", "unknown", "settings-unreadable", t);
    if (facts.projects === "ok") return result("projects", "healthy", "root-readable-unverified", t);
    if (facts.projects === "failed") return result("projects", "failed", "root-unreadable", t);
    return result("projects", "unknown", "root-unverified", t);
  })();

  const tmux = ((): ComponentResult => {
    if (facts.tmux === "ok") return result("tmux", "healthy", "snapshot", t);
    if (facts.tmux === "failed") return result("tmux", "unavailable", "tmux-unavailable", t);
    return result("tmux", "unknown", "tmux-unverified", t);
  })();

  const probes = hostProbesResult(facts.hostProbes, t);

  return [
    database,
    projects,
    tmux,
    integration("ttyd", integrations?.ttyd ?? false, explicit?.ttyd ?? false, () => {
      const side = ttydSide(facts.ttyd);
      if (side === null) return result("ttyd", "needs-configuration", "urls-missing", t);
      if (facts.ttyd.probeBase === null) return result("ttyd", "unknown", `external-unverified-${side}`, t);
      if (facts.ttyd.reachable === true) return result("ttyd", "healthy", `http-ok-${side}`, t);
      if (facts.ttyd.reachable === false) return result("ttyd", "failed", "http-unreachable", t);
      return result("ttyd", "unknown", "http-unverified", t);
    }),
    integration("github", integrations?.github ?? false, explicit?.github ?? false, () =>
      facts.githubCli === "present"
        ? result("github", "unknown", "configured-unverified", t)
        : cliResult("github", facts.githubCli, "configured-unverified")),
    integration("linear", integrations?.linear ?? false, explicit?.linear ?? false, () =>
      facts.linearCredential
        ? result("linear", "unknown", "credential-present", t)
        : result("linear", "needs-configuration", "credential-missing", t)),
    integration("claude", integrations?.agents.claude ?? false, explicit?.agents.claude ?? false, () =>
      cliResult("claude", facts.clis.claude, "cli-present")),
    integration("codex", integrations?.agents.codex ?? false, explicit?.agents.codex ?? false, () =>
      cliResult("codex", facts.clis.codex, "cli-present")),
    integration("opencode", integrations?.agents.opencode ?? false, explicit?.agents.opencode ?? false, () =>
      cliResult("opencode", facts.clis.opencode, "cli-present")),
    metrics,
    probes,
    envFile,
  ];
}

export const MAX_PACKAGE_ROWS = 500;

type FactMatch = { state: "match" } | { state: "no" } | { state: "ambiguous" } | { state: "unresolved" };

// A bare (architecture-less) policy fact resolves only to an unambiguous native or Arch:all
// candidate. Two eligible candidates stay unknown instead of being matched to both.
function matchFact(
  fact: { name: string; arch: string | null },
  name: string,
  arch: string,
  nativeArch: string | null,
  archesForName: Set<string>,
): FactMatch {
  if (fact.name !== name) return { state: "no" };
  if (fact.arch !== null) return fact.arch === arch ? { state: "match" } : { state: "no" };
  if (nativeArch === null) {
    const all = [...archesForName].filter((candidate) => candidate === "all");
    return all.length === 1 ? (arch === "all" ? { state: "match" } : { state: "no" }) : { state: "unresolved" };
  }
  const eligible = [...archesForName].filter((candidate) => candidate === nativeArch || candidate === "all");
  if (eligible.length > 1) return { state: "ambiguous" };
  if (eligible.length === 0) return { state: "no" };
  return eligible[0] === arch ? { state: "match" } : { state: "no" };
}

// One normalized projection used by both the Environment section and alert counts. No subprocess
// runs here: candidates/holds/simulation/architecture all come from the shared observation.
export function buildPackageSection(
  observation: PackageObservation,
  deps: Pick<EnvironmentDeps, "now">,
): PackagesSection {
  const nowMs = deps.now();
  const observedAt = new Date(nowMs).toISOString();
  const warnings = [...observation.warnings];
  if (observation.state === "failed" && observation.rows.length === 0) {
    return {
      observedAt, lastGoodAt: observation.lastGoodAt, state: "unavailable",
      candidates: [], counts: null, truncated: false, warnings,
    };
  }

  const nativeArch = observation.nativeArch;
  const byKey = new Map<string, PackageRow>();
  let duplicateConflict = false;
  for (const row of observation.rows) {
    if (!validPackageName(row.name)) continue;
    if (row.arch && !validPackageArch(row.arch)) continue;
    const key = packageKey(row.name, row.arch, nativeArch);
    const existing = byKey.get(key);
    if (existing) {
      if (existing.fromVersion !== row.fromVersion || existing.toVersion !== row.toVersion) duplicateConflict = true;
      continue;
    }
    byKey.set(key, row);
  }

  const archesByName = new Map<string, Set<string>>();
  for (const row of byKey.values()) {
    const arch = row.arch || nativeArch || "unknown";
    const set = archesByName.get(row.name) ?? new Set<string>();
    set.add(arch);
    archesByName.set(row.name, set);
  }

  const holds = observation.holds;
  const simulation = observation.simulation;
  let anyUnresolved = false;
  let anyAmbiguous = false;

  const normalized = [...byKey.values()].map((row) => {
    const arch = row.arch || nativeArch || "unknown";
    const arches = archesByName.get(row.name) ?? new Set([arch]);
    let held = false;
    let simVersion: string | null = null;
    let simMatched = false;
    let ambiguous = false;
    let unresolved = false;

    if (holds !== null) {
      for (const fact of holds) {
        const match = matchFact(fact, row.name, arch, nativeArch, arches);
        if (match.state === "match") held = true;
        if (match.state === "ambiguous") { ambiguous = true; anyAmbiguous = true; }
        if (match.state === "unresolved") { unresolved = true; anyUnresolved = true; }
      }
    }
    if (simulation !== null) {
      for (const fact of simulation) {
        const match = matchFact(fact, row.name, arch, nativeArch, arches);
        if (match.state === "match") { simMatched = true; simVersion = fact.version; }
        if (match.state === "ambiguous") { ambiguous = true; anyAmbiguous = true; }
        if (match.state === "unresolved") { unresolved = true; anyUnresolved = true; }
      }
    }
    const selectedNow = simMatched && simVersion !== null && simVersion === row.toVersion;
    const conflict = held && selectedNow;
    const reason: PackageCandidate["reason"] = conflict || ambiguous || unresolved
      ? "unknown"
      : held ? "held"
        : selectedNow ? "selected-now"
          : "deferred-unknown";
    return {
      key: packageKey(row.name, row.arch, nativeArch),
      name: row.name,
      arch,
      fromVersion: row.fromVersion,
      toVersion: row.toVersion,
      security: row.security,
      held,
      selectedNow: selectedNow && !conflict,
      conflict,
      reason,
    } satisfies PackageCandidate;
  }).sort((a, b) => a.key.localeCompare(b.key));

  const truncated = normalized.length > MAX_PACKAGE_ROWS;
  const shown = truncated ? normalized.slice(0, MAX_PACKAGE_ROWS) : normalized;
  const totalKnown = observation.children.candidates.ok || observation.children.candidates.lastGoodAt !== null;
  const selectionKnown = holds !== null && simulation !== null && !anyUnresolved && !anyAmbiguous;
  const counts: PackageCounts | null = totalKnown ? {
    total: normalized.length,
    shown: shown.length,
    remainder: normalized.length - shown.length,
    selectedNow: selectionKnown ? normalized.filter((c) => c.reason === "selected-now").length : null,
    held: selectionKnown ? normalized.filter((c) => c.held).length : null,
    deferred: selectionKnown ? normalized.filter((c) => c.reason === "deferred-unknown").length : null,
    unknown: normalized.filter((c) => c.reason === "unknown").length,
  } : null;

  const hasConflict = normalized.some((candidate) => candidate.conflict);
  const state: PackagesSection["state"] =
    observation.state === "failed" && normalized.length === 0 ? "unavailable"
      : observation.state === "ok" && warnings.length === 0 && !hasConflict && !duplicateConflict && !truncated ? "ok"
        : "partial";
  return {
    observedAt,
    lastGoodAt: observation.lastGoodAt,
    state,
    candidates: shown,
    counts,
    truncated,
    warnings,
  };
}

export type EnvironmentRuntime = {
  components: () => Promise<ComponentsSection>;
  resources: () => Promise<ResourcesSection>;
  packages: () => Promise<PackagesSection>;
  workloads: () => Promise<WorkloadsSection>;
};

export type Observation<T> =
  | { ok: true; value: T; observedAt: number }
  | { ok: false; observedAt: number; lastGood: T | null; lastGoodAt: number | null };

type SourceState<T> = {
  lastGood: { value: T; at: number } | null;
  pending: { promise: Promise<T>; startedAt: number } | null;
};

const realSleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

function raceDeadline<T>(
  promise: Promise<T>, ms: number, sleep: (ms: number) => Promise<void>,
): Promise<{ settled: true; value: T } | { settled: false }> {
  return Promise.race([
    promise.then((value) => ({ settled: true as const, value }), () => ({ settled: false as const })),
    sleep(ms).then(() => ({ settled: false as const })),
  ]);
}

// One source = one completed observation plus (at most) the one original pending operation.
// A caller deadline returns unavailable/last-good without cancelling or duplicating the load.
export function createSourceObserver(now: () => number, sleep: (ms: number) => Promise<void> = realSleep) {
  const states = new Map<string, SourceState<unknown>>();
  const unavailable = <T,>(state: SourceState<T>, at: number): Observation<T> => ({
    ok: false, observedAt: at,
    lastGood: state.lastGood?.value ?? null,
    lastGoodAt: state.lastGood?.at ?? null,
  });

  return function observe<T>(
    key: string, ttlMs: number, budgetMs: number, load: () => Promise<T>,
  ): Promise<Observation<T>> {
    let state = states.get(key) as SourceState<T> | undefined;
    if (!state) {
      state = { lastGood: null, pending: null };
      states.set(key, state as SourceState<unknown>);
    }
    const startedAt = now();
    const pending = state.pending;
    if (pending) {
      const remaining = pending.startedAt + budgetMs - startedAt;
      if (remaining <= 0) return Promise.resolve(unavailable(state, startedAt));
      return raceDeadline(pending.promise, remaining, sleep)
        .then((out) => out.settled
          ? { ok: true, value: out.value, observedAt: pending.startedAt }
          : unavailable(state, now()));
    }
    if (state.lastGood && startedAt - state.lastGood.at < ttlMs) {
      return Promise.resolve({ ok: true, value: state.lastGood.value, observedAt: state.lastGood.at });
    }
    const attemptAt = startedAt;
    const operation = { promise: Promise.resolve() as Promise<T>, startedAt: attemptAt };
    state.pending = operation;
    const promise = (async () => {
      try {
        return await load();
      } finally {
        if (state.pending === operation) state.pending = null;
      }
    })();
    operation.promise = promise;
    promise.then(
      (value) => { state.lastGood = { value, at: attemptAt }; },
      () => {},
    );
    return raceDeadline(promise, budgetMs, sleep)
      .then((out) => out.settled
        ? { ok: true, value: out.value, observedAt: attemptAt }
        : unavailable(state, now()));
  };
}

export function createEnvironmentRuntime(deps: EnvironmentDeps): EnvironmentRuntime {
  const observe = createSourceObserver(deps.now, deps.sleep ?? realSleep);
  let lastTmp: TmpCapacity | null = null;
  let lastTmpAt: string | null = null;
  const observedSections: ComponentFacts["hostProbes"] = { resources: null, workloads: null, packages: null };

  async function paneContext(pane: string, incarnation: string | null): Promise<LastKnownPaneContext | null> {
    const key = ownPaneKey(pane, incarnation);
    if (key === null) return null;
    const observed = await observe(`pane-context:${key}`, 5_000, 200, () => deps.ownDb({ pane, incarnation }));
    return observed.ok && observed.value.ok ? observed.value.paneContext : null;
  }

  async function components(): Promise<ComponentsSection> {
    const nowMs = deps.now();
    const raw = deps.readSettingsRaw();
    const settings: SettingsResult = raw.ok ? parseSettings(raw.raw) : raw;
    const presence = raw.ok && settings.ok ? settingsKeyPresence(raw.raw) : null;

    const panesObs = await observe("tmux", 5_000, 5_000, () => deps.listPanes());
    const dbObs = await observe("own-db", 5 * 60_000, 200, () => deps.ownDb());
    const metricObs = await observe("own-db-metric", 5_000, 200, () => deps.ownDb());
    const [claude, codex, opencode, gh] = await Promise.all([
      observe("cli:claude", 30_000, 200, () => deps.cli("claude")),
      observe("cli:codex", 30_000, 200, () => deps.cli("codex")),
      observe("cli:opencode", 30_000, 200, () => deps.cli("opencode")),
      observe("cli:gh", 30_000, 200, () => deps.cli("gh")),
    ]);

    const ttydEnabled = settings.ok && settings.data.integrations.ttyd;
    const transport = ttydEnabled ? deps.ttydTransport() : { ro: null, rw: null };
    const roConfigured = typeof transport.ro === "string" && transport.ro.length > 0;
    const rwConfigured = typeof transport.rw === "string" && transport.rw.length > 0;
    const probeBase = ttydEnabled ? (loopbackBase(transport.ro ?? undefined) ?? loopbackBase(transport.rw ?? undefined)) : null;
    const ttydObs = probeBase === null
      ? null
      : await observe(`ttyd:${probeBase}`, 30_000, 2_000, () => deps.httpOk(probeBase));

    const root = settings.ok ? settings.data.reposRoot : null;
    const projectsObs = root === null
      ? null
      : await observe(`projects:${root}`, 30_000, 1_000, () => deps.checkProjects(root));

    const dbSnapshot = dbObs.ok ? dbObs.value : dbObs.lastGood;
    const db: DbFacts = dbSnapshot === null
      ? { state: "unknown" }
      : dbSnapshot.ok
        ? { state: "ok", schemaVersion: dbSnapshot.schemaVersion }
        : { state: "failed", error: "database-unavailable" };
    const metricSnapshot = metricObs.ok ? metricObs.value : metricObs.lastGood;
    const metrics: MetricsFacts = metricSnapshot === null
      ? { state: "unknown" }
      : metricSnapshot.ok
        ? { state: "ok", latest: metricSnapshot.latestMetricTs }
        : { state: "failed" };
    const tmux: ProbeState = panesObs.ok ? "ok" : panesObs.lastGood !== null ? "failed" : "unknown";
    const projects: ProbeState = projectsObs === null
      ? "unknown"
      : projectsObs.ok ? (projectsObs.value.ok ? "ok" : "failed") : "unknown";

    const facts: ComponentFacts = {
      nowMs,
      settings,
      presence,
      db,
      projects,
      tmux,
      ttyd: {
        roConfigured, rwConfigured, probeBase,
        reachable: ttydObs === null ? null : ttydObs.ok ? ttydObs.value : null,
      },
      clis: {
        claude: claude.ok ? claude.value : "unknown",
        codex: codex.ok ? codex.value : "unknown",
        opencode: opencode.ok ? opencode.value : "unknown",
      },
      githubCli: gh.ok ? gh.value : "unknown",
      linearCredential: deps.linearCredential(),
      metrics,
      envFile: deps.envFile(),
      hostProbes: { ...observedSections },
    };
    const observedAtById: Partial<Record<ComponentId, number>> = {
      database: dbObs.observedAt,
      metrics: metricObs.observedAt,
      tmux: panesObs.observedAt,
      projects: projectsObs?.observedAt ?? nowMs,
      ttyd: ttydObs?.observedAt ?? nowMs,
      claude: claude.observedAt,
      codex: codex.observedAt,
      opencode: opencode.observedAt,
      github: gh.observedAt,
      "host-probes": nowMs,
    };
    const built = buildComponents(facts).map((component) => {
      const at = observedAtById[component.id];
      return at === undefined ? component : { ...component, observedAt: new Date(at).toISOString() };
    });
    return { observedAt: new Date(nowMs).toISOString(), components: built };
  }

  async function resources(): Promise<ResourcesSection> {
    const nowMs = deps.now();
    let stage: "mount-facts" | "statfs-failed" = "mount-facts";
    try {
      const path = deps.realpath("/tmp");
      const mountsObs = await observe("mountinfo", 5_000, 2_000, () => deps.readMountInfo());
      if (!mountsObs.ok) throw new Error("mount-facts");
      const mounts = parseMountInfo(mountsObs.value);
      if (mounts.length === 0) throw new Error("mount-facts");
      const mount = selectMount(mounts, path);
      if (!mount) throw new Error("mount-facts");
      stage = "statfs-failed";
      const statsObs = await observe("statfs", 5_000, 1_000, () => deps.statfs(path));
      if (!statsObs.ok) throw new Error("statfs-failed");
      const capacity = capacityFromStatfs(statsObs.value);
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
      lastTmpAt = new Date(Math.max(mountsObs.observedAt, statsObs.observedAt)).toISOString();
      observedSections.resources = true;
      return { observedAt: lastTmpAt, lastGoodAt: lastTmpAt, state: "ok", stale: false, tmp, warning: null };
    } catch {
      observedSections.resources = false;
      return {
        observedAt: new Date(nowMs).toISOString(), lastGoodAt: lastTmpAt,
        state: "unavailable", stale: lastTmp !== null, tmp: lastTmp, warning: stage,
      };
    }
  }

  async function packages(): Promise<PackagesSection> {
    const observation = await deps.packages();
    const section = buildPackageSection(observation, deps);
    observedSections.packages = section.state !== "unavailable";
    return section;
  }

  async function workloads(): Promise<WorkloadsSection> {
    const panesObs = await observe("tmux", 5_000, 5_000, () => deps.listPanes());
    const panes = panesObs.ok ? panesObs.value : null;
    const tmuxWarning = panesObs.ok ? null : panesObs.lastGood !== null ? "tmux-unavailable" : "tmux-unverified";
    const psObs = await observe("ps", 5_000, 5_000, () => deps.ps());
    if (!psObs.ok) {
      observedSections.workloads = false;
      const nowMs = deps.now();
      return {
        observedAt: new Date(nowMs).toISOString(), state: "unavailable", groups: [],
        totalProcesses: 0, truncated: false,
        coverage: { ownershipKnown: 0, ownershipUnknown: 0, partial: true, reasons: {} },
      };
    }
    try {
      const section = await collectWorkloads({
        nowMs: deps.now(),
        psOutput: psObs.value,
        panePids: panes?.panePids ?? new Map<string, number>(),
        readProc: deps.readProc,
        paneContext,
        incarnation: panes?.incarnation ?? null,
        tmuxWarning,
        verifyPanes: () => deps.listPanes(),
      });
      observedSections.workloads = section.state !== "unavailable";
      return section;
    } catch {
      observedSections.workloads = false;
      const nowMs = deps.now();
      return {
        observedAt: new Date(nowMs).toISOString(), state: "unavailable", groups: [],
        totalProcesses: 0, truncated: false,
        coverage: { ownershipKnown: 0, ownershipUnknown: 0, partial: true, reasons: {} },
      };
    }
  }

  return { components, resources, packages, workloads };
}

// ── system wiring (collectors own every system read) ──────────────────────────────────────────

const execFileAsync = promisify(execFile);

const C_LOCALE_ENV = { LC_ALL: "C", LANG: "C" } as const;

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

export const MAX_PATH_SEGMENTS = 32;
export const CLI_CHECK_TIMEOUT_MS = 200;

const CLI_NAMES = ["claude", "codex", "opencode", "gh"] as const;
export type CliName = (typeof CLI_NAMES)[number];

export type CliFs = {
  stat: (path: string) => Promise<{ isFile(): boolean }>;
  access: (path: string) => Promise<void>;
};

const realCliFs: CliFs = {
  stat: (path) => stat(path),
  access: (path) => access(path, fsConstants.X_OK),
};

function timeoutAfter(ms: number): Promise<never> {
  return new Promise((_, reject) => {
    const timer = setTimeout(() => reject(new Error("timeout")), ms);
    timer.unref?.();
  });
}

function isPermissionError(error: unknown): boolean {
  const code = (error as NodeJS.ErrnoException | undefined)?.code;
  return code === "EACCES" || code === "EPERM";
}

// Fixed executable names only; never launches the CLI/provider. A permission or budget problem
// cannot prove the CLI missing, so it reports `unknown` instead of a false absence.
export async function probeCliPresence(
  name: CliName,
  env: NodeJS.ProcessEnv = process.env,
  fs: CliFs = realCliFs,
  timeoutMs: number = CLI_CHECK_TIMEOUT_MS,
): Promise<CliState> {
  if (!(CLI_NAMES as readonly string[]).includes(name)) return "unknown";
  const dirs = (env.PATH ?? "").split(delimiter).filter((dir) => dir !== "");
  if (dirs.length > MAX_PATH_SEGMENTS) return "unknown";
  const candidates = dirs.map((dir) => join(dir, name));
  if (name === "opencode") candidates.push(join(env.HOME || homedir(), ".opencode", "bin", "opencode"));
  let inconclusive = false;
  for (const candidate of candidates) {
    try {
      const info = await Promise.race([fs.stat(candidate), timeoutAfter(timeoutMs)]);
      if (!info.isFile()) continue;
      await Promise.race([fs.access(candidate), timeoutAfter(timeoutMs)]);
      return "present";
    } catch (error) {
      if (isPermissionError(error)) inconclusive = true;
      else if (error instanceof Error && error.message === "timeout") return "unknown";
    }
  }
  return inconclusive ? "unknown" : "missing";
}

export const MOUNTINFO_LIMITS = { timeoutMs: 2_000, maxBytes: 256 * 1024 };

export async function readBoundedText(
  path: string,
  limits: { timeoutMs: number; maxBytes: number },
  read: (path: string, signal: AbortSignal) => Promise<string> = (p, signal) =>
    readFile(p, { encoding: "utf8", signal }),
): Promise<string> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), limits.timeoutMs);
  try {
    const text = await read(path, controller.signal);
    if (Buffer.byteLength(text, "utf8") > limits.maxBytes) throw new Error("read-too-large");
    return text;
  } finally {
    clearTimeout(timer);
  }
}

export type SystemEnvironmentOptions = {
  env?: NodeJS.ProcessEnv;
  execFileImpl?: (
    file: string, args: readonly string[],
    options: { encoding: "utf8"; timeout: number; maxBuffer: number; env: NodeJS.ProcessEnv },
  ) => Promise<{ stdout: string }>;
  ownDbRunner?: OwnDbWorkerRunner;
  now?: () => number;
  sleep?: (ms: number) => Promise<void>;
};

export function systemEnvironmentDeps(options: SystemEnvironmentOptions = {}): EnvironmentDeps {
  const env = options.env ?? process.env;
  const run = options.execFileImpl ?? execFileAsync;
  const ownDb = options.ownDbRunner ? createOwnDbReader(options.ownDbRunner) : readOwnDb;
  const exec: EnvironmentDeps["exec"] = async (file, args, opts) => {
    const { stdout } = await run(file, args, {
      encoding: "utf8", timeout: opts.timeout, maxBuffer: opts.maxBuffer,
      env: { ...env, ...C_LOCALE_ENV },
    });
    return stdout;
  };
  const boundedProc = (pid: number) => Promise.race([readProcStat(pid), timeoutAfter(2_000)]);
  return {
    now: options.now ?? Date.now,
    sleep: options.sleep,
    readSettingsRaw: () => readRawSettings(),
    ownDb,
    checkProjects: async (root) => {
      const observed = await observeProjectRoot(root);
      return observed;
    },
    listPanes: () => listLivePanes(),
    ttydTransport: () => ({ ro: env.TTYD_RO_URL ?? null, rw: env.TTYD_RW_URL ?? null }),
    httpOk: boundedHttpOk,
    cli: (name) => probeCliPresence(name as CliName, env),
    linearCredential: () => Boolean(env.LINEAR_API_KEY),
    envFile: () => envFileStatus().state,
    readMountInfo: () => readBoundedText("/proc/self/mountinfo", MOUNTINFO_LIMITS),
    statfs: async (path) => {
      const stats = await statfs(path);
      return { bsize: stats.bsize, blocks: stats.blocks, bfree: stats.bfree, bavail: stats.bavail };
    },
    realpath: (path) => realpathSync(path),
    packages: () => getPackageObservation(),
    exec,
    ps: () => exec("ps", ["-eo", "pid=,ppid=,uid=,stat=,etimes=,rss=,comm="], { timeout: 5_000, maxBuffer: 2 * 1024 * 1024 }),
    readProc: boundedProc,
  };
}

async function observeProjectRoot(root: string): Promise<{ ok: boolean }> {
  try {
    const info = await Promise.race([stat(root), timeoutAfter(1_000)]);
    return { ok: info.isDirectory() };
  } catch {
    return { ok: false };
  }
}
