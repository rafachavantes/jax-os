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
  getPackageObservation, normalizePackageCandidates,
  type NormalizedPackageCandidate, type PackageObservation, type PackageWarning,
} from "./alerts";
import { listLivePanes } from "./workflow-tmux";
import { collectWorkloads, createIdentityReader, readProcStat, type WorkloadsSection } from "./workloads";

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

export type PackageCandidate = NormalizedPackageCandidate;

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
  | { state: "failed"; error: string; lastGoodAt?: string | null }
  | { state: "unknown"; lastGoodAt?: string | null };
export type MetricsFacts =
  | { state: "ok"; latest: string | null }
  | { state: "failed"; lastGoodAt?: string | null }
  | { state: "unknown"; lastGoodAt?: string | null };

export type EnvironmentDeps = {
  now: () => number;
  // Elapsed-time seam. The runtime resolves it once and shares it with the source observer,
  // the persistent identity reader and the workload collection.
  monotonic?: () => number;
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
  // Both configured sides are required by the existing viewer; only validated loopback sides
  // are probed, and `base: null` marks an externally configured side that stays unverified.
  ttyd: {
    roConfigured: boolean;
    rwConfigured: boolean;
    ro: { base: string | null; reachable: boolean | null };
    rw: { base: string | null; reachable: boolean | null };
  };
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

function ttydResult(facts: ComponentFacts["ttyd"], t: number): ComponentResult {
  if (!facts.roConfigured || !facts.rwConfigured) return result("ttyd", "needs-configuration", "urls-missing", t);
  if (facts.ro.reachable === false || facts.rw.reachable === false) return result("ttyd", "failed", "http-unreachable", t);
  const unverified = (["ro", "rw"] as const).filter((side) => facts[side].base === null);
  if (unverified.length > 0) return result("ttyd", "unknown", `external-unverified-${unverified.join("-")}`, t);
  if (facts.ro.reachable === true && facts.rw.reachable === true) return result("ttyd", "healthy", "http-ok-both", t);
  return result("ttyd", "unknown", "http-unverified", t);
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
    if (facts.metrics.state === "unknown") return result("metrics", "unknown", "database-unverified", t, facts.metrics.lastGoodAt ?? null);
    if (facts.metrics.state === "failed") return result("metrics", "unavailable", "database-unavailable", t, facts.metrics.lastGoodAt ?? null);
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
    if (facts.db.state === "failed") return result("database", "failed", facts.db.error, t, facts.db.lastGoodAt ?? null);
    return result("database", "unknown", "database-unverified", t, facts.db.lastGoodAt ?? null);
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
    integration("ttyd", integrations?.ttyd ?? false, explicit?.ttyd ?? false, () => ttydResult(facts.ttyd, t)),
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

// One normalized projection used by both the Environment section and alert counts. No subprocess
// runs here: candidates/holds/simulation/architecture all come from the shared observation, and
// the section keeps the source observation time instead of renewing it on every poll.
export function buildPackageSection(
  observation: PackageObservation,
  _deps?: Pick<EnvironmentDeps, "now">,
): PackagesSection {
  const warnings = [...observation.warnings];
  if (observation.state === "failed" && observation.rows.length === 0) {
    return {
      observedAt: observation.observedAt, lastGoodAt: observation.lastGoodAt, state: "unavailable",
      candidates: [], counts: null, truncated: false, warnings,
    };
  }

  const { candidates: normalized, duplicateConflict, selectionKnown } = normalizePackageCandidates(observation);
  const truncated = normalized.length > MAX_PACKAGE_ROWS;
  const shown = truncated ? normalized.slice(0, MAX_PACKAGE_ROWS) : normalized;
  // Known rows from a partial listing stay an explicitly incomplete known count, never a
  // complete-total claim; a first total failure has no rows and was returned as unavailable.
  const totalKnown = normalized.length > 0
    || observation.children.candidates.ok
    || observation.children.candidates.lastGoodAt !== null;
  const counts: PackageCounts | null = totalKnown ? {
    total: normalized.length,
    shown: shown.length,
    remainder: normalized.length - shown.length,
    selectedNow: selectionKnown ? normalized.filter((c) => c.reason === "selected-now").length : null,
    held: selectionKnown ? normalized.filter((c) => c.reason === "held").length : null,
    deferred: selectionKnown ? normalized.filter((c) => c.reason === "deferred-unknown").length : null,
    unknown: normalized.filter((c) => c.reason === "unknown").length,
  } : null;

  const hasConflict = normalized.some((candidate) => candidate.conflict);
  const state: PackagesSection["state"] =
    observation.state === "failed" && normalized.length === 0 ? "unavailable"
      : observation.state === "ok" && warnings.length === 0 && !hasConflict && !duplicateConflict && !truncated ? "ok"
        : "partial";
  return {
    observedAt: observation.observedAt,
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

// An unsuccessful observation keeps WHY it was unsuccessful: an explicit source rejection is a
// failure, while a caller deadline only means the still-pending original was not observed yet.
export type Observation<T> =
  | { ok: true; value: T; observedAt: number }
  | { ok: false; observedAt: number; lastGood: T | null; lastGoodAt: number | null; cause: "rejected" | "deadline" };

type SourceState<T> = {
  // `at`/`expiresAt` are wall observation time and monotonic cache expiry respectively.
  lastGood: { value: T; at: number; expiresAt: number } | null;
  pending: { promise: Promise<T>; startedAt: number; deadline: number } | null;
};

const realSleep = (ms: number): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

type RaceOutcome<T> =
  | { kind: "value"; value: T }
  | { kind: "rejected" }
  | { kind: "deadline" };

function raceDeadline<T>(
  promise: Promise<T>, ms: number, sleep: (ms: number) => Promise<void>,
): Promise<RaceOutcome<T>> {
  return Promise.race([
    promise.then((value) => ({ kind: "value" as const, value }), () => ({ kind: "rejected" as const })),
    sleep(ms).then(() => ({ kind: "deadline" as const })),
  ]);
}

// One source = one completed observation plus (at most) the one original pending operation.
// `now` is the WALL clock: only returned observation timestamps use it. Deadlines and cache
// expiry use the injected monotonic clock, so a wall step never extends either. A caller
// deadline returns unavailable/last-good without cancelling or duplicating the load.
export function createSourceObserver(
  now: () => number,
  sleep: (ms: number) => Promise<void> = realSleep,
  monotonic: () => number = () => performance.now(),
) {
  const states = new Map<string, SourceState<unknown>>();
  const unavailable = <T,>(state: SourceState<T>, at: number, cause: "rejected" | "deadline"): Observation<T> => ({
    ok: false, observedAt: at,
    lastGood: state.lastGood?.value ?? null,
    lastGoodAt: state.lastGood?.at ?? null,
    cause,
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
    const monoAt = monotonic();
    const pending = state.pending;
    if (pending) {
      const remaining = Math.max(0, pending.deadline - monotonic());
      if (remaining <= 0) return Promise.resolve(unavailable(state, startedAt, "deadline"));
      return raceDeadline(pending.promise, remaining, sleep)
        .then((out) => out.kind === "value" && monotonic() < pending.deadline
          ? { ok: true, value: out.value, observedAt: pending.startedAt }
          : unavailable(state, now(), out.kind === "rejected" && monotonic() < pending.deadline ? "rejected" : "deadline"));
    }
    if (state.lastGood && monotonic() < state.lastGood.expiresAt) {
      return Promise.resolve({ ok: true, value: state.lastGood.value, observedAt: state.lastGood.at });
    }
    const attemptAt = startedAt;
    // The absolute monotonic deadline and cache expiry are recorded BEFORE the original load.
    const operation = { promise: Promise.resolve() as Promise<T>, startedAt: attemptAt, deadline: monoAt + budgetMs };
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
      (value) => { state.lastGood = { value, at: attemptAt, expiresAt: monoAt + ttlMs }; },
      () => {},
    );
    // Loading may have already spent part of the original allowance (worker construction,
    // synchronous setup) before control returns here; that time is subtracted instead of
    // granting the operation a fresh full budget.
    const remaining = Math.max(0, operation.deadline - monotonic());
    return raceDeadline(promise, remaining, sleep)
      .then((out) => out.kind === "value" && monotonic() < operation.deadline
        ? { ok: true, value: out.value, observedAt: attemptAt }
        // A settlement outside the original allowance is a deadline even when its promise won
        // the race against the timer callback.
        : unavailable(state, now(), out.kind === "rejected" && monotonic() < operation.deadline ? "rejected" : "deadline"));
  };
}

export function createEnvironmentRuntime(deps: EnvironmentDeps): EnvironmentRuntime {
  const monotonic = deps.monotonic ?? (() => performance.now());
  const observe = createSourceObserver(deps.now, deps.sleep ?? realSleep, monotonic);
  // One raw-read owner per runtime: caller timeouts and overlapping polls share the same bounded
  // original operations, and only their actual settlement releases capacity.
  const identityReader = createIdentityReader(deps.readProc, { now: monotonic });
  let lastTmp: TmpCapacity | null = null;
  let lastTmpAt: string | null = null;
  const observedSections: ComponentFacts["hostProbes"] = { resources: null, workloads: null, packages: null };

  // A typed unsuccessful own-DB snapshot is a failed attempt, never new last-good data. The load
  // boundary narrows it to a rejection so the existing observer keeps genuine success separate.
  async function successfulOwnDb(req?: OwnDbRequest) {
    const snapshot = await deps.ownDb(req);
    if (!snapshot.ok) throw new Error("database-unavailable");
    return snapshot;
  }

  async function paneContext(pane: string, incarnation: string | null): Promise<LastKnownPaneContext | null> {
    const key = ownPaneKey(pane, incarnation);
    if (key === null) return null;
    const observed = await observe(`pane-context:${key}`, 5_000, 200, () => successfulOwnDb({ pane, incarnation }));
    return observed.ok ? observed.value.paneContext : null;
  }

  async function components(): Promise<ComponentsSection> {
    const nowMs = deps.now();
    const raw = deps.readSettingsRaw();
    const settings: SettingsResult = raw.ok ? parseSettings(raw.raw) : raw;
    const presence = raw.ok && settings.ok ? settingsKeyPresence(raw.raw) : null;

    const panesObs = await observe("tmux", 5_000, 5_000, () => deps.listPanes());
    const dbObs = await observe("own-db", 5 * 60_000, 200, () => successfulOwnDb());
    const metricObs = await observe("own-db-metric", 5_000, 200, () => successfulOwnDb());
    const [claude, codex, opencode, gh] = await Promise.all([
      observe("cli:claude", 30_000, CLI_CHECK_TIMEOUT_MS, () => deps.cli("claude")),
      observe("cli:codex", 30_000, CLI_CHECK_TIMEOUT_MS, () => deps.cli("codex")),
      observe("cli:opencode", 30_000, CLI_CHECK_TIMEOUT_MS, () => deps.cli("opencode")),
      observe("cli:gh", 30_000, CLI_CHECK_TIMEOUT_MS, () => deps.cli("gh")),
    ]);

    const ttydEnabled = settings.ok && settings.data.integrations.ttyd;
    const transport = ttydEnabled ? deps.ttydTransport() : { ro: null, rw: null };
    const roConfigured = httpUrlConfigured(transport.ro);
    const rwConfigured = httpUrlConfigured(transport.rw);
    // Only validated loopback sides are probed, and only when BOTH sides are configured — the
    // existing viewer needs the pair; an external/invalid side stays unverified.
    const ttydComplete = ttydEnabled && roConfigured && rwConfigured;
    const roBase = ttydComplete ? loopbackBase(transport.ro ?? undefined) : null;
    const rwBase = ttydComplete ? loopbackBase(transport.rw ?? undefined) : null;
    const roObs: Observation<boolean> | null = roBase === null
      ? null
      : await observe(`ttyd:ro:${roBase}`, 30_000, 2_000, () => deps.httpOk(roBase));
    const rwObs: Observation<boolean> | null = rwBase === null
      ? null
      : await observe(`ttyd:rw:${rwBase}`, 30_000, 2_000, () => deps.httpOk(rwBase));

    const root = settings.ok ? settings.data.reposRoot : null;
    const projectsObs = root === null
      ? null
      : await observe(`projects:${root}`, 30_000, 1_000, () => deps.checkProjects(root));

    // A failed current attempt is projected as failed with the ORIGINAL last-good time; retained
    // last-good data never becomes a healthy badge for a database that just failed.
    const dbLastGoodAt = dbObs.ok ? null : dbObs.lastGoodAt === null ? null : new Date(dbObs.lastGoodAt).toISOString();
    const db: DbFacts = dbObs.ok
      ? { state: "ok", schemaVersion: dbObs.value.schemaVersion }
      : dbObs.cause === "rejected"
        ? { state: "failed", error: "database-unavailable", lastGoodAt: dbLastGoodAt }
        : { state: "unknown", lastGoodAt: dbLastGoodAt };
    const metricLastGoodAt = metricObs.ok ? null : metricObs.lastGoodAt === null ? null : new Date(metricObs.lastGoodAt).toISOString();
    const metrics: MetricsFacts = metricObs.ok
      ? { state: "ok", latest: metricObs.value.latestMetricTs }
      : metricObs.cause === "rejected"
        ? { state: "failed", lastGoodAt: metricLastGoodAt }
        : { state: "unknown", lastGoodAt: metricLastGoodAt };
    const tmux: ProbeState = panesObs.ok ? "ok" : panesObs.cause === "rejected" ? "failed" : "unknown";
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
        roConfigured, rwConfigured,
        ro: { base: roBase, reachable: roObs === null ? null : roObs.ok ? roObs.value : null },
        rw: { base: rwBase, reachable: rwObs === null ? null : rwObs.ok ? rwObs.value : null },
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
      ttyd: Math.max(roObs?.observedAt ?? nowMs, rwObs?.observedAt ?? nowMs),
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
    const tmuxWarning = panesObs.ok ? null : panesObs.cause === "rejected" ? "tmux-unavailable" : "tmux-unverified";
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
        paneDead: panes?.paneDead,
        readProc: deps.readProc,
        paneContext,
        incarnation: panes?.incarnation ?? null,
        tmuxWarning,
        verifyPanes: () => deps.listPanes(),
        identityReader,
        monotonic,
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

// A configured ttyd side must be a real HTTP(S) URL; an arbitrary nonempty string is incomplete
// configuration, never a transport claim.
function httpUrlConfigured(value: string | null): boolean {
  if (!value) return false;
  try {
    const protocol = new URL(value).protocol;
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}

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

const PROVEN_ABSENT_CODES = new Set(["ENOENT", "ENOTDIR"]);

function errorCode(error: unknown): string | undefined {
  return (error as NodeJS.ErrnoException | undefined)?.code;
}

// Fixed executable names only; never launches the CLI/provider. Only a proven missing/non-file/
// non-executable candidate is absence — permission or other filesystem errors (EIO, ...) stay
// inconclusive/unknown instead of a false absence. The caller's source observer owns the public
// deadline, so no inner timeout race clears ownership while a filesystem read is still pending.
export async function probeCliPresence(
  name: CliName,
  env: NodeJS.ProcessEnv = process.env,
  fs: CliFs = realCliFs,
): Promise<CliState> {
  if (!(CLI_NAMES as readonly string[]).includes(name)) return "unknown";
  const dirs = (env.PATH ?? "").split(delimiter).filter((dir) => dir !== "");
  if (dirs.length > MAX_PATH_SEGMENTS) return "unknown";
  const candidates = dirs.map((dir) => join(dir, name));
  if (name === "opencode") candidates.push(join(env.HOME || homedir(), ".opencode", "bin", "opencode"));
  let inconclusive = false;
  for (const candidate of candidates) {
    try {
      const info = await fs.stat(candidate);
      if (!info.isFile()) continue;
    } catch (error) {
      if (PROVEN_ABSENT_CODES.has(errorCode(error) ?? "")) continue;
      if (isPermissionError(error)) inconclusive = true;
      else return "unknown";
      continue;
    }
    try {
      await fs.access(candidate);
      return "present";
    } catch (error) {
      // Proven non-executable for this candidate; any other error is inconclusive.
      if (!isPermissionError(error)) return "unknown";
      inconclusive = true;
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
  const runFixed = async (file: string, args: readonly string[], timeout: number, maxBuffer: number) => {
    const { stdout } = await run(file, args, { encoding: "utf8", timeout, maxBuffer, env: { ...env, ...C_LOCALE_ENV } });
    return stdout;
  };
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
    ps: () => runFixed("ps", ["-eo", "pid=,ppid=,uid=,stat=,etimes=,rss=,comm="], 5_000, 2 * 1024 * 1024),
    readProc: readProcStat,
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
