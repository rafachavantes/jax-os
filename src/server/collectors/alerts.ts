import { execFile } from "node:child_process";
import { readFile } from "node:fs/promises";
import { promisify } from "node:util";

export type ZombieRow = { pid: number; ppid: number; ageSeconds: number; parentCommand: string | null };
// M5 Task 3: architecture is part of identity — two architectures of one name are two rows.
export type PackageRow = { name: string; arch: string; key: string; fromVersion: string; toVersion: string; security: boolean };

export type AlertSource = "disk" | "load" | "zombies" | "packages";

export type Alert =
  | { kind: "disk"; mount: string; usedPercent: number }
  | { kind: "load"; load5: number; cores: number }
  | { kind: "zombies"; count: number; rows: ZombieRow[] }
  | { kind: "updates"; count: number; rows: PackageRow[] }
  // M5 Task 3: an ordinary probe failure is an explicit source state, not a silently omitted
  // alert. Existing array consumers keep their fields; this is an additive variant.
  | { kind: "source"; source: AlertSource; state: "failed" };

// df -P guarantees one line per filesystem. Filter to /dev/* so pseudo
// filesystems (/run/lock at 5MB) never false-alert.
export function parseDfAlerts(out: string, threshold = 80): Alert[] {
  return out
    .split(/\r?\n/)
    .slice(1)
    .map((l) => l.trim().split(/\s+/))
    .filter((c) => c.length >= 6 && c[0].startsWith("/dev/"))
    .map((c) => ({ mount: c.slice(5).join(" "), usedPercent: parseInt(c[4], 10) }))
    .filter((d) => Number.isFinite(d.usedPercent) && d.usedPercent > threshold)
    .map((d) => ({ kind: "disk" as const, ...d }));
}

export function parseLoadAlert(loadavg: string, cores: number): Alert | null {
  const load5 = Number(loadavg.trim().split(/\s+/)[1]);
  return Number.isFinite(load5) && load5 > cores ? { kind: "load", load5, cores } : null;
}

// spec §9: one `ps -eo pid,ppid,stat,etimes,comm` call; parent command resolved from
// another row's pid in the SAME snapshot, never a second call. Round-3 F3: a missing
// parent row yields `parentCommand: null`, never a placeholder.
export function parseZombieRows(psOut: string): ZombieRow[] {
  const commandByPid = new Map<number, string>();
  const zombies: { pid: number; ppid: number; ageSeconds: number }[] = [];
  for (const line of psOut.split(/\r?\n/).slice(1)) {
    const cols = line.trim().split(/\s+/);
    if (cols.length < 5) continue;
    const pid = parseInt(cols[0], 10);
    const ppid = parseInt(cols[1], 10);
    const stat = cols[2];
    const ageSeconds = parseInt(cols[3], 10);
    const comm = cols[4];
    if (!Number.isFinite(pid) || !Number.isFinite(ppid) || !Number.isFinite(ageSeconds)) continue;
    commandByPid.set(pid, comm);
    if (stat.startsWith("Z")) zombies.push({ pid, ppid, ageSeconds });
  }
  return zombies.map((z) => ({ ...z, parentCommand: commandByPid.get(z.ppid) ?? null }));
}

export type PackageNameArch = { name: string; arch: string | null };
export type PackageSimulationFact = { name: string; arch: string | null; version: string | null };

// Validated Debian identity: lowercase name (2+ chars) and architecture. Foreign architectures
// stay distinct from the native one; nothing is inferred from a suite or package name.
export const DEB_NAME_RE = /^[a-z0-9][a-z0-9+.-]+$/;
export const DEB_ARCH_RE = /^(all|[a-z0-9][a-z0-9-]*)$/;

export function validPackageName(name: string): boolean { return DEB_NAME_RE.test(name); }
export function validPackageArch(arch: string): boolean { return DEB_ARCH_RE.test(arch); }

// M5 F6: identity is `name:arch`. Bare names take the fixed native architecture (read by the
// caller with `dpkg --print-architecture`); when that is unknown, the key stays explicitly
// ambiguous instead of silently collapsing two architectures.
export function packageKey(name: string, arch: string | null, nativeArch: string | null): string {
  const resolved = arch && arch.length > 0 ? arch : nativeArch && nativeArch.length > 0 ? nativeArch : "unknown";
  return `${name}:${resolved}`;
}

function splitNameArch(token: string): PackageNameArch | null {
  if (!/^[a-z0-9][a-z0-9+.-]*(?::[a-z0-9][a-z0-9-]*)?$/.test(token)) return null;
  const at = token.lastIndexOf(":");
  const name = at > 0 ? token.slice(0, at) : token;
  if (!validPackageName(name)) return null;
  if (at <= 0) return { name, arch: null };
  const arch = token.slice(at + 1);
  return validPackageArch(arch) ? { name, arch } : null;
}

const PACKAGE_LINE_RE = /^([^/\s]+)\/(\S+)\s+(\S+)\s+(\S+)\s+\[upgradable from:\s*([^\]]+)\]/;

export type PackageListing = { rows: PackageRow[]; recognized: boolean };

// Parses `apt list --upgradable` (C locale is pinned by the caller). Unrecognized prose is
// never proof of zero: parsed known rows are preserved but the listing is flagged partial.
export function parsePackageListing(aptOut: string): PackageListing {
  const rows: PackageRow[] = [];
  let recognized = true;
  for (const line of aptOut.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed === "Listing...") continue;
    const m = PACKAGE_LINE_RE.exec(line);
    if (!m) { recognized = false; continue; }
    const [, name, suite, toVersion, arch, fromVersion] = m;
    if (!validPackageName(name) || !validPackageArch(arch)) { recognized = false; continue; }
    rows.push({
      name, arch, key: packageKey(name, arch, null),
      fromVersion: fromVersion.trim(), toVersion, security: suite.includes("-security"),
    });
  }
  return { rows, recognized };
}

export function parsePackageRows(aptOut: string): PackageRow[] {
  return parsePackageListing(aptOut).rows;
}

export type PackageHoldListing = { rows: PackageNameArch[]; recognized: boolean };

// `apt-mark showhold` prints one bare (or name:arch) package per line. Empty output is a valid
// zero holds; any unparseable non-empty line makes the whole listing unrecognized.
export function parseHoldListing(out: string): PackageHoldListing {
  const rows: PackageNameArch[] = [];
  let recognized = true;
  for (const line of out.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    const parsed = splitNameArch(trimmed);
    if (parsed) rows.push(parsed);
    else recognized = false;
  }
  return { rows, recognized };
}

export function parseHoldRows(out: string): PackageNameArch[] {
  return parseHoldListing(out).rows;
}

const INST_RE = /^Inst\s+(\S+)\s+\[([^\]]*)\]\s+\((.*)\)\s*$/;
const SIM_SUMMARY_RE = /^\d+ upgraded, \d+ newly installed, \d+ to remove and \d+ not upgraded\.$/;
const SIM_PREAMBLE_RE = [
  /^NOTE: This is only a simulation!$/,
  /^\s+apt-get needs root privileges for real execution\.$/,
  /^\s+Keep also in mind that locking is deactivated,$/,
  /^\s+so don't depend on the relevance to the real current situation!$/,
  /^Reading package lists\.\.\.$/,
  /^Building dependency tree\.\.\.$/,
  /^Reading state information\.\.\.$/,
  /^Calculating upgrade\.\.\.$/,
  /^The following packages will be upgraded:$/,
  /^(?:Need to get|After this operation|Fetched) /,
  /^\s{2,}[a-z0-9][a-z0-9+.:-]*(?:\s+[a-z0-9][a-z0-9+.:-]*)*$/,
];

export type PackageSimulationListing = { rows: PackageSimulationFact[]; recognized: boolean };

// `apt-get --simulate upgrade` marks each selected package with an `Inst` row; the bracketed
// architecture and candidate version are part of the solver fact. A recognized zero-result
// summary is valid; arbitrary/empty prose is not proof of zero.
export function parseSimulationListing(out: string): PackageSimulationListing {
  const rows: PackageSimulationFact[] = [];
  let recognized = true;
  let sawProof = false;
  for (const line of out.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith("Conf ")) { sawProof = true; continue; }
    // Inst/summary are matched on trimmed text; the simulation preamble and the indented
    // package list keep their raw indentation, which is part of their supported shape.
    const inst = INST_RE.exec(trimmed);
    if (inst) {
      const target = splitNameArch(inst[1]);
      if (!target) { recognized = false; continue; }
      const inner = inst[3].trim().split(/\s+/);
      const bracket = /\[([^\]]+)\]\s*$/.exec(inst[3]);
      const bracketArch = bracket && validPackageArch(bracket[1]) ? bracket[1] : null;
      const version = inner.find((token) => /^[0-9]/.test(token)) ?? null;
      rows.push({ name: target.name, arch: target.arch ?? bracketArch, version });
      sawProof = true;
      continue;
    }
    if (SIM_SUMMARY_RE.test(trimmed)) { sawProof = true; continue; }
    if (SIM_PREAMBLE_RE.some((re) => re.test(line) || re.test(trimmed))) continue;
    recognized = false;
  }
  return { rows, recognized: recognized && sawProof };
}

export function parseSimulationRows(out: string): PackageNameArch[] {
  return parseSimulationListing(out).rows.map(({ name, arch }) => ({ name, arch }));
}

export type PackageWarning =
  | "package-query-failed"
  | "candidates-unrecognized"
  | "hold-unavailable"
  | "simulation-unavailable"
  | "architecture-unavailable";

export type PackageChildFacts = { ok: boolean; observedAt: string; lastGoodAt: string | null };

export type PackageObservation = {
  rows: PackageRow[];
  holds: PackageNameArch[] | null;
  simulation: PackageSimulationFact[] | null;
  nativeArch: string | null;
  state: "ok" | "partial" | "failed";
  observedAt: string;
  lastGoodAt: string | null;
  stale: boolean;
  error: "package-query-failed" | null;
  warnings: PackageWarning[];
  children: {
    candidates: PackageChildFacts;
    holds: PackageChildFacts;
    simulation: PackageChildFacts;
    architecture: PackageChildFacts;
  };
};

export type PackageCandidateReason = "held" | "selected-now" | "deferred-unknown" | "unknown";

export type NormalizedPackageCandidate = {
  key: string;
  name: string;
  arch: string;
  fromVersion: string;
  toVersion: string;
  security: boolean;
  held: boolean;
  selectedNow: boolean;
  conflict: boolean;
  reason: PackageCandidateReason;
};

export type NormalizedPackageSet = {
  candidates: NormalizedPackageCandidate[];
  duplicateConflict: boolean;
  selectionKnown: boolean;
};

type FactMatch = { state: "match" } | { state: "no" } | { state: "ambiguous" } | { state: "unresolved" };

// A bare (architecture-less) policy fact resolves only to an unambiguous native or Arch:all
// candidate. Two eligible candidates stay unknown instead of being matched to both.
function matchPolicyFact(
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

// ONE normalized projection shared by the alert count and the Environment section. Identical
// duplicate rows count once; conflicting versions for one canonical key stay one unknown
// candidate and are never silently selected from the first row.
export function normalizePackageCandidates(observation: PackageObservation): NormalizedPackageSet {
  const nativeArch = observation.nativeArch;
  const byKey = new Map<string, { row: PackageRow; conflicting: boolean }>();
  let duplicateConflict = false;
  for (const row of observation.rows) {
    if (!validPackageName(row.name)) continue;
    if (row.arch && !validPackageArch(row.arch)) continue;
    const key = packageKey(row.name, row.arch, nativeArch);
    const existing = byKey.get(key);
    if (!existing) {
      byKey.set(key, { row, conflicting: false });
      continue;
    }
    if (existing.row.fromVersion !== row.fromVersion || existing.row.toVersion !== row.toVersion) {
      existing.conflicting = true;
      duplicateConflict = true;
    }
  }

  const archesByName = new Map<string, Set<string>>();
  for (const { row } of byKey.values()) {
    const arch = row.arch || nativeArch || "unknown";
    const set = archesByName.get(row.name) ?? new Set<string>();
    set.add(arch);
    archesByName.set(row.name, set);
  }

  const holds = observation.holds;
  const simulation = observation.simulation;
  let anyAmbiguous = false;
  let anyUnresolved = false;

  const candidates = [...byKey.values()].map((entry) => {
    const row = entry.row;
    const arch = row.arch || nativeArch || "unknown";
    const arches = archesByName.get(row.name) ?? new Set([arch]);
    let held = false;
    let simVersion: string | null = null;
    let simMatched = false;
    let ambiguous = false;
    let unresolved = false;
    if (holds !== null) {
      for (const fact of holds) {
        const match = matchPolicyFact(fact, row.name, arch, nativeArch, arches);
        if (match.state === "match") held = true;
        if (match.state === "ambiguous") ambiguous = true;
        if (match.state === "unresolved") unresolved = true;
      }
    }
    if (simulation !== null) {
      for (const fact of simulation) {
        const match = matchPolicyFact(fact, row.name, arch, nativeArch, arches);
        if (match.state === "match") { simMatched = true; simVersion = fact.version; }
        if (match.state === "ambiguous") ambiguous = true;
        if (match.state === "unresolved") unresolved = true;
      }
    }
    if (ambiguous) anyAmbiguous = true;
    if (unresolved) anyUnresolved = true;
    const selected = simMatched && simVersion !== null && simVersion === row.toVersion;
    const heldSelected = held && selected;
    const unresolvedEvidence = holds === null || simulation === null || ambiguous || unresolved;
    const reason: PackageCandidateReason = entry.conflicting || unresolvedEvidence || heldSelected
      ? "unknown"
      : held
        ? "held"
        : selected
          ? "selected-now"
          : "deferred-unknown";
    return {
      key: packageKey(row.name, row.arch, nativeArch),
      name: row.name,
      arch,
      fromVersion: row.fromVersion,
      toVersion: row.toVersion,
      security: row.security,
      held,
      selectedNow: reason === "selected-now",
      conflict: entry.conflicting || heldSelected,
      reason,
    } satisfies NormalizedPackageCandidate;
  }).sort((a, b) => a.key.localeCompare(b.key));

  return {
    candidates,
    duplicateConflict,
    // Retained last-good solver facts are historical evidence, never a current ready-work claim:
    // a failed holds/simulation child keeps selection unknown even though rows survive.
    selectionKnown: observation.children.holds.ok
      && observation.children.simulation.ok
      && holds !== null
      && simulation !== null
      && !anyAmbiguous
      && !anyUnresolved,
  };
}

export type FixedExecOptions = { timeout?: number; maxBuffer?: number; env?: NodeJS.ProcessEnv };
export type AsyncExec = (file: string, args: readonly string[], opts?: FixedExecOptions) => Promise<string>;

const execFileAsync = promisify(execFile);
const DEFAULT_EXEC = { timeout: 5_000, maxBuffer: 1024 * 1024 } as const;
// Every fixed subprocess runs in the C locale so apt/dpkg/ps wording never depends on the host.
const C_LOCALE = { LC_ALL: "C", LANG: "C" } as const;

export type RawExecFile = (
  file: string, args: readonly string[],
  options: { encoding: "utf8"; timeout: number; maxBuffer: number; env: NodeJS.ProcessEnv },
) => Promise<{ stdout: string }>;

export function createSystemAsyncExec(run: RawExecFile = execFileAsync, env: NodeJS.ProcessEnv = process.env): AsyncExec {
  return async (file, args, opts = {}) => {
    const { stdout } = await run(file, args, {
      encoding: "utf8",
      timeout: opts.timeout ?? DEFAULT_EXEC.timeout,
      maxBuffer: opts.maxBuffer ?? DEFAULT_EXEC.maxBuffer,
      env: { ...env, ...C_LOCALE, ...opts.env },
    });
    return stdout;
  };
}

const systemAsyncExec = createSystemAsyncExec();

export const PACKAGE_TTL_MS = 5 * 60 * 1000;
export const ARCH_RE = /^[a-z0-9-]{1,32}$/;

function iso(ms: number): string { return new Date(ms).toISOString(); }

function emptyChildren(at: number): PackageObservation["children"] {
  const facts: PackageChildFacts = { ok: false, observedAt: iso(at), lastGoodAt: null };
  return { candidates: { ...facts }, holds: { ...facts }, simulation: { ...facts }, architecture: { ...facts } };
}

// One cached, single-flight package observation shared by `getAlerts` and Environment (M5 F7):
// candidates, holds, simulation and native architecture are read together, each child keeps its
// own success/failure/last-good fact, and a cache miss cannot overlap another run.
export function createPackageObservationReader(exec: AsyncExec = systemAsyncExec, now: () => number = Date.now) {
  let cached: { at: number; obs: PackageObservation } | null = null;
  let inflight: Promise<PackageObservation> | null = null;
  const good = {
    candidates: null as { value: PackageRow[]; at: number } | null,
    holds: null as { value: PackageNameArch[]; at: number } | null,
    simulation: null as { value: PackageSimulationFact[]; at: number } | null,
    architecture: null as { value: string; at: number } | null,
  };
  return {
    reset() { cached = null; inflight = null; },
    peek(): PackageObservation | null { return cached?.obs ?? null; },
    async read(): Promise<PackageObservation> {
      const t = now();
      if (cached && t - cached.at < PACKAGE_TTL_MS) return cached.obs;
      if (inflight) return inflight;
      inflight = (async (): Promise<PackageObservation> => {
        const [candidateResult, holdResult, simulationResult, archResult] = await Promise.allSettled([
          exec("apt", ["list", "--upgradable"], { timeout: 5_000, maxBuffer: 1024 * 1024 }),
          exec("apt-mark", ["showhold"], { timeout: 5_000, maxBuffer: 1024 * 1024 }),
          exec("apt-get", ["--simulate", "upgrade"], { timeout: 10_000, maxBuffer: 2 * 1024 * 1024 }),
          exec("dpkg", ["--print-architecture"], { timeout: 2_000, maxBuffer: 4 * 1024 }),
        ]);
        const warnings: PackageWarning[] = [];
        const children = emptyChildren(t);

        let candidatesOk = false;
        let rows: PackageRow[] = [];
        if (candidateResult.status === "fulfilled") {
          const listing = parsePackageListing(candidateResult.value);
          if (listing.recognized) {
            candidatesOk = true;
            rows = listing.rows;
            good.candidates = { value: listing.rows, at: t };
            children.candidates = { ok: true, observedAt: iso(t), lastGoodAt: iso(t) };
          } else {
            warnings.push("candidates-unrecognized");
            rows = listing.rows.length > 0 ? listing.rows : good.candidates?.value ?? [];
          }
        } else {
          warnings.push("package-query-failed");
          rows = good.candidates?.value ?? [];
        }
        if (!candidatesOk) {
          children.candidates = {
            ok: false, observedAt: iso(t),
            lastGoodAt: good.candidates === null ? null : iso(good.candidates.at),
          };
        }

        let holds: PackageNameArch[] | null = null;
        if (holdResult.status === "fulfilled") {
          const listing = parseHoldListing(holdResult.value);
          if (listing.recognized) {
            holds = listing.rows;
            good.holds = { value: listing.rows, at: t };
            children.holds = { ok: true, observedAt: iso(t), lastGoodAt: iso(t) };
          }
        }
        if (holds === null) {
          warnings.push("hold-unavailable");
          holds = good.holds?.value ?? null;
          children.holds = { ok: false, observedAt: iso(t), lastGoodAt: good.holds === null ? null : iso(good.holds.at) };
        }

        let simulation: PackageSimulationFact[] | null = null;
        if (simulationResult.status === "fulfilled") {
          const listing = parseSimulationListing(simulationResult.value);
          if (listing.recognized) {
            simulation = listing.rows;
            good.simulation = { value: listing.rows, at: t };
            children.simulation = { ok: true, observedAt: iso(t), lastGoodAt: iso(t) };
          }
        }
        if (simulation === null) {
          warnings.push("simulation-unavailable");
          simulation = good.simulation?.value ?? null;
          children.simulation = { ok: false, observedAt: iso(t), lastGoodAt: good.simulation === null ? null : iso(good.simulation.at) };
        }

        let nativeArch: string | null = null;
        if (archResult.status === "fulfilled") {
          const value = archResult.value.trim();
          if (ARCH_RE.test(value)) {
            nativeArch = value;
            good.architecture = { value, at: t };
            children.architecture = { ok: true, observedAt: iso(t), lastGoodAt: iso(t) };
          }
        }
        if (nativeArch === null) {
          warnings.push("architecture-unavailable");
          nativeArch = good.architecture?.value ?? null;
          children.architecture = { ok: false, observedAt: iso(t), lastGoodAt: good.architecture === null ? null : iso(good.architecture.at) };
        }

        const lastGoodAt = good.candidates === null ? null : iso(good.candidates.at);
        const state: PackageObservation["state"] = candidatesOk
          ? (warnings.length > 0 ? "partial" : "ok")
          : "failed";
        const obs: PackageObservation = {
          rows,
          holds,
          simulation,
          nativeArch,
          state,
          observedAt: iso(t),
          lastGoodAt,
          stale: !candidatesOk && lastGoodAt !== null,
          error: candidateResult.status === "rejected" ? "package-query-failed" : null,
          warnings,
          children,
        };
        cached = { at: t, obs };
        return obs;
      })().finally(() => { inflight = null; });
      return inflight;
    },
  };
}

export type PackageObservationReader = ReturnType<typeof createPackageObservationReader>;

export const packageObservation = createPackageObservationReader();
export const getPackageObservation = (): Promise<PackageObservation> => packageObservation.read();

export type GetAlertsOptions = {
  exec?: AsyncExec;
  packages?: () => Promise<PackageObservation>;
  packageSnapshot?: () => PackageObservation | null;
  packageWaitMs?: number;
  loadavg?: () => Promise<string>;
};

export const PACKAGE_ALERT_AWAIT_MS = 1_000;

function unavailableObservation(base: PackageObservation | null, at: number): PackageObservation {
  const children = emptyChildren(at);
  if (base) {
    children.candidates = { ok: false, observedAt: iso(at), lastGoodAt: base.lastGoodAt };
  }
  return {
    rows: base?.rows ?? [], holds: base?.holds ?? null, simulation: base?.simulation ?? null,
    nativeArch: base?.nativeArch ?? null, state: "failed",
    observedAt: iso(at), lastGoodAt: base?.lastGoodAt ?? null,
    stale: base?.lastGoodAt != null,
    error: "package-query-failed",
    warnings: [...new Set([...(base?.warnings ?? []), "package-query-failed" as const])],
    children: { ...(base?.children ?? children), candidates: children.candidates },
  };
}

// A slow package probe may take a full second; the rest of the alerts always resolve. The
// original pending probe is never cancelled or duplicated by this timeout.
async function packagesWithinBudget(
  packages: () => Promise<PackageObservation>,
  snapshot: () => PackageObservation | null,
  waitMs: number,
  now: () => number,
): Promise<PackageObservation> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const deadline = new Promise<"timeout">((resolve) => { timer = setTimeout(() => resolve("timeout"), waitMs); });
  try {
    const result = await Promise.race([
      packages().then((obs) => ({ kind: "obs" as const, obs }), () => ({ kind: "error" as const })),
      deadline.then(() => ({ kind: "timeout" as const })),
    ]);
    if (result.kind === "obs") return result.obs;
    return unavailableObservation(snapshot(), now());
  } finally {
    if (timer) clearTimeout(timer);
  }
}

// Each probe fails independently and becomes an explicit source alert; the call itself never
// rejects for an ordinary source failure (error contract: visible warning, not a blank card).
export async function getAlerts(options: GetAlertsOptions = {}): Promise<Alert[]> {
  const exec = options.exec ?? systemAsyncExec;
  const packages = options.packages ?? getPackageObservation;
  const snapshot = options.packageSnapshot ?? (() => packageObservation.peek());
  const loadavg = options.loadavg ?? (() => readFile("/proc/loadavg", "utf8"));
  const alerts: Alert[] = [];
  try {
    alerts.push(...parseDfAlerts(await exec("df", ["-P", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs"], DEFAULT_EXEC)));
  } catch {
    alerts.push({ kind: "source", source: "disk", state: "failed" });
  }
  try {
    const cores = Number((await exec("nproc", [], DEFAULT_EXEC)).trim());
    const load = parseLoadAlert(await loadavg(), cores);
    if (load) alerts.push(load);
  } catch {
    alerts.push({ kind: "source", source: "load", state: "failed" });
  }
  try {
    const rows = parseZombieRows(await exec("ps", ["-eo", "pid,ppid,stat,etimes,comm"], { timeout: 5_000, maxBuffer: 2 * 1024 * 1024 }));
    if (rows.length > 0) alerts.push({ kind: "zombies", count: rows.length, rows });
  } catch {
    alerts.push({ kind: "source", source: "zombies", state: "failed" });
  }
  const observation = await packagesWithinBudget(
    packages, snapshot, options.packageWaitMs ?? PACKAGE_ALERT_AWAIT_MS, Date.now,
  );
  const normalized = normalizePackageCandidates(observation).candidates;
  if (normalized.length > 0) {
    alerts.push({ kind: "updates", count: normalized.length, rows: normalized });
  }
  // A partial observation keeps its known total but never becomes a complete-count/all-clear
  // badge: the existing source warning rides alongside the rows.
  if (observation.state !== "ok") {
    alerts.push({ kind: "source", source: "packages", state: "failed" });
  }
  return alerts;
}
