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

// M5 F6: identity is `name:arch`. Bare names take the fixed native architecture (read by the
// caller with `dpkg --print-architecture`); when that is unknown, the key stays explicitly
// ambiguous instead of silently collapsing two architectures.
export function packageKey(name: string, arch: string | null, nativeArch: string | null): string {
  const resolved = arch && arch.length > 0 ? arch : nativeArch && nativeArch.length > 0 ? nativeArch : "unknown";
  return `${name}:${resolved}`;
}

function splitNameArch(token: string): PackageNameArch | null {
  if (!/^[A-Za-z0-9][A-Za-z0-9+.-]*(?::[A-Za-z0-9-]+)?$/.test(token)) return null;
  const at = token.lastIndexOf(":");
  return at > 0 ? { name: token.slice(0, at), arch: token.slice(at + 1) } : { name: token, arch: null };
}

const PACKAGE_LINE_RE = /^([^/\s]+)\/(\S+)\s+(\S+)\s+(\S+)\s+\[upgradable from:\s*([^\]]+)\]/;

// Parses `apt list --upgradable` (English locale only; revisit if apt output ever localizes);
// a SUITE name (after "/") carrying "-security" marks it security-sourced. A line
// missing the bracketed version pair is skipped, not crashed on.
export function parsePackageRows(aptOut: string): PackageRow[] {
  const rows: PackageRow[] = [];
  for (const line of aptOut.split(/\r?\n/)) {
    const m = PACKAGE_LINE_RE.exec(line);
    if (!m) continue;
    const [, name, suite, toVersion, arch, fromVersion] = m;
    rows.push({
      name, arch, key: packageKey(name, arch, null),
      fromVersion: fromVersion.trim(), toVersion, security: suite.includes("-security"),
    });
  }
  return rows;
}

// `apt-mark showhold` prints one bare (or name:arch) package per line.
export function parseHoldRows(out: string): PackageNameArch[] {
  const rows: PackageNameArch[] = [];
  for (const line of out.split(/\r?\n/)) {
    const parsed = splitNameArch(line.trim());
    if (parsed) rows.push(parsed);
  }
  return rows;
}

// `apt-get --simulate upgrade` marks each selected package with an `Inst` row.
export function parseSimulationRows(out: string): PackageNameArch[] {
  const rows: PackageNameArch[] = [];
  for (const line of out.split(/\r?\n/)) {
    const m = /^Inst\s+(\S+)\s+\[/.exec(line);
    if (!m) continue;
    const parsed = splitNameArch(m[1]);
    if (parsed) rows.push(parsed);
  }
  return rows;
}

export type PackageObservation = {
  rows: PackageRow[];
  state: "ok" | "failed";
  observedAt: string;
  lastGoodAt: string | null;
  stale: boolean;
  error: "package-query-failed" | null;
};

export type FixedExecOptions = { timeout?: number; maxBuffer?: number; env?: NodeJS.ProcessEnv };
export type AsyncExec = (file: string, args: readonly string[], opts?: FixedExecOptions) => Promise<string>;

const execFileAsync = promisify(execFile);
const DEFAULT_EXEC = { timeout: 5_000, maxBuffer: 1024 * 1024 } as const;
// Every fixed subprocess runs in the C locale so apt/dpkg/ps wording never depends on the host.
const C_LOCALE = { LC_ALL: "C", LANG: "C" } as const;
const systemAsyncExec: AsyncExec = async (file, args, opts = {}) => {
  const { stdout } = await execFileAsync(file, args, {
    encoding: "utf8",
    timeout: opts.timeout ?? DEFAULT_EXEC.timeout,
    maxBuffer: opts.maxBuffer ?? DEFAULT_EXEC.maxBuffer,
    env: { ...process.env, ...C_LOCALE, ...opts.env },
  });
  return stdout;
};

export const PACKAGE_TTL_MS = 5 * 60 * 1000;

// One cached, single-flight package observation shared by `getAlerts` and Environment (M5 F7):
// a cache miss cannot overlap another apt run, and a pending probe never delays resources.
export function createPackageObservationReader(exec: AsyncExec = systemAsyncExec, now: () => number = Date.now) {
  let cache: { at: number; obs: PackageObservation } | null = null;
  let inflight: Promise<PackageObservation> | null = null;
  return {
    reset() { cache = null; inflight = null; },
    async read(): Promise<PackageObservation> {
      const t = now();
      if (cache && t - cache.at < PACKAGE_TTL_MS) return cache.obs;
      if (inflight) return inflight;
      inflight = (async (): Promise<PackageObservation> => {
        try {
          const rows = parsePackageRows(await exec("apt", ["list", "--upgradable"], { timeout: 5_000, maxBuffer: 1024 * 1024 }));
          const obs: PackageObservation = {
            rows, state: "ok", observedAt: new Date(t).toISOString(),
            lastGoodAt: new Date(t).toISOString(), stale: false, error: null,
          };
          cache = { at: t, obs };
          return obs;
        } catch {
          const prev = cache?.obs;
          const obs: PackageObservation = {
            rows: prev?.rows ?? [], state: "failed", observedAt: new Date(t).toISOString(),
            lastGoodAt: prev?.lastGoodAt ?? null, stale: prev !== null, error: "package-query-failed",
          };
          cache = { at: t, obs };
          return obs;
        }
      })().finally(() => { inflight = null; });
      return inflight;
    },
  };
}

export const packageObservation = createPackageObservationReader();
export const getPackageObservation = (): Promise<PackageObservation> => packageObservation.read();

export type GetAlertsOptions = {
  exec?: AsyncExec;
  packages?: () => Promise<PackageObservation>;
};

// Each probe fails independently and becomes an explicit source alert; the call itself never
// rejects for an ordinary source failure (error contract: visible warning, not a blank card).
export async function getAlerts(options: GetAlertsOptions = {}): Promise<Alert[]> {
  const exec = options.exec ?? systemAsyncExec;
  const packages = options.packages ?? getPackageObservation;
  const alerts: Alert[] = [];
  try {
    alerts.push(...parseDfAlerts(await exec("df", ["-P", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs"], DEFAULT_EXEC)));
  } catch {
    alerts.push({ kind: "source", source: "disk", state: "failed" });
  }
  try {
    const cores = Number((await exec("nproc", [], DEFAULT_EXEC)).trim());
    const load = parseLoadAlert(await readFile("/proc/loadavg", "utf8"), cores);
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
  const observation = await packages();
  if (observation.rows.length > 0) {
    alerts.push({ kind: "updates", count: observation.rows.length, rows: observation.rows });
  }
  if (observation.state === "failed") {
    alerts.push({ kind: "source", source: "packages", state: "failed" });
  }
  return alerts;
}
