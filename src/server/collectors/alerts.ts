import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { LOCAL_OPTS } from "../inputLimits";

export type ZombieRow = { pid: number; ppid: number; ageSeconds: number; parentCommand: string | null };
export type PackageRow = { name: string; fromVersion: string; toVersion: string; security: boolean };

export type Alert =
  | { kind: "disk"; mount: string; usedPercent: number }
  | { kind: "load"; load5: number; cores: number }
  | { kind: "zombies"; count: number; rows: ZombieRow[] }
  | { kind: "updates"; count: number; rows: PackageRow[] };

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

export function countZombies(psOut: string): number {
  return psOut
    .split(/\r?\n/)
    .slice(1)
    .filter((l) => l.trim().startsWith("Z")).length;
}

// ponytail: matches the English apt phrasing — verified this box runs an
// English locale; revisit only if apt output ever localizes
export function countUpgradable(aptOut: string): number {
  return aptOut.split(/\r?\n/).filter((l) => l.includes("upgradable from")).length;
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

const PACKAGE_LINE_RE = /^([^/\s]+)\/(\S+)\s+(\S+)\s+\S+\s+\[upgradable from:\s*([^\]]+)\]/;

// Parses `apt list --upgradable` (English locale, same convention as countUpgradable);
// a SUITE name (after "/") carrying "-security" marks it security-sourced. A line
// missing the bracketed version pair is skipped, not crashed on.
export function parsePackageRows(aptOut: string): PackageRow[] {
  const rows: PackageRow[] = [];
  for (const line of aptOut.split(/\r?\n/)) {
    const m = PACKAGE_LINE_RE.exec(line);
    if (!m) continue;
    const [, name, suite, toVersion, fromVersion] = m;
    rows.push({ name, fromVersion: fromVersion.trim(), toVersion, security: suite.includes("-security") });
  }
  return rows;
}

// apt list is ~0.5s today; 1h cache kept as apt-lock insurance per spec
let aptCache: { at: number; rows: PackageRow[] } | null = null;
const APT_TTL_MS = 60 * 60 * 1000;

// Untested shell glue. Each probe fails independently — one broken probe
// must not blank the whole card (these are core utils; failure ≈ never).
export type Utf8ExecSync = (
  file: string,
  args: readonly string[],
  options: { encoding: "utf8"; timeout: number; maxBuffer: number },
) => string;

const APT_OPTS = { encoding: "utf8" as const, timeout: 15_000, maxBuffer: 2 * 1024 * 1024 };

function isLimitError(e: unknown): boolean {
  const code = (e as { code?: unknown }).code;
  return code === "ETIMEDOUT" || code === "ERR_CHILD_PROCESS_STDIO_MAXBUFFER";
}

export function getAlerts(exec: Utf8ExecSync = execFileSync as Utf8ExecSync): Alert[] {
  const alerts: Alert[] = [];
  try {
    const df = exec("df", ["-P", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs"], LOCAL_OPTS);
    alerts.push(...parseDfAlerts(df));
  } catch (e) {
    if (isLimitError(e)) throw e;
  }
  try {
    const cores = Number(exec("nproc", [], LOCAL_OPTS).trim());
    const load = parseLoadAlert(readFileSync("/proc/loadavg", "utf8"), cores);
    if (load) alerts.push(load);
  } catch (e) {
    if (isLimitError(e)) throw e;
  }
  try {
    const rows = parseZombieRows(exec("ps", ["-eo", "pid,ppid,stat,etimes,comm"], LOCAL_OPTS));
    if (rows.length > 0) alerts.push({ kind: "zombies", count: rows.length, rows });
  } catch (e) {
    if (isLimitError(e)) throw e;
  }
  try {
    if (!aptCache || Date.now() - aptCache.at > APT_TTL_MS) {
      const out = exec("apt", ["list", "--upgradable"], APT_OPTS);
      aptCache = { at: Date.now(), rows: parsePackageRows(out) };
    }
    if (aptCache.rows.length > 0) alerts.push({ kind: "updates", count: aptCache.rows.length, rows: aptCache.rows });
  } catch (e) {
    if (isLimitError(e)) throw e;
  }
  return alerts;
}
