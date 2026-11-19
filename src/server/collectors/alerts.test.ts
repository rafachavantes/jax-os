import { describe, expect, it, vi } from "vitest";
import {
  createPackageObservationReader, createSystemAsyncExec, getAlerts, normalizePackageCandidates, packageKey,
  parseDfAlerts, parseHoldListing, parseHoldRows, parseLoadAlert, parsePackageListing, parsePackageRows,
  parseSimulationListing, parseSimulationRows, parseZombieRows, PACKAGE_TTL_MS, type PackageObservation,
} from "./alerts";

vi.mock("./alerts", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./alerts")>();
  return { ...actual, getAlerts: vi.fn(actual.getAlerts) };
});

import { GET } from "../../app/api/mission/alerts/route";

const DF = `Filesystem     1024-blocks     Used Available Capacity Mounted on
/dev/sda1        157197504 92993088  57759796      62% /
/dev/sdb1        100000000 90000000  10000000      90% /data
tmpfs               100000    99000      1000      99% /run/lock
`;

describe("parseDfAlerts", () => {
  it("alerts only /dev/* filesystems over threshold", () => {
    expect(parseDfAlerts(DF)).toEqual([{ kind: "disk", mount: "/data", usedPercent: 90 }]);
  });

  it("never alerts on tmpfs-style rows even if df flags them", () => {
    expect(parseDfAlerts(DF).some((a) => a.kind === "disk" && a.mount === "/run/lock")).toBe(false);
  });

  it("handles mount points containing spaces", () => {
    const out = `Filesystem 1024-blocks Used Available Capacity Mounted on
/dev/sdc1 100 95 5 95% /mnt/my disk
`;
    expect(parseDfAlerts(out)).toEqual([{ kind: "disk", mount: "/mnt/my disk", usedPercent: 95 }]);
  });
});

describe("parseLoadAlert", () => {
  it("uses the 5-min field, alerts only above cores", () => {
    expect(parseLoadAlert("9.75 1.29 1.07 7/1684 3152292", 8)).toBeNull();
    expect(parseLoadAlert("1.00 9.50 1.07 7/1684 3152292", 8)).toEqual({
      kind: "load",
      load5: 9.5,
      cores: 8,
    });
  });
});

const execFor = (files: Record<string, string | Error>) => vi.fn(async (file: string) => {
  const value = files[file];
  if (value instanceof Error) throw value;
  if (value === undefined) throw new Error(`unexpected ${file}`);
  return value;
});

function childFacts(at: string, ok = true, lastGoodAt: string | null = ok ? at : null) {
  return { ok, observedAt: at, lastGoodAt };
}

function observation(over: Partial<PackageObservation> = {}): PackageObservation {
  const at = "2026-11-19T00:00:00.000Z";
  const facts = childFacts(at);
  return {
    rows: [], holds: [], simulation: [], nativeArch: "amd64",
    state: "ok", observedAt: at, lastGoodAt: at, stale: false, error: null, warnings: [],
    children: {
      candidates: { ...facts }, holds: { ...facts }, simulation: { ...facts }, architecture: { ...facts },
    },
    ...over,
  };
}

const emptyPackages = async () => observation();

const ALERT_EXECS = {
  df: DF,
  nproc: "8\n",
  ps: "STAT\n",
};

const loadavgOk = async () => "0.10 0.10 0.10 1/2 3\n";

describe("getAlerts — explicit source states (M5 Task 3)", () => {
  it("a timed-out probe becomes a source warning instead of throwing or vanishing", async () => {
    const exec = vi.fn(async (file: string) => {
      if (file === "df") throw Object.assign(new Error("timeout"), { code: "ETIMEDOUT" });
      if (file === "nproc") return "8\n";
      if (file === "ps") return "STAT\n";
      throw new Error("unexpected");
    });
    const alerts = await getAlerts({ exec, packages: emptyPackages, loadavg: loadavgOk });
    expect(alerts).toContainEqual({ kind: "source", source: "disk", state: "failed" });
  });

  it("an overflowed probe is the same explicit unavailable state, while siblings still return", async () => {
    const exec = vi.fn(async (file: string) => {
      if (file === "df") return DF;
      if (file === "nproc") return "8\n";
      if (file === "ps") throw Object.assign(new Error("overflow"), { code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER" });
      throw new Error("unexpected");
    });
    const alerts = await getAlerts({ exec, packages: emptyPackages, loadavg: loadavgOk });
    expect(alerts).toContainEqual({ kind: "source", source: "zombies", state: "failed" });
    expect(alerts).toContainEqual({ kind: "disk", mount: "/data", usedPercent: 90 });
  });

  it("a failed package observation preserves last-good rows and flags the source", async () => {
    const exec = execFor(ALERT_EXECS);
    const alerts = await getAlerts({
      exec,
      loadavg: loadavgOk,
      packages: async () => observation({
        rows: [{ name: "openssl", arch: "amd64", key: "openssl:amd64", fromVersion: "1", toVersion: "2", security: true }],
        state: "failed", lastGoodAt: "2026-11-18T23:55:00.000Z", stale: true,
        error: "package-query-failed", warnings: ["package-query-failed"],
        children: {
          candidates: childFacts("2026-11-19T00:00:00.000Z", false, "2026-11-18T23:55:00.000Z"),
          holds: childFacts("2026-11-19T00:00:00.000Z", false, null),
          simulation: childFacts("2026-11-19T00:00:00.000Z", false, null),
          architecture: childFacts("2026-11-19T00:00:00.000Z", false, null),
        },
      }),
    });
    expect(alerts).toContainEqual({ kind: "source", source: "packages", state: "failed" });
    expect(alerts.find((a) => a.kind === "updates")).toMatchObject({ count: 1 });
  });

  it("keeps a partial package observation's known rows and its visible source warning", async () => {
    const alerts = await getAlerts({
      exec: execFor(ALERT_EXECS),
      loadavg: loadavgOk,
      packages: async () => observation({
        rows: [{ name: "bar", arch: "amd64", key: "bar:amd64", fromVersion: "1", toVersion: "2", security: false }],
        state: "partial", warnings: ["simulation-unavailable"],
      }),
    });
    expect(alerts.find((a) => a.kind === "updates")).toMatchObject({ count: 1, rows: [{ key: "bar:amd64" }] });
    expect(alerts).toContainEqual({ kind: "source", source: "packages", state: "failed" });
  });

  it("never reads the real /proc/loadavg when the loader is injected", async () => {
    const loadavg = vi.fn(loadavgOk);
    const exec = execFor(ALERT_EXECS);
    await getAlerts({ exec, packages: emptyPackages, loadavg });
    expect(loadavg).toHaveBeenCalledTimes(1);
  });

  it("calls ps with the existing widened column set and keeps the update/alert count from one projection", async () => {
    const psOut = "  PID  PPID STAT ELAPSED COMMAND\n41822     1  Ssl    7260 node\n48211 41822  Z+     7260 node\n";
    const exec = execFor({ df: DF, nproc: "8\n", ps: psOut });
    const alerts = await getAlerts({ exec, packages: emptyPackages, loadavg: loadavgOk });
    expect(exec).toHaveBeenCalledWith("ps", ["-eo", "pid,ppid,stat,etimes,comm"], { timeout: 5_000, maxBuffer: 2 * 1024 * 1024 });
    expect(alerts.find((a) => a.kind === "zombies")).toEqual({
      kind: "zombies", count: 1,
      rows: [{ pid: 48211, ppid: 41822, ageSeconds: 7260, parentCommand: "node" }],
    });
  });

  it("resolves disk/load/zombies while the package probe is still pending, without cancelling it", async () => {
    const exec = execFor(ALERT_EXECS);
    const snapshot = observation({
      rows: [{ name: "openssl", arch: "amd64", key: "openssl:amd64", fromVersion: "1", toVersion: "2", security: false }],
      state: "failed", lastGoodAt: "2026-11-18T23:55:00.000Z", stale: true, warnings: ["package-query-failed"],
    });
    const packages = vi.fn(() => new Promise<PackageObservation>(() => {}));
    const alerts = await getAlerts({
      exec, loadavg: loadavgOk, packages, packageSnapshot: () => snapshot, packageWaitMs: 5,
    });
    expect(packages).toHaveBeenCalledTimes(1);
    expect(alerts).toContainEqual({ kind: "disk", mount: "/data", usedPercent: 90 });
    expect(alerts).toContainEqual({ kind: "source", source: "packages", state: "failed" });
    expect(alerts.find((a) => a.kind === "updates")).toMatchObject({ count: 1 });
  });
});

describe("parseZombieRows", () => {
  it("returns an empty array with no zombies", () => {
    expect(parseZombieRows("  PID  PPID STAT ELAPSED COMMAND\n")).toEqual([]);
  });

  it("resolves the parent command from a row in the same snapshot", () => {
    const out = "  PID  PPID STAT ELAPSED COMMAND\n" +
      "41822     1  Ssl    7260 node\n" +
      "48211 41822  Z+     7260 node\n";
    expect(parseZombieRows(out)).toEqual([
      { pid: 48211, ppid: 41822, ageSeconds: 7260, parentCommand: "node" },
    ]);
  });

  it("reports parentCommand: null, never a placeholder, when the parent row is missing (round-3 F3)", () => {
    const out = "  PID  PPID STAT ELAPSED COMMAND\n" +
      "48211 41822  Z+     7260 node\n";
    expect(parseZombieRows(out)).toEqual([
      { pid: 48211, ppid: 41822, ageSeconds: 7260, parentCommand: null },
    ]);
  });
});

describe("package parsers — architecture identity (M5 Task 2)", () => {
  it("returns an empty listing with no upgrades", () => {
    expect(parsePackageRows("Listing...\n")).toEqual([]);
    expect(parsePackageListing("Listing...\n")).toEqual({ rows: [], recognized: true });
  });

  it("separates security from non-security lines and keys by name:arch", () => {
    const out = "Listing...\n" +
      "openssl/noble-security 3.0.15 amd64 [upgradable from: 3.0.13]\n" +
      "curl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\n" +
      "libc6/noble 2.39-0ubuntu8 all [upgradable from: 2.38]\n";
    expect(parsePackageRows(out)).toEqual([
      { name: "openssl", arch: "amd64", key: "openssl:amd64", fromVersion: "3.0.13", toVersion: "3.0.15", security: true },
      { name: "curl", arch: "amd64", key: "curl:amd64", fromVersion: "8.4.0", toVersion: "8.5.0", security: false },
      { name: "libc6", arch: "all", key: "libc6:all", fromVersion: "2.38", toVersion: "2.39-0ubuntu8", security: false },
    ]);
  });

  it("preserves known rows but flags malformed or unsupported prose", () => {
    const out = "Listing...\ncurl/noble 8.5.0 amd64 [upgradable from\nvim/noble 2:9.1 amd64 [upgradable from: 2:9.0]\n";
    expect(parsePackageListing(out)).toEqual({
      recognized: false,
      rows: [{ name: "vim", arch: "amd64", key: "vim:amd64", fromVersion: "2:9.0", toVersion: "2:9.1", security: false }],
    });
  });

  it("keeps two architectures of the same name as distinct rows", () => {
    const out = "Listing...\nfoo/noble 2 amd64 [upgradable from: 1]\nfoo/noble 2 i386 [upgradable from: 1]\n";
    expect(parsePackageRows(out).map((r) => r.key)).toEqual(["foo:amd64", "foo:i386"]);
  });

  it("normalizes bare and name:arch hold rows and treats empty output as valid empty", () => {
    expect(parseHoldRows("openssl\ncurl\n\nlibc6:i386\nnot a package\n")).toEqual([
      { name: "openssl", arch: null },
      { name: "curl", arch: null },
      { name: "libc6", arch: "i386" },
    ]);
    expect(parseHoldListing("")).toEqual({ rows: [], recognized: true });
    expect(parseHoldListing("not a package")).toEqual({ rows: [], recognized: false });
  });

  it("extracts Inst targets plus the bracketed architecture and candidate version", () => {
    const out = "Reading package lists...\n" +
      "Calculating upgrade...\n" +
      "The following packages will be upgraded:\n" +
      "  openssl\n" +
      "1 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.\n" +
      "Inst openssl [3.0.13] (noble-security 3.0.15 amd64 [amd64])\n" +
      "Inst libc6:i386 [2.38] (noble 2.39 i386 [i386])\n" +
      "Inst bar [1] (stable 2 stable [all])\n" +
      "Conf openssl (3.0.15 noble-security [amd64])\n";
    expect(parseSimulationListing(out).rows).toEqual([
      { name: "openssl", arch: "amd64", version: "3.0.15" },
      { name: "libc6", arch: "i386", version: "2.39" },
      { name: "bar", arch: "all", version: "2" },
    ]);
    expect(parseSimulationRows(out)).toEqual([
      { name: "openssl", arch: "amd64" },
      { name: "libc6", arch: "i386" },
      { name: "bar", arch: "all" },
    ]);
  });

  it("accepts the real unprivileged simulation preamble with the indented package list", () => {
    const out = [
      "NOTE: This is only a simulation!",
      " apt-get needs root privileges for real execution.",
      " Keep also in mind that locking is deactivated,",
      " so don't depend on the relevance to the real current situation!",
      "Reading package lists...",
      "Building dependency tree...",
      "Reading state information...",
      "The following packages will be upgraded:",
      "  bar",
      "Inst bar [1] (2 stable [all])",
      "1 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.",
    ].join("\n");
    expect(parseSimulationListing(out)).toEqual({ rows: [{ name: "bar", arch: "all", version: "2" }], recognized: true });
  });

  it("recognizes a zero-upgrade summary but never empty/arbitrary prose as proof of zero", () => {
    expect(parseSimulationListing("Reading package lists...\n0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.\n").recognized).toBe(true);
    expect(parseSimulationListing("").recognized).toBe(false);
    expect(parseSimulationListing("some arbitrary prose").recognized).toBe(false);
  });

  it("bare keys use the fixed native architecture, and ambiguity stays unknown", () => {
    expect(packageKey("openssl", null, "amd64")).toBe("openssl:amd64");
    expect(packageKey("openssl", "i386", "amd64")).toBe("openssl:i386");
    expect(packageKey("openssl", null, null)).toBe("openssl:unknown");
  });
});

describe("shared normalized candidates (observation completion Task 1)", () => {
  const row = (over: Partial<PackageObservation["rows"][number]> = {}) => ({
    name: "bar", arch: "amd64", key: "bar:amd64", fromVersion: "1", toVersion: "2", security: false, ...over,
  });

  it("dedupes identical rows once while keeping conflicting versions as one unknown candidate", () => {
    const identical = normalizePackageCandidates(observation({ rows: [row(), row()] }));
    expect(identical.candidates).toHaveLength(1);
    expect(identical.duplicateConflict).toBe(false);

    const conflicting = normalizePackageCandidates(observation({
      rows: [row(), row({ fromVersion: "0", toVersion: "3" })],
      simulation: [{ name: "bar", arch: "amd64", version: "2" }],
    }));
    expect(conflicting.candidates).toHaveLength(1);
    expect(conflicting.candidates[0]).toMatchObject({ key: "bar:amd64", reason: "unknown", selectedNow: false, conflict: true });
    expect(conflicting.duplicateConflict).toBe(true);
  });

  it("keeps missing selection evidence unknown instead of a solver-deferral claim", () => {
    const missing = normalizePackageCandidates(observation({ rows: [row()], simulation: null }));
    expect(missing.candidates[0]).toMatchObject({ reason: "unknown", selectedNow: false });
    expect(missing.selectionKnown).toBe(false);

    const noHolds = normalizePackageCandidates(observation({ rows: [row()], holds: null }));
    expect(noHolds.candidates[0]).toMatchObject({ reason: "unknown", selectedNow: false });
    expect(noHolds.selectionKnown).toBe(false);
  });

  it("keeps a held-and-selected conflict out of the held and selected classification", () => {
    const conflict = normalizePackageCandidates(observation({
      rows: [row()],
      holds: [{ name: "bar", arch: "amd64" }],
      simulation: [{ name: "bar", arch: "amd64", version: "2" }],
    }));
    expect(conflict.candidates[0]).toMatchObject({ held: true, selectedNow: false, conflict: true, reason: "unknown" });
  });
});

describe("fixed subprocess locale (M5 Task 1)", () => {
  it("pins LC_ALL/LANG=C and applies the requested budgets on every apt/dpkg call", async () => {
    const calls: { file: string; options: Record<string, unknown> }[] = [];
    const exec = createSystemAsyncExec(async (file, _args, options) => {
      calls.push({ file, options });
      return { stdout: "" };
    }, { PATH: "/bin", NODE_ENV: "test" });
    await exec("apt", ["list", "--upgradable"], { timeout: 5_000, maxBuffer: 1024 * 1024 });
    await exec("dpkg", ["--print-architecture"], { timeout: 2_000, maxBuffer: 4 * 1024 });
    for (const call of calls) expect(call.options.env).toMatchObject({ LC_ALL: "C", LANG: "C" });
    expect(calls[0].options).toMatchObject({ timeout: 5_000, maxBuffer: 1024 * 1024 });
    expect(calls[1].options).toMatchObject({ timeout: 2_000, maxBuffer: 4 * 1024 });
  });
});

describe("shared package observation (M5 Task 2)", () => {
  const healthyExec = () => vi.fn(async (file: string) => {
    if (file === "apt") return "Listing...\ncurl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\n";
    if (file === "apt-mark") return "curl\n";
    if (file === "apt-get") return "Inst curl [8.4.0] (noble 8.5.0 amd64 [amd64])\n";
    if (file === "dpkg") return "amd64\n";
    throw new Error(`unexpected ${file}`);
  });

  it("single-flights one combined observation and caches it for five minutes", async () => {
    let now = 1_000_000;
    const exec = healthyExec();
    const reader = createPackageObservationReader(exec, () => now);
    const [a, b] = await Promise.all([reader.read(), reader.read()]);
    expect(exec).toHaveBeenCalledTimes(4);
    expect(exec).toHaveBeenCalledWith("apt", ["list", "--upgradable"], { timeout: 5_000, maxBuffer: 1024 * 1024 });
    expect(exec).toHaveBeenCalledWith("apt-get", ["--simulate", "upgrade"], { timeout: 10_000, maxBuffer: 2 * 1024 * 1024 });
    expect(exec).toHaveBeenCalledWith("dpkg", ["--print-architecture"], { timeout: 2_000, maxBuffer: 4 * 1024 });
    expect(a).toMatchObject({ rows: [{ key: "curl:amd64" }], state: "ok", nativeArch: "amd64", stale: false });
    expect(b).toBe(a);

    now += PACKAGE_TTL_MS - 1;
    await reader.read();
    expect(exec).toHaveBeenCalledTimes(4);
  });

  it("a repeated failed refresh keeps the real last-good time and rows", async () => {
    let now = 1_000_000;
    let failApt = false;
    let failHolds = false;
    const exec = vi.fn(async (file: string) => {
      if (file === "apt") {
        if (failApt) throw new Error("apt lock");
        return "Listing...\ncurl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\n";
      }
      if (file === "apt-mark") {
        if (failHolds) throw new Error("apt lock");
        return "curl\n";
      }
      if (file === "apt-get") return "Inst curl [8.4.0] (noble 8.5.0 amd64 [amd64])\n";
      if (file === "dpkg") return "amd64\n";
      throw new Error("unexpected");
    });
    const reader = createPackageObservationReader(exec, () => now);
    const good = await reader.read();
    now += PACKAGE_TTL_MS + 1;
    failApt = true;
    failHolds = true;
    const failed = await reader.read();
    expect(failed).toMatchObject({ state: "failed", stale: true, lastGoodAt: good.lastGoodAt });
    expect(failed.rows.map((r) => r.key)).toEqual(["curl:amd64"]);
    expect(failed.warnings).toContain("package-query-failed");
  });

  it("a failed child keeps its own last-good fact without hiding the candidates", async () => {
    let round = 0;
    const exec = vi.fn(async (file: string) => {
      if (file === "apt") return "Listing...\ncurl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\n";
      if (file === "dpkg") return "amd64\n";
      if (file === "apt-mark") {
        round += 1;
        if (round === 1) return "curl\n";
        throw new Error("hold query failed");
      }
      if (file === "apt-get") return "Inst curl [8.4.0] (noble 8.5.0 amd64 [amd64])\n";
      throw new Error("unexpected");
    });
    let now = 1_000_000;
    const reader = createPackageObservationReader(exec, () => now);
    await reader.read();
    now += PACKAGE_TTL_MS + 1;
    const second = await reader.read();
    expect(second.state).toBe("partial");
    expect(second.holds).toEqual([{ name: "curl", arch: null }]);
    expect(second.children.holds.ok).toBe(false);
    expect(second.children.holds.lastGoodAt).not.toBeNull();
    expect(second.warnings).toContain("hold-unavailable");
    expect(second.children.candidates.ok).toBe(true);
  });

  it("a first failed observation has no stale or last-good claim", async () => {
    const exec = vi.fn(async () => { throw new Error("apt missing"); });
    const reader = createPackageObservationReader(exec, () => 1_000_000);
    const first = await reader.read();
    expect(first).toMatchObject({ state: "failed", stale: false, lastGoodAt: null, rows: [] });
  });

  it("counts the deduped normalized rows, not the raw listing", async () => {
    const exec = vi.fn(async (file: string) => {
      if (file === "apt") {
        return "Listing...\nbar/stable 2 amd64 [upgradable from: 1]\nbar/stable 2 amd64 [upgradable from: 1]\n";
      }
      if (file === "apt-mark") return "";
      if (file === "apt-get") return "Reading package lists...\n0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.\n";
      if (file === "dpkg") return "amd64\n";
      throw new Error(`unexpected ${file}`);
    });
    const reader = createPackageObservationReader(exec, () => 1_000_000);
    const alerts = await getAlerts({
      exec: execFor(ALERT_EXECS), loadavg: loadavgOk, packages: () => reader.read(),
    });
    expect(alerts.find((a) => a.kind === "updates")).toMatchObject({ count: 1, rows: [{ key: "bar:amd64" }] });
  });

  it("keeps one pending run across repeated polls instead of starting duplicates", async () => {
    let resolveApt: ((value: string) => void) | undefined;
    const exec = vi.fn(async (file: string) => {
      if (file === "apt") {
        return new Promise<string>((resolve) => { resolveApt = resolve; });
      }
      if (file === "apt-mark") return "";
      if (file === "dpkg") return "amd64\n";
      if (file === "apt-get") return "0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.\n";
      throw new Error("unexpected");
    });
    const reader = createPackageObservationReader(exec, () => 1_000_000);
    const first = reader.read();
    const second = reader.read();
    expect(exec).toHaveBeenCalledTimes(4);
    resolveApt?.("Listing...\n");
    await Promise.all([first, second]);
    expect(exec).toHaveBeenCalledTimes(4);
  });
});

describe("mission-alert route (M5 Task 2)", () => {
  it("awaits the exported collector and preserves the public envelope", async () => {
    vi.mocked(getAlerts).mockResolvedValueOnce([{ kind: "source", source: "disk", state: "failed" }]);
    const response = await GET();
    expect(await response.json()).toEqual({
      ok: true,
      data: [{ kind: "source", source: "disk", state: "failed" }],
    });
    expect(getAlerts).toHaveBeenCalledTimes(1);
  });
});
