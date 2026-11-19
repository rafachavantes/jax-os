import { describe, expect, it, vi } from "vitest";
import {
  createPackageObservationReader, getAlerts, packageKey, parseDfAlerts, parseHoldRows, parseLoadAlert,
  parsePackageRows, parseSimulationRows, parseZombieRows, PACKAGE_TTL_MS,
} from "./alerts";

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

const execFor = (files: Record<string, string>) => vi.fn(async (file: string) => {
  const value = files[file];
  if (value === undefined) throw new Error(`unexpected ${file}`);
  return value;
});

const emptyPackages = async () => ({
  rows: [], state: "ok" as const, observedAt: "2026-11-19T00:00:00.000Z",
  lastGoodAt: "2026-11-19T00:00:00.000Z", stale: false, error: null,
});

describe("getAlerts — explicit source states (M5 Task 3)", () => {
  it("a timed-out probe becomes a source warning instead of throwing or vanishing", async () => {
    const exec = vi.fn(async (file: string) => {
      if (file === "df") throw Object.assign(new Error("timeout"), { code: "ETIMEDOUT" });
      if (file === "nproc") return "8\n";
      if (file === "ps") return "STAT\n";
      throw new Error("unexpected");
    });
    const alerts = await getAlerts({ exec, packages: emptyPackages });
    expect(alerts).toContainEqual({ kind: "source", source: "disk", state: "failed" });
  });

  it("an overflowed probe is the same explicit unavailable state, while siblings still return", async () => {
    const exec = vi.fn(async (file: string) => {
      if (file === "df") return DF;
      if (file === "nproc") return "8\n";
      if (file === "ps") throw Object.assign(new Error("overflow"), { code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER" });
      throw new Error("unexpected");
    });
    const alerts = await getAlerts({ exec, packages: emptyPackages });
    expect(alerts).toContainEqual({ kind: "source", source: "zombies", state: "failed" });
    expect(alerts).toContainEqual({ kind: "disk", mount: "/data", usedPercent: 90 });
  });

  it("a failed package observation preserves last-good rows and flags the source", async () => {
    const exec = execFor({ df: DF, nproc: "8\n", ps: "STAT\n" });
    const alerts = await getAlerts({
      exec,
      packages: async () => ({
        rows: [{ name: "openssl", arch: "amd64", key: "openssl:amd64", fromVersion: "1", toVersion: "2", security: true }],
        state: "failed", observedAt: "2026-11-19T00:00:00.000Z",
        lastGoodAt: "2026-11-18T23:55:00.000Z", stale: true, error: "package-query-failed",
      }),
    });
    expect(alerts).toContainEqual({ kind: "source", source: "packages", state: "failed" });
    expect(alerts.find((a) => a.kind === "updates")).toMatchObject({ count: 1 });
  });

  it("calls ps with the existing widened column set and keeps the update/alert count from one projection", async () => {
    const psOut = "  PID  PPID STAT ELAPSED COMMAND\n41822     1  Ssl    7260 node\n48211 41822  Z+     7260 node\n";
    const exec = execFor({ df: DF, nproc: "8\n", ps: psOut });
    const alerts = await getAlerts({ exec, packages: emptyPackages });
    expect(exec).toHaveBeenCalledWith("ps", ["-eo", "pid,ppid,stat,etimes,comm"], { timeout: 5_000, maxBuffer: 2 * 1024 * 1024 });
    expect(alerts.find((a) => a.kind === "zombies")).toEqual({
      kind: "zombies", count: 1,
      rows: [{ pid: 48211, ppid: 41822, ageSeconds: 7260, parentCommand: "node" }],
    });
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

describe("parsePackageRows — architecture identity (M5 F6)", () => {
  it("returns an empty array with no upgrades", () => {
    expect(parsePackageRows("Listing...\n")).toEqual([]);
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

  it("skips a malformed line missing the bracketed version pair, never crashes", () => {
    const out = "Listing...\ncurl/noble 8.5.0 amd64 [upgradable from\nvim/noble 2:9.1 amd64 [upgradable from: 2:9.0]\n";
    expect(parsePackageRows(out)).toEqual([
      { name: "vim", arch: "amd64", key: "vim:amd64", fromVersion: "2:9.0", toVersion: "2:9.1", security: false },
    ]);
  });

  it("keeps two architectures of the same name as distinct rows", () => {
    const out = "Listing...\nfoo/noble 2 amd64 [upgradable from: 1]\nfoo/noble 2 i386 [upgradable from: 1]\n";
    expect(parsePackageRows(out).map((r) => r.key)).toEqual(["foo:amd64", "foo:i386"]);
  });
});

describe("hold and simulation parsers (M5 Task 3)", () => {
  it("normalizes bare and name:arch hold rows, ignoring blank lines", () => {
    expect(parseHoldRows("openssl\ncurl\n\nlibc6:i386\nnot a package\n")).toEqual([
      { name: "openssl", arch: null },
      { name: "curl", arch: null },
      { name: "libc6", arch: "i386" },
    ]);
  });

  it("extracts Inst targets from a simulation, including architecture suffixes", () => {
    const out = "Reading package lists...\n" +
      "Inst openssl [3.0.13] (noble-security 3.0.15 amd64 [amd64])\n" +
      "Inst libc6:i386 [2.38] (noble 2.39 i386 [i386])\n" +
      "Conf openssl (3.0.15 noble-security [amd64])\n";
    expect(parseSimulationRows(out)).toEqual([
      { name: "openssl", arch: null },
      { name: "libc6", arch: "i386" },
    ]);
  });

  it("bare keys use the fixed native architecture, and ambiguity stays unknown", () => {
    expect(packageKey("openssl", null, "amd64")).toBe("openssl:amd64");
    expect(packageKey("openssl", "i386", "amd64")).toBe("openssl:i386");
    expect(packageKey("openssl", null, null)).toBe("openssl:unknown");
  });
});

describe("shared package observation (M5 F7)", () => {
  it("single-flights concurrent reads and caches the combined observation for 5 minutes", async () => {
    let now = 1_000_000;
    let resolveExec: ((value: string) => void) | undefined;
    const exec = vi.fn(() => new Promise<string>((resolve) => { resolveExec = resolve; }));
    const reader = createPackageObservationReader(exec, () => now);
    const first = reader.read();
    const second = reader.read();
    expect(exec).toHaveBeenCalledTimes(1);
    resolveExec!("Listing...\ncurl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\n");
    const [a, b] = await Promise.all([first, second]);
    expect(a).toMatchObject({ state: "ok", stale: false });
    expect(b.rows).toHaveLength(1);

    now += PACKAGE_TTL_MS - 1;
    await reader.read();
    expect(exec).toHaveBeenCalledTimes(1);

    now += 2;
    exec.mockResolvedValueOnce("Listing...\n");
    const refreshed = await reader.read();
    expect(exec).toHaveBeenCalledTimes(2);
    expect(refreshed.rows).toEqual([]);
  });

  it("a failed refresh retains the last-good rows and marks them stale", async () => {
    let now = 1_000_000;
    const exec = vi.fn()
      .mockResolvedValueOnce("Listing...\ncurl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\n")
      .mockRejectedValueOnce(new Error("apt lock"));
    const reader = createPackageObservationReader(exec, () => now);
    await reader.read();
    now += PACKAGE_TTL_MS + 1;
    const failed = await reader.read();
    expect(failed.state).toBe("failed");
    expect(failed.stale).toBe(true);
    expect(failed.lastGoodAt).not.toBeNull();
    expect(failed.rows.map((r) => r.key)).toEqual(["curl:amd64"]);
  });
});
