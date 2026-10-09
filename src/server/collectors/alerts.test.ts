import { describe, expect, it, vi } from "vitest";
import { countUpgradable, countZombies, getAlerts, parseDfAlerts, parseLoadAlert, parsePackageRows, parseZombieRows } from "./alerts";

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

describe("countZombies", () => {
  it("counts stat lines starting with Z, skipping the header", () => {
    expect(countZombies("STAT\nSsl\nZ\nZs\nR+\n")).toBe(2);
    expect(countZombies("STAT\nSsl\n")).toBe(0);
  });
});

describe("countUpgradable", () => {
  it("counts package rows, ignores the Listing header", () => {
    expect(countUpgradable("Listing...\n")).toBe(0);
    expect(
      countUpgradable(
        "Listing...\ncurl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\nvim/noble 2:9.1 amd64 [upgradable from: 2:9.0]\n",
      ),
    ).toBe(2);
  });
});

const LOCAL_OPTS = { encoding: "utf8" as const, timeout: 5_000, maxBuffer: 1024 * 1024 };
const APT_OPTS = { encoding: "utf8" as const, timeout: 15_000, maxBuffer: 2 * 1024 * 1024 };

describe("getAlerts executor bounds", () => {
  it("treats a local-command timeout as source failure, not parsed alerts", () => {
    const exec = vi.fn(() => {
      throw Object.assign(new Error("timeout"), { code: "ETIMEDOUT", stdout: DF });
    });
    expect(() => getAlerts(exec)).toThrow();
    expect(exec).toHaveBeenCalledWith(
      "df",
      ["-P", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs"],
      LOCAL_OPTS,
    );
  });

  it("treats df overflow as source failure, not partial parsed success", () => {
    const exec = vi.fn(() => {
      throw Object.assign(new Error("overflow"), {
        code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
        stdout: DF,
      });
    });
    expect(() => getAlerts(exec)).toThrow();
  });

  it("uses the longer apt cap and treats apt overflow as source failure", () => {
    const exec = vi.fn((file: string) => {
      if (file === "df") return DF;
      if (file === "nproc") return "8\n";
      if (file === "ps") return "STAT\nSsl\n";
      throw Object.assign(new Error("overflow"), {
        code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
        stdout: "Listing...\ncurl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\n",
      });
    });
    expect(() => getAlerts(exec)).toThrow();
    expect(exec).toHaveBeenCalledWith("apt", ["list", "--upgradable"], APT_OPTS);
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

describe("parsePackageRows", () => {
  it("returns an empty array with no upgrades", () => {
    expect(parsePackageRows("Listing...\n")).toEqual([]);
  });

  it("separates security from non-security lines", () => {
    const out = "Listing...\n" +
      "openssl/noble-security 3.0.15 amd64 [upgradable from: 3.0.13]\n" +
      "curl/noble 8.5.0 amd64 [upgradable from: 8.4.0]\n";
    expect(parsePackageRows(out)).toEqual([
      { name: "openssl", fromVersion: "3.0.13", toVersion: "3.0.15", security: true },
      { name: "curl", fromVersion: "8.4.0", toVersion: "8.5.0", security: false },
    ]);
  });

  it("skips a malformed line missing the bracketed version pair, never crashes", () => {
    const out = "Listing...\ncurl/noble 8.5.0 amd64 [upgradable from\nvim/noble 2:9.1 amd64 [upgradable from: 2:9.0]\n";
    expect(parsePackageRows(out)).toEqual([
      { name: "vim", fromVersion: "2:9.0", toVersion: "2:9.1", security: false },
    ]);
  });
});

describe("getAlerts — success path (round-1 plan review F5)", () => {
  it("calls ps with the widened column set, returns row-shaped alerts, and caches the package rows", () => {
    const psOut = "  PID  PPID STAT ELAPSED COMMAND\n41822     1  Ssl    7260 node\n48211 41822  Z+     7260 node\n";
    const aptOut = "Listing...\nopenssl/noble-security 3.0.15 amd64 [upgradable from: 3.0.13]\n";
    const zombieAlert = {
      kind: "zombies" as const, count: 1,
      rows: [{ pid: 48211, ppid: 41822, ageSeconds: 7260, parentCommand: "node" }],
    };
    const updatesAlert = {
      kind: "updates" as const, count: 1,
      rows: [{ name: "openssl", fromVersion: "3.0.13", toVersion: "3.0.15", security: true }],
    };

    const firstExec = vi.fn((file: string) => {
      if (file === "df") return DF;
      if (file === "nproc") return "8\n";
      if (file === "ps") return psOut;
      return aptOut;
    });
    const firstAlerts = getAlerts(firstExec);
    expect(firstExec).toHaveBeenCalledWith("ps", ["-eo", "pid,ppid,stat,etimes,comm"], LOCAL_OPTS);
    expect(firstAlerts.find((a) => a.kind === "zombies")).toEqual(zombieAlert);
    expect(firstAlerts.find((a) => a.kind === "updates")).toEqual(updatesAlert);

    // Same TTL window: a second call must reuse the cached package rows, never
    // re-invoking apt.
    const secondExec = vi.fn((file: string) => {
      if (file === "df") return DF;
      if (file === "nproc") return "8\n";
      if (file === "ps") return psOut;
      throw new Error("apt must not be called again -- the cache is still within its TTL");
    });
    const secondAlerts = getAlerts(secondExec);
    expect(secondAlerts.find((a) => a.kind === "updates")).toEqual(updatesAlert);
  });
});
