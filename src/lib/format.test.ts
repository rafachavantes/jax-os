import { describe, expect, it } from "vitest";
import type { HealthLatest } from "@/server/db/metrics";
import { fmtUptime, meterColorClass, meterValuesCompact } from "./format";

describe("meterColorClass", () => {
  it("accent below 70, warning at/above 70, danger at/above 90", () => {
    expect(meterColorClass(0)).toBe("bg-accent");
    expect(meterColorClass(69)).toBe("bg-accent");
    expect(meterColorClass(70)).toBe("bg-warning");
    expect(meterColorClass(89)).toBe("bg-warning");
    expect(meterColorClass(90)).toBe("bg-danger");
    expect(meterColorClass(100)).toBe("bg-danger");
  });
});

describe("fmtUptime", () => {
  it("formats days+hours once past a day, else hours+minutes", () => {
    expect(fmtUptime(100)).toBe("0h 1m");
    expect(fmtUptime(3661)).toBe("1h 1m");
    expect(fmtUptime(2 * 86400 + 3 * 3600)).toBe("2d 3h");
  });
  it("dashes on null/undefined", () => {
    expect(fmtUptime(null)).toBe("—");
    expect(fmtUptime(undefined)).toBe("—");
  });
});

describe("meterValuesCompact (spec item 14, round 3)", () => {
  const LATEST: HealthLatest = {
    ts: "2026-09-18T00:00:00.000Z", cpuPct: 42, memUsedMb: 4096, memTotalMb: 8192, swapUsedMb: 0,
    diskUsedGb: 40, diskTotalGb: 100, load1: 1, load5: 1.2, load15: 1, uptimeS: 100,
  };
  it("formats short-form: percent bare, mem in G, disk without / total, load as used/cores", () => {
    expect(meterValuesCompact(LATEST, 8)).toEqual({ cpu: "42%", mem: "4.0G", disk: "40%", load: "1.20/8" });
  });
  it("dashes every field on a null latest", () => {
    expect(meterValuesCompact(null, 8)).toEqual({ cpu: "—", mem: "—", disk: "—", load: "—" });
  });
});
