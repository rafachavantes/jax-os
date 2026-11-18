import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ZombieRow } from "@/server/collectors/alerts";
import type { HealthLatest } from "@/server/db/metrics";

const harness = vi.hoisted(() => ({
  results: new Map<string, { data: unknown; isError: boolean }>(),
  calls: [] as { queryKey: unknown[]; queryFn: (ctx: { signal?: AbortSignal }) => unknown }[],
}));

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: unknown[]; queryFn: (ctx: { signal?: AbortSignal }) => unknown }) => {
    harness.calls.push(opts);
    return harness.results.get(opts.queryKey.join(":")) ?? { data: undefined, isError: false };
  },
}));

import { groupZombies, meterPercent, meterValues, ServerAlerts, ServerPanel } from "./ServerPanel";

const LATEST: HealthLatest = {
  ts: "2026-09-18T00:00:00.000Z", cpuPct: 42, memUsedMb: 4096, memTotalMb: 8192,
  swapUsedMb: 0, diskUsedGb: 40, diskTotalGb: 100, load1: 1, load5: 1.2, load15: 1, uptimeS: 100,
};

beforeEach(() => {
  harness.calls.length = 0;
  harness.results.clear();
});

const health = (over: Record<string, unknown> = {}) => ({
  data: { ok: true, data: { latest: LATEST, cores: 8, series: [], spark: [], services: null, ...over } }, isError: false,
});

describe("meterValues / meterPercent / groupZombies (retained pure helpers)", () => {
  it("formats each meter independently, including a null disk side", () => {
    expect(meterValues(LATEST, 8)).toEqual({ cpu: "42%", mem: "4.0 GB / 8.0 GB", disk: "40 / 100 GB", load: "1.20 / 8" });
    expect(meterValues({ ...LATEST, diskTotalGb: null }, 8).disk).toBe("40 / — GB");
    expect(meterValues(null, null)).toEqual({ cpu: "—", mem: "—", disk: "— / — GB", load: "—" });
  });

  it("clamps percent only when both sides are finite and total > 0", () => {
    expect(meterPercent(42, 100)).toBe(42);
    expect(meterPercent(150, 100)).toBe(100);
    expect(meterPercent(42, 0)).toBeNull();
    expect(meterPercent(NaN, 100)).toBeNull();
  });

  it("groups zombies by parent and keeps a null parent command", () => {
    const rows: ZombieRow[] = [
      { pid: 1, ppid: 100, ageSeconds: 60, parentCommand: "node" },
      { pid: 2, ppid: 100, ageSeconds: 120, parentCommand: "node" },
      { pid: 3, ppid: 999, ageSeconds: 10, parentCommand: null },
    ];
    expect(groupZombies(rows)).toEqual([
      { ppid: 100, parentCommand: "node", zombies: rows.slice(0, 2) },
      { ppid: 999, parentCommand: null, zombies: rows.slice(2) },
    ]);
  });
});

describe("ServerPanel — compact resource strip (M2 plan Task 5)", () => {
  it("polls the existing health + alerts endpoints and renders glanceable links to stable health anchors", async () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(harness.calls.map((c) => c.queryKey.join(":"))).toEqual(["health:24h", "mission:alerts"]);
    expect(html).toContain('href="/health#cpu"');
    expect(html).toContain('href="/health#mem"');
    expect(html).toContain('href="/health#disk"');
    expect(html).toContain('href="/health#resources"');
    expect(html).toContain('href="/health#alerts"');
    // Compact values, not a full process/package panel.
    expect(html).toContain("42%");
    expect(html).not.toContain("kill -HUP");
    expect(html).not.toContain("sudo apt upgrade");
    expect(html).not.toContain("copyCaption");
  });

  it("says temporary capacity is unavailable rather than substituting disk", () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("tempUnavailable");
  });

  it("shows the alert count (summed from existing sources) and a source warning on failure", () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", {
      data: { ok: true, data: [{ kind: "zombies", count: 2, rows: [] }, { kind: "updates", count: 3, rows: [] }] }, isError: false,
    });
    expect(renderToStaticMarkup(createElement(ServerPanel))).toContain("alertsCount");
    harness.results.set("mission:alerts", { data: undefined, isError: true });
    expect(renderToStaticMarkup(createElement(ServerPanel))).toContain("alertsUnavailable");
  });

  it("first render is a skeleton, never fake values; a failed source is a visible warning", () => {
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("animate-pulse");
    expect(html).not.toContain("42%");
    harness.results.set("health:24h", { data: undefined, isError: true });
    expect(renderToStaticMarkup(createElement(ServerPanel))).toContain("metersUnavailable");
  });
});

describe("ServerAlerts — read-only detail reused on Health (plan F4)", () => {
  it("renders zombie parents/counts and package facts with an anchor, but no kill/upgrade commands", () => {
    harness.results.set("mission:alerts", {
      data: { ok: true, data: [
        { kind: "zombies", count: 2, rows: [
          { pid: 1, ppid: 100, ageSeconds: 60, parentCommand: "node" },
          { pid: 2, ppid: 100, ageSeconds: 120, parentCommand: "node" },
        ] },
        { kind: "updates", count: 2, rows: [
          { name: "openssl", fromVersion: "3.0.13", toVersion: "3.0.15", security: true },
          { name: "vim", fromVersion: "1", toVersion: "2", security: false },
        ] },
      ] }, isError: false,
    });
    const html = renderToStaticMarkup(createElement(ServerAlerts));
    expect(html).toContain('id="alerts"');
    expect(html).toContain("zombiesCount");
    expect(html).toContain("packagesCount");
    expect(html).toContain("security");
    expect(html).not.toContain("kill -HUP");
    expect(html).not.toContain("sudo apt upgrade");
    expect(html).toContain("readOnlyNote");
  });

  it("resolves to quiet empty rows and shows the source warning without crashing", () => {
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const empty = renderToStaticMarkup(createElement(ServerAlerts));
    expect(empty).toContain("zombiesEmpty");
    expect(empty).toContain("packagesEmpty");
    harness.results.set("mission:alerts", { data: undefined, isError: true });
    expect(renderToStaticMarkup(createElement(ServerAlerts))).toContain("alertsUnavailable");
  });

  it("a non-array payload (out-of-shape fixture) degrades to empty, never a crash", () => {
    harness.results.set("mission:alerts", { data: { ok: true, data: { latest: LATEST } }, isError: false });
    expect(() => renderToStaticMarkup(createElement(ServerAlerts))).not.toThrow();
  });
});
