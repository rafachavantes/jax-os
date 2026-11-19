import { createElement, type ReactElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { NextIntlClientProvider } from "next-intl";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ZombieRow } from "@/server/collectors/alerts";
import type { HealthLatest } from "@/server/db/metrics";
import en from "../../../messages/en-US.json";
import pt from "../../../messages/pt-BR.json";

const harness = vi.hoisted(() => ({
  results: new Map<string, { data: unknown; isError: boolean }>(),
  calls: [] as { queryKey: unknown[]; queryFn: (ctx: { signal?: AbortSignal }) => unknown }[],
}));

vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: unknown[]; queryFn: (ctx: { signal?: AbortSignal }) => unknown }) => {
    harness.calls.push(opts);
    return harness.results.get(opts.queryKey.join(":")) ?? { data: undefined, isError: false };
  },
}));

import { groupZombies, meterPercent, meterValues, ServerAlerts, ServerPanel } from "./ServerPanel";

// The REAL catalogs, never a key-echo translator: a missing sibling surfaces as a next-intl
// MISSING_MESSAGE here instead of a literal key in production.
const catalogs = { "en-US": en, "pt-BR": pt } as const;
type Locale = keyof typeof catalogs;

function renderIntl(node: ReactElement, locale: Locale = "en-US") {
  const errors: unknown[] = [];
  const html = renderToStaticMarkup(createElement(NextIntlClientProvider, {
    locale, messages: catalogs[locale], timeZone: "UTC", onError: (error: unknown) => errors.push(error), children: node,
  }));
  return { html, errors };
}

const LATEST: HealthLatest = {
  ts: "2026-09-18T00:00:00.000Z", cpuPct: 42, memUsedMb: 4096, memTotalMb: 8192,
  swapUsedMb: 0, diskUsedGb: 40, diskTotalGb: 100, load1: 1, load5: 1.2, load15: 1, uptimeS: 100,
};

beforeEach(() => {
  harness.calls.length = 0;
  harness.results.clear();
});

const resources = () => ({
  data: { ok: true, data: {
    observedAt: "2026-09-18T00:00:00.000Z", lastGoodAt: "2026-09-18T00:00:00.000Z",
    state: "ok", stale: false, warning: null,
    tmp: { path: "/tmp", mountPath: "/tmp", fsType: "tmpfs", tmpfs: true, sharedRoot: false, usedGb: 1.5, totalGb: 4, availableGb: 2.5 },
  } },
  isError: false,
});

const health = (over: Record<string, unknown> = {}) => ({
  data: { ok: true, data: { latest: LATEST, cores: 8, series: [], spark: [], services: null, ...over } }, isError: false,
});

describe("meterValues / meterPercent / groupZombies (retained pure helpers)", () => {
  it("formats each meter independently, including a null disk side", () => {
    expect(meterValues(LATEST, 8)).toEqual({ cpu: "42%", mem: "4.0 GiB / 8.0 GiB", disk: "40 / 100 GB", load: "1.20 / 8" });
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
    harness.results.set("environment:resources", resources());
    const { html } = renderIntl(createElement(ServerPanel));
    expect(harness.calls.map((c) => c.queryKey.join(":"))).toEqual(["health:24h", "mission:alerts", "environment:resources"]);
    expect(html).toContain('href="/health#cpu"');
    expect(html).toContain('href="/health#mem"');
    expect(html).toContain('href="/health#disk"');
    expect(html).toContain('href="/health#tmp"');
    expect(html).toContain('href="/health#alerts"');
    // Compact values, not a full process/package panel.
    expect(html).toContain("42%");
    expect(html).not.toContain("kill -HUP");
    expect(html).not.toContain("sudo apt upgrade");
    expect(html).not.toContain("copyCaption");
  });

  it("shows a pending skeleton first and says unavailable only after a failed environment read", () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const pending = renderIntl(createElement(ServerPanel)).html;
    expect(pending).toContain("/tmp");
    expect(pending).toMatch(/animate-pulse/);
    harness.results.set("environment:resources", { data: undefined, isError: true });
    const failed = renderIntl(createElement(ServerPanel)).html;
    expect(failed).toContain("unavailable");
  });

  it("shows the independent measured /tmp capacity when the environment source answers", () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    harness.results.set("environment:resources", resources());
    const { html } = renderIntl(createElement(ServerPanel));
    expect(html).toContain("1.5 / 4.0 GB");
    expect(html).toContain('href="/health#tmp"');
  });

  it("shows retained temporary capacity as stale with its original last-good time", () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    harness.results.set("environment:resources", {
      data: { ok: true, data: {
        observedAt: "2026-09-19T00:00:00.000Z", lastGoodAt: "2026-09-18T00:00:00.000Z",
        state: "unavailable", stale: true, warning: "statfs-failed",
        tmp: { path: "/tmp", mountPath: "/tmp", fsType: "tmpfs", tmpfs: true, sharedRoot: false, usedGb: 1.5, totalGb: 4, availableGb: 2.5 },
      } },
      isError: false,
    });
    const { html, errors } = renderIntl(createElement(ServerPanel));
    expect(errors).toEqual([]);
    expect(html).toContain("1.5 / 4.0 GB");
    expect(html).toContain("stale, showing the last good observation");
  });

  it("renders a temporary-capacity skeleton on the first render, never an unavailable claim", () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const { html } = renderIntl(createElement(ServerPanel));
    expect(html).toContain("animate-pulse");
    expect(html).not.toContain("unavailable");
  });

  it("counts each count-less disk/load/source warning once and keeps row counts exact", () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", {
      data: { ok: true, data: [
        { kind: "zombies", count: 2, rows: [] },
        { kind: "updates", count: 3, rows: [] },
        { kind: "disk", mount: "/data", usedPercent: 90 },
        { kind: "source", source: "packages", state: "failed" },
      ] }, isError: false,
    });
    expect(renderIntl(createElement(ServerPanel)).html).toContain("7 alerts");
  });

  it("shows the alert count (summed from existing sources) and a source warning on failure", () => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", {
      data: { ok: true, data: [{ kind: "zombies", count: 2, rows: [] }, { kind: "updates", count: 3, rows: [] }] }, isError: false,
    });
    expect(renderIntl(createElement(ServerPanel)).html).toContain("5 alerts");
    harness.results.set("mission:alerts", { data: undefined, isError: true });
    expect(renderIntl(createElement(ServerPanel)).html).toContain("Server alerts unavailable");
  });

  it("keeps the CPU/memory/disk/tmp links while health is pending or unavailable", () => {
    const pending = renderIntl(createElement(ServerPanel)).html;
    for (const key of ["cpu", "mem", "disk", "tmp"]) {
      expect(pending).toContain(`href="/health#${key}"`);
    }
    harness.results.set("health:24h", { data: undefined, isError: true });
    const failed = renderIntl(createElement(ServerPanel)).html;
    expect(failed).toContain("Server metrics unavailable");
    for (const key of ["cpu", "mem", "disk", "tmp"]) {
      expect(failed).toContain(`href="/health#${key}"`);
    }
  });

  it("rejects through the real envelope when the source answers ok:false", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(
      JSON.stringify({ ok: false, error: "unavailable" }),
      { status: 200, headers: { "content-type": "application/json" } },
    ));
    try {
      renderIntl(createElement(ServerPanel));
      const healthCall = harness.calls.find((call) => call.queryKey.join(":") === "health:24h");
      expect(healthCall).toBeDefined();
      await expect(healthCall!.queryFn({})).rejects.toThrow("unavailable");
    } finally {
      fetchMock.mockRestore();
    }
  });

  it("first render is a skeleton, never fake values; a failed source is a visible warning", () => {
    const { html } = renderIntl(createElement(ServerPanel));
    expect(html).toContain("animate-pulse");
    expect(html).not.toContain("42%");
    harness.results.set("health:24h", { data: undefined, isError: true });
    expect(renderIntl(createElement(ServerPanel)).html).toContain("Server metrics unavailable");
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
    const { html } = renderIntl(createElement(ServerAlerts));
    expect(html).toContain('id="alerts"');
    expect(html).toContain("2 zombie processes");
    expect(html).toContain("2 children");
    expect(html).toContain("2 updates");
    expect(html).toContain("security");
    expect(html).not.toContain("kill -HUP");
    expect(html).not.toContain("sudo apt upgrade");
    expect(html).toContain("Read only");
  });

  it("resolves to quiet empty rows and shows the source warning without crashing", () => {
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const { html: empty } = renderIntl(createElement(ServerAlerts));
    expect(empty).toContain("No zombie processes.");
    expect(empty).toContain("No pending updates.");
    harness.results.set("mission:alerts", { data: undefined, isError: true });
    expect(renderIntl(createElement(ServerAlerts)).html).toContain("Server alerts unavailable");
  });

  it("a non-array payload (out-of-shape fixture) degrades to empty, never a crash", () => {
    harness.results.set("mission:alerts", { data: { ok: true, data: { latest: LATEST } }, isError: false });
    expect(() => renderIntl(createElement(ServerAlerts))).not.toThrow();
  });
});

describe("both actual catalogs resolve every server key (plan Task 5)", () => {
  const alerts = {
    data: { ok: true, data: [
      { kind: "zombies", count: 2, rows: [
        { pid: 1, ppid: 100, ageSeconds: 60, parentCommand: "node" },
        { pid: 2, ppid: 100, ageSeconds: 120, parentCommand: "node" },
      ] },
      { kind: "updates", count: 2, rows: [
        { name: "openssl", fromVersion: "3.0.13", toVersion: "3.0.15", security: true },
        { name: "vim", fromVersion: "1", toVersion: "2", security: false },
      ] },
    ] },
    isError: false,
  };

  it.each(["en-US", "pt-BR"] as const)("%s renders ServerPanel and ServerAlerts with no MISSING_MESSAGE or literal key", (locale) => {
    harness.results.set("health:24h", health());
    harness.results.set("mission:alerts", alerts);
    harness.results.set("environment:resources", resources());
    const panel = renderIntl(createElement(ServerPanel), locale);
    const detail = renderIntl(createElement(ServerAlerts), locale);
    expect(panel.errors).toEqual([]);
    expect(detail.errors).toEqual([]);
    for (const html of [panel.html, detail.html]) {
      expect(html).not.toMatch(/mission\.server|tempUnavailable|alertsCount|alertsTitle|zombieChildren|readOnlyNote/);
    }
    if (locale === "en-US") {
      expect(panel.html).toContain("/tmp");
      expect(panel.html).toContain("1.5 / 4.0 GB");
      expect(panel.html).toContain("4 alerts");
      expect(detail.html).toContain("Server alerts");
      expect(detail.html).toContain("2 children");
      expect(detail.html).toContain("Read only");
    } else {
      expect(panel.html).toContain("/tmp");
      expect(panel.html).toContain("1.5 / 4.0 GB");
      expect(panel.html).toContain("4 alertas");
      expect(detail.html).toContain("Alertas do servidor");
      expect(detail.html).toContain("2 filhos");
      expect(detail.html).toContain("Somente leitura");
    }
  });
});
