import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ZombieRow } from "@/server/collectors/alerts";
import type { HealthLatest } from "@/server/db/metrics";

// round-1 F8: no wrapper hook to mock here (unlike `useMission`/`useTokensApi`), so
// `useQuery` itself is mocked, keyed by `queryKey`.
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

import { copyCommand, groupZombies, meterPercent, meterValues, ServerPanel } from "./ServerPanel";

const LATEST: HealthLatest = {
  ts: "2026-09-18T00:00:00.000Z", cpuPct: 42, memUsedMb: 4096, memTotalMb: 8192,
  swapUsedMb: 0, diskUsedGb: 40, diskTotalGb: 100, load1: 1, load5: 1.2, load15: 1, uptimeS: 100,
};

beforeEach(() => {
  harness.calls.length = 0;
  harness.results.clear();
});

// round-1 F6: pure and mock-free, so the disk pair's independent null handling is
// tested directly instead of through a rendered string.
describe("meterValues (round-1 F6)", () => {
  it("formats all four from a full sample, each disk side independently per null combination, and a bare dash with no data", () => {
    expect(meterValues(LATEST, 8)).toEqual({ cpu: "42%", mem: "4.0 GB / 8.0 GB", disk: "40 / 100 GB", load: "1.20 / 8" });
    expect(meterValues({ ...LATEST, diskTotalGb: null }, 8).disk).toBe("40 / — GB");
    expect(meterValues({ ...LATEST, diskUsedGb: null }, 8).disk).toBe("— / 100 GB");
    expect(meterValues({ ...LATEST, diskUsedGb: null, diskTotalGb: null }, 8).disk).toBe("— / — GB");
    expect(meterValues(null, null)).toEqual({ cpu: "—", mem: "—", disk: "— / — GB", load: "—" });
  });
});

describe("groupZombies — one group per distinct parent (Decision 17)", () => {
  it("groups zombies sharing one parent into a single entry with nested rows", () => {
    const rows: ZombieRow[] = [
      { pid: 1, ppid: 100, ageSeconds: 60, parentCommand: "node" },
      { pid: 2, ppid: 100, ageSeconds: 120, parentCommand: "node" },
    ];
    expect(groupZombies(rows)).toEqual([{ ppid: 100, parentCommand: "node", zombies: rows }]);
  });

  it("keeps different parents in separate groups, never merged", () => {
    const rows: ZombieRow[] = [
      { pid: 1, ppid: 100, ageSeconds: 60, parentCommand: "node" },
      { pid: 2, ppid: 200, ageSeconds: 60, parentCommand: "python3" },
    ];
    expect(groupZombies(rows)).toHaveLength(2);
  });

  it("round-3 F3: a null parentCommand groups on ppid alone, never crashes", () => {
    const rows: ZombieRow[] = [{ pid: 1, ppid: 999, ageSeconds: 10, parentCommand: null }];
    expect(groupZombies(rows)).toEqual([{ ppid: 999, parentCommand: null, zombies: rows }]);
  });
});

describe("copyCommand — the one allowed injected-side-effect test (spec §11)", () => {
  it("writes the exact command string to the injected clipboard and never touches the network", () => {
    const writeText = vi.fn();
    const fetchSpy = vi.fn();
    const originalFetch = globalThis.fetch;
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    try {
      copyCommand("kill -HUP 41822", { writeText });
      expect(writeText).toHaveBeenCalledTimes(1);
      expect(writeText).toHaveBeenCalledWith("kill -HUP 41822");
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      globalThis.fetch = originalFetch;
    }
  });
});

describe("ServerPanel — endpoint wiring (round-1 F8)", () => {
  it("queries exactly /api/health and /api/mission/alerts", async () => {
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    renderToStaticMarkup(createElement(ServerPanel));
    expect(harness.calls.map((c) => c.queryKey.join(":"))).toEqual(["health:24h", "mission:alerts"]);
    const fetchSpy = vi.fn().mockResolvedValue({ ok: true, json: () => Promise.resolve({ ok: true, data: [] }) });
    const original = globalThis.fetch;
    globalThis.fetch = fetchSpy as unknown as typeof fetch;
    try {
      for (const call of harness.calls) await call.queryFn({ signal: undefined });
    } finally {
      globalThis.fetch = original;
    }
    expect(fetchSpy).toHaveBeenNthCalledWith(1, "/api/health", { signal: undefined });
    expect(fetchSpy).toHaveBeenNthCalledWith(2, "/api/mission/alerts", { signal: undefined });
  });

  it("round-2 F4: shows loading placeholders on first render, never the empty-state text, while both queries have no data yet", () => {
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).not.toContain("zombiesEmpty");
    expect(html).not.toContain("packagesEmpty");
    expect(html).not.toContain("metersUnavailable");
    expect(html).not.toContain("alertsUnavailable");
    // One placeholder for the meters grid, one each for the zombies/packages lists.
    expect((html.match(/animate-pulse/g) ?? []).length).toBe(3);
  });

  it("shows empty states with zero rows once both queries resolve", () => {
    harness.results.set("health:24h", { data: { ok: true, data: { latest: null, series: [], seriesMax: null, spark: [], services: null, collectedAgoS: 5, cores: 8 } }, isError: false });
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("zombiesEmpty");
    expect(html).toContain("packagesEmpty");
    expect(html).not.toContain("animate-pulse");
  });

  it("shows the warning (not a crash) when a source fails outright, with no prior data", () => {
    harness.results.set("health:24h", { data: undefined, isError: true });
    harness.results.set("mission:alerts", { data: undefined, isError: true });
    expect(() => renderToStaticMarkup(createElement(ServerPanel))).not.toThrow();
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("metersUnavailable");
    expect(html).toContain("alertsUnavailable");
  });

  it("round-2 F3: a polling failure after a good sample still shows the warning, with the last-good metrics still visible", () => {
    // `isError: true` with a populated `data` models a background refetch failure that
    // keeps the last-good payload — the old `isError && !data?.ok` swallowed this case.
    harness.results.set("health:24h", { data: { ok: true, data: { latest: LATEST, series: [], seriesMax: null, spark: [], services: null, collectedAgoS: 5, cores: 8 } }, isError: true });
    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("metersUnavailable");
    expect(html).toContain("42%");
  });
});

describe("ServerPanel — meter bars (spec §9, Decision 22)", () => {
  it("renders a filled bar width matching meterPercent for a resolved meter", () => {
    harness.results.set("health:24h", { data: { ok: true, data: { latest: LATEST, cores: 8 } }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel, {}));
    expect(html).toContain("width:42%");
  });
});

describe("ServerPanel — server-panel parity fixes (audit 2026-09-19)", () => {
  it("renders the host line only when hostname is present, formatted through the host key", () => {
    harness.results.set("health:24h", {
      data: { ok: true, data: { latest: LATEST, cores: 8, hostname: "jax-server" } },
      isError: false,
    });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain(">host<");
  });

  it("omits the host line when hostname is absent (e.g. a fixture predating the field)", () => {
    harness.results.set("health:24h", { data: { ok: true, data: { latest: LATEST, cores: 8 } }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    // The bare "host" substring never appears outside the host span itself, and "packagesHost"-like
    // false positives aren't a risk here since no other key contains "host".
    expect(html).not.toContain(">host<");
  });

  it("meters render as single-column rows (grid label·bar·value), not the old boxed tile grid", () => {
    harness.results.set("health:24h", { data: { ok: true, data: { latest: LATEST, cores: 8 } }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("grid-cols-[44px_1fr_auto]");
    expect(html).not.toContain("sm:grid-cols-4");
  });

  it("meter bar color follows the shared threshold: accent/warning/danger", () => {
    harness.results.set("health:24h", {
      data: { ok: true, data: { latest: { ...LATEST, cpuPct: 95, memUsedMb: 75, memTotalMb: 100, diskUsedGb: 5, diskTotalGb: 100 }, cores: 8 } },
      isError: false,
    });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("bg-danger"); // cpu 95%
    expect(html).toContain("bg-warning"); // mem 75%
    expect(html).toContain("bg-accent"); // disk 5%
  });

  it("zombies header shows the pluralized count when populated, the plain label when empty", () => {
    harness.results.set("mission:alerts", {
      data: { ok: true, data: [{ kind: "zombies", count: 2, rows: [
        { pid: 1, ppid: 100, ageSeconds: 60, parentCommand: "node" },
        { pid: 2, ppid: 200, ageSeconds: 60, parentCommand: "python3" },
      ] }] },
      isError: false,
    });
    const populated = renderToStaticMarkup(createElement(ServerPanel));
    expect(populated).toContain("zombiesCount");

    harness.results.set("mission:alerts", { data: { ok: true, data: [] }, isError: false });
    const empty = renderToStaticMarkup(createElement(ServerPanel));
    expect(empty).not.toContain("zombiesCount");
  });

  it("renders a code chip per zombie group's kill command, and exactly one copy caption for the whole block", () => {
    harness.results.set("mission:alerts", {
      data: { ok: true, data: [{ kind: "zombies", count: 2, rows: [
        { pid: 1, ppid: 100, ageSeconds: 60, parentCommand: "node" },
        { pid: 2, ppid: 200, ageSeconds: 60, parentCommand: "python3" },
      ] }] },
      isError: false,
    });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("kill -HUP 100");
    expect(html).toContain("kill -HUP 200");
    expect((html.match(/copyCaption/g) ?? []).length).toBe(1);
  });

  it("updates header shows count + security count, truncates rows past 3 with a 'more' line, and one copy chip for the block", () => {
    const rows = [
      { name: "openssl", fromVersion: "3.0.13", toVersion: "3.0.15", security: true },
      { name: "curl", fromVersion: "8.5.0", toVersion: "8.5.0-2", security: true },
      { name: "nodejs", fromVersion: "22.20", toVersion: "22.22", security: false },
      { name: "vim", fromVersion: "1", toVersion: "2", security: false },
      { name: "git", fromVersion: "1", toVersion: "2", security: false },
    ];
    harness.results.set("mission:alerts", { data: { ok: true, data: [{ kind: "updates", count: rows.length, rows }] }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toContain("packagesCount");
    expect(html).toContain("packagesSecurityCount");
    expect((html.match(/→/g) ?? []).length).toBe(3); // VISIBLE_PACKAGES
    expect(html).toContain("packagesMore");
    expect(html).toContain("sudo apt upgrade openssl curl");
    expect((html.match(/copyCaption/g) ?? []).length).toBe(1);
  });
});

describe("ServerPanel — value tooltip + truncated package rows (spec item 14, round 3)", () => {
  it("the meter value span carries a title attribute with the full value", () => {
    harness.results.set("health:24h", { data: { ok: true, data: { latest: LATEST, cores: 8 } }, isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    expect(html).toMatch(/<span[^>]*title="[^"]*4\.0[^"]*GB[^"]*"[^>]*>4\.0G<\/span>/);
  });
  it("a long package name truncates inside one flex-1 min-w-0 wrapper, the security badge stays flex-none", () => {
    const longName = "x".repeat(60);
    harness.results.set("mission:alerts", { data: { ok: true, data: [{ kind: "updates", count: 1, rows: [{ name: longName, fromVersion: "1", toVersion: "2", security: true }] }] } , isError: false });
    const html = renderToStaticMarkup(createElement(ServerPanel));
    const start = html.indexOf(longName);
    const openTag = html.lastIndexOf("<span", start);
    const spanTag = html.slice(openTag, html.indexOf(">", openTag) + 1);
    expect(spanTag).toContain("min-w-0");
    expect(spanTag).toContain("flex-1");
    expect(spanTag).toContain("truncate");
    // F8 (cold review 92a14d784fb3): the security badge itself must stay flex-none so it never
    // shrinks when the truncated name+version span next to it grows — assert its own opening
    // tag, not just that the word "security" appears somewhere on the page.
    const badgeStart = html.indexOf(">security<");
    const badgeOpenTag = html.lastIndexOf("<span", badgeStart);
    const badgeTag = html.slice(badgeOpenTag, html.indexOf(">", badgeOpenTag) + 1);
    expect(badgeTag).toContain("flex-none");
  });
});

describe("meterPercent (spec §9, meter percent formula, round-1 F5)", () => {
  it("used/total*100, clamped to [0,100], only when both are finite numbers and total > 0", () => {
    expect(meterPercent(42, 100)).toBe(42);
    expect(meterPercent(150, 100)).toBe(100); // clamp — also covers load's own min(100,…) cap
    expect(meterPercent(0, 100)).toBe(0);
  });
  it("null for missing data, a zero/negative total, or a non-finite value", () => {
    expect(meterPercent(null, 100)).toBeNull();
    expect(meterPercent(42, null)).toBeNull();
    expect(meterPercent(42, 0)).toBeNull();
    expect(meterPercent(NaN, 100)).toBeNull();
    expect(meterPercent(42, Infinity)).toBeNull();
  });
});
