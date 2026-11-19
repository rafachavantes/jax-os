import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  queries: new Map<string, { data?: unknown; isError: boolean; dataUpdatedAt?: number; error?: unknown }>(),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/health",
}));
vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: unknown[] }) => harness.queries.get(opts.queryKey.join(":")) ?? {
    data: undefined, isError: false, dataUpdatedAt: 0, error: undefined,
  },
}));

import HealthPage from "./page";

const payload = {
  latest: {
    ts: "2026-07-01T00:00:00.000Z", cpuPct: 1, memUsedMb: 4096, memTotalMb: 8192,
    diskUsedGb: 40, diskTotalGb: 100, load5: 0.1, uptimeS: 10, collectedAgoS: 1,
  },
  collectedAgoS: 1,
  cores: 8,
  spark: [],
  series: [],
  services: null,
};

const RESOURCES = {
  observedAt: "2026-11-19T00:00:00.000Z", lastGoodAt: "2026-11-19T00:00:00.000Z",
  state: "ok", stale: false, warning: null,
  tmp: { path: "/tmp", mountPath: "/tmp", fsType: "tmpfs", tmpfs: true, sharedRoot: false, usedGb: 1.5, totalGb: 4, availableGb: 2.5 },
};

function setQuery(key: string, value: { data?: unknown; isError: boolean; dataUpdatedAt?: number; error?: unknown }) {
  harness.queries.set(key, value);
}

function renderPage(): string {
  return renderToStaticMarkup(createElement(HealthPage));
}

beforeEach(() => {
  harness.queries.clear();
});

describe("HealthPage retained poll data and independent diagnostics (M5 Task 5)", () => {
  it("mounts diagnostics and healthy temporary capacity while the initial health query is still pending", () => {
    setQuery("environment:resources", { data: { ok: true, data: RESOURCES }, isError: false });
    const html = renderPage();
    expect(html).toContain("animate-pulse");
    expect(html).toContain("sections.components");
    expect(html).toContain("1.5 GB / 4.0 GB");
    expect(html).toContain('id="tmp"');
  });

  it("keeps the last payload under a warning and does not invent services", () => {
    setQuery("health:24h", { data: { ok: true, data: payload }, isError: true, dataUpdatedAt: 99, error: new Error("down") });
    const html = renderPage();
    expect(html).toContain("unavailable");
    expect(html).toContain("tiles.cpu");
  });

  it("shows a warning without fake metrics on initial failure but keeps every resource anchor", () => {
    setQuery("health:24h", { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down") });
    setQuery("environment:resources", { data: { ok: true, data: RESOURCES }, isError: false });
    const html = renderPage();
    expect(html).toContain("unavailable");
    for (const id of ["cpu", "mem", "disk", "tmp"]) expect(html).toContain(`id="${id}"`);
    expect(html).toContain("1.5 GB / 4.0 GB");
  });

  it("keeps independent Environment sections mounted when metrics fail or have no sample (AC3)", () => {
    setQuery("health:24h", { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down") });
    expect(renderPage()).toContain("sections.components");
    setQuery("health:24h", { data: { ok: true, data: { ...payload, latest: null } }, isError: false, dataUpdatedAt: 0 });
    const html = renderPage();
    expect(html).toContain("waitingFirstSample");
    expect(html).toContain("sections.components");
    for (const id of ["cpu", "mem", "disk", "tmp"]) expect(html).toContain(`id="${id}"`);
  });

  it("keeps every null metric field unavailable instead of inventing zero", () => {
    setQuery("health:24h", {
      data: { ok: true, data: { ...payload, latest: {
        ts: "2026-07-01T00:00:00.000Z", cpuPct: null, memUsedMb: null, memTotalMb: null,
        diskUsedGb: null, diskTotalGb: null, load5: null, uptimeS: 10, collectedAgoS: 1,
      } } },
      isError: false,
      dataUpdatedAt: 0,
    });
    const html = renderPage();
    expect(html).toContain("tiles.cpu");
    expect(html).toContain("—");
    expect(html).not.toContain("0%");
    expect(html).not.toContain("undefined");
    expect(html).not.toContain("NaN");
  });

  it("renders binary memory units and never undefined/zero-invented disk values", () => {
    setQuery("health:24h", {
      data: { ok: true, data: { ...payload, latest: { ...payload.latest, diskUsedGb: 40, diskTotalGb: null } } },
      isError: false,
      dataUpdatedAt: 0,
    });
    const html = renderPage();
    expect(html).toContain("4.0 GiB / 8.0 GiB");
    expect(html).not.toContain("4.0 GB / 8.0 GB");
    expect(html).toContain("—");
    expect(html).not.toContain("undefined");
    expect(html).not.toContain("(0%)");
  });
});
