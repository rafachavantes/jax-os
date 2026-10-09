import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  query: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown },
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
  useQuery: () => harness.query,
}));

import HealthPage from "./page";

const payload = {
  latest: { ts: "2026-07-01T00:00:00.000Z", cpuPct: 1, memUsedMb: 1, memTotalMb: 2, diskUsedGb: 1, diskTotalGb: 2, load5: 0.1, uptimeS: 10, collectedAgoS: 1 },
  collectedAgoS: 1,
  cores: 8,
  spark: [],
  series: [],
  services: null,
};

describe("HealthPage retained poll data", () => {
  it("keeps the last payload under a warning and does not invent services", () => {
    harness.query = { data: { ok: true, data: payload }, isError: true, dataUpdatedAt: 99, error: new Error("down") };
    const html = renderToStaticMarkup(createElement(HealthPage));
    expect(html).toContain("unavailable");
    expect(html).toContain("tiles.cpu");
  });

  it("shows warning without fake metrics on initial failure", () => {
    harness.query = { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down") };
    expect(renderToStaticMarkup(createElement(HealthPage))).toContain("unavailable");
  });
});
