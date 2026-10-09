import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { Routing } from "@/lib/agent-settings";
import type { EndpointComparison } from "@/server/collectors/openrouter";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
vi.mock("@/components/RelativeTime", () => ({
  RelativeTime: () => createElement("span", null, "relative"),
}));

import { EndpointComparisonDialog, EndpointComparisonTable, comparisonBodyMode } from "./EndpointComparisonDialog";

const data: EndpointComparison = {
  connection: "or",
  model: "author/m/1",
  fetched_at: "2026-09-14T00:00:00.000Z",
  stale: false,
  error: null,
  endpoints: [
    {
      tag: "novita",
      provider: "Novita",
      prices: { prompt: 0.14, completion: 0.28 },
      latency: { p90: 2, window: "30m" },
      throughput: { p90: 100, window: "30m" },
      supported_parameters: null,
      partial: false,
      eligible: true,
      quantization: "fp8",
    },
    {
      tag: null,
      provider: null,
      prices: { prompt: null, completion: null },
      latency: null,
      throughput: null,
      supported_parameters: null,
      partial: true,
      eligible: false,
      quantization: null,
    },
  ],
};

const routing: Routing = {
  sort: "price",
  allow_fallbacks: true,
  only: ["novita"],
  max_price: { prompt: 1 },
  preferred_min_throughput: { p90: 50 },
  preferred_max_latency: { p90: 3 },
};

describe("EndpointComparisonTable", () => {
  it("renders the seven approved columns and real row values", () => {
    const html = renderToStaticMarkup(createElement(EndpointComparisonTable, { data, routing }));
    for (const key of ["identity", "input", "output", "latency", "throughput", "eligibility", "goals"]) {
      expect(html).toContain(`comparison.columns.${key}`);
    }
    expect(html).toContain("Novita");
    expect(html).toContain("novita");
    expect(html).toContain("$0.14");
    expect(html).toContain("$0.28");
    expect(html).toContain("2.00s");
    expect(html).toContain("100 tok/s");
    expect(html).toContain("comparison.yes");
    expect(html).toContain("comparison.unknown");
    expect(html).toContain("fp8");
  });

  it("shows an em dash for a missing quantization", () => {
    const html = renderToStaticMarkup(createElement(EndpointComparisonTable, { data, routing }));
    const rows = html.split("<tr>");
    const secondBodyRow = rows[rows.length - 1];
    expect(secondBodyRow).not.toContain("fp8");
    expect(secondBodyRow).toContain("—");
  });

  it("shows em dashes for unknown prices and statistics, never zeros", () => {
    const html = renderToStaticMarkup(createElement(EndpointComparisonTable, { data, routing }));
    expect(html).toContain("—");
    expect(html).not.toContain("$0.00");
    expect(html).not.toContain(">0 tok/s");
  });

  it("shows the successful-empty state distinctly", () => {
    const html = renderToStaticMarkup(createElement(EndpointComparisonTable, { data: { ...data, endpoints: [] }, routing }));
    expect(html).toContain("comparison.empty");
  });

  it("evaluates rows against the current draft policy rather than a saved snapshot", () => {
    const blocked = renderToStaticMarkup(createElement(EndpointComparisonTable, {
      data,
      routing: { sort: "price", allow_fallbacks: true, only: ["other"] },
    }));
    expect(blocked).toContain("comparison.no");
  });

  it("shows the evaluated statistic and measurement window alongside values", () => {
    const html = renderToStaticMarkup(createElement(EndpointComparisonTable, { data, routing }));
    expect(html).toContain("p90 · 30m");
    expect(html).toContain("2.00s");
    expect(html).toContain("100 tok/s");
  });

  it("rounds latency to two decimals and throughput to whole tokens", () => {
    const noisy: EndpointComparison = {
      ...data,
      endpoints: [{ ...data.endpoints[0], latency: { p90: 7.392100000000005, window: "30m" }, throughput: { p90: 95.09000000000015, window: "30m" } }],
    };
    const html = renderToStaticMarkup(createElement(EndpointComparisonTable, { data: noisy, routing }));
    expect(html).toContain("7.39s");
    expect(html).toContain("95 tok/s");
    expect(html).not.toContain("00000");
  });

  it("keeps an absent selected statistic unknown while naming the window", () => {
    const partialData: EndpointComparison = {
      ...data,
      endpoints: [{ ...data.endpoints[0], latency: { p50: 1, window: "30m" }, throughput: null }],
    };
    const html = renderToStaticMarkup(createElement(EndpointComparisonTable, { data: partialData, routing }));
    expect(html).toContain("p90 · 30m");
    expect(html).not.toContain(">1s<");
    expect(html).not.toContain("100 tok/s");
  });
});

describe("comparisonBodyMode", () => {
  it("separates a first-load error from a successful-empty response", () => {
    expect(comparisonBodyMode({ status: "error", data: null })).toBe("error");
    expect(comparisonBodyMode({ status: "loading", data: null })).toBe("loading");
    expect(comparisonBodyMode({ status: "ready", data: { ...data, endpoints: [] } })).toBe("table");
  });

  it("keeps last-good rows visible through a refresh failure", () => {
    expect(comparisonBodyMode({ status: "error", data })).toBe("table");
    expect(comparisonBodyMode({ status: "idle", data })).toBe("table");
  });
});

describe("EndpointComparisonDialog", () => {
  it("renders a native dialog with explicit refresh and close controls", () => {
    const html = renderToStaticMarkup(createElement(EndpointComparisonDialog, {
      connection: "or",
      model: "author/m/1",
      routing,
    }));
    expect(html).toContain("<dialog");
    expect(html).toContain("comparison.refresh");
    expect(html).toContain("comparison.close");
    expect(html).toMatch(/<button[^>]*class="[^"]*dialogClose[^"]*"[^>]*>comparison\.close<\/button>/);
    expect(html.lastIndexOf("comparison.refresh")).toBeGreaterThan(html.indexOf("comparison.softGoal"));
    expect(html).toContain("dialogFooter");
  });
});
