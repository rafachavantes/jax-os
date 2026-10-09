import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { Routing } from "@/lib/agent-settings";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));
vi.mock("./ToolsDraftProvider", () => ({
  useToolsDraft: () => ({ connections: [] }),
}));

import { OpenRouterRouting, migrateStatistic, parseList, routingStatistic, toggleQuantization } from "./OpenRouterRouting";
import { QUANTIZATIONS } from "@/lib/agent-settings";

function routing(over: Partial<Routing> = {}): Routing {
  return { sort: "price", allow_fallbacks: true, ...over };
}

describe("routingStatistic", () => {
  it("uses the local selection when no goal exists", () => {
    const view = routingStatistic(null, "p50");
    expect(view.displayed).toBe("p50");
    expect(view.mixed).toBe(false);
    expect(view.commonKey).toBeNull();
  });

  it("shows the shared key for common single-value goals", () => {
    const view = routingStatistic(routing({
      preferred_min_throughput: { p90: 100 },
      preferred_max_latency: { p90: 2 },
    }), "p50");
    expect(view.displayed).toBe("p90");
    expect(view.commonKey).toBe("p90");
  });

  it("shows a single existing goal's key", () => {
    const view = routingStatistic(routing({ preferred_min_throughput: { p75: 100 } }), "p50");
    expect(view.displayed).toBe("p75");
  });

  it("shows the placeholder for different single-key goals without calling them mixed maps", () => {
    const view = routingStatistic(routing({
      preferred_min_throughput: { p50: 100 },
      preferred_max_latency: { p90: 2 },
    }), "p90");
    expect(view.displayed).toBe("");
    expect(view.mixed).toBe(false);
  });

  it("flags a multi-percentile map as mixed", () => {
    const view = routingStatistic(routing({ preferred_min_throughput: { p50: 100, p90: 50 } }), "p90");
    expect(view.mixed).toBe(true);
    expect(view.displayed).toBe("");
  });
});

describe("migrateStatistic", () => {
  it("never invents a goal or returns a patch when nothing exists", () => {
    expect(migrateStatistic(null, "p50")).toBeNull();
    expect(migrateStatistic(routing(), "p50")).toBeNull();
  });

  it("moves common single-value goals to the chosen key", () => {
    expect(migrateStatistic(routing({
      preferred_min_throughput: { p90: 100 },
      preferred_max_latency: { p90: 2 },
    }), "p50")).toEqual({
      preferred_min_throughput: { p50: 100 },
      preferred_max_latency: { p50: 2 },
    });
  });

  it("moves different single-key goals and a lone goal", () => {
    expect(migrateStatistic(routing({
      preferred_min_throughput: { p50: 100 },
      preferred_max_latency: { p90: 2 },
    }), "p75")).toEqual({
      preferred_min_throughput: { p75: 100 },
      preferred_max_latency: { p75: 2 },
    });
    expect(migrateStatistic(routing({ preferred_min_throughput: { p75: 100 } }), "p99")).toEqual({
      preferred_min_throughput: { p99: 100 },
    });
  });

  it("leaves a mixed map unchanged until an explicit numeric replacement", () => {
    expect(migrateStatistic(routing({ preferred_min_throughput: { p50: 100, p90: 50 } }), "p99")).toBeNull();
  });
});

describe("parseList", () => {
  it("trims and drops empty fragments but keeps the raw text locally", () => {
    expect(parseList("a, b,, c")).toEqual(["a", "b", "c"]);
    expect(parseList("a, ")).toEqual(["a"]);
    expect(parseList("")).toEqual([]);
  });
});

describe("toggleQuantization", () => {
  it("adds and removes a value, keeping the canonical wire order", () => {
    expect(toggleQuantization(undefined, "fp8", true)).toEqual(["fp8"]);
    expect(toggleQuantization(["fp8"], "fp4", true)).toEqual(["fp4", "fp8"]);
    expect(toggleQuantization(["fp4", "fp8"], "fp4", false)).toEqual(["fp8"]);
  });

  it("is a no-op when unchecking a value that was never set", () => {
    expect(toggleQuantization(["fp8"], "fp4", false)).toEqual(["fp8"]);
  });
});

describe("OpenRouterRouting", () => {
  function render(routingValue: Routing | null) {
    return renderToStaticMarkup(createElement(OpenRouterRouting, {
      connection: "or",
      model: "author/m",
      routing: routingValue,
      onChange: () => {},
    }));
  }

  it("offers a selectable statistic before any goal exists", () => {
    const html = render(null);
    expect(html).toContain("routing.statistic");
    expect(html).toContain('value="p90" selected');
  });

  it("shows the mixed explanation for a multi-percentile map", () => {
    const html = render(routing({ preferred_min_throughput: { p50: 100, p90: 50 } }));
    expect(html).toContain("routing.mixedNote");
  });

  function inputFor(html: string, value: string): string {
    return html.match(new RegExp(`<input[^>]*value="${value}"[^>]*>`))?.[0] ?? "";
  }

  it("renders all twelve quantization checkboxes unchecked with no filter", () => {
    const html = render(null);
    expect(html).toContain("routing.quantizations");
    expect(html).toContain("routing.quantizationsHint");
    for (const q of QUANTIZATIONS) {
      expect(inputFor(html, q)).not.toContain("checked");
    }
  });

  it("checks only the routing's own quantizations", () => {
    const html = render(routing({ quantizations: ["fp8", "fp4"] }));
    expect(inputFor(html, "fp4")).toContain("checked");
    expect(inputFor(html, "fp8")).toContain("checked");
    expect(inputFor(html, "fp16")).not.toContain("checked");
  });
});
