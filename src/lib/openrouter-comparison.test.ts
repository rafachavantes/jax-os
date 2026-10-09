import { describe, expect, it } from "vitest";
import type { Routing } from "@/lib/agent-settings";
import type { EndpointRow } from "@/server/collectors/openrouter";
import {
  canCompare,
  ceiling,
  combine,
  comparisonRequestMatches,
  comparisonUrl,
  evaluateEndpoint,
  singlePercentileKey,
  slugMatches,
} from "./openrouter-comparison";

function row(over: Partial<EndpointRow> = {}): EndpointRow {
  return {
    tag: "novita",
    provider: "Novita",
    prices: { prompt: 0.1, completion: 0.2 },
    latency: { p90: 2, window: "30m" },
    throughput: { p90: 100, window: "30m" },
    supported_parameters: null,
    partial: false,
    eligible: true,
    quantization: null,
    ...over,
  };
}

function routing(over: Partial<Routing> = {}): Routing {
  return { sort: "price", allow_fallbacks: true, ...over };
}

describe("slug matching", () => {
  it("matches an exact selector and rejects an arbitrary string prefix", () => {
    expect(slugMatches("novita", "novita")).toBe("yes");
    expect(slugMatches("novita", "novitax")).toBe("no");
    expect(slugMatches("nova", "novita")).toBe("no");
  });

  it("matches a base selector against a region suffix but not a service tier", () => {
    expect(slugMatches("novita", "novita/fp8")).toBe("yes");
    expect(slugMatches("novita", "novita/fast")).toBe("no");
    expect(slugMatches("novita", "novita/flex")).toBe("no");
    expect(slugMatches("novita/fast", "novita/fast")).toBe("yes");
    expect(slugMatches("novita", "novita/fast/fp8")).toBe("no");
  });

  it("returns unknown when the tag is missing and yes when unrestricted", () => {
    expect(slugMatches("novita", null)).toBe("unknown");
  });
});

describe("definite failure beats unknown", () => {
  it("combine picks no over unknown over yes", () => {
    expect(combine(["yes", "unknown", "no"])).toBe("no");
    expect(combine(["yes", "unknown"])).toBe("unknown");
    expect(combine(["yes", "yes"])).toBe("yes");
    expect(combine([])).toBe("yes");
  });

  it("a failing cap wins over an unknown price", () => {
    const judged = evaluateEndpoint(
      row({ prices: { prompt: 5, completion: null } }),
      routing({ max_price: { prompt: 1, completion: 1 } }),
    );
    expect(judged.eligible).toBe("no");
  });
});

describe("endpoint eligibility", () => {
  it("treats empty restrictions as eligible and missing required identity as unknown", () => {
    expect(evaluateEndpoint(row(), routing()).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ tag: null }), routing()).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ tag: null }), routing({ only: ["novita"] })).eligible).toBe("unknown");
  });

  it("honors exact, base and service-tier only selectors", () => {
    expect(evaluateEndpoint(row(), routing({ only: ["novita"] })).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ tag: "novita/fp8" }), routing({ only: ["novita"] })).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ tag: "novita/fast" }), routing({ only: ["novita"] })).eligible).toBe("no");
    expect(evaluateEndpoint(row({ tag: "novita/fast" }), routing({ only: ["novita/fast"] })).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ tag: "other" }), routing({ only: ["novita"] })).eligible).toBe("no");
  });

  it("ignore excludes a match and keeps the rest eligible", () => {
    expect(evaluateEndpoint(row(), routing({ ignore: ["novita"] })).eligible).toBe("no");
    expect(evaluateEndpoint(row({ tag: "deepinfra" }), routing({ ignore: ["novita"] })).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ tag: null }), routing({ ignore: ["novita"] })).eligible).toBe("unknown");
  });

  it("keeps a zero cap valid and a null price unknown", () => {
    expect(evaluateEndpoint(row({ prices: { prompt: 0, completion: 0.2 } }), routing({ max_price: { prompt: 0 } })).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ prices: { prompt: null, completion: 0.2 } }), routing({ max_price: { prompt: 0.5 } })).eligible).toBe("unknown");
    expect(evaluateEndpoint(row({ prices: { prompt: 1, completion: 0.2 } }), routing({ max_price: { prompt: 0.5 } })).eligible).toBe("no");
  });

  it("ceiling is the tri-state cap primitive", () => {
    expect(ceiling(0, 0)).toBe("yes");
    expect(ceiling(null, 0)).toBe("unknown");
    expect(ceiling(2, 1)).toBe("no");
    expect(ceiling(null, undefined)).toBe("yes");
  });

  it("ignores quantization when no filter is configured", () => {
    expect(evaluateEndpoint(row({ quantization: null }), routing()).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ quantization: "fp8" }), routing()).eligible).toBe("yes");
  });

  it("an allowed quantizations filter checks the endpoint's own quantization", () => {
    expect(evaluateEndpoint(row({ quantization: null }), routing({ quantizations: ["fp8"] })).eligible).toBe("unknown");
    expect(evaluateEndpoint(row({ quantization: "fp8" }), routing({ quantizations: ["fp8", "fp4"] })).eligible).toBe("yes");
    expect(evaluateEndpoint(row({ quantization: "fp16" }), routing({ quantizations: ["fp8", "fp4"] })).eligible).toBe("no");
  });
});

describe("goal evaluation", () => {
  it("reports none when no goal is configured", () => {
    expect(evaluateEndpoint(row(), routing()).goals).toBe("none");
  });

  it("a p50-only observation cannot prove a p90 goal", () => {
    const judged = evaluateEndpoint(
      row({ throughput: { p50: 200, window: "30m" } }),
      routing({ preferred_min_throughput: { p90: 100 } }),
    );
    expect(judged.goals).toBe("unknown");
  });

  it("checks every configured percentile of a goal and fails on a definite miss", () => {
    expect(evaluateEndpoint(row(), routing({ preferred_min_throughput: { p90: 300 } })).goals).toBe("no");
    expect(
      evaluateEndpoint(row(), routing({ preferred_min_throughput: { p50: 100, p90: 50 } })).goals,
    ).toBe("unknown");
  });

  it("combines throughput (>=) and latency (<=) goals", () => {
    expect(evaluateEndpoint(row(), routing({
      preferred_min_throughput: { p90: 100 },
      preferred_max_latency: { p90: 2 },
    })).goals).toBe("yes");
    expect(evaluateEndpoint(row(), routing({
      preferred_min_throughput: { p90: 200 },
      preferred_max_latency: { p90: 2 },
    })).goals).toBe("no");
    expect(evaluateEndpoint(row({ latency: null }), routing({
      preferred_min_throughput: { p90: 100 },
      preferred_max_latency: { p90: 2 },
    })).goals).toBe("unknown");
  });
});

describe("selection and request helpers", () => {
  const connection = {
    id: "or",
    label: "OpenRouter",
    adapter: "openrouter",
    base_url: null,
    auth: "api-key" as const,
    health: "registered" as const,
    credential: { kind: "native" as const },
    editable: true,
    reason: null,
    used_by: [],
    models: [
      { id: "m/1", label: "m", origin: "catalog" as const, efforts: [], no_effort: true, compatible: true, reason: null, context: null, effort_template: null },
    ],
  };

  it("only enables comparison for a valid OpenRouter model selection", () => {
    expect(canCompare(connection, "m/1")).toBe(true);
    expect(canCompare(connection, "m/2")).toBe(false);
    expect(canCompare({ ...connection, adapter: "xai" }, "m/1")).toBe(false);
    expect(canCompare(undefined, "m/1")).toBe(false);
  });

  it("builds the explicit request URL and matches request identity", () => {
    expect(comparisonUrl("or", "author/model", false)).toBe(
      "/api/opencode-providers/endpoints?connection=or&model=author%2Fmodel",
    );
    expect(comparisonUrl("or", "author/model", true)).toBe(
      "/api/opencode-providers/endpoints?connection=or&model=author%2Fmodel&refresh=1",
    );
    expect(comparisonRequestMatches({ connection: "or", model: "m" }, { connection: "or", model: "m" })).toBe(true);
    expect(comparisonRequestMatches({ connection: "or", model: "m" }, { connection: "or", model: "m2" })).toBe(false);
  });

  it("reports a single percentile only for a one-key goal", () => {
    expect(singlePercentileKey(undefined)).toBeNull();
    expect(singlePercentileKey({ p90: 1 })).toBe("p90");
    expect(singlePercentileKey({ p50: 1, p90: 2 })).toBeNull();
  });
});
