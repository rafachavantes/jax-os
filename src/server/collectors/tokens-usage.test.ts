import { describe, expect, it } from "vitest";
import { aggregateUsage, type SessionRow } from "./tokens-usage";

// 2026-07-08T12:00:00Z
const NOW = 1783512000;
const DAY = 86400;

const row = (over: Partial<SessionRow>): SessionRow => ({
  model: "gpt-5.5",
  started_at: NOW - 3600,
  input_tokens: 100,
  output_tokens: 50,
  cache_read_tokens: 0,
  cache_write_tokens: 0,
  reasoning_tokens: 0,
  estimated_cost_usd: null,
  ...over,
});

describe("aggregateUsage", () => {
  it("sums all five token columns, treating nulls as 0", () => {
    const agents = [{ name: "default", rows: [row({ input_tokens: 1, output_tokens: 2, cache_read_tokens: 3, cache_write_tokens: 4, reasoning_tokens: null })] }];
    expect(aggregateUsage(agents, 7, NOW).totalTokens).toBe(10);
  });

  it("drops rows older than the range", () => {
    const agents = [{ name: "default", rows: [row({}), row({ started_at: NOW - 8 * DAY })] }];
    const agg = aggregateUsage(agents, 7, NOW);
    expect(agg.totalTokens).toBe(150);
    expect(agg.byAgent[0].tokens).toBe(150);
  });

  it("builds a contiguous daily array of exactly `days` entries ending today (UTC)", () => {
    const agg = aggregateUsage([{ name: "default", rows: [row({})] }], 7, NOW);
    expect(agg.daily).toHaveLength(7);
    expect(agg.daily[6].date).toBe("2026-07-08");
    expect(agg.daily[0].date).toBe("2026-07-02");
    expect(agg.daily[6].tokens).toBe(150);
    expect(agg.daily[3].tokens).toBe(0);
  });

  it("groups by model desc with null model as 'unknown'", () => {
    const agents = [{
      name: "default",
      rows: [row({ model: "glm-5.2", input_tokens: 1000 }), row({ model: null }), row({})],
    }];
    const agg = aggregateUsage(agents, 7, NOW);
    expect(agg.byModel.map((m) => m.name)).toEqual(["glm-5.2", "gpt-5.5", "unknown"]);
  });

  it("keeps agents separate in byAgent and sums cost when present", () => {
    const agents = [
      { name: "default", rows: [row({ estimated_cost_usd: 0.5 })] },
      { name: "coder", rows: [row({ input_tokens: 999900, estimated_cost_usd: 1.25 })] },
    ];
    const agg = aggregateUsage(agents, 7, NOW);
    expect(agg.byAgent).toEqual([
      { name: "default", tokens: 150, cost: 0.5 },
      { name: "coder", tokens: 999950, cost: 1.25 },
    ]);
    expect(agg.totalCost).toBeCloseTo(1.75);
  });

  it("aligns the cutoff to the oldest calendar bucket, not a rolling window", () => {
    const beforeOldestBucket = Date.parse("2026-07-01T20:00:00Z") / 1000; // inside old rolling window
    const inOldestBucket = Date.parse("2026-07-02T01:00:00Z") / 1000;
    const agents = [{
      name: "default",
      rows: [row({ started_at: beforeOldestBucket }), row({ started_at: inOldestBucket, input_tokens: 10, output_tokens: 0 })],
    }];
    const agg = aggregateUsage(agents, 7, NOW);
    expect(agg.totalTokens).toBe(10);
    expect(agg.byModel).toEqual([{ name: "gpt-5.5", tokens: 10, cost: 0 }]);
    expect(agg.byAgent).toEqual([{ name: "default", tokens: 10, cost: 0 }]);
    expect(agg.daily[0]).toEqual({ date: "2026-07-02", tokens: 10 });
    expect(agg.daily.reduce((s, d) => s + d.tokens, 0)).toBe(agg.totalTokens);
  });

  it("zero-cost dataset reports totalCost 0 (tokens-first contract)", () => {
    const agg = aggregateUsage([{ name: "default", rows: [row({})] }], 7, NOW);
    expect(agg.totalCost).toBe(0);
  });
});
