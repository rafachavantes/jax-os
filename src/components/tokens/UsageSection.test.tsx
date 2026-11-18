import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { NativeUsageSnapshot } from "@/server/collectors/tokens-usage";

const harness = vi.hoisted(() => ({
  query: { data: undefined as unknown, isError: false, dataUpdatedAt: 0 },
  paths: [] as string[],
}));

vi.mock("next-intl", () => {
  const translate = (key: string) => key;
  translate.has = () => true;
  return { useTranslations: () => translate, useLocale: () => "en-US" };
});
vi.mock("@/lib/useTokensApi", () => ({
  useTokensApi: (_path: string) => {
    harness.paths.push(_path);
    return harness.query;
  },
}));

import { UsageSection, usagePath } from "./UsageSection";

const DAY = 86400;

function snapshot(over: Partial<NativeUsageSnapshot> = {}): NativeUsageSnapshot {
  return {
    period: 7,
    startDay: "2026-11-12",
    endDay: "2026-11-18",
    totalInput: 130,
    totalOutput: 38,
    totalTokens: 168,
    totalCost: null,
    costCurrency: null,
    costCoverage: { measured: 0, total: 2 },
    daily: [{ date: "2026-11-18", input: 130, output: 38, tokens: 168 }],
    byModel: [{ name: "__unknown__", allInput: 10, allOutput: 5, tokens: 15, cost: null, currency: null, items: 1 }],
    byAgent: [
      { name: "Claude Code", allInput: 10, allOutput: 5, tokens: 15, cost: null, currency: null, items: 1 },
      { name: "build", allInput: 120, allOutput: 33, tokens: 153, cost: null, currency: null, items: 1 },
    ],
    sessions: { lead: 2, child: 0 },
    sources: [
      { source: "claude", state: "available", lastRefresh: 1_794_913_200, progress: null, warnings: [] },
      { source: "codex", state: "empty", lastRefresh: 1_794_913_200, progress: null, warnings: [] },
      { source: "opencode", state: "available", lastRefresh: 1_794_913_200, progress: null, warnings: [] },
    ],
    filters: { agents: ["Claude Code", "build"], models: ["__unknown__"] },
    mixedProjectionSessions: 0,
    excludedV1Items: 0,
    history: { cutoffDay: "2026-08-20", indexing: false, includesCurrentPartialDay: true },
    warnings: [],
    ...over,
  };
}

function render(): string {
  return renderToStaticMarkup(createElement(UsageSection));
}

describe("UsageSection", () => {
  it("keeps healthy totals visible when one source is unavailable", () => {
    harness.query = {
      data: { ok: true, data: snapshot({ sources: [
        { source: "claude", state: "available", lastRefresh: 1_794_913_200, progress: null, warnings: [] },
        { source: "codex", state: "unavailable", lastRefresh: 1_794_000_000, progress: null, warnings: ["read-failed"] },
        { source: "opencode", state: "indexing", lastRefresh: null, progress: { filesDone: 4, filesTotal: 9 }, warnings: [] },
      ] }) },
      isError: false,
      dataUpdatedAt: 9,
    };
    const html = render();
    expect(html).toContain("168");
    expect(html).toContain("state.unavailable");
    expect(html).toContain("state.indexing");
    expect(html).toContain("warning.read-failed");
    expect(html).toContain("4/9");
  });

  it("distinguishes all-unavailable from a healthy empty period", () => {
    harness.query = {
      data: { ok: true, data: snapshot({
        totalInput: 0, totalOutput: 0, totalTokens: 0, daily: [], byModel: [], byAgent: [], sessions: { lead: 0, child: 0 },
        sources: [
          { source: "claude", state: "unavailable", lastRefresh: null, progress: null, warnings: [] },
          { source: "codex", state: "incompatible", lastRefresh: null, progress: null, warnings: [] },
          { source: "opencode", state: "disabled", lastRefresh: null, progress: null, warnings: [] },
        ],
      }) },
      isError: false,
      dataUpdatedAt: 0,
    };
    const unavailableHtml = render();
    expect(unavailableHtml).toContain("allUnavailable");
    expect(unavailableHtml).not.toContain("dailyBurn");

    harness.query = { data: { ok: true, data: snapshot() }, isError: false, dataUpdatedAt: 0 };
    const healthyHtml = render();
    expect(healthyHtml).toContain("dailyBurn");
    expect(healthyHtml).toContain("historyNote");
  });

  it("shows disabled and indexing as distinct states with view-open progress", () => {
    harness.query = {
      data: { ok: true, data: snapshot({
        sources: [
          { source: "claude", state: "disabled", lastRefresh: null, progress: null, warnings: [] },
          { source: "codex", state: "indexing", lastRefresh: null, progress: { filesDone: 1, filesTotal: 3 }, warnings: [] },
          { source: "opencode", state: "incompatible", lastRefresh: null, progress: null, warnings: ["executable-mismatch"] },
        ],
      }) },
      isError: false,
      dataUpdatedAt: 0,
    };
    const html = render();
    expect(html).toContain("state.disabled");
    expect(html).toContain("state.indexing");
    expect(html).toContain("state.incompatible");
    expect(html).toContain("indexing");
    expect(html).toContain("1/3");
  });

  it("never renders an unknown cost as zero spend", () => {
    harness.query = { data: { ok: true, data: snapshot() }, isError: false, dataUpdatedAt: 0 };
    const html = render();
    expect(html).toContain("—");
    expect(html).toContain("costUnavailable");
    expect(html).not.toContain("$0.00");
  });

  it("shows the error warning with the last successful timestamp", () => {
    harness.query = { data: { ok: true, data: snapshot() }, isError: true, dataUpdatedAt: 1_794_913_200_000 };
    expect(render()).toContain("unavailable");
  });

  it("builds one scoped query key from every filter", () => {
    expect(usagePath("7d", "", "")).toBe("usage?range=7d");
    expect(usagePath("30d", "Claude Code", "")).toBe("usage?range=30d&agent=Claude+Code");
    expect(usagePath("90d", "", "__unknown__")).toBe("usage?range=90d&model=__unknown__");
  });

  it("renders only whitelisted fields and never a sentinel payload", () => {
    const raw = { ...snapshot(), rawTranscript: "SENTINEL_SECRET_TEXT" } as NativeUsageSnapshot;
    harness.query = { data: { ok: true, data: raw }, isError: false, dataUpdatedAt: 0 };
    const html = render();
    expect(html).not.toContain("SENTINEL_SECRET_TEXT");
    expect(html).toContain("Claude Code");
  });

  it("explains UTC partial days, the retained window and mixed projections", () => {
    harness.query = {
      data: { ok: true, data: snapshot({ mixedProjectionSessions: 2, costCoverage: { measured: 1, total: 2 } }) },
      isError: false,
      dataUpdatedAt: 0,
    };
    const html = render();
    expect(html).toContain("utcNote");
    expect(html).toContain("mixedProjection");
    expect(html).toContain("cacheNote");
    void DAY;
  });
});
