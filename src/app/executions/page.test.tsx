import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string, values?: Record<string, unknown>) => values ? `${key}:${Object.values(values).join("/")}` : key }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: () => {} }),
  usePathname: () => "/executions",
}));
vi.mock("@tanstack/react-query", () => ({ useQuery: () => ({ data: undefined, isLoading: false, isError: false }) }));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: ({ epochMs }: { epochMs: number }) => `RT(${epochMs})` }));

import { ExecutionsBody } from "./page";
import type { ExecutionFilters, ExecutionRow } from "@/lib/executions";

const filters: ExecutionFilters = { project: null, kind: "all", state: "all", period: "30d", run: null, limit: 20 };

function row(): ExecutionRow {
  return {
    destination: "jax-os", runId: "0123456789ab", latestEventId: 2, startEventId: 1, finishEventId: 2,
    role: "builder", kind: "build", runtime: "codex", model: null, caller: null, profile: null,
    phase: null, target: null, firstDispatchedAt: null, attemptStartedAt: null, finishedAt: null,
    outcome: "success", terminal: true, contractStatus: "ok", durationMs: null, warnings: [],
  };
}

function body(overrides: Partial<Parameters<typeof ExecutionsBody>[0]> = {}) {
  return {
    filters, rows: [], inspections: {}, olderHref: null, loading: false, error: null,
    inspectionError: false, invalid: false, onFilter: () => {}, onRefresh: () => {}, onReset: () => {},
    ...overrides,
  };
}

describe("ExecutionsBody", () => {
  it("renders a loading skeleton before the first page", () => {
    const html = renderToStaticMarkup(createElement(ExecutionsBody, body({ loading: true })));
    expect(html).toContain("animate-pulse");
    expect(html).not.toContain("empty");
  });

  it("distinguishes an unavailable source from an empty history", () => {
    const failed = renderToStaticMarkup(createElement(ExecutionsBody, body({ error: "boom" })));
    expect(failed).toContain("unavailable");
    expect(failed).not.toContain("executions.empty");

    const empty = renderToStaticMarkup(createElement(ExecutionsBody, body()));
    expect(empty).toContain(">empty</p>");
  });

  it("keeps known rows visible while inspection fails", () => {
    const html = renderToStaticMarkup(createElement(ExecutionsBody, body({ rows: [row()], inspectionError: true })));
    expect(html).toContain("inspectionUnavailable");
    expect(html).toContain("jax-os");
    expect(html).toContain("outcome.success");
  });

  it("renders an Older link carrying the cursor and a Refresh action", () => {
    const html = renderToStaticMarkup(createElement(ExecutionsBody, body({
      rows: [row()], olderHref: "/executions?cursor=abc",
    })));
    expect(html).toContain('href="/executions?cursor=abc"');
    expect(html).toContain(">older</a>");
    expect(html).toContain(">refresh</button>");
  });

  it("surfaces an invalid URL instead of silently widening the scope", () => {
    const html = renderToStaticMarkup(createElement(ExecutionsBody, body({ invalid: true })));
    expect(html).toContain("invalidUrl");
    expect(html).not.toContain("executions.empty");
  });

  it("shows the exact run-context chip", () => {
    const html = renderToStaticMarkup(createElement(ExecutionsBody, body({
      filters: { ...filters, project: "jax-os", run: { project: "jax-os", runId: "0123456789ab" }, period: "all" },
    })));
    expect(html).toContain("runContext:jax-os/0123456789ab");
  });
});
