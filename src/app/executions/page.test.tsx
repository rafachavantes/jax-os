import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const rq = vi.hoisted(() => ({
  options: [] as { queryKey: unknown[]; queryFn: (ctx: { signal?: AbortSignal }) => Promise<unknown> }[],
  refetch: vi.fn(),
}));
const nav = vi.hoisted(() => ({ search: "", replace: vi.fn() }));
type FetchMock = {
  mockReset(): void;
  mockImplementation(fn: (url: string, signal?: AbortSignal) => Promise<unknown>): void;
};
const api = vi.hoisted(() => ({ fetchEnvelope: vi.fn() })) as unknown as { fetchEnvelope: FetchMock };

vi.mock("next-intl", () => ({ useTranslations: () => (key: string, values?: Record<string, unknown>) => values ? `${key}:${Object.values(values).join("/")}` : key }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(nav.search),
  useRouter: () => ({ replace: nav.replace }),
  usePathname: () => "/executions",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (options: { queryKey: unknown[]; queryFn: (ctx: { signal?: AbortSignal }) => Promise<unknown> }) => {
    rq.options.push(options);
    return { data: undefined, isLoading: false, isError: false, refetch: rq.refetch };
  },
}));
vi.mock("@/lib/api", () => ({ fetchEnvelope: api.fetchEnvelope }));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: ({ epochMs }: { epochMs: number }) => `RT(${epochMs})` }));

import { ExecutionsBody, ExecutionsPageInner } from "./page";
import type { ExecutionFilters, ExecutionRow } from "@/lib/executions";

const filters: ExecutionFilters = { project: null, kind: "all", state: "all", period: "30d", run: null, limit: 20 };

function row(overrides: Partial<ExecutionRow> = {}): ExecutionRow {
  return {
    destination: "jax-os", runId: "0123456789ab", latestEventId: 2, startEventId: 1, finishEventId: 2,
    role: "builder", kind: "build", runtime: "codex", model: null, caller: null, profile: null,
    phase: null, target: null, firstDispatchedAt: null, attemptStartedAt: null, finishedAt: null,
    outcome: "success", terminal: true, contractStatus: "ok", durationMs: null, warnings: [],
    ...overrides,
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

describe("ExecutionsPageInner explicit refresh", () => {
  it("refetches a fresh first-page query instead of relying on URL replacement", async () => {
    rq.options.length = 0;
    rq.refetch.mockReset();
    nav.search = "";
    nav.replace.mockReset();
    api.fetchEnvelope.mockReset();
    const urls: string[] = [];
    let call = 0;
    api.fetchEnvelope.mockImplementation(async (url: string) => {
      urls.push(url);
      call++;
      return {
        ok: true,
        data: { rows: [row({ runId: call === 1 ? "0123456789ab" : "ffffffffffff" })], nextCursor: null },
      };
    });

    const element = ExecutionsPageInner();
    const props = element.props as Parameters<typeof ExecutionsBody>[0];
    const pageQuery = rq.options.find((options) => options.queryKey[0] === "executions")!;
    const first = await pageQuery.queryFn({ signal: undefined });
    expect(first).toMatchObject({ data: { rows: [{ runId: "0123456789ab" }] } });

    props.onRefresh();
    expect(rq.refetch).toHaveBeenCalledTimes(2);
    expect(nav.replace).not.toHaveBeenCalled();

    const second = await pageQuery.queryFn({ signal: undefined });
    expect(second).toMatchObject({ data: { rows: [{ runId: "ffffffffffff" }] } });
    expect(urls).toHaveLength(2);
    expect(urls[1]).toBe(urls[0]);
    expect(urls[1]).not.toContain("cursor=");
  });

  it("clears an older cursor while it refetches on Refresh", () => {
    rq.options.length = 0;
    rq.refetch.mockReset();
    nav.search = "cursor=abc";
    nav.replace.mockReset();

    const element = ExecutionsPageInner();
    const props = element.props as Parameters<typeof ExecutionsBody>[0];
    props.onRefresh();
    expect(nav.replace).toHaveBeenCalledWith("/executions", { scroll: false });
    expect(rq.refetch).toHaveBeenCalledTimes(2);
    nav.search = "";
  });
});
