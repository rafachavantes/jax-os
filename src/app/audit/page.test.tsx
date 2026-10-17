import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { Envelope } from "@/lib/api";
import type { MutationPage, MutationRow } from "@/server/db/mutations";

const harness = vi.hoisted(() => ({
  query: {
    data: undefined as Envelope<MutationPage> | undefined,
    isError: false,
  },
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => new URLSearchParams(),
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/audit",
}));
vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => harness.query,
}));

import AuditPage from "./page";
import { RowPair, StatusCell } from "@/components/audit/AuditRows";

const baseRow = (over: Partial<MutationRow> = {}): MutationRow => ({
  id: 1,
  ts: "2026-07-06T22:23:22.226Z",
  kind: "file-edit",
  ok: true,
  error: null,
  payload: { outcome: "done", root: "repos", rel: "a.md" },
  ...over,
});

describe("AuditPage", () => {
  it("shows a warning on initial failure and empty on success with no rows", () => {
    harness.query = { data: undefined, isError: true };
    expect(renderToStaticMarkup(createElement(AuditPage))).toContain("unavailable");
    harness.query = {
      data: { ok: true, data: { rows: [], kinds: [], total: 0, nextCursor: null } },
      isError: false,
    };
    expect(renderToStaticMarkup(createElement(AuditPage))).toContain("empty");
  });

  it("renders first-page rows without an offset query", () => {
    harness.query = {
      data: {
        ok: true,
        data: { rows: [baseRow()], kinds: ["file-edit"], total: 1, nextCursor: "c1" },
      },
      isError: false,
    };
    const html = renderToStaticMarkup(createElement(AuditPage));
    expect(html).toContain("file-edit");
    expect(html).toContain("loadMore");
    expect(html).toContain("aria-expanded");
    expect(html).not.toContain("cursor-pointer");
  });
});

describe("StatusCell", () => {
  it("renders outcome and legacy status cases", () => {
    expect(renderToStaticMarkup(createElement(StatusCell, { row: baseRow({ payload: { outcome: "pending" } }) }))).toContain("statusUncertain");
    expect(renderToStaticMarkup(createElement(StatusCell, { row: baseRow({ payload: { outcome: "abandoned" } }) }))).toContain("statusUncertain");
    expect(renderToStaticMarkup(createElement(StatusCell, { row: baseRow({ payload: { outcome: "done" } }) }))).toContain("statusOk");
    expect(renderToStaticMarkup(createElement(StatusCell, { row: baseRow({ ok: false, payload: { outcome: "failed" } }) }))).toContain("statusFailed");
    expect(renderToStaticMarkup(createElement(StatusCell, { row: baseRow({ ok: null, payload: { session: "x" } }) }))).toContain("—");
    expect(renderToStaticMarkup(createElement(StatusCell, { row: baseRow({ ok: false, error: "disk", payload: {} }) }))).toContain("disk");
  });
});

describe("RowPair", () => {
  it("expands from a labeled button, not a clickable row", () => {
    const html = renderToStaticMarkup(createElement(RowPair, { row: baseRow(), open: false, onToggle: () => {} }));
    expect(html).toContain("aria-expanded=\"false\"");
    expect(html).toContain("expandRow");
    expect(html).not.toContain("cursor-pointer");
    const open = renderToStaticMarkup(createElement(RowPair, { row: baseRow(), open: true, onToggle: () => {} }));
    expect(open).toContain("aria-expanded=\"true\"");
    expect(open).toContain("repos");
  });
});
