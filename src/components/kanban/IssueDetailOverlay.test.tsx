import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { IssueDetail } from "@/server/collectors/linear";

const harness = vi.hoisted(() => {
  const detail: { data: unknown; isLoading: boolean; isError: boolean; dataUpdatedAt?: number } = {
    data: undefined,
    isLoading: false,
    isError: false,
  };
  return { detail, queryKeys: [] as string[][] };
});

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: string[] }) => {
    harness.queryKeys.push(opts.queryKey);
    if (opts.queryKey.includes("issue")) return harness.detail;
    if (opts.queryKey.includes("labels")) return { data: { ok: true, data: detail.issue.labels } };
    if (opts.queryKey.includes("members")) return { data: { ok: true, data: [{ id: "u9", displayName: "Ana" }] } };
    return { data: { ok: true, data: [] } };
  },
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

import { IssueDetailOverlay, applyCommentOptimistic, reconcileCommentCache, shouldConfirmDiscard } from "./IssueDetailOverlay";

// fully populated off-board issue: different team, normally excluded current
// state (canceled), its own assignee/labels/states
const detail: IssueDetail = {
  description: "desc",
  subIssue: { total: 0, done: 0 },
  children: [],
  comments: [{ id: "c1", author: "Ana", body: "hi", createdAt: "2026-01-01T00:00:00Z" }],
  attachments: [],
  parent: undefined,
  commentsTruncated: false,
  childrenTruncated: false,
  commentsIncomplete: false,
  childrenIncomplete: false,
  issue: {
    id: "i9",
    identifier: "MKT-9",
    title: "Off-board cross-team issue",
    priority: 3,
    url: "https://linear.app/x/issue/MKT-9",
    stateId: "s-cancel",
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
    project: "Marketing",
    assignee: "Ana",
    assigneeId: "u9",
    labels: [{ id: "l9", name: "design", color: "#8a2be2" }],
  },
  teamId: "t9",
  states: [
    { id: "s-triage", name: "Triage", color: "#f2c94c", position: 0, type: "triage" },
    { id: "s-prog9", name: "In Progress", color: "#f2c94c", position: 2, type: "started" },
    { id: "s-cancel", name: "Canceled", color: "#95a2b3", position: 4, type: "canceled" },
  ],
};

function setup() {
  harness.queryKeys.length = 0;
}

describe("IssueDetailOverlay", () => {
  it("renders a native dialog whose header, states and controls come from detail data, not the board", () => {
    setup();
    harness.detail = { data: { ok: true, data: detail }, isLoading: false, isError: false };
    const html = renderToStaticMarkup(
      createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }),
    );
    expect(html).toContain("<dialog");
    expect(html).toContain('aria-labelledby="issue-detail-title"');
    expect(html).toContain('id="issue-detail-title"');
    // header identity and title from the detail issue, not a board row
    expect(html).toContain("MKT-9");
    expect(html).toContain("Off-board cross-team issue");
    expect(html).toContain('href="https://linear.app/x/issue/MKT-9"');
    // status choices include the normally excluded canceled state
    expect(html).toContain('value="s-cancel"');
    expect(html).toContain("Canceled");
    expect(html).toContain("Triage");
    // property values and label set from the detail issue
    expect(html).toContain('value="3"'); // priority
    expect(html).toContain("Ana");
    expect(html).toContain("design");
    expect(html).toContain("hi");
    expect(html).toContain('aria-label="commentPlaceholder"');
    // F3: the description edit trigger is a focusable, keyboard-reachable button
    expect(html).toContain('aria-label="descriptionEdit"');
    expect(html).toMatch(/<button[^>]*aria-label="descriptionEdit"/);
  });

  it("keys members and labels queries to the detail team, not the board team", () => {
    setup();
    harness.detail = { data: { ok: true, data: detail }, isLoading: false, isError: false };
    renderToStaticMarkup(createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }));
    expect(harness.queryKeys).toContainEqual(["kanban", "issue", "i9"]);
    expect(harness.queryKeys).toContainEqual(["kanban", "members", "t9"]);
    expect(harness.queryKeys).toContainEqual(["kanban", "labels", "t9"]);
    expect(harness.queryKeys.some((k) => k[0] === "kanban" && k[1] === "members" && k[2] !== "t9")).toBe(false);
  });

  it("shows labeled loading without fabricating editable properties", () => {
    setup();
    harness.detail = { data: undefined, isLoading: true, isError: false };
    const html = renderToStaticMarkup(
      createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }),
    );
    expect(html).toContain("detailLoading");
    expect(html).toContain('aria-busy="true"');
    // no status/priority/assignee controls before the detail resolves
    expect(html).not.toContain('aria-label="propStatus"');
    expect(html).not.toContain('aria-label="commentPlaceholder"');
  });

  it("shows the missing-issue state for the stable not-found code", () => {
    setup();
    harness.detail = { data: { ok: false, error: "issue-not-found", code: "issue-not-found" }, isLoading: false, isError: false };
    const html = renderToStaticMarkup(
      createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }),
    );
    expect(html).toContain("detailMissing");
    expect(html).not.toContain("detailUnavailable");
    expect(html).not.toContain('aria-label="propStatus"');
  });

  it("shows the unavailable state for other failures, without fabricated properties", () => {
    setup();
    harness.detail = { data: { ok: false, error: "linear graphql error" }, isLoading: false, isError: false };
    const html = renderToStaticMarkup(
      createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }),
    );
    expect(html).toContain("detailUnavailable");
    expect(html).toContain("linear graphql error");
    expect(html).not.toContain("detailMissing");
    expect(html).not.toContain('aria-label="propStatus"');
  });

  it("transport failure keeps the last detail under the warning with the last-success time", () => {
    setup();
    harness.detail = {
      data: { ok: true, data: detail },
      isLoading: false,
      isError: true,
      dataUpdatedAt: 9876,
    };
    const html = renderToStaticMarkup(
      createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }),
    );
    // visible stale guidance + retained content stays interactive
    expect(html).toContain("detailUnavailable");
    expect(html).toContain("Off-board cross-team issue");
    // the RelativeTime aside mounts (static render emits the empty span; the
    // relative text is computed client-side after mount)
    expect(html).toContain("<span></span>");
  });

  it("shows attachments and the parent link when present, and their empty states when absent", () => {
    setup();
    harness.detail = { data: { ok: true, data: { ...detail, attachments: [{ id: "a1", title: "PR", url: "https://x" }], parent: { id: "p1", identifier: "MKT-1", title: "Parent", url: "https://linear.app/x" } } }, isLoading: false, isError: false };
    const html = renderToStaticMarkup(createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }));
    expect(html).toContain("detailAttachments");
    expect(html).toContain('href="https://x"');
    expect(html).toContain("MKT-1");

    setup();
    harness.detail = { data: { ok: true, data: detail }, isLoading: false, isError: false };
    const html2 = renderToStaticMarkup(createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }));
    expect(html2).toContain("detailNoAttachments");
    expect(html2).not.toContain("detailParent");
  });

  it("shows the truncation note only for the collection that hit the ceiling", () => {
    setup();
    harness.detail = { data: { ok: true, data: { ...detail, childrenTruncated: true } }, isLoading: false, isError: false };
    const html = renderToStaticMarkup(createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }));
    expect(html).toContain("detailTruncated");
  });

  it("shows the incomplete-list note only for the collection that stopped early, never together with the ceiling note", () => {
    setup();
    harness.detail = { data: { ok: true, data: { ...detail, commentsIncomplete: true } }, isLoading: false, isError: false };
    const html = renderToStaticMarkup(createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }));
    expect(html).toContain("detailIncomplete");
    expect(html).not.toContain("detailTruncated");

    setup();
    harness.detail = { data: { ok: true, data: detail }, isLoading: false, isError: false };
    const html2 = renderToStaticMarkup(createElement(IssueDetailOverlay, { issueId: "i9", onClose: () => {} }));
    expect(html2).not.toContain("detailIncomplete");
  });
});

describe("shouldConfirmDiscard", () => {
  it("non-empty (trimmed) text differs from the default empty baseline", () => {
    expect(shouldConfirmDiscard("hello")).toBe(true);
    expect(shouldConfirmDiscard("   ")).toBe(false);
    expect(shouldConfirmDiscard("")).toBe(false);
  });
  it("compares against an explicit baseline for the description editor", () => {
    expect(shouldConfirmDiscard("same", "same")).toBe(false);
    expect(shouldConfirmDiscard("changed", "same")).toBe(true);
  });
  it("an unchanged description with surrounding whitespace never confirms (round-2 F3)", () => {
    const baseline = "  Notes with surrounding space.  \n";
    expect(shouldConfirmDiscard(baseline, baseline)).toBe(false);
  });
  it("flags a dirty title draft the same way as a dirty description draft (round-2 F2)", () => {
    expect(shouldConfirmDiscard("New title", "Old title")).toBe(true);
    expect(shouldConfirmDiscard("Old title", "Old title")).toBe(false);
  });
});

describe("applyCommentOptimistic / reconcileCommentCache", () => {
  type DetailCache = { ok: true; data: Omit<typeof detail, "comments"> & { comments: { id: string }[] } };
  function fakeQueryClient(cache: unknown) {
    const data = { current: cache };
    return {
      getQueryData: vi.fn((_key: unknown) => data.current),
      setQueryData: vi.fn((_key: unknown, updater: unknown) => {
        data.current = typeof updater === "function" ? (updater as (c: unknown) => unknown)(data.current) : updater;
      }),
      invalidateQueries: vi.fn(),
    };
  }
  const key = ["kanban", "issue", "i9"];
  function envelopeWith(comments: { id: string }[]): DetailCache {
    return { ok: true, data: { ...detail, comments } };
  }

  it("appends the optimistic comment onto the real envelope shape, and no-ops when the envelope is an error", () => {
    const qc = fakeQueryClient(envelopeWith([]));
    applyCommentOptimistic(qc as never, key, { id: "temp-1", author: "You", body: "hi", createdAt: "2026-01-01T00:00:00Z", pending: true });
    expect((qc.getQueryData(key) as DetailCache).data.comments).toEqual([
      { id: "temp-1", author: "You", body: "hi", createdAt: "2026-01-01T00:00:00Z", pending: true },
    ]);

    const qcErr = fakeQueryClient({ ok: false, error: "down" });
    applyCommentOptimistic(qcErr as never, key, { id: "temp-1", author: "You", body: "hi", createdAt: "2026-01-01T00:00:00Z", pending: true });
    expect(qcErr.setQueryData).not.toHaveBeenCalled();
  });

  it("invalidates the exact issue key on success and for unconfirmed/applied-unrecorded, never for a refused temp-removal", () => {
    // Success (guidance === null): reconcileComment itself receives no server
    // comment (diff review a5910333016f F2) — the temp entry stays in the
    // cache exactly as unconfirmed/applied-unrecorded leave it, and this
    // invalidate is what eventually replaces it via refetch.
    const qc0 = fakeQueryClient(envelopeWith([{ id: "temp-1" }]));
    reconcileCommentCache(qc0 as never, key, "temp-1", null);
    expect(qc0.invalidateQueries).toHaveBeenCalledWith({ queryKey: key });
    expect((qc0.getQueryData(key) as DetailCache).data.comments).toEqual([{ id: "temp-1" }]);

    const qc1 = fakeQueryClient(envelopeWith([{ id: "temp-1" }]));
    reconcileCommentCache(qc1 as never, key, "temp-1", { kind: "unconfirmed" });
    expect(qc1.invalidateQueries).toHaveBeenCalledWith({ queryKey: key });

    const qc2 = fakeQueryClient(envelopeWith([{ id: "temp-1" }]));
    reconcileCommentCache(qc2 as never, key, "temp-1", { kind: "applied-unrecorded" });
    expect(qc2.invalidateQueries).toHaveBeenCalledWith({ queryKey: key });

    const qc3 = fakeQueryClient(envelopeWith([{ id: "temp-1" }]));
    reconcileCommentCache(qc3 as never, key, "temp-1", { kind: "refused", error: "nope" });
    expect(qc3.invalidateQueries).not.toHaveBeenCalled();
    expect((qc3.getQueryData(key) as DetailCache).data.comments).toEqual([]);
  });
});