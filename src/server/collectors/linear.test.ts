import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { MutationRejected } from "../../lib/mutationOutcome";
import { IssueNotFound, cacheGet, cachePut, createComment, createIssue, fetchAllPaginated, getIssueDetail, getTeams, mapBoard, mapIssueDetail, mapLabels, mapMembers, mapProjectCounts, mapTeams, updateIssue } from "./linear";

const TEAMS_JSON = {
  data: { teams: { nodes: [{ id: "t1", key: "MOA", name: "Acme Digital" }] } },
};

const BOARD_JSON = {
  data: {
    team: {
      states: {
        nodes: [
          { id: "s-done", name: "Done", color: "#4cb782", position: 3, type: "completed" },
          { id: "s-rev", name: "In Review", color: "#0f7488", position: 1002, type: "started" },
          { id: "s-prog", name: "In Progress", color: "#f2c94c", position: 2, type: "started" },
          { id: "s-todo", name: "Todo", color: "#e2e2e2", position: 1, type: "unstarted" },
          { id: "s-back", name: "Backlog", color: "#bec2c8", position: 0, type: "backlog" },
          { id: "s-cancel", name: "Canceled", color: "#95a2b3", position: 4, type: "canceled" },
          { id: "s-dup", name: "Duplicate", color: "#95a2b3", position: 5, type: "duplicate" },
        ],
      },
      issues: {
        nodes: [
          {
            id: "i1",
            identifier: "MOA-1",
            title: "Full issue",
            priority: 2,
            url: "https://linear.app/x/issue/MOA-1",
            createdAt: "2026-07-06T23:39:19.093Z",
            updatedAt: "2026-07-07T11:12:42.641Z",
            state: { id: "s-prog" },
            project: { id: "proj1", name: "Jax OS" },
            assignee: { id: "u1", displayName: "Rafa" },
            labels: { nodes: [{ id: "l1", name: "bug", color: "#eb5757" }] },
          },
          {
            id: "i2",
            identifier: "MOA-2",
            title: "Bare issue",
            priority: 0,
            url: "https://linear.app/x/issue/MOA-2",
            createdAt: "2026-07-06T23:39:19.093Z",
            updatedAt: "2026-07-07T11:12:42.641Z",
            state: { id: "s-back" },
            project: null,
            assignee: null,
            labels: { nodes: [] },
          },
          {
            id: "i3",
            identifier: "MOA-3",
            title: "Parent with sub-issues",
            priority: 1,
            url: "https://linear.app/x/issue/MOA-3",
            createdAt: "2026-07-06T23:39:19.093Z",
            updatedAt: "2026-07-07T11:12:42.641Z",
            state: { id: "s-prog" },
            project: { name: "Jax OS" },
            assignee: null,
            labels: { nodes: [] },
            children: { nodes: [
              { state: { type: "completed" } },
              { state: { type: "started" } },
              { state: { type: "completed" } },
            ] },
          },
        ],
      },
    },
  },
};

describe("mapTeams", () => {
  it("maps team nodes", () => {
    expect(mapTeams(TEAMS_JSON)).toEqual([{ id: "t1", key: "MOA", name: "Acme Digital" }]);
  });

  it("throws on malformed response", () => {
    expect(() => mapTeams({ data: {} })).toThrow();
  });
});

describe("mapBoard", () => {
  it("filters excluded state types and orders by (typeRank, position)", () => {
    const { states } = mapBoard(BOARD_JSON);
    expect(states.map((s) => s.name)).toEqual([
      "Backlog",
      "Todo",
      "In Progress",
      "In Review",
      "Done",
    ]);
  });

  it("maps full and bare issues (null project/assignee, empty labels)", () => {
    const { issues } = mapBoard(BOARD_JSON);
    expect(issues[0]).toEqual({
      id: "i1",
      identifier: "MOA-1",
      title: "Full issue",
      priority: 2,
      url: "https://linear.app/x/issue/MOA-1",
      createdAt: "2026-07-06T23:39:19.093Z",
      updatedAt: "2026-07-07T11:12:42.641Z",
      stateId: "s-prog",
      project: "Jax OS",
      projectId: "proj1",
      assignee: "Rafa",
      assigneeId: "u1",
      labels: [{ id: "l1", name: "bug", color: "#eb5757" }],
    });
    expect(issues[1].project).toBeUndefined();
    expect(issues[1].assignee).toBeUndefined();
    expect(issues[1].labels).toEqual([]);
  });

  it("counts sub-issues (done = completed state type) on parents, undefined on leaves", () => {
    const { issues } = mapBoard(BOARD_JSON);
    expect(issues[0].subIssue).toBeUndefined(); // MOA-1 has no children
    expect(issues[2].subIssue).toEqual({ total: 3, done: 2 }); // MOA-3: 2 completed of 3
  });

  it("throws on malformed response", () => {
    expect(() => mapBoard({ data: { team: null } })).toThrow();
  });
});

describe("mapProjectCounts", () => {
  it("groups by project name and skips null projects", () => {
    const json = {
      data: {
        issues: {
          nodes: [
            { project: { name: "Jax OS" } },
            { project: { name: "Jax OS" } },
            { project: { name: "Other" } },
            { project: null },
          ],
        },
      },
    };
    expect(mapProjectCounts(json)).toEqual({ "Jax OS": 2, Other: 1 });
  });

  it("throws on malformed response", () => {
    expect(() => mapProjectCounts({})).toThrow();
  });
});

const DETAIL_JSON = {
  data: { issue: {
    id: "i9",
    identifier: "MKT-9",
    title: "Off-board cross-team issue",
    priority: 3,
    url: "https://linear.app/x/issue/MKT-9",
    createdAt: "2026-07-06T23:39:19.093Z",
    updatedAt: "2026-07-07T11:12:42.641Z",
    state: { id: "s-cancel" },
    project: { name: "Marketing" },
    assignee: { id: "u9", displayName: "Ana" },
    labels: { nodes: [{ id: "l9", name: "design", color: "#8a2be2" }] },
    team: {
      id: "t9",
      states: { nodes: [
        { id: "s-triage", name: "Triage", color: "#f2c94c", position: 0, type: "triage" },
        { id: "s-todo9", name: "Todo", color: "#e2e2e2", position: 1, type: "unstarted" },
        { id: "s-prog9", name: "In Progress", color: "#f2c94c", position: 2, type: "started" },
        { id: "s-done9", name: "Done", color: "#4cb782", position: 3, type: "completed" },
        { id: "s-cancel", name: "Canceled", color: "#95a2b3", position: 4, type: "canceled" },
      ] },
    },
    description: "## Hi\n\n- a\n- b",
    children: { nodes: [
      { identifier: "MOA-9", title: "done one", state: { name: "Done", type: "completed" }, assignee: { displayName: "Rafa" } },
      { identifier: "MOA-8", title: "open one", state: { name: "Backlog", type: "backlog" }, assignee: null },
    ] },
    comments: { nodes: [
      { id: "c1", body: "**hey**", createdAt: "2026-07-07T10:00:00.000Z", user: { displayName: "contato" }, botActor: null },
      { id: "c2", body: "bot note", createdAt: "2026-07-07T11:00:00.000Z", user: null, botActor: { name: "Jax" } },
    ] },
  } },
};

// minimal valid detail node for tests that only care about one field
const MINIMAL_DETAIL = {
  id: "i1",
  identifier: "MOA-1",
  title: "t",
  priority: 0,
  url: "https://linear.app/x",
  createdAt: "2026-01-01T00:00:00Z",
  updatedAt: "2026-01-01T00:00:00Z",
  state: { id: "s1" },
  team: { id: "t1", states: { nodes: [] } },
};

describe("mapIssueDetail", () => {
  it("maps description, sub-issue progress, and comments; sorts sub-issues ascending by number", () => {
    const d = mapIssueDetail(DETAIL_JSON);
    expect(d.description).toBe("## Hi\n\n- a\n- b");
    expect(d.subIssue).toEqual({ total: 2, done: 1 });
    // API returned MOA-9 then MOA-8; detail lists them ascending → MOA-8 first
    expect(d.children.map((c) => c.identifier)).toEqual(["MOA-8", "MOA-9"]);
    expect(d.children[0]).toEqual({ identifier: "MOA-8", title: "open one", stateName: "Backlog", stateType: "backlog", assignee: undefined });
    expect(d.children[1]).toEqual({ identifier: "MOA-9", title: "done one", stateName: "Done", stateType: "completed", assignee: "Rafa" });
  });
  it("carries the issue, its team id and the FULL team state set from detail data, off-board or cross-team", () => {
    const d = mapIssueDetail(DETAIL_JSON);
    expect(d.issue).toEqual({
      id: "i9",
      identifier: "MKT-9",
      title: "Off-board cross-team issue",
      priority: 3,
      url: "https://linear.app/x/issue/MKT-9",
      createdAt: "2026-07-06T23:39:19.093Z",
      updatedAt: "2026-07-07T11:12:42.641Z",
      stateId: "s-cancel",
      project: "Marketing",
      assignee: "Ana",
      assigneeId: "u9",
      labels: [{ id: "l9", name: "design", color: "#8a2be2" }],
      subIssue: { total: 2, done: 1 },
    });
    expect(d.teamId).toBe("t9");
    // canceled/triage stay available for an off-board issue currently on one
    expect(d.states.map((s) => s.name)).toEqual(["Triage", "Todo", "In Progress", "Done", "Canceled"]);
    expect(d.states.map((s) => s.type)).toEqual(["triage", "unstarted", "started", "completed", "canceled"]);
  });
  it("throws IssueNotFound when the issue does not exist (data.issue null)", () => {
    expect(() => mapIssueDetail({ data: { issue: null } })).toThrow(IssueNotFound);
    expect(() => mapIssueDetail({ data: { issue: null } })).toThrow("issue-not-found");
  });
  it("throws on malformed required identity/title/state/team fields, never String(undefined)", () => {
    for (const drop of ["id", "identifier", "title", "state", "team"] as const) {
      const bad = { data: { issue: { ...MINIMAL_DETAIL, [drop]: undefined } } };
      expect(() => mapIssueDetail(bad)).toThrow("unexpected issue response");
    }
    expect(() => mapIssueDetail({ data: { issue: { ...MINIMAL_DETAIL, team: { id: "t1", states: null } } } })).toThrow("unexpected issue response");
  });
  it("falls back to botActor when comment has no user, then to em dash", () => {
    const d = mapIssueDetail(DETAIL_JSON);
    expect(d.comments[0].author).toBe("contato");
    expect(d.comments[1].author).toBe("Jax");
    const noAuthor = mapIssueDetail({ data: { issue: { ...MINIMAL_DETAIL, description: "", children: { nodes: [] }, comments: { nodes: [{ id: "c", body: "", createdAt: "", user: null, botActor: null }] } } } });
    expect(noAuthor.comments[0].author).toBe("—");
  });
  it("maps attachments and parent when present", () => {
    const withExtras = {
      data: { issue: {
        ...DETAIL_JSON.data.issue,
        attachments: { nodes: [{ id: "a1", title: "PR #12", url: "https://github.com/x/pr/12" }] },
        parent: { id: "p1", identifier: "MKT-1", title: "Parent issue", url: "https://linear.app/x/issue/MKT-1" },
      } },
    };
    const d = mapIssueDetail(withExtras);
    expect(d.attachments).toEqual([{ id: "a1", title: "PR #12", url: "https://github.com/x/pr/12" }]);
    expect(d.parent).toEqual({ id: "p1", identifier: "MKT-1", title: "Parent issue", url: "https://linear.app/x/issue/MKT-1" });
    expect(d.commentsTruncated).toBe(false);
    expect(d.childrenTruncated).toBe(false);
  });

  it("maps neither attachments nor parent when absent", () => {
    const d = mapIssueDetail(DETAIL_JSON);
    expect(d.attachments).toEqual([]);
    expect(d.parent).toBeUndefined();
  });
  it("throws on malformed response", () => {
    expect(() => mapIssueDetail({ data: {} })).toThrow();
  });
});

describe("getIssueDetail pagination", () => {
  const prevKey = process.env.LINEAR_API_KEY;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    process.env.LINEAR_API_KEY = "test-key";
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    if (prevKey === undefined) delete process.env.LINEAR_API_KEY;
    else process.env.LINEAR_API_KEY = prevKey;
    vi.unstubAllGlobals();
  });

  function jsonOk(body: unknown) {
    fetchMock.mockResolvedValueOnce({ ok: true, json: async () => body });
  }

  // F3: pulls the exact GraphQL variables sent on a given call, so the
  // cursor sequence per call (null → first endCursor → second, …) is
  // asserted directly instead of only counting calls/lengths.
  function variablesOf(call: number): Record<string, unknown> {
    return (JSON.parse((fetchMock.mock.calls[call][1] as RequestInit).body as string) as { variables: Record<string, unknown> }).variables;
  }

  function detailPage(opts: {
    comments?: { count: number; hasNextPage: boolean; endCursor: string | null };
    children?: { count: number; hasNextPage: boolean; endCursor: string | null };
  }) {
    const c = opts.comments ?? { count: 0, hasNextPage: false, endCursor: null };
    const ch = opts.children ?? { count: 0, hasNextPage: false, endCursor: null };
    return {
      data: { issue: {
        ...MINIMAL_DETAIL,
        description: "",
        attachments: { nodes: [] },
        parent: null,
        comments: {
          nodes: Array.from({ length: c.count }, (_, i) => ({ id: `c${i}`, body: "x", createdAt: "2026-01-01T00:00:00Z", user: { displayName: "Rafa" }, botActor: null })),
          pageInfo: { hasNextPage: c.hasNextPage, endCursor: c.endCursor },
        },
        children: {
          nodes: Array.from({ length: ch.count }, (_, i) => ({ identifier: `MOA-${i}`, title: "t", state: { name: "Todo", type: "unstarted" }, assignee: null })),
          pageInfo: { hasNextPage: ch.hasNextPage, endCursor: ch.endCursor },
        },
      } },
    };
  }

  it("concatenates a two-page comments thread; single-page children stay untruncated", async () => {
    jsonOk(detailPage({ comments: { count: 100, hasNextPage: true, endCursor: "cc1" }, children: { count: 1, hasNextPage: false, endCursor: null } }));
    jsonOk(detailPage({ comments: { count: 5, hasNextPage: false, endCursor: null } }));
    const detail = await getIssueDetail("i1");
    expect(detail.comments).toHaveLength(105);
    expect(detail.commentsTruncated).toBe(false);
    expect(detail.childrenTruncated).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(variablesOf(0)).toEqual({ id: "i1", commentsCursor: null, childrenCursor: null });
    expect(variablesOf(1)).toEqual({ id: "i1", commentsCursor: "cc1", childrenCursor: null });
  });

  // Round-2 F1: a distinct issue id per test — the initial page is cached (30s TTL)
  // keyed on { id, commentsCursor: null, childrenCursor: null }, so reusing
  // "i1" here would hit the previous test's cached first page instead of
  // this test's own mocked response.
  it("stops the comments loop at 1,000 and flags truncated; children untouched", async () => {
    jsonOk(detailPage({ comments: { count: 250, hasNextPage: true, endCursor: "c1" } }));
    jsonOk(detailPage({ comments: { count: 250, hasNextPage: true, endCursor: "c2" } }));
    jsonOk(detailPage({ comments: { count: 250, hasNextPage: true, endCursor: "c3" } }));
    jsonOk(detailPage({ comments: { count: 250, hasNextPage: true, endCursor: "c4" } }));
    const detail = await getIssueDetail("i2");
    expect(detail.comments).toHaveLength(1000);
    expect(detail.commentsTruncated).toBe(true);
    expect(detail.commentsIncomplete).toBe(false);
    expect(detail.childrenTruncated).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(variablesOf(0)).toEqual({ id: "i2", commentsCursor: null, childrenCursor: null });
    expect(variablesOf(1)).toEqual({ id: "i2", commentsCursor: "c1", childrenCursor: null });
    expect(variablesOf(2)).toEqual({ id: "i2", commentsCursor: "c2", childrenCursor: null });
    expect(variablesOf(3)).toEqual({ id: "i2", commentsCursor: "c3", childrenCursor: null });
  });

  // Round-2 F1: distinct id — see the note above the previous test.
  it("independently: concatenates a two-page children list; single-page comments stay untruncated", async () => {
    jsonOk(detailPage({ children: { count: 100, hasNextPage: true, endCursor: "kc1" } }));
    jsonOk(detailPage({ children: { count: 5, hasNextPage: false, endCursor: null } }));
    const detail = await getIssueDetail("i3");
    expect(detail.children).toHaveLength(105);
    expect(detail.childrenTruncated).toBe(false);
    expect(detail.commentsTruncated).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(variablesOf(0)).toEqual({ id: "i3", commentsCursor: null, childrenCursor: null });
    expect(variablesOf(1)).toEqual({ id: "i3", commentsCursor: null, childrenCursor: "kc1" });
  });

  // Round-2 F1: distinct id — see the note above the second pagination test.
  it("independently: stops the children loop at 1,000 and flags truncated; comments untouched", async () => {
    jsonOk(detailPage({ children: { count: 250, hasNextPage: true, endCursor: "k1" } }));
    jsonOk(detailPage({ children: { count: 250, hasNextPage: true, endCursor: "k2" } }));
    jsonOk(detailPage({ children: { count: 250, hasNextPage: true, endCursor: "k3" } }));
    jsonOk(detailPage({ children: { count: 250, hasNextPage: true, endCursor: "k4" } }));
    const detail = await getIssueDetail("i4");
    expect(detail.children).toHaveLength(1000);
    expect(detail.childrenTruncated).toBe(true);
    expect(detail.childrenIncomplete).toBe(false);
    expect(detail.commentsTruncated).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(4);
    expect(variablesOf(0)).toEqual({ id: "i4", commentsCursor: null, childrenCursor: null });
    expect(variablesOf(1)).toEqual({ id: "i4", commentsCursor: null, childrenCursor: "k1" });
    expect(variablesOf(2)).toEqual({ id: "i4", commentsCursor: null, childrenCursor: "k2" });
    expect(variablesOf(3)).toEqual({ id: "i4", commentsCursor: null, childrenCursor: "k3" });
  });

  // F1: an early abort (error/malformed page) must never look like a ceiling —
  // the UI reads commentsTruncated/childrenTruncated as "first 1,000 shown",
  // which a handful of items before a timeout is not. Both fields flip to
  // their *Incomplete counterpart instead, in one issue to keep this cheap.
  it("flags an aborted page as incomplete, not truncated, for both comments and children", async () => {
    jsonOk(detailPage({
      comments: { count: 50, hasNextPage: true, endCursor: "cc1" },
      children: { count: 10, hasNextPage: true, endCursor: "kc1" },
    }));
    fetchMock.mockRejectedValueOnce(new Error("timeout"));
    fetchMock.mockRejectedValueOnce(new Error("timeout"));
    const detail = await getIssueDetail("i5");
    expect(detail.commentsTruncated).toBe(false);
    expect(detail.commentsIncomplete).toBe(true);
    expect(detail.childrenTruncated).toBe(false);
    expect(detail.childrenIncomplete).toBe(true);
  });
});

describe("mapMembers / mapLabels", () => {
  it("maps team members", () => {
    expect(mapMembers({ data: { team: { members: { nodes: [{ id: "u1", displayName: "contato" }] } } } }))
      .toEqual([{ id: "u1", displayName: "contato" }]);
  });
  it("maps team labels", () => {
    expect(mapLabels({ data: { team: { labels: { nodes: [{ id: "l1", name: "bug", color: "#eb5757" }] } } } }))
      .toEqual([{ id: "l1", name: "bug", color: "#eb5757" }]);
  });
  it("throws on malformed", () => {
    expect(() => mapMembers({ data: { team: null } })).toThrow();
    expect(() => mapLabels({ data: { team: null } })).toThrow();
  });
});

describe("response cache helpers", () => {
  it("returns a cached entry within the TTL and misses after expiry", () => {
    const cache = new Map<string, { at: number; json: unknown }>();
    cachePut(cache, "k", { data: 1 }, 1_000);
    expect(cacheGet(cache, "k", 1_000 + 29_999)).toEqual({ data: 1 });
    expect(cacheGet(cache, "k", 1_000 + 30_000)).toBeNull();
    expect(cacheGet(cache, "missing", 1_000)).toBeNull();
  });
});

describe("fetchAllPaginated", () => {
  const page = (count: number, hasNextPage: boolean, endCursor: string | null) => ({
    nodes: Array.from({ length: count }, (_, i) => i),
    pageInfo: { hasNextPage, endCursor },
  });

  it("concatenates pages until hasNextPage is false, not truncated", async () => {
    const fetchPage = vi.fn(async () => page(1, false, null));
    const extract = (json: unknown) => json as ReturnType<typeof page>;
    const result = await fetchAllPaginated(page(2, true, "c1"), fetchPage, extract);
    expect(result).toEqual({ nodes: [0, 1, 0], truncated: false, reason: null });
    expect(fetchPage).toHaveBeenCalledTimes(1);
    expect(fetchPage).toHaveBeenCalledWith("c1");
  });

  it("never calls fetchPage when the first page already has no next page", async () => {
    const fetchPage = vi.fn();
    const result = await fetchAllPaginated(page(3, false, null), fetchPage, (j) => j as ReturnType<typeof page>);
    expect(result).toEqual({ nodes: [0, 1, 2], truncated: false, reason: null });
    expect(fetchPage).not.toHaveBeenCalled();
  });

  it("stops at the 1,000-item ceiling and flags truncated when more remain", async () => {
    const later = [page(250, true, "c2"), page(250, true, "c3"), page(250, true, "c4")];
    let call = 0;
    const fetchPage = vi.fn(async () => later[call++]);
    const result = await fetchAllPaginated(page(250, true, "c1"), fetchPage, (j) => j as ReturnType<typeof page>);
    expect(result.nodes).toHaveLength(1000);
    expect(result.truncated).toBe(true);
    expect(result.reason).toBe("ceiling");
    expect(fetchPage).toHaveBeenCalledTimes(3);
  });

  it("does not truncate when the total lands exactly on the ceiling with no next page", async () => {
    const later = [page(250, true, "c2"), page(250, true, "c3"), page(250, false, null)];
    let call = 0;
    const fetchPage = vi.fn(async () => later[call++]);
    const result = await fetchAllPaginated(page(250, true, "c1"), fetchPage, (j) => j as ReturnType<typeof page>);
    expect(result.nodes).toHaveLength(1000);
    expect(result.truncated).toBe(false);
    expect(result.reason).toBeNull();
    expect(fetchPage).toHaveBeenCalledTimes(3);
  });

  // Diff review a7183f365215 F1: a final page can push the total past the
  // ceiling while itself reporting hasNextPage:false — must still flag
  // ceiling, not silently drop the slice as untruncated.
  it("flags the ceiling when the final page crosses 1,000 items but reports no next page", async () => {
    const fetchPage = vi.fn(async () => page(100, false, null));
    const result = await fetchAllPaginated(page(999, true, "c1"), fetchPage, (j) => j as ReturnType<typeof page>);
    expect(result.nodes).toHaveLength(1000);
    expect(result.truncated).toBe(true);
    expect(result.reason).toBe("ceiling");
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });

  // F1 regressions: a page that claims more but can never make forward
  // progress must abort (truncated:true), never loop until the process hangs.
  it("aborts with truncated:true when a page claims more but has no endCursor", async () => {
    const fetchPage = vi.fn();
    const result = await fetchAllPaginated(page(2, true, null), fetchPage, (j) => j as ReturnType<typeof page>);
    expect(result).toEqual({ nodes: [0, 1], truncated: true, reason: "incomplete" });
    expect(fetchPage).not.toHaveBeenCalled();
  });

  it("aborts with truncated:true when a page repeats a cursor already used", async () => {
    const repeat = page(1, true, "c1");
    const fetchPage = vi.fn(async () => repeat);
    const result = await fetchAllPaginated(page(1, true, "c1"), fetchPage, (j) => j as ReturnType<typeof page>);
    expect(result).toEqual({ nodes: [0, 0], truncated: true, reason: "incomplete" });
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });

  it("aborts with truncated:true when a page returns zero nodes while claiming more", async () => {
    const empty = page(0, true, "c2");
    const fetchPage = vi.fn(async () => empty);
    const result = await fetchAllPaginated(page(1, true, "c1"), fetchPage, (j) => j as ReturnType<typeof page>);
    expect(result).toEqual({ nodes: [0], truncated: true, reason: "incomplete" });
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });

  // Round-2 F3: a rejected fetchPage (network failure, timeout) must not discard the
  // first page already loaded — stop and return it with truncated:true.
  it("aborts with truncated:true, keeping the first page's nodes, when a later page's fetch rejects", async () => {
    const fetchPage = vi.fn(async () => { throw new Error("network down"); });
    const result = await fetchAllPaginated(page(2, true, "c1"), fetchPage, (j) => j as ReturnType<typeof page>);
    expect(result).toEqual({ nodes: [0, 1], truncated: true, reason: "incomplete" });
    expect(fetchPage).toHaveBeenCalledTimes(1);
  });
});

describe("linear mutations", () => {
  const prevKey = process.env.LINEAR_API_KEY;
  let fetchMock: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    process.env.LINEAR_API_KEY = "test-key";
    fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
  });
  afterEach(() => {
    if (prevKey === undefined) delete process.env.LINEAR_API_KEY;
    else process.env.LINEAR_API_KEY = prevKey;
    vi.unstubAllGlobals();
  });

  function jsonOk(body: unknown) {
    fetchMock.mockResolvedValue({ ok: true, json: async () => body });
  }

  it("resolves when issueUpdate success is true and does not retry", async () => {
    jsonOk({ data: { issueUpdate: { success: true } } });
    await updateIssue("i1", { priority: 1 });
    expect(fetchMock).toHaveBeenCalledOnce();
  });

  it("throws MutationRejected on explicit success false", async () => {
    jsonOk({ data: { issueUpdate: { success: false } } });
    await expect(updateIssue("i1", { priority: 1 })).rejects.toBeInstanceOf(MutationRejected);
    await expect(updateIssue("i1", { priority: 1 })).rejects.toThrow("issueUpdate did not succeed");
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("treats missing success, null json, HTTP 500, graphql errors, rejected fetch and timeout as unconfirmed", async () => {
    jsonOk({ data: { issueUpdate: {} } });
    await expect(updateIssue("i1", { stateId: "s1" })).rejects.toThrow("issueUpdate outcome unconfirmed");
    await expect(updateIssue("i1", { stateId: "s1" })).rejects.not.toBeInstanceOf(MutationRejected);

    jsonOk(null);
    await expect(updateIssue("i1", { stateId: "s1" })).rejects.not.toBeInstanceOf(MutationRejected);

    fetchMock.mockResolvedValue({ ok: false, status: 500 });
    await expect(updateIssue("i1", { stateId: "s1" })).rejects.toThrow("linear responded 500");
    await expect(updateIssue("i1", { stateId: "s1" })).rejects.not.toBeInstanceOf(MutationRejected);

    jsonOk({ errors: [{ message: "ambiguous" }] });
    await expect(updateIssue("i1", { stateId: "s1" })).rejects.not.toBeInstanceOf(MutationRejected);

    fetchMock.mockRejectedValue(new Error("network"));
    await expect(updateIssue("i1", { stateId: "s1" })).rejects.not.toBeInstanceOf(MutationRejected);

    fetchMock.mockRejectedValue(Object.assign(new Error("aborted"), { name: "TimeoutError" }));
    await expect(updateIssue("i1", { stateId: "s1" })).rejects.not.toBeInstanceOf(MutationRejected);
  });

  it("classifies commentCreate the same way", async () => {
    jsonOk({ data: { commentCreate: { success: true } } });
    await createComment("i1", "hi");
    jsonOk({ data: { commentCreate: { success: false } } });
    await expect(createComment("i1", "hi")).rejects.toBeInstanceOf(MutationRejected);
    jsonOk({ data: { commentCreate: {} } });
    await expect(createComment("i1", "hi")).rejects.toThrow("commentCreate outcome unconfirmed");
  });

  it("clears the cache on success", async () => {
    // 4th mock is a cleanup (deviation from the plan): the module-level
    // responseCache persists across tests in this file, and this test ends
    // with a warm teams entry — without one more successful mutation to clear
    // it, the pre-existing "caches reads" test below would see a cache hit and
    // fail its toHaveBeenCalledOnce.
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => TEAMS_JSON })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ data: { issueCreate: { success: true, issue: { id: "i9", identifier: "MOA-9" } } } }) })
      .mockResolvedValueOnce({ ok: true, json: async () => TEAMS_JSON })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ data: { issueCreate: { success: true, issue: { id: "i9", identifier: "MOA-9" } } } }) });
    await getTeams();
    await getTeams();
    expect(fetchMock).toHaveBeenCalledOnce();
    await createIssue({ teamId: "t1", title: "x" });
    await getTeams();
    expect(fetchMock).toHaveBeenCalledTimes(3);
    await createIssue({ teamId: "t1", title: "x" });
  });

  it("creates an issue on success:true, returns id/identifier", async () => {
    jsonOk({ data: { issueCreate: { success: true, issue: { id: "i9", identifier: "MOA-9" } } } });
    await expect(createIssue({ teamId: "t1", title: "New issue" })).resolves.toEqual({ id: "i9", identifier: "MOA-9" });
  });

  it("sends the exact GraphQL query and { variables: { input } }, required-only and with every optional field (F8)", async () => {
    jsonOk({ data: { issueCreate: { success: true, issue: { id: "i9", identifier: "MOA-9" } } } });
    await createIssue({ teamId: "t1", title: "New issue" });
    const minimal = JSON.parse(fetchMock.mock.calls[0][1].body as string);
    expect(minimal.query).toContain("mutation Create($input: IssueCreateInput!)");
    expect(minimal.query).toContain("issueCreate(input: $input) { success issue { id identifier } }");
    expect(minimal.variables).toEqual({ input: { teamId: "t1", title: "New issue" } });

    const full = { teamId: "t1", title: "New issue", description: "d", projectId: "p1", stateId: "s1", priority: 2, assigneeId: "u1", labelIds: ["l1"] };
    await createIssue(full);
    const withOptionals = JSON.parse(fetchMock.mock.calls[1][1].body as string);
    expect(withOptionals.variables).toEqual({ input: full });
  });

  it("throws MutationRejected on explicit success false", async () => {
    jsonOk({ data: { issueCreate: { success: false } } });
    await expect(createIssue({ teamId: "t1", title: "New issue" })).rejects.toBeInstanceOf(MutationRejected);
    await expect(createIssue({ teamId: "t1", title: "New issue" })).rejects.toThrow("issueCreate did not succeed");
  });

  it("treats missing success or missing issue fields as unconfirmed, never MutationRejected", async () => {
    jsonOk({ data: { issueCreate: { success: true } } });
    await expect(createIssue({ teamId: "t1", title: "x" })).rejects.toThrow("issueCreate outcome unconfirmed");
    await expect(createIssue({ teamId: "t1", title: "x" })).rejects.not.toBeInstanceOf(MutationRejected);
    jsonOk({ data: { issueCreate: {} } });
    await expect(createIssue({ teamId: "t1", title: "x" })).rejects.toThrow("issueCreate outcome unconfirmed");
  });

  it("rejects a missing API key before fetch", async () => {
    delete process.env.LINEAR_API_KEY;
    await expect(updateIssue("i1", { priority: 0 })).rejects.toBeInstanceOf(MutationRejected);
    await expect(updateIssue("i1", { priority: 0 })).rejects.toThrow("LINEAR_API_KEY not configured");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("caches reads and clears the cache after a successful mutation", async () => {
    fetchMock
      .mockResolvedValueOnce({ ok: true, json: async () => TEAMS_JSON })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ data: { issueUpdate: { success: true } } }) })
      .mockResolvedValueOnce({ ok: true, json: async () => TEAMS_JSON });
    await getTeams();
    await getTeams();
    expect(fetchMock).toHaveBeenCalledOnce();
    await updateIssue("i1", { priority: 2 });
    await getTeams();
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

});
