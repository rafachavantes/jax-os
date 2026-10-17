import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => {
  class IssueNotFound extends Error {
    constructor() { super("issue-not-found"); }
  }
  return { getIssueDetail: vi.fn(), IssueNotFound };
});
vi.mock("@/server/collectors/linear", () => ({
  getIssueDetail: (...args: unknown[]) => harness.getIssueDetail(...args),
  IssueNotFound: harness.IssueNotFound,
}));

const settingsState = vi.hoisted(() => ({ linear: true }));
vi.mock("@/server/settings", () => ({
  readGeneralSettings: () => ({ ok: true, data: { integrations: { linear: settingsState.linear } } }),
}));

import { GET } from "./route";

function req(qs: string) {
  return new Request(`http://127.0.0.1/api/kanban/issue${qs}`);
}

const FULL_DETAIL = {
  description: "desc",
  subIssue: { total: 0, done: 0 },
  children: [],
  comments: [],
  teamId: "t9",
  states: [{ id: "s-cancel", name: "Canceled", color: "#95a2b3", position: 4, type: "canceled" }],
  issue: {
    id: "i9",
    identifier: "MKT-9",
    title: "Off-board issue",
    priority: 3,
    url: "https://linear.app/x",
    stateId: "s-cancel",
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
    project: "Marketing",
    assignee: "Ana",
    assigneeId: "u9",
    labels: [{ id: "l9", name: "design", color: "#8a2be2" }],
  },
};

describe("GET /api/kanban/issue", () => {
  it("refuses when linear is disabled, never calling getIssueDetail (spec per-integration table)", async () => {
    settingsState.linear = false;
    try {
      const res = await GET(req("?id=i1"));
      expect(res.status).toBe(200);
      expect(await res.json()).toEqual({ ok: false, error: "disabled" });
      expect(harness.getIssueDetail).not.toHaveBeenCalled();
    } finally {
      settingsState.linear = true;
    }
  });

  it("rejects a missing id without invoking the collector", async () => {
    expect(await (await GET(req(""))).json()).toEqual({ ok: false, error: "missing id" });
    expect(harness.getIssueDetail).not.toHaveBeenCalled();
  });

  it("rejects over-cap ids (UTF-8 bytes) without invoking the collector", async () => {
    expect(await (await GET(req(`?id=${"é".repeat(129)}`))).json()).toEqual({ ok: false, error: "invalid payload" });
    expect(await (await GET(req(`?id=${"a".repeat(257)}`))).json()).toEqual({ ok: false, error: "invalid payload" });
    expect(harness.getIssueDetail).not.toHaveBeenCalled();
  });

  it("passes legal ids, including the 256-byte boundary, to the collector", async () => {
    harness.getIssueDetail.mockReset();
    harness.getIssueDetail.mockResolvedValue(FULL_DETAIL);
    const legal = await (await GET(req(`?id=${"é".repeat(128)}`))).json();
    expect(legal.ok).toBe(true);
    expect(legal.data.teamId).toBe("t9");
    const boundary = await (await GET(req("?id=" + "a".repeat(256)))).json();
    expect(boundary.ok).toBe(true);
    expect(harness.getIssueDetail).toHaveBeenCalledTimes(2);
  });

  it("maps a missing issue to the stable not-found envelope", async () => {
    harness.getIssueDetail.mockReset();
    harness.getIssueDetail.mockRejectedValue(new harness.IssueNotFound());
    const res = await (await GET(req("?id=MOA-99"))).json();
    expect(res.ok).toBe(false);
    expect(res).toEqual({ ok: false, error: "issue-not-found", code: "issue-not-found" });
  });

  it("surfaces a source failure as an error envelope without a code, not as not-found", async () => {
    harness.getIssueDetail.mockReset();
    harness.getIssueDetail.mockRejectedValue(new Error("linear graphql error"));
    const res = await (await GET(req("?id=MOA-1"))).json();
    expect(res.ok).toBe(false);
    expect(res.code).toBeUndefined();
    expect(String(res.error)).toContain("linear");
  });
});