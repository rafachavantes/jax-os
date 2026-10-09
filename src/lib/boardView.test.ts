import { describe, expect, it } from "vitest";
import { filterIssues, nextBoardState, sortIssues } from "./boardView";
import type { KanbanIssue } from "@/server/collectors/linear";

const mk = (o: Partial<KanbanIssue>): KanbanIssue => ({
  id: "i", identifier: "MOA-1", title: "t", priority: 0, url: "", stateId: "s",
  createdAt: "2026-01-01T00:00:00.000Z", updatedAt: "2026-01-01T00:00:00.000Z", labels: [], ...o,
});

describe("filterIssues", () => {
  const issues = [
    mk({ id: "a", identifier: "MOA-1", title: "Alpha", priority: 1, project: "Jax OS", labels: [{ id: "l1", name: "bug", color: "#000" }] }),
    mk({ id: "b", identifier: "MOA-2", title: "Beta", priority: 3, project: "Other", labels: [] }),
  ];
  it("filters by text on title+identifier", () => {
    expect(filterIssues(issues, { query: "alpha", project: null, priority: null, label: null, assignee: null }).map((i) => i.id)).toEqual(["a"]);
    expect(filterIssues(issues, { query: "moa-2", project: null, priority: null, label: null, assignee: null }).map((i) => i.id)).toEqual(["b"]);
  });
  it("combines project + priority + label (AND)", () => {
    expect(filterIssues(issues, { query: "", project: "Jax OS", priority: 1, label: "bug", assignee: null }).map((i) => i.id)).toEqual(["a"]);
    expect(filterIssues(issues, { query: "", project: "Jax OS", priority: 3, label: null, assignee: null })).toEqual([]);
  });
  it("combines assignee with the other three (AND)", () => {
    const withAssignee = [
      mk({ id: "a", identifier: "MOA-1", assignee: "Rafa", project: "Jax OS", priority: 1, labels: [{ id: "l1", name: "bug", color: "#000" }] }),
      mk({ id: "b", identifier: "MOA-2", assignee: "Ana" }),
    ];
    expect(filterIssues(withAssignee, { query: "", project: null, priority: null, label: null, assignee: "Rafa" }).map((i) => i.id)).toEqual(["a"]);
    expect(filterIssues(withAssignee, { query: "", project: null, priority: null, label: null, assignee: "Ana" }).map((i) => i.id)).toEqual(["b"]);
  });
});

describe("sortIssues", () => {
  const issues = [
    mk({ id: "none", priority: 0, identifier: "MOA-3", createdAt: "2026-01-03T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z" }),
    mk({ id: "urgent", priority: 1, identifier: "MOA-1", createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-03T00:00:00Z" }),
    mk({ id: "med", priority: 3, identifier: "MOA-2", createdAt: "2026-01-02T00:00:00Z", updatedAt: "2026-01-02T00:00:00Z" }),
  ];
  it("priority: urgent first, none last", () => {
    expect(sortIssues(issues, "priority").map((i) => i.id)).toEqual(["urgent", "med", "none"]);
  });
  it("created: newest first", () => {
    expect(sortIssues(issues, "created").map((i) => i.id)).toEqual(["none", "med", "urgent"]);
  });
  it("number: ascending identifier number", () => {
    expect(sortIssues(issues, "number").map((i) => i.identifier)).toEqual(["MOA-1", "MOA-2", "MOA-3"]);
  });
  it("does not mutate input", () => {
    const copy = [...issues];
    sortIssues(issues, "priority");
    expect(issues).toEqual(copy);
  });
});

describe("nextBoardState", () => {
  const board = { states: [], issues: [mk({ id: "a", stateId: "s1" }), mk({ id: "b", stateId: "s3" })] };
  it("patches only the dragged issue's stateId, leaving other issues untouched", () => {
    const next = nextBoardState(board, "a", "s2");
    expect(next.issues.find((i) => i.id === "a")?.stateId).toBe("s2");
    // F4: b starts on a state DIFFERENT from the move's target, so this
    // actually proves only the dragged issue changed — b already sharing the
    // target state would pass even if every issue got patched.
    expect(next.issues.find((i) => i.id === "b")?.stateId).toBe("s3");
  });
  it("does not mutate the input board", () => {
    const before = JSON.stringify(board);
    nextBoardState(board, "a", "s2");
    expect(JSON.stringify(board)).toBe(before);
  });
  it("leaves the board unchanged when the issue id is not found", () => {
    expect(nextBoardState(board, "missing", "s2")).toEqual(board);
  });
});
