import { describe, expect, it } from "vitest";
import {
  boardEnabled, hydrateFilters, kanbanFilters, mergeKanbanPatch, normalizeKanbanSelection, parseAuditSearch, parseHealthSearch,
  parseKanbanSearch, serializeAuditSearch, serializeHealthSearch, serializeKanbanSearch,
} from "./cockpitUrl";

describe("kanban URL state", () => {
  it("roundtrips non-default fields and omits defaults", () => {
    const state = parseKanbanSearch(new URLSearchParams("team=t1&issue=i1&view=list&sort=updated&q=foo&project=P&priority=0&label=bug&assignee=u1"));
    expect(state).toEqual({
      team: "t1", issue: "i1", view: "list", sort: "updated", q: "foo", project: "P", priority: 0, label: "bug", assignee: "u1",
    });
    expect(serializeKanbanSearch(new URLSearchParams(), state).toString()).toBe(
      "team=t1&issue=i1&view=list&sort=updated&q=foo&project=P&priority=0&label=bug&assignee=u1",
    );
    expect(serializeKanbanSearch(new URLSearchParams("x=1"), {
      team: null, issue: null, view: "board", sort: "priority", q: "", project: null, priority: null, label: null, assignee: null,
    }).toString()).toBe("x=1");
  });

  it("falls back unknown enums, keeps priority 0, and rejects over-cap UTF-8", () => {
    expect(parseKanbanSearch(new URLSearchParams("view=nope&sort=nope&priority=9")).view).toBe("board");
    expect(parseKanbanSearch(new URLSearchParams("sort=status")).sort).toBe("status");
    expect(parseKanbanSearch(new URLSearchParams("priority=0")).priority).toBe(0);
    expect(parseKanbanSearch(new URLSearchParams(`q=${"é".repeat(129)}`)).q).toBe("");
    expect(parseKanbanSearch(new URLSearchParams(`team=${"é".repeat(128)}`)).team).toBe("é".repeat(128));
    expect(parseKanbanSearch(new URLSearchParams(`team=${"é".repeat(129)}`)).team).toBeNull();
  });

  it("encodes special characters through URLSearchParams", () => {
    const next = serializeKanbanSearch(new URLSearchParams(), {
      team: null, issue: null, view: "board", sort: "priority", q: "a b&c", project: null, priority: null, label: null, assignee: null,
    });
    expect(next.toString()).toBe("q=a+b%26c");
    expect(parseKanbanSearch(next).q).toBe("a b&c");
  });
});

describe("health and audit URL state", () => {
  it("omits default range and preset, keeps unrelated keys", () => {
    expect(serializeHealthSearch(new URLSearchParams("x=1"), { range: "24h" }).toString()).toBe("x=1");
    expect(parseHealthSearch(new URLSearchParams("range=7d")).range).toBe("7d");
    expect(parseHealthSearch(new URLSearchParams("range=nope")).range).toBe("24h");
    expect(serializeAuditSearch(new URLSearchParams("x=1"), { kind: "", preset: "30d" }).toString()).toBe("x=1");
    expect(parseAuditSearch(new URLSearchParams("kind=file-edit&preset=all"))).toEqual({ kind: "file-edit", preset: "all" });
    expect(parseAuditSearch(new URLSearchParams(`kind=${"é".repeat(129)}`)).kind).toBe("");
  });
});

describe("kanbanFilters", () => {
  it("carries the assignee value through, and null when absent (F5)", () => {
    const withAssignee = parseKanbanSearch(new URLSearchParams("assignee=u1"));
    expect(kanbanFilters(withAssignee).assignee).toBe("u1");
    const without = parseKanbanSearch(new URLSearchParams());
    expect(kanbanFilters(without).assignee).toBeNull();
  });
});

describe("normalizeKanbanSelection", () => {
  it("returns hard defaults for junk input", () => {
    for (const junk of [null, undefined, "x", 42, [], { priority: "high" }, { view: "grid" }]) {
      expect(normalizeKanbanSelection(junk)).toEqual({
        team: null, project: null, priority: null, label: null, assignee: null, sort: "priority", view: "board",
      });
    }
  });
  it("keeps each valid field independently, drops invalid ones independently", () => {
    expect(normalizeKanbanSelection({ team: "t1", sort: "nope", view: "list", priority: 9 })).toEqual({
      team: "t1", project: null, priority: null, label: null, assignee: null, sort: "priority", view: "list",
    });
  });
});

describe("hydrateFilters", () => {
  it("keeps a partial URL field and fills the rest from storage", () => {
    const stored = { team: "stored-team", project: "P", priority: 2, label: "bug", assignee: "u1", sort: "updated", view: "list" };
    const result = hydrateFilters(new URLSearchParams("team=url-team"), stored);
    expect(result.team).toBe("url-team");
    expect(result.project).toBe("P");
    expect(result.sort).toBe("updated");
    expect(result.view).toBe("list");
  });
  it("never applies a stored value when the URL already carries all seven fields", () => {
    const full = "team=t1&project=P1&priority=1&label=L1&assignee=u1&sort=created&view=list";
    const stored = { team: "other", project: "Other", priority: 3, label: "Other", assignee: "other", sort: "priority", view: "board" };
    expect(hydrateFilters(new URLSearchParams(full), stored)).toEqual(parseKanbanSearch(new URLSearchParams(full)));
  });
  it("?view=board (present, equal to the hard default) is kept, never overwritten by a different stored view", () => {
    const result = hydrateFilters(new URLSearchParams("view=board"), { view: "list" });
    expect(result.view).toBe("board");
  });
  it("?view=bogus (present but invalid) falls back to the hard default, never to a stored value", () => {
    const result = hydrateFilters(new URLSearchParams("view=bogus"), { view: "list" });
    expect(result.view).toBe("board");
  });
  it("corrupted storage falls back to hard defaults field-by-field, never throws", () => {
    expect(() => hydrateFilters(new URLSearchParams(), "not an object")).not.toThrow();
    expect(hydrateFilters(new URLSearchParams(), "not an object").team).toBeNull();
  });
});

describe("mergeKanbanPatch", () => {
  // KB-1: two synchronous patches (e.g. two FilterBar onChange calls before
  // router.replace commits) must both land when the second is composed onto
  // the first's result, not onto the original stale base.
  it("threads the previous merge result forward so both patched fields survive", () => {
    const base = parseKanbanSearch(new URLSearchParams());
    const afterFirst = mergeKanbanPatch(base, { project: "P1" });
    const afterSecond = mergeKanbanPatch(afterFirst, { priority: 2 });
    expect(afterSecond.project).toBe("P1");
    expect(afterSecond.priority).toBe(2);
  });
});

describe("boardEnabled", () => {
  it("a present urlTeam ignores hydrated entirely", () => {
    expect(boardEnabled("t1", false, "t1")).toBe(true);
    expect(boardEnabled("t1", false, null)).toBe(false);
  });
  it("an absent urlTeam is gated on hydrated even when a teamId already resolved (warm-cache fixture)", () => {
    expect(boardEnabled(null, false, "fallback-team")).toBe(false);
    expect(boardEnabled(null, true, "fallback-team")).toBe(true);
    expect(boardEnabled(null, true, null)).toBe(false);
  });
});
