import { describe, expect, it } from "vitest";
import {
  applySavedOrder, dropEntry, isPrScope, mergeSavedLayout, moveEntry, normalizeProjectKey,
  parseLayoutEntries, setEntryWidth, visibleLayout, type LayoutEntry,
} from "./cockpitLayout";

const a: LayoutEntry = { project: "A", width: "third" };
const x: LayoutEntry = { project: "x", width: "half" };
const b: LayoutEntry = { project: "B", width: "third" };
const y: LayoutEntry = { project: "y", width: "full" };

describe("normalizeProjectKey (spec AC3/7, plan Task 1)", () => {
  it("accepts ordinary and nested relative keys", () => {
    expect(normalizeProjectKey("jax-os")).toBe("jax-os");
    expect(normalizeProjectKey("company/nested.repo")).toBe("company/nested.repo");
  });

  it("rejects empty, absolute, control/NUL, backslash, dot and empty segments, and over 256 chars", () => {
    for (const bad of ["", "/abs", "a\\b", "a\u0000b", "..", "a/../b", "./a", "a//b", "a/", "a/./b", "x".repeat(257)]) {
      expect(normalizeProjectKey(bad)).toBeNull();
    }
    expect(normalizeProjectKey(42)).toBeNull();
    expect(normalizeProjectKey("x".repeat(256))).toBe("x".repeat(256));
  });
});

describe("parseLayoutEntries (plan Task 1)", () => {
  it("accepts a bounded unique width list", () => {
    expect(parseLayoutEntries([a, x, b])).toEqual([a, x, b]);
  });

  it("rejects duplicates, bad widths, non-objects, extra keys, and overflow past 500", () => {
    expect(parseLayoutEntries([a, a])).toBeNull();
    expect(parseLayoutEntries([{ project: "A", width: "wide" }])).toBeNull();
    expect(parseLayoutEntries([{ project: "A" }])).toBeNull();
    expect(parseLayoutEntries([{ project: "A", width: "half", extra: 1 }])).toBeNull();
    expect(parseLayoutEntries([null])).toBeNull();
    expect(parseLayoutEntries("nope")).toBeNull();
    const over = Array.from({ length: 501 }, (_, i) => ({ project: `p${i}`, width: "third" as const }));
    expect(parseLayoutEntries(over)).toBeNull();
    expect(parseLayoutEntries(over.slice(0, 500))).toHaveLength(500);
  });
});

describe("mergeSavedLayout (plan Task 1 — absent anchors)", () => {
  it("reorders submitted entries and places absent ones after their anchors (A,x,B,y + B,A)", () => {
    expect(mergeSavedLayout([a, x, b, y], [b, a])).toEqual([b, a, x, y]);
  });

  it("keeps absent-first entries at the end, preserving their relative order", () => {
    const sub: LayoutEntry = { project: "B", width: "half" };
    expect(mergeSavedLayout([x, y, a, b], [sub, a])).toEqual([sub, a, x, y]);
  });

  it("emits consecutive absent entries behind an already-emitted anchor", () => {
    expect(mergeSavedLayout([a, x, y, b], [b, a])).toEqual([b, a, x, y]);
  });

  it("an empty submission preserves the complete previous order", () => {
    expect(mergeSavedLayout([a, x, b, y], [])).toEqual([a, x, b, y]);
  });

  it("absent widths are retained and new submitted widths win", () => {
    const subA: LayoutEntry = { project: "A", width: "full" };
    const subC: LayoutEntry = { project: "C", width: "half" };
    // x's anchor is A, so it is emitted right after A; C follows in submitted order.
    expect(mergeSavedLayout([a, x], [subA, subC])).toEqual([subA, x, subC]);
  });

  it("an absent head never lets a later absent entry overtake it", () => {
    // x's anchor is null (nothing submitted preceded it in the prior order) -> x waits at the
    // end even though y's anchor A was emitted; y must not jump ahead of x.
    expect(mergeSavedLayout([x, a, y], [a])).toEqual([a, x, y]);
  });

  it("without a prior layout the submission is the layout", () => {
    expect(mergeSavedLayout(null, [b, a])).toEqual([b, a]);
  });
});

describe("ordering and draft operations", () => {
  it("saved order wins, new projects append dir-ascending, no override keeps auto order", () => {
    const cards = [{ dir: "z" }, { dir: "A" }, { dir: "new-b" }, { dir: "new-a" }];
    expect(applySavedOrder(cards, null).map((c) => c.dir)).toEqual(["z", "A", "new-b", "new-a"]);
    expect(applySavedOrder(cards, [{ project: "A", width: "full" }, { project: "z", width: "third" }]).map((c) => c.dir))
      .toEqual(["A", "z", "new-a", "new-b"]);
  });

  it("visibleLayout captures rendered order and saved widths", () => {
    expect(visibleLayout([{ dir: "A" }, { dir: "z" }], [{ project: "z", width: "half" }]))
      .toEqual([{ project: "A", width: "third" }, { project: "z", width: "half" }]);
  });

  it("move and drop reorder through the same entries; boundaries are no-ops", () => {
    expect(moveEntry([a, x, b], "x", -1)).toEqual([x, a, b]);
    expect(moveEntry([a, x, b], "A", -1)).toEqual([a, x, b]);
    expect(moveEntry([a, x, b], "B", 1)).toEqual([a, x, b]);
    expect(dropEntry([a, x, b], "B", "A")).toEqual([b, a, x]);
    expect(dropEntry([a, x, b], "nope", "A")).toEqual([a, x, b]);
  });

  it("width changes only the selected entry", () => {
    expect(setEntryWidth([a, x], "x", "full")).toEqual([a, { project: "x", width: "full" }]);
  });

  it("pr scope accepts only mine/all", () => {
    expect(isPrScope("mine")).toBe(true);
    expect(isPrScope("all")).toBe(true);
    expect(isPrScope("team")).toBe(false);
  });
});
