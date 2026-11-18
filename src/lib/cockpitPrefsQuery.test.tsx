import { describe, expect, it } from "vitest";
import {
  applySnapshot, beginSave, cancelOrganize, draftDrop, draftMove, draftWidth, enterOrganize,
  initialOrganizeState, reloadOrganize, saveConflicted, saveFailed, saveSucceeded,
} from "./cockpitPrefsQuery";
import type { CockpitSnapshot, LayoutEntry } from "./cockpitLayout";

const rev1: CockpitSnapshot = {
  revision: 1, prScope: "mine",
  layout: [{ project: "A", width: "half" }, { project: "B", width: "third" }],
};
const rev2: CockpitSnapshot = {
  revision: 2, prScope: "mine",
  layout: [{ project: "B", width: "full" }, { project: "A", width: "half" }, { project: "C", width: "third" }],
};
const cards = [{ dir: "A" }, { dir: "B" }, { dir: "C" }];

describe("Organize draft survives polling (plan Task 2)", () => {
  it("enter captures the current revision and the rendered order/widths", () => {
    const state = enterOrganize(rev1, cards);
    expect(state.baseRevision).toBe(1);
    expect(state.draft).toEqual([
      { project: "A", width: "half" },
      { project: "B", width: "third" },
      { project: "C", width: "third" },
    ]);
  });

  it("a newer server snapshot never overwrites a dirty draft", () => {
    const dirty = draftMove(enterOrganize(rev1, cards), "C", -1);
    expect(applySnapshot(dirty, rev2)).toBe(dirty);
    // Cancel restores the latest successfully fetched snapshot and establishes it as baseline.
    const cancelled = cancelOrganize(rev2);
    expect(cancelled.draft).toBeNull();
    expect(cancelled.baseline).toBe(rev2);
    // A subsequent poll at revision N+1 updates the clean baseline without resurrecting a draft.
    expect(applySnapshot(cancelled, { ...rev2, revision: 3 }).baseline).toEqual({ ...rev2, revision: 3 });
    expect(applySnapshot(cancelled, rev2).draft).toBeNull();
  });

  it("move/drop/width edits mark the draft dirty and only touch the draft", () => {
    let state = enterOrganize(rev1, cards);
    state = draftMove(state, "B", -1);
    expect(state.draft!.map((e) => e.project)).toEqual(["B", "A", "C"]);
    state = draftDrop(state, "C", "B");
    expect(state.draft!.map((e) => e.project)).toEqual(["C", "B", "A"]);
    state = draftWidth(state, "B", "full");
    expect(state.draft!.find((e) => e.project === "B")?.width).toBe("full");
    expect(state.status).toBe("dirty");
  });

  it("a stale save is a conflict that keeps the draft and offers the latest snapshot baseline", () => {
    const saving = beginSave(enterOrganize(rev1, cards));
    const conflicted = saveConflicted(saving, rev2);
    expect(conflicted.status).toBe("conflict");
    expect(conflicted.draft).not.toBeNull();
    expect(conflicted.baseline).toBe(rev2);
    // Reload replaces the draft from the latest server state (new revision/order/widths).
    const reloaded = reloadOrganize(rev2, cards);
    expect(reloaded.baseRevision).toBe(2);
    expect(reloaded.draft).toEqual([
      { project: "B", width: "full" },
      { project: "A", width: "half" },
      { project: "C", width: "third" },
    ]);
  });

  it("any other save failure keeps the draft and names the error", () => {
    const failed = saveFailed(beginSave(enterOrganize(rev1, cards)), "network");
    expect(failed.status).toBe("error");
    expect(failed.error).toBe("network");
    expect(failed.draft).toEqual(enterOrganize(rev1, cards).draft);
  });

  it("an actual success applies only the successful snapshot and clears the draft", () => {
    const done = saveSucceeded(rev2);
    expect(done).toEqual({ ...initialOrganizeState, baseline: rev2 });
  });

  it("an unrelated PR-scope change never touches a dirty layout draft", () => {
    const dirty = enterOrganize(rev1, cards);
    const scopeOnly: CockpitSnapshot = { ...rev1, prScope: "all" };
    expect(applySnapshot(dirty, scopeOnly)).toBe(dirty);
    // In view mode it updates the baseline and the selected scope comes from the server snapshot.
    expect(applySnapshot(cancelOrganize(rev1), scopeOnly).baseline?.prScope).toBe("all");
  });

  it("draft operations are inert outside Organize", () => {
    const entry: LayoutEntry = { project: "A", width: "third" };
    expect(draftDrop(initialOrganizeState, "A", "B")).toBe(initialOrganizeState);
    expect(draftWidth(initialOrganizeState, "A", "full")).toBe(initialOrganizeState);
    expect(initialOrganizeState.draft).toBeNull();
    expect(entry.width).toBe("third");
  });
});
