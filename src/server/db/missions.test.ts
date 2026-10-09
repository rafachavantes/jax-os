import { describe, expect, it } from "vitest";
import { openDb } from "./index";
import {
  finishMission, getActiveMission, insertMission, resolveMilestoneIndex, setMilestoneState, setMissionStatusLine,
  type MissionMilestone,
} from "./missions";

const NOW = new Date("2026-09-19T12:00:00.000Z");

describe("insertMission / getActiveMission (spec §6 Decision 1, §14 Risk 2)", () => {
  it("initializes status_line to '' and every milestone to pending, readable back via getActiveMission", () => {
    const db = openDb(":memory:");
    const created = insertMission(db, { name: "Ship it", goal: "6 phases tonight", milestones: ["Phase 1", "Phase 2"] }, NOW);
    expect(created).toMatchObject({
      name: "Ship it", goal: "6 phases tonight", statusLine: "", state: "active", endedAt: null,
      milestones: [{ title: "Phase 1", state: "pending" }, { title: "Phase 2", state: "pending" }],
    });
    expect(created.startedAt).toBe(NOW.toISOString());
    expect(getActiveMission(db)).toEqual(created);
  });

  it("getActiveMission returns null with no active mission", () => {
    const db = openDb(":memory:");
    expect(getActiveMission(db)).toBeNull();
  });

  it("the missions_one_active partial index refuses a second concurrent active insert", () => {
    const db = openDb(":memory:");
    insertMission(db, { name: "A", goal: "g", milestones: ["m1"] }, NOW);
    expect(() => insertMission(db, { name: "B", goal: "g", milestones: ["m1"] }, NOW)).toThrow();
  });
});

describe("resolveMilestoneIndex (spec §6 Decision 5)", () => {
  const milestones: MissionMilestone[] = [
    { title: "Phase 1", state: "pending" }, { title: "3", state: "pending" }, { title: "Phase 3", state: "pending" },
  ];
  it("a numeric-shaped ref resolves as a 1-based index", () => {
    expect(resolveMilestoneIndex(milestones, "1")).toBe(0);
    expect(resolveMilestoneIndex(milestones, "3")).toBe(2); // the numeric REF "3" is an index, never matched against the title "3"
  });
  it("a non-numeric ref resolves as an exact, case-sensitive, trimmed title match", () => {
    expect(resolveMilestoneIndex(milestones, "Phase 1")).toBe(0);
    expect(resolveMilestoneIndex(milestones, "phase 1")).toBeNull();
  });
  it("returns null for an out-of-range index or an unmatched title", () => {
    expect(resolveMilestoneIndex(milestones, "0")).toBeNull(); // "0" doesn't match MISSION_MILESTONE_NUMERIC_RE either
    expect(resolveMilestoneIndex(milestones, "9")).toBeNull();
    expect(resolveMilestoneIndex(milestones, "nope")).toBeNull();
  });
});

describe("setMissionStatusLine / setMilestoneState / finishMission (spec §6, §12)", () => {
  it("setMissionStatusLine updates the row and round-trips through getActiveMission", () => {
    const db = openDb(":memory:");
    const created = insertMission(db, { name: "A", goal: "g", milestones: ["m1"] }, NOW);
    const updated = setMissionStatusLine(db, created.id, "phase 1 merged");
    expect(updated?.statusLine).toBe("phase 1 merged");
    expect(getActiveMission(db)?.statusLine).toBe("phase 1 merged");
  });

  it("setMissionStatusLine returns null for a non-existent mission id", () => {
    const db = openDb(":memory:");
    expect(setMissionStatusLine(db, 999, "x")).toBeNull();
  });

  it("setMilestoneState flips exactly the targeted 0-based index, round-tripping the JSON array", () => {
    const db = openDb(":memory:");
    const created = insertMission(db, { name: "A", goal: "g", milestones: ["m1", "m2"] }, NOW);
    const afterFirst = setMilestoneState(db, created.id, 0, "in-progress");
    expect(afterFirst?.milestones).toEqual([{ title: "m1", state: "in-progress" }, { title: "m2", state: "pending" }]);
    const afterSecond = setMilestoneState(db, created.id, 1, "done");
    expect(afterSecond?.milestones).toEqual([{ title: "m1", state: "in-progress" }, { title: "m2", state: "done" }]);
  });

  it("setMilestoneState returns null for an out-of-range index or a non-existent mission id", () => {
    const db = openDb(":memory:");
    const created = insertMission(db, { name: "A", goal: "g", milestones: ["m1"] }, NOW);
    expect(setMilestoneState(db, created.id, 5, "done")).toBeNull();
    expect(setMilestoneState(db, 999, 0, "done")).toBeNull();
  });

  it("finishMission sets state/ended_at, clears the active slot, and allows a new start", () => {
    const db = openDb(":memory:");
    const created = insertMission(db, { name: "A", goal: "g", milestones: ["m1"] }, NOW);
    const finished = finishMission(db, created.id, "done", NOW);
    expect(finished).toMatchObject({ state: "done", endedAt: NOW.toISOString() });
    expect(getActiveMission(db)).toBeNull();
    expect(() => insertMission(db, { name: "B", goal: "g", milestones: ["m1"] }, NOW)).not.toThrow();
  });

  it("finishMission returns null for a non-existent id or a mission that is no longer active", () => {
    const db = openDb(":memory:");
    expect(finishMission(db, 999, "done", NOW)).toBeNull();
    const created = insertMission(db, { name: "A", goal: "g", milestones: ["m1"] }, NOW);
    finishMission(db, created.id, "cancelled", NOW);
    expect(finishMission(db, created.id, "done", NOW)).toBeNull(); // already finished, no longer active
  });

  it("finishMission returns null on a same-outcome repeat call, not the stale already-finished row (cold review F6)", () => {
    const db = openDb(":memory:");
    const created = insertMission(db, { name: "A", goal: "g", milestones: ["m1"] }, NOW);
    finishMission(db, created.id, "done", NOW);
    expect(finishMission(db, created.id, "done", NOW)).toBeNull();
  });
});
