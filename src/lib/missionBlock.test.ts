import { describe, expect, it } from "vitest";
import { milestoneDotStyle, selectMissionBlockState } from "./missionBlock";
import type { Envelope } from "@/lib/api";
import type { UserMission } from "@/server/db/missions";

const MISSION: UserMission = {
  id: 1, name: "Ship 6 phases tonight", goal: "Merge and review all six phases before sleep",
  statusLine: "Fases 1 e 2 merged, rodando review da fase 3", state: "active",
  startedAt: "2026-09-19T10:00:00.000Z", endedAt: null,
  milestones: [
    { title: "Fase 1", state: "done" },
    { title: "Fase 2", state: "pending" },
  ],
};

describe("selectMissionBlockState (spec §10, F1, F2, F3)", () => {
  it("hides the block when there is no query result yet (loading)", () => {
    expect(selectMissionBlockState(undefined, false)).toEqual({ kind: "hidden" });
  });
  it("hides the block when the source resolves to no active mission", () => {
    expect(selectMissionBlockState({ ok: true, data: null }, false)).toEqual({ kind: "hidden" });
  });
  it("F3: hides the block on an {ok:false} envelope", () => {
    const envelope: Envelope<UserMission | null> = { ok: false, error: "unavailable" };
    expect(selectMissionBlockState(envelope, false)).toEqual({ kind: "hidden" });
  });
  it("F2: hides the block when the query errored, even though stale active-mission data is still cached", () => {
    expect(selectMissionBlockState({ ok: true, data: MISSION }, true)).toEqual({ kind: "hidden" });
  });
  it("shows the mission with its own status line and a separate goal line when statusLine is non-empty", () => {
    expect(selectMissionBlockState({ ok: true, data: MISSION }, false)).toEqual({
      kind: "active", mission: MISSION, goalLine: MISSION.goal, statusText: MISSION.statusLine,
    });
  });
  it("F1: an empty statusLine fills the status slot with the goal, and the separate goal line is omitted (not duplicated)", () => {
    const fresh: UserMission = { ...MISSION, statusLine: "" };
    expect(selectMissionBlockState({ ok: true, data: fresh }, false)).toEqual({
      kind: "active", mission: fresh, goalLine: null, statusText: MISSION.goal,
    });
  });
});

describe("milestoneDotStyle (spec §10 dot rail — done/in-progress/pending)", () => {
  it("done renders a solid filled dot", () => {
    expect(milestoneDotStyle("done")).toEqual({ className: "h-2.5 w-2.5 flex-none rounded-full bg-brand", stateKey: "done" });
  });
  it("in-progress renders a half-filled dot, distinct from both done and pending", () => {
    const style = milestoneDotStyle("in-progress");
    expect(style.className).toBe("h-2.5 w-2.5 flex-none rounded-full border border-brand bg-brand/50");
    expect(style.stateKey).toBe("inProgress");
  });
  it("pending renders an outline-only dot, no fill", () => {
    expect(milestoneDotStyle("pending")).toEqual({ className: "h-2.5 w-2.5 flex-none rounded-full border border-line-strong bg-transparent", stateKey: "pending" });
  });
});
