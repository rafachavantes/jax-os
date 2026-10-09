import { describe, expect, it } from "vitest";
import { openDb } from "../../../../server/db";
import { getActiveMission, insertMission, setMissionStatusLine } from "../../../../server/db/missions";
import { insertEvent } from "../../../../server/db/workflows";
import { insertMutation } from "../../../../server/db/mutations";
import { handleMissionStatus, type MissionStatusRouteDeps } from "./handler";

function depsWithDb(): MissionStatusRouteDeps {
  const db = openDb(":memory:");
  return { getDb: () => db, getActiveMission, setMissionStatusLine, insertMutation, insertEvent };
}
function postRequest(body: unknown): Request {
  return new Request("http://localhost/api/mission/status", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
}

describe("handleMissionStatus", () => {
  it("updates status_line, audits one mutation, and inserts one mission-status-updated event with the exact payload contract", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "Ship it", goal: "g", milestones: ["m1", "m2"] });
    const res = await handleMissionStatus(postRequest({ status_line: "phase 1 merged" }), deps);
    expect((await res.json()).data.statusLine).toBe("phase 1 merged");
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-status'").get()).toEqual({ n: 1 });
    const ev = deps.getDb().prepare("SELECT type, project, role, emitter, payload FROM workflow_events").get() as
      { type: string; project: string; role: string; emitter: string; payload: string };
    expect(ev).toMatchObject({ type: "mission-status-updated", project: "_mission", role: "lead", emitter: "wrapper" });
    expect(JSON.parse(ev.payload)).toEqual({ missionId: 1, missionName: "Ship it", missionStatusLine: "phase 1 merged", milestonesDone: 0, milestonesTotal: 2 });
  });

  it("refuses no-active-mission and malformed status (blank), touching neither missions nor workflow_events", async () => {
    const deps = depsWithDb();
    expect(await (await handleMissionStatus(postRequest({ status_line: "x" }), deps)).json()).toEqual({ ok: false, error: "no-active-mission" });
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    expect(await (await handleMissionStatus(postRequest({ status_line: "" }), deps)).json()).toEqual({ ok: false, error: "malformed status" });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 0 });
  });

  it("a transaction failure (insertEvent throwing) rolls back the status update and the mutation insert (spec §8 Atomicity, F3)", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    const failing: MissionStatusRouteDeps = { ...deps, insertEvent: () => { throw new Error("boom"); } };
    const res = await handleMissionStatus(postRequest({ status_line: "on track" }), failing);
    expect(await res.json()).toEqual({ ok: false, error: "mission-write-failed" });
    expect(deps.getActiveMission(deps.getDb())?.statusLine).toBe("");
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-status'").get()).toEqual({ n: 0 });
  });

  it("a mission finishing between the active-mission read and the update yields no-active-mission, not mission-write-failed (cold review F2)", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    const racing: MissionStatusRouteDeps = { ...deps, setMissionStatusLine: () => null };
    const res = await handleMissionStatus(postRequest({ status_line: "on track" }), racing);
    expect(await res.json()).toEqual({ ok: false, error: "no-active-mission" });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-status'").get()).toEqual({ n: 0 });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 0 });
  });
});
