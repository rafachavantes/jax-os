import { describe, expect, it } from "vitest";
import { openDb } from "../../../../server/db";
import { getActiveMission, insertMission } from "../../../../server/db/missions";
import { insertEvent } from "../../../../server/db/workflows";
import { insertMutation } from "../../../../server/db/mutations";
import { handleMissionFinish, type MissionFinishRouteDeps } from "./handler";
import { finishMission } from "../../../../server/db/missions";

function depsWithDb(): MissionFinishRouteDeps {
  const db = openDb(":memory:");
  return { getDb: () => db, getActiveMission, finishMission, insertMutation, insertEvent };
}
function postRequest(body: unknown): Request {
  return new Request("http://localhost/api/mission/finish", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
}

describe("handleMissionFinish", () => {
  it("done sets state/ended_at, audits one mutation, and fires mission-finished with the matching outcome", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "Ship it", goal: "g", milestones: ["m1"] });
    const res = await handleMissionFinish(postRequest({ outcome: "done" }), deps);
    const body = await res.json();
    expect(body.data.state).toBe("done");
    expect(body.data.endedAt).not.toBeNull();
    expect(deps.getActiveMission(deps.getDb())).toBeNull();
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-finish'").get()).toEqual({ n: 1 });
    const ev = deps.getDb().prepare("SELECT type, payload FROM workflow_events").get() as { type: string; payload: string };
    expect(ev.type).toBe("mission-finished");
    expect(JSON.parse(ev.payload)).toEqual({ missionId: 1, missionName: "Ship it", outcome: "done" });
  });

  it("cancel sets state cancelled with a matching outcome", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    const res = await handleMissionFinish(postRequest({ outcome: "cancelled" }), deps);
    expect((await res.json()).data.state).toBe("cancelled");
  });

  it("refuses no-active-mission", async () => {
    const deps = depsWithDb();
    expect(await (await handleMissionFinish(postRequest({ outcome: "done" }), deps)).json()).toEqual({ ok: false, error: "no-active-mission" });
  });

  it("refuses malformed outcome (invalid value or missing) before any write, leaving the mission active with no mutation or event (cold review F2)", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    for (const body of [{ outcome: "success" }, {}]) {
      expect(await (await handleMissionFinish(postRequest(body), deps)).json()).toEqual({ ok: false, error: "malformed outcome" });
    }
    expect(deps.getActiveMission(deps.getDb())?.state).toBe("active");
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-finish'").get()).toEqual({ n: 0 });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 0 });
  });

  it("maps a finishMission race (returns null though the pre-check saw an active mission) to no-active-mission, never a crash (cold review F6)", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    const failing: MissionFinishRouteDeps = { ...deps, finishMission: () => null };
    const res = await handleMissionFinish(postRequest({ outcome: "done" }), failing);
    expect(await res.json()).toEqual({ ok: false, error: "no-active-mission" });
  });

  it("a transaction failure (insertEvent throwing) rolls back the finish update and the mutation insert (cold review F7, spec §8 Atomicity)", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    const failing: MissionFinishRouteDeps = { ...deps, insertEvent: () => { throw new Error("boom"); } };
    const res = await handleMissionFinish(postRequest({ outcome: "done" }), failing);
    expect(await res.json()).toEqual({ ok: false, error: "mission-write-failed" });
    expect(deps.getActiveMission(deps.getDb())?.state).toBe("active");
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-finish'").get()).toEqual({ n: 0 });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 0 });
  });
});
