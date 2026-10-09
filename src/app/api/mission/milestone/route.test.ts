import { describe, expect, it } from "vitest";
import { openDb } from "../../../../server/db";
import { getActiveMission, insertMission, setMilestoneState } from "../../../../server/db/missions";
import { insertEvent } from "../../../../server/db/workflows";
import { insertMutation } from "../../../../server/db/mutations";
import { handleMissionMilestone, type MissionMilestoneRouteDeps } from "./handler";

function depsWithDb(): MissionMilestoneRouteDeps {
  const db = openDb(":memory:");
  return { getDb: () => db, getActiveMission, setMilestoneState, insertMutation, insertEvent };
}
function postRequest(body: unknown): Request {
  return new Request("http://localhost/api/mission/milestone", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
}

describe("handleMissionMilestone", () => {
  it("resolves a milestone by 1-based index or by exact title, updating state and firing mission-status-updated", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "Ship it", goal: "g", milestones: ["Phase 1", "Phase 2"] });
    const res1 = await handleMissionMilestone(postRequest({ milestone: "1", state: "done" }), deps);
    expect((await res1.json()).data.milestones).toEqual([{ title: "Phase 1", state: "done" }, { title: "Phase 2", state: "pending" }]);
    const res2 = await handleMissionMilestone(postRequest({ milestone: "Phase 2", state: "in-progress" }), deps);
    expect((await res2.json()).data.milestones).toEqual([{ title: "Phase 1", state: "done" }, { title: "Phase 2", state: "in-progress" }]);
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-milestone'").get()).toEqual({ n: 2 });
    const evs = deps.getDb().prepare("SELECT payload FROM workflow_events ORDER BY id").all() as { payload: string }[];
    expect(JSON.parse(evs[0].payload)).toEqual({ missionId: 1, missionName: "Ship it", missionStatusLine: "", milestonesDone: 1, milestonesTotal: 2 });
  });

  it("refuses no-active-mission, unknown-milestone (out-of-range index and unmatched title), and malformed state", async () => {
    const deps = depsWithDb();
    expect(await (await handleMissionMilestone(postRequest({ milestone: "1", state: "done" }), deps)).json()).toEqual({ ok: false, error: "no-active-mission" });
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    expect(await (await handleMissionMilestone(postRequest({ milestone: "9", state: "done" }), deps)).json()).toEqual({ ok: false, error: "unknown-milestone" });
    expect(await (await handleMissionMilestone(postRequest({ milestone: "nope", state: "done" }), deps)).json()).toEqual({ ok: false, error: "unknown-milestone" });
    expect(await (await handleMissionMilestone(postRequest({ milestone: "1", state: "banana" }), deps)).json()).toEqual({ ok: false, error: "malformed state" });
  });

  it("a transaction failure (insertEvent throwing) rolls back the milestone update and the mutation insert", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    const failing: MissionMilestoneRouteDeps = { ...deps, insertEvent: () => { throw new Error("boom"); } };
    const res = await handleMissionMilestone(postRequest({ milestone: "1", state: "done" }), failing);
    expect(await res.json()).toEqual({ ok: false, error: "mission-write-failed" });
    expect(deps.getActiveMission(deps.getDb())?.milestones).toEqual([{ title: "m1", state: "pending" }]);
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-milestone'").get()).toEqual({ n: 0 });
  });

  it("a mission finishing between the active-mission read and the update yields no-active-mission, not mission-write-failed (cold review F2)", async () => {
    const deps = depsWithDb();
    insertMission(deps.getDb(), { name: "A", goal: "g", milestones: ["m1"] });
    const racing: MissionMilestoneRouteDeps = { ...deps, setMilestoneState: () => null };
    const res = await handleMissionMilestone(postRequest({ milestone: "1", state: "done" }), racing);
    expect(await res.json()).toEqual({ ok: false, error: "no-active-mission" });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-milestone'").get()).toEqual({ n: 0 });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 0 });
  });
});
