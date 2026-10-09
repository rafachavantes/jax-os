import { describe, expect, it } from "vitest";
import { openDb } from "../../../../server/db";
import { getActiveMission, insertMission } from "../../../../server/db/missions";
import { handleMissionCurrent, type MissionCurrentRouteDeps } from "./handler";

function depsWithDb(): MissionCurrentRouteDeps {
  const db = openDb(":memory:");
  return { getDb: () => db, getActiveMission };
}

describe("handleMissionCurrent", () => {
  it("returns {ok:true, data:null} with no active mission", async () => {
    const deps = depsWithDb();
    expect(await (await handleMissionCurrent(undefined, deps)).json()).toEqual({ ok: true, data: null });
  });

  it("returns the active mission once one is started", async () => {
    const deps = depsWithDb();
    const created = insertMission(deps.getDb(), { name: "Ship it", goal: "g", milestones: ["m1"] });
    expect(await (await handleMissionCurrent(undefined, deps)).json()).toEqual({ ok: true, data: created });
  });

  it("a DB failure returns {ok:false,error}, never a crash", async () => {
    const deps: MissionCurrentRouteDeps = { getDb: depsWithDb().getDb, getActiveMission: () => { throw new Error("db unavailable"); } };
    expect(await (await handleMissionCurrent(undefined, deps)).json()).toEqual({ ok: false, error: "db unavailable" });
  });
});
