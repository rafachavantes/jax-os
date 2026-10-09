import { describe, expect, it } from "vitest";
import { openDb } from "../../../server/db";
import { getActiveMission, insertMission } from "../../../server/db/missions";
import { insertMutation } from "../../../server/db/mutations";
import { handleMissionStart, type MissionRouteDeps } from "./handler";

function depsWithDb(): MissionRouteDeps {
  const db = openDb(":memory:");
  return { getDb: () => db, getActiveMission, insertMission, insertMutation };
}
function postRequest(body: unknown, headers: Record<string, string> = { "content-type": "application/json" }): Request {
  return new Request("http://localhost/api/mission", { method: "POST", headers, body: JSON.stringify(body) });
}

describe("handleMissionStart", () => {
  it("blocks a cross-site request before touching the db", async () => {
    const deps = depsWithDb();
    const req = postRequest({ name: "A", goal: "g", milestones: ["m1"] }, { "sec-fetch-site": "cross-site", "content-type": "application/json" });
    expect(await (await handleMissionStart(req, deps)).json()).toEqual({ ok: false, error: "cross-site request blocked" });
    expect(deps.getActiveMission(deps.getDb())).toBeNull();
  });

  it("creates the mission, audits exactly one mission-start mutation, and fires no workflow_events row at all", async () => {
    const deps = depsWithDb();
    const res = await handleMissionStart(postRequest({ name: "Ship it", goal: "6 phases tonight", milestones: ["Phase 1", "Phase 2"] }), deps);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data).toMatchObject({ name: "Ship it", goal: "6 phases tonight", state: "active", statusLine: "" });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-start'").get()).toEqual({ n: 1 });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 0 });
  });

  it("refuses mission-active with no new row and no new mutation", async () => {
    const deps = depsWithDb();
    await handleMissionStart(postRequest({ name: "A", goal: "g", milestones: ["m1"] }), deps);
    const res = await handleMissionStart(postRequest({ name: "B", goal: "g2", milestones: ["m1"] }), deps);
    expect(await res.json()).toEqual({ ok: false, error: "mission-active" });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM missions").get()).toEqual({ n: 1 });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-start'").get()).toEqual({ n: 1 });
  });

  it("refuses malformed name, malformed goal, too-many-milestones, and malformed milestone (empty/numeric-only/duplicate/whitespace-canonicalized duplicate)", async () => {
    const deps = depsWithDb();
    const cases: [unknown, string][] = [
      [{ name: "", goal: "g", milestones: ["m1"] }, "malformed name"],
      [{ name: "A", goal: "x".repeat(281), milestones: ["m1"] }, "malformed goal"],
      [{ name: "A", goal: "g", milestones: Array.from({ length: 13 }, (_, i) => `m${i}`) }, "too-many-milestones"],
      [{ name: "A", goal: "g", milestones: ["3"] }, "malformed milestone"],
      [{ name: "A", goal: "g", milestones: ["Phase 1", "phase 1"] }, "malformed milestone"],
      // cold review F5: "Phase  1" (two inner spaces) must canonicalize to the same key as
      // "Phase 1" — trim+lowercase alone would miss this, only whitespace collapsing catches it.
      [{ name: "A", goal: "g", milestones: ["Phase 1", "Phase  1"] }, "malformed milestone"],
      [{ name: "A", goal: "g", milestones: [] }, "malformed milestone"],
    ];
    for (const [body, error] of cases) {
      expect(await (await handleMissionStart(postRequest(body), deps)).json()).toEqual({ ok: false, error });
    }
    expect(deps.getActiveMission(deps.getDb())).toBeNull();
  });

  it("a transaction failure rolls back the row insert too, and returns mission-write-failed, never a raw error", async () => {
    const deps = depsWithDb();
    const failing: MissionRouteDeps = { ...deps, insertMutation: () => { throw new Error("boom"); } };
    const res = await handleMissionStart(postRequest({ name: "A", goal: "g", milestones: ["m1"] }), failing);
    expect(await res.json()).toEqual({ ok: false, error: "mission-write-failed" });
    expect(deps.getActiveMission(deps.getDb())).toBeNull();
  });

  it("maps a genuine missions_one_active unique-constraint violation to mission-active, never mission-write-failed (cold review F1)", async () => {
    const deps = depsWithDb();
    // Simplest reproduction of the race the partial unique index guards (spec §14 Risk 2): the
    // pre-check and the insert now run inside the SAME transaction (Task 5's implementation), so
    // the only way to observe the index actually firing here is a stubbed insertMission throwing
    // the exact shape better-sqlite3 throws for it — a real concurrent second connection racing
    // the same `.immediate()` transaction is not reproducible in a single in-process :memory: db.
    const constraintError = Object.assign(new Error("UNIQUE constraint failed: missions.state"), { code: "SQLITE_CONSTRAINT_UNIQUE" });
    const failing: MissionRouteDeps = { ...deps, insertMission: () => { throw constraintError; } };
    const res = await handleMissionStart(postRequest({ name: "A", goal: "g", milestones: ["m1"] }), failing);
    expect(await res.json()).toEqual({ ok: false, error: "mission-active" });
  });
});
