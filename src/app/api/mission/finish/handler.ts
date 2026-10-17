import { NextResponse } from "next/server";
import { getDb } from "../../../../server/db";
import { finishMission, getActiveMission } from "../../../../server/db/missions";
import { insertEvent } from "../../../../server/db/workflows";
import { insertMutation } from "../../../../server/db/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";

export type MissionFinishRouteDeps = {
  getDb: typeof getDb;
  getActiveMission: typeof getActiveMission;
  finishMission: typeof finishMission;
  insertMutation: typeof insertMutation;
  insertEvent: typeof insertEvent;
};
const systemDeps: MissionFinishRouteDeps = { getDb, getActiveMission, finishMission, insertMutation, insertEvent };

export async function handleMissionFinish(req: Request, deps: MissionFinishRouteDeps = systemDeps) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  const read = await readJsonCapped(req, 256);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value as { outcome?: unknown } | null;
  const outcome = body?.outcome;
  // Cold review F2: outcome is validated explicitly, before any write — a missing or invalid
  // value used to fall through to the migration's `state` CHECK constraint (a raw DB failure);
  // now it never reaches the transaction at all.
  if (outcome !== "done" && outcome !== "cancelled") return NextResponse.json({ ok: false, error: "malformed outcome" });
  const db = deps.getDb();
  const active = deps.getActiveMission(db);
  if (!active) return NextResponse.json({ ok: false, error: "no-active-mission" });
  try {
    const mission = db.transaction(() => {
      const updated = deps.finishMission(db, active.id, outcome);
      // Cold review F6: finishMission now returns null on a zero-row UPDATE (no longer active —
      // a race between the check above and this call). That is the SAME "no active mission"
      // outcome the pre-check above already reports, so it gets the identical refusal code,
      // never mission-write-failed.
      if (!updated) throw new Error("no-active-mission");
      deps.insertMutation(db, { ts: new Date().toISOString(), kind: "mission-finish", mission_id: updated.id, outcome });
      deps.insertEvent(db, {
        run_id: null, project: "_mission", role: "lead", type: "mission-finished", source: "deterministic", emitter: "wrapper",
        payload: { missionId: updated.id, missionName: updated.name, outcome },
      });
      return updated;
    }).immediate();
    return NextResponse.json({ ok: true, data: mission });
  } catch (err) {
    if (err instanceof Error && err.message === "no-active-mission") return NextResponse.json({ ok: false, error: "no-active-mission" });
    return NextResponse.json({ ok: false, error: "mission-write-failed" });
  }
}
