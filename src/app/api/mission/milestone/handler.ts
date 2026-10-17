import { NextResponse } from "next/server";
import { getDb } from "../../../../server/db";
import { getActiveMission, resolveMilestoneIndex, setMilestoneState } from "../../../../server/db/missions";
import { insertEvent } from "../../../../server/db/workflows";
import { insertMutation } from "../../../../server/db/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";

export type MissionMilestoneRouteDeps = {
  getDb: typeof getDb;
  getActiveMission: typeof getActiveMission;
  setMilestoneState: typeof setMilestoneState;
  insertMutation: typeof insertMutation;
  insertEvent: typeof insertEvent;
};
const systemDeps: MissionMilestoneRouteDeps = { getDb, getActiveMission, setMilestoneState, insertMutation, insertEvent };

export async function handleMissionMilestone(req: Request, deps: MissionMilestoneRouteDeps = systemDeps) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  const read = await readJsonCapped(req, 4096);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value as { milestone?: unknown; state?: unknown } | null;
  const milestoneRef = body?.milestone;
  const state = body?.state;
  if (typeof milestoneRef !== "string" || milestoneRef.length === 0) return NextResponse.json({ ok: false, error: "unknown-milestone" });
  // No CHECK constraint backs `state` (it lives inside the milestones JSON blob, not its own
  // column) — unlike `finish`'s `outcome`, this one IS validated here, since nothing else would.
  if (state !== "done" && state !== "in-progress") return NextResponse.json({ ok: false, error: "malformed state" });
  const db = deps.getDb();
  try {
    const mission = db.transaction(() => {
      // Cold review F2: read the active mission INSIDE the transaction, and map a null update
      // result (a concurrent finish between the read and the write) to the same "no-active-mission"
      // outcome as a missing mission up front — never mission-write-failed.
      const active = deps.getActiveMission(db);
      if (!active) throw new Error("no-active-mission");
      const index = resolveMilestoneIndex(active.milestones, milestoneRef);
      if (index === null) throw new Error("unknown-milestone");
      const updated = deps.setMilestoneState(db, active.id, index, state);
      if (!updated) throw new Error("no-active-mission");
      deps.insertMutation(db, { ts: new Date().toISOString(), kind: "mission-milestone", mission_id: updated.id, milestone: milestoneRef, state });
      deps.insertEvent(db, {
        run_id: null, project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper",
        payload: {
          missionId: updated.id, missionName: updated.name, missionStatusLine: updated.statusLine,
          milestonesDone: updated.milestones.filter((m) => m.state === "done").length, milestonesTotal: updated.milestones.length,
        },
      });
      return updated;
    }).immediate();
    return NextResponse.json({ ok: true, data: mission });
  } catch (err) {
    if (err instanceof Error && err.message === "no-active-mission") return NextResponse.json({ ok: false, error: "no-active-mission" });
    if (err instanceof Error && err.message === "unknown-milestone") return NextResponse.json({ ok: false, error: "unknown-milestone" });
    return NextResponse.json({ ok: false, error: "mission-write-failed" });
  }
}
