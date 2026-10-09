import { NextResponse } from "next/server";
import { getDb } from "../../../../server/db";
import { getActiveMission, setMissionStatusLine } from "../../../../server/db/missions";
import { insertEvent } from "../../../../server/db/workflows";
import { insertMutation } from "../../../../server/db/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { LIMITS, validMissionText } from "../../../../lib/workflow";

export type MissionStatusRouteDeps = {
  getDb: typeof getDb;
  getActiveMission: typeof getActiveMission;
  setMissionStatusLine: typeof setMissionStatusLine;
  insertMutation: typeof insertMutation;
  insertEvent: typeof insertEvent;
};
const systemDeps: MissionStatusRouteDeps = { getDb, getActiveMission, setMissionStatusLine, insertMutation, insertEvent };

export async function handleMissionStatus(req: Request, deps: MissionStatusRouteDeps = systemDeps) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  const read = await readJsonCapped(req, 4096);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value as { status_line?: unknown } | null;
  const statusLine = body?.status_line;
  // status's OWN incoming text must not be blank (spec §7 malformed status) — validMissionText
  // requires non-empty; the STORED value's own '' case (Task 4) is a separate, later concern.
  if (!validMissionText(statusLine, LIMITS.missionStatusLine)) return NextResponse.json({ ok: false, error: "malformed status" });
  const db = deps.getDb();
  try {
    const mission = db.transaction(() => {
      // Cold review F2: read the active mission INSIDE the transaction, and map a null update
      // result (a concurrent finish between the read and the write) to the same "no-active-mission"
      // outcome as a missing mission up front — never mission-write-failed.
      const active = deps.getActiveMission(db);
      if (!active) throw new Error("no-active-mission");
      const updated = deps.setMissionStatusLine(db, active.id, statusLine);
      if (!updated) throw new Error("no-active-mission");
      deps.insertMutation(db, { ts: new Date().toISOString(), kind: "mission-status", mission_id: updated.id, status_line: updated.statusLine });
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
    return NextResponse.json({ ok: false, error: "mission-write-failed" });
  }
}
