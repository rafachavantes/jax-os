import { NextResponse } from "next/server";
import { getDb } from "../../../server/db";
import { getActiveMission, insertMission } from "../../../server/db/missions";
import { insertMutation } from "../../../server/db/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../server/api";
import { LIMITS, MISSION_MILESTONE_NUMERIC_RE, validMissionText } from "../../../lib/workflow";

export type MissionRouteDeps = {
  getDb: typeof getDb;
  getActiveMission: typeof getActiveMission;
  insertMission: typeof insertMission;
  insertMutation: typeof insertMutation;
};
const systemDeps: MissionRouteDeps = { getDb, getActiveMission, insertMission, insertMutation };

export async function handleMissionStart(req: Request, deps: MissionRouteDeps = systemDeps) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  const read = await readJsonCapped(req, 4096);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value as { name?: unknown; goal?: unknown; milestones?: unknown } | null;
  const name = body?.name;
  const goal = body?.goal;
  const milestones = body?.milestones;
  if (!validMissionText(name, LIMITS.missionName)) return NextResponse.json({ ok: false, error: "malformed name" });
  if (!validMissionText(goal, LIMITS.missionGoal)) return NextResponse.json({ ok: false, error: "malformed goal" });
  if (!Array.isArray(milestones) || milestones.length < 1) return NextResponse.json({ ok: false, error: "malformed milestone" });
  if (milestones.length > LIMITS.missionMilestonesMax) return NextResponse.json({ ok: false, error: "too-many-milestones" });
  const seen = new Set<string>();
  for (const m of milestones) {
    if (!validMissionText(m, LIMITS.milestoneTitle)) return NextResponse.json({ ok: false, error: "malformed milestone" });
    if (MISSION_MILESTONE_NUMERIC_RE.test(m)) return NextResponse.json({ ok: false, error: "malformed milestone" });
    // Cold review F5: canonicalize INTERNAL whitespace too, not just the ends — validMissionText
    // already refuses leading/trailing whitespace, but "Phase  1" (two inner spaces) and
    // "Phase 1" would otherwise hash to different keys and both be accepted as distinct titles.
    const key = m.trim().toLowerCase().replace(/\s+/g, " ");
    if (seen.has(key)) return NextResponse.json({ ok: false, error: "malformed milestone" });
    seen.add(key);
  }
  const db = deps.getDb();
  try {
    const mission = db.transaction(() => {
      // Cold review F1: the active-mission check moves INSIDE this same immediate transaction,
      // right next to the insert — a check-then-act gap here is exactly the race
      // `missions_one_active` (Task 2) exists to catch. `.immediate()` takes the write lock
      // before this callback runs, so nothing can insert an active row between this check and
      // the INSERT below.
      if (deps.getActiveMission(db)) throw new Error("mission-active");
      const created = deps.insertMission(db, { name, goal, milestones: milestones as string[] });
      deps.insertMutation(db, { ts: new Date().toISOString(), kind: "mission-start", mission_id: created.id, name: created.name, goal: created.goal, milestones: created.milestones });
      return created;
    }).immediate();
    return NextResponse.json({ ok: true, data: mission });
  } catch (err) {
    // Two ways this same "there's already an active mission" fact can surface: the pre-check
    // above throwing its own sentinel, or a genuine missions_one_active unique-index violation
    // if the pre-check somehow missed it (a second connection, or insertMission racing outside
    // this process) — better-sqlite3 tags that one `SQLITE_CONSTRAINT_UNIQUE`, the only unique
    // index this table has. Both map to `mission-active`, never the generic `mission-write-failed`.
    if (err instanceof Error && err.message === "mission-active") return NextResponse.json({ ok: false, error: "mission-active" });
    if ((err as { code?: string })?.code === "SQLITE_CONSTRAINT_UNIQUE") return NextResponse.json({ ok: false, error: "mission-active" });
    return NextResponse.json({ ok: false, error: "mission-write-failed" });
  }
}
