import type Database from "better-sqlite3";
import { MISSION_MILESTONE_NUMERIC_RE } from "../../lib/workflow";

// Spec §4a: these types are a NEW, unrelated concept from src/lib/mission.ts's
// Mission/MissionModel/MissionCard/MissionState (the existing per-project dashboard model).
// Never import from src/lib/mission.ts here, and never rename these to bare `Mission*` — the
// collision is the whole reason this spec put them in their own file.
export type MilestoneState = "pending" | "in-progress" | "done";
export type MissionMilestone = { title: string; state: MilestoneState };
export type UserMissionState = "active" | "done" | "cancelled";
export type UserMission = {
  id: number;
  name: string;
  goal: string;
  statusLine: string;
  state: UserMissionState;
  startedAt: string;
  endedAt: string | null;
  milestones: MissionMilestone[];
};

const COLS = "id, name, goal, status_line, state, started_at, ended_at, milestones";
type RawRow = {
  id: number; name: string; goal: string; status_line: string; state: UserMissionState;
  started_at: string; ended_at: string | null; milestones: string;
};

function rowOf(r: RawRow): UserMission {
  return {
    id: r.id, name: r.name, goal: r.goal, statusLine: r.status_line, state: r.state,
    startedAt: r.started_at, endedAt: r.ended_at, milestones: JSON.parse(r.milestones) as MissionMilestone[],
  };
}

// Spec §6 Decision 3: at most one row can ever be 'active' (missions_one_active, migration 11)
// — this is the read half of that invariant; every route uses it as its pre-check.
export function getActiveMission(db: Database.Database): UserMission | null {
  const row = db.prepare(`SELECT ${COLS} FROM missions WHERE state = 'active' LIMIT 1`).get() as RawRow | undefined;
  return row ? rowOf(row) : null;
}

// Spec §6 Decision 1: status_line starts '' and every milestone starts 'pending'. The caller
// (the /api/mission route) has already validated name/goal/milestone shape and count — this is
// a plain insert, not a second validation pass.
export function insertMission(
  db: Database.Database,
  input: { name: string; goal: string; milestones: string[] },
  now = new Date(),
): UserMission {
  const milestones: MissionMilestone[] = input.milestones.map((title) => ({ title, state: "pending" }));
  const startedAt = now.toISOString();
  const r = db.prepare(
    `INSERT INTO missions (name, goal, status_line, state, started_at, ended_at, milestones)
     VALUES (?, ?, '', 'active', ?, NULL, ?)`,
  ).run(input.name, input.goal, startedAt, JSON.stringify(milestones));
  return {
    id: Number(r.lastInsertRowid), name: input.name, goal: input.goal, statusLine: "",
    state: "active", startedAt, endedAt: null, milestones,
  };
}

export function setMissionStatusLine(db: Database.Database, id: number, statusLine: string): UserMission | null {
  db.prepare("UPDATE missions SET status_line = ? WHERE id = ? AND state = 'active'").run(statusLine, id);
  const row = db.prepare(`SELECT ${COLS} FROM missions WHERE id = ? AND state = 'active'`).get(id) as RawRow | undefined;
  return row ? rowOf(row) : null;
}

// Spec §6 Decision 5: a whole-string positive integer resolves as a 1-based index into the
// array; otherwise an exact, case-sensitive, trimmed title match. Returns the 0-based array
// index, or null if neither resolves.
export function resolveMilestoneIndex(milestones: MissionMilestone[], ref: string): number | null {
  if (MISSION_MILESTONE_NUMERIC_RE.test(ref)) {
    const i = Number(ref) - 1;
    return i >= 0 && i < milestones.length ? i : null;
  }
  const i = milestones.findIndex((m) => m.title === ref);
  return i === -1 ? null : i;
}

export function setMilestoneState(db: Database.Database, id: number, index: number, state: MilestoneState): UserMission | null {
  const row = db.prepare(`SELECT ${COLS} FROM missions WHERE id = ? AND state = 'active'`).get(id) as RawRow | undefined;
  if (!row) return null;
  const milestones = JSON.parse(row.milestones) as MissionMilestone[];
  if (!milestones[index]) return null;
  milestones[index] = { ...milestones[index], state };
  db.prepare("UPDATE missions SET milestones = ? WHERE id = ?").run(JSON.stringify(milestones), id);
  return { ...rowOf(row), milestones };
}

export function finishMission(db: Database.Database, id: number, outcome: "done" | "cancelled", now = new Date()): UserMission | null {
  const endedAt = now.toISOString();
  // Cold review F6: the UPDATE's own affected-row count is the ONLY signal for "was there an
  // active row to finish" — comparing the post-write `state` column to `outcome` is wrong: a
  // same-outcome repeat call (finish "done" twice) would find the row already sitting at
  // `outcome` from the FIRST call and wrongly return it as freshly finished.
  const result = db.prepare("UPDATE missions SET state = ?, ended_at = ? WHERE id = ? AND state = 'active'").run(outcome, endedAt, id);
  if (result.changes === 0) return null;
  const row = db.prepare(`SELECT ${COLS} FROM missions WHERE id = ?`).get(id) as RawRow | undefined;
  return row ? rowOf(row) : null;
}
