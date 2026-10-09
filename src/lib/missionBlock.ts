import type { Envelope } from "@/lib/api";
import type { MilestoneState, UserMission } from "@/server/db/missions";

// Spec §10 + F1 (round 2): the block is either absent, or active with two derived
// text slots. `goalLine` is null exactly when statusLine is still "" (fresh start,
// §6 Decision 1) — in that case `statusText` carries the goal instead, and the
// separate goal line is omitted so the goal is never shown twice on screen.
export type MissionBlockState =
  | { kind: "hidden" }
  | { kind: "active"; mission: UserMission; goalLine: string | null; statusText: string };

// F2/F3 (plan review fc8a146c7553): `isError` hides the block even when TanStack
// Query still holds stale `data` from a prior successful fetch — an error must
// never keep a mission on screen. `data` is typed as the repo's own `Envelope<T>`
// union (reused, not redefined) — `!data.ok` (and the `data.data === null` "no
// active mission" case) both resolve to hidden, same as a network error.
export function selectMissionBlockState(
  data: Envelope<UserMission | null> | undefined,
  isError: boolean,
): MissionBlockState {
  if (isError || !data || !data.ok || data.data === null) return { kind: "hidden" };
  const mission = data.data;
  const hasStatus = mission.statusLine !== "";
  return {
    kind: "active",
    mission,
    goalLine: hasStatus ? mission.goal : null,
    statusText: hasStatus ? mission.statusLine : mission.goal,
  };
}

export type MilestoneDotStyle = { className: string; stateKey: "done" | "inProgress" | "pending" };

// F4 (plan review fc8a146c7553): reuses only the rounded-dot shape/size from
// ProjectCard.tsx:433-443; the three fill states below are this component's own
// classes (DS tokens) — filled = done, half-opacity fill = in-progress, outline-only
// = pending. No new color token: bg-brand/50 is Tailwind's own opacity modifier on
// the existing --color-brand token.
export function milestoneDotStyle(state: MilestoneState): MilestoneDotStyle {
  if (state === "done") return { className: "h-2.5 w-2.5 flex-none rounded-full bg-brand", stateKey: "done" };
  if (state === "in-progress") return { className: "h-2.5 w-2.5 flex-none rounded-full border border-brand bg-brand/50", stateKey: "inProgress" };
  return { className: "h-2.5 w-2.5 flex-none rounded-full border border-line-strong bg-transparent", stateKey: "pending" };
}
