"use client";

import { useTranslations } from "next-intl";
import { useQuery } from "@tanstack/react-query";
import { fetchEnvelope, type Envelope } from "@/lib/api";
import { milestoneDotStyle, selectMissionBlockState } from "@/lib/missionBlock";
import type { UserMission } from "@/server/db/missions";

// Spec §10: reads its own small, independent query — NOT buildMission/MissionModel
// (spec §4a naming collision: this UserMission is Rafa's declared cross-project
// objective, unrelated to the existing per-project Mission dashboard model). Absent
// entirely with no active mission, still loading, or the source unreachable — this
// block never raises its own warning; that job belongs elsewhere (Global Constraints).
export function MissionBlock() {
  const t = useTranslations("missionBlock");
  const query = useQuery({
    queryKey: ["mission-block", "current"],
    queryFn: ({ signal }: { signal?: AbortSignal }) => fetchEnvelope<UserMission | null>("/api/mission/current", signal),
    refetchInterval: 10_000,
  });
  const state = selectMissionBlockState(query.data as Envelope<UserMission | null> | undefined, query.isError);
  if (state.kind === "hidden") return null;
  const { mission, goalLine, statusText } = state;
  return (
    <section className="flex flex-col gap-2 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <span className="text-[15px] font-semibold text-ink min-w-0 [overflow-wrap:anywhere]">{mission.name}</span>
      {goalLine ? <p className="text-[12px] text-muted min-w-0 [overflow-wrap:anywhere]">{goalLine}</p> : null}
      <p className="line-clamp-2 text-[13px] text-body-ink min-w-0 [overflow-wrap:anywhere]">{statusText}</p>
      <div role="list" aria-label={t("milestonesLabel")} className="flex flex-wrap gap-1.5">
        {mission.milestones.map((m, i) => {
          const style = milestoneDotStyle(m.state);
          return (
            <span
              key={i}
              role="listitem"
              title={m.title}
              aria-label={`${m.title}: ${t(`milestoneState.${style.stateKey}`)}`}
              className={style.className}
            />
          );
        })}
      </div>
    </section>
  );
}
