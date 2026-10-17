"use client";

import { useTranslations } from "next-intl";
import type { MissionTimelineEntry } from "@/lib/mission";

// F3 (diff review): the expanded timeline (ProjectCard.tsx) and EventTicker rendered different
// subsets of the same MissionTimelineEntry — buildTicker already spreads the whole entry into a
// ticker item, so both places share this one component now and can't drift again.
export function TimelineEntryText({ entry }: { entry: MissionTimelineEntry }) {
  const t = useTranslations("mission");
  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <span className="text-body-ink">{entry.type}</span>
      {entry.pane ? <span>{entry.pane}</span> : null}
      {entry.capsuleStatus ? <span className="text-body-ink">{entry.capsuleStatus}</span> : null}
      {entry.outcome ? <span className="text-body-ink">{entry.outcome}</span> : null}
      {entry.contractStatus ? <span>{entry.contractStatus}</span> : null}
      {entry.stage ? <span>{t(`lastRun.stage.${entry.stage}`)}</span> : null}
      {entry.diagnostic ? <span className="text-body-ink">{entry.diagnostic}</span> : null}
    </span>
  );
}
