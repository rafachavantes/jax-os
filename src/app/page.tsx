"use client";

import { useTranslations } from "next-intl";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useMission } from "@/lib/useMission";
import { useGeneralSettings } from "@/lib/settingsQuery";
import { fetchEnvelope } from "@/lib/api";
import { buildMission, buildTicker, summarizeBuckets, toggleFilter } from "@/lib/mission";
import type { MissionBucket } from "@/lib/mission";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { PitCounts } from "@/components/mission/PitCounts";
import { ProjectsColumn } from "@/components/mission/ProjectsColumn";
import { InboxStrip } from "@/components/mission/InboxStrip";
import { ServerPanel } from "@/components/mission/ServerPanel";
import { MissionBlock } from "@/components/mission/MissionBlock";
import { EventTicker } from "@/components/mission/EventTicker";
import type { ProjectScan } from "@/server/collectors/projects";
import type { Commit } from "@/server/collectors/activity";
import type { HubEnvelopeData } from "@/server/db/workflows";
import type { PrResult } from "@/server/collectors/github";
import type { WorktreeResult } from "@/server/collectors/worktrees";

export default function MissionPage() {
  const t = useTranslations();
  const tNav = useTranslations("nav");
  const projects = useMission<ProjectScan>("projects", 10_000);
  const hub = useMission<HubEnvelopeData>("hub", 10_000);
  const prs = useMission<Record<string, PrResult>>("prs", 60_000);
  const worktrees = useMission<Record<string, WorktreeResult>>("worktrees", 60_000);
  const activity = useMission<Commit[]>("activity", 10_000);
  const [filter, setFilter] = useState<MissionBucket | null>(null);
  const [expandedDir, setExpandedDir] = useState<string | null>(null);

  // enhancement overlay: counts failure hides quietly (spec carve-out) —
  // the kanban page raises the visible warning for the same Linear outage
  const counts = useQuery({
    queryKey: ["kanban", "project-counts"],
    queryFn: ({ signal }) => fetchEnvelope<Record<string, number>>("/api/kanban/project-counts", signal),
    refetchInterval: 60_000,
  });
  const countsByName = counts.data?.ok
    ? Object.fromEntries(Object.entries(counts.data.data).map(([k, v]) => [k.toLowerCase(), v]))
    : null;

  const projectResult = projects.data;

  // F1/F3 (diff review e52d9e6dc555): loading/error/malformed settings must NOT resolve to
  // github off — that would flip every PR section to "disabled" before settings land (and crash
  // on the out-of-shape harness envelope) — so only an explicit ok:true decides.
  const { data: settings } = useGeneralSettings();
  const githubEnabled = settings?.ok === true && settings.data.integrations ? settings.data.integrations.github : true;

  const nowMs = Date.now();
  // F2: worktrees is a TRAILING argument (after nowMs), never inserted before it --
  // every 4-argument buildMission caller elsewhere must keep binding nowMs to position 4.
  const mission = buildMission(projectResult, hub.data, prs.data, nowMs, worktrees.data, githubEnabled);
  // Array.isArray guard: `activity.data.data` is typed Commit[], but an out-of-whitelist test
  // harness returns a projects-shaped envelope for every source — a non-iterable value must
  // degrade to an empty ticker, never crash the page (hard rule 3).
  const ticker = buildTicker(mission.model.cards, Array.isArray(activity.data?.data) ? activity.data.data : [], prs.data?.ok ? prs.data.data : {}, nowMs);
  const summary = summarizeBuckets(mission.readiness, mission.model.cards);
  // F3: worktrees is deliberately OUT of this array (spec §7 quiet-failure exception,
  // decision C2) -- every other source here still raises the page-level warning.
  const staleSource = [projects, hub, prs, activity, counts].find((q) => q.isError);
  const lastGood = staleSource && staleSource.dataUpdatedAt > 0 ? staleSource.dataUpdatedAt : undefined;

  return (
    <div className="mx-auto flex w-full min-w-0 max-w-[1320px] flex-col gap-[22px] [animation:jax-rise_.4s_ease]">
      {staleSource ? (
        <SourceWarning
          label={t("mission.inbox.unavailable")}
          aside={lastGood ? <RelativeTime epochMs={lastGood} /> : undefined}
        />
      ) : null}

      {summary.counts ? (
        <PitCounts
          counts={summary.counts}
          active={filter}
          onToggle={(b) => setFilter((f) => toggleFilter(f, b))}
          hiddenCount={mission.model.hiddenCount}
          autoHiddenCount={mission.model.autoHiddenCount}
        />
      ) : (
        <div className="h-9 w-full max-w-md animate-pulse rounded-full bg-surface-2" />
      )}

      <MissionBlock />

      <InboxStrip mission={mission} onExpand={setExpandedDir} />

      <ProjectsColumn
        mission={mission} reposRoot={mission.model.reposRoot} counts={countsByName} filter={filter}
        expandedDir={expandedDir}
        onToggleExpand={(dir) => setExpandedDir((d) => (d === dir ? null : dir))}
      />

      <div className="grid min-w-0 items-start gap-4 [&>*]:min-w-0 lg:grid-cols-[1.6fr_1fr]">
        <EventTicker items={ticker} />
        <ServerPanel />
      </div>

      <a href="/audit" className="self-start text-[12px] text-muted hover:text-body-ink">
        {tNav("audit")} →
      </a>
    </div>
  );
}
