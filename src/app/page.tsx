"use client";

import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { flushSync } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { useMission } from "@/lib/useMission";
import { useCockpitSnapshot } from "@/lib/cockpitPrefsQuery";
import { useGeneralSettings } from "@/lib/settingsQuery";
import { fetchEnvelope } from "@/lib/api";
import { boardCards, boardShowsKey, buildMission, buildTicker, summarizeBuckets, toggleFilter } from "@/lib/mission";
import type { MissionBucket } from "@/lib/mission";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { ProjectsColumn } from "@/components/mission/ProjectsColumn";
import { InboxStrip } from "@/components/mission/InboxStrip";
import { ServerPanel } from "@/components/mission/ServerPanel";
import { MissionBlock } from "@/components/mission/MissionBlock";
import { AnswerDraftsProvider } from "@/components/mission/AnswerDrafts";
import { EventTicker } from "@/components/mission/EventTicker";
import type { ProjectScan } from "@/server/collectors/projects";
import type { Commit } from "@/server/collectors/activity";
import type { HubEnvelopeData } from "@/server/db/workflows";
import type { PrResult } from "@/server/collectors/github";
import type { WorktreeResult } from "@/server/collectors/worktrees";

export type HomeView = "projects" | "activity";

// ?view= is the meaningful selection (plan Task 5). Read it from the real location after mount
// instead of useSearchParams, whose Suspense boundary would render an empty home during
// production prerender while still letting the surrounding page render; this keeps the same
// URL contract and other parameters intact via the exported href builder.
export function parseHomeView(search: string): HomeView {
  return new URLSearchParams(search).get("view") === "activity" ? "activity" : "projects";
}

export function homeViewHref(pathname: string, search: string, view: HomeView): string {
  const params = new URLSearchParams(search);
  params.set("view", view);
  return `${pathname}?${params.toString()}`;
}

function HomeCockpit() {
  const t = useTranslations();
  const tNav = useTranslations("nav");
  const tViews = useTranslations("mission.views");
  const [view, setView] = useState<HomeView>("projects");
  const [location, setLocation] = useState<{ path: string; search: string }>({ path: "/", search: "" });
  useEffect(() => {
    setView(parseHomeView(window.location.search));
    setLocation({ path: window.location.pathname, search: window.location.search });
  }, []);
  const projects = useMission<ProjectScan>("projects", 10_000);
  const hub = useMission<HubEnvelopeData>("hub", 10_000);
  const activity = useMission<Commit[]>("activity", 10_000);
  const worktrees = useMission<Record<string, WorktreeResult>>("worktrees", 60_000);
  const [filter, setFilter] = useState<MissionBucket | null>(null);
  const [expandedDir, setExpandedDir] = useState<string | null>(null);

  // One polling preference snapshot. The PR area never issues a query before this is known, and
  // an unavailable source renders an explicit PR-unavailable state instead of a forever-loading
  // skeleton (plan F2); the board falls back to its default order/width with the controls
  // disabled inside ProjectsColumn.
  const cockpit = useCockpitSnapshot();
  const cockpitSnapshot = cockpit.data?.data ?? null;
  const preferencesUnavailable = cockpit.isError && cockpitSnapshot === null;
  const prScope = cockpitSnapshot?.prScope ?? "mine";
  const prs = useMission<Record<string, PrResult>>("prs", 60_000, cockpitSnapshot !== null, `scope=${prScope}`);
  const prsEnvelope = preferencesUnavailable ? { ok: false as const, error: "preferences unavailable" } : prs.data;

  // enhancement overlay: counts failure hides quietly (spec carve-out) — the kanban page raises
  // the visible warning for the same Linear outage.
  const counts = useQuery({
    queryKey: ["kanban", "project-counts"],
    queryFn: ({ signal }) => fetchEnvelope<Record<string, number>>("/api/kanban/project-counts", signal),
    refetchInterval: 60_000,
  });
  const countsByName = counts.data?.ok
    ? Object.fromEntries(Object.entries(counts.data.data).map(([k, v]) => [k.toLowerCase(), v]))
    : null;

  // F1/F3 (diff review e52d9e6dc555): only an explicit ok:true with github off disables the PR
  // section — loading/error/malformed settings must not flip it before settings land.
  const { data: settings } = useGeneralSettings();
  const githubEnabled = settings?.ok === true && settings.data.integrations ? settings.data.integrations.github : true;

  const nowMs = Date.now();
  // F2: worktrees is a TRAILING argument (after nowMs), never inserted before it.
  const mission = buildMission(projects.data, hub.data, prsEnvelope, nowMs, worktrees.data, githubEnabled);
  const ticker = buildTicker(
    mission.model.cards,
    Array.isArray(activity.data?.data) ? activity.data.data : [],
    prsEnvelope && prsEnvelope.ok ? prsEnvelope.data : {},
    nowMs,
  );
  const summary = summarizeBuckets(mission.readiness, boardCards(mission.model.cards));
  // F3: worktrees is deliberately OUT of this array (spec §7 quiet-failure exception).
  const staleSource = [projects, hub, activity, counts, cockpitSnapshot !== null ? prs : undefined].find((q) => q?.isError);
  const lastGood = staleSource && staleSource.dataUpdatedAt > 0 ? staleSource.dataUpdatedAt : undefined;

  const viewHref = (next: HomeView) => homeViewHref(location.path, location.search, next);
  const tabClass = (active: boolean) =>
    `rounded-md border px-3 py-1 text-[12px] font-semibold ${active ? "border-brand-soft-border bg-brand-soft text-brand" : "border-line text-muted hover:text-body-ink"}`;

  return (
    <div className="mx-auto flex w-full min-w-0 max-w-[1320px] flex-col gap-[18px] [animation:jax-rise_.4s_ease]">
      {staleSource ? (
        <SourceWarning
          label={t("mission.inbox.unavailable")}
          aside={lastGood ? <RelativeTime epochMs={lastGood} /> : undefined}
        />
      ) : null}

      <MissionBlock />
      <ServerPanel />

      <nav aria-label={tViews("label")} className="flex items-center gap-1.5">
        <a href={viewHref("projects")} aria-current={view === "projects" ? "page" : undefined} className={tabClass(view === "projects")}>{tViews("projects")}</a>
        <a href={viewHref("activity")} aria-current={view === "activity" ? "page" : undefined} className={tabClass(view === "activity")}>{tViews("activity")}</a>
      </nav>

      <AnswerDraftsProvider>
        {/* Both surfaces stay mounted: switching to Activity must not destroy answer drafts,
            focus, expansion or confirmation dialogs (spec AC6). */}
        <div data-testid="home-projects" className={view === "projects" ? "flex flex-col gap-[22px]" : "hidden"}>
          <InboxStrip mission={mission} onExpand={(key) => {
            // Respond on a card the filter hides: drop the filter synchronously so the scroll finds it.
            if (filter !== null && !boardShowsKey(mission.model.cards, filter, key)) flushSync(() => { setFilter(null); setExpandedDir(key); });
            else setExpandedDir(key);
          }} />
          <ProjectsColumn
            mission={mission} reposRoot={mission.model.reposRoot} counts={countsByName} filter={filter}
            expandedDir={expandedDir}
            onToggleExpand={(dir) => setExpandedDir((d) => (d === dir ? null : dir))}
            bucketSummary={summary}
            onToggleFilter={(b) => setFilter((f) => toggleFilter(f, b))}
          />
        </div>
        <div data-testid="home-activity" className={view === "activity" ? "" : "hidden"}>
          <div className="grid min-w-0 items-start gap-4 [&>*]:min-w-0 lg:grid-cols-[1.6fr_1fr]">
            <div className="min-w-0 lg:col-span-2">
              <EventTicker items={ticker} />
            </div>
          </div>
        </div>
      </AnswerDraftsProvider>

      <a href="/audit" className="self-start text-[12px] text-muted hover:text-body-ink">
        {tNav("audit")} →
      </a>
    </div>
  );
}

export default function MissionPage() {
  return <HomeCockpit />;
}
