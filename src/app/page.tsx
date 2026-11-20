"use client";

import { useTranslations } from "next-intl";
import { useEffect, useRef, useState, type MouseEvent } from "react";
import { flushSync } from "react-dom";
import { useQuery } from "@tanstack/react-query";
import { useMission } from "@/lib/useMission";
import { useCockpitSnapshot } from "@/lib/cockpitPrefsQuery";
import { useGeneralSettings } from "@/lib/settingsQuery";
import { fetchEnvelope } from "@/lib/api";
import { boardCards, boardShowsKey, buildMission, buildTicker, summarizeBuckets, toggleFilter } from "@/lib/mission";
import type { MissionBucket, MissionReadiness } from "@/lib/mission";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";
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

export type HomeView = "projects" | "activity";

// Per-tab Home UI snapshot (M3b): filter, expanded card key, the #main-content scroll offset and
// the ?view origin, in sessionStorage only. No DB/state library; guarded and bounded.
export const HOME_SNAPSHOT_KEY = "jax-os.home.snapshot";
export type HomeSnapshot = {
  view: HomeView;
  filter: MissionBucket | null;
  expandedKey: string | null;
  scrollTop: number;
};

const HOME_BUCKETS: readonly MissionBucket[] = ["needs-you", "stuck", "working", "idle"];

export function parseHomeSnapshot(raw: string | null): HomeSnapshot | null {
  if (raw === null) return null;
  let value: unknown;
  try { value = JSON.parse(raw); } catch { return null; }
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  if (record.view !== "projects" && record.view !== "activity") return null;
  const filter = record.filter === null ? null : HOME_BUCKETS.includes(record.filter as MissionBucket) ? record.filter as MissionBucket : undefined;
  if (filter === undefined) return null;
  const expandedKey = record.expandedKey === null
    ? null
    : typeof record.expandedKey === "string" && record.expandedKey.length > 0 && record.expandedKey.length <= 1024
      ? record.expandedKey
      : undefined;
  if (expandedKey === undefined) return null;
  const scrollTop = typeof record.scrollTop === "number" && Number.isFinite(record.scrollTop) && record.scrollTop >= 0
    ? record.scrollTop
    : null;
  if (scrollTop === null) return null;
  return { view: record.view, filter, expandedKey, scrollTop };
}

export function readHomeSnapshot(storage: Pick<Storage, "getItem"> | null): HomeSnapshot | null {
  if (!storage) return null;
  try { return parseHomeSnapshot(storage.getItem(HOME_SNAPSHOT_KEY)); } catch { return null; }
}

export function writeHomeSnapshot(storage: Pick<Storage, "setItem"> | null, snapshot: HomeSnapshot): void {
  if (!storage) return;
  try { storage.setItem(HOME_SNAPSHOT_KEY, JSON.stringify(snapshot)); } catch { /* storage unavailable */ }
}

function browserSessionStorage(): Storage | null {
  try { return typeof window === "undefined" ? null : window.sessionStorage; } catch { return null; }
}

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

// Modifier/button shape of a click that should keep the browser's own navigation (new tab,
// window, download…). Everything else is an ordinary primary click on a plain link.
export type HomeViewClick = {
  button: number; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean;
};

// The whole decision, pure: whether to intercept this click and whether it changes the selection.
// Clicking the already-selected tab is swallowed without a duplicate history entry.
export function homeViewClickPlan(
  current: HomeView, next: HomeView, click: HomeViewClick, pathname: string, search: string,
): { intercept: boolean; push: boolean; href: string } {
  const href = homeViewHref(pathname, search, next);
  const plain = click.button === 0 && !click.metaKey && !click.ctrlKey && !click.shiftKey && !click.altKey;
  return plain ? { intercept: true, push: next !== current, href } : { intercept: false, push: false, href };
}

// Back/Forward and the first client render both read the REAL current location (never a stale
// captured search), so the selection follows the URL in both directions.
export function readHomeLocation(loc: { pathname: string; search: string }): { view: HomeView; location: { path: string; search: string } } {
  return { view: parseHomeView(loc.search), location: { path: loc.pathname, search: loc.search } };
}

// "Board committed with real data": both backbone sources succeeded and the board is not still a
// loading skeleton. degraded (a PR-only outage) is still a real board.
export function homeBoardReady(readiness: MissionReadiness): boolean {
  return readiness.kind === "ready" || readiness.kind === "degraded";
}

function HomeCockpit() {
  const t = useTranslations();
  const tNav = useTranslations("nav");
  const tViews = useTranslations("mission.views");
  // The saved snapshot is read ONLY after mount and applied before the later writers arm, so the
  // first client render is byte-identical to the server render (no hydration mismatch).
  const [saved, setSaved] = useState<HomeSnapshot | null>(null);
  const [initialized, setInitialized] = useState(false);
  const [view, setView] = useState<HomeView>("projects");
  const [location, setLocation] = useState<{ path: string; search: string }>({ path: "/", search: "" });
  useEffect(() => {
    const sync = () => {
      const next = readHomeLocation(window.location);
      const params = new URLSearchParams(window.location.search);
      setView(params.has("view") ? next.view : (saved?.view ?? "projects"));
      setLocation(next.location);
    };
    sync();
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, [saved]);
  const projects = useMission<ProjectScan>("projects", 10_000);
  const hub = useMission<HubEnvelopeData>("hub", 10_000);
  const activity = useMission<Commit[]>("activity", 10_000);
  const worktrees = useMission<Record<string, WorktreeResult>>("worktrees", 60_000);
  const [filter, setFilter] = useState<MissionBucket | null>(null);
  const [expandedDir, setExpandedDir] = useState<string | null>(null);

  // Persist UI state only after the snapshot initialization; restore the saved scroll ONCE after
  // Home's project content is ready (later polls never force it back).
  const readyRef = useRef(false);
  const restoredRef = useRef(false);
  const scrollRef = useRef(0);
  const uiRef = useRef({ view, filter, expandedKey: expandedDir });
  uiRef.current = { view, filter, expandedKey: expandedDir };
  useEffect(() => {
    const snapshot = readHomeSnapshot(browserSessionStorage());
    if (snapshot) {
      setSaved(snapshot);
      setView(new URLSearchParams(window.location.search).has("view") ? parseHomeView(window.location.search) : snapshot.view);
      scrollRef.current = snapshot.scrollTop;
      setFilter(snapshot.filter);
      setExpandedDir(snapshot.expandedKey);
    }
    setInitialized(true);
  }, []);
  // Writers arm only after the restored state is committed; the mount pass can never overwrite it.
  useEffect(() => { if (initialized) readyRef.current = true; }, [initialized]);
  useEffect(() => {
    const storage = browserSessionStorage();
    if (readyRef.current) writeHomeSnapshot(storage, { ...uiRef.current, scrollTop: scrollRef.current });
  }, [view, filter, expandedDir]);
  useEffect(() => {
    const element = document.getElementById("main-content");
    if (!element) return;
    let frame = 0;
    const onScroll = () => {
      // While the saved offset is still being restored, the programmatic scroll must not overwrite it.
      if (!readyRef.current || !restoredRef.current) return;
      scrollRef.current = element.scrollTop;
      if (frame !== 0) return;
      frame = requestAnimationFrame(() => {
        frame = 0;
        writeHomeSnapshot(browserSessionStorage(), { ...uiRef.current, scrollTop: scrollRef.current });
      });
    };
    element.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      element.removeEventListener("scroll", onScroll);
      if (frame !== 0) cancelAnimationFrame(frame);
    };
  }, []);

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
  // Restore the saved #main-content offset ONCE, only when the real board is committed: both
  // projects and hub succeeded and readiness is no longer loading. At most three animation frames
  // wait for the rendered range; a genuinely shorter board clamps once, a later poll never resets.
  const boardReady = homeBoardReady(mission.readiness);
  useEffect(() => {
    if (restoredRef.current || !boardReady) return;
    const element = document.getElementById("main-content");
    if (!element) return;
    const top = scrollRef.current;
    if (top <= 0) { restoredRef.current = true; return; }
    let attempts = 0;
    let frame = 0;
    const settle = () => {
      attempts += 1;
      const max = Math.max(0, element.scrollHeight - element.clientHeight);
      if (max >= top) {
        element.scrollTo({ top });
        restoredRef.current = true;
        return;
      }
      if (attempts >= 3) {
        element.scrollTo({ top: max });
        restoredRef.current = true;
        return;
      }
      frame = requestAnimationFrame(settle);
    };
    frame = requestAnimationFrame(settle);
    return () => { if (!restoredRef.current) cancelAnimationFrame(frame); };
  }, [boardReady]);
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
  // The board, its provider and drafts stay mounted: an ordinary click only pushes the meaningful
  // ?view= URL and updates the local selection instead of reloading the document.
  const onViewClick = (event: MouseEvent<HTMLAnchorElement>, next: HomeView) => {
    const plan = homeViewClickPlan(view, next, event, location.path, location.search);
    if (!plan.intercept) return;
    event.preventDefault();
    if (!plan.push) return;
    window.history.pushState(null, "", plan.href);
    flushSync(() => setView(next));
    setLocation({ path: location.path, search: plan.href.slice(location.path.length) });
    document.getElementById(`home-${next}`)?.querySelector<HTMLAnchorElement>("nav a[aria-current='page']")?.focus();
  };
  const tabClass = (active: boolean) =>
    `rounded-md border px-3 py-1 text-[12px] font-semibold ${active ? "border-brand-soft-border bg-brand-soft text-brand" : "border-line text-muted hover:text-body-ink"}`;
  const navigation = (
    <nav aria-label={tViews("label")} className="flex flex-none items-center gap-1.5">
      <a href={viewHref("projects")} onClick={(event) => onViewClick(event, "projects")} aria-current={view === "projects" ? "page" : undefined} className={tabClass(view === "projects")}>{tViews("projects")}</a>
      <a href={viewHref("activity")} onClick={(event) => onViewClick(event, "activity")} aria-current={view === "activity" ? "page" : undefined} className={tabClass(view === "activity")}>{tViews("activity")}</a>
    </nav>
  );

  return (
    <div className="mx-auto flex min-h-full w-full min-w-0 max-w-[1320px] flex-col gap-[18px] [animation:jax-rise_.4s_ease]">
      {staleSource ? (
        <SourceWarning
          label={t("mission.inbox.unavailable")}
          aside={lastGood ? <RelativeTime epochMs={lastGood} /> : undefined}
        />
      ) : null}

      <MissionBlock />

      {/* Both surfaces stay mounted: switching to Activity must not destroy answer drafts,
          focus, expansion or confirmation dialogs (spec AC6). The draft provider now lives in
          app/Providers.tsx above route children (M3b). */}
      <div id="home-projects" data-testid="home-projects" className={view === "projects" ? "flex flex-col gap-[22px]" : "hidden"}>
          <InboxStrip mission={mission} onExpand={(key) => {
            // Respond on a card the filter hides: drop the filter synchronously so the scroll finds it.
            if (filter !== null && !boardShowsKey(mission.model.cards, filter, key)) flushSync(() => { setFilter(null); setExpandedDir(key); });
            else setExpandedDir(key);
          }} />
          <ProjectsColumn
            navigation={navigation}
            mission={mission} reposRoot={mission.model.reposRoot} counts={countsByName} filter={filter}
            expandedDir={expandedDir}
            onToggleExpand={(dir) => setExpandedDir((d) => (d === dir ? null : dir))}
            bucketSummary={summary}
            onToggleFilter={(b) => setFilter((f) => toggleFilter(f, b))}
          />
        </div>
      <div id="home-activity" data-testid="home-activity" className={view === "activity" ? "flex flex-col gap-3.5" : "hidden"}>
        {navigation}
        <div className="grid min-w-0 items-start gap-4 [&>*]:min-w-0 lg:grid-cols-[1.6fr_1fr]">
          <div className="min-w-0 lg:col-span-2">
            <EventTicker items={ticker} />
            <a href="/executions" className="mt-2 inline-block text-[12px] text-muted hover:text-body-ink">
              {tNav("executions")} →
            </a>
          </div>
        </div>
      </div>

      <a href="/audit" className="self-start text-[12px] text-muted hover:text-body-ink">
        {tNav("audit")} →
      </a>
      <ServerPanel />
    </div>
  );
}

export default function MissionPage() {
  return <HomeCockpit />;
}
