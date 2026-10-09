"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus, RefreshCw } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import { Suspense, useEffect, useRef, useState } from "react";
import { fetchEnvelope } from "@/lib/api";
import { filterIssues, sortIssues } from "@/lib/boardView";
import {
  kanbanFilters, parseKanbanSearch, searchHref, serializeKanbanSearch, hydrateFilters, boardEnabled, mergeKanbanPatch, STORAGE_KEY,
  type KanbanUrlState,
} from "@/lib/cockpitUrl";
import { moveTo, resolveTeamId } from "@/lib/kanbanMove";
import type { KanbanBoard, LinearTeam } from "@/server/collectors/linear";
import { KanbanColumn } from "@/components/kanban/KanbanColumn";
import { ListView } from "@/components/kanban/ListView";
import { FilterBar } from "@/components/kanban/FilterBar";
import { IssueDetailOverlay } from "@/components/kanban/IssueDetailOverlay";
import { CreateIssueOverlay } from "@/components/kanban/CreateIssueOverlay";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { guidanceText, write, type WriteGuidance } from "@/components/kanban/PropertiesSidebar";

function KanbanPageInner() {
  const t = useTranslations("kanban");
  const tTitle = useTranslations("title");
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const url = parseKanbanSearch(new URLSearchParams(params.toString()));
  const filters = kanbanFilters(url);
  const sort = url.sort;
  const view = url.view;
  const [hydrated, setHydrated] = useState(false);
  const [hydratedTeam, setHydratedTeam] = useState<string | null>(null);

  // KB-1: `pendingUrl` resyncs to `url` every render, then `replaceUrl`
  // updates it immediately (before the next render commits). Two calls made
  // synchronously in the same tick — e.g. back-to-back FilterBar onChange —
  // both compose onto this ref instead of the second overwriting the
  // first's fields against a stale `url` closure.
  const pendingUrl = useRef(url);
  pendingUrl.current = url;

  function replaceUrl(patch: Partial<KanbanUrlState>) {
    const next = mergeKanbanPatch(pendingUrl.current, patch);
    pendingUrl.current = next;
    router.replace(searchHref(pathname, serializeKanbanSearch(new URLSearchParams(params.toString()), next)), { scroll: false });
  }

  const teams = useQuery({
    queryKey: ["kanban", "teams"],
    queryFn: ({ signal }) => fetchEnvelope<LinearTeam[]>("/api/kanban/teams", signal),
    refetchInterval: 60_000,
  });
  const teamList = teams.data?.ok ? teams.data.data : null;
  const zeroTeams = teams.data?.ok && teamList !== null && teamList.length === 0;
  const fallbackTeam = (teamList?.find((tm) => tm.key === "MOA") ?? teamList?.[0])?.id ?? null;
  const teamId = resolveTeamId(url.team, hydratedTeam, teamList, fallbackTeam);

  const board = useQuery({
    queryKey: ["kanban", "board", teamId],
    queryFn: ({ signal }) => fetchEnvelope<KanbanBoard>(`/api/kanban/board?team=${teamId}`, signal),
    refetchInterval: 10_000,
    enabled: boardEnabled(url.team, hydrated, teamId),
  });
  const data = board.data?.ok ? board.data.data : null;

  useEffect(() => {
    let stored: unknown = null;
    try { stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null"); } catch { /* corrupt/absent → defaults */ }
    const restored = hydrateFilters(params, stored);
    replaceUrl(restored);
    // round-2 F4: batch hydratedTeam with hydrated here — url.team (the
    // router's reflection of replaceUrl) can still lag a render behind this.
    setHydratedTeam(restored.team);
    setHydrated(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        team: url.team, project: url.project, priority: url.priority,
        label: url.label, assignee: url.assignee, sort: url.sort, view: url.view,
      }));
    } catch { /* quota/private mode — best-effort */ }
  }, [hydrated, url.team, url.project, url.priority, url.label, url.assignee, url.sort, url.view]);

  const queryClient = useQueryClient();
  const pendingMoves = useRef(new Set<string>());
  const [movePending, setMovePending] = useState(false);
  const [moveGuidance, setMoveGuidance] = useState<WriteGuidance | null>(null);
  const [creating, setCreating] = useState(false);

  async function handleMove(issueId: string, stateId: string) {
    return moveTo(queryClient, teamId ?? "", {
      issueId,
      issue: data?.issues.find((i) => i.id === issueId),
      targetStateId: stateId,
      reservation: pendingMoves.current,
      post: () => write("/api/kanban/move", { issueId, stateId }),
      onPending: setMovePending,
      onGuidance: setMoveGuidance,
      onInvalidate: () => queryClient.invalidateQueries({ queryKey: ["kanban", "board"] }),
    });
  }

  const failed =
    (teams.isError && !teams.data) ||
    (board.isError && !board.data);
  const stale = (teams.isError && !!teams.data) || (board.isError && !!board.data);
  const failError = teams.error instanceof Error
    ? teams.error.message
    : board.error instanceof Error
      ? board.error.message
      : undefined;
  const notConfigured = failError?.includes("LINEAR_API_KEY");

  // client-side filter+sort over the already-polled board (zero new fetch)
  const stateRank = new Map((data?.states ?? []).map((s, i) => [s.id, i] as const));
  const shown = data ? sortIssues(filterIssues(data.issues, filters), sort, stateRank) : [];

  let body: React.ReactNode;
  if (zeroTeams) {
    body = <p className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">{t("zeroTeams")}</p>;
  } else if (failed) {
    body = (
      <SourceWarning
        label={notConfigured ? t("notConfigured") : t("unavailable")}
        detail={notConfigured ? undefined : failError}
      />
    );
  } else if (!data) {
    body = <div className="h-40 animate-pulse rounded-lg border border-line bg-surface" />;
  } else if (data.issues.length === 0) {
    body = <p className="rounded-lg border border-line bg-surface p-5 text-sm text-muted">{t("empty")}</p>;
  } else if (view === "list") {
    body = <ListView states={data.states} issues={shown} onOpenIssue={(issue) => replaceUrl({ issue: issue.id })} />;
  } else {
    body = (
      <div className="jax-scroll grid flex-1 auto-cols-[minmax(226px,1fr)] grid-flow-col gap-3.5 overflow-x-auto pb-1.5">
        {data.states.map((s) => (
          <KanbanColumn
            key={s.id}
            state={s}
            issues={shown.filter((i) => i.stateId === s.id)}
            onDropIssue={handleMove}
            onOpenIssue={(issue) => replaceUrl({ issue: issue.id })}
          />
        ))}
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-[18px] [animation:jax-rise_.4s_ease]">
      <h1 className="sr-only">{tTitle("kanban")}</h1>
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex h-8 items-center gap-2 rounded-full border border-line bg-info-soft px-3">
          <RefreshCw className="h-3.5 w-3.5 text-info" />
          <span className="text-xs font-semibold text-info">
            {t("synced")}
            {board.dataUpdatedAt ? (
              <>
                {" · "}
                <RelativeTime epochMs={board.dataUpdatedAt} />
              </>
            ) : null}
          </span>
        </div>
        <button
          type="button"
          onClick={() => setCreating(true)}
          className="flex h-8 items-center gap-1.5 rounded-full bg-brand px-3 text-xs font-semibold text-on-brand"
        >
          <Plus className="h-3.5 w-3.5" />
          {t("createButton")}
        </button>
        <span className="text-[11px] text-muted">{t("persistHint")}</span>
        {data ? (
          <FilterBar
            issues={data.issues}
            filters={filters}
            onFilters={(next) => replaceUrl({ q: next.query, project: next.project, priority: next.priority, label: next.label, assignee: next.assignee })}
            sort={sort}
            onSort={(next) => replaceUrl({ sort: next })}
            view={view}
            onView={(next) => replaceUrl({ view: next })}
          />
        ) : null}
        <div className="ml-auto flex gap-1 rounded-full border border-line bg-surface-2 p-1">
          {(teamList ?? []).map((tm) => (
            <button
              key={tm.id}
              onClick={() => {
                replaceUrl({ team: tm.id, issue: null });
                setMoveGuidance(null);
              }}
              className={`flex h-7 items-center rounded-full px-2.5 text-xs font-semibold transition-colors ${
                tm.id === teamId ? "bg-brand-soft text-brand" : "text-muted hover:text-ink"
              }`}
            >
              {tm.key}
            </button>
          ))}
        </div>
      </div>
      {stale ? (
        <SourceWarning
          label={t("unavailable")}
          aside={board.dataUpdatedAt ? <RelativeTime epochMs={board.dataUpdatedAt} /> : undefined}
        />
      ) : null}
      {movePending || moveGuidance ? (
        <div
          role="status"
          aria-live="polite"
          className={`rounded-md border px-3.5 py-2 text-xs ${
            moveGuidance
              ? "border-danger bg-danger-soft text-danger"
              : "border-line bg-surface-2 text-muted"
          }`}
        >
          {moveGuidance ? guidanceText(t, moveGuidance) : t("movePending")}
        </div>
      ) : null}
      {body}
      {url.issue ? (
        // keyed by issue: switching A -> B remounts the overlay, so A's draft,
        // sending state and outcomes can never modify B (lifetime ownership).
        // The overlay resolves everything from the detail endpoint itself, even
        // when the board lacks the issue or is unavailable.
        <IssueDetailOverlay
          key={url.issue}
          issueId={url.issue}
          onClose={() => replaceUrl({ issue: null })}
          boardIssueIds={new Set((data?.issues ?? []).map((i) => i.id))}
          onOpenParent={(id) => replaceUrl({ issue: id })}
        />
      ) : null}
      {creating && teamId ? (
        <CreateIssueOverlay
          teamId={teamId}
          states={data?.states ?? []}
          issues={data?.issues ?? []}
          defaultProjectName={filters.project}
          onClose={() => setCreating(false)}
          onCreated={(id) => { setCreating(false); replaceUrl({ issue: id }); }}
        />
      ) : null}
    </div>
  );
}

export default function KanbanPage() {
  return (
    <Suspense>
      <KanbanPageInner />
    </Suspense>
  );
}
