import type { QueryClient } from "@tanstack/react-query";
import type { Envelope } from "@/lib/api";
import { nextBoardState } from "@/lib/boardView";
import type { KanbanBoard, LinearTeam } from "@/server/collectors/linear";
import { moveIssue } from "@/components/kanban/KanbanColumn";
import { reconcileWrite } from "@/components/kanban/PropertiesSidebar";

// Optimistic move transaction (phase 6 spec §6.3, §9, round-1 F1): previously
// a KanbanPageInner closure, now a standalone module for direct testing
// without a DOM (page.tsx cannot carry a named export besides its reserved
// fields — Next.js 15 rejects it at build time).
// Wraps moveIssue unchanged; never touches the cache for a same-column drop.
// The board query caches fetchEnvelope's shape ({ok:true,data:KanbanBoard}),
// never a bare KanbanBoard (round-1 F1) — unwrap before nextBoardState,
// re-wrap on every write back; skip the optimistic patch when not ok.
//
// round-2 F1: settlement used to replace the WHOLE board via
// reconcileWrite(snapshot, applied, g) captured once at call time — a
// concurrent move on another issue landing in between would be silently
// erased or resurrected. It now patches only the moved issue's own stateId
// against whatever is in the cache AT SETTLEMENT TIME, leaving every other
// in-flight optimistic change untouched.
export async function moveTo(
  queryClient: QueryClient,
  teamId: string,
  opts: Parameters<typeof moveIssue>[0],
): Promise<void> {
  const key = ["kanban", "board", teamId];
  const envelope = queryClient.getQueryData<Envelope<KanbanBoard>>(key);
  if (!envelope?.ok || !opts.issue || opts.issue.stateId === opts.targetStateId) {
    return moveIssue(opts);
  }
  const fromStateId = opts.issue.stateId;
  const applied = nextBoardState(envelope.data, opts.issueId, opts.targetStateId);
  queryClient.setQueryData(key, { ok: true, data: applied });
  return moveIssue({
    ...opts,
    onGuidance: (g) => {
      opts.onGuidance(g);
      const reconciledStateId = reconcileWrite(fromStateId, opts.targetStateId, g);
      queryClient.setQueryData<Envelope<KanbanBoard>>(key, (current) => {
        if (!current?.ok) return current;
        const issues = current.data.issues.map((i) => (i.id === opts.issueId ? { ...i, stateId: reconciledStateId } : i));
        return { ok: true, data: { ...current.data, issues } };
      });
    },
  });
}

// round-2 F4: the FIRST enabled board query must target the just-restored
// team, never url.team's stale reflection of it — so an absent URL team
// falls back to the hydrated one. round-3 F1: once the URL DOES carry a
// team (e.g. a team-button click), it must win over the now-stale hydrated
// value instead of being permanently shadowed by it. Standalone for direct
// testing, same reason as moveTo (this file's tests never run an effect).
export function resolveTeamId(
  urlTeam: string | null,
  hydratedTeam: string | null,
  teamList: LinearTeam[] | null,
  fallbackTeam: string | null,
): string | null {
  const restoredTeam = urlTeam ?? hydratedTeam;
  if (!teamList) return restoredTeam;
  return teamList.some((tm) => tm.id === restoredTeam) ? restoredTeam : fallbackTeam;
}
