import type { QueryClient } from "@tanstack/react-query";
import type { Envelope } from "@/lib/api";
import { postWorkflowAction } from "@/lib/mission";
import type { HubEnvelopeData, ProjectPrefs } from "@/server/db/workflows";

export const HUB_QUERY_KEY = ["mission", "hub"];

// Mirrors setProjectPrefs' own patch semantics (src/server/db/workflows.ts) so the
// optimistic value matches what the server will actually persist.
export function nextPrefs(prev: ProjectPrefs, patch: { pinned?: boolean; hidden?: boolean }, now = new Date()): ProjectPrefs {
  return {
    pinned: patch.pinned ?? prev.pinned,
    hiddenAt: patch.hidden === undefined ? prev.hiddenAt : patch.hidden ? now.toISOString() : null,
  };
}

// Item 2 (optimistic hide/pin, mission card UX): patches the hub query's cached prefs for
// `dir` immediately so the card re-buckets on this same render (buildMission derives
// visibility from `hub.data.prefs`), POSTs, then invalidates on success so the next mission
// poll lands right away instead of waiting out the 10s interval — and rolls back to the
// pre-patch snapshot on failure. Same shape as kanbanMove.ts's moveTo for the kanban board;
// `post` is injectable for tests (default: the real fetch-backed postWorkflowAction).
export async function postPrefsPatch(
  queryClient: QueryClient,
  dir: string,
  patch: { pinned?: boolean; hidden?: boolean },
  post: (url: string, body: unknown) => Promise<Envelope<ProjectPrefs>> = postWorkflowAction,
): Promise<Envelope<ProjectPrefs>> {
  const snapshot = queryClient.getQueryData<Envelope<HubEnvelopeData>>(HUB_QUERY_KEY);
  if (snapshot?.ok) {
    const prev = snapshot.data.prefs[dir] ?? { pinned: false, hiddenAt: null };
    const patched: Envelope<HubEnvelopeData> = {
      ok: true,
      data: { ...snapshot.data, prefs: { ...snapshot.data.prefs, [dir]: nextPrefs(prev, patch) } },
    };
    queryClient.setQueryData(HUB_QUERY_KEY, patched);
  }
  const result = await post("/api/mission/prefs", { project: dir, ...patch });
  if (!result.ok) {
    if (snapshot) queryClient.setQueryData(HUB_QUERY_KEY, snapshot);
    return result;
  }
  void queryClient.invalidateQueries({ queryKey: HUB_QUERY_KEY });
  return result;
}
