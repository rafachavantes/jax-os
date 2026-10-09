import type { MutationPage, MutationRow } from "@/server/db/mutations";

export type AuditViewState = {
  identity: string;
  // increments on every filter switch (UI or history navigation), so a
  // response captured under an older selection can never mutate the current
  // one, including an A -> B -> A return to the same filter identity
  generation: number;
  rows: MutationRow[];
  // cursor: string once established, null both before any first page and
  // after exhaust. loadMoreUsed distinguishes "no older-page consumption yet"
  // from "load-more has run": before the first load-more, polls may refresh
  // nextCursor (a >limit first page gains a cursor); afterwards the cursor is
  // captured/advanced exclusively by load-more settlements, and a final null
  // stays exhausted across polls
  cursor: string | null;
  loadMoreUsed: boolean;
  error: string | null;
  openId: number | null;
  loadMorePending: boolean;
};

export type AuditAction =
  | { type: "filter"; identity: string }
  | { type: "firstPage"; identity: string; page: MutationPage }
  | { type: "loadMoreStart"; generation: number }
  | { type: "loadMoreDone"; identity: string; generation: number; page: MutationPage }
  | { type: "loadMoreFailed"; identity: string; generation: number; error: string }
  | { type: "open"; id: number | null };

export function auditFilterIdentity(kind: string, preset: string): string {
  return `${kind}|${preset}`;
}

export function emptyAuditState(identity: string): AuditViewState {
  return {
    identity,
    generation: 0,
    rows: [],
    cursor: null,
    loadMoreUsed: false,
    error: null,
    openId: null,
    loadMorePending: false,
  };
}

export function mergeAuditRows(current: MutationRow[], incoming: MutationRow[]): MutationRow[] {
  const byId = new Map<number, MutationRow>();
  for (const row of current) byId.set(row.id, row);
  for (const row of incoming) byId.set(row.id, row);
  return [...byId.values()].sort((a, b) => (a.ts === b.ts ? b.id - a.id : a.ts < b.ts ? 1 : -1));
}

// The production audit-page transition seam: the page consumes every actual
// successful first-page/load-more result exactly once through this reducer,
// filter-scoped by identity and generation. A first-page refresh merges into
// all retained rows (including former page-1 rows), never resets an
// already-used older-page cursor and never clears a load-more error — only
// that error's own retry or a new filter owns it. A successful empty first
// page is a real empty result: rows and pagination are cleared.
export function auditReducer(state: AuditViewState, action: AuditAction): AuditViewState {
  switch (action.type) {
    case "filter":
      if (action.identity === state.identity) return state;
      return { ...emptyAuditState(action.identity), generation: state.generation + 1 };
    case "firstPage":
      if (action.identity !== state.identity) return state;
      if (action.page.rows.length === 0) {
        return { ...state, rows: [], cursor: null, loadMoreUsed: false };
      }
      return {
        ...state,
        rows: mergeAuditRows(state.rows, action.page.rows),
        cursor: state.loadMoreUsed ? state.cursor : action.page.nextCursor,
      };
    case "loadMoreStart":
      if (action.generation !== state.generation || state.loadMorePending || !state.cursor) {
        return state;
      }
      // from here on the cursor is owned by load-more settlements: polls must
      // never revive a consumed or exhausted cursor
      return { ...state, loadMorePending: true, error: null, loadMoreUsed: true };
    case "loadMoreDone": {
      const current = action.identity === state.identity && action.generation === state.generation;
      return {
        ...state,
        ...(current
          ? {
              rows: mergeAuditRows(state.rows, action.page.rows),
              cursor: action.page.nextCursor,
              error: null,
            }
          : {}),
        loadMorePending: false,
      };
    }
    case "loadMoreFailed":
      return {
        ...state,
        ...(action.identity === state.identity && action.generation === state.generation
          ? { error: action.error }
          : {}),
        loadMorePending: false,
      };
    case "open":
      return { ...state, openId: action.id };
  }
}

// selector used before dispatching loadMoreStart: only a pending-free state
// with an established, non-exhausted cursor can load more
export function beginAuditLoadMore(state: AuditViewState): { cursor: string } | null {
  if (state.loadMorePending || !state.cursor) return null;
  return { cursor: state.cursor };
}