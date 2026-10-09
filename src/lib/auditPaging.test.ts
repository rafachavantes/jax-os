import { describe, expect, it } from "vitest";
import type { MutationPage, MutationRow } from "@/server/db/mutations";
import {
  auditReducer, beginAuditLoadMore, emptyAuditState, mergeAuditRows,
  type AuditAction, type AuditViewState,
} from "./auditPaging";

const row = (over: Partial<MutationRow> & Pick<MutationRow, "id" | "ts">): MutationRow => ({
  kind: "x",
  ok: true,
  error: null,
  payload: { outcome: "done" },
  ...over,
});

const page = (rows: MutationRow[], nextCursor: string | null = null): MutationPage => ({
  rows, kinds: ["x"], total: rows.length, nextCursor,
});

// drives the page's own transition seam: the same reducer the page runs
function drive(state: AuditViewState, ...actions: AuditAction[]): AuditViewState {
  return actions.reduce((s, a) => auditReducer(s, a), state);
}

describe("mergeAuditRows", () => {
  it("lets incoming win on the same id and sorts (ts,id) DESC", () => {
    const current = [row({ id: 1, ts: "2026-07-01T00:00:00.000Z", payload: { outcome: "pending" } })];
    const incoming = [
      row({ id: 1, ts: "2026-07-01T00:00:00.000Z", payload: { outcome: "done" } }),
      row({ id: 2, ts: "2026-07-02T00:00:00.000Z" }),
    ];
    const merged = mergeAuditRows(current, incoming);
    expect(merged.map((r) => r.id)).toEqual([2, 1]);
    expect((merged[1].payload as { outcome: string }).outcome).toBe("done");
  });
});

describe("audit page transition seam (reducer)", () => {
  const idA = "a|30d";
  const idB = "b|30d";
  const first = row({ id: 10, ts: "2026-07-10T00:00:00.000Z" });
  const older = row({ id: 9, ts: "2026-07-09T00:00:00.000Z" });
  const newest = row({ id: 11, ts: "2026-07-11T00:00:00.000Z" });

  it("pollA -> pollB without any load-more: merges rows, keeps the established cursor and no duplicates", () => {
    let state = emptyAuditState(idA);
    state = drive(
      state,
      { type: "firstPage", identity: idA, page: page([first], "c1") },
      { type: "firstPage", identity: idA, page: page([older, first], "c1") },
    );
    expect(state.rows.map((r) => r.id)).toEqual([10, 9]);
    expect(state.cursor).toBe("c1");
    expect(state.openId).toBeNull();
  });

  it("established cursor stays across later first pages; updated outcomes replace rows in place", () => {
    let state = emptyAuditState(idA);
    state = drive(
      state,
      { type: "firstPage", identity: idA, page: page([first], "c1") },
    );
    expect(beginAuditLoadMore(state)?.cursor).toBe("c1");
    state = drive(state, { type: "loadMoreStart", generation: 0 });
    state = drive(state, { type: "loadMoreDone", identity: idA, generation: 0, page: page([older], "c2") });
    expect(state.rows.map((r) => r.id)).toEqual([10, 9]);
    expect(state.cursor).toBe("c2");
    const done = row({ id: 10, ts: "2026-07-10T00:00:00.000Z", payload: { outcome: "done" } });
    state = drive(state, { type: "firstPage", identity: idA, page: page([done], "c1-new") });
    expect(state.cursor).toBe("c2"); // older-page cursor never reset by a poll
    expect(state.rows).toHaveLength(2);
    expect((state.rows[0].payload as { outcome: string }).outcome).toBe("done");
  });

  it("last page -> rerender -> poll: the final load-more null stays exhausted", () => {
    let state = emptyAuditState(idA);
    state = drive(
      state,
      { type: "firstPage", identity: idA, page: page([first], "c1") },
      { type: "loadMoreStart", generation: 0 },
      { type: "loadMoreDone", identity: idA, generation: 0, page: page([older], null) },
    );
    expect(state.cursor).toBeNull();
    expect(beginAuditLoadMore(state)).toBeNull();
    // a new poll inserts rows again, but exhaustion stays: no stale cursor revival
    state = drive(state, { type: "firstPage", identity: idA, page: page([newest, first], "c1") });
    expect(state.cursor).toBeNull();
    expect(state.rows.map((r) => r.id)).toEqual([11, 10, 9]);
  });

  it("failed load-more -> poll (rerender) -> retry: only the retry owns the error", () => {
    let state = emptyAuditState(idA);
    state = drive(
      state,
      { type: "firstPage", identity: idA, page: page([first], "c1") },
      { type: "loadMoreStart", generation: 0 },
      { type: "loadMoreFailed", identity: idA, generation: 0, error: "page-fail" },
    );
    expect(state.error).toBe("page-fail");
    expect(state.loadMorePending).toBe(false);
    // a first-page refresh never clears the load-more error
    state = drive(state, { type: "firstPage", identity: idA, page: page([first], "c1") });
    expect(state.error).toBe("page-fail");
    // retry sends the same cursor and owns the error
    expect(beginAuditLoadMore(state)?.cursor).toBe("c1");
    state = drive(state, { type: "loadMoreStart", generation: 0 }, { type: "loadMoreDone", identity: idA, generation: 0, page: page([older], "c2") });
    expect(state.error).toBeNull();
    expect(state.rows.map((r) => r.id)).toEqual([10, 9]);
  });

  it("filter switch with an outstanding page request: the late result never mutates the new selection", () => {
    let state = emptyAuditState(idA);
    state = drive(
      state,
      { type: "firstPage", identity: idA, page: page([first], "c1") },
      { type: "loadMoreStart", generation: 0 },
    );
    expect(state.loadMorePending).toBe(true);
    // user switches to B while A's load-more is in flight
    state = drive(state, { type: "filter", identity: idB });
    expect(state.identity).toBe(idB);
    expect(state.rows).toEqual([]);
    expect(state.generation).toBe(1);
    // the outstanding A result settles late: selection untouched, pending released
    state = drive(state, { type: "loadMoreDone", identity: idA, generation: 0, page: page([row({ id: 99, ts: "z" })], "nope") });
    expect(state.identity).toBe(idB);
    expect(state.rows).toEqual([]);
    expect(state.cursor).toBeNull();
    expect(state.loadMorePending).toBe(false);
    // B can load more on its own terms
    expect(drive(state, { type: "filter", identity: idB })).toEqual(state); // same-identity filter is a no-op
  });

  it("A -> B -> A lifetime: a gen-0 response cannot mutate the re-selected A", () => {
    let state = emptyAuditState(idA);
    state = drive(
      state,
      { type: "firstPage", identity: idA, page: page([first], "c1") },
      { type: "loadMoreStart", generation: 0 },
      { type: "filter", identity: idB },
      { type: "filter", identity: idA },
    );
    expect(state.generation).toBe(2);
    expect(state.rows).toEqual([]);
    // the stale in-flight load-more of the previous A lifetime settles
    state = drive(state, { type: "loadMoreDone", identity: idA, generation: 0, page: page([older], "stale") });
    expect(state.rows).toEqual([]);
    expect(state.cursor).toBeNull();
    expect(state.loadMorePending).toBe(false);
    // fresh first page for the new A lifetime works
    state = drive(state, { type: "firstPage", identity: idA, page: page([first], "c1") });
    expect(state.rows.map((r) => r.id)).toEqual([10]);
    expect(beginAuditLoadMore(state)?.cursor).toBe("c1");
  });

  it("clears rows and pagination on a successful empty first page; later non-empty re-establishes", () => {
    let state = drive(emptyAuditState(idA), { type: "firstPage", identity: idA, page: page([first], "c") });
    state = drive(state, { type: "firstPage", identity: idA, page: page([]) });
    expect(state.rows).toEqual([]);
    expect(state.cursor).toBeNull();
    expect(state.loadMoreUsed).toBe(false);
    state = drive(state, { type: "firstPage", identity: idA, page: page([older], "c2") });
    expect(state.rows.map((r) => r.id)).toEqual([9]);
    expect(state.cursor).toBe("c2");
  });

  it("ignores a duplicate load-more and late responses from another filter", () => {
    let state = drive(emptyAuditState(idA), { type: "firstPage", identity: idA, page: page([first], "c1") });
    state = drive(state, { type: "loadMoreStart", generation: 0 });
    expect(drive(state, { type: "loadMoreStart", generation: 0 })).toEqual(state);
    // late first page for A while showing B is ignored
    const late = drive(emptyAuditState(idB), { type: "firstPage", identity: idA, page: page([row({ id: 99, ts: "z" })], "nope") });
    expect(late.rows).toEqual([]);
    // a load-more settled for a different filter does not touch A's rows, only frees the pending flag
    state = drive(state, { type: "loadMoreDone", identity: idB, generation: 0, page: page([row({ id: 2, ts: "s" })], null) });
    expect(state.rows.map((r) => r.id)).toEqual([10]);
    expect(state.cursor).toBe("c1");
    expect(state.loadMorePending).toBe(false);
    expect(beginAuditLoadMore(state)?.cursor).toBe("c1");
  });

  it("gap fill: exhausted-looking first page -> more rows give a cursor -> older page fills without duplicates", () => {
    const r = (id: number) => row({ id, ts: `2026-07-${String(id).padStart(2, "0")}T00:00:00.000Z` });
    let state = emptyAuditState(idA);
    // initial first page has 30 rows and says exhausted (no load-more ran yet)
    const initial = Array.from({ length: 30 }, (_, i) => r(i + 1));
    state = drive(state, { type: "firstPage", identity: idA, page: page(initial, null) });
    expect(beginAuditLoadMore(state)).toBeNull();
    // 15 more rows arrive while still on first-page polling: the poll may
    // refresh the not-yet-consumed cursor instead of freezing the null
    const grown = Array.from({ length: 45 }, (_, i) => r(i + 1));
    state = drive(state, { type: "firstPage", identity: idA, page: page(grown, "c45") });
    expect(beginAuditLoadMore(state)?.cursor).toBe("c45");
    // the older page fills the gap exactly once: no duplicate ids
    const older = Array.from({ length: 15 }, (_, i) => r(i + 46));
    state = drive(state, { type: "loadMoreStart", generation: 0 });
    state = drive(state, { type: "loadMoreDone", identity: idA, generation: 0, page: page(older, "c60") });
    expect(state.rows).toHaveLength(60);
    expect(new Set(state.rows.map((x) => x.id)).size).toBe(60);
    // load-more used: later polls keep the captured cursor, checked joins keep rows
    const polled = Array.from({ length: 50 }, (_, i) => r(i + 1));
    state = drive(state, { type: "firstPage", identity: idA, page: page(polled, "c50-new") });
    expect(state.cursor).toBe("c60");
    expect(state.rows).toHaveLength(60);
  });
});