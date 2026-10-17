import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  teams: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown },
  board: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown },
  boardEnabled: true as boolean | undefined,
  detail: { data: undefined as unknown, isLoading: false, isError: false },
  params: new URLSearchParams(),
}));

vi.mock("next/navigation", () => ({
  useSearchParams: () => harness.params,
  useRouter: () => ({ replace: vi.fn() }),
  usePathname: () => "/kanban",
}));
vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: ({ queryKey, enabled }: { queryKey: string[]; enabled?: boolean }) => {
    if (queryKey.includes("board")) {
      harness.boardEnabled = enabled ?? true;
      return harness.board;
    }
    if (queryKey.includes("teams")) return harness.teams;
    if (queryKey.includes("issue")) return harness.detail;
    return { data: { ok: true, data: [] } };
  },
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));

import KanbanPage from "./page";
import { moveTo, resolveTeamId } from "@/lib/kanbanMove";

const detailPayload = {
  description: "desc",
  subIssue: { total: 0, done: 0 },
  children: [],
  comments: [],
  attachments: [],
  parent: undefined,
  commentsTruncated: false,
  childrenTruncated: false,
  commentsIncomplete: false,
  childrenIncomplete: false,
  teamId: "t9",
  states: [{ id: "s1", name: "Todo", color: "#e2e2e2", position: 1, type: "unstarted" }],
  issue: {
    id: "i9",
    identifier: "OTHER-9",
    title: "Cross-team issue",
    priority: 0,
    url: "https://linear.app/x",
    stateId: "s1",
    createdAt: "2026-01-01T00:00:00Z",
    updatedAt: "2026-01-01T00:00:00Z",
    labels: [],
  },
};

const TEAM_MOA = { id: "t1", key: "MOA", name: "Acme" };
const TEAM_X = { id: "t2", key: "MKT", name: "Marketing" };

function teamsOk(teams: Array<typeof TEAM_MOA>) {
  return { data: { ok: true, data: teams }, isError: false, dataUpdatedAt: 1, error: undefined };
}
function emptyBoard() {
  return { data: { ok: true, data: { states: [], issues: [] } }, isError: false, dataUpdatedAt: 2, error: undefined };
}

describe("KanbanPage retained poll data", () => {
  it("shows last board under a warning", () => {
    harness.teams = teamsOk([TEAM_MOA]);
    harness.board = {
      data: { ok: true, data: { states: [], issues: [] } },
      isError: true,
      dataUpdatedAt: 2,
      error: new Error("down"),
    };
    const html = renderToStaticMarkup(createElement(KanbanPage));
    expect(html).toContain("unavailable");
    expect(html).toContain("empty");
  });

  it("shows failure without a fabricated board", () => {
    harness.teams = { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down") };
    harness.board = { data: undefined, isError: false, dataUpdatedAt: 0, error: undefined };
    expect(renderToStaticMarkup(createElement(KanbanPage))).toContain("unavailable");
  });

  it("mounts the issue overlay for an off-board deep link even when the board is unavailable", () => {
    harness.teams = teamsOk([TEAM_MOA]);
    harness.board = { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down") };
    harness.detail = { data: { ok: true, data: detailPayload }, isLoading: false, isError: false };
    harness.params = new URLSearchParams("issue=i9");
    const html = renderToStaticMarkup(createElement(KanbanPage));
    // the detail overlay mounts despite no board data
    expect(html).toContain("<dialog");
    expect(html).toContain('aria-labelledby="issue-detail-title"');
    expect(html).toContain("OTHER-9");
  });

  it("mounts the issue overlay for an issue missing from the current board", () => {
    harness.teams = teamsOk([TEAM_MOA]);
    harness.board = {
      data: {
        ok: true,
        data: {
          states: [],
          issues: [{ id: "i1", identifier: "MOA-1", title: "On board", priority: 0, url: "https://x", stateId: "s1", createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z", labels: [] }],
        },
      },
      isError: false,
      dataUpdatedAt: 2,
      error: undefined,
    };
    harness.detail = { data: { ok: true, data: detailPayload }, isLoading: false, isError: false };
    harness.params = new URLSearchParams("issue=OTHER-9");
    const html = renderToStaticMarkup(createElement(KanbanPage));
    expect(html).toContain("<dialog");
    expect(html).toContain("OTHER-9");
  });
});

describe("KanbanPage teams selection", () => {
  it("zero teams shows the translated empty state and disables the board query", () => {
    harness.teams = teamsOk([]);
    harness.board = { data: undefined, isError: false, dataUpdatedAt: 0, error: undefined };
    harness.boardEnabled = true;
    harness.params = new URLSearchParams();
    const html = renderToStaticMarkup(createElement(KanbanPage));
    expect(html).toContain("zeroTeams");
    expect(harness.boardEnabled).toBe(false);
  });

  it("an invalid URL team falls back to MOA when present", () => {
    harness.teams = teamsOk([TEAM_MOA, TEAM_X]);
    harness.board = emptyBoard();
    harness.params = new URLSearchParams("team=bogus");
    const html = renderToStaticMarkup(createElement(KanbanPage));
    // MOA is the active chip (fallbackTeam prefers MOA, not the first team); MKT stays inactive
    expect(html).toMatch(/bg-brand-soft text-brand">MOA</);
    expect(html).not.toMatch(/bg-brand-soft text-brand">MKT</);
  });

  it("an invalid URL team falls back to the first team without MOA", () => {
    harness.teams = teamsOk([TEAM_X]);
    harness.board = emptyBoard();
    harness.params = new URLSearchParams("team=bogus");
    const html = renderToStaticMarkup(createElement(KanbanPage));
    expect(html).toMatch(/bg-brand-soft text-brand">MKT</);
  });

  it("a URL team enables the board query even before teams resolves", () => {
    harness.teams = { data: undefined, isError: false, dataUpdatedAt: 0, error: undefined };
    harness.board = emptyBoard();
    harness.params = new URLSearchParams("team=t1");
    renderToStaticMarkup(createElement(KanbanPage));
    expect(harness.boardEnabled).toBe(true);
  });

  it("a warm teams cache with no URL team does not enable the board before hydration", () => {
    harness.teams = teamsOk([TEAM_MOA]);
    harness.board = emptyBoard();
    harness.params = new URLSearchParams();
    renderToStaticMarkup(createElement(KanbanPage));
    expect(harness.boardEnabled).toBe(false);
  });
});

describe("moveTo", () => {
  type BoardCache = { ok: true; data: { states: never[]; issues: { id: string; stateId: string }[] } };
  function fakeQueryClient(cache: unknown) {
    const data = { current: cache };
    return {
      getQueryData: vi.fn((_key: unknown) => data.current),
      setQueryData: vi.fn((_key: unknown, updater: unknown) => {
        data.current = typeof updater === "function" ? (updater as (c: unknown) => unknown)(data.current) : updater;
      }),
    };
  }
  function boardEnvelope(): BoardCache {
    return { ok: true, data: { states: [], issues: [{ id: "i1", stateId: "s1" }] } };
  }

  // F1: seeds/reads the real {ok:true,data} envelope, never a bare board.
  // F4: a deferred POST proves the cache shows the move WHILE pending, not
  // just in the final settled/reverted state.
  it("shows the move while the POST is pending, then reverts to the exact snapshot on a refusal", async () => {
    const qc = fakeQueryClient(boardEnvelope());
    let resolvePost!: (v: { ok: true } | { ok: false; error: string }) => void;
    const post = new Promise<{ ok: true } | { ok: false; error: string }>((r) => { resolvePost = r; });
    const opts = {
      issueId: "i1", issue: { stateId: "s1" }, targetStateId: "s2",
      reservation: new Set<string>(),
      post: vi.fn(() => post),
      onPending: vi.fn(), onGuidance: vi.fn(), onInvalidate: vi.fn(),
    };
    const inFlight = moveTo(qc as never, "t1", opts);
    await Promise.resolve(); // flush moveTo's synchronous optimistic patch before the POST settles
    expect((qc.getQueryData(["kanban", "board", "t1"]) as BoardCache).data.issues[0].stateId).toBe("s2");

    resolvePost({ ok: false, error: "nope" });
    await inFlight;
    expect((qc.getQueryData(["kanban", "board", "t1"]) as BoardCache).data.issues[0].stateId).toBe("s1");
    expect(opts.onGuidance).toHaveBeenCalledWith({ kind: "refused", error: "nope" });
  });

  it("never touches the cache for a same-column drop", async () => {
    const qc = fakeQueryClient(boardEnvelope());
    const opts = {
      issueId: "i1", issue: { stateId: "s1" }, targetStateId: "s1",
      reservation: new Set<string>(), post: vi.fn(), onPending: vi.fn(), onGuidance: vi.fn(), onInvalidate: vi.fn(),
    };
    await moveTo(qc as never, "t1", opts);
    expect(qc.setQueryData).not.toHaveBeenCalled();
    expect(opts.post).not.toHaveBeenCalled();
  });

  it("skips the optimistic patch when the cached board envelope is not ok", async () => {
    const qc = fakeQueryClient({ ok: false, error: "down" });
    const opts = {
      issueId: "i1", issue: { stateId: "s1" }, targetStateId: "s2",
      reservation: new Set<string>(), post: vi.fn().mockResolvedValue({ ok: true }),
      onPending: vi.fn(), onGuidance: vi.fn(), onInvalidate: vi.fn(),
    };
    await moveTo(qc as never, "t1", opts);
    expect(qc.setQueryData).not.toHaveBeenCalled();
  });

  // round-2 F1: settling one of two in-flight moves (refused) must not disturb the other issue's still-pending optimistic patch.
  it("a refused move reverts only its own issue, leaving another pending move's patch intact", async () => {
    const qc = fakeQueryClient({ ok: true, data: { states: [], issues: [{ id: "i1", stateId: "s1" }, { id: "i2", stateId: "s1" }] } });
    const stateOf = (id: string) => (qc.getQueryData(["kanban", "board", "t1"]) as BoardCache).data.issues.find((i) => i.id === id)?.stateId;
    let resolveA!: (v: { ok: true } | { ok: false; error: string }) => void;
    let resolveB!: (v: { ok: true } | { ok: false; error: string }) => void;
    const postA = new Promise<{ ok: true } | { ok: false; error: string }>((r) => { resolveA = r; });
    const postB = new Promise<{ ok: true } | { ok: false; error: string }>((r) => { resolveB = r; });
    const opts = (issueId: string, post: typeof postA) => ({
      issueId, issue: { stateId: "s1" }, targetStateId: "s2", reservation: new Set<string>(),
      post: vi.fn(() => post), onPending: vi.fn(), onGuidance: vi.fn(), onInvalidate: vi.fn(),
    });
    const optsA = opts("i1", postA);
    const optsB = opts("i2", postB);

    const inFlightA = moveTo(qc as never, "t1", optsA);
    await Promise.resolve();
    const inFlightB = moveTo(qc as never, "t1", optsB);
    await Promise.resolve();
    expect(stateOf("i1")).toBe("s2");
    expect(stateOf("i2")).toBe("s2");

    resolveA({ ok: false, error: "nope" });
    await inFlightA;
    expect(stateOf("i1")).toBe("s1"); // A reverted
    expect(stateOf("i2")).toBe("s2"); // B untouched, still optimistic

    resolveB({ ok: true });
    await inFlightB;
    expect(stateOf("i2")).toBe("s2"); // B applied
  });
});

describe("resolveTeamId", () => {
  // round-2 F4: an absent URL team falls back to the hydrated one, whose
  // router reflection can still be stale for a render right after hydration
  // flips `hydrated` true — the resolved value must win over the fallback.
  it("falls back to the hydrated team when the URL has none yet, still falls back further once absent from a resolved list, and trusts a URL team directly while unresolved", () => {
    expect(resolveTeamId(null, "t2", [TEAM_MOA, TEAM_X], "t1")).toBe("t2");
    expect(resolveTeamId("bogus", null, [TEAM_MOA, TEAM_X], "t1")).toBe("t1");
    expect(resolveTeamId("t2", null, null, "t1")).toBe("t2");
  });

  // round-3 F1: clicking another team after hydration updates the URL but
  // never the hydrated value — the URL team must win once it's present,
  // instead of staying frozen on the stale hydrated one.
  it("lets a URL team win over the hydrated team once the URL has one", () => {
    expect(resolveTeamId("t2", "t1", [TEAM_MOA, TEAM_X], "t1")).toBe("t2");
  });
});