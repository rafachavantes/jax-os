import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  mission: {
    data: undefined as unknown,
    isError: false,
    dataUpdatedAt: 0,
    error: undefined as unknown,
  },
  perEndpoint: {} as Record<string, { data: unknown; isError: boolean; dataUpdatedAt: number; error: unknown }>,
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => harness.mission,
}));
const useMissionSpy = vi.hoisted(() => vi.fn());
vi.mock("@/lib/useMission", () => ({
  useMission: (...args: unknown[]) => {
    useMissionSpy(...args);
    return harness.perEndpoint[args[0] as string] ?? harness.mission;
  },
}));
const summarizeBucketsSpy = vi.hoisted(() => vi.fn((_readiness: unknown, _cards: unknown[]) => ({ counts: { "needs-you": 0, stuck: 0, working: 0, idle: 0 }, worst: "idle" })));
const buildMissionSpy = vi.hoisted(() => vi.fn(() => ({
  readiness: { kind: "ready" },
  model: {
    cards: [], uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
    inbox: [], inboxTotalCount: 0, inboxPrRemainder: 0, prTruncatedAny: false,
    codexSource: "ok", tmuxSource: "ok",
  },
})));
vi.mock("@/lib/mission", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mission")>();
  return {
    ...actual,
    buildMission: buildMissionSpy,
    summarizeBuckets: summarizeBucketsSpy,
  };
});
vi.mock("@/components/mission/PitCounts", () => ({ PitCounts: () => createElement("div", null, "pits") }));
vi.mock("@/components/mission/ProjectsColumn", () => ({ ProjectsColumn: () => createElement("div", null, "projects") }));
vi.mock("@/components/mission/InboxStrip", () => ({ InboxStrip: () => createElement("div", null, "inbox") }));
vi.mock("@/components/mission/ServerPanel", () => ({ ServerPanel: () => createElement("div", null, "server") }));
vi.mock("@/components/mission/MissionBlock", () => ({ MissionBlock: () => createElement("div", null, "missionblock") }));
vi.mock("@/components/mission/ActivityFeed", () => ({ ActivityFeed: () => createElement("div", null, "activity") }));

import { boardCards, toggleFilter, type MissionCard } from "@/lib/mission";
import { createAnswerDraftStore, draftKeys, draftUpdater, initialDraft, settleAnswer } from "@/components/mission/AnswerDrafts";
import MissionPage, {
  HOME_SNAPSHOT_KEY, homeBoardReady, homeViewHref, parseHomeSnapshot, parseHomeView, readHomeSnapshot, writeHomeSnapshot,
} from "./page";

describe("MissionPage retained poll data", () => {
  afterEach(() => {
    harness.perEndpoint = {};
  });

  it("shows a warning with last-success time above retained content", () => {
    harness.mission = {
      data: { ok: true, data: { projects: [], skipped: 0 } },
      isError: true,
      dataUpdatedAt: 1_700_000_000_000,
      error: new Error("down"),
    };
    const html = renderToStaticMarkup(createElement(MissionPage));
    expect(html).toContain("inbox.unavailable");
    expect(html).toContain("projects");
    expect(useMissionSpy).not.toHaveBeenCalledWith("agents", expect.anything());
  });

  it("shows a warning without fake data on initial failure", () => {
    harness.mission = { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down") };
    const html = renderToStaticMarkup(createElement(MissionPage));
    expect(html).toContain("inbox.unavailable");
  });
});

describe("MissionPage worktrees is a quiet-failure source (plan review round 1, F3)", () => {
  afterEach(() => {
    harness.perEndpoint = {};
  });

  it("never raises the page-level warning on its own, even with every other source fine", () => {
    harness.mission = { data: { ok: true, data: { projects: [], skipped: 0 } }, isError: false, dataUpdatedAt: 1_700_000_000_000, error: undefined };
    harness.perEndpoint = { worktrees: { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down") } };
    const html = renderToStaticMarkup(createElement(MissionPage));
    // fails before Step 8's page.tsx edit lands -- today's page.tsx never fetches "worktrees"
    expect(useMissionSpy).toHaveBeenCalledWith("worktrees", expect.anything());
    expect(html).not.toContain("inbox.unavailable");  // F3's own regression guard
  });
});

describe("toggleFilter — click-to-filter (round-1 F7)", () => {
  it("sets, clears (same pill again), and switches (a different pill)", () => {
    expect(toggleFilter(null, "stuck")).toBe("stuck");
    expect(toggleFilter("stuck", "stuck")).toBeNull();
    expect(toggleFilter("stuck", "working")).toBe("working");
  });
});

describe("MissionPage keeps the ticker/server grid inside the viewport (mobile overflow fix)", () => {
  it("constrains the grid and its children so mono truncate rows can't force horizontal scroll", () => {
    harness.mission = { data: { ok: true, data: { projects: [], skipped: 0 } }, isError: false, dataUpdatedAt: 1_700_000_000_000, error: undefined };
    const html = renderToStaticMarkup(createElement(MissionPage));
    expect(html).toContain('class="grid min-w-0 items-start gap-4 [&amp;&gt;*]:min-w-0 lg:grid-cols-[1.6fr_1fr]"');
    expect(html).toContain("min-w-0 max-w-[1320px]");
  });
});

describe("MissionPage renders MissionBlock above InboxStrip (spec §10)", () => {
  it("renders the mission block before the inbox strip in the output", () => {
    harness.mission = { data: { ok: true, data: { projects: [], skipped: 0 } }, isError: false, dataUpdatedAt: 1_700_000_000_000, error: undefined };
    const html = renderToStaticMarkup(createElement(MissionPage));
    expect(html).toContain("missionblock");
    expect(html.indexOf("missionblock")).toBeLessThan(html.indexOf("inbox"));
  });
});

describe("MissionPage threads githubEnabled from settings (diff review e52d9e6dc555 F1/F3)", () => {
  beforeEach(() => {
    buildMissionSpy.mockClear();
  });

  it("passes githubEnabled=false when settings say github is off (F1)", () => {
    harness.mission = {
      data: { ok: true, data: { integrations: { github: false } }, webhookConfigured: false },
      isError: false,
      dataUpdatedAt: 0,
      error: undefined,
    };
    renderToStaticMarkup(createElement(MissionPage));
    expect(buildMissionSpy).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.anything(), expect.any(Number), expect.anything(), false);
  });

  it("settings still loading (undefined data): githubEnabled stays true, not false (F1)", () => {
    harness.mission = { data: undefined, isError: false, dataUpdatedAt: 0, error: undefined };
    renderToStaticMarkup(createElement(MissionPage));
    // Every source (including the settings hook) resolves to undefined data here, so only the
    // trailing githubEnabled arg carries a defined value to assert on.
    expect(buildMissionSpy).toHaveBeenCalledWith(undefined, undefined, undefined, expect.any(Number), undefined, true);
  });

  it("settings query errored (ok:false): githubEnabled stays true, not false (F1)", () => {
    harness.mission = { data: { ok: false, error: "x" }, isError: false, dataUpdatedAt: 0, error: undefined };
    renderToStaticMarkup(createElement(MissionPage));
    expect(buildMissionSpy).toHaveBeenCalledWith(expect.anything(), expect.anything(), expect.anything(), expect.any(Number), expect.anything(), true);
  });
});

describe("MissionPage counts board cards (multi-session cards, spec §12)", () => {
  it("passes boardCards(model.cards) to summarizeBuckets, not the project cards", () => {
    const base: MissionCard = {
      dir: "p", name: "p", stage: "build", branch: "main", builder: "claude-code",
      gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-11-10T10:00:00-03:00",
      headline: "working", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
      sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
      codexSource: "ok", codexUntracked: false,
      heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
    };
    const s1 = { ...base, dir: "p", key: "p#session:A", headline: "idle" as const };
    const s2 = { ...base, dir: "p", key: "p#session:B", headline: "working" as const };
    const cards = [{ ...base, group: { cards: [s1, s2], footer: null } }];
    buildMissionSpy.mockReturnValueOnce({
      readiness: { kind: "ready" } as const,
      model: {
        cards, uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
        inbox: [], inboxTotalCount: 0, inboxPrRemainder: 0, prTruncatedAny: false,
        codexSource: "ok" as const, tmuxSource: "ok" as const, archiveAfterDays: 14, hiddenCount: 0, autoHiddenCount: 0, reposRoot: "",
      },
    } as never);
    summarizeBucketsSpy.mockClear();
    harness.mission = { data: { ok: true, data: { projects: [], skipped: 0 } }, isError: false, dataUpdatedAt: 0, error: undefined };
    renderToStaticMarkup(createElement(MissionPage));
    expect(summarizeBucketsSpy).toHaveBeenCalled();
    expect(summarizeBucketsSpy.mock.calls.at(-1)![1]).toEqual(boardCards(cards));
  });
});


describe("Home UI snapshot (M3b Task 4)", () => {
  it("round-trips filter, expansion, scroll and the ?view origin through guarded storage", () => {
    const data = new Map<string, string>();
    const storage = { getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v); } };
    writeHomeSnapshot(storage, { view: "activity", filter: "working", expandedKey: "jax-os#project", scrollTop: 412.5 });
    expect(readHomeSnapshot(storage)).toEqual({ view: "activity", filter: "working", expandedKey: "jax-os#project", scrollTop: 412.5 });
    expect(data.has(HOME_SNAPSHOT_KEY)).toBe(true);
    expect(parseHomeView(new URL(homeViewHref("/", "", "activity"), "http://x").search)).toBe("activity");
  });

  it("falls back safely on corrupt/unavailable data and rejects out-of-envelope values", () => {
    expect(parseHomeSnapshot("{not json")).toBeNull();
    expect(parseHomeSnapshot(JSON.stringify({ view: "other", filter: null, expandedKey: null, scrollTop: 0 }))).toBeNull();
    expect(parseHomeSnapshot(JSON.stringify({ view: "projects", filter: "bogus", expandedKey: null, scrollTop: 0 }))).toBeNull();
    expect(parseHomeSnapshot(JSON.stringify({ view: "projects", filter: null, expandedKey: "x".repeat(1025), scrollTop: 0 }))).toBeNull();
    expect(parseHomeSnapshot(JSON.stringify({ view: "projects", filter: null, expandedKey: null, scrollTop: -1 }))).toBeNull();
    expect(parseHomeSnapshot(JSON.stringify({ view: "projects", filter: null, expandedKey: null, scrollTop: Number.POSITIVE_INFINITY }))).toBeNull();
    const throwing = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
    expect(readHomeSnapshot(throwing)).toBeNull();
    expect(() => writeHomeSnapshot(throwing, { view: "projects", filter: null, expandedKey: null, scrollTop: 0 })).not.toThrow();
    expect(readHomeSnapshot(null)).toBeNull();
  });
});

describe("Home snapshot initialization and scroll readiness (M3b corrections)", () => {
  it("arms the scroll restore only with real projects/hub data and a non-loading board", () => {
    expect(homeBoardReady({ kind: "ready" })).toBe(true);
    expect(homeBoardReady({ kind: "degraded", degraded: ["prs"] })).toBe(true);
    expect(homeBoardReady({ kind: "loading", pending: ["hub"] })).toBe(false);
    expect(homeBoardReady({ kind: "failed", failed: ["hub"], pending: [] })).toBe(false);
  });

  it("does not read the saved snapshot during the first render, so server and client markup match", () => {
    harness.perEndpoint = {};
    harness.mission = { data: { ok: true, data: { projects: [], skipped: 0 } }, isError: false, dataUpdatedAt: 1, error: undefined };
    const storage = {
      getItem: (k: string) => k === HOME_SNAPSHOT_KEY ? JSON.stringify({ view: "activity", filter: "working", expandedKey: "x", scrollTop: 99 }) : null,
      setItem: () => {},
    };
    vi.stubGlobal("window", {
      sessionStorage: storage, location: { pathname: "/", search: "" }, addEventListener: () => {}, removeEventListener: () => {},
    });
    try {
      const html = renderToStaticMarkup(createElement(MissionPage));
      expect(html).toContain('data-testid="home-projects" class="flex flex-col gap-[22px]"');
      expect(html).toContain('data-testid="home-activity" class="hidden"');
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("answer drafts survive route content unmount (M3b Task 4)", () => {
  it("keeps a draft across remount while another question never inherits it and an answer clears only its event", () => {
    const store = createAnswerDraftStore();
    const key = draftKeys.question(11);
    let local = "";
    const update = draftUpdater(store, key, (v: string) => { local = v; });
    update("typed answer");
    // route content unmounts: a fresh form instance re-reads the same store
    expect(initialDraft(store, key, () => "")).toBe("typed answer");
    expect(initialDraft(store, draftKeys.question(12), () => "")).toBe("");
    settleAnswer(store, true, { event_id: 11 });
    expect(initialDraft(store, key, () => "")).toBe("");
    // a failed answer keeps the draft
    store.set(key, "still here");
    settleAnswer(store, false, { event_id: 11 });
    expect(initialDraft(store, key, () => "")).toBe("still here");
  });
});
