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
    summarizeBuckets: () => ({ counts: { "needs-you": 0, stuck: 0, working: 0, idle: 0 }, worst: "idle" }),
  };
});
vi.mock("@/components/mission/PitCounts", () => ({ PitCounts: () => createElement("div", null, "pits") }));
vi.mock("@/components/mission/ProjectsColumn", () => ({ ProjectsColumn: () => createElement("div", null, "projects") }));
vi.mock("@/components/mission/InboxStrip", () => ({ InboxStrip: () => createElement("div", null, "inbox") }));
vi.mock("@/components/mission/ServerPanel", () => ({ ServerPanel: () => createElement("div", null, "server") }));
vi.mock("@/components/mission/MissionBlock", () => ({ MissionBlock: () => createElement("div", null, "missionblock") }));
vi.mock("@/components/mission/ActivityFeed", () => ({ ActivityFeed: () => createElement("div", null, "activity") }));

import { toggleFilter } from "@/lib/mission";
import MissionPage from "./page";

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
