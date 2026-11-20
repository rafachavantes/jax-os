import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  results: new Map<string, { data: unknown; isError: boolean }>(),
  calls: [] as { key: string; enabled?: boolean }[],
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string, values?: Record<string, unknown>) =>
    values && typeof values.count === "number" ? `${key}:${values.count}` : key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({}),
  useQuery: (opts: { queryKey: unknown[]; enabled?: boolean }) => {
    const key = opts.queryKey.join(":");
    harness.calls.push({ key, enabled: opts.enabled });
    return harness.results.get(key) ?? { data: undefined, isError: false, dataUpdatedAt: 0 };
  },
}));

import MissionPage, { homeViewClickPlan, homeViewHref, parseHomeView, readHomeLocation } from "@/app/page";

const PROJECT = {
  dir: "Alpha", name: "Alpha", stage: "build", gate: null, legacyGateField: false,
  builder: "claude-code", branch: "main", updated: "2026-11-18T10:00:00.000Z", now: "working",
  residuals: [], statusMtime: "2026-11-18T10:00:00.000Z", archived: false,
};
const HUB = {
  byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok",
  prefs: {}, settings: { archiveAfterDays: 14 },
};
const COCKPIT = { ok: true, data: { revision: 1, layout: null, prScope: "mine" } };

function seed(over: Record<string, { data: unknown; isError: boolean }> = {}) {
  harness.results.clear();
  harness.calls.length = 0;
  const base: Record<string, { data: unknown; isError: boolean }> = {
    "mission:cockpit-prefs": { data: COCKPIT, isError: false },
    "mission:projects:": { data: { ok: true, data: { projects: [PROJECT], skipped: 0, reposRoot: "/repos" } }, isError: false },
    "mission:hub:": { data: { ok: true, data: HUB }, isError: false },
    "mission:activity:": { data: { ok: true, data: [] }, isError: false },
    "mission:worktrees:": { data: { ok: true, data: {} }, isError: false },
    "mission:alerts": { data: { ok: true, data: [] }, isError: false },
    "mission:alerts:null": { data: { ok: true, data: [] }, isError: false },
    "health:24h": { data: { ok: true, data: { latest: null, cores: 8, series: [], spark: [], services: null } }, isError: false },
    "mission-block:current": { data: { ok: true, data: null }, isError: false },
  };
  for (const [key, value] of Object.entries({ ...base, ...over })) harness.results.set(key, value);
}

const render = () => renderToStaticMarkup(createElement(MissionPage));

beforeEach(() => {
  seed();
});

describe("Home cockpit hierarchy (M2 plan Task 5, spec AC1/AC2/AC10)", () => {  it("keeps the duplicate permanent count row out and renders the compact Projects controls", () => {
    const html = render();
    expect(html).toContain("projects-controls");
    expect(html).toContain("server-strip");
    expect(html).not.toContain("pitHint");
  });

  it("puts navigation and filters in the card toolbar and server resources last", () => {
    seed({ "mission:prs:scope=mine": { data: { ok: true, data: { Alpha: { ok: true, prs: [], truncated: false } } }, isError: false } });
    const html = render();
    const controls = html.slice(html.indexOf('data-testid="projects-controls"'), html.indexOf('class="project-grid"'));
    expect(controls).toContain('aria-label="label"');
    expect(controls).toContain("controls.filter");
    expect(controls).toContain("controls.mine");
    expect(controls).toContain("controls.display");
    expect(html.indexOf('data-testid="server-strip"')).toBeGreaterThan(html.indexOf('href="/audit"'));
  });

  it("local cards, attention and resources render while the PR query is still unresolved", () => {
    // No "mission:prs:scope=mine" entry: the query is in flight/delayed.
    const html = render();
    expect(html).toContain("Alpha");
    expect(html).toContain("server-strip");
    expect(html).toContain("home-projects");
    expect(html).toContain("home-activity");
    expect(harness.calls.some((c) => c.key === "mission:prs:scope=mine")).toBe(true);
  });

  it("preferences unavailable: no PR request is issued and the board carries the explicit unavailable state", () => {
    seed({ "mission:cockpit-prefs": { data: undefined, isError: true } });
    const html = render();
    expect(harness.calls.filter((c) => c.key.startsWith("mission:prs:")).every((c) => c.enabled === false)).toBe(true);
    expect(html).toContain("controls.unavailable");
    expect(html).toContain('data-width="third"');
  });

  it("Mine/All scope comes from the preference snapshot and keys the PR query", () => {
    seed({ "mission:cockpit-prefs": { data: { ok: true, data: { revision: 2, layout: null, prScope: "all" } }, isError: false } });
    render();
    expect(harness.calls.some((c) => c.key === "mission:prs:scope=all")).toBe(true);
  });

  it("?view stores the Activity selection while preserving other URL parameters, and both surfaces stay mounted", () => {
    expect(parseHomeView("view=activity&foo=1")).toBe("activity");
    expect(parseHomeView("foo=1")).toBe("projects");
    expect(homeViewHref("/", "view=activity&foo=1", "projects")).toBe("/?view=projects&foo=1");
    expect(homeViewHref("/", "", "activity")).toBe("/?view=activity");
    const html = render();
    expect(html).toContain("view=projects");
    expect(html).toContain("view=activity");
    expect(html).toContain("home-projects");
    expect(html).toContain("home-activity");
    expect(html).toMatch(/home-projects"[^>]*class="flex/);
    expect(html).toMatch(/home-activity"[^>]*class="hidden"/);
  });

  it("the mission block is collapsible (details) so progress is reachable without a large permanent area", () => {
    seed({
      "mission-block:current": { data: { ok: true, data: {
        id: 1, name: "M", goal: "g", statusLine: "s", state: "active",
        startedAt: "2026-11-18T00:00:00.000Z", endedAt: null, autonomous: false,
        milestones: [{ title: "m1", state: "done" }],
      } }, isError: false },
    });
    const html = render();
    expect(html).toContain("<details");
    expect(html).toContain("m1");
  });
});

// M2 plan F1 (RED): a real buildMission over a project with five live unresolved questions —
// validated holders on the visible card — must keep the same counted attention title, the
// bounded preview and the full-set control while the PR source is pending, failed as a whole, or
// failed for that one project. PRs no longer determine attention membership.
const questionPayload = (eventId: number) => ({
  questions: [{ question: `Pergunta ${eventId}?`, options: [{ label: "sim" }] }],
});
const livePane = (eventId: number) => ({
  pane: `%${eventId}`, tmuxIncarnation: `${eventId}:1`, session: "s1", role: "lead",
  live: true, working: false, capsule: null,
  pendingQuestion: { eventId, pane: `%${eventId}`, tmuxIncarnation: `${eventId}:1`, payload: questionPayload(eventId) },
  lastEventTs: "2026-11-18T10:00:00.000Z", subagentCount: 0, runtime: "claude",
});
const HUB_WITH_REQUESTS = {
  ...HUB,
  byProject: {
    Alpha: {
      panes: [1, 2, 3, 4, 5].map(livePane), codexSessions: [], codexUntracked: false,
      pendingQuestion: null, freshnessCount: 0, newestEventTs: null, timeline: [],
      activeRuns: [], lastRun: null, heartbeat: [], lastAction: null,
    },
  },
};

function expectCountedAttention(html: string) {
  // The count-bearing title (mock renders `title:<count>`), never the empty/neutral fallbacks.
  expect(html).toContain("title:5");
  expect(html).not.toContain("titleEmpty");
  expect(html).not.toContain("titleUnknown");
  // Bounded preview: first three requests + the full-set control; the rest stay behind it.
  expect(html).toContain("Pergunta 1?");
  expect(html).toContain("Pergunta 3?");
  expect(html).not.toContain("Pergunta 4?");
  expect(html).toContain("showAll:5");
}

describe("Home cockpit — attention count independent of PR readiness (M2 plan F1)", () => {
  it("keeps the counted attention title while the PR query is still pending", () => {
    seed({ "mission:hub:": { data: { ok: true, data: HUB_WITH_REQUESTS }, isError: false } });
    expectCountedAttention(render());
  });

  it("keeps the counted attention title while the PR envelope failed", () => {
    seed({
      "mission:hub:": { data: { ok: true, data: HUB_WITH_REQUESTS }, isError: false },
      "mission:prs:scope=mine": { data: { ok: false, error: "gh down" }, isError: false },
    });
    expectCountedAttention(render());
  });

  it("keeps the counted attention title while one project's own PR fetch failed", () => {
    seed({
      "mission:hub:": { data: { ok: true, data: HUB_WITH_REQUESTS }, isError: false },
      "mission:prs:scope=mine": { data: { ok: true, data: { Alpha: { ok: false, error: "gh down" } } }, isError: false },
    });
    expectCountedAttention(render());
  });

  it("a genuine hub failure still shows the attention source warning, never a confident empty title", () => {
    seed({ "mission:hub:": { data: { ok: false, error: "hub down" }, isError: false } });
    const html = render();
    expect(html).toContain("titleUnknown");
    expect(html).not.toContain("titleEmpty");
  });
});

const plainClick = (over: Partial<{ button: number; metaKey: boolean; ctrlKey: boolean; shiftKey: boolean; altKey: boolean }> = {}) => ({
  button: 0, metaKey: false, ctrlKey: false, shiftKey: false, altKey: false, ...over,
});

describe("Home view navigation keeps the mounted board (M2 plan Task 2)", () => {
  it("an ordinary primary click intercepts and pushes the meaningful view URL, preserving other parameters", () => {
    expect(homeViewClickPlan("projects", "activity", plainClick(), "/", "?view=projects&foo=1"))
      .toEqual({ intercept: true, push: true, href: "/?view=activity&foo=1" });
  });

  it("clicking the selected view is swallowed without a duplicate history entry", () => {
    expect(homeViewClickPlan("projects", "projects", plainClick(), "/", ""))
      .toEqual({ intercept: true, push: false, href: "/?view=projects" });
    expect(homeViewClickPlan("activity", "activity", plainClick(), "/", "?view=activity"))
      .toEqual({ intercept: true, push: false, href: "/?view=activity" });
  });

  it.each([
    ["middle button", { button: 1 }],
    ["meta", { metaKey: true }],
    ["ctrl", { ctrlKey: true }],
    ["shift", { shiftKey: true }],
    ["alt", { altKey: true }],
  ])("a %s click keeps the browser's own navigation", (_label, over) => {
    expect(homeViewClickPlan("projects", "activity", plainClick(over), "/", ""))
      .toMatchObject({ intercept: false, push: false });
  });

  it("mount and popstate read the view from the actual location, not a stale search", () => {
    expect(readHomeLocation({ pathname: "/", search: "?view=activity&foo=1" }))
      .toEqual({ view: "activity", location: { path: "/", search: "?view=activity&foo=1" } });
    expect(readHomeLocation({ pathname: "/", search: "" }).view).toBe("projects");
  });
});
