import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  results: new Map<string, { data: unknown; isError: boolean }>(),
  calls: [] as { key: string; enabled?: boolean }[],
}));

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key, useLocale: () => "en-US" }));
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

describe("Home cockpit hierarchy (M2 plan Task 5, spec AC1/AC2/AC10)", () => {
  it("keeps the duplicate permanent count row out and renders the compact Projects controls", () => {
    const html = render();
    expect(html).toContain("projects-controls");
    expect(html).toContain("server-strip");
    expect(html).not.toContain("pitHint");
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
