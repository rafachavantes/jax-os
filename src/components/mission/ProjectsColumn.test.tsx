import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
// PrefsButtons (item 2) reads a QueryClient via useQueryClient — no provider exists in these
// static-markup renders, same stand-in AfkToggle.test.tsx already uses for the same hook.
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => ({}) }));
// The real module's PrefsButtons is kept (spread from the actual module) so the hidden/
// autoHidden management rows under test render the real control, not a stand-in; only
// ProjectCard itself is replaced by the lightweight data-card marker this file already used.
vi.mock("./ProjectCard", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./ProjectCard")>();
  return {
    ...actual,
    ProjectCard: ({ card, expanded, onToggle }: { card: { dir: string; headline: string; codexSource: string }; expanded: boolean; onToggle: () => void }) => {
      onToggle(); // called during render so the test's OWN onToggleExpand closure records which dir it fired for — proves each card's onToggle is wired to ITS OWN dir, not a shared/misrouted one
      return createElement("div", { "data-card": card.dir, "data-headline": card.headline, "data-expanded": String(expanded) }, `card-${card.dir}`);
    },
  };
});
vi.mock("./SourceWarning", () => ({
  SourceWarning: ({ label, detail }: { label: string; detail: string }) =>
    createElement("div", { "data-testid": "source-warning" }, `${label} — ${detail}`),
}));

import type { Mission, MissionCard } from "@/lib/mission";
import { disclosureVisibleCards, hiddenAutoHiddenSummary, ProjectsColumn } from "./ProjectsColumn";

// Any fixed string: the component renders whatever reposRoot the server passes it.
const REPOS_ROOT_DISPLAY = "/home/rafa/repos";

function missionOf(over: Partial<Mission["model"]> = {}): Mission {
  return {
    readiness: { kind: "ready" },
    model: {
      cards: [], uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
      inbox: [], inboxTotalCount: 0, inboxPrRemainder: 0, prTruncatedAny: false,
      codexSource: "ok", tmuxSource: "ok", archiveAfterDays: 14, hiddenCount: 0, autoHiddenCount: 0, reposRoot: "", ...over,
    },
  };
}

// Phase 3: a full MissionCard base, extended with Task 5's required fields, so each card in a
// visibility fixture only overrides what its own test cares about.
const cardBase: MissionCard = {
  dir: "p1", name: "p1", stage: "build", branch: "main", builder: "claude-code",
  gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-09-10T12:00:00.000Z",
  headline: "idle", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
  codexSource: "ok", codexUntracked: false,
  heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};

function missionWithVisibility(
  cards: (Partial<MissionCard> & { dir: string })[],
  counts: { hiddenCount?: number; autoHiddenCount?: number } = {},
): Mission {
  return missionOf({
    cards: cards.map((c) => ({ ...cardBase, ...c })),
    hiddenCount: counts.hiddenCount ?? 0,
    autoHiddenCount: counts.autoHiddenCount ?? 0,
  });
}

const workingCard = { dir: "p1", headline: "working", codexSource: "ok", visibility: "visible" };

describe("ProjectsColumn — reposRoot display (spec: reposRoot/vaultPath consumers)", () => {
  it("renders the server-supplied reposRoot in the header", () => {
    const html = renderToStaticMarkup(createElement(ProjectsColumn, {
      mission: missionOf(), reposRoot: REPOS_ROOT_DISPLAY, counts: null, expandedDir: null, onToggleExpand: () => {},
    }));
    expect(html).toContain(REPOS_ROOT_DISPLAY);
  });
});

describe("ProjectsColumn — Codex source warnings (MOA-469 §4)", () => {
  it("a failed Codex source renders a warning without hiding the working card", () => {
    const html = renderToStaticMarkup(createElement(ProjectsColumn, {
      reposRoot: REPOS_ROOT_DISPLAY, expandedDir: null, onToggleExpand: () => {}, mission: missionOf({ cards: [workingCard as never], codexSource: "failed" }),
      counts: null,
    }));
    expect(html).toContain("source-warning");
    expect(html).toContain("card-p1");
    expect(html).toContain('data-headline="working"');
  });

  it("a healthy Codex source shows no warning", () => {
    const html = renderToStaticMarkup(createElement(ProjectsColumn, {
      reposRoot: REPOS_ROOT_DISPLAY, expandedDir: null, onToggleExpand: () => {}, mission: missionOf({ cards: [workingCard as never] }),
      counts: null,
    }));
    expect(html).not.toContain("source-warning");
    expect(html).toContain("card-p1");
  });

  it("a truncated Codex session list surfaces a note, not a hard warning", () => {
    const html = renderToStaticMarkup(createElement(ProjectsColumn, {
      reposRoot: REPOS_ROOT_DISPLAY, expandedDir: null, onToggleExpand: () => {}, mission: missionOf({ codexSource: "truncated" }),
      counts: null,
    }));
    expect(html).toContain("codexTruncated");
    expect(html).not.toContain("source-warning");
  });

  it("a failed tmux source renders a warning without hiding the working card (C4)", () => {
    const html = renderToStaticMarkup(createElement(ProjectsColumn, {
      reposRoot: REPOS_ROOT_DISPLAY, expandedDir: null, onToggleExpand: () => {}, mission: missionOf({ cards: [workingCard as never], tmuxSource: "failed" }),
      counts: null,
    }));
    expect(html).toContain("source-warning");
    expect(html).toContain("card-p1");
    expect(html).toContain('data-headline="working"');
  });

  it("a healthy tmux source shows no warning", () => {
    const html = renderToStaticMarkup(createElement(ProjectsColumn, {
      reposRoot: REPOS_ROOT_DISPLAY, expandedDir: null, onToggleExpand: () => {}, mission: missionOf({ cards: [workingCard as never] }),
      counts: null,
    }));
    expect(html).not.toContain("source-warning");
  });
});

describe("ProjectsColumn — filter prop (round-1 F7)", () => {
  const cards = [
    { dir: "a", headline: "working", codexSource: "ok", visibility: "visible" },
    { dir: "b", headline: "waiting", codexSource: "ok", visibility: "visible" },
    { dir: "c", headline: "needs-you", codexSource: "ok", visibility: "visible" },
    { dir: "d", headline: "idle", codexSource: "ok", visibility: "visible" },
  ];

  it("the working filter includes waiting cards, excludes the rest — omitting filter still shows everything (existing cases above)", () => {
    const html = renderToStaticMarkup(createElement(ProjectsColumn, {
      reposRoot: REPOS_ROOT_DISPLAY, expandedDir: null, onToggleExpand: () => {}, mission: missionOf({ cards: cards as never }), counts: null, filter: "working",
    }));
    // Phase 3: assert the wrapper's own marker — a bare "card-d" substring also matches the
    // `data-card-dir` attribute every rendered card now carries.
    expect(html).toContain('data-card-dir="a"');
    expect(html).toContain('data-card-dir="b"');
    expect(html).not.toContain('data-card-dir="c"');
    expect(html).not.toContain('data-card-dir="d"');
  });
});

describe("Phase 3 — visibility filtering and management bars", () => {
  it("a hidden or autoHidden card is excluded from the main grid but its count still shows", () => {
    const mission = missionWithVisibility([
      { dir: "v", visibility: "visible" as const },
      { dir: "h", visibility: "hidden" as const },
    ], { hiddenCount: 1, autoHiddenCount: 0 });
    const html = renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: null, onToggleExpand: () => {} }));
    expect(html).not.toContain('data-card-dir="h"');
    expect(html).toContain('data-card-dir="v"');
    // The file's next-intl mock drops the namespace, so the key renders bare ("hiddenBar").
    expect(html).toContain("hiddenBar");
  });

  it("a pinned card (visibility visible) always renders regardless of prefs.pinned", () => {
    const mission = missionWithVisibility([{ dir: "p", visibility: "visible" as const, prefs: { pinned: true, hiddenAt: null } }], { hiddenCount: 0, autoHiddenCount: 0 });
    expect(renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: null, onToggleExpand: () => {} }))).toContain('data-card-dir="p"');
  });

  // Plan review round 1 (72cd1de344ad) F4, reworked by Decision 17: hiding a project must not be
  // a dead end. The management rows still exist (real `<button>`s, ProjectCard.tsx's own
  // PrefsButtons with its `bucket` prop), now behind the collapsed disclosure — the summary's
  // toggle is the entry point, and disclosureVisibleCards' own test covers the opened path.
  it("a hidden card is collapsed by default behind a real gerenciar toggle, not a dead end", () => {
    const mission = missionWithVisibility(
      [{ dir: "h", name: "Hidden H", visibility: "hidden" as const, prefs: { pinned: false, hiddenAt: "2026-09-19T00:00:00.000Z" } }],
      { hiddenCount: 1, autoHiddenCount: 0 },
    );
    const html = renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: null, onToggleExpand: () => {} }));
    expect(html).toContain("<button");
    expect(html).toContain("actions.manage");
    expect(html).not.toContain('data-card-dir="h"');
    expect(html).not.toContain("Hidden H"); // the hidden summary carries no names; no per-project row until opened
  });

  it("MOA-486 follow-up: an autoHidden card's name is NOT in the collapsed summary (compact count only), mostrar toggle present, pin button not yet rendered", () => {
    const mission = missionWithVisibility(
      [{ dir: "a", name: "Auto A", visibility: "autoHidden" as const, prefs: { pinned: false, hiddenAt: null } }],
      { hiddenCount: 0, autoHiddenCount: 1 },
    );
    const html = renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: null, onToggleExpand: () => {} }));
    expect(html).not.toContain("Auto A"); // compact line: count only, no project names inline
    expect(html).toContain("projects.autoHiddenShow"); // the lowercase "mostrar" disclosure toggle
    expect(html).not.toContain("actions.pin"); // the per-project PrefsButtons row stays collapsed
  });
});

describe("ProjectsColumn — uncarded-projects collapsed toggle (MOA-486 follow-up)", () => {
  it("collapsed: the count sentence and show toggle render, names sit inside <details> markup", () => {
    const mission = missionOf({ uncardedProjectNames: ["repo-a", "repo-b"] });
    const html = renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: null, onToggleExpand: () => {} }));
    expect(html).toContain("uncarded"); // count sentence key, no {names} interpolated into it
    expect(html).toContain("uncardedShow"); // "ver" toggle label
    // the names live inside the <details> element (collapsed by default, no `open` attribute)
    const detailsMatch = html.match(/<details[^>]*>([\s\S]*)<\/details>/);
    expect(detailsMatch).not.toBeNull();
    expect(detailsMatch![1]).toContain("uncardedNames");
    expect(html).not.toMatch(/<details[^>]*\bopen\b/);
  });

  it("no uncarded names: nothing renders, not even an empty <details>", () => {
    const mission = missionOf({ uncardedProjectNames: [] });
    const html = renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: null, onToggleExpand: () => {} }));
    expect(html).not.toContain("<details");
    expect(html).not.toContain("uncarded");
  });
});

describe("hiddenAutoHiddenSummary (spec §8, Decision 17; MOA-486 follow-up: counts only, no names)", () => {
  // `t` mirrors ProjectsColumn's OWN translator, already scoped to the "mission.projects"
  // namespace (`useTranslations("mission.projects")`) — the function under test must call
  // `t("hiddenBar")`/`t("autoHiddenBar")` unscoped, the same way the component's existing
  // (pre-this-plan) code already does, never `t("projects.hiddenBar")`.
  const t = (key: string, values?: Record<string, unknown>) => (values ? `${key}:${JSON.stringify(values)}` : key);
  it("both segments render when both counts are > 0 — plain ICU-plural counts, no project names", () => {
    const s = hiddenAutoHiddenSummary(2, 3, t);
    expect(s.hidden).toBe('hiddenBar:{"count":2}');
    expect(s.autoHidden).toBe('autoHiddenBar:{"count":3}');
  });
  it("only the present segment renders when the other count is 0", () => {
    expect(hiddenAutoHiddenSummary(0, 1, t).hidden).toBeNull();
    expect(hiddenAutoHiddenSummary(3, 0, t).autoHidden).toBeNull();
  });
  it("both null when both are 0", () => {
    expect(hiddenAutoHiddenSummary(0, 0, t)).toEqual({ hidden: null, autoHidden: null });
  });
});

describe("ProjectsColumn — lifted expandedDir (spec §6, Decision 13a, round-1 F9)", () => {
  it("passes expanded={c.dir === expandedDir} and its own onToggleExpand(dir) down to each card", () => {
    const toggled: string[] = [];
    const mission = missionWithVisibility([{ ...cardBase, dir: "p1" }, { ...cardBase, dir: "p2" }]);
    const html = renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: "p2", onToggleExpand: (dir) => toggled.push(dir) }));
    expect(html).toContain('data-card="p1" data-headline="idle" data-expanded="false"');
    expect(html).toContain('data-card="p2" data-headline="idle" data-expanded="true"');
    expect(toggled).toEqual(["p1", "p2"]); // each card's own onToggle fired once, for its own dir
  });
});

describe("ProjectsColumn — hidden/auto-hidden disclosure (spec §8, Decision 17, round-1 F9; MOA-486 follow-up: compact line)", () => {
  it("renders one compact summary line — counts + toggles only, no project names, no per-project row by default (collapsed)", () => {
    const mission = missionWithVisibility([{ ...cardBase, dir: "hidden1", name: "Hidden One", visibility: "hidden" as const }, { ...cardBase, dir: "auto1", name: "Auto One", visibility: "autoHidden" as const }]);
    const html = renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: null, onToggleExpand: () => {} }));
    // Plan deviation: this file's next-intl mock drops the namespace, so the keys render bare —
    // "hiddenBar"/"autoHiddenBar", not "projects.hiddenBar" (same convention the
    // hiddenAutoHiddenSummary describe above documents).
    expect(html).toContain("hiddenBar");
    expect(html).toContain("autoHiddenBar");
    expect(html).toContain("projects.autoHiddenShow"); // lowercase "mostrar" toggle — gerenciar cannot reveal autoHidden cards (verified: setShowHidden never touches showAuto), so it stays a separate link
    expect(html).toContain("actions.manage"); // "gerenciar" toggle (unchanged key)
    // MOA-486 follow-up: the compact line drops project names entirely from BOTH halves — the
    // three-line noise ("(Patient Insight Journal (PIJ))") is gone; names only ever appear once a
    // segment is expanded (disclosureVisibleCards' own test below covers that path).
    expect(html).not.toContain("Auto One");
    expect(html).not.toContain("actions.pin");
    expect(html).not.toContain("Hidden One");
  });
  it("disclosureVisibleCards: empty when collapsed, real arrays once open, independently per segment (the open-state path, round-1 F9 — renderToStaticMarkup only shows the collapsed state above)", () => {
    const auto = [{ ...cardBase, dir: "a1" }];
    const hidden = [{ ...cardBase, dir: "h1" }, { ...cardBase, dir: "h2" }];
    expect(disclosureVisibleCards(false, false, auto, hidden)).toEqual({ auto: [], hidden: [] });
    expect(disclosureVisibleCards(true, false, auto, hidden)).toEqual({ auto, hidden: [] });
    expect(disclosureVisibleCards(false, true, auto, hidden)).toEqual({ auto: [], hidden });
    expect(disclosureVisibleCards(true, true, auto, hidden)).toEqual({ auto, hidden });
  });
});

describe("ProjectsColumn — flex-wrap rows, max 3 per row, partial rows stretch (Rafa 2026-09-20)", () => {
  it("cards are growing flex items with a 1/2/3-per-row basis; the expanded card takes a 2-of-3 basis", () => {
    const mission = missionWithVisibility([{ ...cardBase, dir: "p1" }, { ...cardBase, dir: "p2" }]);
    const html = renderToStaticMarkup(createElement(ProjectsColumn, { reposRoot: REPOS_ROOT_DISPLAY, mission, counts: null, expandedDir: "p2", onToggleExpand: () => {} }));
    const p1Tag = html.match(/<div[^>]*data-card-dir="p1"[^>]*>/)?.[0] ?? "";
    const p2Tag = html.match(/<div[^>]*data-card-dir="p2"[^>]*>/)?.[0] ?? "";
    expect(html).toContain("flex flex-wrap gap-4");
    expect(html).not.toContain("grid-cols");
    for (const tag of [p1Tag, p2Tag]) {
      expect(tag).toContain("grow");
      expect(tag).toContain("basis-full");
      expect(tag).toContain("sm:basis-[45%]");
    }
    expect(p1Tag).toContain("lg:basis-[30%]");
    expect(p2Tag).toContain("lg:basis-[63%]");
    expect(p2Tag).not.toContain("lg:basis-[30%]");
  });
});
