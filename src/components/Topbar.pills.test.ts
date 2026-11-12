import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({ mission: new Map<string, { data: unknown }>(), calls: [] as string[] }));

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key, useLocale: () => "en-US" }));
vi.mock("next/navigation", () => ({ usePathname: () => "/" }));
vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: React.ReactNode }) => createElement("a", { href }, children),
}));
vi.mock("./MobileNav", () => ({ MobileNav: () => null }));
vi.mock("./LanguageSelector", () => ({ LanguageSelector: () => null }));
vi.mock("./ThemeToggle", () => ({ ThemeToggle: () => null }));
vi.mock("./mission/AfkToggle", () => ({ AfkToggle: () => null }));
vi.mock("@/lib/useMission", () => ({
  useMission: (endpoint: string) => {
    harness.calls.push(endpoint);
    return harness.mission.get(endpoint) ?? { data: undefined };
  },
}));
vi.mock("@/lib/useTokensApi", () => ({ useTokensApi: () => ({ data: undefined }) }));

const board = vi.hoisted(() => ({ mission: undefined as unknown }));
vi.mock("@/lib/mission", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mission")>();
  return { ...actual, buildMission: () => board.mission };
});

import type { Mission, MissionCard } from "@/lib/mission";
import { Topbar } from "./Topbar";

const base: MissionCard = {
  dir: "p", name: "p", stage: "build", branch: "main", builder: "claude-code",
  gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-11-10T10:00:00-03:00",
  headline: "idle", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
  codexSource: "ok", codexUntracked: false,
  heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};

const mission: Mission = {
  readiness: { kind: "ready" },
  model: {
    cards: [
      {
        ...base, headline: "needs-you",
        group: {
          cards: [
            { ...base, key: "p#session:A", headline: "needs-you" },
            { ...base, key: "p#session:B", headline: "working" },
            { ...base, key: "p#session:C", headline: "idle" },
          ],
          footer: null,
        },
      },
      { ...base, dir: "w", headline: "working" },
      {
        ...base, dir: "hidden", visibility: "hidden",
        group: { cards: [{ ...base, dir: "hidden", key: "hidden#session:H", visibility: "hidden", headline: "needs-you" }], footer: null },
      },
    ],
    uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
    inbox: [], inboxTotalCount: 0, inboxPrRemainder: 0, prTruncatedAny: false,
    codexSource: "ok", tmuxSource: "ok", archiveAfterDays: 14, hiddenCount: 1, autoHiddenCount: 0, reposRoot: "",
  },
};

describe("Topbar pill counts board cards, never group headers or hidden cards", () => {
  it("counts the needs-you card once (group header ignored) and polls only projects + hub", () => {
    board.mission = mission;
    harness.calls = [];
    harness.mission.set("projects", { data: { ok: true, data: { projects: [], skipped: 0 } } });
    harness.mission.set("hub", { data: { ok: true, data: { byProject: {}, historicalTruncated: false, codexSource: "ok", tmuxSource: "ok" } } });
    const html = renderToStaticMarkup(createElement(Topbar, { initialTheme: "dark", sidebar: "expanded", onToggleSidebar: () => {} }));
    // needs-you: group A card + hidden group card (skipped) → 1; working: B + plain w → 2.
    expect(html).toContain("1 states.needs-you");
    expect(html).not.toContain("2 states.needs-you");
    expect([...new Set(harness.calls)]).toEqual(["projects", "hub"]);
    expect(harness.calls).not.toContain("worktrees");
  });
});
