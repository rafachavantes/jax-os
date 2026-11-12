import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: () => null }));
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({}),
  useQuery: () => ({ data: { ok: true, data: { integrations: { github: true } } } }),
}));

import { composeIndexedReply, type MissionCard, type MissionPane, type MissionPendingQuestion } from "@/lib/mission";
import { ProjectCard } from "./ProjectCard";
import { InboxStrip } from "./InboxStrip";
import { Mosaic, MosaicAnswerOptions } from "@/components/sessions/Mosaic";
import { MosaicCellDetail } from "@/components/sessions/MosaicCellDetail";

const stale: MissionPendingQuestion = {
  pane: "%1", eventId: 77, answerable: false,
  questions: [{ question: "Stale?", multiSelect: false, options: [{ label: "stale-opt" }] }],
};
const live: MissionPendingQuestion = {
  pane: "%2", eventId: 78,
  questions: [{ question: "Live?", multiSelect: false, options: [{ label: "alpha" }, { label: "beta" }] }],
};

function pane(over: Partial<MissionPane> & { pane: string }): MissionPane {
  return {
    tmuxIncarnation: "1:1", session: "s1", tmuxSession: "s1", role: "lead", live: true, state: "needs-you",
    lastEventTs: null, pendingQuestion: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null,
    capsuleMergeAsk: null, capsuleQuestion: null, subagentCount: 0, capsuleEventId: null, ...over,
  };
}

const card: MissionCard = {
  dir: "p", name: "p", stage: "build", branch: "main", builder: "claude-code",
  gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-11-10T10:00:00-03:00",
  headline: "needs-you", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [{ transport: "tmux", session: "s1", panes: [pane({ pane: "%1", pendingQuestion: stale }), pane({ pane: "%2", pendingQuestion: live })] }],
  pendingQuestions: [stale, live],
  timeline: [], pr: { status: "unknown" },
  codexSource: "ok", codexUntracked: false,
  heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};

const mission = {
  readiness: { kind: "ready" as const },
  model: {
    cards: [card], uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
    inbox: [{ kind: "question" as const, dir: "p", name: "p", pane: "%1", eventId: 77, question: "Stale?", answerable: false as const }],
    inboxTotalCount: 1, inboxPrRemainder: 0, prTruncatedAny: false,
    codexSource: "ok" as const, tmuxSource: "ok" as const,
    archiveAfterDays: 14, hiddenCount: 0, autoHiddenCount: 0, reposRoot: "",
  },
};

function renderMosaic(): string {
  return renderToStaticMarkup(createElement(Mosaic, {
    readiness: mission.readiness, cards: mission.model.cards, panesTail: null, busy: false,
    onAction: () => {}, onOpenDetail: () => {}, onOpenMobile: () => {},
  }));
}

describe("stale-answer agreement across card, inbox and Mosaic (spec §8.3)", () => {
  it("every surface shows the same reason and no controls for event 77", () => {
    const cardHtml = renderToStaticMarkup(createElement(ProjectCard, { card, expanded: false, onToggle: () => {} }));
    const inboxHtml = renderToStaticMarkup(createElement(InboxStrip, { mission, onExpand: () => {} }));
    const mosaicHtml = renderMosaic();
    for (const html of [cardHtml, inboxHtml, mosaicHtml]) {
      expect(html).toContain("actions.questionFromEarlierSession");
      expect(html).not.toContain("stale-opt");
    }
    // Both Mosaic containers (desktop list and mobile list) carry the unanswerable cell.
    const desktop = mosaicHtml.match(/<div class="hidden flex-col gap-5 lg:flex">([\s\S]*)<\/div><\/div><div class="flex flex-col gap-3 lg:hidden">/);
    const mobile = mosaicHtml.split('<div class="flex flex-col gap-3 lg:hidden">')[1] ?? "";
    expect(desktop).not.toBeNull();
    expect(desktop![1]).toContain('data-testid="mosaic-unanswerable"');
    expect(mobile).toContain('data-testid="mosaic-unanswerable"');
  });

  it("the mobile detail shows the reason and offers neither take-control nor a reply input", () => {
    const html = renderToStaticMarkup(createElement(MosaicCellDetail, {
      card, pane: card.sessions[0].transport === "tmux" ? card.sessions[0].panes[0] : pane({ pane: "%1" }),
      session: "s1", lines: null, busy: false, onAction: () => {}, onClose: () => {},
    }));
    expect(html).toContain("actions.questionFromEarlierSession");
    expect(html).toContain('data-testid="mosaic-unanswerable"');
    expect(html).not.toContain("stale-opt");
    expect(html).not.toContain("sessions.takeControl");
    expect(html).not.toContain("actions.freeformPlaceholder");
  });

  it("the answerable pane keeps its numbered options and targets event 78, never 77", () => {
    const mosaicHtml = renderMosaic();
    expect(mosaicHtml).toContain("1 · alpha");
    expect(mosaicHtml).toContain("2 · beta");
    const optionsHtml = renderToStaticMarkup(createElement(MosaicAnswerOptions, {
      pendingQuestion: live, busy: false, onAction: () => {},
    }));
    expect(optionsHtml).toContain("alpha");
    expect(optionsHtml).not.toContain("stale-opt");
    const posted = { event_id: live.eventId, reply: composeIndexedReply(live, [[1]]) };
    expect(posted).toEqual({ event_id: 78, reply: "1: 1" });
  });
});
