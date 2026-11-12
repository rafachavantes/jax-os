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

import type { MissionCard, MissionPane } from "@/lib/mission";
import { ProjectCard } from "./ProjectCard";
import {
  AnswerDraftsProvider, draftKeys, draftUpdater, initialDraft, settleAnswer, useAnswerDraft, createAnswerDraftStore,
} from "./AnswerDrafts";

function pane(over: Partial<MissionPane> & { pane: string }): MissionPane {
  return {
    tmuxIncarnation: "1:1", session: "s1", role: "lead", live: true, state: "needs-you",
    lastEventTs: null, pendingQuestion: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null,
    capsuleMergeAsk: null, capsuleQuestion: null, subagentCount: 0, capsuleEventId: null, tmuxSession: "s1", ...over,
  };
}

function card(over: Partial<MissionCard> = {}): MissionCard {
  return {
    dir: "p", name: "p", stage: "build", branch: "main", builder: "claude-code",
    gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-11-10T10:00:00-03:00",
    headline: "idle", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
    sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
    codexSource: "ok", codexUntracked: false,
    heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
    ...over,
  };
}

const freeformCard = (eventId: number | null): MissionCard => card({
  headline: eventId === null ? "idle" : "needs-you",
  sessions: eventId === null ? [] : [{ transport: "tmux", session: "s1", panes: [pane({ pane: "%1", capsuleEventId: eventId, capsuleStatus: "needs_input" })] }],
});

const twoQuestions = [
  { question: "Q1", multiSelect: false, options: [{ label: "one" }, { label: "two" }] },
  { question: "Q2", multiSelect: false, options: [{ label: "three" }, { label: "four" }] },
];

function renderInProvider(c: MissionCard, store: ReturnType<typeof createAnswerDraftStore>): string {
  return renderToStaticMarkup(
    createElement(AnswerDraftsProvider, { store }, createElement(ProjectCard, { card: c, expanded: true, onToggle: () => {} })),
  );
}

describe("AnswerDrafts — pure write-through", () => {
  it("updates the store before the local setter, and a different event starts empty", () => {
    const s = createAnswerDraftStore();
    const local = vi.fn();
    draftUpdater(s, draftKeys.freeform(5), local)("Use Postgres");
    expect(s.get("freeform:5")).toBe("Use Postgres");
    expect(local).toHaveBeenCalledWith("Use Postgres");
    expect(initialDraft(s, "freeform:5", () => "")).toBe("Use Postgres");
    expect(initialDraft(s, "freeform:6", () => "")).toBe("");
  });

  it("a null store falls back to the initial value and still calls the local setter", () => {
    const local = vi.fn();
    expect(initialDraft(null, "freeform:5", () => "init")).toBe("init");
    draftUpdater(null, "freeform:5", local)("x");
    expect(local).toHaveBeenCalledWith("x");
  });

  it("success clears both draft kinds for the event; failure keeps them; a body without an event clears nothing", () => {
    const s = createAnswerDraftStore();
    s.set("freeform:5", "a");
    s.set("question:5", [[1]]);
    settleAnswer(s, false, { event_id: 5 });
    expect(s.get("freeform:5")).toBe("a");
    expect(s.get("question:5")).toEqual([[1]]);
    settleAnswer(s, true, { reply: "no event id" });
    expect(s.get("freeform:5")).toBe("a");
    settleAnswer(s, true, { event_id: 5, reply: "go" });
    expect(s.get("freeform:5")).toBeUndefined();
    expect(s.get("question:5")).toBeUndefined();
  });
});

describe("AnswerDrafts — hook write-through", () => {
  it("a probe's update reaches the store and survives a remount", () => {
    const s = createAnswerDraftStore();
    let captured: { value: string; update: (v: string) => void } | null = null;
    function Probe() {
      const [value, update] = useAnswerDraft("freeform:5", () => "");
      captured = { value, update };
      return createElement("span", null, value);
    }
    renderToStaticMarkup(createElement(AnswerDraftsProvider, { store: s }, createElement(Probe)));
    expect(captured!.value).toBe("");
    captured!.update("abc");
    expect(s.get("freeform:5")).toBe("abc");
    const second = renderToStaticMarkup(createElement(AnswerDraftsProvider, { store: s }, createElement(Probe)));
    expect(captured!.value).toBe("abc");
    expect(second).toContain("abc");
  });
});

describe("AnswerDrafts — forms render the stored draft", () => {
  it("a freeform textarea recovers the stored text", () => {
    const s = createAnswerDraftStore();
    draftUpdater(s, draftKeys.freeform(5), () => {})("Use Postgres");
    const html = renderInProvider(freeformCard(5), s);
    expect(html).toContain(">Use Postgres</textarea>");
  });

  it("a multi-part question keeps its selected options", () => {
    const s = createAnswerDraftStore();
    draftUpdater(s, draftKeys.question(9), () => {})([[1], [2]]);
    const html = renderInProvider(card({ headline: "needs-you", pendingQuestions: [{ pane: "%1", eventId: 9, questions: twoQuestions }] }), s);
    expect(html).toMatch(/bg-brand-soft[^>]*>one</);
    expect(html).toMatch(/bg-brand-soft[^>]*>four</);
  });

  it("stored selections of a different length are ignored", () => {
    const s = createAnswerDraftStore();
    draftUpdater(s, draftKeys.question(9), () => {})([[1]]);
    const html = renderInProvider(card({ headline: "needs-you", pendingQuestions: [{ pane: "%1", eventId: 9, questions: twoQuestions }] }), s);
    expect(html).not.toMatch(/bg-brand-soft[^>]*>one</);
    expect(html).not.toMatch(/bg-brand-soft[^>]*>four</);
  });

  it("a disappearing question keeps its draft and restores it when the event returns", () => {
    const s = createAnswerDraftStore();
    draftUpdater(s, draftKeys.freeform(5), () => {})("half typed");
    const shown = renderInProvider(freeformCard(5), s);
    expect(shown).toContain("half typed");
    const gone = renderInProvider(freeformCard(null), s);
    expect(gone).not.toContain("half typed");
    expect(s.get("freeform:5")).toBe("half typed");
    const back = renderInProvider(freeformCard(5), s);
    expect(back).toContain("half typed");
    const other = renderInProvider(freeformCard(6), s);
    expect(other).not.toContain("half typed");
    expect(other).toContain("<textarea");
  });
});
