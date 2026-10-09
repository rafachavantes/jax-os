// src/lib/mosaicView.test.ts
import { describe, expect, it } from "vitest";
import type { MissionCard, MissionPane } from "./mission";
import { findMobilePane } from "./mosaicView";

const cardBase: MissionCard = {
  dir: "p1", name: "Alpha", stage: "build", branch: "main", builder: "claude-code",
  gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-09-19T10:00:00.000Z",
  headline: "working", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
  codexSource: "ok", codexUntracked: false,
  heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};
function pane(overrides: Partial<MissionPane>): MissionPane {
  return {
    pane: "%1", tmuxIncarnation: "1:1", session: "s1", role: "lead", state: "working",
    lastEventTs: null, pendingQuestion: null, subagentCount: 0, capsuleEventId: null, capsuleStatus: null,
    capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null,
    live: true, tmuxSession: "s1",
    ...overrides,
  };
}

describe("findMobilePane", () => {
  it("finds the card/pane across tmux sessions, returning the pane's tmuxSession as `session`", () => {
    const p = pane({ pane: "%2" });
    const cards: MissionCard[] = [{ ...cardBase, sessions: [{ transport: "tmux", session: "s1", panes: [p] }] }];
    expect(findMobilePane(cards, "%2")).toEqual({ card: cards[0], pane: p, session: "s1" });
  });
  it("returns null when the pane id is not found, or is null", () => {
    const cards: MissionCard[] = [{ ...cardBase, sessions: [{ transport: "tmux", session: "s1", panes: [pane({})] }] }];
    expect(findMobilePane(cards, "%9")).toBeNull();
    expect(findMobilePane(cards, null)).toBeNull();
  });
  it("returns null when the matched pane's tmuxSession is null (round-3 F3 — e.g. a refresh dropped the session while the dialog was open)", () => {
    const cards: MissionCard[] = [{ ...cardBase, sessions: [{ transport: "tmux", session: "s1", panes: [pane({ pane: "%2", tmuxSession: null })] }] }];
    expect(findMobilePane(cards, "%2")).toBeNull();
  });
  it("skips codex sessions (no pane field to match)", () => {
    const cards: MissionCard[] = [{ ...cardBase, sessions: [{ transport: "codex", threadId: "t1", state: "working", lastEventTs: null, subagentCount: 0, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null }] }];
    expect(findMobilePane(cards, "%1")).toBeNull();
  });
});
