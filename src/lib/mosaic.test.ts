import { describe, expect, it } from "vitest";
import { mosaicGroups, mosaicPageState, selectSpotlightPane, type MosaicGroup } from "./mosaic";
import type { MissionCard, MissionPane, MissionReadiness } from "./mission";

const pane = (over: Partial<MissionPane> = {}): MissionPane => ({
  pane: "%1", tmuxIncarnation: "1:1", session: "s1", tmuxSession: "s1", role: "lead",
  state: "idle", lastEventTs: null, pendingQuestion: null, capsuleStatus: null,
  capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null,
  subagentCount: 0, capsuleEventId: null, live: true, ...over,
});
const card = (over: Partial<MissionCard> = {}): MissionCard => ({
  dir: "p1", name: "p1", stage: "build", branch: "main", builder: "claude-code", gate: null,
  legacyGateField: false, now: "", residuals: [], updated: "", headline: "idle", hubReady: true,
  lastEventAgo: null, freshnessCount: null, freshnessWarning: false, sessions: [], pendingQuestions: [],
  timeline: [], pr: { status: "unknown" }, codexSource: "ok", codexUntracked: false, heartbeat: [],
  lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible", ...over,
});

describe("mosaicGroups (Sessions mosaic spec §5 Decisions 1/2)", () => {
  it("groups only cards with at least one LIVE tmux pane", () => {
    const withPane = card({ dir: "p1", sessions: [{ transport: "tmux", session: "s1", panes: [pane()] }] });
    const noPane = card({ dir: "p2", sessions: [] });
    expect(mosaicGroups([withPane, noPane])).toEqual([{ dir: "p1", name: "p1", panes: [pane()] }]);
  });

  it("a card whose ONLY pane has live:false gets no group at all — a stale pendingQuestion never resurrects it (round-1 F1)", () => {
    const staleQ = { pane: "%1", eventId: 1, questions: [{ question: "q?", multiSelect: false, options: [] }] };
    const deadWithQuestion = card({
      dir: "p1",
      sessions: [{ transport: "tmux", session: "s1", panes: [pane({ live: false, pendingQuestion: staleQ })] }],
    });
    expect(mosaicGroups([deadWithQuestion])).toEqual([]);
  });

  it("a codex-only card (no tmux panes at all) contributes no group (non-goal: Codex-native sessions)", () => {
    const codexOnly = card({ dir: "p1", sessions: [{ transport: "codex", threadId: "t1", state: "working", lastEventTs: null, subagentCount: 0, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null }] });
    expect(mosaicGroups([codexOnly])).toEqual([]);
  });
});

describe("selectSpotlightPane (Sessions mosaic spec §8 Decision 16, round-2 F3)", () => {
  it("returns null when no pane needs Rafa", () => {
    const groups: MosaicGroup[] = [{ dir: "p1", name: "p1", panes: [pane({ state: "working" })] }];
    expect(selectSpotlightPane(groups)).toBeNull();
  });

  it("the FIRST needs-you pane in traversal order wins, even across groups — 'oldest pending question wins'", () => {
    const groups: MosaicGroup[] = [
      { dir: "p1", name: "p1", panes: [pane({ pane: "%1", state: "idle" })] },
      { dir: "p2", name: "p2", panes: [pane({ pane: "%2", capsuleStatus: "blocked" }), pane({ pane: "%3", capsuleStatus: "needs_input" })] },
    ];
    expect(selectSpotlightPane(groups)).toEqual({ dir: "p2", pane: pane({ pane: "%2", capsuleStatus: "blocked" }) });
  });

  it("a pane with tmuxSession:null is still spotlight-eligible (round-2 F1 — only its click/Take-Control path is disabled, never its spotlight/attention rendering, spec §8 Decision 13a)", () => {
    const groups: MosaicGroup[] = [{ dir: "p1", name: "p1", panes: [pane({ tmuxSession: null, capsuleStatus: "needs_input" })] }];
    expect(selectSpotlightPane(groups)).toEqual({ dir: "p1", pane: pane({ tmuxSession: null, capsuleStatus: "needs_input" }) });
  });
});

describe("mosaicPageState (Sessions mosaic spec §7 Decision 18, round-2 F4)", () => {
  const READY: MissionReadiness = { kind: "ready" };
  const LOADING: MissionReadiness = { kind: "loading", pending: ["projects"] };
  const FAILED: MissionReadiness = { kind: "failed", failed: ["hub"], pending: [] };
  it("loading and failed pass through regardless of groups", () => {
    expect(mosaicPageState(LOADING, [])).toBe("loading");
    expect(mosaicPageState(FAILED, [{ dir: "p1", name: "p1", panes: [pane()] }])).toBe("failed");
  });
  it("ready with zero groups is empty; ready with any group is ready", () => {
    expect(mosaicPageState(READY, [])).toBe("empty");
    expect(mosaicPageState(READY, [{ dir: "p1", name: "p1", panes: [pane()] }])).toBe("ready");
  });
});
