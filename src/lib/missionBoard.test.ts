import { describe, expect, it } from "vitest";
import {
  boardCardKeyFor, boardCards, boardEntries, cardKey, seatKey, sessionDisplay, sessionStatusOf,
  summarizeBuckets, unlinkedKey, projectSectionKey,
  type BoardEntry, type MissionBucket, type MissionCard, type MissionCardScope, type MissionGroup,
} from "./mission";
import type { HubSessionStatus } from "../server/db/workflows";
import { U } from "./multiSessionFixtures";

function card(over: Partial<MissionCard> & { dir: string }): MissionCard {
  return {
    name: over.dir, stage: "build", branch: "main", builder: "claude-code",
    gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-11-10T10:00:00-03:00",
    headline: "idle", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
    sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
    codexSource: "ok", codexUntracked: false,
    heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
    ...over,
  };
}

const groupOf = (project: MissionCard, cards: MissionCard[]): MissionCard => ({ ...project, group: { cards, footer: null } });

describe("sessionStatusOf", () => {
  const s = (over: Partial<HubSessionStatus>): HubSessionStatus => ({ latestKind: null, latestTarget: null, lastSummary: null, ...over });
  it("maps run kinds to stages, branch only for build/diff/merge, Now from the finished summary", () => {
    expect(sessionStatusOf(s({ latestKind: "plan", latestTarget: "docs/p.md", lastSummary: "x" })))
      .toEqual({ stage: "spec", branch: null, now: "x" });
    expect(sessionStatusOf(s({ latestKind: "build", latestTarget: "feat/a" })))
      .toEqual({ stage: "build", branch: "feat/a", now: null });
    expect(sessionStatusOf(s({ latestKind: "diff", latestTarget: "feat/a" })))
      .toEqual({ stage: "review", branch: "feat/a", now: null });
    expect(sessionStatusOf(s({ latestKind: "merge", latestTarget: "feat/a" })))
      .toEqual({ stage: "ship", branch: "feat/a", now: null });
    expect(sessionStatusOf(s({ latestKind: "spec", latestTarget: "docs/s.md" })))
      .toEqual({ stage: "spec", branch: null, now: null });
  });
  it("null status and no run are null; a blank summary leaves Now null", () => {
    expect(sessionStatusOf(null)).toBeNull();
    expect(sessionStatusOf(undefined)).toBeNull();
    expect(sessionStatusOf(s({ latestKind: null }))).toBeNull();
    expect(sessionStatusOf(s({ latestKind: "build", latestTarget: "feat/a", lastSummary: "  " })))
      .toEqual({ stage: "build", branch: "feat/a", now: null });
  });
});

describe("card keys", () => {
  it("seatKey: a session id, an unidentified pane key, and no pane at all", () => {
    expect(seatKey("p", { sessionId: "s1", panes: [] })).toBe("p#session:s1");
    expect(seatKey("p", { sessionId: null, panes: [{ pane: "%3", tmuxIncarnation: "1:2" }] })).toBe("p#pane:%3:1:2");
    expect(seatKey("p", { sessionId: null, panes: [] })).toBe("p#pane:unknown");
  });
  it("cardKey prefers the card's own key; unlinked and project-section keys are distinct", () => {
    expect(cardKey(card({ dir: "p" }))).toBe("p");
    expect(cardKey(card({ dir: "p", key: "p#session:A" }))).toBe("p#session:A");
    expect(unlinkedKey("p")).toBe("p#unlinked");
    expect(projectSectionKey("p")).toBe("p#project");
  });
});

describe("sessionDisplay", () => {
  const scope = (over: Partial<Extract<MissionCardScope, { kind: "session" }>>): Extract<MissionCardScope, { kind: "session" }> => ({
    kind: "session", sessionId: U("a"), transport: "tmux", runtime: "claude", name: null, tmuxSessions: [], panes: [], ...over,
  });
  it("name, then the 8-char short id, then new session", () => {
    expect(sessionDisplay(scope({ name: "multi-sessao" }))).toEqual({ kind: "name", text: "multi-sessao" });
    expect(sessionDisplay(scope({ name: "  " }))).toEqual({ kind: "short", text: U("a").slice(0, 8) });
    expect(sessionDisplay(scope({ sessionId: null }))).toEqual({ kind: "new", text: null });
  });
});

describe("boardCards and boardEntries", () => {
  const s1 = card({ dir: "p", key: "p#session:A", headline: "idle", scope: { kind: "session", sessionId: "A", transport: "tmux", runtime: "claude", name: null, tmuxSessions: [], panes: [] } });
  const s2 = card({ dir: "p", key: "p#session:B", headline: "working", scope: { kind: "session", sessionId: "B", transport: "tmux", runtime: "claude", name: null, tmuxSessions: [], panes: [] } });
  const project = groupOf(card({ dir: "p", headline: "working" }), [s1, s2]);

  it("a grouped project flattens to its cards", () => {
    expect(boardCards([project]).map(cardKey)).toEqual(["p#session:A", "p#session:B"]);
    expect(boardCards([card({ dir: "x" })])[0].dir).toBe("x");
  });

  it("filters cards inside a group and drops a group with no matching card", () => {
    const entries = boardEntries([project], "working");
    expect(entries).toHaveLength(1);
    expect(entries[0].kind).toBe("group");
    if (entries[0].kind === "group") expect(entries[0].cards.map(cardKey)).toEqual(["p#session:B"]);
    expect(boardEntries([project], "stuck")).toEqual([]);
    const all = boardEntries([project], null);
    expect(all).toHaveLength(1);
    if (all[0].kind === "group") expect(all[0].cards).toHaveLength(2);
  });

  it("a hidden project is omitted and a plain card is filter-matched", () => {
    const hidden = card({ dir: "h", visibility: "hidden" });
    const working = card({ dir: "w", headline: "working" });
    expect(boardEntries([hidden, working], null).map((e) => (e.kind === "plain" ? cardKey(e.card) : "group"))).toEqual(["w"]);
    expect(boardEntries([working], "idle")).toEqual([]);
  });

  it("pill parity: the cards shown for a filter equal the pill count for that bucket", () => {
    const cards = [
      card({ dir: "plain-idle", headline: "idle" }),
      card({ dir: "plain-work", headline: "working" }),
      card({ dir: "plain-wait", headline: "waiting" }),
      card({ dir: "plain-needs", headline: "needs-you" }),
      card({ dir: "plain-stuck", headline: "stuck" }),
      groupOf(card({ dir: "g", headline: "needs-you" }), [
        card({ dir: "g", key: "g#session:A", headline: "needs-you" }),
        card({ dir: "g", key: "g#session:B", headline: "working" }),
        card({ dir: "g", key: "g#session:C", headline: "idle" }),
      ]),
    ];
    const counts = summarizeBuckets({ kind: "ready" }, boardCards(cards)).counts!;
    for (const filter of ["needs-you", "stuck", "working", "idle"] as MissionBucket[]) {
      const shown = boardEntries(cards, filter).flatMap((e: BoardEntry) => (e.kind === "group" ? e.cards : [e.card]));
      expect(shown.length).toBe(counts[filter]);
    }
  });
});

describe("boardCardKeyFor", () => {
  const question = { pane: "%1", eventId: 7, questions: [{ question: "q", multiSelect: false, options: [{ label: "a" }] }] };
  const a = card({ dir: "p", key: "p#session:A", pendingQuestions: [] });
  const b = card({ dir: "p", key: "p#session:B", pendingQuestions: [question] });
  const project = groupOf(card({ dir: "p" }), [a, b]);

  it("finds the session card holding a pending question", () => {
    expect(boardCardKeyFor(project, 7)).toBe("p#session:B");
  });
  it("finds a capsule event on a Codex session card", () => {
    const codex = card({
      dir: "p", key: "p#session:C",
      sessions: [{ transport: "codex", threadId: "p#session:C", state: "needs-you", lastEventTs: null, subagentCount: 0, capsuleEventId: 41, capsuleStatus: "needs_input", capsuleMinutes: null, capsuleDeclaredAt: null }],
    });
    const g = groupOf(card({ dir: "p" }), [codex]);
    expect(boardCardKeyFor(g, 41)).toBe("p#session:C");
  });
  it("a plain card holds its own question; nobody holding yields null; a null event yields null", () => {
    expect(boardCardKeyFor(card({ dir: "q", pendingQuestions: [question] }), 7)).toBe("q");
    expect(boardCardKeyFor(project, 99)).toBeNull();
    expect(boardCardKeyFor(project, null)).toBeNull();
    expect(boardCardKeyFor(project, undefined)).toBeNull();
  });
});

// Keeps the exported MissionGroup type honest against the fixture shape above.
const _group: MissionGroup = { cards: [], footer: null };
void _group;
