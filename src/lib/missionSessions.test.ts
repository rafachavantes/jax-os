import { describe, expect, it } from "vitest";
import {
  boardCardKeyFor, boardCards, boardEntries, bucketOf, buildTicker, cardKey, sessionDisplay, stuckBoardCards, type MissionCard,
} from "./mission";
import {
  missionOf, mkActivity, mkHubProject, mkPane, mkProject, mkSeat, mkSplit, runSummary, NOW, U,
} from "./multiSessionFixtures";
import { buildMission, type Mission } from "./mission";
import type { HubActiveRun, HubPane, HubSessionSeat, HubSessionSplit } from "../server/db/workflows";
import type { WorktreeResult } from "../server/collectors/worktrees";

const A = U("a");
const B = U("b");
const X = U("x");

function activeRun(runId: string): HubActiveRun {
  return { runId, startedAt: new Date(NOW - 5 * 60_000).toISOString(), kind: "build", runtime: "claude", target: "feat/x", count: 1, lastStep: null };
}

function seat(id: string | null, pane: string, over: Partial<HubSessionSeat> = {}): HubSessionSeat {
  return mkSeat({ sessionId: id, panes: [{ pane, tmuxIncarnation: "1:1" }], ...over });
}

function pane(p: string, over: Partial<HubPane> = {}): HubPane {
  return mkPane({ pane: p, tmuxIncarnation: "1:1", ...over });
}

function keys(m: Mission): string[] {
  return boardCards(m.model.cards).map(cardKey);
}

function cardOf(m: Mission, key: string): MissionCard {
  return boardCards(m.model.cards).find((c) => cardKey(c) === key)!;
}

describe("board decision table (spec §5)", () => {
  const project = () => mkProject({ dir: "p" });
  const cases: { name: string; split?: HubSessionSplit; expected: string[] }[] = [
    { name: "0 seats, no split", expected: ["p"] },
    {
      name: "0 seats, one unlinked run",
      split: mkSplit({ seats: [], unlinked: mkActivity({ activeRuns: [runSummary("r1")], activeRun: activeRun("r1") }) }),
      expected: ["p#unlinked"],
    },
    { name: "1 seat, none", split: mkSplit({ seats: [seat(A, "%1")] }), expected: [`p#session:${A}`] },
    {
      name: "1 seat, one unlinked run",
      split: mkSplit({ seats: [seat(A, "%1")], unlinked: mkActivity({ activeRuns: [runSummary("r1")], activeRun: activeRun("r1") }) }),
      expected: [`p#session:${A}`, "p#unlinked"],
    },
    {
      name: "2 seats, none",
      split: mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] }),
      expected: [`p#session:${A}`, `p#session:${B}`],
    },
    {
      name: "2 seats, one unlinked run",
      split: mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")], unlinked: mkActivity({ activeRuns: [runSummary("r1")], activeRun: activeRun("r1") }) }),
      expected: [`p#session:${A}`, `p#session:${B}`, "p#unlinked"],
    },
  ];
  for (const c of cases) {
    it(c.name, () => {
      const m = missionOf({ p: mkHubProject({ panes: [pane("%1"), pane("%2", { harnessSession: B })], ...(c.split ? { split: c.split } : {}) }) });
      expect(keys(m)).toEqual(c.expected);
      if (c.name === "0 seats, one unlinked run") expect(m.model.cards[0].group).toBeDefined();
    });
  }
});

describe("seat order and transitions", () => {
  it("keeps seat order when activity timestamps swap", () => {
    const splitA = mkSplit({
      seats: [
        mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }], activity: mkActivity({ newestEventTs: "2026-11-12T13:00:00.000Z" }) }),
        mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }], activity: mkActivity({ newestEventTs: "2026-11-12T14:00:00.000Z" }) }),
      ],
    });
    const splitB = mkSplit({
      seats: [
        mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }], activity: mkActivity({ newestEventTs: "2026-11-12T14:00:00.000Z" }) }),
        mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }], activity: mkActivity({ newestEventTs: "2026-11-12T13:00:00.000Z" }) }),
      ],
    });
    const mk = (split: HubSessionSplit) => missionOf({ p: mkHubProject({ panes: [pane("%1"), pane("%2", { harnessSession: B })], split }) });
    expect(keys(mk(splitA))).toEqual([`p#session:${A}`, `p#session:${B}`]);
    expect(keys(mk(splitB))).toEqual([`p#session:${A}`, `p#session:${B}`]);
  });

  it("1→2→1 keeps A's key, and the 1-seat plain card takes it", () => {
    const one = mkSplit({ seats: [seat(A, "%1")] });
    const two = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] });
    const panes = [pane("%1"), pane("%2", { harnessSession: B })];
    const m1 = missionOf({ p: mkHubProject({ panes: [panes[0]], split: one }) });
    const m2 = missionOf({ p: mkHubProject({ panes, split: two }) });
    const m1b = missionOf({ p: mkHubProject({ panes: [panes[0]], split: one }) });
    expect(keys(m1)).toEqual([`p#session:${A}`]);
    expect(keys(m2)).toEqual([`p#session:${A}`, `p#session:${B}`]);
    expect(keys(m1b)).toEqual([`p#session:${A}`]);
    expect(m1.model.cards[0].key).toBe(`p#session:${A}`);
  });

  it("2→1→0→2 returns B's build to B's card and shows it on the unlinked card in between", () => {
    const panes = [pane("%1"), pane("%2", { harnessSession: B })];
    const bRun = runSummary("rB");
    const bActive = activeRun("rB");
    const two = mkSplit({
      seats: [seat(A, "%1"), seat(B, "%2", { activity: mkActivity({ activeRuns: [bRun], activeRun: bActive }) })],
    });
    const twoModel = missionOf({ p: mkHubProject({ panes, split: two }) });
    expect(cardOf(twoModel, `p#session:${B}`).activeRuns).toHaveLength(1);

    const one = mkSplit({
      seats: [seat(A, "%1")],
      unlinked: mkActivity({ activeRuns: [bRun], activeRun: bActive }),
    });
    const oneModel = missionOf({ p: mkHubProject({ panes: [panes[0]], split: one }) });
    expect(keys(oneModel)).toEqual([`p#session:${A}`, "p#unlinked"]);
    expect(cardOf(oneModel, "p#unlinked").headline).toBe("working");
    expect(cardOf(oneModel, "p#unlinked").activeRun?.runId).toBe("rB");
    const workKeys = boardEntries(oneModel.model.cards, "working").flatMap((e) => (e.kind === "group" ? e.cards : [e.card])).map(cardKey);
    expect(workKeys).toContain("p#unlinked");

    const zero = mkSplit({
      seats: [],
      unlinked: mkActivity({ activeRuns: [bRun], activeRun: bActive }),
    });
    const zeroModel = missionOf({ p: mkHubProject({ panes: [panes[0]], split: zero }) });
    expect(keys(zeroModel)).toEqual(["p#unlinked"]);
    expect(keys(zeroModel)).not.toContain(`p#session:${A}`);
    expect(boardEntries(zeroModel.model.cards, "working").flatMap((e) => (e.kind === "group" ? e.cards : [e.card])).map(cardKey)).toEqual(["p#unlinked"]);

    const twoAgain = missionOf({ p: mkHubProject({ panes, split: two }) });
    expect(cardOf(twoAgain, `p#session:${B}`).activeRuns).toHaveLength(1);
    expect(keys(twoAgain)).not.toContain("p#unlinked");
  });

  it("a plain card whose own run is in flight reads waiting, never idle", () => {
    const running = missionOf({
      p: mkHubProject({
        panes: [pane("%1")],
        split: mkSplit({ seats: [seat(A, "%1", { activity: mkActivity({ activeRuns: [runSummary("r1")] }) })] }),
      }),
    });
    expect(running.model.cards[0].headline).toBe("waiting");
    expect(bucketOf(running.model.cards[0].headline)).toBe("working");
    const idle = missionOf({ p: mkHubProject({ panes: [pane("%1")], split: mkSplit({ seats: [seat(A, "%1")] }) }) });
    expect(idle.model.cards[0].headline).toBe("idle");
  });
});

describe("sources and ownership", () => {
  it("keeps the Codex warning and puts a closed session's run on the unlinked card", () => {
    const m = missionOf(
      { p: mkHubProject({ split: mkSplit({ seats: [], unlinked: mkActivity({ activeRuns: [runSummary("r1")], activeRun: activeRun("r1") }) }) }) },
      { codexSource: "failed" },
    );
    expect(m.model.codexSource).toBe("failed");
    expect(keys(m)).toEqual(["p#unlinked"]);
  });

  it("a Codex session that is also an open tmux seat keeps its run on that card", () => {
    const thread = U("c");
    const seatTmux = mkSeat({ sessionId: thread, transport: "tmux", panes: [{ pane: "%1", tmuxIncarnation: "1:1" }], activity: mkActivity({ activeRuns: [runSummary("r1")], activeRun: activeRun("r1") }) });
    const m = missionOf({
      p: mkHubProject({
        panes: [pane("%1", { harnessSession: thread }), pane("%2", { harnessSession: B })],
        split: mkSplit({ seats: [seatTmux, seat(B, "%2")] }),
      }),
    }, { codexSource: "failed" });
    expect(cardOf(m, `p#session:${thread}`).activeRuns).toHaveLength(1);
    expect(keys(m)).not.toContain("p#unlinked");
  });

  it("hub failure keeps the error contract and renders no group; no hub at all renders unknowns", () => {
    const failed = missionOf({ p: mkHubProject({ split: mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] }) }) }, { hubOk: false });
    expect(failed.readiness.kind).toBe("failed");
    expect(failed.model.cards[0].group).toBeUndefined();
    const loading = buildMission({ ok: true, data: { projects: [mkProject({ dir: "p" })], skipped: 0, reposRoot: "" } }, undefined, undefined, NOW);
    expect(loading.model.cards[0].group).toBeUndefined();
    expect(loading.model.cards[0].headline).toBe("unknown");
  });
});

describe("sorting by the most urgent card", () => {
  it("a gate-only project group does not outrank a working plain project; a needs-you card does", () => {
    const gatedProject = mkProject({ dir: "g", gate: "awaiting-approval" });
    const split = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] });
    const working = mkHubProject({ panes: [pane("%9", { working: true })] });
    const m = missionOf(
      { g: mkHubProject({ panes: [pane("%1"), pane("%2", { harnessSession: B })], split }), w: working },
      { projects: [gatedProject, mkProject({ dir: "w" })] },
    );
    expect(m.model.cards.map((c) => c.dir)).toEqual(["w", "g"]);

    const blocked = { status: "blocked" as const, minutes: null, declaredAt: new Date(NOW - 1000).toISOString(), eventId: 3, mergeAsk: null, question: null };
    const needs = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] });
    const m2 = missionOf(
      {
        g: mkHubProject({ panes: [pane("%1", { capsule: blocked }), pane("%2", { harnessSession: B })], split: needs }),
        w: working,
      },
      { projects: [mkProject({ dir: "g" }), mkProject({ dir: "w" })] },
    );
    expect(m2.model.cards.map((c) => c.dir)).toEqual(["g", "w"]);
  });
});

describe("session card scope", () => {
  it("two agents in two panes of one tmux session are two cards with the same tmux session", () => {
    const split = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] });
    const m = missionOf({ p: mkHubProject({ panes: [pane("%1"), pane("%2", { harnessSession: B })], split }) });
    const a = cardOf(m, `p#session:${A}`).scope!;
    const b = cardOf(m, `p#session:${B}`).scope!;
    if (a.kind !== "session" || b.kind !== "session") throw new Error("session scope expected");
    expect(a.tmuxSessions).toEqual(["s1"]);
    expect(b.tmuxSessions).toEqual(["s1"]);
    expect(a.panes).toEqual(["%1"]);
    expect(b.panes).toEqual(["%2"]);
  });

  it("one session visible in two panes is one card with both panes", () => {
    const split = mkSplit({ seats: [mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }, { pane: "%2", tmuxIncarnation: "1:1" }] }), seat(B, "%3")] });
    const m = missionOf({ p: mkHubProject({ panes: [pane("%1"), pane("%2"), pane("%3", { harnessSession: B })], split }) });
    expect(keys(m)).toHaveLength(2);
    const scope = cardOf(m, `p#session:${A}`).scope!;
    if (scope.kind !== "session") throw new Error("session scope expected");
    expect(scope.panes).toEqual(["%1", "%2"]);
  });

  it("a Codex thread also hosted by a tmux pane is one card with the tmux location", () => {
    const thread = U("c");
    const split = mkSplit({ seats: [mkSeat({ sessionId: thread, transport: "tmux", panes: [{ pane: "%1", tmuxIncarnation: "1:1" }] }), seat(B, "%2")] });
    const m = missionOf({ p: mkHubProject({ panes: [pane("%1", { harnessSession: thread }), pane("%2", { harnessSession: B })], split }) });
    expect(keys(m)).toEqual([`p#session:${thread}`, `p#session:${B}`]);
    const scope = cardOf(m, `p#session:${thread}`).scope!;
    if (scope.kind !== "session") throw new Error("session scope expected");
    expect(scope.tmuxSessions.length).toBeGreaterThan(0);
  });

  it("mixed runtimes report the right runtime and transport", () => {
    const codexThread = U("c");
    const split = mkSplit({
      seats: [
        seat(A, "%1"),
        mkSeat({ sessionId: codexThread, transport: "codex" }),
      ],
    });
    const m = missionOf({
      p: mkHubProject({
        panes: [pane("%1", { runtime: "claude" })],
        codexSessions: [{ threadId: codexThread, status: "idle", activeFlags: [], sourceOk: true, working: false, attention: false, pendingAttention: false, runInFlight: false, capsule: null, lastEventTs: null, subagentCount: 0 }],
        split,
      }),
    });
    const claude = cardOf(m, `p#session:${A}`).scope!;
    const codex = cardOf(m, `p#session:${codexThread}`).scope!;
    expect(claude.kind === "session" && claude.runtime).toBe("claude");
    expect(claude.kind === "session" && claude.transport).toBe("tmux");
    expect(codex.kind === "session" && codex.runtime).toBe("codex");
    expect(codex.kind === "session" && codex.transport).toBe("codex");
  });

  it("a pane with no recorded session is keyed by its pane key and displays as new", () => {
    const split = mkSplit({ seats: [mkSeat({ sessionId: null, panes: [{ pane: "%4", tmuxIncarnation: "1:1" }] }), seat(B, "%5")] });
    const m = missionOf({ p: mkHubProject({ panes: [pane("%4", { harnessSession: null }), pane("%5", { harnessSession: B })], split }) });
    expect(keys(m)).toContain("p#pane:%4:1:1");
    const scope = cardOf(m, "p#pane:%4:1:1").scope!;
    if (scope.kind !== "session") throw new Error("session scope expected");
    expect(scope.sessionId).toBeNull();
    expect(sessionDisplay(scope).kind).toBe("new");
  });
});

describe("ownership of pane questions and capsules", () => {
  const question = { eventId: 5, pane: "%1", tmuxIncarnation: "1:1", payload: { questions: [{ question: "Pick", options: [{ label: "a" }] }] } };

  it("a question recorded by another conversation is on no card but stays in the inbox", () => {
    const split = mkSplit({ seats: [seat(B, "%1"), seat(X, "%2")] });
    const m = missionOf({
      p: mkHubProject({
        panes: [
          pane("%1", { harnessSession: B, questionSession: X, pendingQuestion: question }),
          pane("%2", { harnessSession: X }),
        ],
        split,
      }),
    });
    for (const card of boardCards(m.model.cards)) expect(card.pendingQuestions).toEqual([]);
    expect(boardCardKeyFor(m.model.cards[0], 5)).toBeNull();
    expect(m.model.inbox.some((r) => r.kind === "question" && r.eventId === 5)).toBe(true);
  });

  it("a legacy question with no recorded session stays with the seat holding the pane, plain or grouped", () => {
    const one = mkSplit({ seats: [seat(A, "%1")] });
    const plainModel = missionOf({ p: mkHubProject({ panes: [pane("%1", { questionSession: null, pendingQuestion: question })], split: one }) });
    expect(plainModel.model.cards[0].pendingQuestions).toHaveLength(1);
    expect(boardCardKeyFor(plainModel.model.cards[0], 5)).toBe(`p#session:${A}`);

    const two = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] });
    const grouped = missionOf({
      p: mkHubProject({
        panes: [pane("%1", { questionSession: null, pendingQuestion: question }), pane("%2", { harnessSession: B })],
        split: two,
      }),
    });
    expect(cardOf(grouped, `p#session:${A}`).pendingQuestions).toHaveLength(1);
    expect(cardOf(grouped, `p#session:${B}`).pendingQuestions).toEqual([]);
    expect(boardCardKeyFor(grouped.model.cards[0], 5)).toBe(`p#session:${A}`);
  });
});

describe("session status, project facts and worktrees", () => {
  it("session stage/branch/Now come from the seat's own runs; status.md stays on the project", () => {
    const status = { latestKind: "build" as const, latestTarget: "feat/a", lastSummary: "done a" };
    const split = mkSplit({
      seats: [
        seat(A, "%1", { activity: mkActivity({ status }) }),
        seat(B, "%2"),
      ],
    });
    const m = missionOf(
      {
        p: mkHubProject({
          panes: [pane("%1"), pane("%2", { harnessSession: B })],
          split,
        }),
      },
      { projects: [mkProject({ dir: "p", stage: "test", branch: "status-branch", builder: "builder-x", now: "status-now", gate: "awaiting-approval" })] },
    );
    const a = cardOf(m, `p#session:${A}`);
    const b = cardOf(m, `p#session:${B}`);
    expect(a.sessionStatus).toEqual({ stage: "build", branch: "feat/a", now: "done a" });
    expect(a.stage).toBe("build");
    expect(a.branch).toBe("feat/a");
    expect(b.sessionStatus).toBeNull();
    for (const c of [a, b]) {
      expect(c.gate).toBeNull();
      expect(c.builder).toBe("");
      expect(c.now).toBe("");
      expect(c.residuals).toEqual([]);
    }
    const projectCard = m.model.cards[0];
    expect(projectCard.stage).toBe("test");
    expect(projectCard.branch).toBe("status-branch");
    expect(projectCard.builder).toBe("builder-x");
    expect(projectCard.now).toBe("status-now");
    expect(projectCard.gate).toBe("awaiting-approval");
  });

  it("retained worktrees land on their session's card and the rest on the footer", () => {
    const split = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] });
    const worktrees: Record<string, WorktreeResult> = {
      p: { ok: true, count: 3, oldestAgeDays: 2, sessions: [{ session: A, count: 1 }, { session: null, count: 1 }, { session: "closed", count: 1 }] },
    };
    const m = missionOf({ p: mkHubProject({ panes: [pane("%1"), pane("%2", { harnessSession: B })], split }) }, { worktrees });
    expect(cardOf(m, `p#session:${A}`).retainedWorktrees).toBe(1);
    expect(m.model.cards[0].group!.footer).toEqual({ retainedWorktrees: 2, unownedLastRun: null });
  });

  it("no worktree session list means a null footer count, and old worktrees never create a card", () => {
    const split = mkSplit({ seats: [seat(A, "%1")] });
    const worktrees: Record<string, WorktreeResult> = { p: { ok: true, count: 3, oldestAgeDays: 2 } };
    const m = missionOf({ p: mkHubProject({ panes: [pane("%1")], split }) }, { worktrees });
    expect(m.model.cards[0].group).toBeUndefined();
    expect(keys(m)).toEqual([`p#session:${A}`]);

    const lastRun = { runId: "r9", role: "builder" as const, kind: "build" as const, runtime: "claude" as const, target: null, finishedAt: "2026-11-12T10:00:00.000Z", outcome: "success", contractStatus: "ok" as const, stage: "build" as const, diagnostic: null, headSha: null, findings: null, runtimeModel: null, profileName: null, reportRel: null, targetRel: null, verifyCommand: null, buildCommand: null };
    const splitTwo = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")], unownedLastRun: lastRun });
    const m2 = missionOf({ p: mkHubProject({ panes: [pane("%1"), pane("%2", { harnessSession: B })], split: splitTwo }) }, { worktrees });
    expect(m2.model.cards[0].group!.footer!.retainedWorktrees).toBeNull();
  });

  it("an unowned last run appears in the footer, and no footer when nothing is left", () => {
    const lastRun = { runId: "r9", role: "builder" as const, kind: "build" as const, runtime: "claude" as const, target: "feat/x", finishedAt: "2026-11-12T10:00:00.000Z", outcome: "success", contractStatus: "ok" as const, stage: "build" as const, diagnostic: null, headSha: null, findings: null, runtimeModel: null, profileName: null, reportRel: null, targetRel: null, verifyCommand: null, buildCommand: null };
    const split = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")], unownedLastRun: lastRun });
    const m = missionOf({ p: mkHubProject({ panes: [pane("%1"), pane("%2", { harnessSession: B })], split }) });
    expect(m.model.cards[0].group!.footer!.unownedLastRun!.labelKey).toBe("buildOk");

    const empty = mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] });
    const m2 = missionOf({ p: mkHubProject({ panes: [pane("%1"), pane("%2", { harnessSession: B })], split: empty }) });
    expect(m2.model.cards[0].group!.footer).toBeNull();
  });
});

describe("unlinked card", () => {
  it("lists every in-flight run with the newest as activeRun", () => {
    const newer = { ...runSummary("r9"), startedAt: new Date(NOW - 60_000).toISOString() };
    const older = { ...runSummary("r8"), startedAt: new Date(NOW - 600_000).toISOString() };
    const m = missionOf({
      p: mkHubProject({
        split: mkSplit({ seats: [], unlinked: mkActivity({ activeRuns: [newer, older], activeRun: activeRun("r9") }) }),
      }),
    });
    const unlinked = cardOf(m, "p#unlinked");
    expect(unlinked.activeRuns).toHaveLength(2);
    expect(unlinked.activeRun?.runId).toBe("r9");
  });
});

describe("board consumers: pills, inbox rows, stuck rail, ticker", () => {
  it("a grouped project emits no gate row; the identical plain project keeps it", () => {
    const blocked = { status: "blocked" as const, minutes: null, declaredAt: new Date(NOW - 1000).toISOString(), eventId: 3, mergeAsk: null, question: null };
    const grouped = missionOf({
      p: mkHubProject({
        panes: [pane("%1", { capsule: blocked }), pane("%2", { harnessSession: B })],
        split: mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] }),
      }),
    }, { projects: [mkProject({ dir: "p", gate: "awaiting-approval" })] });
    expect(grouped.model.inbox.some((r) => r.kind === "gate")).toBe(false);

    const plain = missionOf({
      p: mkHubProject({ panes: [pane("%1", { capsule: blocked })] }),
    }, { projects: [mkProject({ dir: "p", gate: "awaiting-approval" })] });
    expect(plain.model.inbox.some((r) => r.kind === "gate")).toBe(true);
  });

  it("one question row per event, answerability copied", () => {
    const q11 = { eventId: 11, pane: "%1", tmuxIncarnation: "1:1", payload: { questions: [{ question: "a?", options: [{ label: "x" }] }] } };
    const q12 = { eventId: 12, pane: "%2", tmuxIncarnation: "1:1", payload: { questions: [{ question: "b?", options: [{ label: "y" }] }] } };
    const m = missionOf({
      p1: mkHubProject({ panes: [pane("%1", { pendingQuestion: q11 })] }),
      p2: mkHubProject({ panes: [pane("%2", { pendingQuestion: q12, questionAnswerable: false })] }),
    });
    const rows = m.model.inbox.filter((r) => r.kind === "question");
    expect(rows.map((r) => (r.kind === "question" ? r.eventId : 0))).toEqual([11, 12]);
    expect(rows[1]).toMatchObject({ answerable: false });
  });

  it("attention rows carry the capsule event id in both transports, null when absent", () => {
    const tmuxCapsule = { status: "needs_input" as const, minutes: null, declaredAt: new Date(NOW - 1000).toISOString(), eventId: 31, mergeAsk: null, question: null };
    const codexCapsule = { status: "needs_input" as const, minutes: null, declaredAt: new Date(NOW - 1000).toISOString(), eventId: 41, mergeAsk: null, question: null };
    const m = missionOf({
      t: mkHubProject({ panes: [pane("%1", { capsule: tmuxCapsule })] }),
      c: mkHubProject({ codexSessions: [{ threadId: U("c"), status: "idle", activeFlags: [], sourceOk: true, working: false, attention: true, pendingAttention: false, runInFlight: false, capsule: codexCapsule, lastEventTs: null, subagentCount: 0 }] }),
      n: mkHubProject({ codexSessions: [{ threadId: U("n"), status: "idle", activeFlags: [], sourceOk: true, working: false, attention: false, pendingAttention: true, runInFlight: false, capsule: null, lastEventTs: null, subagentCount: 0 }] }),
    });
    const rows = m.model.inbox.filter((r) => r.kind === "attention");
    expect(rows).toContainEqual(expect.objectContaining({ transport: "tmux", capsuleEventId: 31 }));
    expect(rows).toContainEqual(expect.objectContaining({ transport: "codex", threadId: U("c"), capsuleEventId: 41 }));
    expect(rows).toContainEqual(expect.objectContaining({ transport: "codex", threadId: U("n"), capsuleEventId: null }));
  });

  it("the stuck rail lists session cards, not project cards", () => {
    const stuckCapsule = { status: "waiting" as const, minutes: 1, declaredAt: new Date(NOW - 60 * 60_000).toISOString(), eventId: 1, mergeAsk: null, question: null };
    const grouped = missionOf({
      g: mkHubProject({
        panes: [pane("%1", { capsule: stuckCapsule }), pane("%2", { harnessSession: B })],
        split: mkSplit({ seats: [seat(A, "%1"), seat(B, "%2")] }),
      }),
    });
    expect(stuckBoardCards(grouped.model).map(cardKey)).toEqual([`g#session:${A}`]);

    const plain = missionOf({ p: mkHubProject({ panes: [pane("%1", { capsule: stuckCapsule })] }) });
    expect(stuckBoardCards(plain.model).map(cardKey)).toEqual(["p"]);
  });

  it("the ticker keeps project-card events once, not per session card", () => {
    const events = [
      { id: 1, ts: "2026-11-12T11:00:00.000Z", type: "question", pane: "%1", role: "lead", emitter: "claude", payload: null },
      { id: 2, ts: "2026-11-12T11:01:00.000Z", type: "turn-stopped", pane: "%2", role: "lead", emitter: "claude", payload: null },
    ];
    const m = missionOf({
      p: mkHubProject({
        panes: [pane("%1"), pane("%2", { harnessSession: B })],
        timeline: events,
        split: mkSplit({
          seats: [
            seat(A, "%1", { activity: mkActivity({ timeline: [events[0]] }) }),
            seat(B, "%2", { activity: mkActivity({ timeline: [events[1]] }) }),
          ],
        }),
      }),
    });
    const ticker = buildTicker(m.model.cards, [], {}, NOW);
    expect(ticker.filter((i) => i.source === "event")).toHaveLength(2);
  });
});
