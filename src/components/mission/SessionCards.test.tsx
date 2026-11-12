import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: () => null }));
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({}),
  useQuery: () => ({ data: { ok: true, data: { integrations: { github: true } } } }),
}));
const postSpy = vi.hoisted(() => vi.fn(async () => ({ ok: true, data: {} })));
vi.mock("@/lib/mission", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mission")>();
  return { ...actual, postWorkflowAction: postSpy };
});

import {
  boardCardKeyFor, cardKey, type InboxRow, type Mission, type MissionBucket, type MissionCard, type MissionPane,
} from "@/lib/mission";
import { pendingUiAction, ProjectCard } from "./ProjectCard";
import { expandAndScroll, InboxStrip, QuestionRowButtons } from "./InboxStrip";

const t = (key: string) => key;

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

function pane(over: Partial<MissionPane> & { pane: string }): MissionPane {
  return {
    tmuxIncarnation: "1:1", session: "s1", role: "lead", live: true, state: "needs-you",
    lastEventTs: null, pendingQuestion: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null,
    capsuleMergeAsk: null, capsuleQuestion: null, subagentCount: 0, capsuleEventId: null, tmuxSession: "s1", ...over,
  };
}

const sessionScope = { kind: "session" as const, sessionId: "A", transport: "tmux" as const, runtime: "claude" as const, name: null, tmuxSessions: ["s1"], panes: ["%1"] };

const multiQuestion = {
  pane: "%1", eventId: 9,
  questions: [{ question: "Pick one", multiSelect: false, options: [{ label: "one" }, { label: "two" }] }],
};
const twoQuestions = {
  pane: "%1", eventId: 9,
  questions: [
    { question: "Pick one", multiSelect: false, options: [{ label: "one" }, { label: "two" }] },
    { question: "And another", multiSelect: false, options: [{ label: "three" }, { label: "four" }] },
  ],
};
const singleQuestion = {
  pane: "%1", eventId: 9,
  questions: [{ question: "Pick one", multiSelect: false, options: [{ label: "one" }, { label: "two" }] }],
};

function renderCard(c: MissionCard, expanded = false): string {
  return renderToStaticMarkup(createElement(ProjectCard, { card: c, expanded, onToggle: () => {} }));
}

function missionWithCards(cards: MissionCard[], inbox: InboxRow[]): Mission {
  return {
    readiness: { kind: "ready" },
    model: {
      cards, uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
      inbox, inboxTotalCount: inbox.length, inboxPrRemainder: 0, prTruncatedAny: false,
      codexSource: "ok", tmuxSource: "ok",
      archiveAfterDays: 14, hiddenCount: 0, autoHiddenCount: 0, reposRoot: "",
    },
  };
}

describe("unanswerable questions (spec §8.1, §8.3)", () => {
  it("the compact card shows the reason and no option buttons or Respond", () => {
    const html = renderCard(card({ dir: "p", headline: "needs-you", pendingQuestions: [{ ...singleQuestion, answerable: false }] }));
    expect(html).toContain("actions.questionFromEarlierSession");
    expect(html).not.toContain(">one<");
    expect(html).not.toContain("actions.respond");
  });

  it("a multi-part unanswerable question has no Respond either", () => {
    const multi = { pane: "%1", eventId: 9, answerable: false as const, questions: [{ question: "a", multiSelect: false, options: [{ label: "x" }] }, { question: "b", multiSelect: true, options: [{ label: "y" }] }] };
    const html = renderCard(card({ dir: "p", headline: "needs-you", pendingQuestions: [multi] }));
    expect(html).toContain("actions.questionFromEarlierSession");
    expect(html).not.toContain("actions.respond");
  });

  it("the inbox row for the same event shows the same reason and no button", () => {
    const c = card({ dir: "p", headline: "needs-you", pendingQuestions: [{ ...singleQuestion, answerable: false }] });
    const html = renderToStaticMarkup(createElement(InboxStrip, {
      mission: missionWithCards([c], [{ kind: "question", dir: "p", name: "p", pane: "%1", eventId: 9, question: "Pick one", answerable: false }]),
      onExpand: () => {},
    }));
    expect(html).toContain("actions.questionFromEarlierSession");
    expect(html).not.toContain("<button type=\"button\" class=\"rounded-md px-3 py-1.5 text-[12.5px] font-medium");
  });

  it("the expanded card shows the reason instead of the structured form, and keeps the form when answerable", () => {
    const stale = card({ dir: "p", headline: "needs-you", pendingQuestions: [{ ...multiQuestion, answerable: false }] });
    const staleHtml = renderCard(stale, true);
    expect(staleHtml).toContain("actions.questionFromEarlierSession");
    expect(staleHtml).not.toContain(">one<");
    expect(staleHtml).not.toContain("actions.submit");

    const live = card({ dir: "p", headline: "needs-you", pendingQuestions: [twoQuestions] });
    const liveHtml = renderCard(live, true);
    expect(liveHtml).toContain(">one<");
    expect(liveHtml).toContain("actions.submit");
  });
});

describe("merge approval only from the session's own ask (spec §8.5)", () => {
  const sessionWith = (over: Partial<MissionPane>): MissionCard => card({
    dir: "p", headline: "needs-you", scope: sessionScope,
    sessions: [{ transport: "tmux", session: "s1", panes: [pane({ pane: "%1", capsuleEventId: 7, capsuleStatus: "needs_input", ...over })] }],
  });

  it("a session card's plain needs_input capsule is freeform, never the project gate fallback", () => {
    const action = pendingUiAction(sessionWith({}), undefined, t);
    expect(action?.kind).toBe("freeform");
  });

  it("a session card's own merge ask lights Approve with its own branch and target", () => {
    const action = pendingUiAction(sessionWith({ capsuleMergeAsk: 1, capsuleMergeBranch: "feat/a", capsuleMergeTarget: "main" }), undefined, t);
    expect(action).toMatchObject({ kind: "approve", eventId: 7, branch: "feat/a", target: "main", replyYes: "actions.replyMergeAsk" });
  });

  it("a plain card keeps the gate fallback", () => {
    const plain = card({
      dir: "p", headline: "needs-you", gate: "awaiting-approval", branch: "feat/x",
      sessions: [{ transport: "tmux", session: "s1", panes: [pane({ pane: "%1", capsuleEventId: 7, capsuleStatus: "needs_input", capsuleMergeAsk: null })] }],
    });
    expect(pendingUiAction(plain, undefined, t)).toMatchObject({ kind: "approve", eventId: 7 });
  });

  it("a grouped project card gets no approve at all", () => {
    const project = card({
      dir: "p", headline: "needs-you", gate: "awaiting-approval",
      sessions: [{ transport: "tmux", session: "s1", panes: [pane({ pane: "%1", capsuleEventId: 7, capsuleStatus: "needs_input", capsuleMergeAsk: null })] }],
      group: { cards: [], footer: null },
    });
    expect(pendingUiAction(project, undefined, t)?.kind).not.toBe("approve");
  });
});

describe("Respond opens only the card holding the event (spec §8.4)", () => {
  afterEach(() => vi.unstubAllGlobals());

  const grouped = card({
    dir: "p", pendingQuestions: [twoQuestions],
    group: {
      cards: [
        card({ dir: "p", key: "p#session:A", scope: sessionScope, headline: "idle" }),
        card({ dir: "p", key: "p#session:B", scope: { ...sessionScope, sessionId: "B", panes: ["%2"] }, headline: "needs-you", pendingQuestions: [twoQuestions] }),
      ],
      footer: null,
    },
  });

  it("a multi-part question held by session B targets B's card", () => {
    const scrollIntoView = vi.fn();
    const querySelector = vi.fn(() => ({ scrollIntoView }));
    vi.stubGlobal("document", { querySelector });
    const onExpand = vi.fn();
    const html = renderToStaticMarkup(createElement(QuestionRowButtons, {
      card: grouped, eventId: 9, busy: false, answeredEventId: null, onExpand, onAction: () => {},
    }));
    expect(html).toContain("actions.respond");
    const key = boardCardKeyFor(grouped, 9)!;
    expect(key).toBe("p#session:B");
    expandAndScroll(key, onExpand);
    expect(onExpand).toHaveBeenCalledWith("p#session:B");
    expect(querySelector).toHaveBeenCalledWith('[data-card-key="p#session:B"]');
    expect(scrollIntoView).toHaveBeenCalled();
  });

  it("an event no card holds has no Respond", () => {
    const html = renderToStaticMarkup(createElement(QuestionRowButtons, {
      card: grouped, eventId: 99, busy: false, answeredEventId: null, onExpand: () => {}, onAction: () => {},
    }));
    expect(html).not.toContain("actions.respond");
  });

  it("an attention row with a capsule held by session B renders Respond; an unheld capsule renders none", () => {
    const b = card({
      dir: "p", key: "p#session:B", scope: sessionScope, headline: "needs-you",
      sessions: [{ transport: "tmux", session: "s1", panes: [pane({ pane: "%1", capsuleEventId: 31, capsuleStatus: "blocked" })] }],
    });
    const project = card({ dir: "p", group: { cards: [b], footer: null } });
    const inbox: InboxRow[] = [{ kind: "attention", dir: "p", name: "p", status: "blocked", transport: "tmux", pane: "%1", tmuxIncarnation: "1:1", capsuleEventId: 31 }];
    const held = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([project], inbox), onExpand: () => {} }));
    expect(held).toContain("actions.respond");
    expect(boardCardKeyFor(project, 31)).toBe("p#session:B");

    const unheldInbox: InboxRow[] = [{ kind: "attention", dir: "p", name: "p", status: "blocked", transport: "tmux", pane: "%1", tmuxIncarnation: "1:1", capsuleEventId: 99 }];
    const unheld = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([project], unheldInbox), onExpand: () => {} }));
    expect(unheld).not.toContain("actions.respond");
    expect(unheld).toContain("openTmux");
  });

  it("a Codex capsule held by a Codex seat card renders Respond", () => {
    const codex = card({
      dir: "p", key: "p#session:C", headline: "needs-you",
      scope: { kind: "session", sessionId: "C", transport: "codex", runtime: "codex", name: null, tmuxSessions: [], panes: [] },
      sessions: [{ transport: "codex", threadId: "C", state: "needs-you", lastEventTs: null, subagentCount: 0, capsuleEventId: 41, capsuleStatus: "needs_input", capsuleMinutes: null, capsuleDeclaredAt: null }],
    });
    const project = card({ dir: "p", group: { cards: [codex], footer: null } });
    const inbox: InboxRow[] = [{ kind: "attention", dir: "p", name: "p", status: "needs_input", transport: "codex", threadId: "C", capsuleEventId: 41 }];
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([project], inbox), onExpand: () => {} }));
    expect(html).toContain("actions.respond");
    expect(boardCardKeyFor(project, 41)).toBe("p#session:C");
  });

  it("a plain single-session card holding capsule 31 targets its own card key", () => {
    const plain = card({
      dir: "p", headline: "needs-you",
      sessions: [{ transport: "tmux", session: "s1", panes: [pane({ pane: "%1", capsuleEventId: 31, capsuleStatus: "blocked" })] }],
    });
    expect(boardCardKeyFor(plain, 31)).toBe(cardKey(plain));
    const inbox: InboxRow[] = [{ kind: "attention", dir: "p", name: "p", status: "blocked", transport: "tmux", pane: "%1", tmuxIncarnation: "1:1", capsuleEventId: 31 }];
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([plain], inbox), onExpand: () => {} }));
    expect(html).toContain("actions.respond");
  });

  it("a legacy pane question stays on the seat holding the pane and the inbox Respond finds it", () => {
    const a = card({
      dir: "p", key: "p#session:A", scope: sessionScope, headline: "needs-you",
      pendingQuestions: [{ ...twoQuestions, pane: "%1" }],
    });
    const b = card({ dir: "p", key: "p#session:B", scope: { ...sessionScope, sessionId: "B", panes: ["%2"] }, headline: "idle" });
    const project = card({ dir: "p", pendingQuestions: [{ ...twoQuestions, pane: "%1" }], group: { cards: [a, b], footer: null } });
    expect(boardCardKeyFor(project, 9)).toBe("p#session:A");
    const cardHtml = renderCard(a, true);
    expect(cardHtml).toContain(">one<");
    const inbox: InboxRow[] = [{ kind: "question", dir: "p", name: "p", pane: "%1", eventId: 9, question: "Pick one" }];
    const inboxHtml = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([project], inbox), onExpand: () => {} }));
    expect(inboxHtml).toContain("actions.respond");
  });
});

// ---- Task 8: group frame, session card content and board layout --------------------------------

import { ProjectsColumn } from "./ProjectsColumn";
import { missionOf, mkActivity, mkHubProject, mkPane, mkProject, mkSeat, mkSplit, runSummary, U as FU } from "@/lib/multiSessionFixtures";
import type { HubActiveRun, HubSessionSplit } from "@/server/db/workflows";
import { AnswerDraftsProvider } from "./AnswerDrafts";

const A = FU("a");
const B = FU("b");
const C = FU("c");

function run(runId: string): HubActiveRun {
  return { runId, startedAt: null, kind: "build", runtime: "claude", target: "feat/x", count: 1, lastStep: null };
}

function segment(html: string, key: string): string {
  const start = html.indexOf(`data-card-key="${key}"`);
  if (start < 0) return "";
  const rest = html.slice(start);
  const next = rest.indexOf('data-card-key="', 10);
  return next < 0 ? rest : rest.slice(0, next);
}


function renderColumn(m: Mission, expandedDir: string | null = null, filter: MissionBucket | null = null, store?: ReturnType<typeof import("./AnswerDrafts").createAnswerDraftStore>): string {
  const column = createElement(ProjectsColumn, {
    mission: m, reposRoot: "/repos", counts: null, filter, expandedDir, onToggleExpand: () => {},
  });
  return renderToStaticMarkup(store ? createElement(AnswerDraftsProvider, { store }, column) : column);
}

describe("group frame render (spec §7.1–§7.3, §13)", () => {
  it("one header, session cards without project footer facts, dashed unlinked card and one footer line", () => {
    const m = missionOf(
      {
        Alpha: mkHubProject({
          panes: [mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A }), mkPane({ pane: "%2", tmuxIncarnation: "1:1", harnessSession: B })],
          split: mkSplit({
            seats: [
              mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }], name: "multi-sessao" }),
              mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }] }),
            ],
            unlinked: mkActivity({ activeRuns: [runSummary("r1"), runSummary("r2")], activeRun: run("r1") }),
          }),
        }),
      },
      {
        prs: { Alpha: { ok: true, truncated: false, prs: [{ number: 12, title: "t", isDraft: false, url: "https://x", ciState: "green", mergeReady: false, updatedAt: "2026-11-12T10:00:00.000Z" }] } },
        worktrees: { Alpha: { ok: true, count: 1, oldestAgeDays: 1, sessions: [{ session: null, count: 1 }] } },
      },
    );
    const html = renderColumn(m);
    expect((html.match(/board\.sessionsChip/g) ?? []).length).toBe(1);
    expect((html.match(/Alpha</g) ?? []).length).toBe(1);
    expect((html.match(/aria-label="actions\.more"/g) ?? []).length).toBe(1);
    expect((html.match(/PR #12/g) ?? []).length).toBe(1);
    expect(segment(html, `Alpha#session:${A}`)).not.toContain("PR #12");
    expect(segment(html, `Alpha#session:${B}`)).not.toContain("PR #12");
    for (const key of [`Alpha#session:${A}`, `Alpha#session:${B}`, "Alpha#unlinked", "Alpha#project"]) {
      expect(html).toContain(`data-card-key="${key}"`);
    }
    const unlinked = segment(html, "Alpha#unlinked");
    expect(unlinked).toContain("border-dashed");
    expect(unlinked).toContain("board.unlinkedTitle");
    expect((unlinked.match(/expanded\.sessionLineNoRole/g) ?? []).length).toBe(2);
    expect((html.match(/board\.footerNoSession/g) ?? []).length).toBe(1);
  });

  it("the display chain: name + short id, short id alone, new session with location, truncate for long names", () => {
    const panes = [
      mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A }),
      mkPane({ pane: "%2", tmuxIncarnation: "1:1", harnessSession: B }),
      mkPane({ pane: "%4", tmuxIncarnation: "1:1", session: null, harnessSession: null }),
    ];
    const split = mkSplit({
      seats: [
        mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }], name: "multi-sessao" }),
        mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }] }),
        mkSeat({ sessionId: null, panes: [{ pane: "%4", tmuxIncarnation: "1:1" }] }),
      ],
    });
    const html = renderColumn(missionOf({ Alpha: mkHubProject({ panes, split }) }));
    const a = segment(html, `Alpha#session:${A}`);
    expect(a).toContain("multi-sessao");
    expect(a).toContain(A.slice(0, 8));
    const b = segment(html, `Alpha#session:${B}`);
    expect(b).toContain(B.slice(0, 8));
    expect(b).not.toContain("multi-sessao");
    const c = segment(html, "Alpha#pane:%4:1:1");
    expect(c).toContain("board.newSession");
    expect(c).toContain("board.location");

    split.seats[0].name = "n".repeat(80);
    const longHtml = renderColumn(missionOf({ Alpha: mkHubProject({ panes, split }) }));
    expect(longHtml).toContain("n".repeat(80));
    expect(segment(longHtml, `Alpha#session:${A}`)).toContain("truncate");
  });

  it("session stage/branch/Now come from its runs; status.md facts never reach grouped cards", () => {
    const panes = [mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A }), mkPane({ pane: "%2", tmuxIncarnation: "1:1", harnessSession: B })];
    const split = mkSplit({
      seats: [
        mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }], activity: mkActivity({ status: { latestKind: "build", latestTarget: "feat/a", lastSummary: "done a" } }) }),
        mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }] }),
      ],
    });
    const distinctive = mkProject({ dir: "Alpha", stage: "test", branch: "status-branch", builder: "builder-x", now: "status-now", gate: "awaiting-approval" });
    const html = renderColumn(missionOf({ Alpha: mkHubProject({ panes, split }) }, { projects: [distinctive] }));
    const a = segment(html, `Alpha#session:${A}`);
    expect(a).toContain("stages.build");
    expect(a).toContain("feat/a");
    expect(a).toContain("done a");
    const b = segment(html, `Alpha#session:${B}`);
    expect(b).not.toContain("stages.");
    expect(b).not.toContain("status-branch");
    expect(html).not.toContain("stages.test");
    expect(html).not.toContain("status-now");
    expect(html).not.toContain("builder-x");
  });

  it("Dispatch, residuals and the loop summary live only in the project section", () => {
    const panes = [mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A }), mkPane({ pane: "%2", tmuxIncarnation: "1:1", harnessSession: B })];
    const split = mkSplit({ seats: [mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }] }), mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }] })] });
    const m = missionOf({ Alpha: mkHubProject({ panes, split, loopSummary: { specCount: 1, planCount: 1, buildCount: 2, diffCount: 1, rounds: 2, approved: false, wallDays: 1 } }) }, {
      projects: [mkProject({ dir: "Alpha", residuals: ["residual-x"] })],
    });
    const sessionExpanded = renderColumn(m, `Alpha#session:${A}`);
    expect(sessionExpanded).not.toContain('data-testid="dispatch-row"');
    expect(sessionExpanded).not.toContain("residual-x");

    const projectExpanded = renderColumn(m, "Alpha#project");
    expect((projectExpanded.match(/data-testid="dispatch-row"/g) ?? []).length).toBe(1);
    expect(projectExpanded).toContain("actions.dispatch");
    expect(projectExpanded).toContain("residual-x");
    expect(projectExpanded).toContain("expanded.loopSummary");
    expect(projectExpanded).toContain("board.dispatchHint");
  });

  it("a plain single-session column renders no group frame", () => {
    const m = missionOf({
      Alpha: mkHubProject({
        panes: [mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A })],
        split: mkSplit({ seats: [mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }] })] }),
      }),
    });
    const html = renderColumn(m);
    expect(html).not.toContain("board.sessionsChip");
    expect(html).not.toContain("board.unlinkedTitle");
    expect(html).not.toContain('data-testid="group-header"');
    expect(html).not.toContain("board.");
  });

  it("an unlinked dispatch result shows its run and Cancel on the unlinked card only", () => {
    const m = missionOf({
      Alpha: mkHubProject({
        panes: [mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A })],
        split: mkSplit({
          seats: [mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }] })],
          unlinked: mkActivity({ activeRuns: [runSummary("r1")], activeRun: run("r1") }),
        }),
      }),
    });
    const html = renderColumn(m);
    expect((html.match(/actions\.cancel/g) ?? []).length).toBe(1);
    expect(segment(html, "Alpha#unlinked")).toContain("expanded.sessionLineNoRole");
    expect(segment(html, `Alpha#session:${A}`)).not.toContain("expanded.sessionLineNoRole");
  });

  it("prefs stay project-scoped on the group header, and a hidden project renders no group", () => {
    const panes = [mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A }), mkPane({ pane: "%2", tmuxIncarnation: "1:1", harnessSession: B })];
    const split = mkSplit({ seats: [mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }] }), mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }] })] });
    const pinned = missionOf({ Alpha: mkHubProject({ panes, split }) }, { prefs: { Alpha: { pinned: true, hiddenAt: null } } });
    const pinnedHtml = renderColumn(pinned);
    expect(segment(pinnedHtml, "Alpha#project")).toContain("actions.unpin");
    for (const key of [`Alpha#session:${A}`, `Alpha#session:${B}`]) expect(segment(pinnedHtml, key)).not.toContain("actions.unpin");

    const hidden = missionOf({ Alpha: mkHubProject({ panes, split }) }, { prefs: { Alpha: { pinned: false, hiddenAt: "2026-11-11T00:00:00.000Z" } } });
    const hiddenHtml = renderColumn(hidden);
    expect(hiddenHtml).not.toContain('data-testid="group-header"');
    expect(hiddenHtml).not.toContain(`data-card-key="Alpha#session:${A}"`);
    expect(hiddenHtml).toContain("hiddenBar");
  });

  it("the filter applies inside groups: one matching card keeps the group, none drops it", () => {
    const panes = [mkPane({ pane: "%1", tmuxIncarnation: "1:1", harnessSession: A }), mkPane({ pane: "%2", tmuxIncarnation: "1:1", harnessSession: B, working: true })];
    const split = mkSplit({ seats: [mkSeat({ sessionId: A, panes: [{ pane: "%1", tmuxIncarnation: "1:1" }] }), mkSeat({ sessionId: B, panes: [{ pane: "%2", tmuxIncarnation: "1:1" }] })] });
    const m = missionOf({ Alpha: mkHubProject({ panes, split }) });
    const working = renderColumn(m, null, "working");
    expect(working).toContain("board.sessionsChip");
    expect(working).toContain(`data-card-key="Alpha#session:${B}"`);
    expect(working).not.toContain(`data-card-key="Alpha#session:${A}"`);

    const stuck = renderColumn(m, null, "stuck");
    expect(stuck).not.toContain("board.sessionsChip");
    expect(stuck).not.toContain(`data-card-key="Alpha#session:${A}"`);
    expect(stuck).toContain("empty");
  });
});
