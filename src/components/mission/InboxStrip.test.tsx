import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("./SourceWarning", () => ({
  SourceWarning: ({ label, detail }: { label: string; detail: string }) =>
    createElement("div", { "data-testid": "source-warning" }, `${label} — ${detail}`),
}));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: () => null }));
const postSpy = vi.hoisted(() => vi.fn(async () => ({ ok: true, data: {} })));
vi.mock("@/lib/mission", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/mission")>();
  return { ...actual, postWorkflowAction: postSpy };
});

import type { InboxRow, Mission, MissionCard } from "@/lib/mission";
import { expandAndScroll, runQuestionAction, runStuckCancel, InboxStrip } from "./InboxStrip";

function missionOf(inbox: InboxRow[]): Mission {
  return {
    readiness: { kind: "ready" },
    model: {
      cards: [], uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
      inbox, inboxTotalCount: inbox.length, inboxPrRemainder: 0, prTruncatedAny: false,
      codexSource: "ok", tmuxSource: "ok",
      archiveAfterDays: 14, hiddenCount: 0, autoHiddenCount: 0, reposRoot: "",
    },
  };
}

describe("InboxStrip — hidden when nothing waits (mobile AFK-move follow-up)", () => {
  it("renders nothing (null) for an empty, ready, non-degraded mission", () => {
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission: missionOf([]), onExpand: () => {} }));
    expect(html).toBe("");
  });
  it("still renders the warning for a failed source even with an empty inbox", () => {
    const failedMission: Mission = { ...missionOf([]), readiness: { kind: "failed", failed: ["prs"], pending: [] } };
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission: failedMission, onExpand: () => {} }));
    expect(html).toContain("data-testid=\"source-warning\"");
    expect(html).toContain("unavailable");
  });
});

describe("InboxStrip — transport-aware attention rows (MOA-469 §4)", () => {
  it("a native Codex attention row is visible and has no tmux action", () => {
    const html = renderToStaticMarkup(createElement(InboxStrip, {
      mission: missionOf([
        { kind: "attention", dir: "p1", name: "p1", status: "needs_input", transport: "codex", threadId: "0191f0aa-1234-7000-8000-000000000001" },
      ]),
      onExpand: () => {},
    }));
    expect(html).toContain("codexSession");
    expect(html).toContain("0191f0aa");
    expect(html).not.toContain('href="/tmux"');
    expect(html).not.toContain("openTmux");
  });

  it("a Claude pane attention row retains its tmux action", () => {
    const html = renderToStaticMarkup(createElement(InboxStrip, {
      mission: missionOf([
        { kind: "attention", dir: "p1", name: "p1", status: "blocked", transport: "tmux", pane: "%1", tmuxIncarnation: "1:1" },
      ]),
      onExpand: () => {},
    }));
    expect(html).toContain('href="/tmux"');
    expect(html).toContain("openTmux");
  });
});

describe("InboxStrip — accepts onExpand (spec §6, Decision 13a, wired by Task 9, round-1 F9)", () => {
  it("renders a non-empty inbox's real rows, not just the trivial empty-model case, with onExpand supplied (unused by any row until Task 9)", () => {
    const inbox: InboxRow[] = [{ kind: "gate", dir: "p1", name: "Alpha", gate: "awaiting-approval", updated: "2026-09-19T10:00:00.000Z" }];
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission: missionOf(inbox), onExpand: () => {} }));
    expect(html).toContain("Alpha");
    expect(html).toContain("title"); // the count-bearing key path was taken, not titleEmpty
    expect(html).not.toContain("titleEmpty");
  });
});

function missionWithCards(inbox: InboxRow[], cards: MissionCard[]): Mission {
  const m = missionOf(inbox);
  return { ...m, model: { ...m.model, cards } };
}

const cardBase: MissionCard = {
  dir: "p1", name: "p1", stage: "build", branch: "main", builder: "claude-code",
  gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-09-19T10:00:00.000Z",
  headline: "stuck", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
  codexSource: "ok", codexUntracked: false,
  heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};

describe("InboxStrip — stuck rows and two-segment title (spec §6, Decisions 7/8, round-1 F6/F7)", () => {
  it("both counts when both present; PR rows excluded from the needs-you count (F6); n===0/m>0 renders stuckTitle alone (F7)", () => {
    const both = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([{ kind: "gate", dir: "p2", name: "p2", gate: "awaiting-approval", updated: "2026-09-19T10:00:00.000Z" }], [cardBase]), onExpand: () => {} }));
    expect(both).toContain("title");
    expect(both).toContain("stuckSuffix");
    // Plan deviation: the PR-only fixture carries NO stuck cards (the plan's draft paired it with
    // cardBase, which routes the title to the stuckTitle branch and makes the "count path" assertion
    // unable to pass) — F6's own point is the baseTitle key path, so the PR-only inbox stands alone.
    const prOnly = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([{ kind: "pr", dir: "p3", name: "p3", number: 1, url: "https://x", ci: "green" }], []), onExpand: () => {} })); // F6: PR-only still takes the "count" key path, not titleEmpty
    expect(prOnly).toContain("title");
    expect(prOnly).not.toContain("titleEmpty");
    const stuckOnly = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([], [cardBase]), onExpand: () => {} })); // cardBase.headline === "stuck", zero inbox rows
    expect(stuckOnly).toContain("stuckTitle");
    expect(stuckOnly).not.toContain("titleEmpty");
    expect(stuckOnly).not.toContain("stuckSuffix");
  });
  it("a stuck card with activeRun renders Cancelar run; without activeRun renders none", () => {
    const withRun = { ...cardBase, activeRun: { runId: "r1", startedAt: null, kind: "build" as const, runtime: "claude" as const, target: "x", count: 1, lastStep: null, stage: "build" as const, descriptionKey: "building" as const, targetKind: "branch" as const } };
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([], [withRun]), onExpand: () => {} }));
    expect(html).toContain("actions.cancel");
    const noRun = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([], [{ ...cardBase, activeRun: null }]), onExpand: () => {} }));
    expect(noRun).not.toContain("actions.cancel");
  });
});

describe("InboxStrip — question row actions (spec §6, Decision 9)", () => {
  const mergeQ = { question: "May I merge `x` into `main`?", multiSelect: false, options: [{ label: "Sim" }, { label: "Não" }] };
  it("a merge-shaped question row renders its own options — no regex-driven merge kind (native answer delivery D1)", () => {
    const card = { ...cardBase, headline: "needs-you" as const, pendingQuestions: [{ pane: "%1", eventId: 9, questions: [mergeQ] }] };
    const html = renderToStaticMarkup(createElement(InboxStrip, {
      mission: missionWithCards([{ kind: "question", dir: "p1", name: "p1", pane: "%1", eventId: 9, question: mergeQ.question }], [card]),
      onExpand: () => {},
    }));
    expect(html).not.toContain("actions.approveMerge");
    expect(html).toContain("Sim");
    expect(html).toContain("Não");
  });
  it("a bare-capsule (hasQuestionText false) question row renders no button — attention rows never had one, unchanged", () => {
    const html = renderToStaticMarkup(createElement(InboxStrip, {
      mission: missionOf([{ kind: "attention", dir: "p1", name: "p1", status: "blocked", transport: "tmux", pane: "%1", tmuxIncarnation: "1:1" }]),
      onExpand: () => {},
    }));
    expect(html).not.toContain("actions.respond");
    expect(html).toContain("blocked");
  });
  it("a gate row with no live capsule on any pane renders nothing at all (spec §5a narrowing)", () => {
    const html = renderToStaticMarkup(createElement(InboxStrip, {
      mission: missionWithCards([{ kind: "gate", dir: "p1", name: "p1", gate: "awaiting-approval", updated: "2026-09-19T10:00:00.000Z" }], [{ ...cardBase, headline: "needs-you" as const, gate: "awaiting-approval" as const }]),
      onExpand: () => {},
    }));
    expect(html).not.toContain("actions.approveMerge");
  });

  it("a gate row with a live lead-role tmux capsule renders Aprovar merge / Não", () => {
    const card = { ...cardBase, headline: "needs-you" as const, gate: "awaiting-approval" as const, branch: "feat/x",
      sessions: [{ transport: "tmux" as const, session: "s", panes: [
        { pane: "%1", tmuxIncarnation: "1:1", session: "s", role: "lead" as const, state: "needs-you" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: 9, capsuleStatus: "needs_input" as const, capsuleMinutes: null, capsuleDeclaredAt: null, live: true, tmuxSession: null, subagentCount: 0, capsuleMergeAsk: null, capsuleQuestion: null },
      ] }] };
    const html = renderToStaticMarkup(createElement(InboxStrip, {
      mission: missionWithCards([{ kind: "gate", dir: "p1", name: "p1", gate: "awaiting-approval", updated: "2026-09-19T10:00:00.000Z" }], [card]),
      onExpand: () => {},
    }));
    expect(html).toContain("actions.approveMerge");
  });
  it("a card with two pending questions renders two rows, each resolving its own eventId's options (F2)", () => {
    const q1 = { pane: "%1", eventId: 9, questions: [{ question: "Env?", multiSelect: false, options: [{ label: "dev" }, { label: "stg" }] }] };
    const q2 = { pane: "%2", eventId: 10, questions: [{ question: "Deploy?", multiSelect: false, options: [{ label: "yes" }, { label: "no" }] }] };
    const card = { ...cardBase, headline: "needs-you" as const, pendingQuestions: [q1, q2] };
    const html = renderToStaticMarkup(createElement(InboxStrip, {
      mission: missionWithCards([
        { kind: "question", dir: "p1", name: "p1", pane: "%1", eventId: 9, question: q1.questions[0].question },
        { kind: "question", dir: "p1", name: "p1", pane: "%2", eventId: 10, question: q2.questions[0].question },
      ], [card]),
      onExpand: () => {},
    }));
    // Pre-fix, both rows resolve pendingQuestions[0] (q1) — the second row's own options (yes/no) never render.
    expect(html).toContain(">dev<");
    expect(html).toContain(">stg<");
    expect(html).toContain(">yes<");
    expect(html).toContain(">no<");
  });
  it("an empty-text question row (structured but blank) renders the bare-capsule fallback, no quoted text or button (round-1 F8)", () => {
    const card = { ...cardBase, headline: "needs-you" as const, sessions: [{ transport: "tmux" as const, session: "s", panes: [{ pane: "%1", tmuxIncarnation: "1:1", session: "s", role: "lead" as const, state: "needs-you" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: 5, capsuleStatus: "blocked" as const, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 }] }] };
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission: missionWithCards([{ kind: "question", dir: "p1", name: "p1", pane: "%1", eventId: 1, question: "" }], [card]), onExpand: () => {} }));
    expect(html).not.toContain("actions.respond");
    expect(html).not.toContain("actions.approveMerge");
    expect(html).not.toContain("“”"); // no blank quoted text (JSX decodes &ldquo;/&rdquo; to real quote chars)
    expect(html).toContain("blocked"); // InboxLine's `t` is scoped to "mission.inbox" already, bare key
    expect(html).toContain('href="/tmux"');
  });
});

describe("InboxStrip — direct handler tests, no jsdom (round-2 F4)", () => {
  it("expandAndScroll calls onExpand(dir) and scrollIntoView on the matching card", () => {
    const scrollIntoView = vi.fn();
    vi.stubGlobal("document", { querySelector: vi.fn(() => ({ scrollIntoView })) });
    const onExpand = vi.fn();
    expandAndScroll("p1", onExpand);
    expect(onExpand).toHaveBeenCalledExactlyOnceWith("p1");
    expect(scrollIntoView).toHaveBeenCalledOnce();
    vi.unstubAllGlobals();
  });
  it("runQuestionAction posts url/body and toggles busy around it", async () => {
    postSpy.mockClear();
    const setBusy = vi.fn();
    await runQuestionAction("/api/workflow/answer", { event_id: 9, reply: "1" }, setBusy);
    expect(postSpy).toHaveBeenCalledExactlyOnceWith("/api/workflow/answer", { event_id: 9, reply: "1" });
    expect(setBusy).toHaveBeenNthCalledWith(1, true);
    expect(setBusy).toHaveBeenNthCalledWith(2, false);
  });
  it("runStuckCancel posts cancel with run_id and dismisses the confirm dialog", async () => {
    postSpy.mockClear();
    const setBusy = vi.fn();
    const setConfirming = vi.fn();
    await runStuckCancel("r1", setBusy, setConfirming);
    expect(postSpy).toHaveBeenCalledExactlyOnceWith("/api/workflow/cancel", { run_id: "r1" });
    expect(setConfirming).toHaveBeenCalledExactlyOnceWith(false);
  });
});
