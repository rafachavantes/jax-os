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

import type { InboxRow, MissionCard, MissionPane } from "@/lib/mission";
import { ProjectCard } from "./ProjectCard";
import { InboxStrip } from "./InboxStrip";

function baseCard(over: Partial<MissionCard> = {}): MissionCard {
  return {
    dir: "alpha", name: "alpha", stage: "build", branch: "feat/a", builder: "claude-code",
    gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-11-10T10:00:00-03:00",
    headline: "idle", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
    sessions: [], pendingQuestions: [], timeline: [], pr: { status: "ok", count: 0, ci: "none", truncated: false, newest: null },
    activeRun: null, activeRuns: [], lastRun: null, retainedWorktrees: null, loopSummary: null,
    codexSource: "ok", codexUntracked: false,
    heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
    ...over,
  };
}

const pane = (over: Partial<MissionPane> & { pane: string }): MissionPane => ({
  pane: over.pane, tmuxIncarnation: "100:1000", session: "jax-os", role: "lead", live: true,
  state: "idle", lastEventTs: null, pendingQuestion: null, capsuleStatus: null, capsuleMinutes: null,
  capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, subagentCount: 0,
  capsuleEventId: null, tmuxSession: "jax-os", ...over,
});

const workingRun = {
  runId: "run-1", startedAt: null, kind: "build" as const, runtime: "claude" as const, target: "feat/x",
  count: 1, lastStep: null, stage: "build" as const, descriptionKey: "building" as const, targetKind: "branch" as const,
};

const fixtures: { name: string; card: MissionCard; expanded: boolean }[] = [
  { name: "plain-idle", card: baseCard(), expanded: false },
  {
    name: "plain-working-run",
    expanded: false,
    card: baseCard({
      headline: "working", activeRun: workingRun,
      sessions: [{ transport: "tmux", session: "jax-os", panes: [pane({ pane: "%1", state: "working" })] }],
    }),
  },
  {
    name: "plain-approve",
    expanded: false,
    card: baseCard({
      headline: "needs-you", gate: "awaiting-approval",
      sessions: [{ transport: "tmux", session: "jax-os", panes: [pane({
        pane: "%1", role: "lead", state: "needs-you", capsuleEventId: 7, capsuleStatus: "needs_input",
        capsuleMergeAsk: 1, capsuleMergeBranch: "feat/a", capsuleMergeTarget: "main", capsuleQuestion: "May I merge?",
      })] }],
    }),
  },
  {
    name: "plain-expanded",
    expanded: true,
    card: baseCard({
      residuals: ["finish the docs", "rotate the token"], retainedWorktrees: 2,
      loopSummary: { specCount: 1, planCount: 1, buildCount: 2, diffCount: 1, rounds: 2, wallDays: 1 },
    }),
  },
];

describe("plain card golden markup (recorded on UNCHANGED code)", () => {
  for (const f of fixtures) {
    it(`${f.name} renders byte-identically`, async () => {
      const html = renderToStaticMarkup(createElement(ProjectCard, {
        card: f.card, issues: 2, expanded: f.expanded, onToggle: () => {},
      }));
      expect(html).not.toContain("data-card-key");
      await expect(html).toMatchFileSnapshot(`./__golden__/${f.name}.html`);
    });
  }

  it("inbox-gate: a gate row and a question row render byte-identically", async () => {
    const gateCard = baseCard({
      dir: "gateway", name: "gateway", headline: "needs-you", gate: "awaiting-approval",
      sessions: [{ transport: "tmux", session: "jax-os", panes: [pane({
        pane: "%2", role: "lead", state: "needs-you", capsuleEventId: 8, capsuleStatus: "needs_input", capsuleMergeAsk: null,
      })] }],
    });
    const questionCard = baseCard({
      dir: "quiz", name: "quiz", headline: "needs-you",
      pendingQuestions: [{ pane: "%1", eventId: 9, questions: [{ question: "Env?", multiSelect: false, options: [{ label: "dev" }, { label: "stg" }] }] }],
    });
    const inbox: InboxRow[] = [
      { kind: "gate", dir: "gateway", name: "gateway", gate: "awaiting-approval", updated: "2026-11-10T10:00:00-03:00" },
      { kind: "question", dir: "quiz", name: "quiz", pane: "%1", eventId: 9, question: "Env?" },
    ];
    const mission = {
      readiness: { kind: "ready" as const },
      model: {
        cards: [gateCard, questionCard], uncardedProjectNames: [], historicalTruncated: false, skipped: 0,
        inbox, inboxTotalCount: inbox.length, inboxPrRemainder: 0, prTruncatedAny: false,
        codexSource: "ok" as const, tmuxSource: "ok" as const,
        archiveAfterDays: 14, hiddenCount: 0, autoHiddenCount: 0, reposRoot: "",
      },
    };
    const html = renderToStaticMarkup(createElement(InboxStrip, { mission, onExpand: () => {} }));
    expect(html).not.toContain("actions.respond");
    await expect(html).toMatchFileSnapshot("./__golden__/inbox-gate.html");
  });
});
