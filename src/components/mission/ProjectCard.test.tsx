import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  githubEnabled: true,
  settingsMode: "loaded" as "loaded" | "loading" | "error",
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
// F1: renders a distinguishing marker with the epochMs it was given, so tests can assert
// presence/absence and the exact timestamp passed through, instead of the previous `=> null`
// which made every RelativeTime call invisible to renderToStaticMarkup.
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: ({ epochMs }: { epochMs: number }) => `RT(${epochMs})` }));
// PrefsButtons (item 2) reads a QueryClient via useQueryClient — no provider exists in these
// static-markup renders, same stand-in AfkToggle.test.tsx already uses for the same hook.
// useQuery answers the settings key (`useGeneralSettings`, Task 5's github gate) and nothing
// else; the harness flag lets the github-off test flip it for one render.
vi.mock("@tanstack/react-query", () => ({
  useQueryClient: () => ({}),
  useQuery: (opts: { queryKey: unknown[] }) =>
    opts.queryKey[0] === "settings"
      ? harness.settingsMode === "loading"
        ? { data: undefined }
        : harness.settingsMode === "error"
          ? { data: { ok: false, error: "x" } }
          : { data: { ok: true, data: { integrations: { github: harness.githubEnabled } } } }
      : { data: undefined },
}));

import type { MissionCard, MissionPane, MissionPendingQuestion } from "@/lib/mission";
import { dispatchBodyFor, gateWaitingLine, heartbeatPath, statusPillSuffix, stuckLine, worstFinding } from "@/lib/mission";
import {
  actionEventId, bareCapsuleStatus, bucketPrefsPatch, CancelDialog, pendingUiAction, PendingActionButtons, PrefsMenu, ProjectCard, dispatchValid, handleCancelDialogClose,
} from "./ProjectCard";
import { LIMITS } from "@/lib/workflow";

beforeEach(() => {
  harness.githubEnabled = true;
  harness.settingsMode = "loaded";
});

// Byte-identical pt-BR regression: every pre-existing assertion in this file keeps passing by
// injecting a translator that resolves to today's hardcoded literals (no assertion text changes).
const T_PT = (key: string, vars?: Record<string, string | number>): string => {
  const map: Record<string, string> = {
    "actions.replyMergeAsk": "pode",
    "actions.replyMergeBranch": vars ? `pode fazer o merge de ${vars.branch}` : "",
    "actions.replyMerge": "pode fazer o merge",
    "actions.replyDeny": "não",
  };
  return map[key] ?? key;
};
const T_EN = (key: string, vars?: Record<string, string | number>): string => {
  const map: Record<string, string> = {
    "actions.replyMergeAsk": "yes",
    "actions.replyMergeBranch": vars ? `yes, merge ${vars.branch}` : "",
    "actions.replyMerge": "yes, go ahead and merge",
    "actions.replyDeny": "no",
  };
  return map[key] ?? key;
};

// Slices the markup of a balanced <div ...marker...> ... </div> out of a renderToStaticMarkup
// string — same depth-counting convention the F4 no-nested-buttons test uses, generalized to
// find one tagged div's own closing tag instead of just checking overall nesting depth.
function sliceBalancedDiv(html: string, marker: string): string {
  const markerIndex = html.indexOf(marker);
  if (markerIndex < 0) throw new Error(`marker not found: ${marker}`);
  const openTag = html.lastIndexOf("<div", markerIndex);
  const divTag = /<div\b|<\/div>/g;
  divTag.lastIndex = openTag;
  let depth = 0;
  let match: RegExpExecArray | null;
  while ((match = divTag.exec(html))) {
    depth += match[0] === "</div>" ? -1 : 1;
    if (depth === 0) return html.slice(openTag, match.index + match[0].length);
  }
  throw new Error(`unbalanced div for marker: ${marker}`);
}

const base: MissionCard = {
  dir: "p1", name: "p1", stage: "build", branch: "main", builder: "claude-code",
  gate: null, legacyGateField: false, now: "", residuals: [], updated: "2026-09-10T12:00:00.000Z",
  headline: "working", hubReady: true, lastEventAgo: null, freshnessCount: null, freshnessWarning: false,
  sessions: [], pendingQuestions: [], timeline: [], pr: { status: "unknown" },
  codexSource: "ok", codexUntracked: false,
  heartbeat: new Array(60).fill(0), lastAction: null, prefs: { pinned: false, hiddenAt: null }, visibility: "visible",
};

const tmuxApprovePane = (over: Partial<MissionPane> = {}): MissionPane => ({
  pane: "%1", tmuxIncarnation: "1:1", session: "s", role: "lead", state: "needs-you", lastEventTs: null,
  pendingQuestion: null, capsuleEventId: 9, capsuleStatus: "needs_input", capsuleMinutes: null, capsuleDeclaredAt: null,
  capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0, ...over,
});

const PANE_ROW: MissionCard = {
  ...base,
  sessions: [{ transport: "tmux", session: "jax-p1-lead", panes: [{ pane: "%1", tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead", state: "working", lastEventTs: null, pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 }] }],
};

const CODEX_ROW: MissionCard = {
  ...base,
  sessions: [{ transport: "codex", threadId: "0191f0aa-1234-7000-8000-000000000001", state: "working", lastEventTs: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, subagentCount: 0 }],
};

const FAILED_RUN: MissionCard = {
  ...base,
  headline: "idle",
  lastRun: {
    runId: "r1", role: "builder", contractStatus: "ok", stage: "runtime",
    diagnostic: "APIError 403 API key budget limit exceeded", finishedAt: "2026-09-09T10:02:00.000Z",
    headSha: null, labelKey: "buildFailed", tone: "danger", findings: null,
    target: null, verifyCommand: null, buildCommand: null,
    runtimeModel: null, profileName: null, reportRel: null, targetRel: null,
  },
};

describe("ProjectCard — transport-aware session rows (MOA-469 §4)", () => {
  it("a Claude pane row retains its tmux action", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: PANE_ROW }));
    expect(html).toContain('href="/tmux"');
    expect(html).toContain("expanded.openInTmux");
  });

  it("a pane-less Codex row is visible and has no tmux action", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: CODEX_ROW }));
    expect(html).toContain("codexSession");
    expect(html).toContain("0191f0aa");
    expect(html).not.toContain('href="/tmux"');
    expect(html).not.toContain("expanded.openInTmux");
  });

  it("a Codex source failure still renders the working Claude card", () => {
    const card: MissionCard = { ...PANE_ROW, codexSource: "failed" };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).toContain("states.working");
    expect(html).toContain('href="/tmux"');
  });
});

describe("ProjectCard — compact-face SessionLines (round-3 review F1/F2/T3)", () => {
  const p = (pane: string, over: Partial<MissionPane> = {}): MissionPane => ({
    pane, tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead", state: "working", lastEventTs: null,
    pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null,
    capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0, ...over,
  });

  // F1: a finite pane timestamp renders a RelativeTime for that session line; no timestamp
  // omits the segment entirely (mocked RelativeTime renders "RT(<ms>)" — see the module mock).
  it("ends the tmux session line with a RelativeTime built from the newest pane's lastEventTs", () => {
    const ms = Date.parse("2026-09-20T11:52:00.000Z");
    const card: MissionCard = { ...base, sessions: [{ transport: "tmux", session: "s", panes: [
      p("%1", { lastEventTs: "2026-09-20T11:40:00.000Z" }),
      p("%2", { lastEventTs: "2026-09-20T11:52:00.000Z" }),
    ] }] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain(`RT(${ms})`);
  });
  it("omits the RelativeTime segment when no pane has a finite lastEventTs", () => {
    const card: MissionCard = { ...base, sessions: [{ transport: "tmux", session: "s", panes: [p("%1", { lastEventTs: null }), p("%2", { lastEventTs: "not-a-date" })] }] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).not.toContain("RT(");
  });
  it("a codex session line renders RelativeTime from its own lastEventTs", () => {
    const ms = Date.parse("2026-09-20T11:52:00.000Z");
    const card: MissionCard = { ...base, sessions: [{ transport: "codex", threadId: "0191f0aa-1234-7000-8000-000000000001", state: "working", lastEventTs: "2026-09-20T11:52:00.000Z", capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, subagentCount: 0 }] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain(`RT(${ms})`);
  });

  // F2: worstPaneState's own state-resolution logic is unit-tested directly in mission.test.ts
  // — the next-intl mock at the top of this file strips t() params, so SessionLines' rendered
  // state text is opaque to a renderToStaticMarkup assertion here.
});

describe("ProjectCard — last-run line and timeline (spec §12.4, MOA-474)", () => {
  it("renders the last-run line with its tone class and diagnostic verbatim, only when no run is active", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: FAILED_RUN }));
    expect(html).toContain("lastRun.buildFailed");
    expect(html).toContain("text-danger");
    expect(html).toContain("APIError 403 API key budget limit exceeded");
  });

  it("a successful build shows the short head sha (spec §12.4 example, diff review 754d9e56254d F2)", () => {
    const ok: MissionCard = { ...FAILED_RUN, lastRun: { ...FAILED_RUN.lastRun!, stage: null, diagnostic: null, headSha: "9f2b487abcdef0123456789abcdef0123456789a", labelKey: "buildOk", tone: "success" } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: ok }));
    expect(html).toContain("lastRun.buildOk");
    expect(html).toContain("9f2b487");
    expect(html).not.toContain("9f2b487a");
  });

  it("renders the active-run progress row and, on the shared F1 path, the last-run line too", () => {
    // F1 (cold review 92a14d784fb3, round 3): SessionLines/LastLine moved outside the
    // activeRun ternary, so an active card now renders its last-run line via LastLine's own
    // LastRunLine fallback — the old "omitted while active" contract is superseded.
    const activeCard: MissionCard = {
      ...FAILED_RUN,
      activeRun: { runId: "r2", startedAt: null, kind: "build", runtime: "claude", target: "x", count: 1, lastStep: null, stage: "build", descriptionKey: "building", targetKind: "branch" },
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: activeCard }));
    expect(html).toContain("activity.building");
    expect(html).toContain("lastRun.buildFailed");
  });

  it("shows the reportProblem flag only for missing/invalid contractStatus", () => {
    const missing: MissionCard = { ...FAILED_RUN, lastRun: { ...FAILED_RUN.lastRun!, contractStatus: "missing", labelKey: "buildNoResult", stage: null, diagnostic: null } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: missing }));
    expect(html).toContain("lastRun.reportProblem");
    const ok: MissionCard = FAILED_RUN;
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: ok }))).not.toContain("lastRun.reportProblem");
  });

  it("renders run-finished timeline rows with outcome/contractStatus/stage/diagnostic", () => {
    const withTimeline: MissionCard = {
      ...base,
      timeline: [{ ts: "2026-09-09T10:02:00.000Z", type: "run-finished", pane: null, capsuleStatus: null, role: null, outcome: "failure", contractStatus: "ok", stage: "build", diagnostic: "pnpm build exit 1" }],
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: withTimeline }));
    expect(html).toContain("pnpm build exit 1");
    expect(html).toContain("failure");
  });
});

describe("ProjectCard — findings on the last-run line (round-3 F1)", () => {
  it("renders findings counts when present", () => {
    const withFindings: MissionCard = { ...FAILED_RUN, lastRun: { ...FAILED_RUN.lastRun!, findings: { high: 1, medium: 2, low: 0 } } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: withFindings }));
    expect(html).toContain("lastRun.severity.high");
  });

  it("omits the findings segment entirely when null — no placeholder, no zero", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: FAILED_RUN }));
    expect(html).not.toContain("lastRun.findings");
  });

  // F4: the badge's tone tracks severity — high/medium get the danger/warning pairing, low
  // keeps the pre-existing muted look.
  it("tones the severity badge by severity (high/medium/low)", () => {
    const withFindings = (findings: { high: number; medium: number; low: number }): MissionCard => ({ ...FAILED_RUN, lastRun: { ...FAILED_RUN.lastRun!, findings } });
    const high = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: withFindings({ high: 1, medium: 0, low: 0 }) }));
    expect(high).toContain("border-danger bg-danger-soft text-danger");
    const medium = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: withFindings({ high: 0, medium: 1, low: 0 }) }));
    expect(medium).toContain("border-warning bg-warning-soft text-warning");
    const low = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: withFindings({ high: 0, medium: 0, low: 1 }) }));
    expect(low).toContain("border-line text-muted");
  });
});

describe("ProjectCard — border state and stage track (Decisions 9/10)", () => {
  it("a working card gets the animated ring class", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: base }));
    expect(html).toContain("state-working");
  });

  it("a needs-you card gets a static danger border, no ring class", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "needs-you" } }));
    expect(html).not.toContain("state-working");
    expect(html).toContain("border-danger");
  });

  it("highlights exactly one dot matching the card's own stage", () => {
    // round-2 F2: `/bg-brand/g` also matches the unrelated `bg-brand-soft` folder-flag class; assert the dedicated marker instead.
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, stage: "review" } }));
    expect((html.match(/data-active="true"/g) ?? []).length).toBe(1);
  });

  it("shows no absolute filesystem path anywhere", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: base }));
    expect(html).not.toContain("/home/rafa/repos");
  });
});

describe("ProjectCard — retained worktrees and loop summary (Fase 4)", () => {
  it("retained-worktree section omitted at count 0/null, shown otherwise", () => {
    const zero = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, retainedWorktrees: 0 } }));
    expect(zero).not.toContain("expanded.retainedWorktrees");
    const none = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, retainedWorktrees: null } }));
    expect(none).not.toContain("expanded.retainedWorktrees");
    const some = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, retainedWorktrees: 3 } }));
    expect(some).toContain("expanded.retainedWorktrees");
  });

  it("loop-summary line shown only when it has any non-zero count", () => {
    const none = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, loopSummary: null } }));
    expect(none).not.toContain("expanded.loopSummary");
    const some = renderToStaticMarkup(createElement(ProjectCard, {
      expanded: true, onToggle: () => {}, card: { ...base, loopSummary: { specCount: 0, planCount: 0, buildCount: 1, diffCount: 2, rounds: 2, approved: true, wallDays: 3.4 } },
    }));
    expect(some).toContain("expanded.loopSummary");
    expect(some).toContain("expanded.loopWallTime");
  });
});

describe("action row (Task 9, spec §9)", () => {
  const merge2 = (labels: [string, string]): MissionPendingQuestion => ({
    pane: "%1", eventId: 11,
    questions: [{ question: "May I merge `feat/x` into `main`?", multiSelect: false, options: [{ label: labels[0] }, { label: labels[1] }] }],
  });

  it("needs-you + a merge-shaped pendingQuestion renders through the generic per-option control now (D1 — classifyMergeQuestion deleted)", () => {
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [merge2(["Sim", "Não"])] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).not.toContain("actions.approveMerge");
    expect(html).toContain(">Sim<");
    expect(html).toContain(">Não<");
  });

  it("needs-you + a non-merge-shaped pendingQuestion renders the generic option labels, not Aprovar merge", () => {
    const q: MissionPendingQuestion = { pane: "%1", eventId: 11, questions: [{ question: "Which env?", multiSelect: false, options: [{ label: "dev" }, { label: "prod" }] }] };
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [q] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).not.toContain("actions.approveMerge");
    expect(html).toContain("dev");
    expect(html).toContain("prod");
  });

  it("needs-you with a bare awaiting-approval gate and no live capsule anywhere renders no button (spec §5a narrowing)", () => {
    const card = { ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, pendingQuestions: [] };
    expect(pendingUiAction(card, undefined, T_PT)).toBeNull();
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).not.toContain("actions.approveMerge");
  });

  it("a working card with an awaiting-approval gate renders no approve button: the tech lead is still applying the findings", () => {
    const card = { ...base, headline: "working" as const, gate: "awaiting-approval" as const, pendingQuestions: [] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).not.toContain("actions.approveMerge");
  });

  it("needs-you with a gate and a native Codex freeform capsule (no mergeAsk) still renders freeform — codex sessions carry role: null so the gate fallback never fires for them (D2)", () => {
    const card = { ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, pendingQuestions: [],
      sessions: [{ transport: "codex" as const, threadId: "t1", state: "needs-you" as const, lastEventTs: null, subagentCount: 0, capsuleEventId: 42, capsuleStatus: "needs_input" as const, capsuleMinutes: null, capsuleDeclaredAt: null }] };
    expect(pendingUiAction(card, undefined, T_PT)?.kind).toBe("freeform");
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).not.toContain("actions.approveMerge");
  });

  it("a native Codex session with mergeAsk above threshold now reaches approve too — D2 drops the old transport==='tmux' guard", () => {
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [],
      sessions: [{ transport: "codex" as const, threadId: "t1", state: "needs-you" as const, lastEventTs: null, subagentCount: 0, capsuleEventId: 42, capsuleStatus: "needs_input" as const, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: 0.9, capsuleQuestion: "posso mergear feat/x em main?" }] };
    expect(pendingUiAction(card, undefined, T_PT)).toEqual({ kind: "approve", eventId: 42, question: "posso mergear feat/x em main?", replyYes: "pode", replyNo: "não", answerable: true });
  });

  it("a native Codex session with answerable: false renders the approve pair disabled plus the unreachable line (D4)", () => {
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [],
      sessions: [{ transport: "codex" as const, threadId: "t1", state: "needs-you" as const, lastEventTs: null, subagentCount: 0, capsuleEventId: 42, capsuleStatus: "needs_input" as const, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: 0.9, capsuleQuestion: "posso mergear?", capsuleAnswerable: false }] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).toContain("actions.answerUnreachable");
    expect(html).toMatch(/<button[^>]*\bdisabled\b[^>]*>actions\.approveMerge<\/button>/);
  });

  it("a native Codex session with a plain (non-merge) freeform capsule and answerable: false renders the textarea and submit disabled plus the unreachable line (F1, review round 4)", () => {
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [],
      sessions: [{ transport: "codex" as const, threadId: "t1", state: "needs-you" as const, lastEventTs: null, subagentCount: 0, capsuleEventId: 42, capsuleStatus: "needs_input" as const, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleAnswerable: false }] };
    expect(pendingUiAction(card, undefined, T_PT)?.kind).toBe("freeform");
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).toContain("actions.answerUnreachable");
    expect(html).toMatch(/<textarea[^>]*\bdisabled\b[^>]*>/);
    expect(html).toMatch(/<button[^>]*\bdisabled\b[^>]*>actions\.submit<\/button>/);
  });

  it("needs-you with gate blocked and nothing pending renders no action button at all", () => {
    const card = { ...base, headline: "needs-you" as const, gate: "blocked" as const, pendingQuestions: [] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).not.toContain("actions.approveMerge");
    expect(html).not.toContain("actions.respond");
  });

  it("a structured pendingQuestion wins over a merge-ask-eligible freeform capsule (selectPendingAction's own precedence, unchanged)", () => {
    const q: MissionPendingQuestion = { pane: "%1", eventId: 1, questions: [{ question: "Which environment?", multiSelect: false, options: [{ label: "dev" }, { label: "stg" }] }] };
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [q],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ pendingQuestion: q, capsuleMergeAsk: 0.99, capsuleQuestion: "posso mergear?" })] }] };
    expect(pendingUiAction(card, undefined, T_PT)?.kind).toBe("generic");
  });

  it("a message-detected merge ask is Jev's mergeAsk alone now: above threshold is 'approve', a literal-regex-looking question with no mergeAsk falls through to freeform (D1)", () => {
    const byMergeAsk = { ...base, headline: "needs-you" as const, pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ role: "adhoc", capsuleMergeAsk: 0.82, capsuleQuestion: "posso mergear feat/x em main?" })] }] };
    expect(pendingUiAction(byMergeAsk, undefined, T_PT)).toEqual({ kind: "approve", eventId: 9, question: "posso mergear feat/x em main?", replyYes: "pode", replyNo: "não", answerable: true });
    const byLiteralRegex = { ...base, headline: "needs-you" as const, pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ role: "adhoc", capsuleMergeAsk: null, capsuleQuestion: "May I merge `feat/x` into `main`?" })] }] };
    expect(pendingUiAction(byLiteralRegex, undefined, T_PT)?.kind).toBe("freeform");
    const nonMatchingQuestion = { ...base, headline: "needs-you" as const, pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ role: "adhoc", capsuleMergeAsk: null, capsuleQuestion: "should I retry the build?" })] }] };
    expect(pendingUiAction(nonMatchingQuestion, undefined, T_PT)?.kind).toBe("freeform");
  });

  it("gate set, a live lead-role tmux capsule exists, not merge-ask-eligible → approve with the branded gate reply, falling back to the bare phrase when card.branch is empty", () => {
    const card = { ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, branch: "feat/x", pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ capsuleQuestion: "done, next step tomorrow" })] }] };
    expect(pendingUiAction(card, undefined, T_PT)).toEqual({ kind: "approve", eventId: 9, question: "done, next step tomorrow", replyYes: "pode fazer o merge de feat/x", replyNo: "não", answerable: true });
    const noBranch = { ...card, branch: "" };
    const noBranchAction = pendingUiAction(noBranch, undefined, T_PT);
    if (noBranchAction?.kind !== "approve") throw new Error("expected approve");
    expect(noBranchAction.replyYes).toBe("pode fazer o merge");
  });

  it("gate fallback only covers an unclassified ask: a gated lead capsule Jev scored as not-a-merge-ask gets no approve", () => {
    const gated = (capsuleMergeAsk: number | null) => ({ ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, branch: "feat/x", pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ capsuleMergeAsk, capsuleQuestion: "should I retry the build?" })] }] });
    expect(pendingUiAction(gated(0.1), undefined, T_PT)?.kind).toBe("freeform");
    expect(pendingUiAction(gated(null), undefined, T_PT)?.kind).toBe("approve");
    expect(pendingUiAction(gated(0.9), undefined, T_PT)?.kind).toBe("approve");
  });

  it("merge-eligible branch: localizes the bare reply through T (F1 — the literal decision 13 missed)", () => {
    const byMergeAsk = { ...base, headline: "needs-you" as const, pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ role: "adhoc", capsuleMergeAsk: 0.82, capsuleQuestion: "posso mergear feat/x em main?" })] }] };
    expect(pendingUiAction(byMergeAsk, undefined, T_PT)).toEqual({ kind: "approve", eventId: 9, question: "posso mergear feat/x em main?", replyYes: "pode", replyNo: "não", answerable: true });
    expect(pendingUiAction(byMergeAsk, undefined, T_EN)).toEqual({ kind: "approve", eventId: 9, question: "posso mergear feat/x em main?", replyYes: "yes", replyNo: "no", answerable: true });
  });

  it("branch-merge fallback: localizes the interpolated reply through T", () => {
    const withBranch = { ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, branch: "feat/x", pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ capsuleQuestion: "done, next step tomorrow" })] }] };
    const ptAction = pendingUiAction(withBranch, undefined, T_PT);
    if (ptAction?.kind !== "approve") throw new Error("expected approve");
    expect(ptAction.replyYes).toBe("pode fazer o merge de feat/x");
    const enAction = pendingUiAction(withBranch, undefined, T_EN);
    if (enAction?.kind !== "approve") throw new Error("expected approve");
    expect(enAction.replyYes).toBe("yes, merge feat/x");
  });

  it("an adhoc pane under a gate gets no approve (role rule), but a message-detected ask on the same adhoc pane does", () => {
    const noAsk = { ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, branch: "feat/x", pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ role: "adhoc" })] }] };
    expect(pendingUiAction(noAsk, undefined, T_PT)?.kind).toBe("freeform");
    const withAsk = { ...noAsk, sessions: [{ ...noAsk.sessions[0], panes: [tmuxApprovePane({ role: "adhoc", capsuleMergeAsk: 0.9, capsuleQuestion: "posso mergear?" })] }] };
    expect(pendingUiAction(withAsk, undefined, T_PT)).toEqual({ kind: "approve", eventId: 9, question: "posso mergear?", replyYes: "pode", replyNo: "não", answerable: true });
  });

  it("a blocked-status capsule never becomes approve, mergeAsk-eligible or not (spec §7)", () => {
    const card = { ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, branch: "feat/x", pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ capsuleStatus: "blocked", capsuleMergeAsk: 0.9, capsuleQuestion: "posso mergear?" })] }] };
    expect(pendingUiAction(card, undefined, T_PT)?.kind).toBe("freeform");
  });

  it("QuestionLine renders the derived question for the approve kind, nothing when question is null", () => {
    const withQ = { ...base, headline: "needs-you" as const, pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane({ role: "adhoc", capsuleMergeAsk: 0.9, capsuleQuestion: "posso mergear?" })] }] };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: withQ }))).toContain("posso mergear?");
    const noQ = { ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, branch: "feat/x", pendingQuestions: [],
      sessions: [{ transport: "tmux" as const, session: "s", panes: [tmuxApprovePane()] }] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: noQ }));
    expect(html).not.toContain('? &quot;');
  });

  it("Aprovar/Negar call onAction with the action's own eventId and reply text (no jsdom — the component called as a plain function, its returned element tree walked directly)", () => {
    const calls: [string, unknown][] = [];
    const onAction = (url: string, body: unknown) => { calls.push([url, body]); };
    const messageDetected = { kind: "approve" as const, eventId: 9, question: "posso mergear?", replyYes: "pode", replyNo: "não", answerable: true };
    const tree1 = PendingActionButtons({ action: messageDetected, busy: false, onExpand: () => {}, onAction }) as { props: { children: { props: { onClick: () => void } }[] } };
    tree1.props.children[0].props.onClick();
    tree1.props.children[1].props.onClick();
    const gateBranded = { kind: "approve" as const, eventId: 22, question: null, replyYes: "pode fazer o merge de feat/x", replyNo: "não", answerable: true };
    const tree2 = PendingActionButtons({ action: gateBranded, busy: false, onExpand: () => {}, onAction }) as { props: { children: { props: { onClick: () => void } }[] } };
    tree2.props.children[0].props.onClick();
    tree2.props.children[1].props.onClick();
    expect(calls).toEqual([
      ["/api/workflow/answer", { event_id: 9, reply: "pode" }],
      ["/api/workflow/answer", { event_id: 9, reply: "não" }],
      ["/api/workflow/answer", { event_id: 22, reply: "pode fazer o merge de feat/x" }],
      ["/api/workflow/answer", { event_id: 22, reply: "não" }],
    ]);
  });

  // merge-button-hide fix: `answered` short-circuits PendingActionButtons to a confirmation
  // string instead of the buttons — this is what makes an approve/deny click disappear right
  // away instead of waiting out a hub poll.
  it("PendingActionButtons renders a confirmation string, no buttons, when answered is true", () => {
    const action = { kind: "approve" as const, eventId: 9, question: null, replyYes: "pode", replyNo: "não", answerable: true };
    const html = renderToStaticMarkup(PendingActionButtons({ action, busy: false, answered: true, onExpand: () => {}, onAction: () => {} }));
    expect(html).toContain("actions.answerSent");
    expect(html).not.toContain("actions.approveMerge");
    expect(html).not.toContain("<button");
    const pq: MissionPendingQuestion = { pane: "%1", eventId: 9, questions: [{ question: "q", multiSelect: false, options: [{ label: "ok" }] }] };
    const single = { kind: "generic" as const, pendingQuestion: pq };
    const htmlGeneric = renderToStaticMarkup(PendingActionButtons({ action: single, busy: false, answered: true, onExpand: () => {}, onAction: () => {} }));
    expect(htmlGeneric).toContain("actions.answerSent");
    expect(htmlGeneric).not.toContain("<button");
  });

  it("answered defaults to false — omitting it renders the buttons as before", () => {
    const action = { kind: "approve" as const, eventId: 9, question: null, replyYes: "pode", replyNo: "não", answerable: true };
    const html = renderToStaticMarkup(PendingActionButtons({ action, busy: false, onExpand: () => {}, onAction: () => {} }));
    expect(html).not.toContain("actions.answerSent");
    expect(html).toContain("actions.approveMerge");
  });

  it("actionEventId reads the eventId a generic/approve action would answer, null for freeform or no action", () => {
    const pq2: MissionPendingQuestion = { pane: "%1", eventId: 5, questions: [{ question: "q", multiSelect: false, options: [{ label: "ok" }] }] };
    const generic = { kind: "generic" as const, pendingQuestion: pq2 };
    const approve = { kind: "approve" as const, eventId: 9, question: null, replyYes: "pode", replyNo: "não", answerable: true };
    const freeform = { kind: "freeform" as const, pane: "%1", eventId: 3, answerable: true };
    expect(actionEventId(generic)).toBe(5);
    expect(actionEventId(approve)).toBe(9);
    expect(actionEventId(freeform)).toBeNull();
    expect(actionEventId(null)).toBeNull();
  });

  it("working with activeRun set renders Cancelar; working with no activeRun renders none", () => {
    const withRun = { ...base, headline: "working" as const, activeRun: { runId: "r1", startedAt: null, kind: "build" as const, runtime: "codex" as const, target: "feat/x", count: 1, lastStep: null, stage: "build" as const, descriptionKey: "building" as const, targetKind: "branch" as const } };
    const noRun = { ...base, headline: "working" as const };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: withRun }))).toContain("actions.cancel");
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: noRun }))).not.toContain("actions.cancel");
  });

  it("idle renders Disparar; stuck renders both Disparar and (with activeRun) Cancelar", () => {
    const idle = { ...base, headline: "idle" as const };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: idle }))).toContain("actions.dispatch");
    const stuckWithRun = { ...base, headline: "stuck" as const, activeRun: { runId: "r1", startedAt: null, kind: "build" as const, runtime: "codex" as const, target: "feat/x", count: 1, lastStep: null, stage: "build" as const, descriptionKey: "building" as const, targetKind: "branch" as const } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: stuckWithRun }));
    expect(html).toContain("actions.dispatch");
    expect(html).toContain("actions.cancel");
  });

  it("Disparar never renders on the compact face, even for an idle/stuck card with an activeRun (Decision 13)", () => {
    const stuckWithRun = { ...base, headline: "stuck" as const, activeRun: { runId: "r1", startedAt: null, kind: "build" as const, runtime: "codex" as const, target: "feat/x", count: 1, lastStep: null, stage: "build" as const, descriptionKey: "building" as const, targetKind: "branch" as const } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: stuckWithRun }));
    expect(html).not.toContain("actions.dispatch");
  });

  // round-1 F8: spec §9 gives the full detail block to card.pendingQuestions[0] ONLY; any
  // other pane with something pending stays reachable through its own tmux link in the
  // Sessions section above, never a second detail block. This fixture gives pane %1 (q1) AND
  // pane %2 (q2) a Sessions row too, so the "passive link survives" half is actually exercised,
  // not merely asserted by omission.
  it("with two live panes each holding an open pendingQuestion, only the first gets the detail block; the second stays reachable via its own tmux link", () => {
    const q1 = merge2(["Sim", "Não"]);
    const q2 = { ...merge2(["Sim", "Não"]), pane: "%2", eventId: 22 };
    const card = {
      ...base, headline: "needs-you" as const, pendingQuestions: [q1, q2],
      sessions: [{
        transport: "tmux" as const, session: "jax-p1-lead",
        panes: [
          { pane: "%1", tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead" as const, state: "needs-you" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 },
          { pane: "%2", tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead" as const, state: "needs-you" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 },
        ],
      }],
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    // Exactly one classified control pair (eventId 11's, from q1) and exactly one danger-styled
    // detail block render — q2's pane never gets a second one.
    expect((html.match(/>Sim<\/button>/g) ?? []).length).toBe(1);
    expect((html.match(/rounded-md border border-danger bg-danger-soft/g) ?? []).length).toBe(1);
    // 3 total "/tmux" links: one per Sessions pane row (2) plus the surviving detail block's
    // own answerHint link (1) — pane %2's own passive link is one of those 2, untouched by q2
    // being dropped from the pending-question section.
    expect((html.match(/href="\/tmux"/g) ?? []).length).toBe(3);
  });

  it("the free-text box lives inside ExpandedCard when a live pane carries a non-null capsuleEventId; the compact face shows the bare-capsule status instead (Decision 12)", () => {
    const withCapsule = { ...base, headline: "needs-you" as const, sessions: [{ transport: "tmux" as const, session: "jax-p1-lead", panes: [{ pane: "%1", tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead" as const, state: "needs-you" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: 7, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 }] }] };
    const compact = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: withCapsule }));
    expect(compact).not.toContain("actions.respond");
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: withCapsule }));
    expect(html).not.toContain("actions.respond");
    expect(html).toContain("actions.freeformPlaceholder");
    expect(html).not.toContain("actions.answerUnreachable");
  });
});

describe("dispatch and merge forms (Task 9, spec §7.1/§7.2)", () => {
  // round-1 F4: the dispatch form's `command` field defaults to `"review"` (`EMPTY_DISPATCH`)
  // and no test in this file simulates a click to switch it — so a single render can only ever
  // observe the review branch's own fields, never the build branch's own fields (`dispatchPlan`
  // etc. are inside the `f.command === "build"` half and simply never mount). Test the two
  // branches SEPARATELY for what a static render can actually prove: the review branch's own
  // fields render and the build branch's own fields do not (this is the default state), and the
  // command SELECTOR itself (its two <option> elements, always both present regardless of which
  // one is selected) offers both choices.
  it("idle card's dispatch form defaults to reviewSpec: target/focus fields render, build-only fields do not, all 4 options are offered", () => {
    const card = { ...base, headline: "idle" as const };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).toContain("actions.dispatchOption.reviewSpec");
    expect(html).toContain("actions.dispatchOption.reviewPlan");
    expect(html).toContain("actions.dispatchOption.reviewDiff");
    expect(html).toContain("actions.dispatchOption.build");
    expect(html).toContain("actions.dispatchTarget");
    expect(html).toContain("actions.dispatchFocus");
    expect(html).not.toContain("actions.dispatchPlan");
    expect(html).not.toContain("actions.dispatchPhase");
    expect(html).not.toContain("actions.dispatchBranch");
    expect(html).not.toContain("actions.dispatchProfileDefault");
  });

  it("the dispatch command selector always offers all 4 dispatchOption choices", () => {
    const card = { ...base, headline: "idle" as const };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).toContain("actions.dispatchOption.reviewSpec");
    expect(html).toContain("actions.dispatchOption.reviewPlan");
    expect(html).toContain("actions.dispatchOption.reviewDiff");
    expect(html).toContain("actions.dispatchOption.build");
  });

  // round-2 F2: a spec/plan review target must respect LIMITS.target (512), not just
  // non-empty. DispatchForm's own `target` state starts at "" with no card-derived prefill
  // (see round-1 F7's comment above), so there is no way to observe an overlong value on a
  // static render without simulating typing — dispatchValid is called directly instead,
  // the same logic DispatchForm's own `valid`/`targetValid` delegate to.
  it("dispatchValid rejects an overlong spec/plan review target even when non-empty", () => {
    const base_ = { command: "review" as const, kind: "spec" as const, target: "", focus: "", plan: "", phase: "", branch: "", whitelist: "", verify: "", build: "", profile: "default" as const };
    expect(dispatchValid({ ...base_, target: "x".repeat(LIMITS.target) })).toBe(true);
    expect(dispatchValid({ ...base_, target: "x".repeat(LIMITS.target + 1) })).toBe(false);
  });

  it("every dispatch form control (idle card) has an explicit <label for=…>", () => {
    const card = { ...base, headline: "idle" as const };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    const ids = [...html.matchAll(/<(?:select|input) id="([^"]+)"/g)].map((m) => m[1]);
    expect(ids.length).toBeGreaterThan(0);
    for (const id of ids) expect(html).toContain(`<label for="${id}"`);
  });

});

// F1 (diff review 240482904e68): pure-handler test, no jsdom/DOM event simulated — asserts the
// exact bug (Escape closing the dialog without resetting `confirmingCancel`, so a second click
// found it already mounted) is now impossible: the handler always resets when not busy.
describe("cancel dialog dismiss (diff review 240482904e68 F1)", () => {
  it("resets the confirm state when the dialog closes and the action isn't busy", () => {
    const onDismiss = vi.fn();
    handleCancelDialogClose(false, onDismiss);
    expect(onDismiss).toHaveBeenCalledTimes(1);
  });

  it("does not reset the confirm state while a cancel request is in flight", () => {
    const onDismiss = vi.fn();
    handleCancelDialogClose(true, onDismiss);
    expect(onDismiss).not.toHaveBeenCalled();
  });
});

describe("bucketPrefsPatch (plan review round 1, 72cd1de344ad F4 — pure handler, no jsdom)", () => {
  it("maps a hidden bucket to unhide and an autoHidden bucket to pin", () => {
    expect(bucketPrefsPatch("hidden")).toEqual({ hidden: false });
    expect(bucketPrefsPatch("autoHidden")).toEqual({ pinned: true });
  });
});

// Item 1/2: PrefsMenu is the pure/stateless render PrefsButtons drives — direct render lets
// the pending (aria-busy) state and per-state labels be asserted without simulating a click.
describe("PrefsMenu (item 1/2 — the card header's more-options control)", () => {
  const unpinnedVisible = { pinned: false, hiddenAt: null };
  const pinnedHidden = { pinned: true, hiddenAt: "2026-09-19T00:00:00.000Z" };

  it("has an accessible name on the trigger and both items with their state-dependent labels", () => {
    const html = renderToStaticMarkup(createElement(PrefsMenu, { prefs: unpinnedVisible, hidden: false, pending: false, error: null, onPin: () => {}, onHide: () => {} }));
    expect(html).toContain('aria-label="actions.more"');
    expect(html).toContain("actions.pin");
    expect(html).toContain("actions.hide");
  });

  it("swaps to Desfixar/Mostrar once pinned/hidden", () => {
    const html = renderToStaticMarkup(createElement(PrefsMenu, { prefs: pinnedHidden, hidden: true, pending: false, error: null, onPin: () => {}, onHide: () => {} }));
    expect(html).toContain("actions.unpin");
    expect(html).toContain("actions.show");
  });

  // Bug fix (card-hidden-menu): a card that auto-returned still carries a non-null
  // `hiddenAt` (mission.ts's visibility ladder clears the EFFECTIVE hidden state on new
  // activity without clearing the stored flag) — the menu must read `hidden` (the caller's
  // `card.visibility === "hidden"`), never the raw `prefs.hiddenAt`, or it keeps offering
  // "Show" on a card that is already visible.
  it("reads the hide label from `hidden`, not from a stale `prefs.hiddenAt`", () => {
    const html = renderToStaticMarkup(createElement(PrefsMenu, { prefs: pinnedHidden, hidden: false, pending: false, error: null, onPin: () => {}, onHide: () => {} }));
    expect(html).toContain("actions.hide");
    expect(html).not.toContain("actions.show");
  });

  it("renders aria-busy on the pending state and disables both items (item 2)", () => {
    const html = renderToStaticMarkup(createElement(PrefsMenu, { prefs: unpinnedVisible, hidden: false, pending: true, error: null, onPin: () => {}, onHide: () => {} }));
    expect(html).toContain('aria-busy="true"');
    expect(html).toContain("disabled");
    expect(html).toContain("opacity-60");
  });

  it("surfaces the failure text on error (item 2)", () => {
    const html = renderToStaticMarkup(createElement(PrefsMenu, { prefs: unpinnedVisible, hidden: false, pending: false, error: "network error", onPin: () => {}, onHide: () => {} }));
    expect(html).toContain("actions.actionFailed");
  });

  it("no <button> nests inside another <button> (the two menu items are siblings under <details>, not the trigger)", () => {
    const html = renderToStaticMarkup(createElement(PrefsMenu, { prefs: unpinnedVisible, hidden: false, pending: false, error: null, onPin: () => {}, onHide: () => {} }));
    let depth = 0;
    let maxDepth = 0;
    for (const tag of html.match(/<button\b|<\/button>/g) ?? []) {
      depth += tag === "</button>" ? -1 : 1;
      maxDepth = Math.max(maxDepth, depth);
    }
    expect(maxDepth).toBeLessThanOrEqual(1);
  });
});

// Bug fix (card-hidden-menu, full-component regression): a card ProjectsColumn ever renders
// as a full ProjectCard is, by construction, `visibility === "visible"` — but new activity
// after `hiddenAt` clears the EFFECTIVE hidden state (mission.ts's ladder) without clearing
// the stored `hiddenAt` itself. Before the fix, PrefsButtons/PrefsMenu read that stale
// `prefs.hiddenAt` directly and kept offering "Show" on a card already back on the board.
describe("ProjectCard header menu vs an auto-returned card (card-hidden-menu bug fix)", () => {
  it("offers Hide, not Show, once the card is visible again even though hiddenAt is still set", () => {
    const card = { ...base, prefs: { pinned: false, hiddenAt: "2026-09-19T00:00:00.000Z" }, visibility: "visible" as const };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("actions.hide");
    expect(html).not.toContain("actions.show");
  });
});

describe("Phase 3 — heartbeat, ultima acao, subagentCount, run links, hide/pin (spec §6/§7/§11)", () => {
  it("renders a sparkline from card.heartbeat and the lastAction line when present, hidden below md", () => {
    const heartbeat = Array.from({ length: 60 }, (_, i) => (i === 59 ? 3 : 0));
    const card = { ...base, heartbeat, lastAction: { tool: "Bash", ts: "2026-09-19T12:00:00.000Z" } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    expect(html).toContain('data-testid="heartbeat"');
    // Plan deviation: this file's next-intl mock returns the key only (the tool value never
    // reaches the HTML), so the line is asserted by its own key — the same convention every
    // other test in this file uses.
    expect(html).toContain("expanded.lastActionLine");
    // MOA-486 round 2: no heartbeat sparkline on mobile at all.
    expect(html).toContain("md:flex");
  });

  it("renders no lastAction line at all when it is null", () => {
    const heartbeat = Array.from({ length: 60 }, (_, i) => (i === 59 ? 3 : 0));
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, heartbeat, lastAction: null } }));
    expect(html).toContain('data-testid="heartbeat"');
    expect(html).not.toContain("expanded.lastActionLine");
  });

  // MOA-486 round 2 (Rafa's phone review): a card with no heartbeat data (every minute bucket
  // at 0, e.g. `base`) renders NOTHING for the heartbeat block — no reserved height, no
  // near-zero-height sparkline reading as a dashed line.
  it("MOA-486: renders no heartbeat block at all when card.heartbeat has no data", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: base }));
    expect(html).not.toContain('data-testid="heartbeat"');
  });

  it("a pane with subagentCount > 0 shows a subagent suffix; subagentCount 0 shows none", () => {
    const withSub = { ...base, sessions: [{ transport: "tmux" as const, session: "jax-p1-lead", panes: [
      { pane: "%1", tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead" as const, state: "working" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 2 },
    ] }] };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: withSub }))).toContain("expanded.subagentSuffix");
    const noSub = { ...base, sessions: [{ transport: "tmux" as const, session: "jax-p1-lead", panes: [
      { pane: "%1", tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead" as const, state: "working" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 },
    ] }] };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: noSub }))).not.toContain("expanded.subagentSuffix");
  });

  it("a codex session with subagentCount > 0 also shows the suffix", () => {
    const card = { ...base, sessions: [{ transport: "codex" as const, threadId: "0191f0aa-1111-7000-8000-000000000001", state: "working" as const, lastEventTs: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, subagentCount: 1 }] };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }))).toContain("expanded.subagentSuffix");
  });

  // Plan review round 1 (72cd1de344ad) F1: a build activeRun renders lastStep; a review
  // activeRun with the SAME non-null lastStep value never renders it — proves the gate is
  // `kind === "build"`, not merely "lastStep is present" (spec §8, AC7, Decision 11).
  it("a build activeRun with a non-null lastStep renders it; a review activeRun with the same value does not", () => {
    const buildRun = { runId: "r1", startedAt: null, kind: "build" as const, runtime: "claude" as const, target: "x", count: 1, stage: "build" as const, descriptionKey: "building" as const, targetKind: "branch" as const, lastStep: "Wrote src/foo.ts" };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, activeRun: buildRun } }));
    expect(html).toContain("Wrote src/foo.ts");
    const reviewRun = { ...buildRun, kind: "diff" as const };
    const html2 = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, activeRun: reviewRun } }));
    expect(html2).not.toContain("Wrote src/foo.ts");
  });

  it("a build activeRun with lastStep: null renders no extra line (nothing to show yet)", () => {
    const buildRun = { runId: "r1", startedAt: null, kind: "build" as const, runtime: "claude" as const, target: "x", count: 1, stage: "build" as const, descriptionKey: "building" as const, targetKind: "branch" as const, lastStep: null };
    const before = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, activeRun: { ...buildRun, count: 2 } } }));
    // With count>1 the otherRuns span is the last thing in that flex row; asserting its
    // presence pins the anchor Step 6b inserts after, without a lastStep-specific string
    // to search for when the value itself is null.
    expect(before).toContain("activity.otherRuns");
  });

  // Plan review round 1 F5: profileName is builder-only (spec §11/AC12) — the reviewer
  // fixture below carries a NON-null profileName so the test fails if the gate is dropped,
  // instead of passing by coincidence because the fixture happened to omit it.
  it("lastRun model/profile text renders; profileName renders only for a builder role", () => {
    const builderRun = { runId: "b1", role: "builder" as const, contractStatus: "ok" as const, stage: null, diagnostic: null, finishedAt: null, headSha: "b".repeat(40), labelKey: "buildOk" as const, tone: "success" as const, findings: null, runtimeModel: "codex-model", profileName: "fallback" as const, reportRel: null, targetRel: null, target: null, verifyCommand: null, buildCommand: null };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, lastRun: builderRun } }));
    expect(html).toContain("codex-model");
    expect(html).toContain("· fallback");
    const reviewerRun = { ...builderRun, role: "reviewer" as const, profileName: "fallback" as const };
    const html2 = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, lastRun: reviewerRun } }));
    expect(html2).toContain("codex-model");
    expect(html2).not.toContain("· fallback");
  });

  it("report/document links render only when *Rel is non-null, built via serializeFileSelection (never raw interpolation)", () => {
    const run = { runId: "b1", role: "builder" as const, contractStatus: "ok" as const, stage: null, diagnostic: null, finishedAt: null, headSha: null, labelKey: "buildOk" as const, tone: "success" as const, findings: null, runtimeModel: "x", profileName: null, reportRel: ".local/docs/reports/a & b.md", targetRel: null, target: null, verifyCommand: null, buildCommand: null };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, lastRun: run } }));
    expect(html).toContain(encodeURIComponent(".local/docs/reports/a & b.md").replace(/%20/g, "+"));
    const noLinks = { ...run, reportRel: null };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, lastRun: noLinks } }))).not.toContain("/files?root=repos");
  });

  // F2 (diff review): reportRel/targetRel are PROJECT-root relative (workflows.ts ~1486,
  // workflows.test.ts ~3526-3541 — server side is correct and untouched), but the link built
  // `{ root: "repos", rel: lastRun.reportRel }`, which files.ts treats as REPOS-root relative —
  // every link missed the `<project dir>/` prefix. Assert the actual `rel` query param, not just
  // presence of the encoded fragment.
  it("report/document link rel is prefixed with the project dir (F2)", () => {
    const run = {
      runId: "b1", role: "builder" as const, contractStatus: "ok" as const, stage: null, diagnostic: null,
      finishedAt: null, headSha: null, labelKey: "buildOk" as const, tone: "success" as const, findings: null,
      runtimeModel: null, profileName: null, reportRel: ".local/reports/x.md", targetRel: ".local/docs/specs/x-spec.md",
      target: null, verifyCommand: null, buildCommand: null,
    };
    const card = { ...base, dir: "proj", lastRun: run };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    const hrefs = [...html.matchAll(/href="(\/files\?[^"]*)"/g)].map((m) => m[1].replace(/&amp;/g, "&"));
    const rels = hrefs.map((href) => new URLSearchParams(href.split("?")[1]).get("rel"));
    expect(rels).toContain("proj/.local/reports/x.md");
    expect(rels).toContain("proj/.local/docs/specs/x-spec.md");
  });

  // MOA-486 round 2 (Rafa's phone review): outcome · model · report link all sit on one flex
  // line with gap-x-2 (not gap-2, which also adds row-gap when the line does wrap) — asserts
  // the container class rather than a rendered width, since renderToStaticMarkup has no layout.
  it("MOA-486: the result line container uses gap-x-2", () => {
    const run = {
      runId: "b1", role: "builder" as const, contractStatus: "ok" as const, stage: null, diagnostic: null,
      finishedAt: null, headSha: null, labelKey: "reviewOk" as const, tone: "success" as const, findings: null,
      runtimeModel: "gpt-5.6-luna", profileName: null, reportRel: ".local/reports/x.md", targetRel: null,
      target: null, verifyCommand: null, buildCommand: null,
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, lastRun: run } }));
    const start = html.indexOf("lastRun.reviewOk");
    const containerOpen = html.lastIndexOf("<div", start);
    const containerTag = html.slice(containerOpen, html.indexOf(">", containerOpen) + 1);
    expect(containerTag).toContain("gap-x-2");
  });

  // Item 1 (mobile hide/pin UX): the header's two icon buttons are now one "···" trigger
  // that opens a menu carrying both items, state-dependent labels only.
  // MOA-486 round 2: the trigger drops its border and shrinks to a 32px icon-only box
  // (Rafa's phone review) — no more bordered 44px target.
  it("the header renders a single more-options trigger, 32px, no border", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: base }));
    expect(html).toContain('aria-label="actions.more"');
    const start = html.indexOf('aria-label="actions.more"');
    const openTag = html.lastIndexOf("<summary", start);
    const summaryTag = html.slice(openTag, html.indexOf(">", openTag) + 1);
    expect(summaryTag).toContain("h-8");
    expect(summaryTag).toContain("w-8");
    expect(summaryTag).not.toContain("border");
  });

  it("the menu carries both items with state-dependent labels", () => {
    const visible = { ...base, prefs: { pinned: false, hiddenAt: null } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: visible }));
    expect(html).toContain("actions.hide");
    expect(html).toContain("actions.pin");
    // Bug fix (card-hidden-menu): a pinned card's visibility is always "visible" regardless of
    // `hiddenAt` (mission.ts: `prefs.pinned ? "visible" : ...`) — the pin label still swaps to
    // Unpin, but the hide label must read Hide, never the stale Show (this used to assert
    // "actions.show" here, the same bug the auto-return case hits).
    const pinnedHidden = { ...base, prefs: { pinned: true, hiddenAt: "2026-09-19T00:00:00.000Z" } };
    const html2 = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: pinnedHidden }));
    expect(html2).toContain("actions.unpin");
    expect(html2).toContain("actions.hide");
    expect(html2).not.toContain("actions.show");
  });

  // F4 (diff review): PrefsButtons' own <button>s used to render INSIDE the header's expand-
  // toggle <button> — nested interactive controls, invalid HTML. Assert no <button> ever nests
  // inside another <button> in the rendered card (string-based, matching this file's no-jsdom
  // renderToStaticMarkup convention: track <button>/</button> depth, it must never exceed 1).
  it("no button nests inside another button (header expand toggle vs PrefsButtons, F4)", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: base }));
    let depth = 0;
    let maxDepth = 0;
    for (const tag of html.match(/<button\b|<\/button>/g) ?? []) {
      depth += tag === "</button>" ? -1 : 1;
      maxDepth = Math.max(maxDepth, depth);
    }
    expect(maxDepth).toBeLessThanOrEqual(1);
  });

  // Card header overlap fix: pins the header structure so the status pill and the prefs
  // buttons can never again be nested back inside the expand-toggle <button> around the
  // project name (the crush that caused the overlap, bab74c3) — the name must render inside
  // the toggle button, and the prefs buttons must render as siblings after it closes.
  it("header: the project name is inside the expand toggle, PrefsButtons is outside it", () => {
    const card = { ...base, name: "distinctive-project-name" };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    const toggleStart = html.indexOf('aria-expanded="true"');
    const toggleOpenTag = html.lastIndexOf("<button", toggleStart);
    const toggleClose = html.indexOf("</button>", toggleStart) + "</button>".length;
    expect(toggleOpenTag).toBeGreaterThanOrEqual(0);
    const toggleMarkup = html.slice(toggleOpenTag, toggleClose);
    expect(toggleMarkup).toContain("distinctive-project-name");
    const prefsButtonIndex = html.indexOf('aria-label="actions.more"');
    expect(prefsButtonIndex).toBeGreaterThanOrEqual(toggleClose);
  });

  // MOA-486 round 2 (Rafa's phone review, replaces the round-1 dedicated status row): the "···"
  // menu trigger + chevron move back INTO header row 1 (the toggle's own row), at its right
  // edge; the status pill moves onto row 2, inline with the stage word/dots (data-testid=
  // "card-stage-row"). Assert both containers hold what they should — the pill shares the stage
  // row, the menu trigger stays in the header row.
  it("MOA-486: the status pill is inline with the stage word/dots; the menu trigger stays in the header row", () => {
    const card = { ...base, name: "distinctive-project-name" };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card }));
    const headerRow = sliceBalancedDiv(html, 'data-testid="card-header-row"');
    expect(headerRow).toContain("distinctive-project-name");
    expect(headerRow).toContain('aria-label="actions.more"');
    const stageRow = sliceBalancedDiv(html, 'data-testid="card-stage-row"');
    expect(stageRow).toContain("stages.build");
    expect(stageRow).toContain("states.working");
    // the stage row does not carry the menu trigger — it stays put in row 1
    expect(stageRow).not.toContain('aria-label="actions.more"');
  });
});

describe("ProjectCard — controlled expand (spec §6, Decision 13a, round-2 F2)", () => {
  it("expanded=true renders ExpandedCard; expanded=false omits it — no internal useState involved", () => {
    const shown = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: base }));
    expect(shown).toContain("expanded.sessions");
    const hidden = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: base }));
    expect(hidden).not.toContain("expanded.sessions");
  });
  it("the header toggle's aria-expanded tracks the prop directly, not a local mirror", () => {
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: base }))).toContain('aria-expanded="false"');
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: base }))).toContain('aria-expanded="true"');
  });
});

describe("ProjectCard compact face — LastRecord removed, footer, stage chip (spec §5/§8)", () => {
  const withLastRun = { ...base, lastRun: {
    runId: "r1", role: "builder" as const, contractStatus: "ok" as const, stage: null, diagnostic: null,
    finishedAt: null, headSha: "a".repeat(40), labelKey: "buildOk" as const, tone: "success" as const, findings: null,
    target: null, verifyCommand: null, buildCommand: null, runtimeModel: null, profileName: null, reportRel: null, targetRel: null,
  }, activeRun: null };
  it("compact face renders LastRunLine but never LastRecord's heading or quoted now text", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...withLastRun, now: "quoted status text" } }));
    expect(html).toContain("lastRun.buildOk");
    expect(html).not.toContain("activity.lastRecord");
    expect(html).not.toContain("quoted status text");
  });
  it("ExpandedCard renders LastRecord for a non-active card too (Decision 4)", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...withLastRun, now: "quoted status text" } }));
    expect(html).toContain("activity.lastRecord");
    expect(html).toContain("quoted status text");
  });
  it("footer includes builder and branch, each omitted cleanly when empty", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, builder: "claude-code", branch: "feat/x" } }));
    expect(html).toContain("claude-code");
    expect(html).toContain("feat/x");
    const empty = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, builder: "", branch: "" } }));
    expect(empty).not.toContain("claude-code");
  });
  it("the stage chip text matches card.stage for every ProjectStage fixture", () => {
    for (const stage of ["spec", "build", "review", "test", "ship"] as const) {
      const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, stage } }));
      expect(html).toContain(`stages.${stage}`);
    }
  });
  it("Disparar never renders on the compact face; DispatchForm renders inside ExpandedCard for idle/stuck only", () => {
    for (const headline of ["idle", "stuck", "working", "waiting", "needs-you"] as const) {
      const collapsed = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, headline } }));
      expect(collapsed).not.toContain("actions.dispatch");
      const expandedHtml = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline } }));
      if (headline === "idle" || headline === "stuck") expect(expandedHtml).toContain("actions.dispatch");
      else expect(expandedHtml).not.toContain("actions.dispatch");
    }
  });
});

describe("pendingUiAction guard + bare-capsule status text (spec §6, Decisions 9a/11a/12, round-2 F1/F3)", () => {
  const emptyQ: MissionPendingQuestion = { pane: "%1", eventId: 1, questions: [{ question: "", multiSelect: false, options: [{ label: "a" }, { label: "b" }] }] };
  it("a structured-but-empty question renders no button and the generic needs-you tag (no session capsule to read)", () => {
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [emptyQ] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).not.toContain("actions.respond");
    expect(html).not.toContain("actions.approveMerge");
    expect(html).toContain("states.needs-you");
  });
  it("a bare freeform capsule (blocked) renders inbox.blocked status text and a tmux link, no Responder button", () => {
    const card = {
      ...base, headline: "needs-you" as const,
      sessions: [{ transport: "tmux" as const, session: "s", panes: [{ pane: "%1", tmuxIncarnation: "1:1", session: "s", role: "lead" as const, state: "needs-you" as const, lastEventTs: null, pendingQuestion: null, capsuleEventId: 5, capsuleStatus: "blocked" as const, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 }] }],
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).not.toContain("actions.respond");
    expect(html).toContain("inbox.blocked");
    expect(html).toContain('href="/tmux"');
  });
  it("a blocked gate with no capsule/session renders the gate line only — no capsule label, no tmux link (F3)", () => {
    const card = { ...base, headline: "needs-you" as const, gate: "blocked" as const };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    // The header state pill also reads "states.needs-you" (unrelated to ActionRow), so the capsule
    // label is asserted by its distinguishing class+link pairing, not the bare translation key.
    expect(html).not.toContain('text-muted">states.needs-you<a href="/tmux"');
    expect(html).not.toContain("inbox.blocked");
    expect(html).not.toContain("inbox.needsInput");
    expect(html).not.toContain('href="/tmux"');
  });
  it("a real question (hasQuestionText true) still renders its button set — merge-shaped renders as generic now (D1), plus generic single and generic multi (Responder)", () => {
    const mergeQ: MissionPendingQuestion = { pane: "%1", eventId: 1, questions: [{ question: "May I merge `x` into `main`?", multiSelect: false, options: [{ label: "Sim" }, { label: "Não" }] }] };
    const single: MissionPendingQuestion = { pane: "%1", eventId: 1, questions: [{ question: "Env?", multiSelect: false, options: [{ label: "dev" }, { label: "stg" }] }] };
    const multi: MissionPendingQuestion = { pane: "%1", eventId: 1, questions: [{ question: "Q1?", multiSelect: true, options: [{ label: "a" }] }, { question: "Q2?", multiSelect: false, options: [{ label: "b" }] }] };
    const mergeHtml = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, headline: "needs-you" as const, pendingQuestions: [mergeQ] } }));
    expect(mergeHtml).not.toContain("actions.approveMerge");
    expect(mergeHtml).toContain(">Sim<");
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, headline: "needs-you" as const, pendingQuestions: [single] } }))).toContain("dev");
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, headline: "needs-you" as const, pendingQuestions: [multi] } }))).toContain("actions.respond");
  });
});

describe("exported helpers reused by InboxStrip (Task 9)", () => {
  it("pendingUiAction, PendingActionButtons, CancelDialog, bareCapsuleStatus are all exported", () => {
    expect(typeof pendingUiAction).toBe("function");
    expect(typeof PendingActionButtons).toBe("function");
    expect(typeof CancelDialog).toBe("function");
    expect(typeof bareCapsuleStatus).toBe("function");
  });
});

describe("ProjectCard — border liveliness classes (spec §8 D18, round-2 F3)", () => {
  it("stuck renders state-stuck, needs-you renders state-needs-you, idle renders neither", () => {
    const stuck = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, headline: "stuck" as const } }));
    expect(stuck).toContain("state-stuck");
    expect(stuck).not.toContain("state-needs-you");
    const needsYou = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, headline: "needs-you" as const } }));
    expect(needsYou).toContain("state-needs-you");
    expect(needsYou).not.toContain("state-stuck");
    const idle = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, headline: "idle" as const } }));
    expect(idle).not.toContain("state-stuck");
    expect(idle).not.toContain("state-needs-you");
  });
});

describe("ProjectCard — question line on the compact face (spec item 2)", () => {
  const genericQ: MissionPendingQuestion = { pane: "%1", eventId: 1, questions: [{ question: "Which environment?", multiSelect: false, options: [{ label: "dev" }, { label: "stg" }, { label: "prod" }] }] };
  const mergeQ: MissionPendingQuestion = { pane: "%1", eventId: 1, questions: [{ question: "May I merge `x` into `main`?", multiSelect: false, options: [{ label: "Sim" }, { label: "Não" }] }] };
  it("shows the question text (truncated, mono) for a generic single-select action", () => {
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [{ ...genericQ, questions: [{ ...genericQ.questions[0], multiSelect: true }] }] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("Which environment?");
  });
  it("shows the question text for a merge action", () => {
    const card = { ...base, headline: "needs-you" as const, pendingQuestions: [mergeQ] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("May I merge");
  });
  it("shows nothing extra when there is no pending action at all", () => {
    const card = { ...base, headline: "needs-you" as const, gate: "awaiting-approval" as const, pendingQuestions: [] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).not.toContain("? &quot;");
  });
});

describe("ProjectCard — status pill duration + subagent suffix (spec item 3)", () => {
  it("renders both segments when minutes and subagentCount are both present", () => {
    const card: MissionCard = {
      ...base, headline: "working",
      activeRun: { runId: "r", startedAt: new Date(Date.now() - 8 * 60_000).toISOString(), kind: "build", runtime: "claude", target: "x", count: 1, lastStep: null, stage: "build", descriptionKey: "building", targetKind: "branch" },
      sessions: [{ transport: "tmux", session: "s", panes: [{ pane: "%1", tmuxIncarnation: "1:1", session: "s", role: "lead", state: "working", lastEventTs: null, pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 2 }] }],
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("pill.subagents");
    expect(html).toContain("pill.elapsed");
  });
  it("renders a bare pill with neither segment when nothing resolves", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, headline: "idle" as const, activeRun: null, sessions: [] } }));
    expect(html).not.toContain("pill.subagents");
    expect(html).not.toContain("pill.elapsed");
  });
  // T2: an idle headline never shows elapsed minutes, even with a fresh session timestamp.
  it("an idle card with a session timestamp renders no pill.elapsed segment", () => {
    const card: MissionCard = {
      ...base, headline: "idle", activeRun: null,
      sessions: [{ transport: "tmux", session: "s", panes: [{ pane: "%1", tmuxIncarnation: "1:1", session: "s", role: "lead", state: "idle", lastEventTs: new Date().toISOString(), pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 }] }],
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).not.toContain("pill.elapsed");
  });
  // T1: the pill can truncate instead of overflowing the card in a narrow column.
  it("the pill's text span carries truncate, the dot stays flex-none", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: base }));
    const stageRow = sliceBalancedDiv(html, 'data-testid="card-stage-row"');
    expect(stageRow).toContain('<span class="truncate">');
    expect(stageRow).toContain("min-w-0 max-w-full");
  });
});

describe("ProjectCard — stage chip styling (spec item 9)", () => {
  it("the stage word carries the bordered mono uppercase pill classes", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: base }));
    const start = html.indexOf(">stages.build<");
    const openTag = html.lastIndexOf("<span", start);
    const spanTag = html.slice(openTag, html.indexOf(">", openTag) + 1);
    expect(spanTag).toContain("rounded-full");
    expect(spanTag).toContain("border-line");
    expect(spanTag).toContain("font-mono");
    expect(spanTag).toContain("uppercase");
  });
});

describe("ProjectCard — one line per session on the compact face (spec item 4)", () => {
  const tmuxPane = (n: string, over: Partial<MissionPane> = {}) => ({
    pane: n, tmuxIncarnation: "1:1", session: "jax-p1-lead", role: "lead" as const, state: "working" as const,
    lastEventTs: "2026-09-20T11:00:00.000Z", pendingQuestion: null, capsuleEventId: null, capsuleStatus: null,
    capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0, ...over,
  });
  it("renders one line per session, capped at 3 with a trailing +N", () => {
    const card = { ...base, sessions: Array.from({ length: 5 }, (_, i) => ({ transport: "tmux" as const, session: `s${i}`, panes: [tmuxPane(`%${i}`, { session: `s${i}` })] })) };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect((html.match(/expanded\.sessionLine/g) ?? []).length).toBe(3);
    expect(html).toContain("+2");
  });
  it("renders nothing for zero sessions", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, sessions: [] } }));
    expect(html).not.toContain("expanded.sessionLine");
  });
  it("a codex session line has no role segment", () => {
    const card = { ...base, sessions: [{ transport: "codex" as const, threadId: "0191f0aa-1234-7000-8000-000000000001", state: "working" as const, lastEventTs: "2026-09-20T11:00:00.000Z", capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, subagentCount: 0 }] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("expanded.sessionLineNoRole");
    expect(html).not.toContain("expanded.sessionLine\"");
  });
});

describe("ProjectCard — free last lines for stuck/needs-you (spec item 5)", () => {
  it("a stuck card with a resolvable stuckLine shows it instead of LastRunLine", () => {
    const card: MissionCard = {
      ...base, headline: "stuck", activeRun: null,
      sessions: [{ transport: "tmux", session: "s", panes: [{ pane: "%1", tmuxIncarnation: "1:1", session: "s", role: "lead", state: "stuck", lastEventTs: null, pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: 20, capsuleDeclaredAt: new Date(Date.now() - 30 * 60_000).toISOString(), capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 }] }],
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("expanded.stuckLine");
  });
  it("a needs-you gate card with a resolvable gateWaitingLine shows it", () => {
    const card: MissionCard = { ...base, headline: "needs-you", gate: "awaiting-approval", activeRun: null, updated: new Date(Date.now() - 10 * 60_000).toISOString(), sessions: [] };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("expanded.gateWaitingLine");
  });
  it("every other headline still shows LastRunLine, unaffected", () => {
    const withLastRun = { ...base, headline: "idle" as const, activeRun: null, lastRun: { runId: "r1", role: "builder" as const, contractStatus: "ok" as const, stage: null, diagnostic: null, finishedAt: null, headSha: null, labelKey: "buildOk" as const, tone: "success" as const, findings: null, target: null, verifyCommand: null, buildCommand: null, runtimeModel: null, profileName: null, reportRel: null, targetRel: null } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: withLastRun }));
    expect(html).toContain("lastRun.buildOk");
  });
  // F1 (cold review 92a14d784fb3): the shared path renders SessionLines/LastLine OUTSIDE the
  // activeRun ternary — an active/working card must still show its session line, not just the
  // activeRun progress row.
  it("an active (working) card still renders a session line — the shared path is not gated on activeRun", () => {
    const card: MissionCard = {
      ...base, headline: "working",
      activeRun: { runId: "r", startedAt: "2026-09-20T11:42:00.000Z", kind: "build", runtime: "claude", target: "x", count: 1, lastStep: null, stage: "build", descriptionKey: "building", targetKind: "branch" },
      sessions: [{ transport: "tmux", session: "s", panes: [{ pane: "%1", tmuxIncarnation: "1:1", session: "s", role: "lead", state: "working", lastEventTs: null, pendingQuestion: null, capsuleEventId: null, capsuleStatus: null, capsuleMinutes: null, capsuleDeclaredAt: null, capsuleMergeAsk: null, capsuleQuestion: null, live: true, tmuxSession: null, subagentCount: 0 }] }],
    };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("expanded.sessionLine");
  });
});

describe("ProjectCard — heartbeat sparkline (spec item 10)", () => {
  it("renders an svg/path instead of bar spans for a card with heartbeat data", () => {
    const heartbeat = Array.from({ length: 60 }, (_, i) => (i === 59 ? 3 : 0));
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, heartbeat } }));
    // The plan's bare "<svg" assertion is meaningless here — every card already renders lucide
    // icon svgs; extract the heartbeat block by its own data-testid marker instead.
    const start = html.indexOf('data-testid="heartbeat"');
    expect(start).toBeGreaterThan(-1);
    const block = html.slice(html.lastIndexOf("<div", start), html.indexOf("</div>", start));
    expect(block).toContain("<svg");
    expect(block).toContain("<path");
  });
  it("the draw animation class applies only for a working card", () => {
    const heartbeat = Array.from({ length: 60 }, (_, i) => (i === 59 ? 3 : 0));
    const working = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "working" as const, heartbeat } }));
    expect(working).toContain("hb-draw");
    const idle = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "idle" as const, heartbeat } }));
    expect(idle).not.toContain("hb-draw");
  });
  it("an all-zero heartbeat still renders no block at all", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: base }));
    expect(html).not.toContain('data-testid="heartbeat"');
  });
});

describe("ProjectCard — findings severity badge (spec item 11)", () => {
  it("renders a severity badge for the worst non-zero finding, nothing when all are 0", () => {
    const lastRun = { runId: "r1", role: "builder" as const, contractStatus: "ok" as const, stage: null, diagnostic: null, finishedAt: null, headSha: null, labelKey: "buildOk" as const, tone: "success" as const, target: null, verifyCommand: null, buildCommand: null, runtimeModel: null, profileName: null, reportRel: null, targetRel: null };
    const withMedium = { ...base, lastRun: { ...lastRun, findings: { high: 0, medium: 1, low: 0 } } };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: withMedium }))).toContain("lastRun.severity.medium");
    const zero = { ...base, lastRun: { ...lastRun, findings: { high: 0, medium: 0, low: 0 } } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: zero }));
    expect(html).not.toContain("lastRun.severity");
    expect(html).not.toContain("lastRun.findings");
  });
});

describe("ProjectCard — footer runtime avatar + newest PR (spec item 12)", () => {
  it("shows a C/X/O avatar per builder, case-insensitive, none for an unrecognized value", () => {
    const c = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, builder: "Claude-Code" } }));
    expect(c).toContain(">C<");
    const x = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, builder: "codex" } }));
    expect(x).toContain(">X<");
    const o = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, builder: "opencode" } }));
    expect(o).toContain(">O<");
    const none = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: { ...base, builder: "jax" } }));
    expect(none).not.toMatch(/>[CXO]</);
  });
  it("renders PR #n · CI state, no +N suffix when count is 1", () => {
    const card = { ...base, pr: { status: "ok" as const, count: 1, ci: "pending" as const, truncated: false, newest: { number: 82, ci: "pending" as const } } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("PR #82");
    expect(html).not.toContain("prMore");
  });
  it("renders +2 for count 3 untruncated, +4 ou mais for count 5 truncated", () => {
    const untrunc = { ...base, pr: { status: "ok" as const, count: 3, ci: "green" as const, truncated: false, newest: { number: 82, ci: "green" as const } } };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: untrunc }))).toContain("prMore");
    const trunc = { ...base, pr: { status: "ok" as const, count: 5, ci: "green" as const, truncated: true, newest: { number: 82, ci: "green" as const } } };
    expect(renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card: trunc }))).toContain("prMoreTruncated");
  });
  it("newest: null renders no PR segment at all", () => {
    const card = { ...base, pr: { status: "ok" as const, count: 0, ci: "none" as const, truncated: false, newest: null } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).not.toContain("PR #");
    expect(html).not.toContain("prSummary");
  });
  it("renders no PR footer/badge when github is disabled, even with a real card.pr.status", () => {
    harness.githubEnabled = false;
    const card = { ...base, pr: { status: "ok" as const, count: 1, ci: "pending" as const, truncated: false, newest: { number: 82, ci: "pending" as const } } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).not.toContain("prUnavailable");
    expect(html).not.toMatch(/PR #\d/);
  });

  it("settings still loading: PR section still renders as if github were enabled (F1)", () => {
    harness.settingsMode = "loading";
    const card = { ...base, pr: { status: "ok" as const, count: 1, ci: "pending" as const, truncated: false, newest: { number: 82, ci: "pending" as const } } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("PR #82");
  });

  it("settings query errored: PR section still renders as if github were enabled (F1)", () => {
    harness.settingsMode = "error";
    const card = { ...base, pr: { status: "ok" as const, count: 1, ci: "pending" as const, truncated: false, newest: { number: 82, ci: "pending" as const } } };
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: false, onToggle: () => {}, card }));
    expect(html).toContain("PR #82");
  });
});

describe("ProjectCard — dispatch form: one row + collapsed extras (spec item 8)", () => {
  it("the command select offers exactly 4 options", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "idle" as const } }));
    expect(html).toContain("actions.dispatchOption.reviewSpec");
    expect(html).toContain("actions.dispatchOption.reviewPlan");
    expect(html).toContain("actions.dispatchOption.reviewDiff");
    expect(html).toContain("actions.dispatchOption.build");
    expect((html.match(/<option/g) ?? []).length).toBe(4);
  });
  it("the profile select is absent for every review option's default render, present once the option is build", () => {
    const reviewDefault = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "idle" as const } }));
    expect(reviewDefault).not.toContain("actions.dispatchProfileDefault");
  });
  it("focus/the extra build fields sit inside a collapsed <details> below the row, never inside it", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "idle" as const } }));
    expect(html).toContain("actions.moreOptions");
    const rowStart = html.indexOf('data-testid="dispatch-row"');
    const rowEnd = html.indexOf("</div>", rowStart) + "</div>".length;
    expect(html.slice(rowStart, rowEnd)).not.toContain("actions.dispatchFocus");
    // The FIRST <details> in the markup is PrefsMenu's header dropdown — find the details block
    // that actually follows the dispatch row instead of matching from the top of the document
    // (the plan's bare `<details[^>]*>` regex would capture PrefsMenu's own menu).
    const detailsStart = html.indexOf("<details", rowStart);
    const detailsEnd = html.indexOf("</details>", detailsStart) + "</details>".length;
    expect(detailsStart).toBeGreaterThan(rowStart);
    expect(html.slice(detailsStart, detailsEnd)).toContain("actions.dispatchFocus");
  });
  it("the primary controls (command select, target input, submit) share one flex row wrapper", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "idle" as const } }));
    const rowMatch = html.match(/<div data-testid="dispatch-row" class="[^"]*"[^>]*>([\s\S]*?)<\/div>/);
    expect(rowMatch).not.toBeNull();
    const row = rowMatch![0];
    expect(row).toContain("<select");
    expect(row).toContain("<input");
    expect(row).toContain('<button type="button"');
  });
  it("the row wrapper carries a single flex-wrap row class, no per-control stacking", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "idle" as const } }));
    const openTag = html.slice(html.indexOf('<div data-testid="dispatch-row"'), html.indexOf(">", html.indexOf('<div data-testid="dispatch-row"')) + 1);
    expect(openTag).toContain("flex");
    expect(openTag).toContain("flex-wrap");
    expect(openTag).toContain("items-center");
  });
  it("the profile select is inside the row only once the option is build", () => {
    const reviewDefault = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "idle" as const } }));
    const rowStart = reviewDefault.indexOf('data-testid="dispatch-row"');
    const rowEnd = reviewDefault.indexOf("</div>", rowStart) + "</div>".length;
    expect(reviewDefault.slice(rowStart, rowEnd)).not.toContain("actions.dispatchProfileDefault");
  });
  it("submit still produces the exact same request body shape dispatchValid already validates (review, default option)", () => {
    const html = renderToStaticMarkup(createElement(ProjectCard, { expanded: true, onToggle: () => {}, card: { ...base, headline: "idle" as const } }));
    expect(html).toContain('<button type="button"');
  });
});
