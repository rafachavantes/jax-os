// The pure join (spec B §7.1, §7.2). Three independently-polled envelopes — status.md scans,
// the hub (DB + live tmux), and gh PR results — come together here as card models and inbox
// rows. No I/O, no node built-ins, no VALUE import from src/server/: this file is imported
// directly by "use client" components AND by vitest with no DB, no tmux, no gh (test 12).
// Every src/server/ import below is `import type` — erased at build, not a runtime dependency
// (the same convention src/server/collectors/projects.ts documents in the opposite direction).

import type { Envelope } from "./api";
import type { Project, ProjectGate, ProjectScan, ProjectStage } from "../server/collectors/projects";
import type { Capsule, HubActiveRun, HubActiveRunSummary, HubCodexSession, HubEnvelopeData, HubLastRun, HubPane, HubProject, HubTimelineEvent, LoopSummary, PaneRuntime, PendingQuestion } from "../server/db/workflows";
import type { CiState, Pr, PrResult } from "../server/collectors/github";
import type { Commit } from "../server/collectors/activity";
import type { WorktreeResult } from "../server/collectors/worktrees";
import { CONTRACT_STATUSES, STAGES, paneKey, type ContractStatus, type Role, type Stage } from "./workflow";

export type MissionState = "unknown" | "needs-you" | "stuck" | "working" | "waiting" | "idle";

export type MissionQuestion = {
  question: string;
  header?: string;
  multiSelect: boolean;
  options: { label: string; description?: string }[];
};

export type MissionPendingQuestion = { pane: string; eventId: number; questions: MissionQuestion[] };

export type MissionPane = {
  pane: string;
  // Sessions mosaic spec §5 Decision 1 (round-1 F1): copied straight from HubPane.live — a dead
  // pane's stale pendingQuestion/capsuleStatus must never surface as a mosaic cell. Callers filter
  // on THIS field before ever calling mosaicPaneState (Task 2), never by re-deriving liveness
  // from `.state`.
  live: boolean;
  tmuxIncarnation: string;
  session: string;
  // Sessions mosaic spec §5 Decision 4a (round-2 F1): the RAW HubPane.session, no fallback.
  // `session` above stays the pre-existing DISPLAY label (`p.session ?? p.pane`) — every action
  // that reaches real tmux (Take Control, the desktop detail-view target) must read
  // `.tmuxSession`, never `.session`, or it can target a bare pane id as if it were a tmux
  // session name (Take Control 404s: `{ok:false, error:"session not found"}`).
  tmuxSession: string | null;
  role: "lead" | "adhoc";
  state: MissionState;
  lastEventTs: string | null;
  pendingQuestion: MissionPendingQuestion | null;
  // Spec §6, Decision 11a (round-2 F1): the same value the strip already reads under a
  // different name (InboxRow's attention.status) — narrowed identically to how paneState
  // and the attentionRows push already narrow p.capsule?.status, just exposed here too so
  // ActionRow/pendingUiAction (Part 2) has an equivalent on the CARD side.
  capsuleStatus: "needs_input" | "blocked" | null;
  // Round-3 item 5: straight pass-through from HubPane.capsule — the fields already exist
  // server-side (workflows.ts:839's Capsule type), just not yet projected onto the client shape.
  capsuleMinutes: number | null;
  capsuleDeclaredAt: string | null;
  // Merge-ask spec §3 point 6: mirrored from HubPane.capsule.mergeAsk/question — tmux panes
  // only, never a native Codex session (MissionCodexSession carries neither field, §5/§7's
  // transport guard).
  capsuleMergeAsk: number | null;
  // Merge question contract: from HubPane.capsule.mergeBranch/mergeTarget. Optional so fixtures that never
  // exercise the button need no change; absent means "no canonical merge question".
  capsuleMergeBranch?: string; capsuleMergeTarget?: string;
  capsuleQuestion: string | null;
  // D4: optional (not required like the two fields above) — every real HubPane.capsule already
  // carries it, this is only about not forcing every test fixture in the repo that never
  // exercises this feature to also set it (see the tsc-baseline note in Global Constraints).
  capsuleAnswerable?: boolean;
  // Session-runtime-label fix: mirrored from HubPane.runtime — optional for the same
  // fixture-blast-radius reason as capsuleAnswerable above. Absent/"unknown" means the caller
  // falls back to card.builder (ProjectCard.tsx).
  runtime?: PaneRuntime;
  // Phase 3 §7: open subagents on this pane (Decision 7), rendered as a suffix when > 0.
  subagentCount: number;
  // Phase 2 Decision 21: the last capsule's event id — the free-text "Responder" target when the
  // pane is needs-you by capsule (needs_input/blocked) with no structured question.
  capsuleEventId: number | null;
};

// Discriminated live-session row (MOA-469 §4): a tmux pane group or a native Codex thread. A
// native row never fabricates a pane or incarnation and renders no tmux action.
// The card's session list shows only what is alive: a dead tmux pane can't work, ask or be
// answered (MOA-469 already keeps it off the inbox), so it is history, not a session. A Codex
// thread the app-server no longer has loaded (state "unknown": notLoaded/systemError/no runtime
// row) is dead the same way. Headline/staleness still read card.sessions.
export function liveSessions(sessions: MissionSession[]): MissionSession[] {
  return sessions.flatMap((s): MissionSession[] => {
    if (s.transport !== "tmux") return s.state === "unknown" ? [] : [s];
    const panes = s.panes.filter((p) => p.live);
    return panes.length > 0 ? [{ ...s, panes }] : [];
  });
}

export type MissionSession =
  | { transport: "tmux"; session: string; panes: MissionPane[] }
  | {
      transport: "codex"; threadId: string; state: MissionState; lastEventTs: string | null; subagentCount: number;
      capsuleEventId: number | null; capsuleStatus: "needs_input" | "blocked" | null; capsuleMinutes: number | null;
      capsuleDeclaredAt: string | null;
      // D2/D4: mirrored from HubCodexSession.capsule.mergeAsk/question/answerable the same way
      // MissionPane already mirrors HubPane.capsule — optional for the same fixture-blast-radius
      // reason as MissionPane.capsuleAnswerable above.
      capsuleMergeAsk?: number | null; capsuleQuestion?: string | null; capsuleAnswerable?: boolean; capsuleMergeBranch?: string; capsuleMergeTarget?: string;
    };

export type MissionTimelineEntry = {
  ts: string; type: string; pane: string | null; role: string | null; capsuleStatus: string | null;
  outcome: string | null; contractStatus: ContractStatus | null; stage: Stage | null; diagnostic: string | null;
};

export type MissionCardPr =
  | { status: "unknown" }
  | { status: "disabled" }
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "ok"; count: number; ci: CiState; truncated: boolean; newest: { number: number; ci: CiState } | null };

export type MissionActiveRun = HubActiveRun & {
  stage: "build" | "review" | null;
  descriptionKey: "building" | "reviewSpec" | "reviewPlan" | "reviewDiff" | "generic";
  targetKind: "branch" | "document" | null;
};

// Same stage/descriptionKey/targetKind projection as MissionActiveRun, one per in-flight run
// (Sessions card line) instead of just the newest.
export type MissionActiveRunSummary = HubActiveRunSummary & {
  stage: "build" | "review" | null;
  descriptionKey: "building" | "reviewSpec" | "reviewPlan" | "reviewDiff" | "generic";
  targetKind: "branch" | "document" | null;
};

export type LastRunTone = "success" | "danger" | "warning";
export type LastRunLabelKey =
  | "buildOk" | "buildFailed" | "buildBlocked" | "buildInterrupted" | "buildCancelled" | "buildNoResult"
  | "reviewOk" | "reviewChanges" | "reviewRejected" | "reviewInterrupted" | "reviewCancelled" | "reviewNoVerdict";

export type MissionLastRun = {
  runId: string;
  role: Role;
  contractStatus: ContractStatus;
  stage: Stage | null;
  diagnostic: string | null;
  finishedAt: string | null;
  headSha: string | null;
  labelKey: LastRunLabelKey;
  tone: LastRunTone;
  // Round-3 F1: straight pass-through from HubLastRun, same style as stage/diagnostic.
  findings: { high: number; medium: number; low: number } | null;
  // Phase 2 (Decision 9): straight pass-through from HubLastRun for the --checks prefill.
  target: string | null;
  verifyCommand: string | null;
  buildCommand: string | null;
  // Phase 3 §11 (Decision 17): pass-through, same style as stage/diagnostic. profileName is
  // builder-only (null for a reviewer row, enforced where the run is classified server-side).
  runtimeModel: string | null;
  profileName: string | null;
  reportRel: string | null;
  targetRel: string | null;
};

export type MissionCard = {
  dir: string;
  name: string;
  stage: ProjectStage;
  branch: string;
  builder: string;
  flag?: string;
  gate: ProjectGate | null;
  legacyGateField: boolean;
  now: string;
  residuals: string[];
  updated: string;
  headline: MissionState;
  hubReady: boolean;
  lastEventAgo: string | null;
  freshnessCount: number | null;
  freshnessWarning: boolean;
  sessions: MissionSession[];
  pendingQuestions: MissionPendingQuestion[];
  timeline: MissionTimelineEntry[];
  pr: MissionCardPr;
  activeRun?: MissionActiveRun | null;
  activeRuns?: MissionActiveRunSummary[];
  lastRun?: MissionLastRun | null;
  retainedWorktrees?: number | null;
  loopSummary?: LoopSummary | null;
  codexSource: "ok" | "failed" | "truncated" | "absent";
  codexUntracked: boolean;
  // Phase 3 §6/§10 (Decision 20, round 3 F1): heartbeat/hide/pin data joined from the envelope,
  // never through `hp` — a status-only project with no HubProject entry keeps its real prefs.
  heartbeat: number[];
  lastAction: { tool: string; ts: string } | null;
  prefs: { pinned: boolean; hiddenAt: string | null };
  visibility: "visible" | "hidden" | "autoHidden" | "archived";
};

export type InboxRow =
  | { kind: "gate"; dir: string; name: string; gate: "awaiting-approval" | "blocked"; updated: string }
  | { kind: "question"; dir: string; name: string; pane: string; eventId: number; question: string }
  | { kind: "attention"; dir: string; name: string; status: "blocked" | "needs_input"; transport: "tmux"; pane: string; tmuxIncarnation: string }
  | { kind: "attention"; dir: string; name: string; status: "needs_input"; transport: "codex"; threadId: string }
  | { kind: "pr"; dir: string; name: string; number: number; url: string; ci: CiState };

export type MissionModel = {
  cards: MissionCard[];
  uncardedProjectNames: string[];
  historicalTruncated: boolean;
  skipped: number;
  inbox: InboxRow[];
  inboxTotalCount: number;
  inboxPrRemainder: number;
  // Finding 5 (branch review): `inboxPrRemainder` is derived from the fetched PR sample per
  // project — if ANY contributing project's own fetch was truncated (github.ts's own PR-count
  // cap), the true remainder beyond the inbox cap can be larger than what was actually counted.
  // Callers use this to render the remainder as "at least N" instead of an exact count.
  prTruncatedAny: boolean;
  codexSource: "ok" | "failed" | "truncated" | "absent";
  // tmux liveness is a source independent of Codex (C4): a failed collector is a visible warning,
  // never a quiet healthy zero-agents board.
  tmuxSource: "ok" | "failed";
  // Phase 3 §10 (Decision 20): the envelope's settings value copied once, plus client-derived
  // per-visibility counts (used by the projects column's management rows).
  archiveAfterDays: number;
  hiddenCount: number;
  autoHiddenCount: number;
  reposRoot: string;
};

export type SourceName = "projects" | "hub" | "prs";

// Confidence about the three sources — NOT data. There is deliberately no "empty" variant here:
// the empty title is a two-clause check the caller makes against `model` itself
// (readiness.kind === "ready" && model.inboxTotalCount === 0, InboxStrip's job, Task 8), not a
// type-level claim this module enforces. TypeScript is structural and cannot make a variant
// "unconstructible" — an earlier draft claimed exactly that about {kind:"empty"} and it wasn't
// true (cold review round 3, Finding 8).
export type MissionReadiness =
  | { kind: "loading"; pending: SourceName[] }
  // `pending` here too (post-merge cold review, Finding 4, MEDIUM): a failed backbone source
  // (e.g. `hub`) does not imply `projects` has resolved — the two are independent envelopes, and
  // dropping `projects`'s own still-loading status here left ProjectsColumn with no way to tell
  // "hub failed, projects still in flight" apart from "hub failed, projects already resolved
  // empty" — it fell through to a false empty state (Task 9).
  | { kind: "failed"; failed: SourceName[]; pending: SourceName[] }
  | { kind: "degraded"; degraded: SourceName[] }
  | { kind: "ready" };

export type Mission = { readiness: MissionReadiness; model: MissionModel };

const FRESHNESS_WARNING_THRESHOLD = 20;
const INBOX_PR_CAP = 5;
const EMPTY_SCAN: ProjectScan = { projects: [], skipped: 0, reposRoot: "" }; // spec §7.3's own pinned empty value

// Same formula as evaluateStaleness (src/server/db/workflows.ts): declaredAt + max(2min, 20% of
// the declared window). Duplicated deliberately — this module cannot import the DB-backed
// original without pulling a src/server value import into a client-importable file.
function isPastStuckDeadline(declaredAtIso: string, capsuleMinutes: number, nowMs: number): boolean {
  const declaredAtMs = Date.parse(declaredAtIso);
  const n = capsuleMinutes;
  const deadlineMs = declaredAtMs + (n * 60 + Math.max(120, n * 60 * 0.2)) * 1000;
  return nowMs > deadlineMs;
}

// Per-pane state. Liveness is checked FIRST (MOA-469 live correction): a dead pane's stale
// needs_input/blocked capsule or pending question must never surface as current attention.
// `working` already combines "live" and "has an open turn" (Task 4's
// isWorking()) — since the branch below only reaches it after `p.live` is confirmed true, using
// `working` here is exactly equivalent to a standalone "open turn" check in that position.
export function paneState(p: HubPane, nowMs: number): MissionState {
  if (!p.live) return "idle";
  if (p.pendingQuestion || p.capsule?.status === "blocked" || p.capsule?.status === "needs_input") {
    return "needs-you";
  }
  if (p.working) return "working";
  // Phase 3 Decision 8: a stopped main turn with an open subagent still reads as working.
  if (p.subagentCount > 0) return "working";
  if (p.capsule?.status === "waiting") {
    return typeof p.capsule.minutes === "number" &&
      isPastStuckDeadline(p.capsule.declaredAt, p.capsule.minutes, nowMs)
      ? "stuck" : "waiting";
  }
  return "idle";
}

export type MosaicPaneState = "working" | "waiting-for-input" | "blocked-permission" | "idle-stale";

// Sessions mosaic spec §6: the 3-level Codeman pulse + idle-stale fold, checked top to bottom,
// first match wins. Called ONLY on panes that already survived the live===true filter (Task 3's
// mosaicGroups) — a dead pane never reaches this function (round-1 F1), so it never needs
// `.live` itself. Deliberately collapses paneState's four non-attention values (idle, waiting,
// stuck, unknown) into one static "idle-stale" reading — full six-way nuance stays one click
// away in the existing card/viewer, never lost, just not painted on a small cell.
export function mosaicPaneState(pane: Pick<MissionPane, "state" | "capsuleStatus" | "pendingQuestion">): MosaicPaneState {
  if (pane.capsuleStatus === "blocked") return "blocked-permission";
  if (pane.capsuleStatus === "needs_input" || pane.pendingQuestion !== null) return "waiting-for-input";
  if (pane.state === "working") return "working";
  return "idle-stale";
}

export type MosaicPaneAction =
  | { kind: "options"; pendingQuestion: MissionPendingQuestion }
  | { kind: "respond" }
  | { kind: "open-pane"; capsuleEventId: number | null }
  | { kind: "none" };

// Sessions mosaic spec §8 Decision 17 (round-1 F2/F3): PANE-scoped — reads pendingQuestion/
// capsuleStatus/capsuleEventId directly, NEVER pendingUiAction(card, …)/selectPendingAction(card):
// a card-level match can select ANOTHER pane's question, and a card's gate/mergeNow action must
// never leak onto an unrelated pane's cell. Deliberately does NOT reuse hasQuestionText (Decision
// 9a) — that predicate takes a card-shaped `{pendingQuestions}` input; this one takes a single
// pane's own pendingQuestion, a different shape, so the "real text" check is inlined below instead
// of forcing an adapter over a one-line predicate. `capsuleEventId` is carried on the "open-pane"
// result (round-2 review F1) because that is the event the blocked-permission action targets —
// Part 2's "abrir pane"/detail link needs it, and reading it straight off `pane` (never through
// `pendingUiAction`) keeps this function pane-scoped by construction.
export function mosaicPaneAction(pane: Pick<MissionPane, "pendingQuestion" | "capsuleStatus" | "capsuleEventId">): MosaicPaneAction {
  const pq = pane.pendingQuestion;
  if (pq !== null) {
    const q0 = pq.questions[0];
    const singleSimple = pq.questions.length === 1 && q0 !== undefined && !q0.multiSelect
      && typeof q0.question === "string" && q0.question.trim().length > 0;
    return singleSimple ? { kind: "options", pendingQuestion: pq } : { kind: "respond" };
  }
  if (pane.capsuleStatus === "needs_input") return { kind: "respond" };
  if (pane.capsuleStatus === "blocked") return { kind: "open-pane", capsuleEventId: pane.capsuleEventId };
  return { kind: "none" };
}

// Native Codex session state (MOA-469 §2/§4). Live runtime flags take precedence; idle consults
// this session's own pending attention and in-flight runs before its last capsule. "unknown" means
// the live source is unavailable — it is never evidence of work, and headlineFor excludes it so a
// source warning never hides a genuinely working Claude pane. Unavailable statuses are classified
// BEFORE any stale capsule/attention is consulted (MOA-469 live correction): a not-loaded/systemError
// session must never promote old attention into a current action.
function codexSessionState(s: HubCodexSession, nowMs: number): MissionState {
  if (!s.sourceOk) return "unknown";
  if (s.status === "notLoaded" || s.status === "systemError" || s.status === "unknown") return "unknown";
  if (s.attention || s.pendingAttention) return "needs-you";
  if (s.working) return "working";
  // Phase 3 Decision 8: same subagent fold as paneState above, keyed on harness_session server-side.
  if (s.subagentCount > 0) return "working";
  if (s.capsule?.status === "blocked" || s.capsule?.status === "needs_input") return "needs-you";
  if (s.runInFlight) return "waiting";
  if (s.capsule?.status === "waiting") {
    return typeof s.capsule.minutes === "number" &&
      isPastStuckDeadline(s.capsule.declaredAt, s.capsule.minutes, nowMs)
      ? "stuck" : "waiting";
  }
  return "idle";
}

// Spec §6 Decision 11a (round-2 F1): the blocked/needs_input narrowing both MissionPane and the
// codex MissionSession arm expose as `capsuleStatus` — the same narrowing paneState applies above.
// Explicitly typed so TS keeps the literal union through a contextually-typed object literal.
function capsuleStatusOf(capsule: Capsule | null): "needs_input" | "blocked" | null {
  const s = capsule?.status;
  return s === "blocked" || s === "needs_input" ? s : null;
}

const STATE_RANK: Record<MissionState, number> = {
  "needs-you": 5, unknown: 4, stuck: 3, working: 2, waiting: 1, idle: 0,
};

function worstState(states: MissionState[]): MissionState {
  return states.reduce((worst, s) => (STATE_RANK[s] > STATE_RANK[worst] ? s : worst), "idle" as MissionState);
}

function headlineFor(gate: ProjectGate | null, panes: HubPane[], codexSessions: HubCodexSession[], nowMs: number): MissionState {
  const states = [
    ...panes.map((p) => paneState(p, nowMs)),
    // Native "unknown" (source unavailable / not loaded) is a visible warning on the row, never a
    // headline downgrade that hides a genuinely working pane (MOA-469 §4).
    ...codexSessions.map((s) => codexSessionState(s, nowMs)).filter((s) => s !== "unknown"),
  ];
  // A gate waits on Rafa only once the tech lead has stopped. After `approve-with-changes` the
  // tech lead is still applying the findings: the branch is merge-eligible, but nobody owes an
  // approval yet, so a working pane keeps the card "working" and the merge button off
  // (2026-09-21, arc: "Merge now" shown while the fixes were being applied).
  // Same while it waits on a run it dispatched itself (2026-09-24, arc: a stale gate showed
  // needs-you during a correction build).
  if (gate !== null) {
    if (states.includes("working")) return "working";
    return states.includes("waiting") ? "waiting" : "needs-you";
  }
  if (states.length === 0) return "idle";
  return worstState(states);
}

// A pending question's payload is the raw `question` event payload (Record<string, unknown>,
// unvalidated) — this module has no I/O, so it defensively re-parses it into typed
// MissionQuestion[] rather than trusting the shape (`src/lib/workflow.ts`'s own
// `questionShapes()` does the equivalent validation server-side, for the answer-claiming path).
function parseQuestions(payload: Record<string, unknown>): MissionQuestion[] {
  const raw = payload.questions;
  if (!Array.isArray(raw)) return [];
  const out: MissionQuestion[] = [];
  for (const q of raw) {
    if (!q || typeof q !== "object") continue;
    const r = q as Record<string, unknown>;
    if (typeof r.question !== "string" || !Array.isArray(r.options)) continue;
    const options = r.options
      .filter((o): o is Record<string, unknown> => !!o && typeof o === "object" && typeof (o as Record<string, unknown>).label === "string")
      .map((o) => ({ label: o.label as string, description: typeof o.description === "string" ? o.description : undefined }));
    out.push({
      question: r.question,
      header: typeof r.header === "string" ? r.header : undefined,
      multiSelect: r.multiSelect === true,
      options,
    });
  }
  return out;
}

function toMissionPendingQuestion(pq: PendingQuestion): MissionPendingQuestion {
  return { pane: pq.pane, eventId: pq.eventId, questions: parseQuestions(pq.payload) };
}

// Shared by toMissionActiveRun (the single newest-run summary) and toMissionActiveRunSummary
// (one line per in-flight run) — the same kind -> stage/description/targetKind projection either
// way, so the two can never drift.
function stageForKind(kind: HubActiveRun["kind"]): { stage: MissionActiveRun["stage"]; descriptionKey: MissionActiveRun["descriptionKey"]; targetKind: MissionActiveRun["targetKind"] } {
  switch (kind) {
    case "build":
      return { stage: "build", descriptionKey: "building", targetKind: "branch" };
    case "spec":
      return { stage: "review", descriptionKey: "reviewSpec", targetKind: "document" };
    case "plan":
      return { stage: "review", descriptionKey: "reviewPlan", targetKind: "document" };
    case "diff":
      return { stage: "review", descriptionKey: "reviewDiff", targetKind: "branch" };
    default:
      return { stage: null, descriptionKey: "generic", targetKind: null };
  }
}

function toMissionActiveRun(run: HubActiveRun | null | undefined): MissionActiveRun | null {
  if (!run) return null;
  return { ...run, ...stageForKind(run.kind) };
}

function toMissionActiveRunSummary(run: HubActiveRunSummary): MissionActiveRunSummary {
  return { ...run, ...stageForKind(run.kind) };
}

// The rendering table (spec §12.4): (role, contractStatus, outcome, stage) -> labelKey/tone.
// Order matters and is fixed: `stage: worker` always wins first (an interrupted worker close
// has no useful cause, regardless of role or a stale outcome value); `cancelled` next (its
// own literal, never a report-derived label); a genuinely legacy row (no stage AND no
// outcome at all -- §Backward compatibility, predates this spec) gets the *same* literal
// label its role would otherwise use for "no verdict"/"no result", but at `warning` tone
// instead of `danger` -- the two are the same TEXT with a different confidence color, so
// `reviewNoVerdict` is intentionally reused for both the legacy case and the "we know it's
// missing/invalid" case below, distinguished only by `tone`.
function classifyLastRun(role: Role, contractStatus: ContractStatus, outcome: string | null, stage: Stage | null): { labelKey: LastRunLabelKey; tone: LastRunTone } {
  if (stage === "worker") {
    return role === "reviewer" ? { labelKey: "reviewInterrupted", tone: "warning" } : { labelKey: "buildInterrupted", tone: "warning" };
  }
  if (contractStatus === "cancelled") {
    return role === "reviewer" ? { labelKey: "reviewCancelled", tone: "warning" } : { labelKey: "buildCancelled", tone: "warning" };
  }
  if (stage === null && outcome === null) {
    return role === "reviewer" ? { labelKey: "reviewNoVerdict", tone: "warning" } : { labelKey: "buildNoResult", tone: "warning" };
  }
  if (role === "reviewer") {
    if (outcome === "approve") return { labelKey: "reviewOk", tone: "success" };
    if (outcome === "approve-with-changes") return { labelKey: "reviewChanges", tone: "success" };
    if (outcome === "reject") return { labelKey: "reviewRejected", tone: "danger" };
    return { labelKey: "reviewNoVerdict", tone: "danger" }; // missing/invalid, or interrupted row 1a
  }
  if (outcome === "success") return { labelKey: "buildOk", tone: "success" };
  if (outcome === "blocked") return { labelKey: "buildBlocked", tone: "danger" };
  return { labelKey: "buildFailed", tone: "danger" }; // failure, any stage, incl. interrupted row 1a
}

function toMissionLastRun(run: HubLastRun | null | undefined): MissionLastRun | null {
  if (!run) return null;
  const { labelKey, tone } = classifyLastRun(run.role, run.contractStatus, run.outcome, run.stage);
  return {
    runId: run.runId, role: run.role, contractStatus: run.contractStatus, stage: run.stage,
    diagnostic: run.diagnostic, finishedAt: run.finishedAt, headSha: run.headSha, labelKey, tone,
    findings: run.findings, target: run.target, verifyCommand: run.verifyCommand, buildCommand: run.buildCommand,
    runtimeModel: run.runtimeModel, profileName: run.profileName, reportRel: run.reportRel, targetRel: run.targetRel,
  };
}

// capsuleStatus renders only for turn-stopped rows — every other timeline entry shows type/pane
// alone (§7.4's mockup).
function timelineCapsuleStatus(e: HubTimelineEvent): string | null {
  if (e.type !== "turn-stopped" || !e.payload || typeof e.payload !== "object") return null;
  const status = (e.payload as Record<string, unknown>).capsule_status;
  return typeof status === "string" ? status : null;
}

// outcome/contractStatus/stage/diagnostic render only for run-finished timeline rows (the
// same conditional pattern timelineCapsuleStatus already uses for turn-stopped) -- raw,
// untranslated values, matching how e.type/capsuleStatus already render as raw enum strings
// on this same list (ProjectCard.tsx's timeline <li>).
function timelineRunFinishedFields(e: HubTimelineEvent): Pick<MissionTimelineEntry, "outcome" | "contractStatus" | "stage" | "diagnostic"> {
  if (e.type !== "run-finished" || !e.payload || typeof e.payload !== "object") {
    return { outcome: null, contractStatus: null, stage: null, diagnostic: null };
  }
  const p = e.payload as Record<string, unknown>;
  const contractStatus = typeof p.contract_status === "string" && (CONTRACT_STATUSES as readonly string[]).includes(p.contract_status)
    ? p.contract_status as ContractStatus : null;
  // Reviewer verdict counts only on an ok report (spec §7).
  const outcome = e.role === "reviewer"
    ? (contractStatus === "ok" && typeof p.verdict === "string" ? p.verdict : null)
    : (typeof p.result === "string" ? p.result : null);
  const stage = typeof p.stage === "string" && (STAGES as readonly string[]).includes(p.stage)
    ? p.stage as Stage : null;
  const diagnostic = typeof p.diagnostic === "string" ? p.diagnostic : null;
  return { outcome, contractStatus, stage, diagnostic };
}

// §5: "no CI" (`ciState: "none"`) only when NO pr in the set has any check; otherwise the worst
// among the ones that do (failing beats pending beats green) — mirrors
// src/server/collectors/github.ts's summarizePrState() exactly (duplicated for the same reason
// isPastStuckDeadline is: no value import from src/server/ into this file).
function worstCi(prs: Pr[]): CiState {
  let sawPending = false;
  let sawGreen = false;
  for (const pr of prs) {
    if (pr.ciState === "failing") return "failing";
    if (pr.ciState === "pending") sawPending = true;
    else if (pr.ciState === "green") sawGreen = true;
  }
  if (sawPending) return "pending";
  if (sawGreen) return "green";
  return "none";
}

// Finding 6 (branch review): hub event timestamps are UTC `...Z` while `status.md`'s `updated`
// can carry an offset like `...-03:00` — comparing the raw strings lexicographically (the
// previous `localeCompare`) sorted `10` before `12` regardless of timezone, misordering e.g.
// `2026-08-27T12:00:00.000Z` (13:00 local) before `2026-08-27T10:00:00-03:00` (13:00 UTC, same
// instant) vs an actually-later one. Parsed epoch ms compares real instants; an invalid/missing
// timestamp sorts to +Infinity (last, not first) so it never masquerades as "oldest".
function sortEpochMs(c: MissionCard): number {
  const ms = Date.parse(c.lastEventAgo ?? c.updated ?? "");
  return Number.isNaN(ms) ? Infinity : ms;
}

// Item 12 (round 3): the footer's single "most relevant" PR — newest updatedAt wins, ties
// broken by the higher number; if NO pr in the list has a finite updatedAt, the highest
// number wins instead (never null unless the list itself is empty).
export function newestPr(prs: Pr[]): { number: number; ci: CiState } | null {
  if (prs.length === 0) return null;
  const withDates = prs.filter((pr) => Number.isFinite(Date.parse(pr.updatedAt)));
  const pool = withDates.length > 0 ? withDates : prs;
  const byDate = withDates.length > 0;
  const best = pool.reduce((a, b) => {
    if (byDate) {
      const da = Date.parse(a.updatedAt), db = Date.parse(b.updatedAt);
      if (da !== db) return da > db ? a : b;
    }
    return a.number > b.number ? a : b;
  });
  return { number: best.number, ci: best.ciState };
}

// The join, over whatever IS known. `hub`/`prs` being undefined or failed never prevents a
// MissionModel from being built — every card still exists, with "unknown" headline/PR fields
// where a source hasn't answered yet. Splitting this out from `buildMission` below keeps that
// function itself tiny: compute the model, then compute readiness from the SAME already-built
// cards (Finding 4 needs to see each card's resolved `pr.status` to decide whether `prs` still
// counts as pending).
function buildModel(
  projects: ProjectScan,
  projectsReady: boolean,
  hub: Envelope<HubEnvelopeData> | undefined,
  prs: Envelope<Record<string, PrResult>> | undefined,
  githubEnabled: boolean,
  nowMs: number,
  worktrees?: Envelope<Record<string, WorktreeResult>> | undefined,
): MissionModel {
  const hubReady = hub !== undefined && hub.ok === true;
  const hubMap: Record<string, HubProject> = hub && hub.ok ? hub.data.byProject : {};
  const historicalTruncated = hub && hub.ok ? hub.data.historicalTruncated : false;
  // Still loading (hub undefined) is not a failure: no "unavailable" warning flashes before the first fetch lands.
  const hubMapCodexSource: "ok" | "failed" | "truncated" | "absent" = hub === undefined ? "ok" : hub.ok ? hub.data.codexSource : "failed";
  const hubMapTmuxSource: "ok" | "failed" = hub === undefined ? "ok" : hub.ok ? hub.data.tmuxSource : "failed";
  const cardedDirs = new Set(projects.projects.map((p) => p.dir));

  // Finding 3 (cold review round 3): spec §4.1's needs-you condition includes "any pane's last
  // capsule is blocked or needs_input" as its OWN trigger, independent of a pending question. The
  // inbox used to be built only from gate + question + merge-ready-PR rows, so a pane that was
  // needs-you purely by capsule produced a needs-you CARD with nothing in the strip. Collected
  // alongside pendingQuestions in the same per-pane loop below, and skipped when the pane already
  // has a pending question (that row already covers it — no double-count).
  const attentionRows: InboxRow[] = [];

  const cards: MissionCard[] = projects.projects.map((project) => {
    const hp = hubMap[project.dir];
    const panes = hp?.panes ?? [];
    const codexSessions = hp?.codexSessions ?? [];

    const headline: MissionState = hubReady ? headlineFor(project.gate, panes, codexSessions, nowMs) : "unknown";

    const sessionOrder: string[] = [];
    const sessionsByName = new Map<string, MissionPane[]>();
    const pendingQuestions: MissionPendingQuestion[] = [];
    for (const p of panes) {
      const pendingQuestion = p.pendingQuestion ? toMissionPendingQuestion(p.pendingQuestion) : null;
      // MOA-469 live correction: actionable inbox/question rows come only from a LIVE pane — a dead
      // pane's stale capsule/question stays on its session display row below, never on the inbox.
      if (p.live) {
        if (pendingQuestion) {
          pendingQuestions.push(pendingQuestion);
        } else if (p.capsule?.status === "blocked" || p.capsule?.status === "needs_input") {
          attentionRows.push({ kind: "attention", dir: project.dir, name: project.name, status: p.capsule.status, transport: "tmux", pane: p.pane, tmuxIncarnation: p.tmuxIncarnation });
        }
      }
      // `session`/`role` are null when workflow_sessions has no row yet for this pane — fall back
      // to the pane id as its DISPLAY label and "adhoc" as the neutral role, rather than dropping
      // the pane or widening MissionPane's types for a transient gap. Finding 2 (branch review):
      // the GROUPING key used to be the same bare pane id, which collapsed two incarnations of the
      // same pane (e.g. two `%1` rows with different `tmuxIncarnation`, both session-less) into one
      // synthetic session bucket. `paneKey` (tuple-safe, `src/lib/workflow.ts`) keeps them apart
      // internally; the display label stays the bare pane id/session name either way.
      const displaySession = p.session ?? p.pane;
      const groupKey = p.session ?? paneKey(p.pane, p.tmuxIncarnation);
      const mp: MissionPane = {
        pane: p.pane, tmuxIncarnation: p.tmuxIncarnation, session: displaySession, role: p.role ?? "adhoc",
        state: paneState(p, nowMs), lastEventTs: p.lastEventTs, pendingQuestion,
        subagentCount: p.subagentCount,
        capsuleEventId: p.capsule?.eventId ?? null,
        capsuleStatus: capsuleStatusOf(p.capsule),
        capsuleMinutes: p.capsule?.minutes ?? null,
        capsuleDeclaredAt: p.capsule?.declaredAt ?? null,
        capsuleMergeAsk: p.capsule?.mergeAsk ?? null,
        capsuleMergeBranch: p.capsule?.mergeBranch, capsuleMergeTarget: p.capsule?.mergeTarget,
        capsuleQuestion: p.capsule?.question ?? null,
        capsuleAnswerable: p.capsule?.answerable ?? true,
        runtime: p.runtime,
        live: p.live,
        tmuxSession: p.session,
      };
      if (!sessionsByName.has(groupKey)) sessionOrder.push(groupKey);
      const list = sessionsByName.get(groupKey) ?? [];
      list.push(mp);
      sessionsByName.set(groupKey, list);
    }
    const sessions: MissionSession[] = [
      ...sessionOrder.map((key) => {
        const groupPanes = sessionsByName.get(key)!;
        return { transport: "tmux" as const, session: groupPanes[0].session, panes: groupPanes };
      }),
      ...codexSessions.map((s) => {
        const state = codexSessionState(s, nowMs);
        if (state === "needs-you") {
          attentionRows.push({ kind: "attention", dir: project.dir, name: project.name, status: "needs_input", transport: "codex", threadId: s.threadId });
        }
        return { transport: "codex" as const, threadId: s.threadId, state, lastEventTs: s.lastEventTs, subagentCount: s.subagentCount, capsuleEventId: s.capsule?.eventId ?? null,
          capsuleStatus: capsuleStatusOf(s.capsule), capsuleMinutes: s.capsule?.minutes ?? null, capsuleDeclaredAt: s.capsule?.declaredAt ?? null,
          capsuleMergeAsk: s.capsule?.mergeAsk ?? null, capsuleQuestion: s.capsule?.question ?? null,
          capsuleMergeBranch: s.capsule?.mergeBranch, capsuleMergeTarget: s.capsule?.mergeTarget,
          capsuleAnswerable: s.capsule?.answerable ?? true };
      }),
    ];

    // freshnessCount/newestEventTs are already computed server-side (Task 4), already exclude
    // emitter='jaxos', and are NOT capped the way `timeline` is — reading them directly here
    // (instead of re-deriving from a 20-row-capped event list) is what keeps the freshness badge
    // correct past 20 events.
    const freshnessCount = hubReady ? (hp?.freshnessCount ?? 0) : null;
    const freshnessWarning = freshnessCount !== null && freshnessCount >= FRESHNESS_WARNING_THRESHOLD;
    const lastEventAgo = hubReady ? (hp?.newestEventTs ?? null) : null;
    const heartbeat = hubReady ? (hp?.heartbeat ?? new Array(60).fill(0)) : new Array(60).fill(0);
    const lastAction = hubReady ? (hp?.lastAction ?? null) : null;
    // Optional-chained: the envelope is external input — a pre-Part-1/malformed payload missing
    // `prefs`/`settings` must fall back to the documented defaults, never crash the board
    // (AGENTS.md hard rule 3; an out-of-whitelist Topbar test fixture still carries the old shape).
    const prefs = hub && hub.ok ? (hub.data.prefs?.[project.dir] ?? { pinned: false, hiddenAt: null }) : { pinned: false, hiddenAt: null };
    // Filtering rule: lastActivity counts AGENT activity only. hp.newestEventTs already excludes
    // emitter='jaxos' (computed upstream, Part 1, workflows.ts:1387-1421, SQL `... AND emitter !=
    // 'jaxos'`) — every synthetic jaxos-only event kind (e.g. question-answered, EVENT_MATRIX-
    // restricted to emitters: ["jaxos"]) is excluded there, not here. This block treats
    // hp.newestEventTs as an opaque, already-filtered scalar and must never re-derive a max
    // timestamp by scanning hp.timeline (which filters nothing, workflows.ts:728, and so still
    // carries jaxos-emitted rows for display).
    const lastActivityCandidates = [hp?.newestEventTs, project.updated, project.statusMtime]
      .map((v) => (v ? Date.parse(v) : NaN)).filter((n) => Number.isFinite(n));
    const lastActivityMs = lastActivityCandidates.length > 0 ? Math.max(...lastActivityCandidates) : null;
    const archiveAfterDays = hub && hub.ok ? (hub.data.settings?.archiveAfterDays ?? 14) : 14;
    // Order pinned > archived > hidden > autoHidden > visible — the spec names pinned as the one
    // override; the other three are mutually protective in practice and no acceptance criterion
    // exercises a tie, so this order is a deliberate, documented pick.
    const visibility: MissionCard["visibility"] = prefs.pinned
      ? "visible"
      : project.archived && !(freshnessCount !== null && freshnessCount > 0)
        ? "archived"
        : prefs.hiddenAt !== null && (lastActivityMs === null || lastActivityMs <= Date.parse(prefs.hiddenAt))
          ? "hidden"
          : lastActivityMs !== null && nowMs - lastActivityMs >= archiveAfterDays * 24 * 60 * 60 * 1000
            ? "autoHidden"
            : "visible";
    const timeline: MissionTimelineEntry[] = hubReady
      ? (hp?.timeline ?? []).map((e) => ({ ts: e.ts, type: e.type, pane: e.pane, role: e.role, capsuleStatus: timelineCapsuleStatus(e), ...timelineRunFinishedFields(e) }))
      : [];

    let pr: MissionCardPr;
    // github off (settings): a terminal "disabled" on every card, never "unknown" — the empty
    // `prs` map is the correct, final, resolved answer, not a still-in-flight poll (buildMission
    // below only keeps "prs" pending on "unknown").
    if (!githubEnabled) pr = { status: "disabled" };
    else if (prs === undefined) pr = { status: "loading" };
    else if (!prs.ok) pr = { status: "error", error: prs.error };
    else {
      const result = prs.data[project.dir];
      // Finding 4 (cold review round 3, closed at the source in round 4): a missing map entry used
      // to be genuinely ambiguous between "no GitHub remote" and "not in this snapshot yet". Task
      // 5's collector now resolves a no-remote project to an EXPLICIT entry ({ok:true, prs:[],
      // truncated:false} — handled by the `else` branch below, same as a real zero-open-PRs
      // remote), so a missing entry can mean only one thing: this project is newer than the prs
      // poll's own 60s-old getProjects() snapshot, and nobody has checked yet (mission/projects
      // polls at 10s, mission/prs at 60s, and the two routes call getProjects() independently).
      // `unknown` is never merge-ready, and buildMission below keeps "prs" pending whenever any
      // card is unknown, so a real merge-ready PR nobody has fetched yet can never be silently
      // reported as "nothing waits on you".
      if (!result) pr = { status: "unknown" };
      else if (!result.ok) pr = { status: "error", error: result.error };
      else pr = { status: "ok", count: result.prs.length, ci: worstCi(result.prs), truncated: result.truncated, newest: newestPr(result.prs) };
    }

    let retainedWorktrees: number | null;
    if (worktrees === undefined || !worktrees.ok) retainedWorktrees = null;
    else {
      const wr = worktrees.data[project.dir];
      retainedWorktrees = wr && wr.ok && wr.count > 0 ? wr.count : null;
    }

    return {
      dir: project.dir, name: project.name, stage: project.stage, branch: project.branch,
      builder: project.builder, flag: project.flag, gate: project.gate, legacyGateField: project.legacyGateField,
      now: project.now, residuals: project.residuals, updated: project.updated,
      headline, hubReady, lastEventAgo, freshnessCount, freshnessWarning, sessions, pendingQuestions, timeline, pr,
      activeRun: hubReady ? toMissionActiveRun(hp?.activeRun) : null,
      activeRuns: hubReady ? (hp?.activeRuns ?? []).map(toMissionActiveRunSummary) : [],
      lastRun: hubReady ? toMissionLastRun(hp?.lastRun) : null,
      retainedWorktrees, loopSummary: hubReady ? (hp?.loopSummary ?? null) : null,
      codexSource: hubMapCodexSource,
      codexUntracked: hubReady ? (hp?.codexUntracked ?? false) : false,
      heartbeat, lastAction, prefs, visibility,
    };
  });

  cards.sort((a, b) => {
    const rankDiff = STATE_RANK[b.headline] - STATE_RANK[a.headline];
    if (rankDiff !== 0) return rankDiff;
    const tsDiff = sortEpochMs(a) - sortEpochMs(b); // ascending = oldest first
    if (tsDiff !== 0) return tsDiff;
    return a.dir.localeCompare(b.dir);
  });

  // Post-merge cold review, Finding 3 (MEDIUM): `cardedDirs` is built from `projects.projects`,
  // which is `EMPTY_SCAN` (empty array) whenever the raw `projects` envelope hasn't resolved
  // `{ok:true}` yet — pending and a real "no projects" scan are indistinguishable at this point.
  // Gating on `hubReady` alone therefore listed every hub project as "not on this board" while
  // `projects` was still loading or had failed, next to the column's own skeleton or source
  // warning. Gated on `projectsReady` too so the remainder is only ever computed once there is a
  // real card set to compare against.
  const uncardedProjectNames =
    projectsReady && hubReady ? Object.keys(hubMap).filter((dir) => !cardedDirs.has(dir)).sort() : [];

  // MOA-486: only visible cards feed the strip — hiding a card hides its rows too. Nothing new is
  // lost: any activity after `hiddenAt` already flips the card back to "visible" (ladder above).
  const inboxCards = cards.filter((c) => c.visibility === "visible");
  const inboxDirs = new Set(inboxCards.map((c) => c.dir));
  const visibleAttentionRows = attentionRows.filter((r) => inboxDirs.has(r.dir));

  const gateRows: InboxRow[] = inboxCards
    .filter((c) => c.gate !== null && c.headline === "needs-you") // same rule as headlineFor: a working or waiting tech lead owes nothing yet
    .map((c) => ({ kind: "gate" as const, dir: c.dir, name: c.name, gate: c.gate as "awaiting-approval" | "blocked", updated: c.updated }));

  const questionRows: InboxRow[] = inboxCards.flatMap((c) =>
    c.pendingQuestions.map((q) => ({ kind: "question" as const, dir: c.dir, name: c.name, pane: q.pane, eventId: q.eventId, question: q.questions[0]?.question ?? "" })),
  );

  const mergeReady: { dir: string; name: string; pr: Pr }[] = [];
  if (prs !== undefined && prs.ok) {
    for (const card of inboxCards) {
      const result = prs.data[card.dir];
      if (!result || !result.ok) continue;
      for (const pr of result.prs) {
        if (pr.mergeReady) mergeReady.push({ dir: card.dir, name: card.name, pr });
      }
    }
  }
  mergeReady.sort((a, b) => a.dir.localeCompare(b.dir) || a.pr.number - b.pr.number);
  const prRows: InboxRow[] = mergeReady
    .slice(0, INBOX_PR_CAP)
    .map((x) => ({ kind: "pr" as const, dir: x.dir, name: x.name, number: x.pr.number, url: x.pr.url, ci: x.pr.ciState }));
  const inboxPrRemainder = Math.max(0, mergeReady.length - INBOX_PR_CAP);
  const prTruncatedAny = cards.some((c) => c.pr.status === "ok" && c.pr.truncated);
  const hiddenCount = cards.filter((c) => c.visibility === "hidden").length;
  const autoHiddenCount = cards.filter((c) => c.visibility === "autoHidden").length;
  const archiveAfterDaysOut = hub && hub.ok ? (hub.data.settings?.archiveAfterDays ?? 14) : 14;

  return {
    cards,
    uncardedProjectNames,
    historicalTruncated,
    skipped: projects.skipped,
    inbox: [...gateRows, ...questionRows, ...visibleAttentionRows, ...prRows],
    inboxTotalCount: gateRows.length + questionRows.length + visibleAttentionRows.length + mergeReady.length,
    inboxPrRemainder,
    prTruncatedAny,
    codexSource: hubMapCodexSource,
    tmuxSource: hubMapTmuxSource,
    archiveAfterDays: archiveAfterDaysOut,
    hiddenCount,
    autoHiddenCount,
    reposRoot: projects.reposRoot,
  };
}

// The ONE page-level join (cold review round 3, Simplification 1 — recurrence ledger R3,
// threshold crossed a third time). `ProjectsColumn` and `InboxStrip` (Task 8/9) both consume this
// same `Mission`: `model` is always a real, usable `MissionModel`; `readiness` is a separate,
// small statement of confidence, never a gate on whether `model` exists.
export function buildMission(
  projects: Envelope<ProjectScan> | undefined,
  hub: Envelope<HubEnvelopeData> | undefined,
  prs: Envelope<Record<string, PrResult>> | undefined,
  nowMs: number,
  worktrees?: Envelope<Record<string, WorktreeResult>> | undefined,
  githubEnabled = true,
): Mission {
  const projectsReady = projects !== undefined && projects.ok;
  const projectsData = projects !== undefined && projects.ok ? projects.data : EMPTY_SCAN;
  const model = buildModel(projectsData, projectsReady, hub, prs, githubEnabled, nowMs, worktrees);

  // Backbone failure only (spec §7.1's table: projects/hub get a page-level warning, prs does
  // not) — checked, and reported in full, before pending, so a failed source never reads as
  // "still loading" (hard rule 3). `prs` is deliberately never pushed here (Finding 5): its
  // failure is scoped to each card's own `pr` field above, never promoted to a page-level
  // failure that would hide gate/question/attention rows that ARE fully known.
  const failed: SourceName[] = [];
  if (projects !== undefined && !projects.ok) failed.push("projects");
  if (hub !== undefined && !hub.ok) failed.push("hub");
  if (failed.length > 0) {
    // Post-merge cold review, Finding 4 (MEDIUM): a failed source used to be reported alone, with
    // no way to tell whether the OTHER backbone source (`projects`) has resolved yet at all. When
    // it hasn't — e.g. hub failed while projects is still in flight — `ProjectsColumn` (Task 9)
    // needs that distinction to keep its skeleton up instead of falling through to a false empty
    // state. `pending` here lists every still-undefined source, same definition as the `loading`
    // branch below, just not yet exclusive of a failure existing alongside it.
    const stillPending: SourceName[] = [];
    if (projects === undefined) stillPending.push("projects");
    if (hub === undefined) stillPending.push("hub");
    if (prs === undefined) stillPending.push("prs");
    return { readiness: { kind: "failed", failed, pending: stillPending }, model };
  }

  const pending: SourceName[] = [];
  if (projects === undefined) pending.push("projects");
  if (hub === undefined) pending.push("hub");
  if (prs === undefined) pending.push("prs");
  // Finding 4: `prs` can be {ok:true} yet still incomplete for the CURRENT project set — a
  // project newer than its snapshot has no map entry and reads as "unknown" on the card (above),
  // not a confident "no remote". Treat that the same as `prs` still being in flight, so readiness
  // can never call itself ready while a real PR might be sitting there unfetched.
  if (prs !== undefined && prs.ok && !pending.includes("prs") && model.cards.some((c) => c.pr.status === "unknown")) {
    pending.push("prs");
  }
  if (pending.length > 0) return { readiness: { kind: "loading", pending }, model };

  // Cold review round 4, Finding 2 (R3's fifth occurrence of this recurrence class): a failed
  // `prs` envelope is neither `failed` (Finding 5 — a PR outage is not a backbone failure) nor
  // `pending` (it already resolved, just not successfully) — left uncounted here, it fell all the
  // way through to `{kind:"ready"}` while GitHub was down, and an empty inbox then rendered the
  // confident "NOTHING WAITS ON YOU" title. `degraded` names that state instead of stretching
  // `ready` to cover it: known gate/question/attention rows still render (InboxStrip, Task 8) —
  // a PR outage must never hide rows that are fully known — but the confident empty title is
  // reachable only from `ready`.
  //
  // Post-merge cold review, Finding 1 (HIGH): the envelope itself can resolve `{ok:true}` while
  // one project's OWN `PrResult` is `{ok:false}` (each project's PR fetch is independent —
  // `github.ts`'s `getPrsForProjects` returns a per-project `PrResult`, not one outcome for the
  // whole poll). That per-card failure already renders as `pr.status === "error"` on the one card
  // (above), but until now nothing promoted it to `degraded` — a per-project GitHub failure with
  // an otherwise-empty inbox reached the same confident `titleEmpty` this whole branch exists to
  // prevent for a whole-envelope failure. Treated the same way: `prs.ok` alone is not enough to
  // rule out `degraded` — a card-level error counts too. No page-level warning is added for this
  // case (the per-card error is already visible on that one card); this only ever narrows
  // `readiness` away from a confident `ready`.
  const degraded: SourceName[] = [];
  if (prs !== undefined && !prs.ok) degraded.push("prs");
  else if (prs !== undefined && prs.ok && model.cards.some((c) => c.pr.status === "error")) degraded.push("prs");
  if (degraded.length > 0) return { readiness: { kind: "degraded", degraded }, model };

  return { readiness: { kind: "ready" }, model };
}

export type StripTitle = { kind: "empty" } | { kind: "count"; count: number } | { kind: "unknown" };
export type StripWarning = { kind: "none" } | { kind: "skeleton" } | { kind: "source"; sources: SourceName[] };
export type StripPresentation = { title: StripTitle; warning: StripWarning };

// Finding 8 (branch review, LOW): the title/warning branch was inline JSX in InboxStrip, so a
// mistake there could only be caught by rendering the component — the repo has no react-testing
// infra and adding one is out of scope for a LOW finding. Pulled the decision into this pure
// function instead so the adversarial readiness/model combinations (including Finding 3's
// zero-card degraded case below) are tested directly, the same way buildMission's own branches
// are.
export function stripPresentation(readiness: MissionReadiness, model: MissionModel): StripPresentation {
  const title: StripTitle =
    readiness.kind === "ready"
      ? model.inboxTotalCount === 0
        ? { kind: "empty" }
        : { kind: "count", count: model.inboxTotalCount }
      : { kind: "unknown" };

  // Finding 3 (branch review, MEDIUM): a degraded `prs` source has nowhere to show its inline
  // warning (a card's own `pr.status === "error"` line) when there are zero cards at all — the
  // strip used to render only the neutral `titleUnknown` with no visible warning anywhere on the
  // page. `model.cards.length === 0` is the exact case where no card can carry that inline
  // warning: whenever at least one card exists, a `degraded` "prs" reading always means either the
  // whole envelope failed (every card reads `pr.status === "error"`) or one project's own PR fetch
  // failed (that one card reads it) — degraded is never reachable with cards present and none of
  // them showing the error inline.
  const warning: StripWarning =
    readiness.kind === "failed"
      ? { kind: "source", sources: readiness.failed }
      : readiness.kind === "degraded" && model.cards.length === 0
        ? { kind: "source", sources: readiness.degraded }
        : readiness.kind === "loading" && model.inbox.length === 0
          ? { kind: "skeleton" }
          : { kind: "none" };

  return { title, warning };
}

// The four-pill fold (spec §7/§8): MissionState's six values fold into four buckets.
// Exported so the count strip, topbar pill, and click-filter share ONE fold.
export type MissionBucket = "needs-you" | "stuck" | "working" | "idle";
export type MissionCounts = Record<MissionBucket, number>;
export type MissionBucketSummary = { counts: MissionCounts; worst: MissionBucket } | { counts: null; worst: null };

const BUCKET_RANK: readonly MissionBucket[] = ["needs-you", "stuck", "working", "idle"];

export function bucketOf(headline: MissionState): MissionBucket | null {
  if (headline === "waiting") return "working";
  if (headline === "unknown") return null;
  return headline;
}

// Gated on readiness (mirrors stripPresentation's gate, spec §7): loading/failed must
// never render a confident "0 working" -- caller shows a skeleton when null. `degraded`
// (prs-only) computes normally; a PR outage never affects card headlines.
export function summarizeBuckets(readiness: MissionReadiness, cards: MissionCard[]): MissionBucketSummary {
  if (readiness.kind === "loading" || readiness.kind === "failed") return { counts: null, worst: null };
  const counts: MissionCounts = { "needs-you": 0, stuck: 0, working: 0, idle: 0 };
  for (const card of cards) {
    // A hidden card is off the board, so it is off the pills too: a hidden project whose
    // status.md still carries `gate: awaiting-approval` counted as a phantom "needs you" that no
    // visible card explained (2026-09-21, route-converter-se). Same rule as the inbox strip.
    if (card.visibility !== "visible") continue;
    const bucket = bucketOf(card.headline);
    if (bucket) counts[bucket] += 1;
  }
  let worst: MissionBucket = "idle";
  for (const bucket of BUCKET_RANK) {
    if (counts[bucket] > 0) {
      worst = bucket;
      break;
    }
  }
  return { counts, worst };
}

// round-1 F7: the click-to-filter toggle (Decision 3). Lives here rather than in
// `page.tsx` because the App Router rejects non-page named exports from a page module
// ("toggleFilter is not a valid Page export field" — Next build type error).
export function toggleFilter(current: MissionBucket | null, clicked: MissionBucket): MissionBucket | null {
  return current === clicked ? null : clicked;
}

// Client-safe mirror of the pipeline vocabulary (spec §8). NEVER import
// projects.ts's own unexported STAGES constant — it imports node:fs and would break
// the client bundle; only ProjectStage itself is `import type`-ed here.
export const PIPELINE_STAGES: ProjectStage[] = ["spec", "build", "review", "test", "ship"];

export function stageDotIndex(stage: ProjectStage): number {
  return PIPELINE_STAGES.indexOf(stage); // -1 for an unrecognized value — no dot lit, never a crash
}

// ---- Phase 2 (Mission Control B): pure card helpers, spec §6 / §9 ---------------------------

// Merge question contract: the "Approve merge" button is lit by the hook's deterministic
// merge-question alone (merge_ask exactly 1 AND a branch). Old rows carry a Jev fraction or no
// branch and stay ineligible. `typeof` (not `!== null`): the optional fields arrive as undefined.
export function isMergeAskEligible(mergeAsk: number | null, mergeBranch?: string | null): boolean {
  return mergeAsk === 1 && typeof mergeBranch === "string";
}

// parseIndexedReply's grammar (src/lib/workflow.ts:156-187): one `<n>: <i>` or `<n>: <i,j>` line per
// question, in question order — the ONLY reply shape claimAnswer accepts (round-2 F1).
export function composeIndexedReply(pq: MissionPendingQuestion, selections: number[][]): string {
  return pq.questions.map((_, i) => `${i + 1}: ${[...(selections[i] ?? [])].sort((a, b) => a - b).join(",")}`).join("\n");
}

export type PendingAction =
  | { kind: "question"; pendingQuestion: MissionPendingQuestion }
  | {
      kind: "freeform"; pane: string; eventId: number; capsuleStatus: "needs_input" | "blocked" | null;
      transport: "tmux" | "codex"; role: "lead" | "adhoc" | null; mergeAsk: number | null; question: string | null;
      answerable: boolean; mergeBranch?: string; mergeTarget?: string;
    };

// Round-3 F4: ONE pending interaction per card. A structured question first — card.pendingQuestions[0]
// is the first LIVE pane, in pane order, with an open question (buildModel's per-pane loop appends and
// never reorders); otherwise the first needs-you session (tmux panes, in session/pane order, then native
// Codex sessions — the order card.sessions is already built in) whose capsule has an event id. Every
// other session's question/capsule is reachable only through its own link this phase.
export function selectPendingAction(card: MissionCard, eventId?: number): PendingAction | null {
  // F2: an explicit eventId (a specific inbox row) targets that pendingQuestion directly, instead
  // of always defaulting to pendingQuestions[0] — otherwise every row for the same card would answer
  // the first row's event. Omitted eventId keeps the original default-single-action behaviour.
  const pq = eventId === undefined ? card.pendingQuestions[0] : card.pendingQuestions.find((q) => q.eventId === eventId);
  if (pq) return { kind: "question", pendingQuestion: pq };
  if (eventId !== undefined) return null;
  for (const s of card.sessions) {
    if (s.transport === "tmux") {
      for (const p of s.panes) {
        if (p.state === "needs-you" && p.capsuleEventId !== null) {
          return {
            kind: "freeform", pane: p.pane, eventId: p.capsuleEventId, capsuleStatus: p.capsuleStatus,
            transport: "tmux", role: p.role, mergeAsk: p.capsuleMergeAsk, question: p.capsuleQuestion,
            answerable: p.capsuleAnswerable ?? true,
            mergeBranch: p.capsuleMergeBranch, mergeTarget: p.capsuleMergeTarget,
          };
        }
      }
    } else if (s.state === "needs-you" && s.capsuleEventId !== null) {
      return {
        kind: "freeform", pane: s.threadId, eventId: s.capsuleEventId, capsuleStatus: s.capsuleStatus,
        transport: "codex", role: null, mergeAsk: s.capsuleMergeAsk ?? null, question: s.capsuleQuestion ?? null,
        answerable: s.capsuleAnswerable ?? true,
        mergeBranch: s.capsuleMergeBranch, mergeTarget: s.capsuleMergeTarget,
      };
    }
  }
  return null;
}

// Spec §6's opening predicate (round-2 F3): the ONE place "is there a real question to
// answer" is decided, used identically by the card face, the strip, and both their tests.
export function hasQuestionText(card: Pick<MissionCard, "pendingQuestions">): boolean {
  const q = card.pendingQuestions[0]?.questions[0]?.question;
  return typeof q === "string" && q.trim().length > 0;
}

// ---- Task 9: the one fetch wrapper the card's action buttons use ---------------------------
// Mirrors src/components/kanban/PropertiesSidebar.tsx's write() classification (same envelope
// family, src/lib/api.ts's Envelope<T>, already imported at the top of this file) but keeps
// `data` on success, which every Fase 2 route returns and kanban's write() discards — written
// locally rather than imported, to avoid a mission-to-kanban cross-feature dependency for one
// function. Never throws.
export async function postWorkflowAction<T>(url: string, body: unknown): Promise<Envelope<T>> {
  let res: Response;
  try {
    res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  } catch {
    return { ok: false, error: "network error" };
  }
  let json: unknown;
  try {
    json = await res.json();
  } catch {
    return { ok: false, error: "invalid response" };
  }
  if (json && typeof json === "object" && !Array.isArray(json) && "ok" in json) return json as Envelope<T>;
  return { ok: false, error: "unexpected response" };
}

// merge-button-hide fix: pulls the numeric event id out of an answer POST body (every
// PendingActionButtons click posts `{event_id, reply}`) so the caller can key its optimistic
// "answered" confirmation to the exact question just replied to. A cancel/dispatch/prefs POST
// body has no event_id and yields null, so it never triggers that path.
export function answerEventId(body: unknown): number | null {
  if (!body || typeof body !== "object" || Array.isArray(body)) return null;
  const id = (body as { event_id?: unknown }).event_id;
  return typeof id === "number" ? id : null;
}

// ---- Ticker (spec §9, Decisions 12/13) — a pure client-side merge, no new API route -----------
const TICKER_EVENT_TYPES = new Set(["question", "question-resolved", "attention-needed", "turn-stopped", "run-finished", "merge-approved"]);
// Rafa (2026-09-19): the activity feed grows unbounded — cap it to the last 7 days. 30 stays the
// total retained: at 10 items/page (EventTicker's paging) that's 3 pages, comfortably above what
// a 7-day window fills for this single-user cockpit.
const TICKER_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;
const TICKER_CAP = 30;

export type TickerItem =
  | ({ source: "event"; dir: string; name: string } & MissionTimelineEntry)
  | { source: "commit"; dir: string; name: string; ts: string; author: string; message: string; hash: string }
  | { source: "pr"; dir: string; name: string; ts: string; number: number; url: string; ci: CiState; title: string };

// ---- Ticker sentence (spec §9, Decisions 19/21) — a pure key lookup, EventTicker-only. -------
// TimelineEntryText (ProjectCard.tsx's own expanded timeline) is untouched — it renders raw
// field concatenation for an engineer-facing log; this is the ticker's human sentence.
export type TickerSentence = { key: string; values?: Record<string, string> };

const TICKER_NS = "sentence.";

// F3 (round-1): run-finished's conditions overlap — checked top to bottom, first match wins.
// Round-2 F4: role is now the FIRST thing checked among the outcome-value rows, same precedence
// classifyLastRun already uses for the identical payload fields (mission.ts's own
// classifyLastRun) — a builder outcome can never match a reviewer-only row or vice versa.
export function tickerSentence(entry: Pick<MissionTimelineEntry, "type" | "capsuleStatus" | "outcome" | "contractStatus" | "stage" | "role">): TickerSentence {
  switch (entry.type) {
    case "question": return { key: TICKER_NS + "question" };
    case "question-resolved": return { key: TICKER_NS + "questionResolved" };
    case "attention-needed": return { key: TICKER_NS + "attentionNeeded" };
    case "merge-approved": return { key: TICKER_NS + "mergeApproved" };
    case "turn-stopped":
      if (entry.capsuleStatus === "needs_input") return { key: TICKER_NS + "turnStoppedNeedsInput" };
      if (entry.capsuleStatus === "blocked") return { key: TICKER_NS + "turnStoppedBlocked" };
      return { key: TICKER_NS + "turnStopped" };
    case "run-finished":
      if (entry.stage === "worker") return { key: TICKER_NS + "runFinishedInterrupted" };
      if (entry.contractStatus === "cancelled") return { key: TICKER_NS + "runFinishedCancelled" };
      if (entry.role === "reviewer" && entry.outcome === "approve") return { key: TICKER_NS + "reviewApprove" };
      if (entry.role === "reviewer" && entry.outcome === "approve-with-changes") return { key: TICKER_NS + "reviewApproveChanges" };
      if (entry.role === "reviewer" && entry.outcome === "reject") return { key: TICKER_NS + "reviewReject" };
      if (entry.role === "builder" && entry.outcome === "success") return { key: TICKER_NS + "buildOk" };
      if (entry.role === "builder" && entry.outcome === "blocked") return { key: TICKER_NS + "buildBlocked" };
      if (entry.role === "builder" && entry.outcome !== null) return { key: TICKER_NS + "buildFailed" };
      return { key: TICKER_NS + "runFinishedNoResult" };
    default: return { key: TICKER_NS + "runFinishedNoResult" };
  }
}

// Native Intl, same rung RelativeTime.tsx already uses for its own formatting — a mission-
// control operations feed reads better as a fixed clock than a locale-relative one.
// Round-2 F1: deliberately HOST-LOCAL time (no `timeZone` option) — the cockpit is
// single-user, host TZ America/Sao_Paulo (AGENTS.md) — never UTC.
export function formatClock(epochMs: number, locale: string): string {
  return new Intl.DateTimeFormat(locale, { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false }).format(epochMs);
}

export function buildTicker(
  cards: Pick<MissionCard, "dir" | "name" | "timeline">[],
  commits: Commit[],
  prsByProject: Record<string, PrResult>,
  nowMs: number = Date.now(),
): TickerItem[] {
  const items: TickerItem[] = [];
  for (const c of cards) {
    for (const e of c.timeline) {
      if (TICKER_EVENT_TYPES.has(e.type)) items.push({ source: "event", dir: c.dir, name: c.name, ...e });
    }
  }
  for (const commit of commits) {
    items.push({ source: "commit", dir: commit.repo, name: commit.repo, ts: new Date(commit.epoch * 1000).toISOString(),
      author: commit.author, message: commit.message, hash: commit.hash });
  }
  for (const c of cards) {
    const result = prsByProject[c.dir];
    if (!result || !result.ok) continue;
    for (const pr of result.prs) {
      items.push({ source: "pr", dir: c.dir, name: c.name, ts: pr.updatedAt, number: pr.number, url: pr.url, ci: pr.ciState, title: pr.title });
    }
  }
  // Cutoff applied BEFORE sort/cap: an invalid `ts` (NaN) is dropped, never sorted as "current".
  const cutoffMs = nowMs - TICKER_WINDOW_MS;
  const withinWindow = items.filter((item) => {
    const ms = Date.parse(item.ts);
    return Number.isFinite(ms) && ms >= cutoffMs;
  });
  withinWindow.sort((a, b) => Date.parse(b.ts) - Date.parse(a.ts));
  return withinWindow.slice(0, TICKER_CAP);
}

// ---- Ticker paging (Rafa, 2026-09-19) — pure, unit-tested clamping math shared by EventTicker.
export function pageSlice<T>(items: T[], page: number, size: number): { items: T[]; page: number; pages: number } {
  const pages = Math.max(1, Math.ceil(items.length / size));
  const clamped = Math.min(Math.max(1, Math.trunc(page)), pages);
  return { items: items.slice((clamped - 1) * size, clamped * size), page: clamped, pages };
}

// ---- Round 3, item 3: status pill duration + subagent suffix ------------------------------
export function statusPillSuffix(card: Pick<MissionCard, "activeRun" | "sessions" | "headline">, nowMs: number): { minutes: number | null; subagentCount: number } {
  let subagentCount = 0;
  // T2/F3: track the newest FINITE epoch directly — a session's lastEventTs can be an
  // unparseable string, and comparing against it via Date.parse (NaN) silently blocked every
  // later, valid timestamp from ever winning.
  let newestMs: number | null = null;
  const consider = (ts: string | null) => {
    if (!ts) return;
    const ms = Date.parse(ts);
    if (Number.isFinite(ms) && (newestMs === null || ms > newestMs)) newestMs = ms;
  };
  for (const s of card.sessions) {
    if (s.transport === "tmux") {
      for (const p of s.panes) {
        subagentCount += p.subagentCount;
        consider(p.lastEventTs);
      }
    } else {
      subagentCount += s.subagentCount;
      consider(s.lastEventTs);
    }
  }
  // T2: elapsed minutes only mean anything while a run is actually elapsing — idle/waiting/
  // unknown headlines never show it (subagentCount is unaffected).
  let minutes: number | null = null;
  if (card.headline === "working" || card.headline === "stuck" || card.headline === "needs-you") {
    const runMs = card.activeRun?.startedAt ? Date.parse(card.activeRun.startedAt) : NaN;
    const ms = Number.isFinite(runMs) ? runMs : newestMs;
    minutes = ms !== null && Number.isFinite(ms) ? Math.max(0, Math.floor((nowMs - ms) / 60000)) : null;
  }
  return { minutes, subagentCount };
}

// F2: the WORST pane state across a tmux session's panes, for the compact-face session line —
// deliberately a DIFFERENT order than STATE_RANK (headlineFor's own worst-of), because a source
// outage ("unknown") on one pane must never outrank a genuinely bad state on another.
const PANE_LINE_STATE_RANK: Record<MissionState, number> = {
  "needs-you": 5, stuck: 4, working: 3, waiting: 2, idle: 1, unknown: 0,
};

export function worstPaneState(panes: MissionPane[]): MissionState {
  return panes.reduce((worst, p) => (PANE_LINE_STATE_RANK[p.state] > PANE_LINE_STATE_RANK[worst] ? p.state : worst), "unknown" as MissionState);
}

// ---- Round 3, item 5: elapsed-minutes + the stuck/gate "last" line precedence -------------
export function elapsedMinutesSince(ts: string, nowMs: number): number | null {
  const ms = Date.parse(ts);
  if (!Number.isFinite(ms)) return null;
  return Math.max(0, Math.floor((nowMs - ms) / 60000));
}

// F7 (cold review 92a14d784fb3) fix: an unparseable `ts` used to get latched in as the running
// "newest" the first time it was seen (the old `newest === null` branch didn't check finiteness),
// which then blocked every later, valid timestamp from ever winning (`Date.parse(ts) >
// Date.parse(newest))` is `NaN > NaN` — always false). Track the running max as a NUMBER
// (`newestMs`), seeded at `-Infinity`, and skip any `ts` whose `Date.parse` isn't finite.
function newestSessionEventTs(card: Pick<MissionCard, "sessions">): string | null {
  let newest: string | null = null;
  let newestMs = -Infinity;
  for (const s of card.sessions) {
    const tss = s.transport === "tmux" ? s.panes.map((p) => p.lastEventTs) : [s.lastEventTs];
    for (const ts of tss) {
      if (!ts) continue;
      const ms = Date.parse(ts);
      if (Number.isFinite(ms) && ms > newestMs) {
        newestMs = ms;
        newest = ts;
      }
    }
  }
  return newest;
}

export type LastLineResult = { key: "stuckLine"; declared: number; elapsed: number } | { key: "stuckLineFallback" | "gateWaitingLine"; elapsed: number } | null;

// Rung 1: the FIRST pane/session whose own `state === "stuck"` — only ITS declared-minutes
// capsule counts, never a non-stuck pane's capsule data even when one happens to carry it
// (F7 fix: the original loop scanned every pane's capsule fields regardless of state). The
// capsule is used only when its declaredAt both parses AND is not in the future
// (`Date.parse(declaredAt) <= nowMs`); a future value falls straight to rung 2, it is never
// clamped in place. Rung 2: the newest lastEventTs across the whole card. Rung 3: card.updated.
// Rung 4: null — never a "· há 0 min" fabricated from nothing.
export function stuckLine(card: Pick<MissionCard, "sessions" | "updated">, nowMs: number): LastLineResult {
  let firstStuckPane: { capsuleMinutes: number | null; capsuleDeclaredAt: string | null } | null = null;
  outer: for (const s of card.sessions) {
    const panes = s.transport === "tmux" ? s.panes : [s];
    for (const p of panes) {
      if (p.state === "stuck") {
        firstStuckPane = p;
        break outer;
      }
    }
  }
  if (firstStuckPane && firstStuckPane.capsuleMinutes !== null && firstStuckPane.capsuleDeclaredAt !== null) {
    const declaredMs = Date.parse(firstStuckPane.capsuleDeclaredAt);
    if (Number.isFinite(declaredMs) && declaredMs <= nowMs) {
      const elapsed = elapsedMinutesSince(firstStuckPane.capsuleDeclaredAt, nowMs);
      if (elapsed !== null) return { key: "stuckLine", declared: firstStuckPane.capsuleMinutes, elapsed };
    }
  }
  const sessionTs = newestSessionEventTs(card);
  if (sessionTs) {
    const elapsed = elapsedMinutesSince(sessionTs, nowMs);
    if (elapsed !== null) return { key: "stuckLineFallback", elapsed };
  }
  const elapsed = elapsedMinutesSince(card.updated, nowMs);
  return elapsed !== null ? { key: "stuckLineFallback", elapsed } : null;
}

// One rung: card.updated (the gate is set by status.md). Null when unparseable.
export function gateWaitingLine(card: Pick<MissionCard, "updated">, nowMs: number): LastLineResult {
  // The gate's age is status.md's own `updated` — the same timestamp the inbox gate row shows;
  // session activity is the tech lead working on the approval, not the wait itself.
  const elapsed = elapsedMinutesSince(card.updated, nowMs);
  return elapsed !== null ? { key: "gateWaitingLine", elapsed } : null;
}

// ---- Round 3, item 10: the heartbeat sparkline's SVG path (viewBox "0 0 200 26") ----------
export function heartbeatPath(minutes: number[]): string {
  const max = Math.max(1, ...minutes);
  const points = minutes.map((n, i) => {
    const x = (i * 200) / Math.max(1, minutes.length - 1);
    const y = 26 - (n / max) * 22 - 2;
    return `${x.toFixed(2)} ${y.toFixed(2)}`;
  });
  return points.length === 0 ? "" : `M${points[0]} ${points.slice(1).map((p) => `L${p}`).join(" ")}`;
}

// ---- Round 3, item 11: the last-run findings badge — highest non-zero severity wins -------
export function worstFinding(findings: { high: number; medium: number; low: number } | null): { severity: "high" | "medium" | "low"; count: number } | null {
  if (!findings) return null;
  if (findings.high > 0) return { severity: "high", count: findings.high };
  if (findings.medium > 0) return { severity: "medium", count: findings.medium };
  if (findings.low > 0) return { severity: "low", count: findings.low };
  return null;
}

// ---- Round 3, item 8: the dispatch form's 4-way option → request-body mapping -------------
export type DispatchOption = "reviewSpec" | "reviewPlan" | "reviewDiff" | "build";
export type DispatchFields = {
  command: "review" | "build"; kind: "spec" | "plan" | "diff"; target: string; focus: string;
  plan: string; phase: string; branch: string; whitelist: string; verify: string; build: string; profile: "default" | "fallback";
};
export type DispatchRequestBody =
  | { project: string; command: "review"; kind: "spec" | "plan" | "diff"; target: string; focus?: string }
  | { project: string; command: "build"; plan: string; phase: string; branch: string; whitelist: string; verify: string; build?: string; profile: "default" | "fallback" };

export function dispatchBodyFor(option: DispatchOption, f: DispatchFields, projectDir: string): DispatchRequestBody {
  if (option === "build") {
    return { project: projectDir, command: "build", plan: f.plan, phase: f.phase, branch: f.branch, whitelist: f.whitelist, verify: f.verify, ...(f.build ? { build: f.build } : {}), profile: f.profile };
  }
  const kind = option === "reviewSpec" ? "spec" : option === "reviewPlan" ? "plan" : "diff";
  return { project: projectDir, command: "review", kind, target: f.target, ...(f.focus ? { focus: f.focus } : {}) };
}
