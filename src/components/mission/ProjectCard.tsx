"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { Bot, ChevronDown, ChevronRight, Clock3, File, FolderGit2, GitBranch, GitPullRequest, Hourglass, MoreHorizontal } from "lucide-react";
import { useTranslations } from "next-intl";
import {
  answerEventId, composeIndexedReply, liveSessions, dispatchBodyFor, elapsedMinutesSince, hasQuestionText, isMergeAskEligible, PIPELINE_STAGES, postWorkflowAction, selectPendingAction, stageDotIndex, statusPillSuffix,
  stuckLine, gateWaitingLine, heartbeatPath, worstFinding, worstPaneState,
  type MissionActiveRunSummary, type MissionCard, type MissionPendingQuestion, type MissionState,
} from "@/lib/mission";
import { postPrefsPatch } from "@/lib/prefsAction";
import { useGeneralSettings } from "@/lib/settingsQuery";
import {
  LIMITS, RUN_ID_RE, validOptionalFocus, validPhaseToken, validRefName, validSingleLineCommand,
} from "@/lib/workflow";
import { RelativeTime } from "@/components/RelativeTime";
import { serializeFileSelection } from "@/lib/filesUrl";
import { TimelineEntryText } from "@/components/mission/TimelineEntryText";

const BTN = "rounded-md px-3 py-1.5 text-[12.5px] font-medium disabled:cursor-not-allowed disabled:opacity-50";
const BTN_PRIMARY = `${BTN} border border-brand bg-brand-soft text-brand`;
const BTN_DANGER = `${BTN} border border-danger bg-danger-soft text-danger`;
const BTN_GHOST = `${BTN} border border-line`;

const STATE_STYLES: Record<MissionState, { tag: string; dot: string }> = {
  "needs-you": { tag: "border-danger bg-danger-soft text-danger", dot: "bg-danger" },
  stuck: { tag: "border-warning bg-warning-soft text-warning", dot: "bg-warning" },
  working: { tag: "border-accent-soft-border bg-accent-soft text-accent", dot: "bg-accent" },
  waiting: { tag: "border-info bg-info-soft text-info", dot: "bg-info" },
  idle: { tag: "border-line bg-surface-3 text-muted", dot: "bg-muted" },
  unknown: { tag: "border-line bg-surface-3 text-muted", dot: "bg-muted" },
};

// Decision 9: only `working` animates (comet ring + breathe glow in globals.css); the
// other five are static. `needs-you` adds a soft glow (--danger-soft token). `idle` is
// the neutral border at 60% opacity; `unknown` is the same border at full opacity so a
// source-down card is never mistaken for a confident idle one.
const BORDER_CLASS: Record<MissionState, string> = {
  "needs-you": "border-danger tile-ring state-needs-you",
  stuck: "border-warning tile-ring state-stuck",
  working: "border-line tile-ring state-working",
  waiting: "border-info",
  idle: "border-line/60",
  unknown: "border-line",
};

const LAST_RUN_TONE_CLASS: Record<"success" | "danger" | "warning", string> = {
  success: "text-success", danger: "text-danger", warning: "text-warning",
};

// F4: the severity badge on the last-run line — high/medium get the same danger/warning
// tone pairing every other severity surface in this file uses; low keeps the pre-existing
// muted, borderless-fill look.
const FINDING_SEVERITY_CLASS: Record<"high" | "medium" | "low", string> = {
  high: "border-danger bg-danger-soft text-danger",
  medium: "border-warning bg-warning-soft text-warning",
  low: "border-line text-muted",
};

type PendingUiAction =
  | { kind: "generic"; pendingQuestion: MissionPendingQuestion }
  | { kind: "freeform"; pane: string; eventId: number; answerable: boolean }
  | { kind: "approve"; eventId: number; question: string | null; replyYes: string; replyNo: string; answerable: boolean; branch?: string; target?: string }
  | null;

// One place both the compact row and ExpandedCard's detail block read from, so they always
// agree on which single interaction (spec §9, round-3 F4) is live for this card. `eventId` (F2)
// lets a specific inbox row target its own pendingQuestion instead of always card.pendingQuestions[0].
export function pendingUiAction(
  card: MissionCard,
  eventId?: number,
  t?: (key: string, vars?: Record<string, string | number>) => string,
): PendingUiAction {
  const pending = selectPendingAction(card, eventId);
  if (pending?.kind === "question") {
    // Round-2 F3: an empty structured question renders nothing — hasQuestionText is the ONE place both card and strip (Task 9) check this, so they can't drift.
    if (!hasQuestionText({ pendingQuestions: [pending.pendingQuestion] })) return null;
    return { kind: "generic", pendingQuestion: pending.pendingQuestion };
  }
  // Native-answer-delivery spec D2/D4: both triggers below fire for ANY transport now (native
  // Codex included — the old transport==="tmux" guard is gone). isMergeAskEligible and the
  // role/gate check already only admit a real signal; `answerable` rides along so the card can
  // render the pair DISABLED instead of hiding it outright when delivery is known-impossible.
  if (pending?.kind === "freeform" && pending.capsuleStatus === "needs_input") {
    if (isMergeAskEligible(pending.mergeAsk, pending.mergeBranch)) {
      return {
        kind: "approve", eventId: pending.eventId, question: pending.question,
        replyYes: t?.("actions.replyMergeAsk") ?? "pode", replyNo: t?.("actions.replyDeny") ?? "não", answerable: pending.answerable,
        branch: pending.mergeBranch, target: pending.mergeTarget,
      };
    }
    // Gate fallback only when the classifier gave no score (null); a scored non-merge ask must not get a merge button.
    if (pending.mergeAsk === null && pending.role === "lead" && card.gate === "awaiting-approval" && card.headline === "needs-you") {
      return {
        kind: "approve", eventId: pending.eventId, question: pending.question,
        replyYes: card.branch
          ? (t?.("actions.replyMergeBranch", { branch: card.branch }) ?? `pode fazer o merge de ${card.branch}`)
          : (t?.("actions.replyMerge") ?? "pode fazer o merge"),
        replyNo: t?.("actions.replyDeny") ?? "não", answerable: pending.answerable,
      };
    }
  }
  if (pending?.kind === "freeform") return { kind: "freeform", pane: pending.pane, eventId: pending.eventId, answerable: pending.answerable };
  return null;
}

// merge-button-hide fix: the eventId a rendered "generic"/"approve" action would answer — the
// same key answerEventId (lib/mission.ts) pulls out of that click's own POST body, so a caller
// can tell "the action currently on screen" and "the one I just successfully answered" apart.
// null for freeform (no button here) or no action at all.
export function actionEventId(action: PendingUiAction): number | null {
  if (action?.kind === "generic") return action.pendingQuestion.eventId;
  if (action?.kind === "approve") return action.eventId;
  return null;
}

// Decision 11a: bare-capsule status source — walks the same order selectPendingAction walks (tmux panes then codex sessions, first needs-you wins), covering both the freeform and guarded-empty-question cases with one function. ponytail: simpler than threading capsuleStatus a second time through PendingUiAction's freeform variant for a value this function already resolves.
export function bareCapsuleStatus(card: MissionCard): "needs_input" | "blocked" | null {
  for (const s of card.sessions) {
    if (s.transport === "tmux") {
      for (const p of s.panes) if (p.state === "needs-you") return p.capsuleStatus;
    } else if (s.state === "needs-you") return s.capsuleStatus;
  }
  return null;
}

// Decision 9: the merge/generic/approve button set, shared by ActionRow (compact face) and InboxStrip's question/gate rows (Task 9) — one source of truth. Freeform renders nothing here (Decision 12).
export function PendingActionButtons({
  action, busy, answered = false, onExpand, onAction,
}: { action: PendingUiAction; busy: boolean; answered?: boolean; onExpand: () => void; onAction: (url: string, body: unknown) => void }) {
  const t = useTranslations("mission");
  // merge-button-hide fix: a successful POST for THIS action's eventId hides the buttons right
  // away instead of waiting for the next hub poll to prove it — the caller sets `answered` from
  // its own optimistic local state, keyed to that eventId (round-trips through actionEventId).
  if (answered) return <span className="text-[11.5px] text-muted">{t("actions.answerSent")}</span>;
  if (action?.kind === "generic") {
    const single = action.pendingQuestion.questions.length === 1 && !action.pendingQuestion.questions[0].multiSelect;
    if (single) {
      return (
        <>
          {action.pendingQuestion.questions[0].options.map((opt, i) => (
            <button key={opt.label} type="button" className={BTN_GHOST} disabled={busy}
              onClick={() => onAction("/api/workflow/answer", { event_id: action.pendingQuestion.eventId, reply: composeIndexedReply(action.pendingQuestion, [[i + 1]]) })}>
              {opt.label}
            </button>
          ))}
        </>
      );
    }
    return <button type="button" className={BTN_GHOST} disabled={busy} onClick={onExpand}>{t("actions.respond")}</button>;
  }
  if (action?.kind === "approve") {
    return (
      <>
        <button key="approve" type="button" className={BTN_PRIMARY} disabled={busy || !action.answerable}
          onClick={() => onAction("/api/workflow/answer", { event_id: action.eventId, reply: action.replyYes })}>
          {t("actions.approveMerge")}
        </button>
        <button key="deny" type="button" className={BTN_GHOST} disabled={busy || !action.answerable}
          onClick={() => onAction("/api/workflow/answer", { event_id: action.eventId, reply: action.replyNo })}>
          {t("actions.deny")}
        </button>
      </>
    );
  }
  return null;
}

// Item 2: the question text itself, shown above the button row for merge/generic/approve actions.
function QuestionLine({ action }: { action: PendingUiAction }) {
  const t = useTranslations("mission");
  if (action?.kind === "approve") {
    return (
      <>
        {action.branch && action.target ? <span className="block truncate font-mono text-[11px] text-muted">{t("actions.mergeTarget", { branch: action.branch, target: action.target })}</span> : null}
        {action.question ? <span className="block truncate font-mono text-[11px] text-muted">? &quot;{action.question}&quot;</span> : null}
        {!action.answerable ? <span className="block text-[11px] text-danger">{t("actions.answerUnreachable")}</span> : null}
      </>
    );
  }
  if (action?.kind !== "generic") return null;
  const question = action.pendingQuestion.questions[0]?.question;
  if (!question) return null;
  return <span className="block truncate font-mono text-[11px] text-muted">? &quot;{question}&quot;</span>;
}

function ActionRow({
  card, action, busy, answered, expanded, onExpand, onAction,
}: {
  card: MissionCard; action: PendingUiAction; busy: boolean; answered: boolean; expanded: boolean;
  onExpand: () => void; onAction: (url: string, body: unknown) => void;
}) {
  const t = useTranslations("mission");
  const showCancel = (card.headline === "working" || card.headline === "stuck") && !!card.activeRun;
  const [confirmingCancel, setConfirmingCancel] = useState(false);
  const isActionable = action?.kind === "generic" || action?.kind === "approve";
  // F3: the bare-capsule fallback needs an actual capsule (a freeform pane/session with capsuleStatus)
  // or a structured pending question to describe — a blocked/awaiting gate with neither is not a
  // capsule at all, and used to get this same misleading needs-you label + tmux link regardless.
  const status = card.headline === "needs-you" && !isActionable ? bareCapsuleStatus(card) : null;
  const bareCapsule = card.headline === "needs-you" && !isActionable && (status !== null || card.pendingQuestions.length > 0);
  if (!isActionable && !showCancel && !bareCapsule) return null;
  return (
    <div className="flex flex-col gap-1">
      <QuestionLine action={action} />
      <div className="flex flex-wrap items-center gap-1.5">
        {isActionable ? <PendingActionButtons action={action} busy={busy} answered={answered} onExpand={onExpand} onAction={onAction} /> : null}
        {bareCapsule ? (
          <span className="flex items-center gap-1.5 text-[11.5px] text-muted">
            {t(status === "blocked" ? "inbox.blocked" : status === "needs_input" ? "inbox.needsInput" : "states.needs-you")}
            <a href="/tmux" className="font-semibold text-body-ink">{t("expanded.openInTmux")}</a>
          </span>
        ) : null}
        {showCancel ? (
          <button key="cancel" type="button" className={BTN_DANGER} disabled={busy} onClick={() => setConfirmingCancel(true)}>{t("actions.cancel")}</button>
        ) : null}
        {confirmingCancel && card.activeRun ? (
          <CancelDialog
            busy={busy}
            onDismiss={() => setConfirmingCancel(false)}
            onConfirm={() => { setConfirmingCancel(false); onAction("/api/workflow/cancel", { run_id: card.activeRun!.runId }); }}
          />
        ) : null}
      </div>
    </div>
  );
}

// F1 (diff review 240482904e68): Escape fires the dialog's 'cancel' then 'close' event. `onCancel`
// below already preventDefault()s while busy, so `onClose` only ever runs once the browser actually
// closed the dialog — exactly the case that used to leave `confirmingCancel` stuck true (the native
// dialog closed but the button's own state never reset, so a second click found it already mounted
// and the mount-only showModal() effect never ran again). Exported as a pure function so a no-jsdom
// test can assert the dismiss path fires without simulating a real dialog close event.
export function handleCancelDialogClose(busy: boolean, onDismiss: () => void) {
  if (!busy) onDismiss();
}

export function CancelDialog({ busy, onConfirm, onDismiss }: { busy: boolean; onConfirm: () => void; onDismiss: () => void }) {
  const t = useTranslations("mission");
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    if (ref.current && typeof ref.current.showModal === "function") ref.current.showModal();
  }, []);
  return (
    <dialog ref={ref} className="rounded-lg border border-line bg-surface p-5 backdrop:bg-base/60"
      onCancel={(e) => { if (busy) e.preventDefault(); }}
      onClose={() => handleCancelDialogClose(busy, onDismiss)}>
      <p className="text-[13px] text-body-ink">{t("actions.cancelTitle")}</p>
      <p className="mt-1 text-[12px] text-muted">{t("actions.cancelBody")}</p>
      <div className="mt-4 flex justify-end gap-2">
        <button type="button" disabled={busy} className={BTN_GHOST} onClick={onDismiss}>{t("actions.cancelDismiss")}</button>
        <button type="button" disabled={busy} className={BTN_DANGER} onClick={onConfirm}>{t("actions.cancelConfirm")}</button>
      </div>
    </dialog>
  );
}

function FreeformForm({ eventId, answerable, busy, onAction }: { eventId: number; answerable: boolean; busy: boolean; onAction: (url: string, body: unknown) => void }) {
  const t = useTranslations("mission");
  const [reply, setReply] = useState("");
  // round-2 F1: LIMITS.freeformReply (2000) is pre-existing in src/lib/workflow.ts (spec
  // §4.6.2 guard 6, commit 3f08724) — Part 1's own LIMITS edit only appends checks/focus/
  // whitelist, it never touches this field. Already imported above with the rest of LIMITS.
  const valid = reply.trim().length > 0 && reply.length <= LIMITS.freeformReply;
  return (
    <div className="flex flex-col gap-1.5">
      <textarea
        className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]"
        placeholder={t("actions.freeformPlaceholder")} maxLength={LIMITS.freeformReply}
        value={reply} onChange={(e) => setReply(e.target.value)} disabled={!answerable}
      />
      <button type="button" className={BTN_PRIMARY} disabled={busy || !valid || !answerable}
        onClick={() => onAction("/api/workflow/answer", { event_id: eventId, reply })}>
        {t("actions.submit")}
      </button>
      {!answerable ? <span className="block text-[11px] text-danger">{t("actions.answerUnreachable")}</span> : null}
    </div>
  );
}

function GenericQuestionForm({ pendingQuestion, busy, onAction }: { pendingQuestion: MissionPendingQuestion; busy: boolean; onAction: (url: string, body: unknown) => void }) {
  const t = useTranslations("mission");
  const [selections, setSelections] = useState<number[][]>(() => pendingQuestion.questions.map(() => []));
  const valid = selections.every((s) => s.length > 0);
  function toggle(qIdx: number, optIdx: number, multi: boolean) {
    setSelections((prev) => prev.map((sel, i) => {
      if (i !== qIdx) return sel;
      if (!multi) return [optIdx];
      return sel.includes(optIdx) ? sel.filter((v) => v !== optIdx) : [...sel, optIdx].sort((a, b) => a - b);
    }));
  }
  return (
    <div className="flex flex-col gap-2">
      {pendingQuestion.questions.map((q, qIdx) => (
        <div key={qIdx} className="flex flex-col gap-1">
          <span className="text-[11.5px] text-muted">{q.question}</span>
          <div className="flex flex-wrap gap-1">
            {q.options.map((opt, optIdx) => (
              <button key={opt.label} type="button" disabled={busy}
                className={selections[qIdx]?.includes(optIdx + 1) ? BTN_PRIMARY : BTN_GHOST}
                onClick={() => toggle(qIdx, optIdx + 1, q.multiSelect)}>
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      ))}
      <button type="button" className={BTN_PRIMARY} disabled={busy || !valid}
        onClick={() => onAction("/api/workflow/answer", { event_id: pendingQuestion.eventId, reply: composeIndexedReply(pendingQuestion, selections) })}>
        {t("actions.submit")}
      </button>
    </div>
  );
}

function Hint({ show }: { show: boolean }) {
  const t = useTranslations("mission");
  return show ? <span className="text-[11px] text-danger">{t("actions.hintInvalid")}</span> : null;
}

// F2 (diff review 240482904e68): every dispatch/merge control relied on its placeholder alone —
// no label at all for the selects. `sr-only` reuses the codebase's existing visually-hidden-label
// pattern (Shell.tsx skip link, page <h1>s) so the compact placeholder-driven layout is unchanged.
function FieldLabel({ htmlFor, children }: { htmlFor: string; children: React.ReactNode }) {
  return <label htmlFor={htmlFor} className="sr-only">{children}</label>;
}

type DispatchFields = {
  command: "review" | "build"; kind: "spec" | "plan" | "diff"; target: string; focus: string;
  plan: string; phase: string; branch: string; whitelist: string; verify: string; build: string; profile: "default" | "fallback";
};
const EMPTY_DISPATCH: DispatchFields = { command: "review", kind: "diff", target: "", focus: "", plan: "", phase: "", branch: "", whitelist: "", verify: "", build: "", profile: "default" };

// round-2 F2: a spec/plan review target was only checked for non-empty, never against
// LIMITS.target (512, spec §7.4's own merge-target bound, reused here since no separate
// bound is declared for a review target). Exported so ProjectCard.test.tsx can assert this
// boolean directly — the field's OWN useState starts at "" with no card-derived prefill,
// so there is no non-empty-but-invalid value a static render can observe without simulating
// typing (Global Constraints ban that);
// a direct call is the only way to exercise the overlong-target branch at all.
export function dispatchValid(f: DispatchFields): boolean {
  if (f.command === "review") {
    return (f.kind === "diff" ? RUN_ID_RE.test(f.target) : f.target.trim().length > 0 && f.target.length <= LIMITS.target) && validOptionalFocus(f.focus || undefined);
  }
  return f.plan.trim().length > 0 && validPhaseToken(f.phase) && validRefName(f.branch)
    && validSingleLineCommand(f.whitelist, LIMITS.whitelist) && validSingleLineCommand(f.verify, LIMITS.verify)
    && (f.build === "" || validSingleLineCommand(f.build, LIMITS.build));
}

const DISPATCH_OPTIONS = ["reviewSpec", "reviewPlan", "reviewDiff", "build"] as const;
type DispatchOptionUi = (typeof DISPATCH_OPTIONS)[number];

function DispatchForm({ card, busy, onAction }: { card: MissionCard; busy: boolean; onAction: (url: string, body: unknown) => void }) {
  const t = useTranslations("mission");
  const uid = useId();
  const [f, setF] = useState<DispatchFields>(EMPTY_DISPATCH);
  const [option, setOption] = useState<DispatchOptionUi>("reviewSpec");
  const set = <K extends keyof DispatchFields>(k: K, v: DispatchFields[K]) => setF((prev) => ({ ...prev, [k]: v }));
  // The local DispatchFields{command,kind} pair still drives dispatchValid unchanged; keep it
  // synced from the 4-way option so validation stays byte-identical to the pre-reflow form.
  const kind: DispatchFields["kind"] = option === "reviewPlan" ? "plan" : option === "reviewSpec" ? "spec" : "diff";
  const command: DispatchFields["command"] = option === "build" ? "build" : "review";
  const effectiveF: DispatchFields = { ...f, command, kind };
  const valid = dispatchValid(effectiveF);
  const targetValid = kind === "diff" ? RUN_ID_RE.test(f.target) : f.target.trim().length > 0 && f.target.length <= LIMITS.target;
  const focusValid = validOptionalFocus(f.focus || undefined);
  const phaseValid = validPhaseToken(f.phase);
  const branchValid = validRefName(f.branch);
  const whitelistValid = validSingleLineCommand(f.whitelist, LIMITS.whitelist);
  const verifyValid = validSingleLineCommand(f.verify, LIMITS.verify);
  const buildValid = f.build === "" || validSingleLineCommand(f.build, LIMITS.build);
  const hasExtraValue = f.focus !== "" || f.phase !== "" || f.branch !== "" || f.whitelist !== "" || f.verify !== "" || f.build !== "";
  function submit() {
    onAction("/api/workflow/dispatch", dispatchBodyFor(option, effectiveF, card.dir));
  }
  return (
    <section className="flex flex-col gap-2">
      <h3 className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("actions.dispatch")}</h3>
      {/* F3 (cold review 1a4d1da24d88): every primary control — command, target/plan, the
          build-only profile select, and the submit button — shares ONE flex-wrap row. Only
          the rarely-touched extras (focus, or phase/branch/whitelist/verify/build) live
          inside <details> below it. */}
      <div data-testid="dispatch-row" className="flex flex-wrap items-center gap-2">
        <FieldLabel htmlFor={`${uid}-option`}>{t("actions.dispatchCommand")}</FieldLabel>
        <select id={`${uid}-option`} className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]" value={option} onChange={(e) => setOption(e.target.value as DispatchOptionUi)}>
          {DISPATCH_OPTIONS.map((o) => <option key={o} value={o}>{t(`actions.dispatchOption.${o}`)}</option>)}
        </select>
        <FieldLabel htmlFor={`${uid}-target`}>{option === "build" ? t("actions.dispatchPlan") : t("actions.dispatchTarget")}</FieldLabel>
        <input
          id={`${uid}-target`}
          className="min-w-0 flex-1 rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]"
          placeholder={option === "build" ? t("actions.dispatchPlan") : t("actions.dispatchTarget")}
          value={option === "build" ? f.plan : f.target}
          onChange={(e) => set(option === "build" ? "plan" : "target", e.target.value)}
        />
        {option === "build" ? (
          <>
            <FieldLabel htmlFor={`${uid}-profile`}>{t("actions.dispatchProfile")}</FieldLabel>
            <select id={`${uid}-profile`} className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]" value={f.profile} onChange={(e) => set("profile", e.target.value as DispatchFields["profile"])}>
              <option value="default">{t("actions.dispatchProfileDefault")}</option>
              <option value="fallback">{t("actions.dispatchProfileFallback")}</option>
            </select>
          </>
        ) : null}
        <button type="button" className={BTN_PRIMARY} disabled={busy || !valid} onClick={submit}>{t("actions.dispatch")}</button>
      </div>
      {option === "build" ? null : <Hint show={f.target.length > 0 && !targetValid} />}
      <details open={hasExtraValue}>
        <summary className="cursor-pointer text-[11px] font-semibold text-body-ink">{t("actions.moreOptions")}</summary>
        <div className="flex flex-col gap-2 pt-2">
          {option !== "build" ? (
            <>
              <FieldLabel htmlFor={`${uid}-focus`}>{t("actions.dispatchFocus")}</FieldLabel>
              <input id={`${uid}-focus`} className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]" placeholder={t("actions.dispatchFocus")} value={f.focus} onChange={(e) => set("focus", e.target.value)} />
              <Hint show={f.focus.length > 0 && !focusValid} />
            </>
          ) : (
            <>
              <FieldLabel htmlFor={`${uid}-phase`}>{t("actions.dispatchPhase")}</FieldLabel>
              <input id={`${uid}-phase`} className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]" placeholder={t("actions.dispatchPhase")} value={f.phase} onChange={(e) => set("phase", e.target.value)} />
              <Hint show={f.phase.length > 0 && !phaseValid} />
              <FieldLabel htmlFor={`${uid}-branch`}>{t("actions.dispatchBranch")}</FieldLabel>
              <input id={`${uid}-branch`} className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]" placeholder={t("actions.dispatchBranch")} value={f.branch} onChange={(e) => set("branch", e.target.value)} />
              <Hint show={f.branch.length > 0 && !branchValid} />
              <FieldLabel htmlFor={`${uid}-whitelist`}>{t("actions.dispatchWhitelist")}</FieldLabel>
              <input id={`${uid}-whitelist`} className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]" placeholder={t("actions.dispatchWhitelist")} value={f.whitelist} onChange={(e) => set("whitelist", e.target.value)} />
              <Hint show={f.whitelist.length > 0 && !whitelistValid} />
              <FieldLabel htmlFor={`${uid}-verify`}>{t("actions.dispatchVerify")}</FieldLabel>
              <input id={`${uid}-verify`} className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]" placeholder={t("actions.dispatchVerify")} value={f.verify} onChange={(e) => set("verify", e.target.value)} />
              <Hint show={f.verify.length > 0 && !verifyValid} />
              <FieldLabel htmlFor={`${uid}-build`}>{t("actions.dispatchBuild")}</FieldLabel>
              <input id={`${uid}-build`} className="rounded-md border border-line bg-surface px-2 py-1.5 text-[12.5px]" placeholder={t("actions.dispatchBuild")} value={f.build} onChange={(e) => set("build", e.target.value)} />
              <Hint show={f.build.length > 0 && !buildValid} />
            </>
          )}
        </div>
      </details>
    </section>
  );
}

const SESSION_LINES_CAP = 3;

// Item 4: one compact-face line per session (tmux group or native codex thread), capped with
// a trailing "+N". A tmux group's own display name is its session label; a codex row has no
// session name field, so it reuses the same 8-char thread-prefix convention ExpandedCard
// already renders (ProjectCard.tsx's own codex row, minus the trailing ellipsis).
// F1: the newest FINITE epoch across a set of timestamps — mirrors statusPillSuffix's own
// non-finite guard (F3), so an unparseable lastEventTs never blocks a later valid one.
function newestFiniteMs(timestamps: (string | null)[]): number | null {
  let best: number | null = null;
  for (const ts of timestamps) {
    if (!ts) continue;
    const ms = Date.parse(ts);
    if (Number.isFinite(ms) && (best === null || ms > best)) best = ms;
  }
  return best;
}

// One line per in-flight headless run (builder/reviewer — e.g. several parallel Codex
// reviews), rendered next to the tmux/codex session rows in both SessionLines (compact) and
// the expanded sessions section: stage · runtime · running · elapsed minutes. Shared so the
// two never drift on format.
function ActiveRunLine({ run }: { run: MissionActiveRunSummary }) {
  const t = useTranslations("mission");
  const minutes = run.startedAt ? elapsedMinutesSince(run.startedAt, Date.now()) : null;
  return (
    <div className="truncate font-mono text-[11px] text-muted">
      {t("expanded.sessionLineNoRole", { name: run.stage ? t(`stages.${run.stage}`) : "", runtime: run.runtime ?? "?", state: t("activity.running") })}
      {minutes !== null ? t("pill.elapsed", { minutes }) : null}
    </div>
  );
}

function SessionLines({ sessions, activeRuns, builder }: { sessions: MissionCard["sessions"]; activeRuns: MissionCard["activeRuns"]; builder: string }) {
  const t = useTranslations("mission");
  const runs = activeRuns ?? [];
  if (sessions.length === 0 && runs.length === 0) return null;
  const visible = sessions.slice(0, SESSION_LINES_CAP);
  const hidden = sessions.length - visible.length;
  return (
    <div className="flex flex-col gap-0.5">
      {visible.map((s) => {
        if (s.transport === "codex") {
          const ms = newestFiniteMs([s.lastEventTs]);
          return (
            <div key={`codex-${s.threadId}`} className="truncate font-mono text-[11px] text-muted">
              {t("expanded.sessionLineNoRole", { name: `${t("codexSession")} ${s.threadId.slice(0, 8)}`, runtime: "codex", state: t(`states.${s.state}`) })}
              {ms !== null ? <> · <RelativeTime epochMs={ms} /></> : null}
            </div>
          );
        }
        const p0 = s.panes[0];
        // F2: the session line's state is the WORST across every pane, not just the first.
        const state = worstPaneState(s.panes);
        // F1: same for the timestamp — the newest pane wins, not always the first.
        const ms = newestFiniteMs(s.panes.map((p) => p.lastEventTs));
        // Session-runtime-label fix: the PANE's own harness, not the project-level card.builder
        // (which names the project's assigned build runtime, e.g. "codex", and can legitimately
        // differ from whoever's actually sitting in this tmux pane). Falls back to card.builder
        // only when the pane's runtime can't be derived (session-registration-only pane).
        const runtime = p0.runtime && p0.runtime !== "unknown" ? p0.runtime : builder;
        return (
          <div key={`${p0.pane}:${p0.tmuxIncarnation}`} className="truncate font-mono text-[11px] text-muted">
            {t("expanded.sessionLine", { name: s.session, runtime, role: t(`roles.${p0.role}`), state: t(`states.${state}`) })}
            {ms !== null ? <> · <RelativeTime epochMs={ms} /></> : null}
          </div>
        );
      })}
      {runs.map((run) => <ActiveRunLine key={run.runId} run={run} />)}
      {hidden > 0 ? <span className="text-[11px] text-muted">+{hidden}</span> : null}
    </div>
  );
}

const RUNTIME_AVATAR: Record<string, { letter: string; cls: string }> = {
  claude: { letter: "C", cls: "bg-brand" }, "claude-code": { letter: "C", cls: "bg-brand" },
  codex: { letter: "X", cls: "bg-info" }, opencode: { letter: "O", cls: "bg-muted" },
};

function RuntimeAvatar({ builder }: { builder: string }) {
  const entry = RUNTIME_AVATAR[builder.toLowerCase()];
  if (!entry) return null;
  return <span className={`flex h-4 w-4 flex-none items-center justify-center rounded-sm text-[9px] font-bold text-ink ${entry.cls}`}>{entry.letter}</span>;
}

export function ProjectCard({ card, issues, expanded, onToggle }: { card: MissionCard; issues?: number; expanded: boolean; onToggle: () => void }) {
  const t = useTranslations("mission");
  const { data: settings } = useGeneralSettings();
  // F1/F3 (diff review e52d9e6dc555): loading/error settings keep the pre-fetch assumed-enabled
  // behavior — only an explicit ok:true with github off suppresses the PR section.
  const githubEnabled = settings?.ok === true ? settings.data.integrations.github : true;
  const styles = STATE_STYLES[card.headline];
  const startedMs = card.activeRun?.startedAt ? Date.parse(card.activeRun.startedAt) : NaN;
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  // merge-button-hide fix: the eventId of the last answer POST that succeeded — compared against
  // actionEventId(action) below, so the confirmation only shows for the question it actually
  // answered and clears itself once the hub poll moves the card on to something else.
  const [answeredEventId, setAnsweredEventId] = useState<number | null>(null);
  const action = pendingUiAction(card, undefined, t);
  const answered = answeredEventId !== null && answeredEventId === actionEventId(action);
  // MOA-486 round 2: an all-zero heartbeat means nothing has been recorded yet — render no
  // block at all rather than a near-zero-height sparkline that reads as a dashed line.
  const hasHeartbeat = card.heartbeat.some((n) => n > 0);
  const pillSuffix = statusPillSuffix(card, Date.now());

  async function runAction(url: string, body: unknown) {
    setBusy(true);
    setActionError(null);
    const result = await postWorkflowAction<unknown>(url, body);
    setBusy(false);
    if (!result.ok) { setActionError(result.error); return; }
    const eventId = answerEventId(body);
    if (eventId !== null) setAnsweredEventId(eventId);
  }

  return (
    <div className={`relative flex min-w-0 w-full flex-col gap-2.5 rounded-lg border bg-surface px-5 py-[18px] ${BORDER_CLASS[card.headline]}`}>
      {/* header — clicking the expand toggle expands the card in place (§2.3), no navigation.
          F4 (diff review): the toggle is its own <button>, a sibling of the "···" menu's
          buttons — not their ancestor — so no interactive control ever nests inside another.
          Row 1 (MOA-486 round 2, Rafa's phone review): icon + name (the toggle button, `flex-1`
          so it grows to fill the row and pushes the following siblings to the right edge) then
          the "···" menu trigger + chevron, all inline. Row 2: stage word + pipeline dots +
          status pill, inline on one line (data-testid="card-stage-row") — no more dedicated
          row for the pill (round 1's `card-status-row`, dropped after visual review). */}
      <div data-testid="card-header-row" className="flex min-w-0 items-center gap-3 text-left">
        <button type="button" onClick={onToggle} aria-expanded={expanded} className="flex min-w-0 flex-1 items-center gap-3 text-left">
          <span className="flex h-9 w-9 flex-none items-center justify-center rounded-md bg-brand-soft text-brand">
            <FolderGit2 className="h-[18px] w-[18px]" />
          </span>
          <span className="min-w-0 flex-1 truncate text-[15px] font-semibold text-ink">{card.name}</span>
          {card.flag ? (
            <span className="flex-none rounded-full border border-brand-soft-border bg-brand-soft px-[7px] py-0.5 text-[10px] font-bold uppercase tracking-[.06em] text-brand">
              {card.flag}
            </span>
          ) : null}
        </button>
        <PrefsButtons dir={card.dir} prefs={card.prefs} visibility={card.visibility} />
        {expanded ? <ChevronDown className="h-4 w-4 flex-none text-muted" /> : <ChevronRight className="h-4 w-4 flex-none text-muted" />}
      </div>
      <div data-testid="card-stage-row" className="flex flex-wrap items-center gap-1.5">
        <span className="rounded-full border border-line px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[.06em] text-muted">{t(`stages.${card.stage}`)}</span>
        <div className="flex items-center gap-1.5" aria-hidden>
          {PIPELINE_STAGES.map((s, i) => (
            <span
              key={s}
              // round-2 F2: dedicated test marker — `bg-brand-soft` (the flag pill above) also matches a `/bg-brand/` substring regex.
              data-active={i === stageDotIndex(card.stage) ? "true" : undefined}
              className={`h-1.5 w-1.5 rounded-full ${i === stageDotIndex(card.stage) ? "bg-brand" : "bg-surface-3"}`}
            />
          ))}
        </div>
        {/* T1: the pill can be as narrow as its container demands (min-w-0/max-w-full) — a
            long "trabalhando · subagente ×1 · 0 min" line truncates inside it instead of
            overflowing the card; `flex-wrap` on the row above lets the whole pill drop to its
            own line first, truncate is the last resort. */}
        <span className={`flex min-w-0 max-w-full flex-none items-center gap-1.5 rounded-full border px-2 py-1 text-[11px] font-semibold ${styles.tag}`}>
          <span className={`h-1.5 w-1.5 flex-none rounded-full ${styles.dot} [animation:jax-pulse_1.6s_infinite]`} />
          <span className="truncate">
            {t(`states.${card.headline}`)}
            {pillSuffix.subagentCount > 0 ? t("pill.subagents", { count: pillSuffix.subagentCount }) : null}
            {pillSuffix.minutes !== null ? t("pill.elapsed", { minutes: pillSuffix.minutes }) : null}
          </span>
        </span>
      </div>

      {/* MOA-486 round 2: no icon-column indent below the header — body rows use the card's
          full width from its own left padding. No heartbeat block at all below `md:` (mobile),
          and nothing at all (not even on desktop) when card.heartbeat has no data. */}
      {hasHeartbeat ? (
        <div className="hidden items-center justify-between gap-3 md:flex">
          <Heartbeat minutes={card.heartbeat} headline={card.headline} />
          {card.lastAction ? (
            <span className="flex-none text-[11px] text-muted">
              {t("expanded.lastActionLine", { tool: card.lastAction.tool })} · <RelativeTime epochMs={Date.parse(card.lastAction.ts)} />
            </span>
          ) : null}
        </div>
      ) : null}

      {card.legacyGateField ? <span className="text-[11px] italic text-muted">{t("legacyGateHint")}</span> : null}

      {card.activeRun ? (
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
          <span className="flex min-w-0 max-w-full flex-wrap items-center gap-3">
            {card.activeRun.stage ? <span>{t(`stages.${card.activeRun.stage}`)}</span> : null}
            {card.activeRun.target ? (
              <span className="flex min-w-0 max-w-full items-center gap-1">
                {card.activeRun.targetKind === "document" ? <File className="h-3.5 w-3.5 flex-none" aria-hidden /> : null}
                {card.activeRun.targetKind === "branch" ? <GitBranch className="h-3.5 w-3.5 flex-none" aria-hidden /> : null}
                <span className="min-w-0 [overflow-wrap:anywhere]">{card.activeRun.target}</span>
              </span>
            ) : null}
            {card.activeRun.runtime ? <span className="flex items-center gap-1"><Bot className="h-3.5 w-3.5" aria-hidden />{card.activeRun.runtime}</span> : null}
            <span>{t(`activity.${card.activeRun.descriptionKey}`)}</span>
            {card.activeRun.count > 1 ? <span>{t("activity.otherRuns", { count: card.activeRun.count - 1 })}</span> : null}
            {card.activeRun.kind === "build" && card.activeRun.lastStep ? (
              <span className="min-w-0 [overflow-wrap:anywhere]">· {card.activeRun.lastStep}</span>
            ) : null}
          </span>
          {Number.isFinite(startedMs) ? (
            <span className="flex flex-none items-center gap-1.5">
              {t("activity.started")}
              <Clock3 className="h-3.5 w-3.5" aria-hidden />
              <RelativeTime epochMs={startedMs} />
            </span>
          ) : null}
        </div>
      ) : null}
      <div className="flex flex-col gap-1.5">
        <SessionLines sessions={liveSessions(card.sessions)} activeRuns={card.activeRuns} builder={card.builder} />
        <LastLine card={card} />
      </div>

      <div className="flex flex-wrap items-center gap-4 border-t border-line-subtle pt-[13px] text-xs text-muted">
        {card.builder ? <span className="flex items-center gap-1"><RuntimeAvatar builder={card.builder} />{card.builder}</span> : null}
        {card.branch ? <span className="flex items-center gap-1"><GitBranch className="h-3.5 w-3.5" aria-hidden />{card.branch}</span> : null}
        {typeof issues === "number" ? <span>{t("projects.issueCount", { count: issues })}</span> : null}
        {!githubEnabled ? null : card.pr.status === "ok" && card.pr.newest ? (
          <span className="flex items-center gap-1.5">
            <GitPullRequest className="h-3.5 w-3.5" />
            PR #{card.pr.newest.number} · {t(`card.ci.${card.pr.newest.ci}`)}
            {card.pr.count > 1 ? ` ${t(card.pr.truncated ? "prMoreTruncated" : "prMore", { count: card.pr.count - 1 })}` : ""}
          </span>
        ) : card.pr.status === "loading" || card.pr.status === "unknown" ? (
          <span className="h-3.5 w-20 animate-pulse rounded bg-surface-2" />
        ) : card.pr.status === "error" ? (
          <span className="text-warning">{t("prUnavailable")}</span>
        ) : null}
      </div>

      <ActionRow card={card} action={action} busy={busy} answered={answered} expanded={expanded} onExpand={onToggle} onAction={runAction} />
      {actionError ? <p className="text-[11.5px] text-danger">{t("actions.actionFailed", { error: actionError })}</p> : null}
      {expanded ? <ExpandedCard card={card} action={action} busy={busy} onAction={runAction} /> : null}
    </div>
  );
}

const HEARTBEAT_STROKE: Record<MissionState, string> = {
  "needs-you": "stroke-danger", stuck: "stroke-warning", working: "stroke-accent",
  waiting: "stroke-info", idle: "stroke-muted", unknown: "stroke-muted",
};

function Heartbeat({ minutes, headline }: { minutes: number[]; headline: MissionState }) {
  const path = heartbeatPath(minutes);
  return (
    <div data-testid="heartbeat" className="min-w-0 flex-1" aria-hidden>
      <svg viewBox="0 0 200 26" className="h-4 w-full" preserveAspectRatio="none">
        <path
          d={path} fill="none" strokeWidth="2" pathLength={1}
          className={`${HEARTBEAT_STROKE[headline]} ${headline === "working" ? "hb-draw" : ""}`}
        />
      </svg>
    </div>
  );
}

// Plan review round 1 F4: the pure bucket→patch mapping, factored out so it has its own
// test with no DOM/click involved ("pure handler tests, no jsdom") — a hidden management
// row's one recovery action is unhide; an autoHidden row's is pin (there is no `hidden_at`
// to clear there, since the visibility ladder — mission.ts — only reaches "autoHidden" for
// an unpinned, un-hidden card, spec §10).
export function bucketPrefsPatch(bucket: "hidden" | "autoHidden"): { hidden?: boolean; pinned?: boolean } {
  return bucket === "hidden" ? { hidden: false } : { pinned: true };
}

const MENU_ITEM = "rounded px-2 py-1.5 text-left text-[12px] font-medium text-body-ink hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-50";

// Item 1 (mobile hide/pin UX): the header's own "···" trigger + dropdown, replacing the old
// two-icon pair — pure/stateless so its markup has a direct, no-jsdom renderToStaticMarkup
// test, same split as PendingActionButtons above. MOA-486 round 2 (Rafa's phone review): the
// trigger drops its border and shrinks from a bordered 44px box to a plain 32px (h-8/w-8)
// icon-only square, `text-muted` → `hover:text-ink` — still a comfortable tap target (the box
// is well past the 16px icon it centers), just visually lighter. Both items always render,
// labels only swap per state (pinned/hidden). `pending`/`error` (item 2) drive aria-busy/
// disabled and the failure text; PrefsButtons owns that state and passes it down.
// Bug fix (card-hidden-menu): the hide label used to read `prefs.hiddenAt !== null` directly.
// mission.ts's visibility ladder clears the effective hidden state on new activity WITHOUT
// clearing the stored `hiddenAt` — a card that auto-returns still carries a non-null
// `hiddenAt`, so the raw check kept showing "Show" on an already-visible card. `hidden` is
// the caller's `card.visibility === "hidden"` — the same predicate the board itself uses to
// decide whether the card is shown — so the menu can never disagree with the board.
export function PrefsMenu({
  prefs, hidden, pending, error, onPin, onHide,
}: { prefs: MissionCard["prefs"]; hidden: boolean; pending: boolean; error: string | null; onPin: () => void; onHide: () => void }) {
  const t = useTranslations("mission");
  return (
    <details className="relative flex-none">
      <summary
        aria-label={t("actions.more")}
        className="flex h-8 w-8 list-none items-center justify-center rounded text-muted transition-colors hover:text-ink [&::-webkit-details-marker]:hidden"
        onClick={(e) => e.stopPropagation()}
      >
        <MoreHorizontal className="h-4 w-4" aria-hidden />
      </summary>
      <div aria-busy={pending} className={`absolute right-0 z-10 mt-1 flex w-36 flex-col gap-1 rounded-md border border-line bg-surface p-1.5 shadow-lg ${pending ? "opacity-60" : ""}`}>
        <button type="button" disabled={pending} className={MENU_ITEM} onClick={(e) => { e.stopPropagation(); onPin(); }}>
          {prefs.pinned ? t("actions.unpin") : t("actions.pin")}
        </button>
        <button type="button" disabled={pending} className={MENU_ITEM} onClick={(e) => { e.stopPropagation(); onHide(); }}>
          {hidden ? t("actions.show") : t("actions.hide")}
        </button>
        {error ? <span className="text-[10px] text-danger">{t("actions.actionFailed", { error })}</span> : null}
      </div>
    </details>
  );
}

// Exported (round 1 F4): Task 8 reuses this same component, unchanged, inside
// `ProjectsColumn.tsx`'s hidden/autoHidden management rows via the `bucket` prop — one
// component, three contexts, instead of duplicating button markup per bucket.
// Item 2 (optimistic hide/pin): owns the pending/error state and drives postPrefsPatch
// (src/lib/prefsAction.ts), which patches the hub query cache immediately, POSTs, then
// invalidates (success) or rolls back (failure) — reusing this file's own actionError-paragraph
// pattern for the failure surface (ProjectCard.runAction below) rather than the kanban board's
// heavier WriteGuidance machinery, which exists for Linear's richer write-conflict states that
// don't apply to a boolean prefs flag.
export function PrefsButtons({ dir, prefs, bucket, visibility }: { dir: string; prefs: MissionCard["prefs"]; bucket?: "hidden" | "autoHidden"; visibility?: MissionCard["visibility"] }) {
  const t = useTranslations("mission");
  const queryClient = useQueryClient();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function act(patch: { pinned?: boolean; hidden?: boolean }) {
    if (pending) return; // guard against double clicks while a request is in flight
    setPending(true);
    setError(null);
    const result = await postPrefsPatch(queryClient, dir, patch);
    setPending(false);
    if (!result.ok) setError(result.error);
  }

  if (bucket) {
    return (
      <div className="flex flex-col items-end gap-1">
        <button
          type="button" aria-busy={pending} disabled={pending}
          className={`rounded border border-line-strong px-1.5 py-0.5 text-[10.5px] font-semibold text-body-ink disabled:cursor-not-allowed ${pending ? "opacity-60" : ""}`}
          onClick={(e) => { e.stopPropagation(); void act(bucketPrefsPatch(bucket)); }}
        >
          {t(bucket === "hidden" ? "actions.show" : "actions.pin")}
        </button>
        {error ? <span className="text-[10px] text-danger">{t("actions.actionFailed", { error })}</span> : null}
      </div>
    );
  }
  const hidden = visibility === "hidden";
  return (
    <PrefsMenu
      prefs={prefs} hidden={hidden} pending={pending} error={error}
      onPin={() => void act({ pinned: !prefs.pinned })}
      onHide={() => void act({ hidden: !hidden })}
    />
  );
}

function LastRecord({ card }: { card: MissionCard }) {
  const t = useTranslations("mission");
  const updatedMs = Date.parse(card.updated);

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("activity.lastRecord")}</h3>
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        <span className="flex min-w-0 max-w-full flex-wrap items-center gap-3">
          <span>{t(`stages.${card.stage}`)}</span>
          {card.branch ? (
            <span className="flex min-w-0 max-w-full items-center gap-1">
              <GitBranch className="h-3.5 w-3.5 flex-none" aria-hidden />
              <span className="min-w-0 [overflow-wrap:anywhere]">{card.branch}</span>
            </span>
          ) : null}
          {card.builder ? <span className="flex items-center gap-1"><Bot className="h-3.5 w-3.5" aria-hidden />{card.builder}</span> : null}
        </span>
        {Number.isFinite(updatedMs) ? (
          <span className="flex flex-none items-center gap-1.5"><Clock3 className="h-3.5 w-3.5" aria-hidden /><RelativeTime epochMs={updatedMs} /></span>
        ) : null}
      </div>
      {card.freshnessCount !== null && card.freshnessCount > 0 ? (
        <div className={`flex items-center gap-1.5 text-xs ${card.freshnessWarning ? "font-semibold text-warning" : "text-muted"}`}>
          <Hourglass className="h-3.5 w-3.5" aria-hidden />
          {t("freshness", { count: card.freshnessCount })}
        </div>
      ) : null}
      {card.now ? <p className="min-w-0 max-w-full [overflow-wrap:anywhere] text-[12.5px] italic text-body-ink">&ldquo;{card.now}&rdquo;</p> : null}
    </div>
  );
}

function LastRunLine({ card }: { card: MissionCard }) {
  const t = useTranslations("mission");
  const lastRun = card.lastRun;
  if (!lastRun) return null;
  const showReportProblem = lastRun.contractStatus === "missing" || lastRun.contractStatus === "invalid";
  return (
    // MOA-486 round 2: `gap-x-2` (not `gap-2`, which also adds row-gap on wrap) — outcome ·
    // model · report link read as one line; wrapping only kicks in on true overflow.
    <div className={`flex flex-wrap items-center gap-x-2 text-xs ${LAST_RUN_TONE_CLASS[lastRun.tone]}`}>
      <span className="font-semibold">{t(`lastRun.${lastRun.labelKey}`)}</span>
      {lastRun.labelKey === "buildOk" && lastRun.headSha ? (
        <span className="font-mono text-muted">· {lastRun.headSha.slice(0, 7)}</span>
      ) : null}
      {lastRun.stage || lastRun.diagnostic ? (
        <span className="text-muted">
          · {lastRun.stage ? t(`lastRun.stage.${lastRun.stage}`) : null} — {lastRun.diagnostic ?? t("lastRun.noCause")}
        </span>
      ) : null}
      {worstFinding(lastRun.findings) ? (
        <span className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase ${FINDING_SEVERITY_CLASS[worstFinding(lastRun.findings)!.severity]}`}>
          {worstFinding(lastRun.findings)!.count} {t(`lastRun.severity.${worstFinding(lastRun.findings)!.severity}`)}
        </span>
      ) : null}
      {showReportProblem ? <span className="text-muted">{t("lastRun.reportProblem", { status: lastRun.contractStatus })}</span> : null}
      {lastRun.runtimeModel ? (
        <span className="flex items-center gap-1 text-muted">
          <Bot className="h-3.5 w-3.5" aria-hidden />
          {lastRun.runtimeModel}
          {/* Plan review round 1 F5: profileName is builder-only (spec §11/AC12) — gating on
              role here, not just presence, since a stored row can carry a stale value. */}
          {lastRun.profileName && lastRun.role === "builder" ? <span>· {lastRun.profileName}</span> : null}
        </span>
      ) : null}
      {lastRun.reportRel ? (
        <a href={`/files?${serializeFileSelection(new URLSearchParams(), { root: "repos", rel: `${card.dir}/${lastRun.reportRel}` }).toString()}`} className="font-semibold text-brand">
          {t("expanded.openReport")}
        </a>
      ) : null}
      {lastRun.targetRel ? (
        <a href={`/files?${serializeFileSelection(new URLSearchParams(), { root: "repos", rel: `${card.dir}/${lastRun.targetRel}` }).toString()}`} className="font-semibold text-brand">
          {t("expanded.openDocument")}
        </a>
      ) : null}
    </div>
  );
}

// Item 5: the stuck/needs-you headlines get a free-precedence "last" line instead of
// LastRunLine — every other headline is unaffected.
function LastLine({ card }: { card: MissionCard }) {
  const t = useTranslations("mission");
  const nowMs = Date.now();
  if (card.headline === "stuck") {
    const line = stuckLine(card, nowMs);
    if (!line) return null;
    return <span className="text-xs text-muted">{line.key === "stuckLine" ? t("expanded.stuckLine", { declared: line.declared, elapsed: line.elapsed }) : t("expanded.stuckLineFallback", { elapsed: line.elapsed })}</span>;
  }
  if (card.headline === "needs-you" && card.gate === "awaiting-approval") {
    const line = gateWaitingLine(card, nowMs);
    if (!line) return null;
    return <span className="text-xs text-muted">{t("expanded.gateWaitingLine", { elapsed: line.elapsed })}</span>;
  }
  return <LastRunLine card={card} />;
}

function ExpandedCard({
  card, action, busy, onAction,
}: { card: MissionCard; action: PendingUiAction; busy: boolean; onAction: (url: string, body: unknown) => void }) {
  const t = useTranslations("mission");

  return (
    <div className="flex flex-col gap-4 border-t border-line-subtle pt-[15px]">
      <LastRecord card={card} />
      <section className="flex flex-col gap-2">
        <h3 className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("expanded.sessions")}</h3>
        {!card.hubReady ? (
          <span className="text-[12.5px] text-muted">{t("expanded.unknown")}</span>
        ) : liveSessions(card.sessions).length === 0 && (card.activeRuns ?? []).length === 0 ? (
          <span className="text-[12.5px] text-muted">{t("expanded.noSessions")}</span>
        ) : (
          <>
          {liveSessions(card.sessions).map((s) => {
            // MOA-469 §4: native Codex rows have no pane — no tmux action, no fabricated pane id.
            // Key tmux groups by the first pane+incarnation, not the display label (mission.ts
            // Finding 2); key native rows by threadId.
            if (s.transport === "codex") {
              const paneStyles = STATE_STYLES[s.state];
              return (
                <div key={`codex-${s.threadId}`} className="flex flex-col gap-1">
                  <span className="flex items-center gap-1.5 text-[12.5px] font-semibold text-body-ink">
                    <ChevronRight className="h-3.5 w-3.5 text-muted" />
                    {t("codexSession")}
                  </span>
                  <div className="flex items-center gap-2.5 pl-5 text-[12.5px]">
                    <span className="font-mono text-muted" title={s.threadId}>{s.threadId.slice(0, 8)}…</span>
                    <span className={`rounded-full border px-2 py-0.5 text-[10.5px] font-semibold ${paneStyles.tag}`}>{t(`states.${s.state}`)}</span>
                    {s.subagentCount > 0 ? <span className="text-[10.5px] text-muted">{t("expanded.subagentSuffix", { count: s.subagentCount })}</span> : null}
                    {s.lastEventTs ? (
                      <span className="ml-auto flex-none text-[11px] text-muted"><RelativeTime epochMs={Date.parse(s.lastEventTs)} /></span>
                    ) : null}
                  </div>
                </div>
              );
            }
            return (
            <div key={`${s.panes[0].pane}:${s.panes[0].tmuxIncarnation}`} className="flex flex-col gap-1">
              <span className="flex items-center gap-1.5 text-[12.5px] font-semibold text-body-ink">
                <ChevronRight className="h-3.5 w-3.5 text-muted" />
                {s.session}
              </span>
              {s.panes.map((p) => {
                const paneStyles = STATE_STYLES[p.state];
                const paneMs = p.lastEventTs ? Date.parse(p.lastEventTs) : NaN;
                return (
                  <div key={`${p.pane}:${p.tmuxIncarnation}`} className="flex items-center gap-2.5 pl-5 text-[12.5px]">
                    <span className="font-mono text-muted">{p.pane}</span>
                    <span className="text-muted">{t(`roles.${p.role}`)}</span>
                    <span className={`rounded-full border px-2 py-0.5 text-[10.5px] font-semibold ${paneStyles.tag}`}>{t(`states.${p.state}`)}</span>
                    {p.subagentCount > 0 ? <span className="text-[10.5px] text-muted">{t("expanded.subagentSuffix", { count: p.subagentCount })}</span> : null}
                    {Number.isFinite(paneMs) ? (
                      <span className="ml-auto flex-none text-[11px] text-muted"><RelativeTime epochMs={paneMs} /></span>
                    ) : null}
                    <a href="/tmux" className="flex-none rounded border border-line-strong px-1.5 py-0.5 text-[10.5px] font-semibold text-body-ink">
                      {t("expanded.openInTmux")}
                    </a>
                  </div>
                );
              })}
            </div>
            );
          })}
          {(card.activeRuns ?? []).map((run) => <ActiveRunLine key={run.runId} run={run} />)}
          </>
        )}
      </section>

      <section className="flex flex-col gap-2">
        <h3 className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("expanded.pendingQuestion")}</h3>
        {!card.hubReady ? (
          <span className="text-[12.5px] text-muted">{t("expanded.unknown")}</span>
        ) : card.pendingQuestions.length === 0 && action?.kind !== "freeform" ? (
          <span className="text-[12.5px] text-muted">{t("expanded.noPendingQuestion")}</span>
        ) : (
          <>
            {/* round-1 F8: only pendingQuestions[0] — the same pane pendingUiAction already
                targets — gets a detail block here. Any other pane's pending question is reachable
                only via its own tmux link in the Sessions section above; no second detail block,
                no second action row (spec §9). */}
            {card.pendingQuestions[0] ? (
              <div key={card.pendingQuestions[0].eventId} className="flex flex-col gap-2 rounded-md border border-danger bg-danger-soft px-3 py-2.5">
                <span className="font-mono text-[11px] text-muted">{card.pendingQuestions[0].pane}</span>
                {card.pendingQuestions[0].questions.map((sub, i) => (
                  <div key={i} className="flex flex-col gap-1.5">
                    <span className="text-[12.5px] text-body-ink">{sub.question}</span>
                    <span className="text-[11px] text-muted">{sub.multiSelect ? t("expanded.multiSelect") : null}</span>
                  </div>
                ))}
                {action?.kind === "generic" && !(action.pendingQuestion.questions.length === 1 && !action.pendingQuestion.questions[0].multiSelect) ? (
                  <GenericQuestionForm pendingQuestion={action.pendingQuestion} busy={busy} onAction={onAction} />
                ) : null}
                <span className="text-[11px] text-muted">
                  {t("expanded.answerHint")}{" "}
                  <a href="/tmux" className="font-semibold text-brand">{t("expanded.openInTmux")}</a>
                </span>
              </div>
            ) : null}
            {action?.kind === "freeform" ? <FreeformForm eventId={action.eventId} answerable={action.answerable} busy={busy} onAction={onAction} /> : null}
          </>
        )}
      </section>

      {(card.headline === "idle" || card.headline === "stuck") ? <DispatchForm card={card} busy={busy} onAction={onAction} /> : null}

      <section className="flex flex-col gap-1.5">
        <h3 className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("expanded.timeline")}</h3>
        {card.loopSummary && (card.loopSummary.buildCount > 0 || card.loopSummary.diffCount > 0 || card.loopSummary.specCount > 0 || card.loopSummary.planCount > 0) ? (
          <span className="text-[11px] text-muted">
            {t("expanded.loopSummary", { specCount: card.loopSummary.specCount, planCount: card.loopSummary.planCount,
              buildCount: card.loopSummary.buildCount, diffCount: card.loopSummary.diffCount, rounds: card.loopSummary.rounds })}
            {card.loopSummary.wallDays !== null
              ? ` · ${t("expanded.loopWallTime", { days: Math.round(card.loopSummary.wallDays) })}` : ""}
          </span>
        ) : null}
        {!card.hubReady ? (
          <span className="text-[12.5px] text-muted">{t("expanded.unknown")}</span>
        ) : card.timeline.length === 0 ? (
          <span className="text-[12.5px] text-muted">{t("expanded.noTimeline")}</span>
        ) : (
          <ul className="flex flex-col gap-1 font-mono text-[11.5px] text-muted">
            {card.timeline.map((e, i) => {
              const ms = Date.parse(e.ts);
              return (
                <li key={i} className="flex items-center gap-2.5">
                  {Number.isFinite(ms) ? <RelativeTime epochMs={ms} /> : e.ts}
                  <TimelineEntryText entry={e} />
                </li>
              );
            })}
          </ul>
        )}
      </section>

      <section className="flex flex-col gap-1.5">
        <div className="flex items-center gap-2">
          <h3 className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("expanded.residuals")}</h3>
          {card.residuals.length > 0 ? <span className="text-[11px] text-muted">{card.residuals.length}</span> : null}
        </div>
        {card.residuals.length === 0 ? (
          <span className="text-[12.5px] text-muted">{t("expanded.noResiduals")}</span>
        ) : (
          <ul className="flex flex-col gap-1 text-[12.5px] text-body-ink">
            {card.residuals.map((r, i) => <li key={i}>• {r}</li>)}
          </ul>
        )}
      </section>

      {typeof card.retainedWorktrees === "number" && card.retainedWorktrees > 0 ? (
        <section className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <h3 className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("expanded.retainedWorktrees")}</h3>
            <span className="text-[11px] text-muted">{card.retainedWorktrees}</span>
          </div>
        </section>
      ) : null}
    </div>
  );
}
