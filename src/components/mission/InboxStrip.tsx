"use client";

import { AlertCircle, Flag, GitPullRequest, HelpCircle } from "lucide-react";
import { useTranslations } from "next-intl";
import { answerEventId, postWorkflowAction, stripPresentation, type InboxRow, type Mission, type MissionCard } from "@/lib/mission";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "./SourceWarning";
import { actionEventId, bareCapsuleStatus, CancelDialog, pendingUiAction, PendingActionButtons } from "./ProjectCard";
import { useState } from "react";

type Props = { mission: Mission; onExpand: (dir: string) => void };

function RowIcon({ kind }: { kind: InboxRow["kind"] }) {
  if (kind === "gate") return <Flag className="h-4 w-4 flex-none text-brand" />;
  if (kind === "question") return <HelpCircle className="h-4 w-4 flex-none text-danger" />;
  if (kind === "attention") return <AlertCircle className="h-4 w-4 flex-none text-warning" />;
  return <GitPullRequest className="h-4 w-4 flex-none text-info" />;
}

// Decision 13b: expand the target card and scroll it into view via the data-card-dir attribute ProjectsColumn already stamps on every card.
export function expandAndScroll(dir: string, onExpand: (dir: string) => void) {
  onExpand(dir);
  document.querySelector(`[data-card-dir="${dir}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
}
// merge-button-hide fix: reports success (and the answered eventId) back to the caller instead
// of a bare fire-and-forget POST, so the row can hide its buttons for a confirmation state right
// away instead of waiting for the next hub poll to prove the answer landed.
export async function runQuestionAction(url: string, body: unknown, setBusy: (b: boolean) => void, setAnsweredEventId?: (id: number) => void) {
  setBusy(true);
  const result = await postWorkflowAction(url, body);
  setBusy(false);
  if (result.ok) {
    const id = answerEventId(body);
    if (id !== null) setAnsweredEventId?.(id);
  }
}
// Decision 9: each row looks up its own card, computes pendingUiAction, and reuses the SAME button component the compact face uses.
// F2: `eventId` (the row's own event) is passed for question rows so a card with multiple pending
// questions doesn't have every row answer pendingQuestions[0]; gate rows omit it (unchanged default).
function QuestionRowActions({ card, eventId, onExpand }: { card: MissionCard; eventId?: number; onExpand: (dir: string) => void }) {
  const t = useTranslations("mission");
  const [busy, setBusy] = useState(false);
  const [answeredEventId, setAnsweredEventId] = useState<number | null>(null);
  const action = pendingUiAction(card, eventId, t);
  if (!(action?.kind === "generic" || action?.kind === "approve")) return null;
  const answered = answeredEventId !== null && answeredEventId === actionEventId(action);
  return (
    <PendingActionButtons
      action={action} busy={busy} answered={answered}
      onExpand={() => expandAndScroll(card.dir, onExpand)}
      onAction={(url, body) => runQuestionAction(url, body, setBusy, setAnsweredEventId)}
    />
  );
}
export async function runStuckCancel(runId: string, setBusy: (b: boolean) => void, setConfirming: (c: boolean) => void) {
  setBusy(true);
  await postWorkflowAction("/api/workflow/cancel", { run_id: runId });
  setBusy(false);
  setConfirming(false);
}
// Decision 10: standalone cancel trigger for a stuck row (activeRun only) — reuses CancelDialog exactly as ActionRow does.
function StuckRowActions({ card }: { card: MissionCard }) {
  const t = useTranslations("mission");
  const [busy, setBusy] = useState(false);
  const [confirming, setConfirming] = useState(false);
  if (!card.activeRun) return null;
  return (
    <>
      <button type="button" className="rounded-full border border-danger bg-danger-soft px-2.5 py-0.5 text-[11px] font-semibold text-danger" disabled={busy} onClick={() => setConfirming(true)}>
        {t("actions.cancel")}
      </button>
      {confirming ? <CancelDialog busy={busy} onDismiss={() => setConfirming(false)} onConfirm={() => runStuckCancel(card.activeRun!.runId, setBusy, setConfirming)} /> : null}
    </>
  );
}

function InboxLine({ row, card, onExpand }: { row: InboxRow; card: MissionCard | undefined; onExpand: (dir: string) => void }) {
  const t = useTranslations("mission.inbox");
  if (row.kind === "gate") {
    const updatedMs = Date.parse(row.updated);
    return (
      <div className="flex flex-wrap items-center gap-2.5 text-[13px]">
        <RowIcon kind="gate" />
        <span className="truncate font-semibold text-ink">{row.name}</span>
        <span className="text-body-ink">{t(row.gate === "blocked" ? "blocked" : "approvalPending")}</span>
        {card ? <QuestionRowActions card={card} onExpand={onExpand} /> : null}
        {Number.isFinite(updatedMs) ? (
          <span className="ml-auto flex-none text-[11px] text-muted"><RelativeTime epochMs={updatedMs} /></span>
        ) : null}
      </div>
    );
  }
  if (row.kind === "question") {
    if (row.question.trim().length === 0) {
      const status = card ? bareCapsuleStatus(card) : null;
      return (
        <div className="flex flex-wrap items-center gap-2.5 text-[13px]">
          <RowIcon kind="question" /><span className="truncate font-semibold text-ink">{row.name}</span>
          <span className="text-body-ink">{t(status === "blocked" ? "blocked" : "needsInput")}</span>
          <a href="/tmux" className="ml-auto flex-none rounded-full border border-line-strong px-2.5 py-0.5 text-[11px] font-semibold text-body-ink">{t("openTmux")}</a>
        </div>
      );
    }
    return (
      <div className="flex flex-wrap items-center gap-2.5 text-[13px]">
        <RowIcon kind="question" />
        <span className="truncate font-semibold text-ink">{row.name}</span>
        <span className="truncate text-body-ink">&ldquo;{row.question}&rdquo;</span>
        {card ? <QuestionRowActions card={card} eventId={row.eventId} onExpand={onExpand} /> : null}
        <span className="ml-auto flex-none font-mono text-[11px] text-muted">{row.pane}</span>
        <a href="/tmux" className="flex-none rounded-full border border-line-strong px-2.5 py-0.5 text-[11px] font-semibold text-body-ink">
          {t("openTmux")}
        </a>
      </div>
    );
  }
  if (row.kind === "attention") {
    // Cold review round 3, Finding 3: a pane whose last capsule is blocked/needs_input but has no
    // PARSED pending question (the "question" branch above) used to surface only as a needs-you
    // CARD, with nothing in the strip at all — this row is that same signal, at the same place
    // every other "waiting on you" item lives. MOA-469 §4 adds the native Codex variant: a native
    // session needing attention has no pane and therefore no tmux action.
    if (row.transport === "tmux") {
      return (
        <div className="flex items-center gap-2.5 text-[13px]">
          <RowIcon kind="attention" />
          <span className="truncate font-semibold text-ink">{row.name}</span>
          <span className="text-body-ink">{t(row.status === "blocked" ? "blocked" : "needsInput")}</span>
          <span className="ml-auto flex-none font-mono text-[11px] text-muted">{row.pane}</span>
          <a href="/tmux" className="flex-none rounded-full border border-line-strong px-2.5 py-0.5 text-[11px] font-semibold text-body-ink">
            {t("openTmux")}
          </a>
        </div>
      );
    }
    return (
      <div className="flex items-center gap-2.5 text-[13px]">
        <RowIcon kind="attention" />
        <span className="truncate font-semibold text-ink">{row.name}</span>
        <span className="text-body-ink">{t("needsInput")}</span>
        <span className="ml-auto flex-none font-mono text-[11px] text-muted" title={row.threadId}>{t("codexSession")}</span>
      </div>
    );
  }
  return (
    <div className="flex items-center gap-2.5 text-[13px]">
      <RowIcon kind="pr" />
      <span className="truncate font-semibold text-ink">{row.name}</span>
      <span className="text-body-ink">{t("prLabel", { number: row.number })} · {t(`ci.${row.ci}`)}</span>
      <a href={row.url} target="_blank" rel="noreferrer" className="ml-auto flex-none text-[11px] font-semibold text-brand">
        {t("openGithub")}
      </a>
    </div>
  );
}

function inboxRowKey(row: InboxRow): string {
  if (row.kind === "pr") return `pr-${row.dir}-${row.number}`;
  if (row.kind === "question") return `question-${row.dir}-${row.eventId}`;
  if (row.kind === "attention") return row.transport === "tmux"
    ? `attention-${row.dir}-${row.pane}-${row.tmuxIncarnation}`
    : `attention-${row.dir}-${row.threadId}`;
  return `gate-${row.dir}`;
}

export function InboxStrip({ mission, onExpand }: Props) {
  const t = useTranslations("mission.inbox");
  const tSource = useTranslations("mission.source");
  const { readiness, model } = mission;

  // Title AND the source-warning/skeleton branch below are both decided by the pure
  // `stripPresentation` helper (mission.ts, Finding 8) — every `readiness.kind` except `ready`
  // (`loading`, `failed`, and `degraded` — cold review round 4, Finding 2) reads the neutral
  // `titleUnknown` ("WAITING ON YOU", no count), never the confident `titleEmpty` ("NOTHING WAITS
  // ON YOU"). A failed `prs` envelope with an otherwise-empty inbox is exactly the case `degraded`
  // exists to name: it used to fall through all the way to `ready` and render the confident empty
  // title while GitHub was down (R3's fifth recurrence). `titleEmpty` is the ONLY branch that
  // renders it, gated on a plain two-clause check against ordinary data (Simplification 1,
  // Findings 3/5/8) — there is no type-level claim this string is reachable "only by" a
  // constructor; `degraded` not being `"ready"` is what actually keeps this branch out of reach.
  const presentation = stripPresentation(readiness, model);
  const cardsByDir = new Map(model.cards.map((c) => [c.dir, c]));
  const stuckCards = model.cards.filter((c) => c.headline === "stuck");

  // Round-1 F6: presentation.title.count (mission.ts:620, Part 1) folds in PR rows; Decision 8 excludes them — computed here from model.inbox instead of editing Part 1's own type.
  const needsYouCount = model.inbox.filter((row) => row.kind !== "pr").length;
  const baseTitle =
    presentation.title.kind === "empty"
      ? t("titleEmpty")
      : presentation.title.kind === "count"
        ? t("title", { count: needsYouCount })
        : t("titleUnknown");

  // Nothing waits on Rafa and no source warning applies — the confident empty state, the stuck
  // rail and the source/skeleton banner above all have nothing to say, so the strip itself stays
  // out of the layout instead of showing an empty card.
  // Still loading with nothing known yet: stay out of the layout too, instead of flashing an empty skeleton card.
  if (presentation.warning.kind === "skeleton" && stuckCards.length === 0) return null;
  if (presentation.title.kind === "empty" && stuckCards.length === 0 && presentation.warning.kind === "none") {
    return null;
  }

  return (
    <section className="flex flex-col gap-3.5 rounded-lg border border-brand-soft-border bg-surface px-5 py-[18px] shadow-[var(--glow-soft)]">
      <div className="flex items-center gap-3">
        <span className="text-sm font-bold text-ink">
          {needsYouCount === 0 && stuckCards.length > 0
            ? t("stuckTitle", { count: stuckCards.length })
            : `${baseTitle}${stuckCards.length > 0 ? t("stuckSuffix", { count: stuckCards.length }) : ""}`}
        </span>
      </div>
      {presentation.warning.kind === "source" ? (
        // Additional, never a replacement (Finding 5): `model.inbox` below still renders whatever
        // rows the OTHER, non-failed source(s) already resolved — a failed source no longer erases
        // known rows the way the old two-function design did. Also reached by a `degraded` prs
        // source with zero cards to carry the inline warning itself (Finding 3).
        <SourceWarning label={t("unavailable")} detail={presentation.warning.sources.map((s) => tSource(s)).join(", ")} />
      ) : presentation.warning.kind === "skeleton" ? (
        <div className="h-6 animate-pulse rounded bg-surface-2" />
      ) : null}
      {model.inbox.length > 0 ? (
        <div className="flex flex-col gap-2">
          {
            // The "question" key uses eventId, not pane — two tmux incarnations of the same pane
            // id can each have a pending question at once (cold review Finding 8). This plain //
            // comment sits before the map call rather than inside its returned JSX, since a JSX
            // comment can't be a bare sibling next to <InboxLine> inside a parenthesized (not
            // fragment-wrapped) arrow body (cold review Finding 4).
            model.inbox.map((row) => <InboxLine key={inboxRowKey(row)} row={row} card={cardsByDir.get(row.dir)} onExpand={onExpand} />)
          }
          {model.inboxPrRemainder > 0 ? (
            // Finding 5 (branch review, MEDIUM): the remainder is counted from each project's own
            // fetched PR sample — if any contributing project's fetch was itself truncated
            // (github.ts's PR-count cap), the true remainder can exceed this count. `prTruncatedAny`
            // picks the "at least" phrasing instead of claiming an exact number in that case.
            <span className="text-[11px] text-muted">
              {t(model.prTruncatedAny ? "prRemainderAtLeast" : "prRemainder", { count: model.inboxPrRemainder })}
            </span>
          ) : null}
        </div>
      ) : null}
      {stuckCards.length > 0 ? (
        <div className="flex flex-col gap-2 border-t border-line-subtle pt-3">
          {stuckCards.map((c) => (
            <div key={c.dir} className="flex items-center gap-2.5 text-[13px]">
              <span className="truncate font-semibold text-ink">{c.name}</span>
              <span className="text-body-ink">{t("stuckTitle", { count: 1 }).replace(/\s*·\s*1$/, "")}</span>
              <StuckRowActions card={c} />
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}
