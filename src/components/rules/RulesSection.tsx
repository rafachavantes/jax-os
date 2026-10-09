"use client";

import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useTranslations } from "next-intl";
import { useEffect, useState } from "react";
import { fetchEnvelope } from "@/lib/api";
import { RelativeTime } from "@/components/RelativeTime";
import { compose, norm, ruleOutline, slotVisible, type AppKey, type RuleDocs } from "@/lib/rules";
import { agentsOf, useGeneralSettings } from "@/lib/settingsQuery";
import {
  write,
  writeGuidance,
  type Translator,
  type WriteOutcome,
} from "@/components/kanban/PropertiesSidebar";
import { SourceWarning } from "@/components/mission/SourceWarning";
import type { RulesData } from "@/server/collectors/rules"; // type-only
import { RuleDiffModal } from "./RuleDiffModal";
import css from "./rules.module.css";

// rules-namespace guidance for the shared classified write outcome
function ruleWriteText(t: Translator, outcome: WriteOutcome): string {
  const g = writeGuidance(outcome);
  if (!g) return t("writeRefused", { error: "unavailable" });
  if (g.kind === "refused") return t("writeRefused", { error: g.error });
  if (g.kind === "unconfirmed") return t("writeUnconfirmed");
  return t("writeAppliedUnrecorded");
}

// operation identity of a rule settlement, so a dialog only ever shows the
// outcome of its own operation, never a prior one's error.
export type RuleSlot = keyof RuleDocs; // "global" | "claude" | "codex" | "opencode"
export type RuleStatusOp = `edit:${RuleSlot}` | `apply:${AppKey}`;

// section-owned live outcome of the last rule write/apply — lives here, not in
// the dialogs, so feedback survives modal close.
export type RuleStatus = { tone: "pending" | "success" | "error"; text: string; op: RuleStatusOp } | null;

// dialog-scoped view of the section-owned settlement: the open modal may only
// render an outcome of the SAME operation.
export function ruleStatusForOp(
  status: RuleStatus,
  op: RuleStatusOp | null,
): Exclude<RuleStatus, null> | null {
  return status && status.op === op ? status : null;
}

// production rule-write settlement: the promise runs in the section's scope, so
// every settlement is reported through onStatus even if a dialog unmounts
// mid-flight. Confirmed ok alone calls onSuccess; refused / unconfirmed /
// applied-unrecorded become an error status and keep the dialog open.
export async function settleRuleWrite(opts: {
  op: RuleStatusOp;
  post: () => Promise<WriteOutcome>;
  t: Translator;
  pendingText: string;
  successText: string;
  onStatus: (status: Exclude<RuleStatus, null>) => void;
  onSuccess: () => void;
}): Promise<string | null> {
  opts.onStatus({ tone: "pending", text: opts.pendingText, op: opts.op });
  const outcome = await opts.post();
  if (!outcome.ok) {
    const text = ruleWriteText(opts.t, outcome);
    opts.onStatus({ tone: "error", text, op: opts.op });
    return text;
  }
  opts.onSuccess();
  opts.onStatus({ tone: "success", text: opts.successText, op: opts.op });
  return null;
}

// ---- Task 2: pure workspace derivations (unit/SSR tested) ----

export type RuleDraft = { base: string; baseRevision: string; text: string };

// Stable editing range (F7): captured once at edit start and extended by each
// keystroke. Never re-derived from the edited outline, so typing a literal
// `## ` heading inside a section can't shrink/relocate the visible buffer.
export type EditRange = { start: number; end: number; text: string };

export function seedEditRange(base: string, spanIndex: number | null): EditRange | null {
  const outline = ruleOutline(base);
  const span = spanIndex !== null ? (outline.spans[spanIndex] ?? null) : null;
  if (spanIndex !== null && !span) return null;
  return { start: span ? span.start : 0, end: span ? span.end : base.length, text: span ? span.original : base };
}

export function extendEditRange(range: EditRange, value: string): EditRange {
  return { start: range.start, end: range.start + value.length, text: value };
}

export function spliceDraft(draft: string, range: EditRange, value: string): string {
  return draft.slice(0, range.start) + value + draft.slice(range.end);
}

export function savedSourceOf(data: RulesData, slot: RuleSlot): string {
  return slot === "global" ? data.canonical : data.exceptions[slot];
}

// The exact bytes a source save submits: the global document always
// normalizes; an empty exception is the "no exception" sentinel and stays "".
export function submittedSource(slot: RuleSlot, text: string): string {
  if (slot === "global") return norm(text);
  return text === "" ? "" : norm(text);
}

// A confirmed save becomes the new baseline only when a fresh GET returns the
// exact submitted bytes; a newer local edit (the user typed on) is retained,
// and any other content is a visible conflict — never an inferred revision.
// The draft is compared against the RAW submitted text, not the normalized
// server bytes: a missing final newline is not a newer edit.
export function acceptConfirmedBaseline(
  saved: string,
  submitted: string,
  rawSubmitted: string,
  currentDraft: string,
): "accept" | "keep-newer" | "conflict" {
  if (saved !== submitted) return "conflict";
  return currentDraft !== rawSubmitted ? "keep-newer" : "accept";
}

export function savedRevisionOf(data: RulesData, slot: RuleSlot): string {
  return data.sourceRevisions[slot];
}

export function isDraftDirty(draft: RuleDraft | undefined): boolean {
  return !!draft && draft.text !== draft.base;
}

// A dirty draft whose base revision no longer matches the live source: the
// source changed under it, so it must not be merged or saved blindly.
export function isDraftStale(data: RulesData, slot: RuleSlot, draft: RuleDraft | undefined): boolean {
  return isDraftDirty(draft) && draft!.baseRevision !== savedRevisionOf(data, slot);
}

// The GET payload carries four source revisions; the apply route accepts
// exactly the two-key {global, exception} pair.
export function narrowApplyRevisions(data: RulesData, app: AppKey): { global: string; exception: string } {
  return { global: data.sourceRevisions.global, exception: data.sourceRevisions[app] };
}

export type ApplyPreview = {
  app: AppKey;
  canonical: string;
  exception: string;
  revisions: { global: string; exception: string };
  disk: string;
  diskHash: string;
  missing: boolean;
  path: string;
};

// Fixes the preview to SAVED state only (never an editor draft) and to the
// exact two source revisions the apply route will re-check. An unreadable
// destination cannot open a preview at all (null).
export function captureApplyPreview(data: RulesData, app: AppKey): ApplyPreview | null {
  const info = data.apps[app];
  if ("error" in info) return null;
  return {
    app,
    canonical: data.canonical,
    exception: data.exceptions[app],
    revisions: narrowApplyRevisions(data, app),
    disk: info.disk ?? "",
    diskHash: info.diskHash,
    missing: info.status === "missing",
    path: info.path,
  };
}

// MOA-504 D10: an apply preview whose agent was switched off must not stay open (the server guard
// would refuse the write, but the destination and the Apply button would still be on screen).
export function previewAfterAgents<P extends { app: AppKey }>(p: P | null, agents: Record<AppKey, boolean>): P | null {
  return p && !agents[p.app] ? null : p;
}

const APP_ORDER: AppKey[] = ["claude", "codex", "opencode"];
const STATUS_TAG: Record<"synced" | "drifted" | "missing", "tagSuccess" | "tagWarning" | "tag"> = {
  synced: "tagSuccess",
  drifted: "tagWarning",
  missing: "tag",
};

export function RulesSection() {
  const t = useTranslations("rules");
  const queryClient = useQueryClient();
  const q = useQuery({
    queryKey: ["rules"],
    queryFn: ({ signal }) => fetchEnvelope<RulesData>("/api/rules", signal),
    refetchInterval: 60_000,
  });
  const [slot, setSlot] = useState<RuleSlot>("global");
  const [spanIndex, setSpanIndex] = useState<number | null>(null);
  const [drafts, setDrafts] = useState<Partial<Record<RuleSlot, RuleDraft>>>({});
  const [editing, setEditing] = useState(false);
  const [editRange, setEditRange] = useState<(EditRange & { slot: RuleSlot; span: number | null }) | null>(null);
  const [status, setStatus] = useState<RuleStatus>(null);
  const [saving, setSaving] = useState(false);
  const [preview, setPreview] = useState<ApplyPreview | null>(null);
  const [applying, setApplying] = useState(false);
  // MOA-504 D10: slots and destinations follow the agent switches (fail open while unknown); if the
  // open slot's agent is switched off, fall back to the global document, and an open apply preview
  // (RuleDiffModal) of that agent is closed so its destination and Apply button disappear too.
  const agents = agentsOf(useGeneralSettings().data);
  const visibleApps = APP_ORDER.filter((app) => agents[app]);
  useEffect(() => {
    if (!slotVisible(slot, agents)) {
      setSlot("global");
      setSpanIndex(null);
    }
    setPreview((p) => previewAfterAgents(p, agents));
  }, [slot, agents]);

  const data = q.data?.ok ? q.data.data : null;
  const failed = q.isError && !data;

  function refetch() {
    queryClient.invalidateQueries({ queryKey: ["rules"] });
  }

  if (failed) {
    return (
      <section className="flex flex-col gap-2">
        <h2 className="font-display text-[15px] font-black text-ink">{t("sectionTitle")}</h2>
        <SourceWarning label={t("unavailable")} detail={q.error instanceof Error ? q.error.message : undefined} />
      </section>
    );
  }
  if (!data) return <div className="h-16 animate-pulse rounded bg-surface-2" />;

  const saved = savedSourceOf(data, slot);
  const draft = drafts[slot];
  const effective = draft ? draft.text : saved;
  const dirty = isDraftDirty(draft);
  const stale = isDraftStale(data, slot, draft);
  const outline = ruleOutline(effective);
  const activeSpan = spanIndex !== null ? (outline.spans[spanIndex] ?? null) : null;
  // While editing, the visible buffer is the stable captured range — NOT a span
  // re-derived from the edited outline on every keystroke.
  const bufferActive = editing && editRange !== null && editRange.slot === slot && editRange.span === spanIndex;
  const displayText = bufferActive ? editRange!.text : activeSpan ? activeSpan.original : effective;

  function seedFor(nextSlot: RuleSlot, nextSpan: number | null): (EditRange & { slot: RuleSlot; span: number | null }) | null {
    const base = drafts[nextSlot] ? drafts[nextSlot]!.text : savedSourceOf(data!, nextSlot);
    const range = seedEditRange(base, nextSpan);
    return range ? { ...range, slot: nextSlot, span: nextSpan } : null;
  }

  function select(nextSlot: RuleSlot, nextSpan: number | null) {
    setSlot(nextSlot);
    setSpanIndex(nextSpan);
    if (editing) setEditRange(seedFor(nextSlot, nextSpan));
  }

  function startEdit() {
    if (stale) return;
    if (!draft) setDrafts((prev) => ({ ...prev, [slot]: { base: saved, baseRevision: savedRevisionOf(data!, slot), text: saved } }));
    setEditRange(seedFor(slot, spanIndex));
    setEditing(true);
  }

  function discard() {
    setDrafts((prev) => {
      const next = { ...prev };
      delete next[slot];
      return next;
    });
    setEditRange(null);
    setEditing(false);
  }

  function onEditChange(value: string) {
    const range = bufferActive ? editRange : null;
    // Preserve the editing identity (slot/span) alongside the extended
    // text/range: a type assertion is not a runtime guarantee, and losing them
    // makes the next keystroke take the whole-document replacement branch.
    if (range) setEditRange({ ...range, ...extendEditRange(range, value) });
    setDrafts((prev) => {
      const current = prev[slot] ?? { base: saved, baseRevision: savedRevisionOf(data!, slot), text: saved };
      const text = range ? spliceDraft(current.text, range, value) : value;
      return { ...prev, [slot]: { ...current, text } };
    });
  }

  async function saveSource() {
    const d = drafts[slot];
    if (!d || d.text === d.base || saving || stale) return;
    const op: RuleStatusOp = `edit:${slot}`;
    // The raw draft is what the editor holds; `submitted` is the normalized
    // bytes the server is expected to store.
    const rawSubmitted = d.text;
    const submitted = submittedSource(slot, rawSubmitted);
    const url = slot === "global" ? "/api/rules/canonical" : "/api/rules/exception";
    const body =
      slot === "global"
        ? { content: rawSubmitted, expectedRevision: d.baseRevision }
        : { app: slot, content: rawSubmitted, expectedRevision: d.baseRevision };
    setSaving(true);
    setStatus({ tone: "pending", text: t("saving"), op });
    const outcome = await write(url, body);
    if (!outcome.ok) {
      setStatus({ tone: "error", text: ruleWriteText(t, outcome), op });
      setSaving(false);
      return;
    }
    // Confirmed. Re-read the source and accept the new baseline/revision only
    // when the saved bytes match what was submitted; otherwise stay in a
    // visible conflict state instead of inferring a revision.
    let freshData: RulesData | null = null;
    try {
      const res = await q.refetch();
      if (res.data?.ok) freshData = res.data.data;
    } catch {
      freshData = null;
    }
    setSaving(false);
    if (freshData && savedSourceOf(freshData, slot) === submitted) {
      const freshRevision = savedRevisionOf(freshData, slot);
      const freshSaved = savedSourceOf(freshData, slot);
      setDrafts((prev) => {
        const cur = prev[slot] ?? d;
        const verdict = acceptConfirmedBaseline(freshSaved, submitted, rawSubmitted, cur.text);
        return {
          ...prev,
          [slot]: {
            base: submitted,
            baseRevision: freshRevision,
            // a genuinely newer local edit is retained on the new baseline
            text: verdict === "keep-newer" ? cur.text : submitted,
          },
        };
      });
      setEditing(false);
      setEditRange(null);
      setStatus({ tone: "success", text: t("saved"), op });
      return;
    }
    setStatus({ tone: "error", text: t("sourceConflict"), op });
  }

  function openPreview(app: AppKey) {
    const snapshot = captureApplyPreview(data!, app);
    if (snapshot) setPreview(snapshot);
  }

  async function applyPreview() {
    const p = preview;
    if (!p || applying) return;
    setApplying(true);
    await settleRuleWrite({
      op: `apply:${p.app}`,
      post: () => write("/api/rules/apply", { app: p.app, diskHash: p.diskHash, applySourceRevisions: p.revisions }),
      t,
      pendingText: t("applying"),
      successText: t("applied"),
      onStatus: setStatus,
      onSuccess: () => {
        setPreview(null);
        refetch();
      },
    });
    setApplying(false);
  }

  const previewStale =
    preview !== null &&
    (data.sourceRevisions.global !== preview.revisions.global ||
      data.sourceRevisions[preview.app] !== preview.revisions.exception);
  const statusError = status && status.tone === "error";

  function renderSlotButton(buttonSlot: RuleSlot, span: number | null, label: string) {
    const selected = slot === buttonSlot && spanIndex === span;
    const marker = isDraftDirty(drafts[buttonSlot]) ? " ·" : "";
    return (
      <button
        key={`${buttonSlot}:${span ?? "doc"}`}
        type="button"
        aria-pressed={selected}
        onClick={() => select(buttonSlot, span)}
        className={css.ruleChoice}
      >
        {label}
        {marker}
      </button>
    );
  }

  const globalSectionButtons = slot === "global" && !outline.fullDocumentOnly
    ? outline.spans.map((s, i) => renderSlotButton("global", i, s.label))
    : [];

  return (
    <section className="flex flex-col">
      {q.isError ? (
        <SourceWarning
          label={t("unavailable")}
          aside={q.dataUpdatedAt ? <RelativeTime epochMs={q.dataUpdatedAt} /> : undefined}
        />
      ) : null}

      <div className={css.heading}>
        <div>
          <h2>{t("heading")}</h2>
          <p>{t("headingSubtitle")}</p>
        </div>
      </div>

      <div className={css.processStrip}>
        <span>
          <span className={css.step}>1</span>
          {t("processEdit")}
        </span>
        <span>→</span>
        <span>
          <span className={css.step}>2</span>
          {t("processSave")}
        </span>
        <span>→</span>
        <span>
          <span className={css.step}>3</span>
          {t("processApply")}
        </span>
      </div>

      <div className={`${css.card} ${css.workspace}`}>
        <nav className={css.menu} aria-label={t("sectionTitle")}>
          <div className={css.menuEyebrow}>{t("groupGlobal")}</div>
          {renderSlotButton("global", null, t("sectionTitle"))}
          {globalSectionButtons}
          <div className={css.menuEyebrow}>{t("groupExceptions")}</div>
          {visibleApps.flatMap((app) => {
            const items = [renderSlotButton(app, null, t(`apps.${app}`))];
            if (slot === app && !outline.fullDocumentOnly) {
              for (let i = 0; i < outline.spans.length; i++) items.push(renderSlotButton(app, i, outline.spans[i].label));
            }
            return items;
          })}
        </nav>

        <article className={css.ruleMain}>
          <div className={css.heading}>
            <div>
              <div className={css.eyebrow}>
                {slot === "global" ? t("scopeGlobal") : t("scopeException", { app: t(`apps.${slot}`) })}
              </div>
              <h2 className={css.ruleTitle}>{activeSpan ? activeSpan.label : slot === "global" ? t("sectionTitle") : t(`apps.${slot}`)}</h2>
            </div>
            {!editing ? (
              <button type="button" className={css.secondaryButton} onClick={startEdit} disabled={stale}>
                {t("edit")}
              </button>
            ) : null}
          </div>
          <p className={css.hint}>
            {slot === "global" ? t("contextGlobal") : t("contextException")}
          </p>

          {editing ? (
            <textarea
              className={css.editor}
              value={displayText}
              onChange={(e) => onEditChange(e.target.value)}
              aria-label={t("contentLabel")}
            />
          ) : (
            <pre className={css.reader}>{displayText || (slot === "global" ? "" : t("emptyException"))}</pre>
          )}

          {dirty ? <p className={css.dirtyTag}>{t("dirtyWarning")}</p> : null}
          {stale ? (
            <p role="status" aria-live="polite" className={css.statusError}>
              {t("staleDraft")}
            </p>
          ) : null}

          <div className={css.actions}>
            {editing ? (
              <>
                <button
                  type="button"
                  className={css.primaryButton}
                  onClick={saveSource}
                  disabled={saving || !dirty || stale}
                >
                  {saving ? t("saving") : t("saveSource")}
                </button>
                <button type="button" className={css.secondaryButton} onClick={discard} disabled={saving}>
                  {t("discard")}
                </button>
              </>
            ) : null}
            {status ? (
              <p
                role="status"
                aria-live="polite"
                className={statusError ? css.statusError : status.tone === "success" ? css.statusSuccess : css.statusPending}
              >
                {status.text}
              </p>
            ) : (
              <span className={css.hint}>{dirty ? t("noteDraft") : t("noteSaved")}</span>
            )}
          </div>
        </article>
      </div>

      <div className={css.destinations}>
        <div className={css.heading}>
          <div>
            <h2>{t("destHeading")}</h2>
            <p>{t("destSubtitle")}</p>
          </div>
        </div>
        <div className={css.destinationGrid}>
          {visibleApps.map((app) => {
            const info = data.apps[app];
            if ("error" in info) {
              return (
                <article key={app} className={`${css.card} ${css.destination}`}>
                  <div className={css.destTitle}>
                    <h3>{t(`apps.${app}`)}</h3>
                    <span className={css.tag}>{t("appError")}</span>
                  </div>
                  <code className={css.destCode}>{info.path}</code>
                  <p className={css.destBody}>{t("destUnreadable")}</p>
                </article>
              );
            }
            const synced = info.status === "synced";
            const tagClass = STATUS_TAG[info.status];
            return (
              <article key={app} className={`${css.card} ${css.destination}`}>
                <div className={css.destTitle}>
                  <h3>{t(`apps.${app}`)}</h3>
                  <span className={`${css.tag} ${tagClass === "tag" ? "" : css[tagClass]}`}>
                    {t(`status.${info.status}`)}
                  </span>
                </div>
                <code className={css.destCode}>{info.path}</code>
                <p className={css.destBody}>
                  {info.status === "missing"
                    ? t("destBodyMissing")
                    : synced
                      ? t("destBodySynced")
                      : t("destBodyPending")}
                </p>
                <button type="button" className={css.secondaryButton} onClick={() => openPreview(app)}>
                  {synced ? t("viewDoc") : t("reviewApply")}
                </button>
              </article>
            );
          })}
        </div>
      </div>

      {preview ? (
        <RuleDiffModal
          appLabel={t(`apps.${preview.app}`)}
          expected={compose(preview.exception, preview.canonical)}
          disk={preview.disk}
          missing={preview.missing}
          stale={previewStale}
          path={preview.path}
          applyLabel={preview.missing ? t("createApply", { app: t(`apps.${preview.app}`) }) : t("applyTo", { app: t(`apps.${preview.app}`) })}
          status={ruleStatusForOp(status, `apply:${preview.app}`)}
          onApply={applyPreview}
          onClose={() => {
            if (!applying) setPreview(null);
          }}
        />
      ) : null}
    </section>
  );
}
