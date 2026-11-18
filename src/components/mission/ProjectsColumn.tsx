"use client";

import { useTranslations } from "next-intl";
import { useState } from "react";
import { applySavedOrder, layoutMap, type PrScope } from "@/lib/cockpitLayout";
import {
  beginSave, cancelOrganize, draftDrop, draftMove, draftWidth, enterOrganize,
  initialOrganizeState, reloadOrganize, saveConflicted, saveFailed, saveSucceeded,
  useCockpitSnapshot, useCockpitWrites, type OrganizeState,
} from "@/lib/cockpitPrefsQuery";
import { boardEntries, cardKey, hiddenBoardCards, sessionDisplay, type Mission, type MissionBucket, type MissionBucketSummary, type MissionCard } from "@/lib/mission";
import { PrefsButtons, ProjectCard } from "./ProjectCard";
import { renderBoardEntry } from "./boardEntry";
import { PitCounts } from "./PitCounts";
import { ProjectLayoutControls } from "./ProjectLayoutControls";
import { SourceWarning } from "./SourceWarning";

type Props = {
  mission: Mission;
  reposRoot: string;
  counts: Record<string, number> | null;
  // Optional: an existing test file calls this component with no `filter` at all —
  // undefined/null both mean "show every card" (Decision 3's click-to-filter).
  filter?: MissionBucket | null;
  expandedDir: string | null;
  onToggleExpand: (dir: string) => void;
  // M2 compact Projects controls: the removed PitCounts row's counts/filter live here now.
  bucketSummary?: MissionBucketSummary | null;
  onToggleFilter?: (bucket: MissionBucket) => void;
};

// Decision 17: the two summary segments' TEXT only — the "mostrar"/"gerenciar" disclosure
// toggles and the joining " · " between them are Part 2's JSX. `t` is the caller's OWN
// `useTranslations("mission.projects")` (already scoped, matching this component's existing
// `t("hiddenBar", ...)` call) — never re-prefixed with "projects." here.
// MOA-486 follow-up: dropped the autoHidden half's inline name list ("(Patient Insight Journal
// (PIJ))") — it wrapped badly on the phone. Both halves are now plain ICU-plural counts; names
// only ever show once a segment is expanded (the per-project rows below, unchanged).
export function hiddenAutoHiddenSummary(
  hiddenCount: number,
  autoHiddenCount: number,
  t: (key: string, values?: Record<string, unknown>) => string,
): { autoHidden: string | null; hidden: string | null } {
  const autoHidden = autoHiddenCount > 0 ? t("autoHiddenBar", { count: autoHiddenCount }) : null;
  const hidden = hiddenCount > 0 ? t("hiddenBar", { count: hiddenCount }) : null;
  return { autoHidden, hidden };
}

// Round-1 F9: pulled out of the component so the toggled-open branch has a direct, non-rendering test.
export function disclosureVisibleCards(showAuto: boolean, showHidden: boolean, autoHiddenCards: MissionCard[], hiddenCards: MissionCard[]): { auto: MissionCard[]; hidden: MissionCard[] } {
  return { auto: showAuto ? autoHiddenCards : [], hidden: showHidden ? hiddenCards : [] };
}

// Decision 17: one summary line with two local disclosure toggles; the per-project rows
// (PrefsButtons, unchanged) render below only when their own segment is revealed.
function ProjectVisibilityDisclosure({
  autoHiddenText, hiddenText, autoHiddenCards, hiddenCards,
}: { autoHiddenText: string | null; hiddenText: string | null; autoHiddenCards: MissionCard[]; hiddenCards: MissionCard[] }) {
  const t = useTranslations("mission");
  const [showAuto, setShowAuto] = useState(false);
  const [showHidden, setShowHidden] = useState(false);
  const visible = disclosureVisibleCards(showAuto, showHidden, autoHiddenCards, hiddenCards);
  return (
    <div className="flex flex-col gap-1.5 text-[11px] text-muted">
      <span>
        {autoHiddenText ? (
          <>
            {autoHiddenText}
            {" · "}
            {/* MOA-486 follow-up: "gerenciar" (below) only toggles the manually-hidden half
                (setShowHidden never touches showAuto/autoHiddenCards) — collapsed cards have no
                other reveal path, so this stays a separate lowercase toggle rather than folding
                into gerenciar. */}
            <button type="button" className="font-semibold text-body-ink underline" onClick={() => setShowAuto((s) => !s)}>{t("projects.autoHiddenShow")}</button>
          </>
        ) : null}
        {autoHiddenText && hiddenText ? " · " : null}
        {hiddenText ? (
          <>
            {hiddenText}
            {" · "}
            <button type="button" className="font-semibold text-body-ink underline" onClick={() => setShowHidden((s) => !s)}>{t("actions.manage")}</button>
          </>
        ) : null}
      </span>
      {visible.auto.map((c) => (
        <div key={c.dir} className="flex items-center justify-between gap-2">
          <span className="truncate text-body-ink">{c.name}</span>
          <PrefsButtons dir={c.dir} prefs={c.prefs} bucket="autoHidden" />
        </div>
      ))}
      {visible.hidden.map((c) => (
        <div key={cardKey(c)} className="flex items-center justify-between gap-2">
          <span className="truncate text-body-ink">{c.name}{c.scope?.kind === "session" ? ` · ${sessionDisplay(c.scope).text || t("board.newSession")}` : ""}</span>
          <PrefsButtons dir={c.scope?.kind === "session" ? cardKey(c) : c.dir} prefs={c.prefs} bucket="hidden" />
        </div>
      ))}
    </div>
  );
}

// Diff review F5: a pointer drop that actually changes order announces exactly what a
// keyboard/button move announces, at the resulting actual position. Reuses `draftDrop` (no
// duplicated insertion math); no-op drops, unknown identities, an absent draft or a pending save
// all leave state and live region untouched. Exported because it is the component's own drop
// decision and the only seam a DOM-less test can drive.
export function announceDrop(
  state: OrganizeState,
  cards: { dir: string; name: string }[],
  dragged: string,
  target: string,
  t: (key: string, values?: Record<string, unknown>) => string,
): { next: OrganizeState; announcement: string | null } {
  const next = draftDrop(state, dragged, target);
  if (next.draft === null || next.draft === state.draft) return { next: state, announcement: null };
  const position = next.draft.findIndex((e) => e.project === dragged);
  if (position < 0) return { next: state, announcement: null };
  const name = cards.find((c) => c.dir === dragged)?.name ?? dragged;
  return { next, announcement: t("organize.moved", { name, position: position + 1 }) };
}

export function ProjectsColumn({ mission, reposRoot, counts, filter = null, expandedDir, onToggleExpand, bucketSummary = null, onToggleFilter }: Props) {
  const t = useTranslations("mission.projects");
  const tMission = useTranslations("mission");
  const tSource = useTranslations("mission.source");
  const { readiness, model } = mission;

  // One polling preference snapshot (10s, shared with the page's PR query gate) plus the
  // Organize draft, which a poll never overwrites. Unavailable prefs leave the default
  // urgency/time/dir order and third widths in force, with Save/Reset disabled (plan F2).
  const snapshotQuery = useCockpitSnapshot();
  const writes = useCockpitWrites();
  const snapshot = snapshotQuery.data?.data ?? null;
  const preferencesUnavailable = snapshotQuery.isError && snapshot === null;
  const [organize, setOrganize] = useState(initialOrganizeState);
  const [scopeBusy, setScopeBusy] = useState(false);
  const [scopeError, setScopeError] = useState<string | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const layout = organize.draft ?? snapshot?.layout ?? null;
  const orderedCards = applySavedOrder(model.cards, layout);
  const widths = layoutMap(layout);
  const prScope: PrScope = snapshot?.prScope ?? "mine";

  function startOrganize() {
    if (!snapshot || organize.status === "saving") return;
    setOrganize(enterOrganize(snapshot, model.cards.filter((c) => c.visibility === "visible")));
    setAnnouncement("");
  }

  function moveProject(dir: string, delta: -1 | 1) {
    if (organize.status === "saving" || organize.draft === null) return;
    const index = organize.draft.findIndex((e) => e.project === dir);
    const next = index + delta;
    if (index < 0 || next < 0 || next >= organize.draft.length) return;
    const name = model.cards.find((c) => c.dir === dir)?.name ?? dir;
    setAnnouncement(t("organize.moved", { name, position: next + 1 }));
    setOrganize((state) => draftMove(state, dir, delta));
  }

  function dropProject(dragged: string, target: string) {
    const { next, announcement: message } = announceDrop(
      organize, model.cards, dragged, target, t as (key: string, values?: Record<string, unknown>) => string,
    );
    if (message === null) return;
    setAnnouncement(message);
    setOrganize(next);
  }

  async function persistLayout() {
    if (organize.status === "saving" || organize.draft === null || organize.baseRevision === null) return;
    const started = beginSave(organize);
    setOrganize(started);
    try {
      const result = await writes.saveLayout(started.baseRevision!, started.draft!);
      writes.apply(result.snapshot);
      setOrganize(result.conflict ? saveConflicted(started, result.snapshot) : saveSucceeded(result.snapshot));
    } catch (e) {
      setOrganize(saveFailed(started, e instanceof Error ? e.message : String(e)));
    }
  }

  async function resetLayoutOverride() {
    if (organize.status === "saving") return;
    // Diff review F4: a draft is bound to the revision it was built from (enterOrganize). Reset
    // while Organize is open must submit THAT revision — sending a newer polled revision would
    // silently overwrite the polled change instead of surfacing the expected conflict. Outside
    // Organize there is no draft, so the current snapshot revision is the correct base.
    const revision = organize.draft !== null ? organize.baseRevision : snapshot?.revision ?? null;
    if (revision === null) return;
    const started: typeof organize = organize.draft !== null ? beginSave(organize) : { ...organize, status: "saving" };
    setOrganize(started);
    try {
      const result = await writes.resetLayout(revision);
      writes.apply(result.snapshot);
      setOrganize(result.conflict ? saveConflicted(started, result.snapshot) : saveSucceeded(result.snapshot));
    } catch (e) {
      setOrganize(saveFailed(started, e instanceof Error ? e.message : String(e)));
    }
  }

  function cancelEditing() {
    const latest = snapshot ?? organize.baseline;
    if (latest) setOrganize(cancelOrganize(latest));
  }

  function reloadFromServer() {
    const latest = snapshot ?? organize.baseline;
    if (!latest) return;
    writes.apply(latest);
    setOrganize(reloadOrganize(latest, model.cards.filter((c) => c.visibility === "visible")));
  }

  async function chooseScope(scope: PrScope) {
    if (scope === prScope || scopeBusy) return;
    setScopeBusy(true);
    setScopeError(null);
    try {
      const next = await writes.savePrScope(scope);
      writes.apply(next);
    } catch (e) {
      setScopeError(e instanceof Error ? e.message : String(e));
    } finally {
      setScopeBusy(false);
    }
  }

  // `projects` is the one source that gates whether `model.cards` means anything at all (Task 7's
  // buildModel falls back to an empty scan when `projects` hasn't resolved) — `hub`/`prs` failing
  // or still loading never blanks this column, only individual card fields (Finding 5's principle,
  // applied here too).
  // Post-merge cold review, Finding 4 (MEDIUM): a failed BACKBONE source that ISN'T `projects`
  // (e.g. `hub` down while `projects` is still in flight, `pending` now names it — Task 7) used
  // to match neither branch below and fall through to `model.cards.length === 0`, which reads as
  // a genuine "no projects" empty state. `projects` unresolved — pending in either the `loading`
  // or the `failed` readiness — keeps the skeleton up regardless of which other source failed;
  // the empty state is reachable only once `projects` itself has actually resolved ok.
  const projectsUnresolved =
    (readiness.kind === "loading" && readiness.pending.includes("projects")) ||
    (readiness.kind === "failed" && readiness.pending.includes("projects"));

  // Phase 3 §10: non-visible cards never render in the main grid — each gets a per-project
  // management row below instead. Declared here (not in the else branch that builds `body`)
  // because the return JSX below consumes them too.
  const hiddenCards = hiddenBoardCards(model.cards);
  const autoHiddenCards = model.cards.filter((c) => c.visibility === "autoHidden");

  let body: React.ReactNode;
  if (readiness.kind === "failed" && readiness.failed.includes("projects")) {
    body = <SourceWarning label={t("unavailable")} detail={readiness.failed.map((s) => tSource(s)).join(", ")} />;
  } else if (projectsUnresolved) {
    body = (
      <>
        {readiness.kind === "failed" ? (
          <SourceWarning label={t("unavailable")} detail={readiness.failed.map((s) => tSource(s)).join(", ")} />
        ) : null}
        <div className="h-40 animate-pulse rounded-lg border border-line bg-surface" />
      </>
    );
  } else if (model.cards.length === 0) {
    body = (
      <p className="rounded-lg border border-line bg-surface px-5 py-8 text-sm text-muted">
        {t("empty")}
      </p>
    );
  } else {
    // Decision 3: the `working` filter also matches `waiting` cards, same fold
    // bucketOf() already applies to the count pills — so the filtered grid always matches the
    // pill's own count (spec §7). Groups (spec §5) show only matching cards and disappear when
    // none match; boardEntries owns both rules.
    const entries = boardEntries(orderedCards, filter);
    body = entries.length > 0 ? (
      // Rafa 2026-09-20: flex-wrap instead of the 1→2→3→4 column grid — at most 3 cards per row
      // (30% basis + gaps), and a partial row's cards grow to fill the line (2 cards = half each,
      // 1 = full). Tablet 2 per row, phone 1. The expanded card takes a 2-of-3 basis.
      <div className="project-grid">
        {entries.map((entry) => renderBoardEntry(entry, {
          expandedDir, onToggleExpand, counts, widths,
          organize: organize.draft !== null ? {
            draft: organize.draft,
            disabled: organize.status === "saving",
            onMove: moveProject,
            onWidth: (dir, width) => setOrganize((state) => draftWidth(state, dir, width)),
            onDrop: dropProject,
          } : undefined,
        }))}
      </div>
    ) : (
      // ponytail: reuses the same "empty" copy as the zero-projects case above rather
      // than a new "no cards match this filter" string — a minor wording imprecision,
      // not a new acceptance criterion; add a dedicated key if this reads confusingly
      // at visual validation.
      <p className="rounded-lg border border-line bg-surface px-5 py-8 text-sm text-muted">
        {t("empty")}
      </p>
    );
  }

  return (
    <div className="flex min-w-0 flex-col gap-3.5">
      <div className="flex items-center justify-between gap-2">
        <span className="text-sm font-bold text-ink">{t("title")}</span>
        <span className="font-mono text-[11px] text-muted">{reposRoot}</span>
      </div>
      <ProjectLayoutControls
        unavailable={preferencesUnavailable}
        organize={organize}
        onEnter={startOrganize}
        onSave={() => void persistLayout()}
        onCancel={cancelEditing}
        onReset={() => void resetLayoutOverride()}
        onReload={reloadFromServer}
        prScope={prScope}
        scopeBusy={scopeBusy}
        scopeError={scopeError}
        onScope={(scope) => void chooseScope(scope)}
      />
      <span aria-live="polite" className="sr-only">{announcement}</span>
      {bucketSummary && onToggleFilter && (bucketSummary.counts || filter !== null) ? (
        // M2 plan F2: the permanent four-pill row is one compact native disclosure, folded by
        // default. The pills (same PitCounts state/callback/counts/pressed semantics) live inside;
        // the clear action is a sibling of the summary, visible while folded and independent of
        // bucket-count availability, so it still clears a selected status whose counts are unknown.
        <div className="flex min-w-0 flex-wrap items-center gap-2 text-[11px] text-muted">
          <details className="min-w-0">
            <summary className="cursor-pointer list-none font-semibold text-body-ink [&::-webkit-details-marker]:hidden">
              {t("controls.filter")}: {filter !== null ? tMission(`states.${filter}`) : t("controls.filterAll")}
            </summary>
            {bucketSummary.counts ? (
              <div className="mt-1.5">
                <PitCounts
                  counts={bucketSummary.counts}
                  active={filter}
                  onToggle={onToggleFilter}
                  hiddenCount={model.hiddenCount}
                  autoHiddenCount={model.autoHiddenCount}
                  compact
                />
              </div>
            ) : null}
          </details>
          {filter !== null ? (
            <button type="button" className="rounded border border-line px-1.5 py-0.5 font-semibold text-body-ink" onClick={() => onToggleFilter(filter)}>
              {t("controls.filterClear")}
            </button>
          ) : null}
        </div>
      ) : null}
      {(() => {
        const summary = hiddenAutoHiddenSummary(hiddenCards.length, autoHiddenCards.length, t as (key: string, values?: Record<string, unknown>) => string);
        if (!summary.hidden && !summary.autoHidden) return null;
        return (
          <ProjectVisibilityDisclosure autoHiddenText={summary.autoHidden} hiddenText={summary.hidden} autoHiddenCards={autoHiddenCards} hiddenCards={hiddenCards} />
        );
      })()}
      {/* MOA-469 §4 / C4: the two live sources warn independently — a failed tmux collector is a
          visible source warning (never a quiet healthy zero-agents board), and a failed Codex
          collector must not hide the (fully known) tmux/projects state. */}
      {model.tmuxSource === "failed" ? (
        <SourceWarning label={t("tmuxUnavailable")} detail={tSource("tmux")} />
      ) : null}
      {model.codexSource === "failed" ? (
        <SourceWarning label={t("codexUnavailable")} detail={tSource("codex")} />
      ) : model.codexSource === "truncated" ? (
        <span className="text-[11px] text-muted">{t("codexTruncated")}</span>
      ) : null}
      {body}
      {model.skipped > 0 ? (
        <span className="text-[11px] text-muted">{t("skipped", { count: model.skipped })}</span>
      ) : null}
      {model.uncardedProjectNames.length > 0 ? (
        <>
          {/* MOA-486 follow-up: the full name list used to render inline every time, three lines
              of noise below the cards. A native <details>/<summary> collapses it behind a toggle
              — no state, no JS; the names stay in the collapsed markup (just not visible) so the
              test can assert on them without simulating a click. */}
          <details className="group text-[11px] text-muted">
            <summary className="cursor-pointer list-none [&::-webkit-details-marker]:hidden">
              {t("uncarded", { count: model.uncardedProjectNames.length })}
              {" · "}
              <span className="font-semibold text-body-ink underline group-open:hidden">{t("uncardedShow")}</span>
              <span className="hidden font-semibold text-body-ink underline group-open:inline">{t("uncardedHide")}</span>
            </summary>
            <span>{t("uncardedNames", { names: model.uncardedProjectNames.join(", ") })}</span>
          </details>
          {model.historicalTruncated ? (
            // Post-merge cold review, Finding 2 (HIGH): the merged hub envelope reports
            // `historicalTruncated` when it capped the historical project set — surfacing it here
            // (rather than silently dropping it, as the earlier draft did) keeps the uncarded line
            // from implying it lists every project with activity, when it may only list the capped
            // subset.
            <span className="text-[11px] text-muted">{t("uncardedTruncated")}</span>
          ) : null}
        </>
      ) : model.historicalTruncated ? (
        // Finding 4 (branch review, MEDIUM): `historicalTruncated` used to render this "and
        // others" continuation copy even when `uncardedProjectNames` is empty because `projects`
        // hasn't resolved yet (mission.ts gates the name list on `projectsReady`, not on the
        // truncation flag) — it read as a continuation of a list that was never shown. Standalone
        // wording here says the same thing without implying a preceding list.
        <span className="text-[11px] text-muted">{t("hubListTruncated")}</span>
      ) : null}
    </div>
  );
}
