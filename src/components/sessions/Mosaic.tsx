"use client";

import { useTranslations } from "next-intl";
import {
  mosaicGroups, mosaicPageState, selectSpotlightPane,
  type MosaicGroup, type MosaicPageState, type MosaicSpotlight,
} from "@/lib/mosaic";
import {
  composeIndexedReply, mosaicPaneAction, mosaicPaneState,
  type MissionCard, type MissionPane, type MissionPendingQuestion, type MissionReadiness, type MosaicPaneAction,
} from "@/lib/mission";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";

// Decision-6 bar: a plain flat element reusing the SAME animation NAMES globals.css already
// defines for `.tile-ring` (jax-pulse-border/jax-breathe) — the universal reduced-motion rule
// (globals.css:87-94, `*, *::before, *::after`) already covers any element using them, so no new
// CSS is needed (spec §10 row 27). `working` is deliberately steady (no comet on a 1px-tall bar).
const BAR_CLASS: Record<ReturnType<typeof mosaicPaneState>, string> = {
  "blocked-permission": "bg-danger-soft [animation:jax-pulse-border_1.1s_ease-in-out_infinite]",
  "waiting-for-input": "bg-warning-soft [animation:jax-breathe_2.6s_ease-in-out_infinite]",
  working: "bg-accent",
  "idle-stale": "bg-muted",
};

const CELL_BTN = "rounded-md border border-line px-2.5 py-1 text-[11.5px] font-medium disabled:cursor-not-allowed disabled:opacity-50";
// Round-3 F5: spec §10 D26's 44px minimum applies ONLY in the mobile full-screen dialog, never
// the dense grid cell — a `size` prop picks the class instead of duplicating the component.
const MOBILE_BTN = "min-h-11 min-w-11 rounded-md border border-line px-3 text-[13px] font-medium disabled:cursor-not-allowed disabled:opacity-50";

// Decision 17a: the mosaic's OWN small numbered-option renderer — posts the SAME body
// PendingActionButtons's single-question branch already builds (ProjectCard.tsx), never that
// card-scoped JSX component itself. Exported: Task 2's mobile dialog reuses it verbatim with
// `size="mobile"`.
export function MosaicAnswerOptions({
  pendingQuestion, busy, onAction, size = "compact",
}: { pendingQuestion: MissionPendingQuestion; busy: boolean; onAction: (url: string, body: unknown) => void; size?: "mobile" | "compact" }) {
  const btnClass = size === "mobile" ? MOBILE_BTN : CELL_BTN;
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {pendingQuestion.questions[0].options.map((opt, i) => (
        <button key={opt.label} type="button" className={btnClass} disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            onAction("/api/workflow/answer", { event_id: pendingQuestion.eventId, reply: composeIndexedReply(pendingQuestion, [[i + 1]]) });
          }}>
          {i + 1} · {opt.label}
        </button>
      ))}
    </div>
  );
}

// Round-4 F2: the options branch's own question prompt — shared by the spotlight cell
// (MosaicActionArea) and the mobile dialog (MosaicCellDetail), same as MosaicAnswerOptions above.
// Blank/whitespace-only text renders nothing rather than an empty bar.
export function MosaicQuestionText({ pendingQuestion }: { pendingQuestion: MissionPendingQuestion }) {
  const text = pendingQuestion.questions[0].question.trim();
  if (!text) return null;
  return <p className="max-h-9 overflow-hidden text-[11.5px] font-medium text-body-ink">{text}</p>;
}

function MosaicActionArea({
  action, busy, onAction, onOpen,
}: { action: MosaicPaneAction; busy: boolean; onAction: (url: string, body: unknown) => void; onOpen: (() => void) | null }) {
  const t = useTranslations("mission");
  if (action.kind === "options") {
    return (
      <div className="flex flex-col gap-1">
        <MosaicQuestionText pendingQuestion={action.pendingQuestion} />
        <MosaicAnswerOptions pendingQuestion={action.pendingQuestion} busy={busy} onAction={onAction} />
      </div>
    );
  }
  if (action.kind === "respond") {
    return (
      <button type="button" className={CELL_BTN} disabled={busy || !onOpen}
        onClick={(e) => { e.stopPropagation(); onOpen?.(); }}>
        {t("actions.respond")}
      </button>
    );
  }
  if (action.kind === "open-pane") return <span className="text-[11px] text-muted">{t("inbox.blocked")}</span>;
  return null; // kind === "none"
}

function MosaicCell({
  card, pane, spotlightHere, dim, lines, busy, onAction, onOpen,
}: {
  card: MissionCard; pane: MissionPane; spotlightHere: boolean; dim: boolean;
  lines: string[] | null; busy: boolean; onAction: (url: string, body: unknown) => void; onOpen: (() => void) | null;
}) {
  const t = useTranslations("sessions.mosaic");
  const state = mosaicPaneState(pane);
  const action = mosaicPaneAction(pane);
  const elapsedMs = pane.lastEventTs ? Date.parse(pane.lastEventTs) : NaN;
  const preview = lines && lines.length > 0 ? lines.slice(0, 4).join("\n") : t("noOutput");
  return (
    <div
      data-cell-pane={pane.pane}
      data-spotlight={spotlightHere ? "true" : undefined}
      role={onOpen ? "button" : undefined}
      tabIndex={onOpen ? 0 : undefined}
      onClick={onOpen ?? undefined}
      onKeyDown={onOpen ? (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onOpen(); } } : undefined}
      className={`flex min-h-11 lg:min-h-0 min-w-0 flex-col gap-1.5 rounded-lg border border-line bg-surface px-3 py-2.5 text-left ${spotlightHere ? "lg:col-span-2" : ""} ${dim ? "opacity-[.55]" : ""} ${onOpen ? "cursor-pointer" : ""}`}
    >
      <span className={`h-1 w-full flex-none rounded-full ${BAR_CLASS[state]}`} aria-hidden />
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 text-[11px] text-muted">
        <span className="truncate font-semibold text-body-ink">{card.name}</span>
        <span aria-hidden>·</span>
        <span className="font-mono">{pane.pane}</span>
        {card.builder ? (<><span aria-hidden>·</span><span>{card.builder}</span></>) : null}
        {Number.isFinite(elapsedMs) ? (<><span aria-hidden>·</span><RelativeTime epochMs={elapsedMs} /></>) : null}
      </div>
      <pre className="min-h-0 overflow-hidden whitespace-pre-wrap break-all font-mono text-[10.5px] leading-snug text-muted">{preview}</pre>
      <MosaicActionArea action={action} busy={busy} onAction={onAction} onOpen={onOpen} />
      <span className="mt-auto text-[10.5px] font-semibold">
        {onOpen ? <span className="text-body-ink">{t("openPane")}</span> : <span className="text-muted">{t("unknownSession")}</span>}
      </span>
    </div>
  );
}

function panesTailLookup(panesTail: MosaicProps["panesTail"], paneId: string): string[] | null {
  return panesTail?.find((p) => p.pane === paneId)?.lines ?? null;
}

export type MosaicProps = {
  readiness: MissionReadiness;
  cards: MissionCard[];
  panesTail: { pane: string; lines: string[] | null }[] | null;
  busy: boolean;
  onAction: (url: string, body: unknown) => void;
  onOpenDetail: (tmuxSession: string) => void;
  onOpenMobile: (pane: string) => void;
};

// Decision 19: the mobile flat list's own ordering — spotlight first, everything else in its
// original card/pane traversal order. Exported and direct-tested (no DOM-order assertion needed).
// MosaicGroup carries only `dir`/`name` (Part 1's own shape, not the full MissionCard) — the
// caller resolves the full MissionCard by `dir` when it needs more than that (Mosaic's own
// `cardByDir` map below).
export function mobileCellOrder(groups: MosaicGroup[], spotlight: MosaicSpotlight): { dir: string; pane: MissionPane }[] {
  const flat = groups.flatMap((g) => g.panes.map((pane) => ({ dir: g.dir, pane })));
  if (!spotlight) return flat;
  const idx = flat.findIndex((e) => e.dir === spotlight.dir && e.pane.pane === spotlight.pane.pane);
  if (idx <= 0) return flat;
  const [entry] = flat.splice(idx, 1);
  return [entry, ...flat];
}

export function Mosaic({ readiness, cards, panesTail, busy, onAction, onOpenDetail, onOpenMobile }: MosaicProps) {
  const t = useTranslations("sessions.mosaic");
  const tProjects = useTranslations("mission.projects");
  const tSource = useTranslations("mission.source");
  const groups = mosaicGroups(cards);
  const spotlight = selectSpotlightPane(groups);
  // MosaicGroup only carries `dir`/`name` — cells still need the full MissionCard (e.g.
  // `.builder`), so this map resolves it once per render instead of re-deriving grouping.
  const cardByDir = new Map(cards.map((c) => [c.dir, c]));
  // Decision 18: page state comes from Part 1's own fold, never re-derived from
  // readiness.kind/groups.length here.
  const pageState: MosaicPageState = mosaicPageState(readiness, groups);

  if (pageState === "failed" && readiness.kind === "failed") {
    return <SourceWarning label={tProjects("unavailable")} detail={readiness.failed.map((s) => tSource(s)).join(", ")} />;
  }
  if (pageState === "loading") {
    return (
      <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-3">
        {[0, 1, 2].map((i) => <div key={i} className="h-32 animate-pulse rounded-lg border border-line bg-surface" />)}
      </div>
    );
  }
  if (pageState === "empty") {
    return <p className="rounded-lg border border-line bg-surface px-5 py-8 text-sm text-muted">{t("empty")}</p>;
  }

  function isSpotlight(dir: string, paneId: string): boolean {
    return spotlight !== null && spotlight.dir === dir && spotlight.pane.pane === paneId;
  }

  return (
    <>
      {/* Decision 14/15: grouped grid with headers — desktop only. */}
      <div className="hidden flex-col gap-5 lg:flex">
        {groups.map((g) => (
          <div key={g.dir} className="flex flex-col gap-2">
            <span className="text-[13px] font-bold text-ink">{g.name}</span>
            <div className="grid grid-cols-[repeat(auto-fill,minmax(240px,1fr))] gap-3">
              {g.panes.map((pane) => {
                // Round-3 F3: a local `const` narrows correctly across the closure below; the
                // property access `pane.tmuxSession` does not, which is what motivated the `!`
                // this replaces.
                const session = pane.tmuxSession;
                return (
                  <MosaicCell
                    key={`${g.dir}:${pane.pane}:${pane.tmuxIncarnation}`}
                    card={cardByDir.get(g.dir)!} pane={pane}
                    spotlightHere={isSpotlight(g.dir, pane.pane)}
                    dim={spotlight !== null && !isSpotlight(g.dir, pane.pane)}
                    lines={panesTailLookup(panesTail, pane.pane)}
                    busy={busy} onAction={onAction}
                    onOpen={session !== null ? () => onOpenDetail(session) : null}
                  />
                );
              })}
            </div>
          </div>
        ))}
      </div>
      {/* Decision 19: flat single column, spotlight first, no group headers (every cell already
          shows its own project name in its header row). */}
      <div className="flex flex-col gap-3 lg:hidden">
        {mobileCellOrder(groups, spotlight).map(({ dir, pane }) => (
          <MosaicCell
            key={`${dir}:${pane.pane}:${pane.tmuxIncarnation}`}
            card={cardByDir.get(dir)!} pane={pane} spotlightHere={false} dim={false}
            lines={panesTailLookup(panesTail, pane.pane)}
            busy={busy} onAction={onAction}
            onOpen={pane.tmuxSession !== null ? () => onOpenMobile(pane.pane) : null}
          />
        ))}
      </div>
    </>
  );
}
