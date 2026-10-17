// Sessions mosaic — pure grouping/spotlight/page-state helpers (spec §5, §7 Decision 18, §8
// Decision 16). No JSX, no fetch: the mosaic PAGE (Part 2) calls these and renders their output;
// every branch here is unit-tested without a DOM.
import type { MissionCard, MissionPane, MissionReadiness } from "./mission";
import { mosaicPaneState } from "./mission";

export type MosaicGroup = { dir: string; name: string; panes: MissionPane[] };

// Spec §5 Decisions 1/2: cell = one LIVE tmux pane; a card with zero surviving panes gets no
// group at all (no placeholder cell) — reversing the mockup's own "project without pane" cell,
// per Rafa's D3. Filtering happens HERE, on MissionPane.live, before mosaicPaneState ever runs —
// a dead pane's stale attention state never reaches it (round-1 F1).
export function mosaicGroups(cards: MissionCard[]): MosaicGroup[] {
  const groups: MosaicGroup[] = [];
  for (const card of cards) {
    const panes: MissionPane[] = [];
    for (const s of card.sessions) {
      if (s.transport === "tmux") {
        for (const p of s.panes) if (p.live) panes.push(p);
      }
    }
    if (panes.length > 0) groups.push({ dir: card.dir, name: card.name, panes });
  }
  return groups;
}

export type MosaicSpotlight = { dir: string; pane: MissionPane } | null;

// Spec §8 Decision 16 (round-2 F3): the FIRST pane, in card/pane traversal order (the same order
// mosaicGroups above preserves from buildMission's own card/pane construction order), whose
// mosaicPaneState needs Rafa — "the oldest pending question wins", one spotlight app-wide,
// mirroring selectPendingAction's single-pending-interaction-per-card rule generalized across
// cards (mission.ts round-3 F4).
export function selectSpotlightPane(groups: MosaicGroup[]): MosaicSpotlight {
  for (const g of groups) {
    for (const p of g.panes) {
      const state = mosaicPaneState(p);
      if (state === "waiting-for-input" || state === "blocked-permission") return { dir: g.dir, pane: p };
    }
  }
  return null;
}

export type MosaicPageState = "loading" | "failed" | "empty" | "ready";

// Spec §7 Decision 18 (round-2 F4): page state comes from pillReadinessOf (Topbar.tsx:35),
// reused AS-IS by the caller — NEVER mission.readiness from buildMission, which stays pinned at
// kind:"loading" forever when `prs` is passed as the literal `undefined` the mosaic page uses
// (the exact trap Topbar.tsx's own comment documents for the identical call shape). This
// function only folds that readiness together with the ALREADY-GROUPED live-pane count into the
// one branch the page renders on; pillReadinessOf itself never returns "degraded" (it doesn't
// poll prs), so that MissionReadiness variant is simply never reached here.
export function mosaicPageState(readiness: MissionReadiness, groups: MosaicGroup[]): MosaicPageState {
  if (readiness.kind === "loading") return "loading";
  if (readiness.kind === "failed") return "failed";
  return groups.length === 0 ? "empty" : "ready";
}
