// src/lib/mosaicView.ts
import type { MissionCard, MissionPane } from "./mission";

// Round-1: src/app/tmux/page.tsx (a Next.js page module) may export ONLY its default component
// (next-page-modules-reject-named-exports) — this is the one presentation-only lookup that plan
// needs unit-tested, so it lives here instead (task assignment's one allowed exception). Scans
// every LIVE-or-not tmux pane on the board, not just mosaicGroups' filtered output, so a dialog
// opened right before its pane's own liveness flips still resolves for the render that closes it.
// Round-3 F3: only ever resolves a pane whose `tmuxSession` is non-null — the mobile dialog must
// never target a null session, and returning it as a separate required `session: string` field
// (instead of the caller re-reading `pane.tmuxSession!`) makes that a type-level guarantee, not
// a runtime assertion. A pane whose `tmuxSession` goes null between renders (a background
// refresh) simply stops matching here on the next render.
export function findMobilePane(cards: MissionCard[], paneId: string | null): { card: MissionCard; pane: MissionPane; session: string } | null {
  if (paneId === null) return null;
  for (const card of cards) {
    for (const s of card.sessions) {
      if (s.transport !== "tmux") continue;
      const pane = s.panes.find((p) => p.pane === paneId);
      if (pane && pane.tmuxSession !== null) return { card, pane, session: pane.tmuxSession };
    }
  }
  return null;
}
