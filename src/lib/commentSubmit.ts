import type { WriteGuidance, WriteOutcome } from "@/components/kanban/PropertiesSidebar";
import type { IssueComment } from "@/server/collectors/linear";

// Identity of the exact comment text the user asked to send. A later draft
// (newer text, same issue) must never be cleared by the older request's
// success, and a stale request for a closed/switched issue must not mutate
// the new overlay's composer.
export type CommentDraft = { issueId: string; body: string; revision: number };

export function draftMatches(a: CommentDraft, b: CommentDraft): boolean {
  return a.issueId === b.issueId && a.body === b.body && a.revision === b.revision;
}

export type CommentSendResult = { outcome: WriteOutcome; current: boolean };

// The overlay's real submission path: reserved pending guard (one in-flight
// request, duplicate clicks answered with null), then the outcome is returned
// together with whether the draft the user sees now is still the sent draft.
// Clearing/completing is the caller's call — this helper never mutates UI.
export function createCommentSender(post: (issueId: string, body: string) => Promise<WriteOutcome>) {
  let inflight = false;
  return {
    isPending: () => inflight,
    async send(opts: { draft: CommentDraft; getCurrent: () => CommentDraft }): Promise<CommentSendResult | null> {
      if (inflight || !opts.draft.body.trim()) return null;
      inflight = true;
      try {
        const outcome = await post(opts.draft.issueId, opts.draft.body);
        return { outcome, current: draftMatches(opts.draft, opts.getCurrent()) };
      } finally {
        inflight = false;
      }
    },
  };
}

// Comment-specific reconciliation (phase 6 spec §6.3, F2): unlike
// reconcileWrite, a refusal removes ONLY the fabricated temp entry — never a
// full snapshot restore, which would also discard a second comment or draft
// added while the first send was in flight. Every other outcome leaves the
// list untouched; the next successful comments refetch replaces the whole
// array with the server's truth.
export function reconcileComment(list: IssueComment[], tempId: string, outcome: WriteGuidance | null): IssueComment[] {
  if (outcome?.kind === "refused") return list.filter((c) => c.id !== tempId);
  return list;
}