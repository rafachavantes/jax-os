"use client";

import { useState } from "react";
import type { KanbanIssue, KanbanState } from "@/server/collectors/linear";
import { write, writeGuidance, type WriteGuidance, type WriteOutcome } from "@/components/kanban/PropertiesSidebar";
import { IssueCard } from "./IssueCard";

// production board-move handler behind the drop path: same-column drop is a
// silent no-op, a per-issue synchronous reservation admits exactly one POST
// for same-turn duplicate drops, and the board always reinvalidates on
// settlement so the UI converges on Linear's truth. Exported so the exact
// production path is exercised without a DOM.
export async function moveIssue(opts: {
  issueId: string;
  issue: { stateId: string } | undefined;
  targetStateId: string;
  reservation: Set<string>;
  post: () => Promise<WriteOutcome>;
  onPending: (pending: boolean) => void;
  onGuidance: (guidance: WriteGuidance | null) => void;
  onInvalidate: () => void;
}): Promise<void> {
  if (!opts.issue || opts.issue.stateId === opts.targetStateId) return; // same-column drop: no-op, no log noise
  if (opts.reservation.has(opts.issueId)) return; // per-issue synchronous guard: one request
  opts.reservation.add(opts.issueId);
  opts.onPending(true);
  opts.onGuidance(null);
  try {
    const outcome = await opts.post();
    opts.onGuidance(writeGuidance(outcome));
  } finally {
    opts.reservation.delete(opts.issueId);
    opts.onPending(false);
    // success or failure, converge on Linear's truth
    opts.onInvalidate();
  }
}

type Props = {
  state: KanbanState;
  issues: KanbanIssue[];
  onDropIssue?: (issueId: string, stateId: string) => void;
  onOpenIssue?: (issue: KanbanIssue) => void;
};

export function KanbanColumn({ state, issues, onDropIssue, onOpenIssue }: Props) {
  const [over, setOver] = useState(false);

  return (
    <div
      className={`flex min-w-[226px] flex-col gap-[11px] rounded-lg p-1 transition-colors ${
        over ? "bg-brand-soft" : ""
      }`}
      onDragOver={(e) => {
        // preventDefault is REQUIRED or onDrop never fires (native DnD)
        if (onDropIssue) {
          e.preventDefault();
          setOver(true);
        }
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        setOver(false);
        if (!onDropIssue) return;
        const issueId = e.dataTransfer.getData("text/plain");
        if (issueId) onDropIssue(issueId, state.id);
      }}
    >
      <div className="flex items-center gap-2 px-1">
        {/* Linear state color is runtime data, not a design token */}
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: state.color }} />
        <span className="text-[13px] font-bold text-ink">{state.name}</span>
        <span className="ml-auto rounded-full border border-line bg-surface-2 px-2 py-px text-[11px] text-muted">
          {issues.length}
        </span>
      </div>
      <div className="flex flex-col gap-2.5">
        {issues.map((i) => (
          <IssueCard
            key={i.id}
            issue={i}
            onDragStart={
              onDropIssue
                ? (e) => e.dataTransfer.setData("text/plain", i.id)
                : undefined
            }
            onOpen={onOpenIssue ? () => onOpenIssue(i) : undefined}
          />
        ))}
      </div>
    </div>
  );
}
