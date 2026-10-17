"use client";

import type { KanbanIssue } from "@/server/collectors/linear";
import { RelativeTime } from "@/components/RelativeTime";

// Linear priority: 0 none, 1 urgent, 2 high, 3 medium, 4 low
export const PRIORITY_DOT: Record<number, string> = {
  0: "bg-line",
  1: "bg-danger",
  2: "bg-warning",
  3: "bg-info",
  4: "bg-muted",
};

type Props = {
  issue: KanbanIssue;
  onDragStart?: (e: React.DragEvent) => void;
  onOpen?: () => void;
};

export function IssueCard({ issue, onDragStart, onOpen }: Props) {
  const createdMs = Date.parse(issue.createdAt);
  return (
    <div
      draggable={Boolean(onDragStart)}
      onDragStart={onDragStart}
      className="flex cursor-grab flex-col gap-2.5 rounded-md border border-line bg-surface p-3.5 transition-colors hover:border-line-strong active:cursor-grabbing"
    >
      <div className="flex items-center gap-2">
        <span
          className={`h-2 w-2 flex-none rounded-full ${PRIORITY_DOT[issue.priority] ?? "bg-line"}`}
        />
        <span className="font-mono text-[10.5px] text-muted">{issue.identifier}</span>
        {issue.subIssue ? (
          <span className="rounded-full border border-line bg-surface-2 px-1.5 py-px text-[10px] font-medium text-muted">
            {issue.subIssue.done}/{issue.subIssue.total}
          </span>
        ) : null}
        {issue.project ? (
          <span className="ml-auto truncate rounded-full bg-surface-3 px-2 py-0.5 text-[10px] font-semibold text-muted">
            {issue.project}
          </span>
        ) : null}
      </div>
      <button
        type="button"
        onClick={onOpen}
        disabled={!onOpen}
        className="text-left text-[13px] font-medium leading-[1.35] text-ink disabled:cursor-default"
      >
        {issue.title}
      </button>
      {issue.labels.length > 0 ? (
        <div className="flex flex-wrap items-center gap-1.5">
          {issue.labels.map((l) => (
            <span
              key={l.id}
              className="rounded-full border px-2 py-0.5 text-[10px] font-semibold"
              // Linear label color is runtime data, not a design token
              style={{ color: l.color, borderColor: l.color }}
            >
              {l.name}
            </span>
          ))}
        </div>
      ) : null}
      <div className="flex items-center gap-2 pt-0.5 text-[11px] text-muted">
        {issue.assignee ? (
          <span className="flex items-center gap-2">
            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-accent-soft text-[10px] font-bold text-accent">
              {issue.assignee.charAt(0).toUpperCase()}
            </span>
            {issue.assignee}
          </span>
        ) : null}
        {Number.isFinite(createdMs) ? (
          <span className="ml-auto text-[10.5px]">
            <RelativeTime epochMs={createdMs} />
          </span>
        ) : null}
      </div>
    </div>
  );
}
