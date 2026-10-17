"use client";

import { useTranslations } from "next-intl";
import type { KanbanIssue, KanbanState } from "@/server/collectors/linear";
import { RelativeTime } from "@/components/RelativeTime";
import { PRIORITY_DOT } from "./IssueCard";

type Props = {
  states: KanbanState[];
  issues: KanbanIssue[];
  onOpenIssue: (issue: KanbanIssue) => void;
};

// issues arrives pre-filtered+sorted (shown) from page.tsx — do NOT re-filter/re-sort here.
export function ListView({ states, issues, onOpenIssue }: Props) {
  const t = useTranslations("kanban");
  return (
    <div className="jax-scroll flex flex-1 flex-col gap-4 overflow-y-auto pb-1.5">
      {states.map((s) => {
        const rows = issues.filter((i) => i.stateId === s.id);
        if (rows.length === 0) return null; // skip empty groups
        return (
          <div key={s.id} className="flex flex-col">
            <div className="flex items-center gap-2 px-1 pb-1.5">
              {/* Linear state color is runtime data, not a design token */}
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: s.color }} />
              <span className="text-[13px] font-bold text-ink">{s.name}</span>
              <span className="ml-auto rounded-full border border-line bg-surface-2 px-2 py-px text-[11px] text-muted">
                {rows.length}
              </span>
            </div>
            <div className="flex flex-col divide-y divide-line rounded-md border border-line bg-surface">
              {rows.map((i) => {
                const createdMs = Date.parse(i.createdAt);
                return (
                  <button
                    type="button"
                    key={i.id}
                    onClick={() => onOpenIssue(i)}
                    className="flex w-full cursor-pointer items-center gap-3 px-3.5 py-2.5 text-left transition-colors hover:bg-surface-2"
                  >
                    <span
                      className={`h-2 w-2 flex-none rounded-full ${PRIORITY_DOT[i.priority] ?? "bg-line"}`}
                    />
                    <span className="w-16 flex-none font-mono text-[10.5px] text-muted">
                      {i.identifier}
                    </span>
                    <span className="flex-1 truncate text-[13px] font-medium text-ink">
                      {i.title}
                    </span>
                    {i.subIssue ? (
                      <span className="flex-none rounded-full border border-line bg-surface-2 px-1.5 py-px text-[10px] font-medium text-muted">
                        {i.subIssue.done}/{i.subIssue.total}
                      </span>
                    ) : null}
                    <span className="flex flex-none items-center gap-2 text-[11px] text-muted">
                      {i.assignee ? (
                        <>
                          <span className="flex h-5 w-5 items-center justify-center rounded-full bg-accent-soft text-[10px] font-bold text-accent">
                            {i.assignee.charAt(0).toUpperCase()}
                          </span>
                          {i.assignee}
                        </>
                      ) : (
                        t("unassigned")
                      )}
                    </span>
                    {Number.isFinite(createdMs) ? (
                      <span className="w-14 flex-none text-right text-[10.5px] text-muted">
                        <RelativeTime epochMs={createdMs} />
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
