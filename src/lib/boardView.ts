import type { KanbanBoard, KanbanIssue } from "@/server/collectors/linear";

export type SortKey = "priority" | "updated" | "created" | "number" | "status";
export type Filters = { query: string; project: string | null; priority: number | null; label: string | null; assignee: string | null };

export function filterIssues(issues: KanbanIssue[], f: Filters): KanbanIssue[] {
  const q = f.query.trim().toLowerCase();
  return issues.filter((i) => {
    if (q && !`${i.identifier} ${i.title}`.toLowerCase().includes(q)) return false;
    if (f.project && i.project !== f.project) return false;
    if (f.priority !== null && i.priority !== f.priority) return false;
    if (f.label && !i.labels.some((l) => l.name === f.label)) return false;
    if (f.assignee && i.assignee !== f.assignee) return false;
    return true;
  });
}

// Linear priority ints: 1 urgent … 4 low, 0 none. Sort urgent-first, none-last.
const PRIORITY_RANK: Record<number, number> = { 1: 0, 2: 1, 3: 2, 4: 3, 0: 4 };
const numOf = (identifier: string): number => Number(identifier.match(/(\d+)$/)?.[1] ?? 0);

export function sortIssues(issues: KanbanIssue[], key: SortKey, stateRank?: Map<string, number>): KanbanIssue[] {
  return [...issues].sort((a, b) => {
    switch (key) {
      case "priority": return PRIORITY_RANK[a.priority] - PRIORITY_RANK[b.priority];
      case "updated": return b.updatedAt.localeCompare(a.updatedAt);
      case "created": return b.createdAt.localeCompare(a.createdAt);
      case "number": return numOf(a.identifier) - numOf(b.identifier);
      case "status": return (stateRank?.get(a.stateId) ?? 0) - (stateRank?.get(b.stateId) ?? 0);
    }
  });
}

// Pure apply-only reducer for the optimistic board patch on a move: sets the
// dragged issue's column. No outcome handling — reconcileWrite decides
// keep-vs-rollback on top of this (phase 6 spec §6.3, Decision 11).
export function nextBoardState(board: KanbanBoard, issueId: string, toStateId: string): KanbanBoard {
  return {
    ...board,
    issues: board.issues.map((i) => (i.id === issueId ? { ...i, stateId: toStateId } : i)),
  };
}
