"use client";

import { useTranslations } from "next-intl";
import type { Filters, SortKey } from "@/lib/boardView";
import type { KanbanIssue } from "@/server/collectors/linear";

// Linear priority ints → existing i18n keys (Urgent=1 … Low=4, None=0)
const PRIORITY_OPTIONS: { value: number; key: string }[] = [
  { value: 1, key: "priorityUrgent" },
  { value: 2, key: "priorityHigh" },
  { value: 3, key: "priorityMedium" },
  { value: 4, key: "priorityLow" },
  { value: 0, key: "priorityNone" },
];

const SORT_LABEL: Record<SortKey, string> = {
  priority: "sortPriority",
  updated: "sortUpdated",
  created: "sortCreated",
  number: "sortNumber",
  status: "sortStatus",
};

const CONTROL =
  "h-8 rounded-full border border-line bg-surface-2 px-3 text-xs text-ink outline-none focus:border-line-strong";

export function FilterBar({
  issues,
  filters,
  onFilters,
  sort,
  onSort,
  view,
  onView,
}: {
  issues: KanbanIssue[];
  filters: Filters;
  onFilters: (f: Filters) => void;
  sort: SortKey;
  onSort: (s: SortKey) => void;
  view: "board" | "list";
  onView: (v: "board" | "list") => void;
}) {
  const t = useTranslations("kanban");

  // option lists derived from the full (unfiltered) board so choices never vanish
  const projects = [
    ...new Set(issues.map((i) => i.project).filter((p): p is string => Boolean(p))),
  ].sort();
  const labels = [...new Set(issues.flatMap((i) => i.labels.map((l) => l.name)))].sort();
  const assignees = [
    ...new Set(issues.map((i) => i.assignee).filter((a): a is string => Boolean(a))),
  ].sort();

  return (
    <div className="flex flex-wrap items-center gap-2">
      <input
        type="search"
        value={filters.query}
        onChange={(e) => onFilters({ ...filters, query: e.target.value })}
        aria-label={t("searchPlaceholder")}
        placeholder={t("searchPlaceholder")}
        className={`${CONTROL} w-40 placeholder:text-muted`}
      />
      <select
        value={filters.project ?? ""}
        onChange={(e) => onFilters({ ...filters, project: e.target.value || null })}
        aria-label={`${t("filterProject")}: ${t("filterAll")}`}
        className={CONTROL}
      >
        <option value="">{`${t("filterProject")}: ${t("filterAll")}`}</option>
        {projects.map((p) => (
          <option key={p} value={p}>
            {p}
          </option>
        ))}
      </select>
      <select
        value={filters.assignee ?? ""}
        onChange={(e) => onFilters({ ...filters, assignee: e.target.value || null })}
        aria-label={`${t("filterAssignee")}: ${t("filterAll")}`}
        className={CONTROL}
      >
        <option value="">{`${t("filterAssignee")}: ${t("filterAll")}`}</option>
        {assignees.map((a) => (
          <option key={a} value={a}>
            {a}
          </option>
        ))}
      </select>
      <select
        value={filters.priority ?? ""}
        onChange={(e) =>
          onFilters({ ...filters, priority: e.target.value === "" ? null : Number(e.target.value) })
        }
        aria-label={`${t("filterPriority")}: ${t("filterAll")}`}
        className={CONTROL}
      >
        <option value="">{`${t("filterPriority")}: ${t("filterAll")}`}</option>
        {PRIORITY_OPTIONS.map((p) => (
          <option key={p.value} value={p.value}>
            {t(p.key)}
          </option>
        ))}
      </select>
      <select
        value={filters.label ?? ""}
        onChange={(e) => onFilters({ ...filters, label: e.target.value || null })}
        aria-label={`${t("filterLabel")}: ${t("filterAll")}`}
        className={CONTROL}
      >
        <option value="">{`${t("filterLabel")}: ${t("filterAll")}`}</option>
        {labels.map((l) => (
          <option key={l} value={l}>
            {l}
          </option>
        ))}
      </select>
      <select
        value={sort}
        onChange={(e) => onSort(e.target.value as SortKey)}
        className={CONTROL}
        aria-label={t("sortBy")}
      >
        {(Object.keys(SORT_LABEL) as SortKey[]).map((k) => (
          <option key={k} value={k}>
            {`${t("sortBy")}: ${t(SORT_LABEL[k])}`}
          </option>
        ))}
      </select>
      <div className="flex gap-1 rounded-full border border-line bg-surface-2 p-1">
        {(["board", "list"] as const).map((v) => (
          <button
            key={v}
            onClick={() => onView(v)}
            className={`flex h-7 items-center rounded-full px-2.5 text-xs font-semibold transition-colors ${
              v === view ? "bg-brand-soft text-brand" : "text-muted hover:text-ink"
            }`}
          >
            {t(v === "board" ? "viewBoard" : "viewList")}
          </button>
        ))}
      </div>
    </div>
  );
}
