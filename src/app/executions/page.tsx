"use client";

import { useQuery } from "@tanstack/react-query";
import { History } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import { Suspense } from "react";
import { fetchEnvelope } from "@/lib/api";
import {
  EXECUTION_KINDS, EXECUTION_PERIODS, EXECUTION_STATES, executionFilterIdentity,
  parseExecutionSearch, serializeExecutionSearch, type ExecutionFilters, type ExecutionRow,
} from "@/lib/executions";
import { searchHref } from "@/lib/cockpitUrl";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { ExecutionRows } from "@/components/executions/ExecutionRows";
import type { ExecutionPage } from "@/server/db/workflows";
import type { InspectionRow } from "@/server/collectors/execution-inspection";

function apiHref(filters: ExecutionFilters, cursor: string | null): string {
  const params = serializeExecutionSearch(new URLSearchParams(), filters);
  if (cursor !== null) params.set("cursor", cursor);
  return `/api/executions?${params.toString()}`;
}

function inspectionHref(rows: ExecutionRow[]): string {
  const params = new URLSearchParams();
  for (const row of rows) {
    params.append("r", [encodeURIComponent(row.destination), encodeURIComponent(row.runId), row.startEventId ?? ""].join("|"));
  }
  return `/api/executions/inspection?${params.toString()}`;
}

function inspectionMap(rows: InspectionRow[] | undefined): Record<string, InspectionRow | undefined> {
  const map: Record<string, InspectionRow | undefined> = {};
  for (const row of rows ?? []) map[`${row.destination}/${row.runId}`] = row;
  return map;
}

export type ExecutionsBodyProps = {
  filters: ExecutionFilters;
  rows: ExecutionRow[];
  inspections: Record<string, InspectionRow | undefined>;
  olderHref: string | null;
  loading: boolean;
  error: string | null;
  inspectionError: boolean;
  invalid: boolean;
  onFilter: (filters: ExecutionFilters) => void;
  onRefresh: () => void;
  onReset: () => void;
};

export function ExecutionsBody({
  filters, rows, inspections, olderHref, loading, error, inspectionError, invalid, onFilter, onRefresh, onReset,
}: ExecutionsBodyProps) {
  const t = useTranslations("executions");
  const selectClass = "h-8 rounded-md border border-line bg-surface-2 px-2 text-[13px] text-ink outline-none focus:border-line-strong";

  return (
    <div className="mx-auto flex w-full max-w-[1320px] flex-col gap-[18px] [animation:jax-rise_.4s_ease]">
      <div className="flex items-center gap-3.5 py-0.5">
        <History className="h-5 w-5 flex-none text-brand" aria-hidden />
        <h1 className="min-w-0 text-[15px] text-body-ink">
          <strong className="font-semibold text-ink">{t("title")}</strong> {t("subtitle")}
        </h1>
      </div>

      <form
        className="flex flex-wrap items-end gap-2"
        onSubmit={(event) => {
          event.preventDefault();
          const form = new FormData(event.currentTarget);
          const project = String(form.get("project") ?? "").trim();
          onFilter({ ...filters, project: project === "" ? null : project, run: null });
        }}
      >
        <label className="flex flex-col gap-1 text-[10px] font-semibold uppercase tracking-[.08em] text-muted">
          {t("filters.project")}
          <input
            key={filters.project ?? ""}
            name="project"
            defaultValue={filters.project ?? ""}
            placeholder={t("filters.projectPlaceholder")}
            maxLength={64}
            className={`${selectClass} w-44`}
          />
        </label>
        <label className="flex flex-col gap-1 text-[10px] font-semibold uppercase tracking-[.08em] text-muted">
          {t("filters.kind")}
          <select className={selectClass} value={filters.kind} onChange={(e) => onFilter({ ...filters, kind: e.target.value as ExecutionFilters["kind"], run: null })}>
            {EXECUTION_KINDS.map((kind) => <option key={kind} value={kind}>{t(`filters.kinds.${kind}`)}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-[10px] font-semibold uppercase tracking-[.08em] text-muted">
          {t("filters.state")}
          <select className={selectClass} value={filters.state} onChange={(e) => onFilter({ ...filters, state: e.target.value as ExecutionFilters["state"], run: null })}>
            {EXECUTION_STATES.map((state) => <option key={state} value={state}>{t(`filters.states.${state}`)}</option>)}
          </select>
        </label>
        <div className="flex flex-col gap-1 text-[10px] font-semibold uppercase tracking-[.08em] text-muted">
          {t("filters.period")}
          <div className="flex items-center gap-1">
            {EXECUTION_PERIODS.map((period) => (
              <button
                key={period}
                type="button"
                onClick={() => onFilter({ ...filters, period, run: null })}
                className={`rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors ${filters.period === period ? "border-brand-soft-border bg-brand-soft text-brand" : "border-line text-muted hover:text-body-ink"}`}
              >
                {t(`filters.periods.${period}`)}
              </button>
            ))}
          </div>
        </div>
        <button type="submit" className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-body-ink hover:bg-surface-2">
          {t("filters.apply")}
        </button>
        <button type="button" onClick={onReset} className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted hover:text-body-ink">
          {t("filters.reset")}
        </button>
        {filters.run ? (
          <span className="rounded border border-brand-soft-border bg-brand-soft px-2 py-1 text-[11px] font-semibold text-brand">
            {t("runContext", { destination: filters.run.project, run: filters.run.runId })}
          </span>
        ) : null}
      </form>

      {invalid ? <SourceWarning label={t("invalidUrl")} /> : null}
      {error ? <SourceWarning label={t("unavailable")} detail={error} /> : null}
      {inspectionError && rows.length > 0 ? <SourceWarning label={t("inspectionUnavailable")} /> : null}

      {invalid ? null : loading && rows.length === 0 ? (
        <div className="rounded-lg border border-line bg-surface p-4">
          <div className="h-4 w-52 animate-pulse rounded bg-surface-2" />
        </div>
      ) : !error && rows.length === 0 ? (
        <p className="rounded-lg border border-line bg-surface p-4 text-[12.5px] text-muted">{t("empty")}</p>
      ) : rows.length > 0 ? (
        <ExecutionRows rows={rows} inspections={inspections} />
      ) : null}

      {rows.length > 0 ? (
        <div className="flex items-center gap-3">
          {olderHref ? (
            <a href={olderHref} className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-body-ink hover:bg-surface-2">
              {t("older")}
            </a>
          ) : null}
          <button type="button" onClick={onRefresh} className="rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-muted hover:text-body-ink">
            {t("refresh")}
          </button>
        </div>
      ) : null}
    </div>
  );
}

export function ExecutionsPageInner() {
  const t = useTranslations("executions");
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const search = params.toString();
  const urlParams = new URLSearchParams(search);
  const parsed = parseExecutionSearch(urlParams);
  const cursor = urlParams.get("cursor");
  const filters: ExecutionFilters = parsed.ok
    ? parsed.filters
    : { project: null, kind: "all", state: "all", period: "30d", run: null, limit: 20 };
  const identity = executionFilterIdentity(filters);

  const page = useQuery({
    queryKey: ["executions", identity, cursor],
    queryFn: ({ signal }) => fetchEnvelope<ExecutionPage>(apiHref(filters, cursor), signal),
    refetchInterval: 10_000,
    enabled: parsed.ok,
  });
  const rows = page.data?.data.rows ?? [];
  const signature = rows.map((row) => `${row.destination}/${row.runId}/${row.startEventId ?? ""}`).join(",");
  const inspection = useQuery({
    queryKey: ["executions-inspection", identity, cursor, signature],
    queryFn: ({ signal }) => fetchEnvelope<InspectionRow[]>(inspectionHref(rows), signal),
    refetchInterval: 10_000,
    enabled: parsed.ok && rows.length > 0,
  });

  const onFilter = (next: ExecutionFilters) => {
    router.replace(searchHref(pathname, serializeExecutionSearch(new URLSearchParams(), next)), { scroll: false });
  };
  // Explicit Refresh must run a fresh first-page query even when already on the first page; a
  // same-URL router.replace never forces one. The cursor is still cleared for an older page.
  const onRefresh = () => {
    const next = new URLSearchParams(urlParams);
    next.delete("cursor");
    if (cursor !== null) router.replace(searchHref(pathname, next), { scroll: false });
    void page.refetch();
    void inspection.refetch();
  };
  const onReset = () => router.replace(pathname, { scroll: false });

  let olderHref: string | null = null;
  const nextCursor = page.data?.data.nextCursor ?? null;
  if (parsed.ok && nextCursor !== null) {
    const next = serializeExecutionSearch(new URLSearchParams(), filters);
    next.set("cursor", nextCursor);
    olderHref = searchHref(pathname, next);
  }

  return (
    <ExecutionsBody
      filters={filters}
      rows={rows}
      inspections={inspectionMap(inspection.data?.data)}
      olderHref={olderHref}
      loading={page.isLoading}
      error={page.isError ? (page.error instanceof Error ? page.error.message : t("unavailable")) : null}
      inspectionError={inspection.isError}
      invalid={!parsed.ok}
      onFilter={onFilter}
      onRefresh={onRefresh}
      onReset={onReset}
    />
  );
}

export default function ExecutionsPage() {
  return (
    <Suspense>
      <ExecutionsPageInner />
    </Suspense>
  );
}
