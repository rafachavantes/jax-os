"use client";

import { useState } from "react";
import { useTranslations } from "next-intl";
import { ChevronDown, ChevronRight, GitBranch } from "lucide-react";
import { RelativeTime } from "@/components/RelativeTime";
import { serializeFileSelection } from "@/lib/filesUrl";
import type { ExecutionRow, ExecutionOutcome } from "@/lib/executions";
import type { InspectionRow } from "@/server/collectors/execution-inspection";

export function executionRunHref(destination: string, runId: string): string {
  const params = new URLSearchParams({ project: destination, run: runId, period: "all" });
  return `/executions?${params.toString()}`;
}

export function formatExecutionDuration(ms: number | null): string | null {
  if (ms === null || !Number.isFinite(ms) || ms < 0) return null;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${seconds % 60}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

const OUTCOME_TONE: Record<ExecutionOutcome, string> = {
  open: "border-info text-info",
  success: "border-success text-success",
  failure: "border-danger text-danger",
  blocked: "border-warning text-warning",
  cancelled: "border-line-strong text-muted",
  interrupted: "border-line-strong text-muted",
  approve: "border-success text-success",
  "approve-with-changes": "border-warning text-warning",
  reject: "border-danger text-danger",
  unknown: "border-line-strong text-muted",
};

function executionFileName(runId: string): { report: string; tests: string } {
  return { report: `${runId}.md`, tests: `${runId}.tests.txt` };
}

export type ExecutionOverlay = {
  outcome: ExecutionOutcome;
  terminal: boolean;
  durationMs: number | null;
  warnings: string[];
};

// Small pure client mirror of execution-worker.ts's applyCurrentOverlay: a terminal fact for
// the SAME attempt overlays the frozen row; a changed attempt only asks for refresh and never
// mixes the newer attempt's facts into the frozen row. Warnings from every source are combined,
// never replaced by an empty array.
export function executionOverlay(row: ExecutionRow, inspection: InspectionRow | undefined): ExecutionOverlay {
  const current = inspection?.current ?? null;
  const sameAttempt = current !== null && inspection?.refreshNeeded !== true;
  const live = sameAttempt && current.terminal ? current : null;
  return {
    outcome: live ? live.outcome : row.outcome,
    terminal: live ? live.terminal : row.terminal,
    durationMs: live ? live.durationMs : row.durationMs,
    warnings: [...new Set([
      ...row.warnings,
      ...(sameAttempt && current ? current.warnings : []),
      ...(inspection?.warnings ?? []),
    ])],
  };
}

export function ExecutionRows({
  rows,
  inspections = {},
  initialExpanded = null,
}: {
  rows: ExecutionRow[];
  inspections?: Record<string, InspectionRow | undefined>;
  initialExpanded?: string | null;
}) {
  const t = useTranslations("executions");
  const [expanded, setExpanded] = useState<string | null>(initialExpanded);

  return (
    <div className="flex flex-col gap-2">
      {rows.map((row) => {
        const key = `${row.destination}/${row.runId}`;
        const inspection = inspections[key];
        const overlay = executionOverlay(row, inspection);
        const terminal = overlay.terminal;
        const outcome = overlay.outcome;
        const durationMs = overlay.durationMs;
        const branch = inspection?.branch ?? null;
        const report = inspection?.report ?? null;
        const tests = inspection?.tests ?? null;
        const observation = !terminal && !inspection?.refreshNeeded ? inspection?.observation : undefined;
        const names = executionFileName(row.runId);
        const isOpen = expanded === key;
        const kindLabel = row.kind === "build"
          ? t("kind.build")
          : row.kind === "spec" || row.kind === "plan" || row.kind === "diff"
            ? `${t("kind.review")} · ${t(`subtype.${row.kind}`)}`
            : row.role === "reviewer" ? t("kind.review") : row.role === "builder" ? t("kind.build") : t("kind.unknown");
        const dispatchedMs = row.firstDispatchedAt ? Date.parse(row.firstDispatchedAt) : NaN;
        const duration = formatExecutionDuration(durationMs);

        return (
          <article key={key} className="min-w-0 rounded-lg border border-line bg-surface px-4 py-3">
            <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1">
              <a href={executionRunHref(row.destination, row.runId)} className="min-w-0 [overflow-wrap:anywhere] text-[13px] font-semibold text-ink hover:text-brand">
                {row.destination}
              </a>
              <span className="text-[12px] text-body-ink">{kindLabel}</span>
              <span className="font-mono text-[11px] text-muted">{row.runId}</span>
              <span className={`rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase ${OUTCOME_TONE[outcome]}`}>
                {t(`outcome.${outcome}`)}
              </span>
              {inspection?.refreshNeeded ? <span className="text-[11px] font-semibold text-warning">{t("worker.refresh")}</span> : null}
              {observation ? (
                <span className={`text-[11px] ${observation.status === "observed" ? "text-success" : observation.status === "unknown" ? "text-warning" : "text-muted"}`}>
                  {t(`worker.${observation.status === "observed" ? "observed" : observation.status === "awaiting-confirmation" ? "awaiting" : observation.status === "no-worker-observed" ? "absent" : observation.status === "not-checked" ? "notChecked" : "unknown"}`)}
                  {observation.lastGoodAt ? <> · {t("worker.lastGood")} <RelativeTime epochMs={Date.parse(observation.lastGoodAt)} /></> : null}
                </span>
              ) : null}
            </div>

            <div className="mt-1 flex min-w-0 flex-wrap items-center gap-x-3 gap-y-1 text-[11.5px] text-muted">
              <span>{t("executor.label")}: {row.runtime ?? t("executor.unknown")}{row.model ? ` · ${row.model}` : ""}{row.caller ? ` · ${row.caller}` : ""}</span>
              {Number.isFinite(dispatchedMs) ? <span>{t("firstDispatch")}: <RelativeTime epochMs={dispatchedMs} /></span> : <span>{t("firstDispatch")}: {t("unknown")}</span>}
              <span>{t("duration")}: {duration ?? t("durationUnknown")}</span>
              {branch ? (
                <span className="flex min-w-0 items-center gap-1"><GitBranch className="h-3.5 w-3.5 flex-none" aria-hidden /><span className="min-w-0 [overflow-wrap:anywhere]">{branch}</span></span>
              ) : null}
            </div>

            <button
              type="button"
              onClick={() => setExpanded(isOpen ? null : key)}
              aria-expanded={isOpen}
              className="mt-2 flex items-center gap-1 text-[11px] font-semibold text-body-ink"
            >
              {isOpen ? <ChevronDown className="h-3.5 w-3.5" aria-hidden /> : <ChevronRight className="h-3.5 w-3.5" aria-hidden />}
              {isOpen ? t("hideDetails") : t("details")}
            </button>

            {isOpen ? (
              <div className="mt-2 grid min-w-0 gap-1.5 border-t border-line-subtle pt-2 text-[11.5px] text-muted">
                <span className="min-w-0 [overflow-wrap:anywhere]">{t("phase")}: {row.phase ?? t("unknown")}</span>
                <span className="min-w-0 [overflow-wrap:anywhere]">{t("target")}: {row.target ?? t("unknown")}</span>
                <span className="min-w-0 [overflow-wrap:anywhere]">{t("caller")}: {row.caller ?? t("unknown")}</span>
                <span className="min-w-0 [overflow-wrap:anywhere]">{t("branch")}: {branch ?? t("unknown")}</span>
                <div className="flex flex-wrap gap-3 pt-1">
                  {report ? (
                    <a className="font-semibold text-brand" href={`/files?${serializeFileSelection(new URLSearchParams(), { root: "repos", rel: report.rel }).toString()}`}>
                      {t("report")}
                    </a>
                  ) : <span>{t("reportUnavailable")}</span>}
                  {tests ? (
                    <a className="font-semibold text-brand" href={`/files?${serializeFileSelection(new URLSearchParams(), { root: "repos", rel: tests.rel }).toString()}`}>
                      {t("tests")}
                    </a>
                  ) : <span>{t("testsUnavailable")}</span>}
                </div>
                {overlay.warnings.length > 0 ? (
                  <span className="min-w-0 font-mono [overflow-wrap:anywhere]">{t("warnings")}: {overlay.warnings.join(", ")}</span>
                ) : null}
                <span className="font-mono text-[10.5px]">{names.report}</span>
              </div>
            ) : null}
          </article>
        );
      })}
    </div>
  );
}
