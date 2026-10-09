"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useTranslations } from "next-intl";
import { auditStatusKey, mutationSummary } from "@/lib/auditSummary";
import type { MutationRow } from "@/server/db/mutations";
import { RelativeTime } from "@/components/RelativeTime";

export function StatusCell({ row }: { row: MutationRow }) {
  const t = useTranslations("audit");
  const status = auditStatusKey(row);
  if (status === "legacy-none") return <span className="text-muted">—</span>;
  const ok = status === "success" || status === "legacy-ok";
  const label = status === "uncertain"
    ? t("statusUncertain")
    : status === "failure"
      ? t("statusFailed")
      : ok
        ? t("statusOk")
        : (row.error ?? t("statusError"));
  return (
    <span className={`flex items-center gap-1.5 ${ok ? "text-success" : status === "uncertain" ? "text-muted" : "text-danger"}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${ok ? "bg-success" : status === "uncertain" ? "bg-muted" : "bg-danger"}`} />
      {label}
    </span>
  );
}

export function RowPair({ row, open, onToggle }: { row: MutationRow; open: boolean; onToggle: () => void }) {
  const t = useTranslations("audit");
  return (
    <>
      <tr className="border-b border-line-subtle hover:bg-surface-2">
        <td className="px-3 py-2 text-muted">
          <button
            type="button"
            aria-expanded={open}
            aria-label={t("expandRow")}
            onClick={onToggle}
            className="rounded-sm text-muted outline-none focus-visible:ring-2 focus-visible:ring-brand"
          >
            {open ? <ChevronDown className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
        </td>
        <td className="whitespace-nowrap px-3 py-2 text-muted">
          <RelativeTime epochMs={Date.parse(row.ts)} />
        </td>
        <td className="px-3 py-2">
          <span className="rounded-full border border-line bg-surface-2 px-2 py-0.5 font-mono text-[11px] text-body-ink">
            {row.kind}
          </span>
        </td>
        <td className="max-w-[420px] truncate px-3 py-2 font-mono text-[12px] text-body-ink">
          {mutationSummary(row.kind, row.payload)}
        </td>
        <td className="whitespace-nowrap px-3 py-2">
          <StatusCell row={row} />
        </td>
      </tr>
      {open ? (
        <tr className="border-b border-line-subtle bg-surface-2">
          <td />
          <td colSpan={4} className="px-3 py-2">
            <pre className="jax-scroll overflow-x-auto font-mono text-[11px] leading-relaxed text-muted">
              {typeof row.payload === "string" ? row.payload : JSON.stringify(row.payload, null, 2)}
            </pre>
          </td>
        </tr>
      ) : null}
    </>
  );
}