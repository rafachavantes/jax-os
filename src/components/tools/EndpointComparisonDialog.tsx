"use client";

import { forwardRef, useEffect, useImperativeHandle, useRef, useState } from "react";
import { useTranslations } from "next-intl";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { fetchEnvelope } from "@/lib/api";
import type { Routing } from "@/lib/agent-settings";
import { comparisonUrl, evaluateEndpoint, singlePercentileKey, type Match } from "@/lib/openrouter-comparison";
import type { EndpointComparison, EndpointRow } from "@/server/collectors/openrouter";
import css from "./tools.module.css";

export type EndpointComparisonHandle = { open: (refresh: boolean) => void };

type RequestState = {
  status: "idle" | "loading" | "ready" | "error";
  data: EndpointComparison | null;
  error: string | null;
  stale: boolean;
};

export type ComparisonBodyMode = "loading" | "error" | "table";

// A first-load error is not an empty success: only a request that produced data
// (or a successful-empty response) may render the table.
export function comparisonBodyMode(state: {
  status: "idle" | "loading" | "ready" | "error";
  data: EndpointComparison | null;
}): ComparisonBodyMode {
  if (state.status === "error" && !state.data) return "error";
  if (state.status === "loading" && !state.data) return "loading";
  return "table";
}

function isComparison(value: unknown): value is EndpointComparison {
  return !!value && typeof value === "object" && Array.isArray((value as EndpointComparison).endpoints);
}

function matchLabel(t: (key: string) => string, value: Match | "none"): string {
  if (value === "none") return t("comparison.goalsNone");
  if (value === "yes") return t("comparison.yes");
  if (value === "no") return t("comparison.no");
  return t("comparison.unknown");
}

function price(value: number | null): string {
  return value === null ? "—" : `$${value.toFixed(2)}`;
}

function stat(value: number | undefined, suffix: string, digits: number): string {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toFixed(digits)}${suffix}` : "—";
}

export function EndpointComparisonTable({
  data,
  routing,
}: {
  data: EndpointComparison | null;
  routing: Routing | null;
}) {
  const t = useTranslations("tools");
  const policy = routing ?? { sort: "price" as const, allow_fallbacks: true };
  const throughputKey = singlePercentileKey(policy.preferred_min_throughput) ?? "p90";
  const latencyKey = singlePercentileKey(policy.preferred_max_latency) ?? "p90";
  const rows = data ? data.endpoints.map((row) => ({ row, evaluation: evaluateEndpoint(row, policy) })) : [];
  const eligibleCount = rows.filter((entry) => entry.evaluation.eligible === "yes").length;
  const matchClass = (value: Match | "none") =>
    value === "yes" ? css.matchYes : value === "no" ? css.matchNo : css.matchUnknown;
  return (
    <>
      <div className={css.tableScroll}>
        <table className={css.comparisonTable}>
          <thead>
            <tr>
              {["identity", "input", "output", "latency", "throughput", "eligibility", "goals"].map((key) => (
                <th key={key} scope="col">{t(`comparison.columns.${key}`)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 ? (
              <tr><td colSpan={7}>{t("comparison.empty")}</td></tr>
            ) : (
              rows.map(({ row, evaluation }, index) => (
                <tr key={`${row.tag ?? row.provider ?? index}`}>
                  <td>
                    <strong>{row.provider ?? t("comparison.unknownProvider")}</strong>
                    <small>{row.tag ?? "—"}</small>
                    <small>{row.quantization ?? "—"}</small>
                  </td>
                  <td>{price(row.prices.prompt)}</td>
                  <td>{price(row.prices.completion)}</td>
                  <td>
                    {stat(row.latency?.[latencyKey], "s", 2)}
                    {row.latency ? <small>{`${latencyKey} · ${row.latency.window}`}</small> : null}
                  </td>
                  <td>
                    {stat(row.throughput?.[throughputKey], " tok/s", 0)}
                    {row.throughput ? <small>{`${throughputKey} · ${row.throughput.window}`}</small> : null}
                  </td>
                  <td><span className={matchClass(evaluation.eligible)}>{matchLabel(t, evaluation.eligible)}</span></td>
                  <td><span className={matchClass(evaluation.goals)}>{matchLabel(t, evaluation.goals)}</span></td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
      {data && data.endpoints.length ? (
        <p className={css.hint}>{t("comparison.eligibleCount", { eligible: String(eligibleCount), count: String(rows.length) })}</p>
      ) : null}
    </>
  );
}

export const EndpointComparisonDialog = forwardRef<
  EndpointComparisonHandle,
  { connection: string; model: string; routing: Routing | null }
>(function EndpointComparisonDialog({ connection, model, routing }, ref) {
  const t = useTranslations("tools");
  const dialogRef = useRef<HTMLDialogElement>(null);
  const contextRef = useRef({ connection, model });
  contextRef.current = { connection, model };
  const controllerRef = useRef<AbortController | null>(null);
  const requestIdRef = useRef(0);
  const [state, setState] = useState<RequestState>({ status: "idle", data: null, error: null, stale: false });

  function abort() {
    controllerRef.current?.abort();
    controllerRef.current = null;
    requestIdRef.current += 1;
  }

  useEffect(() => abort, []);

  useEffect(() => {
    abort();
    setState({ status: "idle", data: null, error: null, stale: false });
    dialogRef.current?.close();
  }, [connection, model]);

  async function run(refresh: boolean) {
    const current = contextRef.current;
    if (!current.connection || !current.model) return;
    abort();
    const id = requestIdRef.current;
    const controller = new AbortController();
    controllerRef.current = controller;
    setState((prev) => ({ ...prev, status: "loading", error: null }));
    const url = comparisonUrl(current.connection, current.model, refresh);
    try {
      const envelope = await fetchEnvelope<EndpointComparison>(url, controller.signal);
      if (id !== requestIdRef.current) return;
      if (!isComparison(envelope.data)) {
        setState((prev) => ({ ...prev, status: "error", error: "unavailable" }));
        return;
      }
      setState({ status: "ready", data: envelope.data, error: envelope.data.error, stale: envelope.data.stale });
    } catch (err) {
      if (id !== requestIdRef.current) return;
      if (err instanceof DOMException && err.name === "AbortError") return;
      setState((prev) => ({ ...prev, status: "error", error: err instanceof Error ? err.message : "unavailable", stale: prev.data !== null }));
    }
  }

  useImperativeHandle(ref, () => ({
    open: (refresh: boolean) => {
      const dialog = dialogRef.current;
      if (dialog && typeof dialog.showModal === "function" && !dialog.open) dialog.showModal();
      void run(refresh);
    },
  }), []);

  const data = state.data;
  const policy = routing ?? { sort: "price" as const, allow_fallbacks: true };

  return (
    <dialog
      ref={dialogRef}
      aria-label={t("comparison.title")}
      onClose={() => abort()}
      onCancel={(e) => {
        e.preventDefault();
        dialogRef.current?.close();
      }}
      className={css.comparisonDialog}
    >
      <div className={css.comparisonBody}>
        <div className={css.dialogHeader}>
          <div>
            <h2>{t("comparison.title")}</h2>
            <p>{t("comparison.subtitle", { model })}</p>
          </div>
          <button type="button" className={css.dialogClose} onClick={() => dialogRef.current?.close()}>
            {t("comparison.close")}
          </button>
        </div>
        {data?.stale || state.stale ? (
          <p className="rounded-md border border-warning bg-warning-soft px-3 py-2 text-xs text-warning">
            {t("comparison.stale")}
          </p>
        ) : null}
        {state.status === "error" ? <SourceWarning label={t("comparison.error")} detail={state.error ?? undefined} /> : null}
        {data?.error ? <SourceWarning label={t("comparison.error")} detail={data.error} /> : null}
        <div className={css.comparisonMeta}>
          <span>{data ? t("comparison.fetchedAt") : t("comparison.loading")}</span>
          {data ? <RelativeTime epochMs={Date.parse(data.fetched_at)} /> : null}
        </div>
        {state.status === "loading" && !data ? (
          <p className={css.hint}>{t("comparison.loading")}</p>
        ) : comparisonBodyMode(state) === "error" ? null : (
          <EndpointComparisonTable data={data} routing={policy} />
        )}
        <p className={css.hint}>{t("comparison.softGoal")}</p>
        <div className={css.dialogFooter}>
          <button type="button" className={css.compareButton} disabled={state.status === "loading"} onClick={() => void run(true)}>
            {t("comparison.refresh")}
          </button>
        </div>
      </div>
    </dialog>
  );
});
