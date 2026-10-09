"use client";

import { useQuery } from "@tanstack/react-query";
import { ScrollText } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import { Suspense, useEffect, useReducer, useRef } from "react";
import { parseAuditSearch, searchHref, serializeAuditSearch } from "@/lib/cockpitUrl";
import { fetchEnvelope, type Envelope } from "@/lib/api";
import type { MutationPage } from "@/server/db/mutations";
import {
  auditFilterIdentity, auditReducer, beginAuditLoadMore, emptyAuditState,
} from "@/lib/auditPaging";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { RowPair } from "@/components/audit/AuditRows";

const PAGE_SIZE = 50;
const PRESETS = ["7d", "30d", "all"] as const;
type Preset = (typeof PRESETS)[number];

function fromFor(preset: Preset): string | undefined {
  if (preset === "all") return undefined;
  const days = preset === "7d" ? 7 : 30;
  return new Date(Date.now() - days * 86_400_000).toISOString().slice(0, 10);
}

function AuditPageInner() {
  const t = useTranslations("audit");
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const url = parseAuditSearch(new URLSearchParams(params.toString()));
  const kind = url.kind;
  const preset = url.preset;
  const loadMoreGuard = useRef(false);

  const from = fromFor(preset);
  const identity = auditFilterIdentity(kind, preset);
  const base = `kind=${encodeURIComponent(kind)}${from ? `&from=${from}` : ""}`;

  const q = useQuery<Envelope<MutationPage>>({
    queryKey: ["audit", kind, preset],
    queryFn: ({ signal }) => fetchEnvelope<MutationPage>(`/api/audit?${base}&limit=${PAGE_SIZE}`, signal),
    refetchInterval: 10_000,
  });

  // each actual successful first-page result is consumed exactly once, into
  // filter-scoped state (never reapplied from cached data on every render);
  // the lazy init seeds the already-available initial result so the first
  // render is not an empty flash
  const [view, dispatch] = useReducer(
    auditReducer,
    emptyAuditState(identity),
    (base) => (q.data?.ok ? auditReducer(base, { type: "firstPage", identity, page: q.data.data }) : base),
  );

  // history navigation changes the URL without setFilters: reset the captured
  // filter identity there too, not only on explicit UI filter changes
  useEffect(() => {
    dispatch({ type: "filter", identity });
  }, [identity]);

  useEffect(() => {
    if (!q.data?.ok) return;
    dispatch({ type: "firstPage", identity, page: q.data.data });
  }, [q.data, identity]);

  const display = view.identity === identity ? view : emptyAuditState(identity);
  const viewRef = useRef(display);
  viewRef.current = display;

  function setFilters(nextKind: string, nextPreset: Preset) {
    router.replace(
      searchHref(pathname, serializeAuditSearch(new URLSearchParams(params.toString()), { kind: nextKind, preset: nextPreset })),
      { scroll: false },
    );
  }

  async function loadMore() {
    if (loadMoreGuard.current) return;
    const s = viewRef.current;
    const started = beginAuditLoadMore(s);
    if (!started) return;
    // capture the selection this request belongs to; a late settle for an
    // older selection must not mutate the current one
    const capturedIdentity = s.identity;
    const capturedGeneration = s.generation;
    loadMoreGuard.current = true;
    dispatch({ type: "loadMoreStart", generation: capturedGeneration });
    try {
      const res: Envelope<MutationPage> = await (await fetch(
        `/api/audit?${base}&limit=${PAGE_SIZE}&cursor=${encodeURIComponent(started.cursor)}`,
      )).json();
      if (!res.ok || !res.data) {
        dispatch({
          type: "loadMoreFailed",
          identity: capturedIdentity,
          generation: capturedGeneration,
          error: !res.ok ? res.error : "unavailable",
        });
        return;
      }
      dispatch({ type: "loadMoreDone", identity: capturedIdentity, generation: capturedGeneration, page: res.data });
    } catch {
      dispatch({
        type: "loadMoreFailed",
        identity: capturedIdentity,
        generation: capturedGeneration,
        error: "unavailable",
      });
    } finally {
      loadMoreGuard.current = false;
    }
  }

  const failed = q.isError && display.rows.length === 0;
  const rows = display.rows;
  const hasMore = display.cursor !== null;

  return (
    <div className="mx-auto flex w-full max-w-[1320px] flex-col gap-[22px] [animation:jax-rise_.4s_ease]">
      <div className="flex items-center gap-3.5 py-0.5">
        <ScrollText className="h-5 w-5 flex-none text-brand" />
        <h1 className="text-[15px] text-body-ink">
          <strong className="font-semibold text-ink">{t("title")}</strong> {t("subtitle")}
        </h1>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <select
          value={kind}
          onChange={(e) => setFilters(e.target.value, preset)}
          className="h-8 rounded-md border border-line bg-surface-2 px-2 text-[13px] text-ink outline-none focus:border-line-strong"
        >
          <option value="">{t("allKinds")}</option>
          {(q.data?.ok ? q.data.data.kinds : []).map((k) => (
            <option key={k} value={k}>
              {k}
            </option>
          ))}
        </select>
        <div className="flex items-center gap-1">
          {PRESETS.map((p) => (
            <button
              key={p}
              onClick={() => setFilters(kind, p)}
              className={`rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors ${
                preset === p ? "border-brand-soft-border bg-brand-soft text-brand" : "border-line text-muted hover:text-body-ink"
              }`}
            >
              {t(p === "7d" ? "range7d" : p === "30d" ? "range30d" : "rangeAll")}
            </button>
          ))}
        </div>
      </div>

      {q.isError && rows.length > 0 ? (
        <SourceWarning
          label={t("unavailable")}
          aside={q.dataUpdatedAt ? <RelativeTime epochMs={q.dataUpdatedAt} /> : undefined}
        />
      ) : null}

      <div className="overflow-x-auto rounded-lg border border-line bg-surface">
        {failed && rows.length === 0 ? (
          <div className="p-4">
            <SourceWarning label={t("unavailable")} detail={q.data && !q.data.ok ? q.data.error : undefined} />
          </div>
        ) : !q.data && rows.length === 0 ? (
          <div className="p-4">
            <div className="h-4 w-40 animate-pulse rounded bg-surface-2" />
          </div>
        ) : rows.length === 0 ? (
          <p className="p-4 text-[12.5px] text-muted">{t("empty")}</p>
        ) : (
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="border-b border-line text-left text-[11px] uppercase tracking-[.08em] text-muted">
                <th className="w-8 px-3 py-2" />
                <th className="px-3 py-2 font-medium">{t("colWhen")}</th>
                <th className="px-3 py-2 font-medium">{t("colKind")}</th>
                <th className="px-3 py-2 font-medium">{t("colWhat")}</th>
                <th className="px-3 py-2 font-medium">{t("colStatus")}</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <RowPair
                  key={r.id}
                  row={r}
                  open={display.openId === r.id}
                  onToggle={() => dispatch({ type: "open", id: display.openId === r.id ? null : r.id })}
                />
              ))}
            </tbody>
          </table>
        )}
      </div>

      {display.error ? (
        <div className="flex items-center gap-2 text-[12px] text-danger">
          <span>{t("loadMoreError")}</span>
          <button
            type="button"
            onClick={loadMore}
            className="rounded-md border border-line px-2 py-1 font-medium text-body-ink hover:bg-surface-2"
          >
            {t("retry")}
          </button>
        </div>
      ) : hasMore ? (
        <button
          type="button"
          onClick={loadMore}
          disabled={display.loadMorePending}
          className="self-start rounded-md border border-line px-3 py-1.5 text-[12px] font-medium text-body-ink transition-transform hover:bg-surface-2 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {t("loadMore")}
        </button>
      ) : null}
    </div>
  );
}

export default function AuditPage() {
  return (
    <Suspense>
      <AuditPageInner />
    </Suspense>
  );
}