"use client";

import { useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { RelativeTime } from "@/components/RelativeTime";
import { useTokensApi } from "@/lib/useTokensApi";
import {
  type NativeUsageSnapshot,
  type RangeKey,
  type UsageSlice,
} from "@/server/collectors/tokens-usage";
import type { SourceCoverage, SourceState } from "@/server/collectors/native-usage";

const RANGE_KEYS: RangeKey[] = ["7d", "30d", "90d"];
const UNKNOWN_MODEL_FILTER = "__unknown__";
const DATA_COLORS = ["--data-coral", "--data-sage", "--data-amber", "--data-teal", "--data-plum"];

const fmtTokens = (n: number, locale: string) =>
  new Intl.NumberFormat(locale, { notation: "compact", maximumFractionDigits: 1 }).format(n);
const fmtMoney = (n: number, currency: string, locale: string) =>
  new Intl.NumberFormat(locale, { style: "currency", currency }).format(n);

const STATE_STYLE: Record<SourceState, string> = {
  available: "bg-success-soft text-success",
  empty: "bg-surface-3 text-muted",
  indexing: "bg-warning-soft text-warning",
  unavailable: "bg-danger-soft text-danger",
  incompatible: "bg-danger-soft text-danger",
  disabled: "bg-surface-3 text-muted",
};

export function usagePath(range: RangeKey, agent: string, model: string): string {
  const params = new URLSearchParams({ range });
  if (agent) params.set("agent", agent);
  if (model) params.set("model", model);
  return `usage?${params.toString()}`;
}

function BurnChart({ daily, locale }: { daily: NativeUsageSnapshot["daily"]; locale: string }) {
  const max = Math.max(1, ...daily.map((d) => d.tokens));
  const week = daily.length <= 7;
  const labelEvery = Math.ceil(daily.length / 7);
  const dateOpts: Intl.DateTimeFormatOptions = week
    ? { weekday: "short", timeZone: "UTC" }
    : { day: "2-digit", month: "2-digit", timeZone: "UTC" };
  return (
    <div className="flex h-[150px] items-end gap-[3px]">
      {daily.map((d, i) => {
        const last = i === daily.length - 1;
        return (
          <div key={d.date} className="flex h-full min-w-0 flex-1 flex-col items-center justify-end gap-1.5">
            {week ? (
              <span className="font-mono text-[10.5px] text-muted">{fmtTokens(d.tokens, locale)}</span>
            ) : null}
            <div
              className={`w-full rounded-t-lg ${last ? "bg-brand" : ""}`}
              style={{
                height: `${Math.max(4, (d.tokens / max) * 100)}%`,
                background: last ? undefined : "linear-gradient(180deg, var(--warm-600), var(--warm-700))",
              }}
            />
            <span className="text-[11px] text-muted">
              {i % labelEvery === 0 || last
                ? new Date(`${d.date}T00:00:00Z`).toLocaleDateString(locale, dateOpts)
                : " "}
            </span>
          </div>
        );
      })}
    </div>
  );
}

function BreakdownPanel({
  title,
  slices,
  locale,
  currency,
  unknownLabel,
}: {
  title: string;
  slices: UsageSlice[];
  locale: string;
  currency: string | null;
  unknownLabel: string;
}) {
  const t = useTranslations("tokens.usage");
  const max = Math.max(1, ...slices.map((s) => s.tokens));
  return (
    <div className="flex min-w-0 flex-col gap-4 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <span className="text-sm font-bold text-ink">{title}</span>
      {slices.length === 0 ? <span className="text-xs text-muted">{t("empty")}</span> : null}
      {slices.map((s, i) => {
        const color = `var(${DATA_COLORS[i % DATA_COLORS.length]})`;
        const name = s.name === UNKNOWN_MODEL_FILTER ? unknownLabel : s.name;
        return (
          <div key={s.name} className="flex min-w-0 flex-col gap-[7px]">
            <div className="flex min-w-0 items-center gap-2">
              <span className="h-[9px] w-[9px] flex-none rounded-[3px]" style={{ background: color }} />
              <span className="min-w-0 truncate text-[12.5px] font-medium text-body-ink">{name}</span>
              <span className="ml-auto flex-none font-mono text-[11.5px] text-muted">{fmtTokens(s.tokens, locale)}</span>
              <span className="w-[64px] flex-none text-right font-mono text-[11.5px] text-ink">
                {s.cost !== null && s.currency ? fmtMoney(s.cost, s.currency, locale) : "—"}
              </span>
            </div>
            <div className="h-[7px] overflow-hidden rounded-full bg-surface-inset">
              <div className="h-full rounded-full" style={{ width: `${(s.tokens / max) * 100}%`, background: color }} />
            </div>
          </div>
        );
      })}
      {slices.some((s) => s.cost === null) ? (
        <span className="text-[11px] text-muted">{currency ? t("partialCost") : t("costUnavailable")}</span>
      ) : null}
    </div>
  );
}

function CoverageRow({ source }: { source: SourceCoverage }) {
  const t = useTranslations("tokens.usage");
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-x-2.5 gap-y-1 text-[11.5px]">
      <span className="font-semibold uppercase tracking-wide text-body-ink">{t(`source.${source.source}`)}</span>
      <span className={`rounded-full px-2 py-0.5 text-[10.5px] font-semibold ${STATE_STYLE[source.state]}`}>
        {t(`state.${source.state}`)}
      </span>
      {source.state === "indexing" ? (
        <span className="text-muted">
          {t("indexing")}
          {source.progress ? ` · ${source.progress.filesDone}/${source.progress.filesTotal}` : ""}
        </span>
      ) : null}
      {source.lastRefresh ? (
        <span className="text-muted">
          {t("lastRefresh")} <RelativeTime epochMs={source.lastRefresh * 1000} />
        </span>
      ) : null}
      {source.warnings.map((warning) => (
        <span key={warning} className="rounded bg-surface-3 px-1.5 py-0.5 font-mono text-[10px] text-warning">
          {t.has(`warning.${warning}`) ? t(`warning.${warning}`) : warning}
        </span>
      ))}
    </div>
  );
}

export function UsageSection() {
  const t = useTranslations("tokens.usage");
  const locale = useLocale();
  const [range, setRange] = useState<RangeKey>("7d");
  const [agent, setAgent] = useState("");
  const [model, setModel] = useState("");
  const path = usagePath(range, agent, model);
  const { data, isError, dataUpdatedAt } = useTokensApi<NativeUsageSnapshot>(path, (envelope) =>
    envelope?.data?.history.indexing ? 1_000 : 60_000,
  );
  const snapshot = data?.ok ? data.data : undefined;
  const healthySources =
    snapshot?.sources.some((s) => s.state === "available" || s.state === "empty") ?? false;
  const everySourceUnavailable =
    snapshot !== undefined &&
    !snapshot.sources.some((s) => s.state === "available" || s.state === "empty" || s.state === "indexing");

  return (
    <div className="flex min-w-0 flex-col gap-5">
      <div className="flex min-w-0 flex-wrap items-center gap-3">
        <div className="flex gap-1 rounded-full border border-line bg-surface-2 p-1">
          {RANGE_KEYS.map((r) => (
            <button
              key={r}
              onClick={() => setRange(r)}
              className={`h-[30px] rounded-full px-4 text-xs font-semibold ${range === r ? "bg-brand text-on-brand" : "text-muted"}`}
            >
              {r}
            </button>
          ))}
        </div>
        <label className="flex min-w-0 items-center gap-2 text-xs text-muted">
          {t("agent")}
          <select
            value={agent}
            onChange={(event) => setAgent(event.target.value)}
            className="h-[30px] max-w-[180px] rounded-md border border-line bg-surface px-2 text-xs text-body-ink"
          >
            <option value="">{t("allAgents")}</option>
            {(snapshot?.filters.agents ?? []).map((name) => (
              <option key={name} value={name}>{name}</option>
            ))}
          </select>
        </label>
        <label className="flex min-w-0 items-center gap-2 text-xs text-muted">
          {t("model")}
          <select
            value={model}
            onChange={(event) => setModel(event.target.value)}
            className="h-[30px] max-w-[200px] rounded-md border border-line bg-surface px-2 text-xs text-body-ink"
          >
            <option value="">{t("allModels")}</option>
            {(snapshot?.filters.models ?? []).map((name) => (
              <option key={name} value={name}>
                {name === UNKNOWN_MODEL_FILTER ? t("unknownModel") : name}
              </option>
            ))}
          </select>
        </label>
        {snapshot ? (
          <div className="ml-auto flex min-w-0 flex-wrap items-baseline gap-2.5">
            <span className="text-[13px] text-muted">{t("total")}</span>
            <span className="font-display text-[26px] font-black text-ink">{fmtTokens(snapshot.totalTokens, locale)}</span>
            <span className="text-[11.5px] text-muted">
              {t("input")} {fmtTokens(snapshot.totalInput, locale)} · {t("output")} {fmtTokens(snapshot.totalOutput, locale)}
            </span>
            <span className="text-[13px] text-muted">·</span>
            <span className="font-display text-[26px] font-black text-brand">
              {snapshot.totalCost !== null && snapshot.costCurrency
                ? fmtMoney(snapshot.totalCost, snapshot.costCurrency, locale)
                : "—"}
            </span>
            <span className="text-[11px] text-muted">{t("cost")}</span>
          </div>
        ) : null}
      </div>

      {isError ? (
        <SourceWarning
          label={t("unavailable")}
          aside={dataUpdatedAt ? <RelativeTime epochMs={dataUpdatedAt} /> : undefined}
        />
      ) : null}

      {snapshot ? (
        <>
          <div className="flex min-w-0 flex-col gap-2 rounded-lg border border-line bg-surface px-[18px] py-3.5">
            <span className="text-xs font-bold text-ink">{t("sources")}</span>
            {snapshot.sources.map((source) => (
              <CoverageRow key={source.source} source={source} />
            ))}
          </div>

          {everySourceUnavailable ? (
            <SourceWarning label={t("allUnavailable")} />
          ) : (
            <>
              <div className="flex min-w-0 flex-col gap-[18px] rounded-lg border border-line bg-surface px-[22px] py-5">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <span className="text-sm font-bold text-ink">{t("dailyBurn")}</span>
                  <span className="text-[11px] text-muted">{t("sessions")}: {t("leadSessions")} {snapshot.sessions.lead} · {t("childSessions")} {snapshot.sessions.child}</span>
                </div>
                {snapshot.totalTokens === 0 ? (
                  <span className="text-xs text-muted">{t("empty")}</span>
                ) : (
                  <BurnChart daily={snapshot.daily} locale={locale} />
                )}
              </div>
              <div className="grid min-w-0 gap-4 lg:grid-cols-2">
                <BreakdownPanel title={t("byModel")} slices={snapshot.byModel} locale={locale} currency={snapshot.costCurrency} unknownLabel={t("unknownModel")} />
                <BreakdownPanel title={t("byAgent")} slices={snapshot.byAgent} locale={locale} currency={snapshot.costCurrency} unknownLabel={t("unknownModel")} />
              </div>
            </>
          )}

          <div className="flex flex-col gap-1 text-[11px] text-muted">
            <span>{t("utcNote")}</span>
            <span>{t("historyNote", { cutoff: snapshot.history.cutoffDay })}</span>
            <span>{t("cacheNote")}</span>
            {snapshot.costCoverage.measured > 0 ? (
              <span>{t("costCoverage", { measured: snapshot.costCoverage.measured, total: snapshot.costCoverage.total })}</span>
            ) : null}
            {snapshot.mixedProjectionSessions > 0 ? (
              <span>{t("mixedProjection", { count: snapshot.mixedProjectionSessions })}</span>
            ) : null}
            {!healthySources && !everySourceUnavailable ? <span>{t("noHealthySource")}</span> : null}
          </div>
        </>
      ) : null}
    </div>
  );
}
