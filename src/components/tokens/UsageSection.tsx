"use client";

import { useState } from "react";
import { useLocale, useTranslations } from "next-intl";
import { SourceWarning } from "@/components/mission/SourceWarning";
import { RelativeTime } from "@/components/RelativeTime";
import { useTokensApi } from "@/lib/useTokensApi";
import type { RangeKey, Slice, UsagePayload } from "@/server/collectors/tokens-usage";

const RANGE_KEYS: RangeKey[] = ["7d", "30d", "90d"];
const DATA_COLORS = ["--data-coral", "--data-sage", "--data-amber", "--data-teal", "--data-plum"];

const fmtTokens = (n: number, locale: string) =>
  new Intl.NumberFormat(locale, { notation: "compact", maximumFractionDigits: 1 }).format(n);
const fmtCost = (n: number, locale: string) =>
  new Intl.NumberFormat(locale, { style: "currency", currency: "USD" }).format(n);

function BurnChart({ daily, locale }: { daily: UsagePayload["daily"]; locale: string }) {
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
          <div key={d.date} className="flex h-full flex-1 flex-col items-center justify-end gap-1.5">
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

function BreakdownPanel({ title, slices, locale }: { title: string; slices: Slice[]; locale: string }) {
  const max = Math.max(1, ...slices.map((s) => s.tokens));
  return (
    <div className="flex flex-col gap-4 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <span className="text-sm font-bold text-ink">{title}</span>
      {slices.map((s, i) => {
        const color = `var(${DATA_COLORS[i % DATA_COLORS.length]})`;
        return (
          <div key={s.name} className="flex flex-col gap-[7px]">
            <div className="flex items-center gap-2">
              <span className="h-[9px] w-[9px] flex-none rounded-[3px]" style={{ background: color }} />
              <span className="text-[12.5px] font-medium text-body-ink">{s.name}</span>
              <span className="ml-auto font-mono text-[11.5px] text-muted">{fmtTokens(s.tokens, locale)}</span>
              {s.cost > 0 ? (
                <span className="w-[58px] text-right font-mono text-[11.5px] text-ink">{fmtCost(s.cost, locale)}</span>
              ) : null}
            </div>
            <div className="h-[7px] overflow-hidden rounded-full bg-surface-inset">
              <div className="h-full rounded-full" style={{ width: `${(s.tokens / max) * 100}%`, background: color }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function UsageSection() {
  const t = useTranslations("tokens.usage");
  const locale = useLocale();
  const [range, setRange] = useState<RangeKey>("7d");
  const { data, isError, dataUpdatedAt } = useTokensApi<UsagePayload>(`usage?range=${range}`, 60_000);

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-3.5">
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
        {data?.ok ? (
          <>
            <span className="ml-auto text-[13px] text-muted">{t("total")}</span>
            <span className="font-display text-[26px] font-black text-ink">{fmtTokens(data.data.totalTokens, locale)}</span>
            {data.data.totalCost > 0 ? (
              <>
                <span className="text-[13px] text-muted">·</span>
                <span className="font-display text-[26px] font-black text-brand">{fmtCost(data.data.totalCost, locale)}</span>
              </>
            ) : null}
          </>
        ) : null}
      </div>

      {isError ? (
        <SourceWarning
          label={t("unavailable")}
          aside={data?.ok && dataUpdatedAt ? <RelativeTime epochMs={dataUpdatedAt} /> : undefined}
        />
      ) : null}

      {data?.ok ? (
        <>
          <div className="flex flex-col gap-[18px] rounded-lg border border-line bg-surface px-[22px] py-5">
            <span className="text-sm font-bold text-ink">{t("dailyBurn")}</span>
            {data.data.totalTokens === 0 ? (
              <span className="text-xs text-muted">{t("empty")}</span>
            ) : (
              <BurnChart daily={data.data.daily} locale={locale} />
            )}
          </div>
          <div className="grid gap-4 lg:grid-cols-2">
            <BreakdownPanel title={t("byModel")} slices={data.data.byModel} locale={locale} />
            <BreakdownPanel title={t("byAgent")} slices={data.data.byAgent} locale={locale} />
          </div>
          {data.data.skipped.length > 0 ? (
            <span className="text-xs text-muted">
              {t("skipped", { count: data.data.skipped.length, names: data.data.skipped.join(", ") })}
            </span>
          ) : null}
        </>
      ) : null}
    </div>
  );
}
