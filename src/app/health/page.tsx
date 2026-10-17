"use client";

import { useQuery } from "@tanstack/react-query";
import { Cpu, Gauge, HardDrive, MemoryStick, Server } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useTranslations } from "next-intl";
import { Suspense } from "react";
import { fetchEnvelope } from "@/lib/api";
import { parseHealthSearch, searchHref, serializeHealthSearch } from "@/lib/cockpitUrl";
import { fmtUptime } from "@/lib/format";
import type { HealthPayload, HealthRange } from "@/server/db/metrics"; // type-only
import { LineChart, Sparkline } from "@/components/health/Charts";
import { RelativeTime } from "@/components/RelativeTime";
import { SourceWarning } from "@/components/mission/SourceWarning";

const RANGES: HealthRange[] = ["1h", "24h", "7d", "90d"];
const STALE_S = 180;

function fmtGb(mb: number | null | undefined): string {
  return mb === null || mb === undefined ? "—" : `${(mb / 1024).toFixed(1)} GB`;
}

const DOT: Record<string, string> = {
  active: "bg-success",
  running: "bg-success",
  inactive: "bg-surface-3",
  failed: "bg-danger",
  unknown: "bg-warning",
};

function HealthPageInner() {
  const t = useTranslations("health");
  const tTitle = useTranslations("title");
  const params = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const range = parseHealthSearch(new URLSearchParams(params.toString())).range;
  function setRange(next: HealthRange) {
    router.replace(searchHref(pathname, serializeHealthSearch(new URLSearchParams(params.toString()), { range: next })), { scroll: false });
  }

  const q = useQuery({
    queryKey: ["health", range],
    queryFn: ({ signal }) => fetchEnvelope<HealthPayload>(`/api/health?range=${range}`, signal),
    refetchInterval: 10_000,
  });
  const data = q.data?.ok ? q.data.data : null;
  const pollFailed = q.isError && !data;
  const stale = data !== null && data.collectedAgoS !== null && data.collectedAgoS > STALE_S;

  if (pollFailed) {
    return (
      <div className="p-1">
        <SourceWarning label={t("unavailable")} detail={q.error instanceof Error ? q.error.message : undefined} />
      </div>
    );
  }
  if (!data) {
    return <div className="h-6 animate-pulse rounded bg-surface-2" />;
  }
  if (data.latest === null) {
    return (
      <div className="flex items-center gap-3 p-1 text-[13px] text-muted">
        <Server className="h-4 w-4" />
        {t("waitingFirstSample")}
      </div>
    );
  }

  const l = data.latest;
  const diskPct =
    l.diskUsedGb !== null && l.diskTotalGb ? Math.round((l.diskUsedGb / l.diskTotalGb) * 100) : null;

  const tiles = [
    { key: "cpu", icon: Cpu, value: l.cpuPct !== null ? `${l.cpuPct.toFixed(0)}%` : "—", spark: data.spark.map((p) => p.cpuPct) },
    { key: "mem", icon: MemoryStick, value: `${fmtGb(l.memUsedMb)} / ${fmtGb(l.memTotalMb)}`, spark: data.spark.map((p) => p.memUsedMb) },
    { key: "disk", icon: HardDrive, value: l.diskUsedGb !== null ? `${l.diskUsedGb.toFixed(0)} / ${l.diskTotalGb?.toFixed(0)} GB (${diskPct}%)` : "—", spark: data.spark.map((p) => p.diskUsedGb) },
    { key: "load", icon: Gauge, value: l.load5 !== null ? `${l.load5.toFixed(2)} / ${data.cores}` : "—", spark: data.spark.map((p) => p.load5) },
  ] as const;

  return (
    <div className={`mx-auto flex w-full max-w-[1320px] flex-col gap-[22px] [animation:jax-rise_.4s_ease] ${stale ? "opacity-90" : ""}`}>
      <h1 className="sr-only">{tTitle("server")}</h1>
      {q.isError ? (
        <SourceWarning
          label={t("unavailable")}
          aside={q.dataUpdatedAt ? <RelativeTime epochMs={q.dataUpdatedAt} /> : undefined}
        />
      ) : stale ? (
        <SourceWarning label={t("collectorDown")} detail={t("lastSampleAgo", { min: Math.round((data.collectedAgoS ?? 0) / 60) })} />
      ) : null}

      <div className="flex items-center justify-between text-[12px] text-muted">
        <span>{t("uptime", { value: fmtUptime(l.uptimeS) })}</span>
        <span className="flex items-center gap-1">
          {t("collectedAgo")} <RelativeTime epochMs={Date.parse(l.ts)} />
        </span>
      </div>

      <div className={`grid grid-cols-2 gap-4 xl:grid-cols-4 ${stale ? "opacity-60" : ""}`}>
        {tiles.map((tile) => (
          <div key={tile.key} className="flex flex-col gap-2 rounded-lg border border-line bg-surface px-5 py-[18px]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted">{t(`tiles.${tile.key}`)}</span>
              <tile.icon className="h-4 w-4 text-muted" />
            </div>
            <span className="font-display text-[26px] font-black leading-none text-ink">{tile.value}</span>
            <Sparkline values={[...tile.spark]} />
          </div>
        ))}
      </div>

      <div className="flex items-center gap-1">
        {RANGES.map((r) => (
          <button
            key={r}
            onClick={() => setRange(r)}
            className={`rounded-md border px-2.5 py-1 text-[11px] font-medium transition-colors ${
              range === r ? "border-brand-soft-border bg-brand-soft text-brand" : "border-line text-muted hover:text-body-ink"
            }`}
          >
            {t(`ranges.${r}`)}
          </button>
        ))}
        {range === "90d" ? (
          <span className="ml-3 flex items-center gap-3 text-[11px] text-muted">
            <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-4 bg-current" /> {t("legendAvg")}</span>
            <span className="flex items-center gap-1"><span className="inline-block h-0.5 w-4 bg-current opacity-35" /> {t("legendMax")}</span>
          </span>
        ) : null}
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <div className="rounded-lg border border-line bg-surface p-4">
          <span className="text-xs font-medium text-muted">{t("charts.cpu")}</span>
          <LineChart points={data.series} maxPoints={data.seriesMax} field="cpuPct" colorClass="text-brand" format={(v) => `${v.toFixed(0)}%`} />
        </div>
        <div className="rounded-lg border border-line bg-surface p-4">
          <span className="text-xs font-medium text-muted">{t("charts.mem")}</span>
          <LineChart points={data.series} maxPoints={data.seriesMax} field="memUsedMb" colorClass="text-accent" format={(v) => fmtGb(v)} />
        </div>
      </div>

      {data.services ? (
        <div className="rounded-lg border border-line bg-surface p-4">
          <span className="text-xs font-medium text-muted">{t("services")}</span>
          <div className="mt-3 grid grid-cols-2 gap-x-6 gap-y-2 md:grid-cols-3 xl:grid-cols-4">
            {Object.entries(data.services.units).map(([name, state]) => (
              <span key={name} className="flex items-center gap-2 text-[12.5px] text-body-ink" title={state}>
                <span className={`h-1.5 w-1.5 flex-none rounded-full ${DOT[state] ?? DOT.unknown}`} />
                <span className="truncate font-mono text-[12px]">{name.replace(/\.(service|timer)$/, "")}</span>
              </span>
            ))}
          </div>
          {data.services.containers.length > 0 ? (
            <>
              <span className="mt-4 block text-xs font-medium text-muted">{t("containers")}</span>
              <div className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1.5 md:grid-cols-3 xl:grid-cols-4">
                {data.services.containers.map((c) => (
                  <span key={c.name} className="flex items-center gap-2 text-[12px] text-muted" title={c.state}>
                    <span className={`h-1.5 w-1.5 flex-none rounded-full ${DOT[c.state] ?? DOT.unknown}`} />
                    <span className="truncate font-mono">{c.name}</span>
                  </span>
                ))}
              </div>
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export default function HealthPage() {
  return (
    <Suspense>
      <HealthPageInner />
    </Suspense>
  );
}
