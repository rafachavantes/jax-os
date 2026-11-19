"use client";

import { useTranslations } from "next-intl";
import { useQuery } from "@tanstack/react-query";
import { Cpu, HardDrive, MemoryStick } from "lucide-react";
import { fetchEnvelope } from "@/lib/api";
import { RelativeTime } from "@/components/RelativeTime";
import type { HealthPayload, HealthLatest } from "@/server/db/metrics";
import type { Alert, PackageRow, ZombieRow } from "@/server/collectors/alerts";
import type { ResourcesSection } from "@/server/collectors/environment";
import { SourceWarning } from "./SourceWarning";

const VISIBLE_PACKAGES = 3;

// Binary memory is explicit MiB/GiB; never a decimal "GB" label for RAM.
function fmtMem(mb: number | null | undefined): string {
  if (mb === null || mb === undefined) return "—";
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GiB` : `${mb.toFixed(0)} MiB`;
}

function fmtVal(n: number | null | undefined, digits = 0): string {
  return n != null ? n.toFixed(digits) : "—";
}

// Shared with Health's own tiles: each meter formats independently, so a null total never
// renders "42 / undefined GB".
export function meterValues(latest: HealthLatest | null, cores: number | null) {
  return {
    cpu: latest?.cpuPct != null ? `${latest.cpuPct.toFixed(0)}%` : "—",
    mem: latest ? `${fmtMem(latest.memUsedMb)} / ${fmtMem(latest.memTotalMb)}` : "—",
    disk: `${fmtVal(latest?.diskUsedGb)} / ${fmtVal(latest?.diskTotalGb)} GB`,
    load: latest?.load5 != null ? `${latest.load5.toFixed(2)} / ${cores ?? "—"}` : "—",
  };
}

export function meterPercent(used: number | null, total: number | null): number | null {
  if (typeof used !== "number" || typeof total !== "number" || !Number.isFinite(used) || !Number.isFinite(total) || total <= 0) return null;
  return Math.min(100, Math.max(0, (used / total) * 100));
}

// One group per distinct ppid, never client-side aggregation beyond a grouping pass over the
// collector's already-flat rows.
export function groupZombies(rows: ZombieRow[]): { ppid: number; parentCommand: string | null; zombies: ZombieRow[] }[] {
  const byParent = new Map<number, ZombieRow[]>();
  for (const row of rows) {
    const list = byParent.get(row.ppid) ?? [];
    list.push(row);
    byParent.set(row.ppid, list);
  }
  return [...byParent.entries()].map(([ppid, zombies]) => ({ ppid, parentCommand: zombies[0].parentCommand, zombies }));
}

function alertRowsOf(data: { ok: true; data: Alert[] } | undefined): Alert[] {
  return Array.isArray(data?.data) ? data.data : [];
}

function alertCount(rows: Alert[]): number {
  return rows.reduce((sum, alert) => sum + ("count" in alert && typeof alert.count === "number" ? alert.count : 0), 0);
}

// Compact home resource strip (M2 plan Task 5): glanceable CPU/RAM/root disk plus an explicit
// /tmp unavailable value (collector lands with M5 — never substituted disk/Shmem) and the alert
// count linking to the read-only Health detail. Lives in its own 10s poll; a slow GitHub call
// never gates it. Sticky without covering controls (background + max z-index tier).
export function ServerPanel() {
  const t = useTranslations("mission.server");
  const health = useQuery({
    queryKey: ["health", "24h"],
    queryFn: ({ signal }) => fetchEnvelope<HealthPayload>("/api/health", signal),
    refetchInterval: 10_000,
  });
  const alerts = useQuery({
    queryKey: ["mission", "alerts"],
    queryFn: ({ signal }) => fetchEnvelope<Alert[]>("/api/mission/alerts", signal),
    refetchInterval: 60_000,
  });
  // M5: the independent actual /tmp observation (never the root disk, never Shmem).
  const environment = useQuery({
    queryKey: ["environment", "resources"],
    queryFn: ({ signal }) => fetchEnvelope<ResourcesSection>("/api/environment?section=resources", signal),
    refetchInterval: 10_000,
  });

  const healthPending = health.data === undefined && !health.isError;
  const healthData = health.data?.ok ? health.data.data : null;
  const latest = healthData?.latest ?? null;
  const mv = {
    cpu: latest?.cpuPct != null ? `${latest.cpuPct.toFixed(0)}%` : "—",
    mem: latest ? `${fmtMem(latest.memUsedMb)} / ${fmtMem(latest.memTotalMb)}` : "—",
    disk: latest?.diskUsedGb != null && latest?.diskTotalGb != null
      ? `${Math.round((latest.diskUsedGb / latest.diskTotalGb) * 100)}%`
      : "—",
  };
  const rows = alertRowsOf(alerts.data);
  const envData = environment.data?.ok ? environment.data.data : null;
  const tmp = envData?.tmp ?? null;
  const tmpPending = environment.data === undefined && !environment.isError;
  const tmpStale = envData?.stale === true;
  const alertsReady = alerts.data !== undefined && alerts.data.ok === true;
  const meters = [
    { key: "cpu" as const, icon: Cpu, value: mv.cpu },
    { key: "mem" as const, icon: MemoryStick, value: mv.mem },
    { key: "disk" as const, icon: HardDrive, value: mv.disk },
  ];

  return (
    <section data-testid="server-strip" className="sticky top-0 z-10 flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1 rounded-lg border border-line bg-surface px-3.5 py-2">
      <span className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">{t("title")}</span>
      {health.isError ? (
        <SourceWarning label={t("metersUnavailable")} />
      ) : healthPending ? (
        <span className="h-4 w-48 animate-pulse rounded bg-surface-2" />
      ) : (
        meters.map(({ key, icon: Icon, value }) => (
          <a key={key} href={`/health#${key}`} className="flex items-center gap-1 font-mono text-[11px] text-muted hover:text-body-ink">
            <Icon className="h-3.5 w-3.5" aria-hidden />
            <span>{t(`meters.${key}`)}</span>
            <span className="font-semibold text-ink">{value}</span>
          </a>
        ))
      )}
      <a href="/health#tmp" className="flex items-center gap-1 font-mono text-[11px] text-muted hover:text-body-ink">
        <span>{t("temp")}</span>
        {tmpPending ? (
          <span className="h-4 w-24 animate-pulse rounded bg-surface-2" />
        ) : (
          <span className="font-semibold text-ink">
            {tmp ? `${tmp.usedGb.toFixed(1)} / ${tmp.totalGb.toFixed(1)} GB` : t("tempUnavailable")}
          </span>
        )}
      </a>
      {tmpStale && tmp ? (
        <span className="flex items-center gap-1 text-[11px] text-warning">
          {t("tempStale")}
          {envData?.lastGoodAt ? <> · <RelativeTime epochMs={Date.parse(envData.lastGoodAt)} /></> : null}
        </span>
      ) : null}
      {alerts.isError ? (
        <SourceWarning label={t("alertsUnavailable")} />
      ) : !alertsReady ? (
        <span className="h-4 w-20 animate-pulse rounded bg-surface-2" />
      ) : (
        <a href="/health#alerts" className="ml-auto flex items-center gap-1 rounded-full border border-line px-2 py-0.5 text-[11px] font-semibold text-body-ink hover:border-line-strong">
          {t("alertsCount", { count: alertCount(rows) })}
        </a>
      )}
    </section>
  );
}

// The read-only alert detail reused on Health (plan F4): existing sources stay readable, and
// NO kill/upgrade commands are copied here — known candidates wait for M5 diagnostics.
export function ServerAlerts() {
  const t = useTranslations("mission.server");
  const alerts = useQuery({
    queryKey: ["mission", "alerts"],
    queryFn: ({ signal }) => fetchEnvelope<Alert[]>("/api/mission/alerts", signal),
    refetchInterval: 60_000,
  });
  const pending = alerts.data === undefined && !alerts.isError;
  const rows = alertRowsOf(alerts.data);
  const zombiesAlert = rows.find((a): a is Extract<Alert, { kind: "zombies" }> => a.kind === "zombies");
  const updatesAlert = rows.find((a): a is Extract<Alert, { kind: "updates" }> => a.kind === "updates");
  const zombieGroups = zombiesAlert ? groupZombies(zombiesAlert.rows) : [];
  const failedSources = rows
    .filter((a): a is Extract<Alert, { kind: "source" }> => a.kind === "source")
    .map((a) => a.source);
  const packages: PackageRow[] = updatesAlert?.rows ?? [];
  const securityNames = packages.filter((p) => p.security).map((p) => p.name);
  const visiblePackages = packages.slice(0, VISIBLE_PACKAGES);
  const hiddenPackageCount = Math.max(0, packages.length - VISIBLE_PACKAGES);

  return (
    <section id="alerts" className="flex flex-col gap-4 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <span className="text-sm font-bold text-ink">{t("alertsTitle")}</span>
      {alerts.isError ? <SourceWarning label={t("alertsUnavailable")} /> : null}
      {failedSources.map((source) => (
        <SourceWarning key={source} label={t("sourceFailed", { source: t(`sourceNames.${source}`) })} />
      ))}
      {pending ? (
        <span className="h-6 animate-pulse rounded bg-surface-2" />
      ) : (
        <>
          <div className="flex flex-col gap-2">
            <span className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">
              {zombieGroups.length > 0 ? t("zombiesCount", { count: zombiesAlert?.count ?? zombieGroups.length }) : t("zombies")}
            </span>
            {zombieGroups.length === 0 ? (
              <span className="text-[12.5px] text-muted">{t("zombiesEmpty")}</span>
            ) : (
              zombieGroups.map((g) => (
                <div key={g.ppid} className="flex flex-wrap items-center gap-2 rounded-md border border-line-subtle px-3 py-2 text-[12px] text-body-ink">
                  <span className="min-w-0 truncate font-mono">{g.parentCommand ?? t("zombieUnknownParent")} · {g.ppid}</span>
                  <span className="text-[11px] text-muted">{t("zombieChildren", { count: g.zombies.length })}</span>
                </div>
              ))
            )}
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2">
              <span className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">
                {packages.length > 0 ? t("packagesCount", { count: packages.length }) : t("packages")}
              </span>
              {securityNames.length > 0 ? (
                <span className="text-[11px] text-muted">{t("packagesSecurityCount", { count: securityNames.length })}</span>
              ) : null}
            </div>
            {packages.length === 0 ? (
              <span className="text-[12.5px] text-muted">{t("packagesEmpty")}</span>
            ) : (
              <>
                {visiblePackages.map((p) => (
                  <div key={p.name} className="flex min-w-0 items-center gap-2 text-[12px] text-body-ink">
                    {p.security ? <span className="flex-none rounded-full bg-danger-soft px-1.5 py-0 text-[10px] font-bold text-danger">{t("security")}</span> : null}
                    <span className="min-w-0 flex-1 truncate font-mono">{p.name} {p.fromVersion} → {p.toVersion}</span>
                  </div>
                ))}
                {hiddenPackageCount > 0 ? <span className="text-[11px] text-muted">{t("packagesMore", { count: hiddenPackageCount })}</span> : null}
              </>
            )}
          </div>
          <span className="text-[10.5px] italic text-muted">{t("readOnlyNote")}</span>
        </>
      )}
    </section>
  );
}
