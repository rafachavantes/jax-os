"use client";

import { useTranslations } from "next-intl";
import { useQuery } from "@tanstack/react-query";
import { fetchEnvelope } from "@/lib/api";
import { fmtUptime, meterColorClass, meterValuesCompact } from "@/lib/format";
import type { HealthPayload, HealthLatest } from "@/server/db/metrics";
import type { Alert, PackageRow, ZombieRow } from "@/server/collectors/alerts";
import { SourceWarning } from "./SourceWarning";

// Blueprint mockup shows ~3 rows then a "+N more" line (audit 2026-09-19).
const VISIBLE_PACKAGES = 3;

function fmtGb(mb: number | null | undefined): string {
  return mb === null || mb === undefined ? "—" : `${(mb / 1024).toFixed(1)} GB`;
}

function fmtVal(n: number | null | undefined, digits = 0): string {
  return n != null ? n.toFixed(digits) : "—";
}

// round-1 F6: pure and exported for its own test — each meter (and each side of the
// disk pair) formats independently, so a null total never renders "42 / undefined GB".
export function meterValues(latest: HealthLatest | null, cores: number | null) {
  return {
    cpu: latest?.cpuPct != null ? `${latest.cpuPct.toFixed(0)}%` : "—",
    mem: latest ? `${fmtGb(latest.memUsedMb)} / ${fmtGb(latest.memTotalMb)}` : "—",
    disk: `${fmtVal(latest?.diskUsedGb)} / ${fmtVal(latest?.diskTotalGb)} GB`,
    load: latest?.load5 != null ? `${latest.load5.toFixed(2)} / ${cores ?? "—"}` : "—",
  };
}

// Decision 22's meter percent formula (round-1 F5). Used/total are both number | null and can
// legitimately be 0 (src/server/db/metrics.ts) — a bar renders only when this returns non-null;
// meterValues' own numeric/"—" text always renders regardless (unchanged).
export function meterPercent(used: number | null, total: number | null): number | null {
  if (typeof used !== "number" || typeof total !== "number" || !Number.isFinite(used) || !Number.isFinite(total) || total <= 0) return null;
  return Math.min(100, Math.max(0, (used / total) * 100));
}

// Decision 4/§4: this handler NEVER runs the command — it only copies the string to
// the browser's Clipboard API. Exported so it is testable with an injected clipboard,
// the one allowed side-effect test in §11 (a plain function, no DOM/React needed).
export function copyCommand(command: string, clipboard: { writeText: (text: string) => void }): void {
  clipboard.writeText(command);
}

// Decision 17: one group per distinct ppid, never client-side aggregation beyond a
// grouping pass over the collector's already-flat rows.
export function groupZombies(rows: ZombieRow[]): { ppid: number; parentCommand: string | null; zombies: ZombieRow[] }[] {
  const byParent = new Map<number, ZombieRow[]>();
  for (const row of rows) {
    const list = byParent.get(row.ppid) ?? [];
    list.push(row);
    byParent.set(row.ppid, list);
  }
  return [...byParent.entries()].map(([ppid, zombies]) => ({ ppid, parentCommand: zombies[0].parentCommand, zombies }));
}

// Shared code-chip style for a copyable command (mockup's `.cmd`, audit 2026-09-19).
const CHIP_CLASS = "rounded border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-ink";
const COPY_BUTTON_CLASS = "rounded border border-line-strong px-2 py-0.5 text-[10.5px] font-semibold text-body-ink";

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

  // round-2 F4: "no data yet" (loading, nothing cached) differs from "resolved to zero
  // rows" — only the former gets a placeholder; an outright `isError` falls through
  // to the warning below instead.
  const healthPending = health.data === undefined && !health.isError;
  const healthData = health.data?.ok ? health.data.data : null;
  const latest = healthData?.latest ?? null;
  const cores = healthData?.cores ?? null;
  const hostname = healthData?.hostname ?? null;
  const mv = meterValuesCompact(latest, cores);
  const mvFull = meterValues(latest, cores);
  const meters = [
    { key: "cpu" as const, value: mv.cpu, fullValue: mvFull.cpu, pct: meterPercent(latest?.cpuPct ?? null, 100) },
    { key: "mem" as const, value: mv.mem, fullValue: mvFull.mem, pct: meterPercent(latest?.memUsedMb ?? null, latest?.memTotalMb ?? null) },
    { key: "disk" as const, value: mv.disk, fullValue: mvFull.disk, pct: meterPercent(latest?.diskUsedGb ?? null, latest?.diskTotalGb ?? null) },
    { key: "load" as const, value: mv.load, fullValue: mvFull.load, pct: meterPercent(latest?.load5 ?? null, cores) },
  ];

  const alertsPending = alerts.data === undefined && !alerts.isError;
  const alertRows = alerts.data?.ok ? alerts.data.data : [];
  const zombiesAlert = alertRows.find((a): a is Extract<Alert, { kind: "zombies" }> => a.kind === "zombies");
  const updatesAlert = alertRows.find((a): a is Extract<Alert, { kind: "updates" }> => a.kind === "updates");
  const zombieGroups = zombiesAlert ? groupZombies(zombiesAlert.rows) : [];
  const packages: PackageRow[] = updatesAlert?.rows ?? [];
  const securityNames = packages.filter((p) => p.security).map((p) => p.name);
  const visiblePackages = packages.slice(0, VISIBLE_PACKAGES);
  const hiddenPackageCount = Math.max(0, packages.length - VISIBLE_PACKAGES);
  const killCommand = (ppid: number) => `kill -HUP ${ppid}`;
  const aptCommand = `sudo apt upgrade ${securityNames.join(" ")}`;

  return (
    <section className="flex flex-col gap-4 rounded-lg border border-line bg-surface px-5 py-[18px]">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-sm font-bold text-ink">{t("title")}</span>
        {hostname ? (
          <span className="font-mono text-[10px] text-muted">{t("host", { hostname, uptime: fmtUptime(latest?.uptimeS) })}</span>
        ) : null}
      </div>

      {/* round-2 F3: `isError` alone — fetchEnvelope throws on {ok:false}, and TanStack
          keeps the last-good `data` on a failed refetch, so isError still shows here. */}
      {health.isError ? <SourceWarning label={t("metersUnavailable")} /> : null}
      {healthPending ? (
        // round-2 F4: existing skeleton pattern (ActivityFeed.tsx, InboxStrip.tsx) — never the empty/"—" values below.
        <div className="h-6 animate-pulse rounded bg-surface-2" />
      ) : (
        // Mockup layout (audit 2026-09-19): single-column rows, `label · bar · value`, not tiles.
        <div className="flex flex-col gap-1.5">
          {meters.map((m) => (
            <div key={m.key} className="grid grid-cols-[44px_1fr_auto] items-center gap-2 font-mono text-[11px] text-muted">
              <span>{t(`meters.${m.key}`)}</span>
              <div className="h-1.5 overflow-hidden rounded-full bg-surface-3">
                {m.pct !== null ? (
                  <div className={`h-full rounded-full ${meterColorClass(m.pct)}`} style={{ width: `${m.pct}%` }} />
                ) : null}
              </div>
              <span className="text-right font-semibold text-ink" title={m.fullValue}>{m.value}</span>
            </div>
          ))}
        </div>
      )}

      {alerts.isError ? <SourceWarning label={t("alertsUnavailable")} /> : null}

      <div className="flex flex-col gap-2">
        <span className="text-[11px] font-bold uppercase tracking-[.08em] text-muted">
          {zombieGroups.length > 0 ? t("zombiesCount", { count: zombiesAlert?.count ?? zombieGroups.length }) : t("zombies")}
        </span>
        {alertsPending ? (
          <div className="h-6 animate-pulse rounded bg-surface-2" />
        ) : zombieGroups.length === 0 ? (
          <span className="text-[12.5px] text-muted">{t("zombiesEmpty")}</span>
        ) : (
          <>
            {zombieGroups.map((g) => (
              <div key={g.ppid} className="flex flex-col gap-1 rounded-md border border-line-subtle px-3 py-2">
                <span className="font-mono text-[12px] text-body-ink">
                  {g.parentCommand ?? t("zombieUnknownParent")} · {g.ppid}
                </span>
                {g.zombies.map((z) => (
                  <span key={z.pid} className="pl-3 font-mono text-[11px] text-muted">
                    {z.pid} · {t("zombieAgeMin", { min: Math.round(z.ageSeconds / 60) })}
                  </span>
                ))}
                <div className="flex items-center gap-2 pl-3">
                  <span className={CHIP_CLASS}>{killCommand(g.ppid)}</span>
                  <button type="button" onClick={() => copyCommand(killCommand(g.ppid), navigator.clipboard)} className={COPY_BUTTON_CLASS}>
                    {t("copy")}
                  </button>
                </div>
              </div>
            ))}
            {/* Audit 2026-09-19: one caption per block, not repeated per group. */}
            <span className="pl-1 text-[10.5px] italic text-muted">{t("copyCaption")}</span>
          </>
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
        {alertsPending ? (
          <div className="h-6 animate-pulse rounded bg-surface-2" />
        ) : packages.length === 0 ? (
          <span className="text-[12.5px] text-muted">{t("packagesEmpty")}</span>
        ) : (
          <>
            {visiblePackages.map((p) => (
              <div key={p.name} className="flex items-center gap-2 text-[12px] text-body-ink">
                {p.security ? <span className="flex-none rounded-full bg-danger-soft px-1.5 py-0 text-[10px] font-bold text-danger">{t("security")}</span> : null}
                <span className="min-w-0 flex-1 truncate font-mono">{p.name} {p.fromVersion} → {p.toVersion}</span>
              </div>
            ))}
            {hiddenPackageCount > 0 ? (
              <span className="text-[11px] text-muted">{t("packagesMore", { count: hiddenPackageCount })}</span>
            ) : null}
            {securityNames.length > 0 ? (
              <div className="flex items-center gap-2">
                <span className={CHIP_CLASS}>{aptCommand}</span>
                <button type="button" onClick={() => copyCommand(aptCommand, navigator.clipboard)} className={COPY_BUTTON_CLASS}>
                  {t("copy")}
                </button>
              </div>
            ) : null}
            {securityNames.length > 0 ? <span className="text-[10.5px] italic text-muted">{t("copyCaption")}</span> : null}
          </>
        )}
      </div>
    </section>
  );
}
