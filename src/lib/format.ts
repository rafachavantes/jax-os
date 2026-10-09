import type { HealthLatest } from "@/server/db/metrics";

// Threshold-color formula shared by every usage/meter bar (originally SubscriptionCards.tsx's
// inline WindowBar color, now reused by ServerPanel.tsx's per-metric bars).
export function meterColorClass(pct: number): string {
  return pct >= 90 ? "bg-danger" : pct >= 70 ? "bg-warning" : "bg-accent";
}

// Item 14 (round 3): the mockup's short-form meter values — ServerPanel.tsx's own
// meterValues() stays the long form (its own long-form test is unchanged); this is a second,
// additive formatter whose output renders on the row, with the long form moved into a native
// `title` tooltip instead. Lives here (not ServerPanel.tsx) alongside meterColorClass/fmtUptime
// — same "pure formatter, no JSX" shape as both (Global Constraints' helper-placement rule).
export function meterValuesCompact(latest: HealthLatest | null, cores: number | null) {
  return {
    cpu: latest?.cpuPct != null ? `${latest.cpuPct.toFixed(0)}%` : "—",
    mem: latest?.memUsedMb != null ? `${(latest.memUsedMb / 1024).toFixed(1)}G` : "—",
    disk: latest?.diskUsedGb != null && latest?.diskTotalGb ? `${Math.round((latest.diskUsedGb / latest.diskTotalGb) * 100)}%` : "—",
    load: latest?.load5 != null ? `${latest.load5.toFixed(2)}/${cores ?? "—"}` : "—",
  };
}

// Shared by src/app/health/page.tsx's own tile and ServerPanel.tsx's host line.
export function fmtUptime(s: number | null | undefined): string {
  if (s === null || s === undefined) return "—";
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  return d > 0 ? `${d}d ${h}h` : `${h}h ${m}m`;
}
