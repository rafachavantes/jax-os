import { cpus, hostname } from "node:os";
import type Database from "better-sqlite3";

export type HealthRange = "1h" | "24h" | "7d" | "90d";
export type ServiceStatus = { units: Record<string, string>; containers: { name: string; state: string }[] };
export type MetricPoint = {
  ts: string;
  cpuPct: number | null;
  memUsedMb: number | null;
  diskUsedGb: number | null;
  load5: number | null;
};
export type HealthLatest = {
  ts: string; cpuPct: number | null; memUsedMb: number | null; memTotalMb: number | null;
  swapUsedMb: number | null; diskUsedGb: number | null; diskTotalGb: number | null;
  load1: number | null; load5: number | null; load15: number | null; uptimeS: number | null;
};
export type HealthPayload = {
  latest: HealthLatest | null;
  series: MetricPoint[];
  seriesMax: MetricPoint[] | null;
  spark: MetricPoint[];
  services: ServiceStatus | null;
  collectedAgoS: number | null;
  cores: number;
  // Host-level, not per-sample — same value regardless of `latest` (Mission Control server
  // panel's host line; audit 2026-09-19).
  hostname: string;
};

const RANGE_MS: Record<HealthRange, number> = {
  "1h": 3_600_000,
  "24h": 86_400_000,
  "7d": 7 * 86_400_000,
  "90d": 90 * 86_400_000,
};

// SVG polylines choke on 10k points; every-Nth keeps shape at cockpit scale.
// The newest point always survives (it's the "now" the eye anchors on).
export function thin<T>(rows: T[], cap = 500): T[] {
  if (rows.length <= cap) return rows;
  const step = Math.ceil(rows.length / cap);
  const out = rows.filter((_, i) => i % step === 0);
  if (out[out.length - 1] !== rows[rows.length - 1]) out.push(rows[rows.length - 1]);
  return out;
}

type RawRow = {
  ts: string; cpu_pct: number | null; mem_used_mb: number | null; mem_total_mb: number | null;
  swap_used_mb: number | null; disk_used_gb: number | null; disk_total_gb: number | null;
  load1: number | null; load5: number | null; load15: number | null; uptime_s: number | null;
  services: string;
};

function rawPoints(db: Database.Database, sinceIso: string): MetricPoint[] {
  const rows = db
    .prepare("SELECT ts, cpu_pct, mem_used_mb, disk_used_gb, load5 FROM metrics WHERE ts >= ? ORDER BY ts ASC")
    .all(sinceIso) as Pick<RawRow, "ts" | "cpu_pct" | "mem_used_mb" | "disk_used_gb" | "load5">[];
  return rows.map((r) => ({ ts: r.ts, cpuPct: r.cpu_pct, memUsedMb: r.mem_used_mb, diskUsedGb: r.disk_used_gb, load5: r.load5 }));
}

export function queryMetrics(db: Database.Database, range: HealthRange): HealthPayload {
  const nowMs = Date.now();
  const cores = cpus().length;
  const host = hostname();

  const latestRow = db
    .prepare("SELECT * FROM metrics ORDER BY ts DESC, id DESC LIMIT 1")
    .get() as RawRow | undefined;

  if (!latestRow) {
    return { latest: null, series: [], seriesMax: null, spark: [], services: null, collectedAgoS: null, cores, hostname: host };
  }

  const latest: HealthLatest = {
    ts: latestRow.ts, cpuPct: latestRow.cpu_pct, memUsedMb: latestRow.mem_used_mb, memTotalMb: latestRow.mem_total_mb,
    swapUsedMb: latestRow.swap_used_mb, diskUsedGb: latestRow.disk_used_gb, diskTotalGb: latestRow.disk_total_gb,
    load1: latestRow.load1, load5: latestRow.load5, load15: latestRow.load15, uptimeS: latestRow.uptime_s,
  };

  let services: ServiceStatus | null = null;
  try {
    const parsed: unknown = JSON.parse(latestRow.services);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) services = parsed as ServiceStatus;
  } catch {
    // manual tampering only — collector always writes valid JSON
  }

  let series: MetricPoint[];
  let seriesMax: MetricPoint[] | null = null;
  if (range === "90d") {
    // hourly aggregates; hour key is ts.slice(0,13) — rebuild a parseable ts
    const cutoffHour = new Date(nowMs - RANGE_MS["90d"]).toISOString().slice(0, 13);
    const rows = db
      .prepare("SELECT * FROM metrics_hourly WHERE hour >= ? ORDER BY hour ASC")
      .all(cutoffHour) as {
        hour: string; cpu_pct_avg: number | null; cpu_pct_max: number | null;
        mem_used_mb_avg: number | null; mem_used_mb_max: number | null;
        load5_max: number | null; disk_used_gb_last: number | null;
      }[];
    // same-length inputs + same step keep avg/max series index-aligned
    series = thin(rows.map((r) => ({
      ts: `${r.hour}:00:00.000Z`, cpuPct: r.cpu_pct_avg, memUsedMb: r.mem_used_mb_avg,
      diskUsedGb: r.disk_used_gb_last, load5: null,
    })));
    seriesMax = thin(rows.map((r) => ({
      ts: `${r.hour}:00:00.000Z`, cpuPct: r.cpu_pct_max, memUsedMb: r.mem_used_mb_max,
      diskUsedGb: null, load5: r.load5_max,
    })));
  } else {
    series = thin(rawPoints(db, new Date(nowMs - RANGE_MS[range]).toISOString()));
  }

  const spark = thin(rawPoints(db, new Date(nowMs - RANGE_MS["24h"]).toISOString()));
  const collectedAgoS = Math.max(0, Math.round((nowMs - Date.parse(latestRow.ts)) / 1000));

  return { latest, series, seriesMax, spark, services, collectedAgoS, cores, hostname: host };
}
