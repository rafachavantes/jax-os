import { hostname } from "node:os";
import { describe, expect, it } from "vitest";
import { openDb } from "./index";
import { queryMetrics, thin } from "./metrics";

function insertSample(db: ReturnType<typeof openDb>, ts: string, cpu: number | null, mem = 1000) {
  db.prepare(
    `INSERT INTO metrics (ts, cpu_pct, mem_used_mb, mem_total_mb, swap_used_mb, disk_used_gb, disk_total_gb,
       load1, load5, load15, uptime_s, services)
     VALUES (?, ?, ?, 16000, 0, 100.5, 150.0, 0.5, 0.7, 0.9, 3600,
       '{"units":{"docker.service":"active"},"containers":[{"name":"c1","state":"running"}]}')`,
  ).run(ts, cpu, mem);
}

describe("queryMetrics", () => {
  it("returns quiet empty on a fresh db", () => {
    const db = openDb(":memory:");
    const p = queryMetrics(db, "24h");
    expect(p).toMatchObject({ latest: null, series: [], seriesMax: null, spark: [], services: null, collectedAgoS: null });
    expect(p.cores).toBeGreaterThan(0);
    expect(p.hostname).toBe(hostname());
    db.close();
  });

  it("returns latest, parsed services, and raw series for 24h", () => {
    const db = openDb(":memory:");
    const now = Date.now();
    insertSample(db, new Date(now - 2 * 3600_000).toISOString(), 10);
    insertSample(db, new Date(now - 60_000).toISOString(), 20, 2000);
    const p = queryMetrics(db, "24h");
    expect(p.latest?.cpuPct).toBe(20);
    expect(p.latest?.memUsedMb).toBe(2000);
    expect(p.services?.units["docker.service"]).toBe("active");
    expect(p.services?.containers).toEqual([{ name: "c1", state: "running" }]);
    expect(p.series).toHaveLength(2);
    expect(p.seriesMax).toBeNull();
    expect(p.collectedAgoS).toBeGreaterThanOrEqual(59);
    expect(p.collectedAgoS).toBeLessThan(180);
    expect(p.hostname).toBe(hostname());
    db.close();
  });

  it("1h range excludes older raw rows; spark stays 24h", () => {
    const db = openDb(":memory:");
    const now = Date.now();
    insertSample(db, new Date(now - 2 * 3600_000).toISOString(), 10);
    insertSample(db, new Date(now - 60_000).toISOString(), 20);
    const p = queryMetrics(db, "1h");
    expect(p.series).toHaveLength(1);
    expect(p.spark).toHaveLength(2);
    db.close();
  });

  it("90d reads hourly aggregates as avg series + max series", () => {
    const db = openDb(":memory:");
    const hour = new Date(Date.now() - 3600_000).toISOString().slice(0, 13);
    db.prepare(
      "INSERT INTO metrics_hourly (hour, cpu_pct_avg, cpu_pct_max, mem_used_mb_avg, mem_used_mb_max, load5_max, disk_used_gb_last, samples) VALUES (?, 15.5, 42.0, 1500, 1800, 1.2, 100.5, 60)",
    ).run(hour);
    insertSample(db, new Date().toISOString(), 20); // latest still comes from raw
    const p = queryMetrics(db, "90d");
    expect(p.series).toHaveLength(1);
    expect(p.series[0]).toMatchObject({ cpuPct: 15.5, memUsedMb: 1500 });
    expect(p.seriesMax?.[0]).toMatchObject({ cpuPct: 42.0, memUsedMb: 1800 });
    expect(p.series[0].ts.startsWith(hour)).toBe(true);
    db.close();
  });

  it("corrupt services JSON yields services: null (manual tampering guard)", () => {
    const db = openDb(":memory:");
    db.prepare(
      "INSERT INTO metrics (ts, services) VALUES (?, 'not json')",
    ).run(new Date().toISOString());
    expect(queryMetrics(db, "24h").services).toBeNull();
    db.close();
  });
});

describe("thin", () => {
  it("keeps arrays <=500 as-is and thins bigger ones to <=500 keeping the last point", () => {
    const small = Array.from({ length: 500 }, (_, i) => i);
    expect(thin(small)).toHaveLength(500);
    const big = Array.from({ length: 10080 }, (_, i) => i);
    const thinned = thin(big);
    expect(thinned.length).toBeLessThanOrEqual(500);
    expect(thinned[thinned.length - 1]).toBe(10079); // newest sample always survives
  });
});

describe("hourly aggregation SQL (same text the Python collector runs)", () => {
  const AGGREGATE_SQL = `
INSERT OR REPLACE INTO metrics_hourly
  (hour, cpu_pct_avg, cpu_pct_max, mem_used_mb_avg, mem_used_mb_max, load5_max, disk_used_gb_last, samples)
SELECT substr(ts, 1, 13) AS h,
       avg(cpu_pct), max(cpu_pct),
       CAST(avg(mem_used_mb) AS INTEGER), max(mem_used_mb),
       max(load5),
       (SELECT m2.disk_used_gb FROM metrics m2
         WHERE substr(m2.ts, 1, 13) = substr(m.ts, 1, 13)
         ORDER BY m2.ts DESC LIMIT 1),
       count(*)
FROM metrics m
WHERE substr(ts, 1, 13) < ?
  AND substr(ts, 1, 13) NOT IN (SELECT hour FROM metrics_hourly)
GROUP BY substr(ts, 1, 13)
`;
  it("aggregates only completed hours, idempotently", () => {
    const db = openDb(":memory:");
    const ins = db.prepare(
      "INSERT INTO metrics (ts, cpu_pct, mem_used_mb, load5, disk_used_gb, services) VALUES (?, ?, ?, ?, ?, '{}')",
    );
    ins.run("2026-07-12T13:00:00.000Z", 10, 1000, 0.5, 100.1);
    ins.run("2026-07-12T13:30:00.000Z", 30, 2000, 1.5, 100.2);
    ins.run("2026-07-12T14:10:00.000Z", 99, 9000, 9.9, 100.3); // current hour — must NOT aggregate
    db.prepare(AGGREGATE_SQL).run("2026-07-12T14");
    db.prepare(AGGREGATE_SQL).run("2026-07-12T14"); // idempotent
    const rows = db.prepare("SELECT * FROM metrics_hourly ORDER BY hour").all() as Record<string, unknown>[];
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      hour: "2026-07-12T13", cpu_pct_avg: 20, cpu_pct_max: 30,
      mem_used_mb_avg: 1500, mem_used_mb_max: 2000, load5_max: 1.5,
      disk_used_gb_last: 100.2, samples: 2,
    });
    db.close();
  });

  it("never re-aggregates an hour after its raw rows are pruned", () => {
    const db = openDb(":memory:");
    const ins = db.prepare(
      "INSERT INTO metrics (ts, cpu_pct, mem_used_mb, load5, disk_used_gb, services) VALUES (?, ?, ?, ?, ?, '{}')",
    );
    ins.run("2026-07-12T13:00:00.000Z", 90, 5000, 2.0, 100.1);
    ins.run("2026-07-12T13:30:00.000Z", 20, 1200, 0.5, 100.2);
    db.prepare(AGGREGATE_SQL).run("2026-07-12T14");
    // 7d prune eats the hour's peak row; re-running must not degrade the aggregate
    db.prepare("DELETE FROM metrics WHERE ts = '2026-07-12T13:00:00.000Z'").run();
    db.prepare(AGGREGATE_SQL).run("2026-07-12T14");
    const row = db.prepare("SELECT * FROM metrics_hourly WHERE hour = '2026-07-12T13'").get() as Record<string, unknown>;
    expect(row).toMatchObject({ cpu_pct_max: 90, mem_used_mb_max: 5000, samples: 2 });
    db.close();
  });
});
