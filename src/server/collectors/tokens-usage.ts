export type SessionRow = {
  model: string | null;
  started_at: number; // unix seconds
  input_tokens: number | null;
  output_tokens: number | null;
  cache_read_tokens: number | null;
  cache_write_tokens: number | null;
  reasoning_tokens: number | null;
  estimated_cost_usd: number | null;
};

export type Slice = { name: string; tokens: number; cost: number };

export type UsageData = {
  totalTokens: number;
  totalCost: number;
  daily: { date: string; tokens: number }[];
  byModel: Slice[];
  byAgent: Slice[];
};

const rowTokens = (r: SessionRow) =>
  (r.input_tokens ?? 0) +
  (r.output_tokens ?? 0) +
  (r.cache_read_tokens ?? 0) +
  (r.cache_write_tokens ?? 0) +
  (r.reasoning_tokens ?? 0);

// ponytail: UTC day buckets (Rafa is UTC-3; a 22h session lands on the next
// bar). Switch to server-local bucketing if that ever bothers.
const isoDay = (epoch: number) => new Date(epoch * 1000).toISOString().slice(0, 10);

export function aggregateUsage(
  agents: { name: string; rows: SessionRow[] }[],
  days: number,
  nowEpoch: number,
): UsageData {
  const daily = Array.from({ length: days }, (_, i) => ({
    date: isoDay(nowEpoch - (days - 1 - i) * 86400),
    tokens: 0,
  }));
  // Cutoff aligned to the oldest calendar bucket, not a rolling window —
  // otherwise rows between the rolling cutoff and that day's UTC midnight
  // would count in totals but miss every daily bucket.
  const cutoff = Date.parse(`${daily[0].date}T00:00:00Z`) / 1000;
  const inRange = agents.map((a) => ({
    name: a.name,
    rows: a.rows.filter((r) => r.started_at >= cutoff),
  }));
  const all = inRange.flatMap((a) => a.rows);
  const byDate = new Map(daily.map((d) => [d.date, d]));
  const models = new Map<string, Slice>();
  for (const r of all) {
    const day = byDate.get(isoDay(r.started_at));
    if (day) day.tokens += rowTokens(r);
    const key = r.model ?? "unknown";
    const m = models.get(key) ?? { name: key, tokens: 0, cost: 0 };
    m.tokens += rowTokens(r);
    m.cost += r.estimated_cost_usd ?? 0;
    models.set(key, m);
  }

  const byAgent = inRange.map((a) => ({
    name: a.name,
    tokens: a.rows.reduce((s, r) => s + rowTokens(r), 0),
    cost: a.rows.reduce((s, r) => s + (r.estimated_cost_usd ?? 0), 0),
  }));
  return {
    totalTokens: byAgent.reduce((s, a) => s + a.tokens, 0),
    totalCost: byAgent.reduce((s, a) => s + a.cost, 0),
    daily,
    byModel: [...models.values()].sort((a, b) => {
      if (b.tokens !== a.tokens) return b.tokens - a.tokens;
      if (a.name === "unknown") return 1;
      if (b.name === "unknown") return -1;
      return 0;
    }),
    byAgent,
  };
}

import { existsSync, readdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import Database from "better-sqlite3";

export const RANGES = { "7d": 7, "30d": 30, "90d": 90 } as const;
export type RangeKey = keyof typeof RANGES;

export type UsagePayload = UsageData & { skipped: string[] };

const HERMES_HOME = join(homedir(), ".hermes");

// Each hermes profile (= agent) is an isolated HERMES_HOME with its own
// state.db: ~/.hermes (default) + ~/.hermes/profiles/<name>/.
export function discoverAgentDbs(): { name: string; path: string }[] {
  const dbs: { name: string; path: string }[] = [];
  const root = join(HERMES_HOME, "state.db");
  if (existsSync(root)) dbs.push({ name: "default", path: root });
  const profilesDir = join(HERMES_HOME, "profiles");
  if (existsSync(profilesDir)) {
    for (const dir of readdirSync(profilesDir).sort()) {
      const p = join(profilesDir, dir, "state.db");
      if (existsSync(p)) dbs.push({ name: dir, path: p });
    }
  }
  return dbs;
}

// Untested shell glue — aggregation above is the tested part.
export function getTokensUsage(range: RangeKey): UsagePayload {
  const days = RANGES[range];
  const nowEpoch = Math.floor(Date.now() / 1000);
  const dbs = discoverAgentDbs();
  if (dbs.length === 0) throw new Error("no hermes state.db found");
  const agents: { name: string; rows: SessionRow[] }[] = [];
  const skipped: string[] = [];
  for (const { name, path } of dbs) {
    try {
      const db = new Database(path, { readonly: true, fileMustExist: true });
      try {
        agents.push({
          name,
          rows: db
            .prepare(
              `SELECT model, started_at, input_tokens, output_tokens, cache_read_tokens,
                      cache_write_tokens, reasoning_tokens, estimated_cost_usd
                 FROM sessions WHERE started_at >= ?`,
            )
            .all(nowEpoch - days * 86400) as SessionRow[],
        });
      } finally {
        db.close();
      }
    } catch {
      skipped.push(name);
    }
  }
  if (agents.length === 0) throw new Error("no readable hermes state.db");
  return { ...aggregateUsage(agents, days, nowEpoch), skipped };
}
