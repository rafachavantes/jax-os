import type Database from "better-sqlite3";
import {
  queryNativeUsage,
  UNKNOWN_MODEL_FILTER,
  type NativeUsageSnapshot,
  type UsagePeriod,
  type UsageSlice,
} from "../db/native-usage";

export { UNKNOWN_MODEL_FILTER };
export type { NativeUsageSnapshot, UsagePeriod, UsageSlice };

export const RANGES = { "7d": 7, "30d": 30, "90d": 90 } as const;
export type RangeKey = keyof typeof RANGES;

export function isRange(value: string): value is RangeKey {
  return Object.prototype.hasOwnProperty.call(RANGES, value);
}

export type UsageParams = { range: RangeKey; agent?: string | null; model?: string | null };

export function getTokensUsage(
  db: Database.Database,
  params: UsageParams,
  now: number = Math.floor(Date.now() / 1000),
): NativeUsageSnapshot {
  return queryNativeUsage(db, {
    period: RANGES[params.range],
    agent: params.agent ?? null,
    model: params.model ?? null,
    now,
  });
}
