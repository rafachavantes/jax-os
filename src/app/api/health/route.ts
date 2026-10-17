import { collectorResponse } from "@/server/api";
import { getDb } from "@/server/db";
import { queryMetrics, type HealthRange } from "@/server/db/metrics";

export const dynamic = "force-dynamic";

const RANGES: HealthRange[] = ["1h", "24h", "7d", "90d"];

export async function GET(req: Request) {
  const raw = new URL(req.url).searchParams.get("range");
  const range: HealthRange = RANGES.includes(raw as HealthRange) ? (raw as HealthRange) : "24h";
  return collectorResponse(() => queryMetrics(getDb(), range));
}
