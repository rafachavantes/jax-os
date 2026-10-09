import { NextResponse } from "next/server";
import { LIMITS } from "@/lib/workflow";
import { collectorResponse, readJsonCapped, requireSameOrigin } from "@/server/api";
import { getDb } from "@/server/db";
import { markDelivered } from "@/server/db/workflows";

// Hermes acks exactly the ids it accepted from the poll (partial batches fine).
export async function POST(req: Request) {
  const guard = requireSameOrigin(req);
  if (guard) return guard;
  const read = await readJsonCapped(req, LIMITS.requestBytes);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const ids = (read.value as { ids?: unknown } | null)?.ids;
  if (
    !Array.isArray(ids) || ids.length === 0 || ids.length > LIMITS.ackIdsMax ||
    !ids.every((i) => Number.isInteger(i) && i > 0)
  ) {
    return NextResponse.json({ ok: false, error: `ids must be 1-${LIMITS.ackIdsMax} positive integers` });
  }
  return collectorResponse(() => ({ acked: markDelivered(getDb(), ids as number[]) }));
}
