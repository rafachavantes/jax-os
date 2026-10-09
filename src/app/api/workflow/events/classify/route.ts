import { NextResponse } from "next/server";
import { CAPSULE_STATUSES, LIMITS } from "../../../../../lib/workflow";
import { collectorResponse, readJsonCapped, requireSameOrigin } from "../../../../../server/api";
import { getDb } from "../../../../../server/db";
import { classifyDeferred } from "../../../../../server/db/workflows";

// The poller settles a deferred turn-stopped here. jaxflow's scripts never open the db for
// writing (jaxflow spec I6), so this route is the write path — the same shape events/ack uses.
export async function POST(req: Request) {
  const guard = requireSameOrigin(req);
  if (guard) return guard;
  const read = await readJsonCapped(req, LIMITS.requestBytes);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value as Record<string, unknown> | null;
  if (!body || typeof body !== "object") return NextResponse.json({ ok: false, error: "body must be an object" });

  const keys = Object.keys(body).sort().join(",");
  const id = body.id;
  if (!Number.isInteger(id) || (id as number) <= 0) {
    return NextResponse.json({ ok: false, error: "id must be a positive integer" });
  }
  let outcome: { status: string; mergeAsk?: number } | { failed: true } | { indeterminate: true };
  if (keys === "capsule_status,id" || keys === "capsule_status,id,merge_ask") {
    const s = body.capsule_status;
    if (typeof s !== "string" || !CAPSULE_STATUSES.includes(s as never) || s === "unknown") {
      return NextResponse.json({ ok: false, error: "capsule_status not allowed" });
    }
    if (keys === "capsule_status,id,merge_ask") {
      const m = body.merge_ask;
      if (typeof m !== "number" || !Number.isFinite(m) || m < 0 || m > 1) {
        return NextResponse.json({ ok: false, error: "merge_ask not allowed" });
      }
      outcome = { status: s, mergeAsk: m };
    } else {
      outcome = { status: s };
    }
  } else if (keys === "failed,id" && body.failed === true) {
    outcome = { failed: true };
  } else if (keys === "id,indeterminate" && body.indeterminate === true) {
    outcome = { indeterminate: true };
  } else {
    return NextResponse.json({ ok: false, error: "body must be {id, capsule_status}, {id, capsule_status, merge_ask}, {id, failed: true}, or {id, indeterminate: true}" });
  }
  return collectorResponse(() => ({ settled: classifyDeferred(getDb(), id as number, outcome as never) }));
}
