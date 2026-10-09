import { NextResponse } from "next/server";
import { LIMITS } from "../../../../lib/workflow";
import { collectorResponse, readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { parseIngress, toForwardBody } from "../../../../server/collectors/workflow-events";
import { forwardNotification, notificationTargetFromEnv } from "../../../../server/collectors/workflow-webhook";
import { getDb } from "../../../../server/db";
import { insertEvent, listPending, markDelivered, RUN_ALREADY_FINISHED } from "../../../../server/db/workflows";

export const dynamic = "force-dynamic";

// Hermes fallback poll (spec §4.2, cursorless): the server-side pending filter
// IS the cursor. Rows are forward-shaped so push and poll carry identical bodies.
export async function GET(req: Request) {
  const u = new URL(req.url);
  if (u.searchParams.get("pending") !== "1") {
    return NextResponse.json({ ok: false, error: "pending=1 required" }); // dashboard listing is Phase D
  }
  const n = Number(u.searchParams.get("limit"));
  const limit = Number.isInteger(n) && n > 0 ? Math.min(n, LIMITS.pollLimitMax) : 50;
  const viewer = process.env.TTYD_RO_URL;
  return collectorResponse(() => listPending(getDb(), limit).map((r) => toForwardBody(r, viewer)));
}

// Ingress (wrapper + hooks). Insert first; then at most ONE bounded forward.
// Persistence success is never masked by forward failure (spec §4.2).
export async function POST(req: Request) {
  const guard = requireSameOrigin(req);
  if (guard) return guard;
  const read = await readJsonCapped(req, LIMITS.requestBytes);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const parsed = parseIngress(read.value);
  if (!parsed.ok) return NextResponse.json({ ok: false, error: `invalid event: ${parsed.error}` });
  if (parsed.event.emitter === "jaxos") {
    // question-answered is appended in-process by the answer route (Phase C), never over HTTP
    return NextResponse.json({ ok: false, error: "emitter jaxos is internal" });
  }

  let row;
  try {
    row = insertEvent(getDb(), parsed.event);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    // A second run-finished for the same run_id is the one insert failure with real
    // "someone else already won" semantics (spec §2.6 cold review G1) — a distinct
    // 409, never the generic 200 {ok:false} every other insert failure keeps.
    if (message === RUN_ALREADY_FINISHED) {
      return NextResponse.json({ ok: false, error: message }, { status: 409 });
    }
    // NOT persisted — the wrapper's fail-closed signal
    return NextResponse.json({ ok: false, error: message });
  }
  const data = { id: row.id, ts: row.ts, delivery: row.delivery };
  if (row.delivery !== "pending") return NextResponse.json({ ok: true, forwarded: false, data });

  const target = notificationTargetFromEnv();
  if (!target) return NextResponse.json({ ok: true, forwarded: false, data }); // no webhook configured → poll fallback

  let forwardBody: string;
  try {
    forwardBody = JSON.stringify(toForwardBody(row, process.env.TTYD_RO_URL));
  } catch {
    // persisted but unforwardable — leave it pending for the poll, never 5xx
    return NextResponse.json({ ok: true, forwarded: false, data });
  }
  const fwd = await forwardNotification(forwardBody, target);
  if (fwd.ok) {
    try {
      markDelivered(getDb(), [row.id]);
      data.delivery = "delivered";
    } catch (e) {
      console.error("[workflow] markDelivered failed after a 200 forward (event stays pending, poll may re-deliver):", e);
    }
  }
  return NextResponse.json({ ok: true, forwarded: fwd.ok, data });
}
