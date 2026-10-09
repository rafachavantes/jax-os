import { NextResponse } from "next/server";
import { logMutation } from "../../../../server/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { SMALL_JSON_CAP, boundedString } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const body = await readJsonCapped(req, SMALL_JSON_CAP);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error });
  const payload = body.value;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return NextResponse.json({ ok: false, error: "invalid payload" });
  const { session } = payload as { session?: unknown };
  if (!boundedString(session, 256)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  try {
    logMutation({ ts: new Date().toISOString(), kind: "release-control", session });
  } catch {
    return NextResponse.json({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "control record unavailable",
    });
  }
  return NextResponse.json({ ok: true });
}
