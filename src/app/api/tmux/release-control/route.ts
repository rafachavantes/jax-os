import { NextResponse } from "next/server";
import { logMutation } from "../../../../server/mutations";
import { readJsonObject } from "../../../../server/api";
import { SMALL_JSON_CAP, boundedString } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  const parsed = await readJsonObject(req, SMALL_JSON_CAP);
  if ("error" in parsed) return parsed.error;
  const payload = parsed.payload;
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
