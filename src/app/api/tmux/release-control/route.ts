import { NextResponse } from "next/server";
import { logMutation } from "../../../../server/mutations";
import { readJsonObject } from "../../../../server/api";
import { SMALL_JSON_CAP, boundedString } from "../../../../server/inputLimits";
import { EXACT_SHAPE_BODY, exactPayload, parseExactControlPayload } from "../../../../lib/sessionAccess";
import type { ExactControlDeps } from "../take-control/route";

const releaseDeps = { log: logMutation } as Pick<ExactControlDeps, "log">;

// Exact Release: records the supplied original selector/binding WITHOUT requiring the native
// target to still exist. `targetPresent: "unverified"` is an audit fact, never a remote lease —
// and a failed audit cannot restore the old RW frame (the client already dropped it).
export async function handleReleaseControl(
  req: Request,
  deps: Pick<ExactControlDeps, "log"> = releaseDeps,
) {
  const parsed = await readJsonObject(req, SMALL_JSON_CAP);
  if ("error" in parsed) return parsed.error;
  const payload = parsed.payload;

  const exact = parseExactControlPayload(payload);
  if (exact === null) return NextResponse.json(EXACT_SHAPE_BODY);
  if (exact !== "legacy") {
    try {
      deps.log({ ...exactPayload(exact.selector, exact.expected), kind: "release-control", targetPresent: "unverified" });
    } catch {
      return NextResponse.json({
        ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
        error: "control record unavailable",
      });
    }
    return NextResponse.json({ ok: true });
  }

  const { session } = payload as { session?: unknown };
  if (!boundedString(session, 256)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  try {
    deps.log({ ts: new Date().toISOString(), kind: "release-control", session });
  } catch {
    return NextResponse.json({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "control record unavailable",
    });
  }
  return NextResponse.json({ ok: true });
}

export async function POST(req: Request) {
  return handleReleaseControl(req);
}
