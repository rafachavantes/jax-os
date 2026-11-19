import { NextResponse } from "next/server";
import { getAgents } from "../../../../server/collectors/tmux";
import { logMutation } from "../../../../server/mutations";
import { readJsonObject } from "../../../../server/api";
import { SMALL_JSON_CAP, boundedString } from "../../../../server/inputLimits";
import {
  EXACT_SHAPE_BODY,
  SESSION_ACCESS_ERROR_TEXT,
  SessionAccessError,
  exactPayload,
  parseExactControlPayload,
  type SessionAccessCode,
} from "../../../../lib/sessionAccess";
import {
  readSessionAccessConfig,
  resolveSessionAccess,
  takenSessionAccess,
  type SessionAccessConfig,
} from "../../../../server/session-access";

export type ExactControlDeps = {
  resolve: typeof resolveSessionAccess;
  config: () => SessionAccessConfig;
  log: typeof logMutation;
};

const exactDeps: ExactControlDeps = {
  resolve: resolveSessionAccess,
  config: readSessionAccessConfig,
  log: logMutation,
};

export function sessionAccessFailure(e: unknown) {
  const code: SessionAccessCode = e instanceof SessionAccessError ? e.code : "source-failure";
  return NextResponse.json({ ok: false, code, error: SESSION_ACCESS_ERROR_TEXT[code] });
}

// Exact Take: a FRESH resolution of the same expected binding, then the existing take-control
// audit insert. Only a successful insert returns TakenSessionAccess (the only rwUrl carrier).
export async function handleTakeControl(req: Request, deps: ExactControlDeps = exactDeps) {
  const parsed = await readJsonObject(req, SMALL_JSON_CAP);
  if ("error" in parsed) return parsed.error;
  const payload = parsed.payload;

  const exact = parseExactControlPayload(payload);
  if (exact === null) return NextResponse.json(EXACT_SHAPE_BODY);
  if (exact !== "legacy") {
    let resolved;
    let taken;
    try {
      resolved = await deps.resolve(exact.selector, exact.expected);
      taken = takenSessionAccess(resolved, deps.config());
    } catch (e) {
      return sessionAccessFailure(e);
    }
    try {
      deps.log({ ...exactPayload(exact.selector, resolved.expected, resolved.sessionName), kind: "take-control" });
    } catch {
      return NextResponse.json({
        ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
        error: "control record unavailable",
      });
    }
    return NextResponse.json({ ok: true, data: taken });
  }

  // Legacy `{session}` branch (mobile Mosaic reply gating): unchanged validation and audit.
  const { session } = payload as { session?: unknown };
  if (!boundedString(session, 256)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  let exists = false;
  try {
    exists = getAgents().some((a) => a.session === session);
  } catch {
    // tmux genuinely broken — nothing to grant control over
  }
  if (!exists) {
    return NextResponse.json({ ok: false, error: "session not found" });
  }
  try {
    logMutation({ ts: new Date().toISOString(), kind: "take-control", session });
  } catch {
    return NextResponse.json({
      ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "control record unavailable",
    });
  }
  return NextResponse.json({ ok: true });
}

export async function POST(req: Request) {
  return handleTakeControl(req);
}
