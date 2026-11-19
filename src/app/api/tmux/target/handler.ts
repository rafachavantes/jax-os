import { NextResponse } from "next/server";
import type { Envelope } from "@/lib/api";
import {
  SESSION_ACCESS_ERROR_TEXT,
  SessionAccessError,
  parseSessionAccessQuery,
  type ResolvedSessionAccess,
  type SessionAccessCode,
} from "../../../../lib/sessionAccess";
import { resolveSessionAccess } from "../../../../server/session-access";

export type TargetHandlerDeps = { resolve: typeof resolveSessionAccess };

const systemDeps: TargetHandlerDeps = { resolve: resolveSessionAccess };

function failure(code: SessionAccessCode) {
  return NextResponse.json({ ok: false, code, error: SESSION_ACCESS_ERROR_TEXT[code] } as Envelope<ResolvedSessionAccess> & { code: SessionAccessCode });
}

// Identity-only dynamic read: invalid input fails before any native/DB access, and a source or
// binding failure is a bounded 200 envelope — never raw command/path/secret exception text.
export async function handleTargetGet(input: string | URL, deps: TargetHandlerDeps = systemDeps) {
  let params: URLSearchParams;
  try {
    params = new URL(input, "http://127.0.0.1").searchParams;
  } catch {
    return failure("invalid-selector");
  }
  const query = parseSessionAccessQuery(params);
  if (query.kind === "invalid") return failure("invalid-selector");
  try {
    const data = await deps.resolve(query.selector, query.expected);
    return NextResponse.json({ ok: true, data } satisfies Envelope<ResolvedSessionAccess>);
  } catch (e) {
    return failure(e instanceof SessionAccessError ? e.code : "source-failure");
  }
}
