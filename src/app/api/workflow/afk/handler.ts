import { NextResponse } from "next/server";
import { getAfkEnabled, getForwardTypes, setAfk, setForwardTypes } from "../../../../server/db/workflows";
import { collectorResponse, readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { getDb } from "../../../../server/db";
import { FORWARDABLE_EVENT_TYPES, type EventType } from "../../../../lib/workflow";

// Deps split so a test never opens the real ~/.jax-os/jaxos.db (mirrors answer/handler.ts's
// established Deps + systemDeps pattern in this same directory tree).
export type AfkRouteDeps = {
  getDb: typeof getDb;
  getAfkEnabled: typeof getAfkEnabled;
  setAfk: typeof setAfk;
  getForwardTypes: typeof getForwardTypes;
  setForwardTypes: typeof setForwardTypes;
};

const systemDeps: AfkRouteDeps = { getDb, getAfkEnabled, setAfk, getForwardTypes, setForwardTypes };

export async function handleAfkGet(_req?: Request, deps: AfkRouteDeps = systemDeps) {
  return collectorResponse(() => ({
    enabled: deps.getAfkEnabled(deps.getDb()),
    forwardTypes: deps.getForwardTypes(deps.getDb()),
  }));
}

// MOA-487 §5 Decision 6: forwardTypes, if present, must be a JSON array of forwardable types
// with no duplicates.
function isValidForwardTypes(v: unknown): v is EventType[] {
  if (!Array.isArray(v)) return false;
  if (new Set(v).size !== v.length) return false;
  return v.every((t) => (FORWARDABLE_EVENT_TYPES as readonly unknown[]).includes(t));
}

export async function handleAfkPost(req: Request, deps: AfkRouteDeps = systemDeps) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  const read = await readJsonCapped(req, 1024);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  // `readJsonCapped` accepts any valid JSON value — the optional chain covers every non-object
  // shape (`null`, an array, a string, a number) with one check (Finding 14).
  const body = read.value as { enabled?: unknown; forwardTypes?: unknown } | null;
  const enabledPresent = body?.enabled !== undefined;
  const forwardTypesPresent = body?.forwardTypes !== undefined;
  if (!enabledPresent && !forwardTypesPresent) {
    return NextResponse.json({ ok: false, error: "enabled or forwardTypes required" });
  }
  if (enabledPresent && typeof body!.enabled !== "boolean") {
    return NextResponse.json({ ok: false, error: "enabled must be a boolean" });
  }
  if (forwardTypesPresent && !isValidForwardTypes(body!.forwardTypes)) {
    return NextResponse.json({ ok: false, error: "forwardTypes must be an array of forwardable event types with no duplicates" });
  }
  return collectorResponse(() => {
    const db = deps.getDb();
    if (enabledPresent) deps.setAfk(db, body!.enabled as boolean);
    if (forwardTypesPresent) deps.setForwardTypes(db, body!.forwardTypes as EventType[]);
    return { enabled: deps.getAfkEnabled(db), forwardTypes: deps.getForwardTypes(db) };
  });
}
