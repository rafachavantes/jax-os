import { NextResponse } from "next/server";
import { getDb } from "../../../../server/db";
import { getArchiveAfterDays, setArchiveAfterDays } from "../../../../server/db/workflows";
import { collectorResponse, readJsonCapped, requireSameOrigin } from "../../../../server/api";

export type SettingsRouteDeps = {
  getDb: typeof getDb;
  getArchiveAfterDays: typeof getArchiveAfterDays;
  setArchiveAfterDays: typeof setArchiveAfterDays;
};
const systemDeps: SettingsRouteDeps = { getDb, getArchiveAfterDays, setArchiveAfterDays };

export async function handleSettingsGet(_req?: Request, deps: SettingsRouteDeps = systemDeps) {
  return collectorResponse(() => ({ archiveAfterDays: deps.getArchiveAfterDays(deps.getDb()) }));
}

export async function handleSettingsPost(req: Request, deps: SettingsRouteDeps = systemDeps) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  const read = await readJsonCapped(req, 1024);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value as { archiveAfterDays?: unknown } | null;
  if (!Number.isInteger(body?.archiveAfterDays)) return NextResponse.json({ ok: false, error: "invalid payload" });
  const days = body!.archiveAfterDays as number;
  // setArchiveAfterDays throws outside 1-30 (Part 1) — collectorResponse turns that into a quiet
  // {ok:false, error} 200, never a 5xx, and never writes on the throwing path.
  return collectorResponse(() => {
    deps.setArchiveAfterDays(deps.getDb(), days);
    return { archiveAfterDays: days };
  });
}
