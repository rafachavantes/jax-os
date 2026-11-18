import { NextResponse } from "next/server";
import { getDb } from "../../../../server/db";
import {
  getCockpitPrefs, resetCockpitLayout, saveCockpitLayout, setCockpitPrScope,
  type LayoutWriteResult,
} from "../../../../server/db/cockpit-prefs";
import { collectorResponse, readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { isPrScope, parseLayoutEntries, type CockpitSnapshot } from "../../../../lib/cockpitLayout";

export type CockpitPrefsDeps = {
  getDb: typeof getDb;
  getCockpitPrefs: typeof getCockpitPrefs;
  saveCockpitLayout: typeof saveCockpitLayout;
  resetCockpitLayout: typeof resetCockpitLayout;
  setCockpitPrScope: typeof setCockpitPrScope;
};

const systemDeps: CockpitPrefsDeps = { getDb, getCockpitPrefs, saveCockpitLayout, resetCockpitLayout, setCockpitPrScope };

const WRITE_CAP = 192 * 1024;

type WriteData = { conflict: boolean; snapshot: CockpitSnapshot };

function invalid() {
  return NextResponse.json({ ok: false, error: "invalid payload" });
}

function writeData(result: LayoutWriteResult): WriteData {
  return result.ok ? { conflict: false, snapshot: result.snapshot } : { conflict: true, snapshot: result.snapshot };
}

function revisionOf(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : null;
}

// Thin read: the DB helper owns defaults and the corrupt-row source error.
export async function handleCockpitPrefsGet(deps: CockpitPrefsDeps = systemDeps) {
  return collectorResponse(() => deps.getCockpitPrefs(deps.getDb()));
}

// Discriminated action body. Shapes are closed per action — an extra key is a malformed request,
// not an ignored field — and every storage exception becomes the safe {ok:false} warning.
export async function handleCockpitPrefsPost(req: Request, deps: CockpitPrefsDeps = systemDeps) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  const read = await readJsonCapped(req, WRITE_CAP);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value;
  if (!body || typeof body !== "object" || Array.isArray(body)) return invalid();
  const record = body as Record<string, unknown>;
  const action = record.action;

  if (action === "layout-save") {
    const keys = Object.keys(record);
    if (keys.length !== 3 || keys.some((k) => !["action", "expectedRevision", "projects"].includes(k))) return invalid();
    const expectedRevision = revisionOf(record.expectedRevision);
    const projects = parseLayoutEntries(record.projects);
    if (expectedRevision === null || projects === null) return invalid();
    return collectorResponse(() => writeData(deps.saveCockpitLayout(deps.getDb(), expectedRevision, projects)));
  }

  if (action === "layout-reset") {
    const keys = Object.keys(record);
    if (keys.length !== 2 || keys.some((k) => !["action", "expectedRevision"].includes(k))) return invalid();
    const expectedRevision = revisionOf(record.expectedRevision);
    if (expectedRevision === null) return invalid();
    return collectorResponse(() => writeData(deps.resetCockpitLayout(deps.getDb(), expectedRevision)));
  }

  if (action === "pr-scope") {
    const keys = Object.keys(record);
    if (keys.length !== 2 || keys.some((k) => !["action", "scope"].includes(k))) return invalid();
    if (!isPrScope(record.scope)) return invalid();
    const scope = record.scope;
    return collectorResponse(() => deps.setCockpitPrScope(deps.getDb(), scope));
  }

  return invalid();
}
