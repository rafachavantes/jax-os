import { NextResponse } from "next/server";
import { getDb } from "../../../../server/db";
import { setProjectPrefs } from "../../../../server/db/workflows";
import { collectorResponse, readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { validBasename } from "../../../../server/inputLimits";

export type PrefsRouteDeps = { getDb: typeof getDb; setProjectPrefs: typeof setProjectPrefs };
const systemDeps: PrefsRouteDeps = { getDb, setProjectPrefs };

// Boolean-shaped body (spec §10): {project, hidden?, pinned?}, at least one of hidden/pinned
// present. setProjectPrefs (Part 1) already audits the write via insertMutation — no separate
// runMutation wrapper, same shape as POST /api/workflow/afk.
export async function handlePrefsPost(req: Request, deps: PrefsRouteDeps = systemDeps) {
  const originError = requireSameOrigin(req);
  if (originError) return originError;
  const read = await readJsonCapped(req, 1024);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value as { project?: unknown; hidden?: unknown; pinned?: unknown } | null;
  if (!validBasename(body?.project)) return NextResponse.json({ ok: false, error: "invalid payload" });
  if (body!.hidden !== undefined && typeof body!.hidden !== "boolean") return NextResponse.json({ ok: false, error: "invalid payload" });
  if (body!.pinned !== undefined && typeof body!.pinned !== "boolean") return NextResponse.json({ ok: false, error: "invalid payload" });
  if (body!.hidden === undefined && body!.pinned === undefined) return NextResponse.json({ ok: false, error: "invalid payload" });
  const project = body!.project as string;
  return collectorResponse(() =>
    deps.setProjectPrefs(deps.getDb(), project, { hidden: body!.hidden as boolean | undefined, pinned: body!.pinned as boolean | undefined }),
  );
}
