import { NextResponse } from "next/server";
import { gateVaultRoot, ROOTS, createEntry, type Root } from "../../../../server/collectors/files";
import { readGeneralSettings } from "../../../../server/settings";
import { fileEffect, runMutation } from "../../../../server/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { SMALL_JSON_CAP, validBasename, validRel } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const body = await readJsonCapped(req, SMALL_JSON_CAP);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error });
  const payload = body.value;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return NextResponse.json({ ok: false, error: "invalid payload" });
  const { root, relParentDir, basename, kind } = payload as Record<string, unknown>;
  if (
    !(typeof root === "string" && Object.hasOwn(ROOTS, root)) ||
    !validRel(relParentDir, true) ||
    !validBasename(basename) ||
    (kind !== "file" && kind !== "folder")
  ) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const settings = readGeneralSettings();
  if (gateVaultRoot(root, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "file-create", root, relParentDir, basename, entryKind: kind },
    () => fileEffect(() => createEntry(root as Root, relParentDir, basename, kind)),
  );
  if (result.ok) return NextResponse.json({ ok: true });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
