import { NextResponse } from "next/server";
import { gateVaultRoot, ROOTS, renameEntry, type Root } from "../../../../server/collectors/files";
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
  const { root, relFrom, relTo } = payload as Record<string, unknown>;
  if (
    !(typeof root === "string" && Object.hasOwn(ROOTS, root)) ||
    !validRel(relFrom) ||
    !validRel(relTo) ||
    !validBasename(relTo.slice(relTo.lastIndexOf("/") + 1))
  ) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const settings = readGeneralSettings();
  if (gateVaultRoot(root, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "file-rename", root, from: relFrom, to: relTo },
    () => fileEffect(() => renameEntry(root as Root, relFrom, relTo)),
  );
  if (result.ok) return NextResponse.json({ ok: true });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
