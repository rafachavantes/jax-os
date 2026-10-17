import { NextResponse } from "next/server";
import { gateVaultRoot, READ_CAP, ROOTS, writeFile, type Root } from "../../../../server/collectors/files";
import { readGeneralSettings } from "../../../../server/settings";
import { fileEffect, runMutation } from "../../../../server/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { WRITE_JSON_CAP, boundedString, validRel } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const body = await readJsonCapped(req, WRITE_JSON_CAP);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error });
  const payload = body.value;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return NextResponse.json({ ok: false, error: "invalid payload" });
  const { root, rel, content, baseHash } = payload as Record<string, unknown>;
  if (
    !(typeof root === "string" && Object.hasOwn(ROOTS, root)) ||
    !validRel(rel) ||
    !boundedString(content, READ_CAP, true) ||
    typeof baseHash !== "string" ||
    !/^[a-fA-F0-9]{64}$/.test(baseHash)
  ) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const settings = readGeneralSettings();
  if (gateVaultRoot(root, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "file-edit", root, rel },
    () => fileEffect(() => writeFile(root as Root, rel, content, baseHash.toLowerCase())),
  );
  if (result.ok) return NextResponse.json({ ok: true, hash: result.value.hash });
  const { value, status: _status, ...failure } = result;
  return NextResponse.json({ ...failure, ...(value ? { hash: value.hash } : {}) });
}
