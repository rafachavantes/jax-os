import { NextResponse } from "next/server";
import { gateVaultRoot, ROOTS, UPLOAD_CAP, uploadFile, type Root } from "../../../../server/collectors/files";
import { readGeneralSettings } from "../../../../server/settings";
import { fileEffect, runMutation } from "../../../../server/mutations";
import { readBodyCapped, requireSameOrigin } from "../../../../server/api";
import { validBasename, validRel } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const cap = UPLOAD_CAP + 64 * 1024;
  const declared = req.headers.get("content-length");
  const length = declared !== null && /^\d+$/.test(declared) ? Number(declared) : NaN;
  if (!Number.isSafeInteger(length) || length <= 0 || length > cap)
    return NextResponse.json({ ok: false, error: "invalid content length" });
  const body = await readBodyCapped(req, cap);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error });
  const form = await new Response(body.value, {
    headers: { "content-type": req.headers.get("content-type") ?? "" },
  }).formData().catch(() => null);
  if (!form) return NextResponse.json({ ok: false, error: "invalid payload" });
  const root = form.get("root");
  const relParentDir = form.get("relParentDir") ?? "";
  const file = form.get("file");
  if (typeof root !== "string" || !Object.hasOwn(ROOTS, root)
      || !validRel(relParentDir, true) || !(file instanceof File)
      || file.size > UPLOAD_CAP || !validBasename(file.name))
    return NextResponse.json({ ok: false, error: "invalid payload" });
  const settings = readGeneralSettings();
  if (gateVaultRoot(root, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  let bytes: Buffer;
  try {
    bytes = Buffer.from(await file.arrayBuffer());
  } catch {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "file-upload", root, relParentDir, basename: file.name },
    () => fileEffect(() => uploadFile(root as Root, relParentDir, file.name, bytes)),
  );
  if (result.ok) return NextResponse.json({ ok: true });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
