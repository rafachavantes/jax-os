import { NextResponse } from "next/server";
import { gateVaultRoot, parseTrashRel, ROOTS, restoreTrashEntry, type Root } from "../../../../../server/collectors/files";
import { readGeneralSettings } from "../../../../../server/settings";
import { fileEffect, runMutation } from "../../../../../server/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../../server/api";
import { SMALL_JSON_CAP, validRel } from "../../../../../server/inputLimits";

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const body = await readJsonCapped(req, SMALL_JSON_CAP);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error });
  const payload = body.value;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return NextResponse.json({ ok: false, error: "invalid payload" });
  const { root, trashRel } = payload as Record<string, unknown>;
  // Shape-validate with the dedicated parser BEFORE runMutation (diff review 0e652b904cea F3) — a
  // traversal-shaped trashRel is rejected here, never reaches the audit log. The length bound is
  // checked on the PARSED original rel, not the ".jax-trash/<timestamp>/" envelope (diff review
  // 617d73a5a89b F1) — a valid 4096-byte original path would otherwise fail the bound once wrapped.
  const parsed = typeof trashRel === "string" ? parseTrashRel(trashRel) : null;
  if (!(typeof root === "string" && Object.hasOwn(ROOTS, root)) || !parsed || !validRel(parsed.rel)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const settings = readGeneralSettings();
  if (gateVaultRoot(root, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  // trashRel is known upfront here (from delete's response) — no details-callback needed.
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "file-trash-restore", root, trashRel },
    () => fileEffect(() => restoreTrashEntry(root as Root, trashRel as string)),
  );
  if (result.ok) return NextResponse.json({ ok: true, restoredRel: result.value.restoredRel });
  // Preserve restoredRel on applied-unrecorded (diff review 617d73a5a89b F2) — the audit row failed
  // to finalize but the rename already happened, so the client still needs it.
  const { value, status: _status, ...failure } = result;
  return NextResponse.json({ ...failure, ...(value ? { restoredRel: value.restoredRel } : {}) });
}
