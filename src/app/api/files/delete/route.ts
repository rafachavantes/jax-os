import { NextResponse } from "next/server";
import { gateVaultRoot, ROOTS, removeExpiredTrashEntryAt, strictRel, sweepTrash, trashEntry, type Root } from "../../../../server/collectors/files";
import { readGeneralSettings } from "../../../../server/settings";
import { fileEffect, runMutation } from "../../../../server/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { SMALL_JSON_CAP, validRel } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const body = await readJsonCapped(req, SMALL_JSON_CAP);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error });
  const payload = body.value;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return NextResponse.json({ ok: false, error: "invalid payload" });
  const { root, rel } = payload as Record<string, unknown>;
  if (
    !(typeof root === "string" && Object.hasOwn(ROOTS, root)) ||
    !validRel(rel) ||
    !strictRel(rel)
  ) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const settings = readGeneralSettings();
  if (gateVaultRoot(root, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "file-delete", root, rel },
    () => fileEffect(() => trashEntry(root as Root, rel)),
    (value) => ({ trashRel: value.trashRel }),
  );
  if (result.ok) {
    // Opportunistic sweep (§5 item 6: "no cron, reuses the request already touching that root's
    // trash"). Each expired entry is removed INSIDE its own runMutation effect — the pending audit
    // row lands before the rm, never after (round 1 plan review F1) — so an audit/mutation failure
    // on entry N stops the loop at N; entries already removed stay removed and recorded. A sweep
    // hiccup (the try/catch below) must never turn an already-successful delete into a failure
    // response — the primary mutation is done and audited by the runMutation call above it.
    try {
      for (const expired of sweepTrash(root as Root)) {
        const swept = await runMutation(
          { ts: new Date().toISOString(), kind: "trash-sweep", root, trashRel: expired.trashRel },
          () => fileEffect(() => removeExpiredTrashEntryAt(expired.absPath)),
        );
        if (!swept.ok) break;
      }
    } catch {
      // swallow — see comment above
    }
    return NextResponse.json({ ok: true, trashRel: result.value.trashRel });
  }
  // Preserve trashRel on applied-unrecorded (diff review 617d73a5a89b F2) — the audit row failed to
  // finalize but the trash-rename already happened, so the client still needs it for Undo.
  const { value, status: _status, ...failure } = result;
  return NextResponse.json({ ...failure, ...(value ? { trashRel: value.trashRel } : {}) });
}
