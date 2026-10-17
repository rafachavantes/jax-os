import { NextResponse } from "next/server";
import { parseSettings, readGeneralSettings, writeGeneralSettings } from "../../../server/settings";
import { envFileStatus } from "../../../server/envFile";
import { nativeCredentialsConfigured } from "../../../server/collectors/subscriptions";
import { getDb } from "../../../server/db";
import { readJsonCapped, requireSameOrigin } from "../../../server/api";
import { SMALL_JSON_CAP } from "../../../server/inputLimits";

export const dynamic = "force-dynamic";

// Spec's Notifications row: read-only configured/missing status for both env vars that back the
// notification webhook — never persisted to settings.json, never editable here (Non-goals: no
// URL/secret field). Computed fresh on every request, sibling to `data` on an ok:true response
// only (the error shape stays exactly what the reader produced — no extra field).
function webhookConfigured(): boolean {
  return Boolean(process.env.NOTIFICATION_WEBHOOK_URL) && Boolean(process.env.NOTIFICATION_WEBHOOK_SECRET);
}

export async function GET() {
  const result = readGeneralSettings();
  if (!result.ok) return NextResponse.json(result);
  return NextResponse.json({ ...result, webhookConfigured: webhookConfigured(), credentialsConfigured: nativeCredentialsConfigured(), envFileState: envFileStatus().state });
}

// Decision 15: full-object save, merged server-side — never trusts the client to have every
// field. Only settings-unreadable (an OS-level read failure) refuses outright; a malformed
// current file merges onto the schema defaults instead, so the Geral form is itself how the
// owner repairs a malformed settings.json (Self-review (g)).
const TOP_KEYS = ["ownerName", "locale", "reposRoot", "vaultPath", "monitoredUnits", "integrations"] as const;

export async function PUT(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const body = await readJsonCapped(req, SMALL_JSON_CAP);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error });
  const payload = body.value;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return NextResponse.json({ ok: false, error: "invalid payload" });

  const current = readGeneralSettings();
  if (!current.ok && current.error === "settings-unreadable") {
    return NextResponse.json({ ok: false, error: "settings-unreadable" });
  }
  // 497A's parseSettings is the one shape validator (its `{}` input resolves to all schema
  // defaults — a missing field falls back independently, no local DEFAULTS copy needed).
  let base = current.ok ? current.data : null;
  if (!base) {
    const defaults = parseSettings({});
    if (!defaults.ok) return NextResponse.json({ ok: false, error: "settings-malformed" });
    base = defaults.data;
  }
  const merged: Record<string, unknown> = { ...base };
  for (const key of TOP_KEYS) {
    if (Object.hasOwn(payload, key)) merged[key] = (payload as Record<string, unknown>)[key];
  }
  const validated = parseSettings(merged);
  if (!validated.ok) return NextResponse.json({ ok: false, error: "settings-malformed" });
  // The writer audits its own mutations row; a throw (audit store unavailable/failed) must
  // surface as the house {ok:false} envelope, never a 5xx (error contract).
  try {
    writeGeneralSettings(getDb(), validated.data);
  } catch (e) {
    return NextResponse.json({ ok: false, error: e instanceof Error ? e.message : String(e) });
  }
  return NextResponse.json({ ok: true, data: validated.data, webhookConfigured: webhookConfigured(), credentialsConfigured: nativeCredentialsConfigured(), envFileState: envFileStatus().state });
}
