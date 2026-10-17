import { NextResponse } from "next/server";
import { gateVaultScope, parseScope } from "../../../../server/collectors/files";
import { searchContent } from "../../../../server/collectors/fileContentSearch";
import { readGeneralSettings } from "../../../../server/settings";
import { boundedString } from "../../../../server/inputLimits";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const url = new URL(req.url);
  const q = url.searchParams.get("q") ?? "";
  if (!boundedString(q, 256, true)) return NextResponse.json({ ok: false, error: "invalid payload" });
  const scope = parseScope(url.searchParams.get("scope"));
  if (!scope) return NextResponse.json({ ok: false, error: "invalid payload" });
  const settings = readGeneralSettings();
  const gated = gateVaultScope(scope, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null);
  if (gated === "refuse") return NextResponse.json({ ok: false, error: "disabled" });
  try {
    const result = await searchContent(gated, q, req.signal);
    return NextResponse.json({ ok: true, data: result.hits, truncated: result.truncated });
  } catch {
    return NextResponse.json({ ok: false, error: req.signal.aborted ? "search cancelled" : "content search unavailable" });
  }
}
