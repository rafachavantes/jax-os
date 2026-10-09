import { NextResponse } from "next/server";
import { buildIndex, gateVaultScope, parseScope } from "@/server/collectors/files";
import { readGeneralSettings } from "@/server/settings";

export const dynamic = "force-dynamic";

// Read-only (CSRF note, spec §4): follows the read precedent, no requireSameOrigin — same as
// tree/read/raw/download/search today.
export async function GET(req: Request) {
  const scope = parseScope(new URL(req.url).searchParams.get("scope"));
  if (!scope) return NextResponse.json({ ok: false, error: "invalid payload" });
  const settings = readGeneralSettings();
  const gated = gateVaultScope(scope, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null);
  if (gated === "refuse") return NextResponse.json({ ok: false, error: "disabled" });
  try {
    const result = await buildIndex(gated);
    return NextResponse.json({
      ok: true,
      data: result.data,
      truncated: result.truncated,
      truncatedRoots: result.truncatedRoots,
      builtAt: result.builtAt,
    });
  } catch (e) {
    return NextResponse.json({ ok: false, error: e instanceof Error ? e.message : String(e) });
  }
}
