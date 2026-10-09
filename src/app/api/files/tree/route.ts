import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { gateVaultRoot, listDir, ROOTS, type Root } from "@/server/collectors/files";
import { readGeneralSettings } from "@/server/settings";
import { validRel } from "@/server/inputLimits";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const u = new URL(req.url);
  const root = u.searchParams.get("root") ?? "";
  const rel = u.searchParams.get("rel") ?? "";
  if (!Object.hasOwn(ROOTS, root)) return NextResponse.json({ ok: false, error: "bad root" });
  const settings = readGeneralSettings();
  if (gateVaultRoot(root, settings.ok && settings.data.integrations.vault && settings.data.vaultPath !== null)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  if (!validRel(rel, true)) return NextResponse.json({ ok: false, error: "invalid payload" });
  return collectorResponse(() => listDir(root as Root, rel));
}
