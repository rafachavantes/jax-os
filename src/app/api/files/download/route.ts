import { NextResponse } from "next/server";
import { gateVaultRoot, ROOTS, type Root } from "@/server/collectors/files";
import { downloadFile } from "@/server/collectors/fileDownload";
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
  if (!validRel(rel, false)) return NextResponse.json({ ok: false, error: "invalid payload" });
  let file: Awaited<ReturnType<typeof downloadFile>> | undefined;
  try {
    file = await downloadFile(root as Root, rel);
    return new NextResponse(file.body, {
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Length": String(file.size),
        "Content-Disposition": file.disposition,
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
      },
    });
  } catch {
    await file?.body.cancel().catch(() => undefined);
    return NextResponse.json({ ok: false, error: "read failed" });
  }
}
