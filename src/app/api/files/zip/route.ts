import { NextResponse } from "next/server";
import { basename } from "node:path";
import { gateVaultRoot, planZip, ROOTS, strictRel, zipStream, type Root } from "../../../../server/collectors/files";
import { contentDisposition } from "../../../../server/collectors/fileDownload";
import { readGeneralSettings } from "../../../../server/settings";
import { validRel } from "../../../../server/inputLimits";

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
  if (!validRel(rel, true) || !strictRel(rel, true)) return NextResponse.json({ ok: false, error: "invalid payload" });
  let plan: Awaited<ReturnType<typeof planZip>>;
  try {
    plan = await planZip(root as Root, rel);
  } catch {
    return NextResponse.json({ ok: false, error: "zip unavailable" });
  }
  if (!plan.ok) return NextResponse.json({ ok: false, error: plan.error });
  const folderName = rel === "" ? root : basename(rel);
  // Node 22's global ReadableStream has static .from over an async generator (runtime-confirmed);
  // the DOM-lib types Next type-checks against predate it, so shim just that static's type.
  const body = (ReadableStream as unknown as {
    from: (gen: AsyncGenerator<Uint8Array>) => ReadableStream<Uint8Array>;
  }).from(zipStream(plan.entries));
  return new NextResponse(body, {
    headers: {
      "Content-Type": "application/zip",
      "Content-Disposition": contentDisposition(`${folderName}.zip`),
      "Cache-Control": "no-store",
      "X-Content-Type-Options": "nosniff",
      "X-Secrets-Excluded": String(plan.secretsExcluded),
      "X-Zip-Entries": String(plan.entries.length),
    },
  });
}
