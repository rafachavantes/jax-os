import { NextResponse } from "next/server";
import {
  gateVaultRoot, imageMime, mediaMime, readRawPreviewAt, resolveExisting, ROOTS,
  RAW_PDF_CAP, RAW_MEDIA_CAP, type Root,
} from "@/server/collectors/files";
import { openRawStreamAt, parseRange, streamHandle } from "@/server/collectors/fileDownload";
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

  const imgMime = imageMime(rel);
  if (imgMime) {
    // Unchanged whole-buffer path (images, incl. SVG) — only Content-Disposition changed to
    // "inline" (spec §7: harmless for <img>, and consistent with every other type raw now serves).
    try {
      const bytes = readRawPreviewAt(resolveExisting(root as Root, rel));
      return new NextResponse(new Uint8Array(bytes), {
        headers: {
          "Content-Type": imgMime,
          "Cache-Control": "no-store",
          "Content-Security-Policy": "default-src 'none'; sandbox",
          "X-Content-Type-Options": "nosniff",
          "Content-Disposition": "inline",
        },
      });
    } catch (error) {
      const message = error instanceof Error && error.message === "too large" ? "too large" : "read failed";
      return NextResponse.json({ ok: false, error: message });
    }
  }

  const media = mediaMime(rel);
  if (media) {
    const cap = media.kind === "pdf" ? RAW_PDF_CAP : RAW_MEDIA_CAP;
    let opened: { handle: Awaited<ReturnType<typeof openRawStreamAt>>["handle"]; size: number };
    try {
      opened = await openRawStreamAt(resolveExisting(root as Root, rel), cap);
    } catch (error) {
      const message = error instanceof Error && error.message === "too large" ? "too large" : "read failed";
      return NextResponse.json({ ok: false, error: message }, { status: message === "too large" ? 413 : 200 });
    }
    const { handle, size } = opened;
    const range = parseRange(req.headers.get("range"), size);
    if (range === "unsatisfiable") {
      await handle.close().catch(() => undefined);
      return new NextResponse(null, { status: 416, headers: { "Content-Range": `bytes */${size}` } });
    }
    const headers: Record<string, string> = {
      "Content-Type": media.mime,
      "Cache-Control": "no-store",
      "Content-Security-Policy": "default-src 'none'; sandbox",
      "X-Content-Type-Options": "nosniff",
      "Content-Disposition": "inline",
      "Accept-Ranges": "bytes",
    };
    const body = streamHandle(handle, size, range ?? undefined);
    if (range) {
      headers["Content-Range"] = `bytes ${range.start}-${range.end}/${size}`;
      headers["Content-Length"] = String(range.end - range.start + 1);
      return new NextResponse(body, { status: 206, headers });
    }
    headers["Content-Length"] = String(size);
    return new NextResponse(body, { status: 200, headers });
  }

  // html/htm never reach here — spec §5 item 10 routes them through /api/files/read + srcdoc
  // (Part 2). Any other unrecognized extension falls through unchanged.
  return NextResponse.json({ ok: false, error: "unsupported type" });
}
