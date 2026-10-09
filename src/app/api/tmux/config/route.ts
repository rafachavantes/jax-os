import { NextResponse } from "next/server";
import type { Envelope, TtydConfig } from "@/lib/api";
import { readGeneralSettings } from "@/server/settings";

export const dynamic = "force-dynamic";

export async function GET() {
  // MOA-496: disabled acts like "not configured" (the terminal panel already renders the
  // existing not-configured SourceWarning for it) and TTYD_RO_URL/TTYD_RW_URL are never read.
  const settings = readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.ttyd)) {
    return NextResponse.json({ ok: true, data: { configured: false, reachable: false } } satisfies Envelope<TtydConfig>);
  }
  const roUrl = process.env.TTYD_RO_URL;
  const rwUrl = process.env.TTYD_RW_URL;
  const configured = Boolean(roUrl && rwUrl);
  let reachable = false;
  if (configured) {
    try {
      // Decision 10: a 4xx/5xx now reports unreachable — matches what a human clicking the
      // iframe would actually see. Plain GET of ttyd's HTML is still side-effect-free: tmux
      // attach only runs on WebSocket connect, never on page GET.
      const res = await fetch(roUrl as string, { signal: AbortSignal.timeout(800) });
      reachable = res.ok;
    } catch {}
  }
  const data: TtydConfig = configured
    ? { configured, reachable, roUrl, rwUrl }
    : { configured, reachable };
  return NextResponse.json({ ok: true, data } satisfies Envelope<TtydConfig>);
}
