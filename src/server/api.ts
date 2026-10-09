import { NextResponse } from "next/server";
import type { Envelope } from "@/lib/api";

// Error contract: source unavailable → 200 {ok:false} (visible warning UI);
// empty data stays {ok:true, data:[]} (quiet empty). Never a 500 — the
// serialization itself is inside the guard so a non-serializable collector
// value degrades to {ok:false} too.
export async function collectorResponse<T>(fn: () => T | Promise<T>) {
  try {
    return NextResponse.json({ ok: true, data: await fn() } satisfies Envelope<T>);
  } catch (e) {
    return NextResponse.json({
      ok: false,
      error: e instanceof Error ? e.message : String(e),
    } satisfies Envelope<T>);
  }
}

// One shared CSRF allowlist for every workflow POST route (cold review finding 8e). The three
// pre-existing routes each checked only `=== "cross-site"`, silently accepting `same-site` — a
// hostile page served by any OTHER process bound to 127.0.0.1 on a different port reports
// same-site, not cross-site, to Fetch Metadata (site = registrable domain, not origin). This is
// the stricter allowlist `answer/handler.ts` already used: accept same-origin/no-header, reject
// everything else. A loopback script caller (curl, workflow_poll.py) sends no header at all and
// is accepted by both the old and the new guard identically.
export function requireSameOrigin(req: Request): NextResponse | null {
  const fetchSite = req.headers.get("sec-fetch-site");
  if (fetchSite && fetchSite !== "same-origin" && fetchSite !== "none") {
    return NextResponse.json({ ok: false, error: "cross-site request blocked" });
  }
  return null;
}

export async function readBodyCapped(
  req: Request, maxBytes: number,
): Promise<{ ok: true; value: Uint8Array<ArrayBuffer> } | { ok: false; error: string }> {
  const declared = Number(req.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > maxBytes)
    return { ok: false, error: "request too large" };
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  try { reader = req.body?.getReader(); }
  catch { return { ok: false, error: "could not read request body" }; }
  if (!reader) return { ok: true, value: new Uint8Array(0) };
  const chunks: Uint8Array[] = [];
  let seen = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      seen += value.byteLength;
      if (seen > maxBytes) {
        await reader.cancel().catch(() => {});
        return { ok: false, error: "request too large" };
      }
      chunks.push(value);
    }
    const bytes = new Uint8Array(seen);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
    return { ok: true, value: bytes };
  } catch {
    return { ok: false, error: "could not read request body" };
  } finally {
    reader.releaseLock();
  }
}

// Bounded JSON body read for the workflow POST routes: enforces the byte cap
// and turns a broken body stream into a contract-shaped {ok:false} instead of
// a 5xx (error contract). Returns the parsed value or an error reason.
export async function readJsonCapped(
  req: Request,
  maxBytes: number,
): Promise<{ ok: true; value: unknown } | { ok: false; error: string }> {
  const body = await readBodyCapped(req, maxBytes);
  if (!body.ok) return body;
  try {
    const text = new TextDecoder().decode(body.value);
    return { ok: true, value: text === "" ? {} : JSON.parse(text) };
  } catch {
    return { ok: false, error: "invalid JSON" };
  }
}
