import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { getDb } from "@/server/db";
import { decodeMutationCursor, queryMutations } from "@/server/db/mutations";
import { boundedString } from "@/server/inputLimits";

export const dynamic = "force-dynamic";

const DAY = /^\d{4}-\d{2}-\d{2}$/;

function parseLimit(v: string | null): number {
  if (v === null || v === "") return 50;
  const n = Number(v);
  if (!Number.isInteger(n) || n <= 0) return 50;
  return Math.min(n, 200);
}

function validAuditBound(v: string): boolean {
  if (Buffer.byteLength(v, "utf8") > 64 || v.length === 0 || /[\u0000-\u001f\u007f]/.test(v)) return false;
  return Number.isFinite(Date.parse(DAY.test(v) ? `${v}T00:00:00Z` : v));
}

export async function GET(req: Request) {
  const u = new URL(req.url);
  if (u.searchParams.has("offset")) {
    return NextResponse.json({ ok: false, error: "offset is not supported" });
  }
  const kindRaw = u.searchParams.get("kind") || undefined;
  if (kindRaw !== undefined && !boundedString(kindRaw, 256)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const from = u.searchParams.get("from") || undefined;
  const to = u.searchParams.get("to") || undefined;
  if ((from && !validAuditBound(from)) || (to && !validAuditBound(to))) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  let cursor: { ts: string; id: number } | undefined;
  if (u.searchParams.has("cursor")) {
    const decoded = decodeMutationCursor(u.searchParams.get("cursor") ?? "");
    if (!decoded) return NextResponse.json({ ok: false, error: "invalid cursor" });
    cursor = decoded;
  }
  return collectorResponse(() =>
    queryMutations(getDb(), {
      kind: kindRaw,
      from,
      to,
      limit: parseLimit(u.searchParams.get("limit")),
      cursor,
    }),
  );
}
