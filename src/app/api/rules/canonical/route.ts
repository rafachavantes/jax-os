import { NextResponse } from "next/server";
import { readJsonCapped, requireSameOrigin } from "@/server/api";
import { norm } from "@/lib/rules";
import { getDb } from "@/server/db";
import { getRuleDocs, setRuleDoc } from "@/server/db/rules";

const CONTENT_CAP = 256 * 1024;
const REVISION_RE = /^[0-9a-f]{64}$/;

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;

  const body = await readJsonCapped(req, CONTENT_CAP + 4096);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error }, { status: 400 });
  const { content, expectedRevision } = (body.value ?? {}) as Record<string, unknown>;
  if (typeof content !== "string" || Buffer.byteLength(content, "utf8") > CONTENT_CAP) {
    return NextResponse.json({ ok: false, error: "invalid content" }, { status: 400 });
  }
  if (typeof expectedRevision !== "string" || !REVISION_RE.test(expectedRevision)) {
    return NextResponse.json({ ok: false, error: "invalid expectedRevision" }, { status: 400 });
  }

  // ALL DB work inside one try: getDb/getRuleDocs/setRuleDoc failures
  // (including the unseeded-table guard) are internal errors → 500. 400 is
  // reserved for invalid input (spec §5).
  const normalized = norm(content);
  try {
    const db = getDb();
    const previous = getRuleDocs(db).global;
    const outcome = setRuleDoc(
      db,
      "global",
      normalized,
      { ts: new Date().toISOString(), kind: "rules-edit", ok: true, slot: "global", previous, content: normalized },
      expectedRevision,
    );
    if (!outcome.ok) {
      return NextResponse.json(
        { ok: false, code: "stale-revision", error: "rule source changed since it was read" },
        { status: 409 },
      );
    }
    return NextResponse.json({ ok: true, data: { content: normalized, revision: outcome.revision } });
  } catch (e) {
    return NextResponse.json({ ok: false, error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}
