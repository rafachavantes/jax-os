import { NextResponse } from "next/server";
import { readJsonCapped, requireSameOrigin } from "@/server/api";
import { APP_KEYS, norm, type AppKey } from "@/lib/rules";
import { getDb } from "@/server/db";
import { getRuleDocs, setRuleDoc } from "@/server/db/rules";

const CONTENT_CAP = 256 * 1024;
const REVISION_RE = /^[0-9a-f]{64}$/;

export async function POST(req: Request) {
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;

  const body = await readJsonCapped(req, CONTENT_CAP + 4096);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error }, { status: 400 });
  const { app, content, expectedRevision } = (body.value ?? {}) as Record<string, unknown>;
  if (
    typeof app !== "string" ||
    !(APP_KEYS as readonly string[]).includes(app) ||
    typeof content !== "string" ||
    Buffer.byteLength(content, "utf8") > CONTENT_CAP
  ) {
    return NextResponse.json({ ok: false, error: "invalid payload" }, { status: 400 });
  }
  if (typeof expectedRevision !== "string" || !REVISION_RE.test(expectedRevision)) {
    return NextResponse.json({ ok: false, error: "invalid expectedRevision" }, { status: 400 });
  }

  const appKey = app as AppKey;
  // "" is the "no exception" sentinel (spec §3) and must stay exactly "" —
  // norm("") would produce "\n", which is NOT the sentinel.
  const normalized = content === "" ? "" : norm(content);
  // ALL DB work inside one try (same rationale as the canonical route): DB
  // failures are 500, never 400.
  try {
    const db = getDb();
    const previous = getRuleDocs(db)[appKey];
    const outcome = setRuleDoc(
      db,
      appKey,
      normalized,
      {
        ts: new Date().toISOString(),
        kind: "rules-exception-edit",
        ok: true,
        slot: appKey,
        previous,
        content: normalized,
      },
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
