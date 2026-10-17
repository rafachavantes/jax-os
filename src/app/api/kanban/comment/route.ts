import { NextResponse } from "next/server";
import { createComment } from "../../../../server/collectors/linear";
import { linearEnabled } from "../../../../server/gates";
import { runMutation } from "../../../../server/mutations";
import { readJsonObject } from "../../../../server/api";
import { COMMENT_CAP, COMMENT_JSON_CAP, boundedString } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  if (!linearEnabled()) return NextResponse.json({ ok: false, error: "disabled" });
  const parsed = await readJsonObject(req, COMMENT_JSON_CAP);
  if ("error" in parsed) return parsed.error;
  const payload = parsed.payload;
  const { issueId, body } = payload as { issueId?: unknown; body?: unknown };
  if (!boundedString(issueId, 256) || !boundedString(body, COMMENT_CAP) || !body.trim()) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const result = await runMutation(
    {
      ts: new Date().toISOString(),
      kind: "linear-comment",
      issueId,
      preview: body.length > 200 ? body.slice(0, 200) + "…" : body,
    },
    () => createComment(issueId, body),
  );
  if (result.ok) return NextResponse.json({ ok: true });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
