import { NextResponse } from "next/server";
import { createComment } from "../../../../server/collectors/linear";
import { readGeneralSettings } from "../../../../server/settings";
import { runMutation } from "../../../../server/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { COMMENT_CAP, COMMENT_JSON_CAP, boundedString } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  const settings = readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.linear)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const bodyRead = await readJsonCapped(req, COMMENT_JSON_CAP);
  if (!bodyRead.ok) return NextResponse.json({ ok: false, error: bodyRead.error });
  const payload = bodyRead.value;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return NextResponse.json({ ok: false, error: "invalid payload" });
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
