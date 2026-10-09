import { NextResponse } from "next/server";
import { moveIssue } from "../../../../server/collectors/linear";
import { readGeneralSettings } from "../../../../server/settings";
import { runMutation } from "../../../../server/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { SMALL_JSON_CAP, boundedString } from "../../../../server/inputLimits";

export async function POST(req: Request) {
  const settings = readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.linear)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  const csrf = requireSameOrigin(req);
  if (csrf) return csrf;
  const body = await readJsonCapped(req, SMALL_JSON_CAP);
  if (!body.ok) return NextResponse.json({ ok: false, error: body.error });
  const payload = body.value;
  if (!payload || typeof payload !== "object" || Array.isArray(payload))
    return NextResponse.json({ ok: false, error: "invalid payload" });
  const { issueId, stateId } = payload as { issueId?: unknown; stateId?: unknown };
  if (!boundedString(issueId, 256) || !boundedString(stateId, 256)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "linear-move", issueId, stateId },
    () => moveIssue(issueId, stateId),
  );
  if (result.ok) return NextResponse.json({ ok: true });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
