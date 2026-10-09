import { NextResponse } from "next/server";
import { updateIssue, type IssuePatch } from "../../../../server/collectors/linear";
import { readGeneralSettings } from "../../../../server/settings";
import { runMutation } from "../../../../server/mutations";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { SMALL_JSON_CAP, COMMENT_CAP, boundedString } from "../../../../server/inputLimits";

const FIELD_KIND: Record<string, string> = {
  priority: "linear-priority",
  assigneeId: "linear-assignee",
  labelIds: "linear-label",
  title: "linear-title",
  description: "linear-description",
};

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
  const { issueId, patch } = payload as { issueId?: unknown; patch?: unknown };
  if (!boundedString(issueId, 256) || patch === null || typeof patch !== "object" || Array.isArray(patch)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const keys = Object.keys(patch);
  const allowed = keys.filter((k) => Object.hasOwn(FIELD_KIND, k));
  if (keys.length !== 1 || allowed.length !== 1) {
    return NextResponse.json({ ok: false, error: "patch must set exactly one allowed field" });
  }
  const field = allowed[0];
  const value = (patch as Record<string, unknown>)[field];
  const valid = field === "priority"
    ? typeof value === "number" && Number.isInteger(value) && value >= 0 && value <= 4
    : field === "assigneeId"
      ? value === null || boundedString(value, 256)
      : field === "title"
        ? boundedString(value, 255)
        : field === "description"
          ? boundedString(value, COMMENT_CAP, true)
          : Array.isArray(value) && value.length <= 100 && value.every(id => boundedString(id, 256));
  if (!valid) return NextResponse.json({ ok: false, error: "invalid payload" });
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: FIELD_KIND[field], issueId, [field]: value },
    () => updateIssue(issueId, { [field]: value } as IssuePatch),
  );
  if (result.ok) return NextResponse.json({ ok: true });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
