import { NextResponse } from "next/server";
import { createIssue, type IssueCreateInput } from "../../../../server/collectors/linear";
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
  const { teamId, title, description, projectId, stateId, priority, assigneeId, labelIds } = payload as Record<string, unknown>;
  if (!boundedString(teamId, 256) || !boundedString(title, 255)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  if (description !== undefined && !boundedString(description, COMMENT_CAP, true)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  if (projectId !== undefined && projectId !== null && !boundedString(projectId, 256)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  if (stateId !== undefined && !boundedString(stateId, 256)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  if (priority !== undefined && !(typeof priority === "number" && Number.isInteger(priority) && priority >= 0 && priority <= 4)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  if (assigneeId !== undefined && assigneeId !== null && !boundedString(assigneeId, 256)) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  if (labelIds !== undefined && !(Array.isArray(labelIds) && labelIds.length <= 100 && labelIds.every((id) => boundedString(id, 256)))) {
    return NextResponse.json({ ok: false, error: "invalid payload" });
  }
  const input: IssueCreateInput = {
    teamId, title,
    ...(description !== undefined ? { description } : {}),
    ...(projectId !== undefined ? { projectId } : {}),
    ...(stateId !== undefined ? { stateId } : {}),
    ...(priority !== undefined ? { priority } : {}),
    ...(assigneeId !== undefined ? { assigneeId } : {}),
    ...(labelIds !== undefined ? { labelIds } : {}),
  } as IssueCreateInput;
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "linear-issue-create", teamId, title },
    () => createIssue(input),
  );
  if (result.ok) return NextResponse.json({ ok: true, id: result.value.id, identifier: result.value.identifier });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
