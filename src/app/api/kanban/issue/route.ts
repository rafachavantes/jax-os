import { NextResponse } from "next/server";
import { getIssueDetail, IssueNotFound } from "@/server/collectors/linear";
import { readGeneralSettings } from "@/server/settings";
import { boundedString } from "@/server/inputLimits";

export const dynamic = "force-dynamic";

// collectorResponse with one extra distinction: a valid id that Linear has no
// issue for is "not-found" (stable code), everything else stays "unavailable".
export async function GET(req: Request) {
  const settings = readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.linear)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  const id = new URL(req.url).searchParams.get("id") ?? "";
  if (!id) return NextResponse.json({ ok: false, error: "missing id" });
  if (!boundedString(id, 256)) return NextResponse.json({ ok: false, error: "invalid payload" });
  try {
    return NextResponse.json({ ok: true, data: await getIssueDetail(id) });
  } catch (e) {
    if (e instanceof IssueNotFound) {
      return NextResponse.json({ ok: false, error: "issue-not-found", code: "issue-not-found" });
    }
    return NextResponse.json({ ok: false, error: e instanceof Error ? e.message : String(e) });
  }
}
