import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { getBoard } from "@/server/collectors/linear";
import { readGeneralSettings } from "@/server/settings";
import { boundedString } from "@/server/inputLimits";

export const dynamic = "force-dynamic";

// MOA-496: gated FIRST, before any input validation or credential read — a disabled integration
// never touches LINEAR_API_KEY (decision 6: disabled is its own distinguishable answer).
export async function GET(req: Request) {
  const settings = readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.linear)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  const team = new URL(req.url).searchParams.get("team") ?? "";
  if (!team) return NextResponse.json({ ok: false, error: "missing team" });
  if (!boundedString(team, 256)) return NextResponse.json({ ok: false, error: "invalid payload" });
  return collectorResponse(() => getBoard(team));
}
