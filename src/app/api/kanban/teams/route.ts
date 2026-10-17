import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { getTeams } from "@/server/collectors/linear";
import { readGeneralSettings } from "@/server/settings";

export const dynamic = "force-dynamic";

export async function GET() {
  const settings = readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.linear)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  return collectorResponse(getTeams);
}
