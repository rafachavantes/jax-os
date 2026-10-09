import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { getHermesMeta } from "@/server/collectors/hermes-meta";
import { readGeneralSettings } from "@/server/settings";

export const dynamic = "force-dynamic";

export async function GET() {
  // MOA-496: refuses before any execFile/~/.hermes read; `hermes` is never shelled out to.
  const settings = readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.hermesTokens)) {
    return NextResponse.json({ ok: false, error: "disabled" });
  }
  return collectorResponse(getHermesMeta);
}
