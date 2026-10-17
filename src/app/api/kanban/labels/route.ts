import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { getLabels } from "@/server/collectors/linear";
import { linearEnabled } from "@/server/gates";
import { boundedString } from "@/server/inputLimits";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  if (!linearEnabled()) return NextResponse.json({ ok: false, error: "disabled" });
  const team = new URL(req.url).searchParams.get("team") ?? "";
  if (!team) return NextResponse.json({ ok: false, error: "missing team" });
  if (!boundedString(team, 256)) return NextResponse.json({ ok: false, error: "invalid payload" });
  return collectorResponse(() => getLabels(team));
}
