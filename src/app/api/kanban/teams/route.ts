import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { getTeams } from "@/server/collectors/linear";
import { linearEnabled } from "@/server/gates";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!linearEnabled()) return NextResponse.json({ ok: false, error: "disabled" });
  return collectorResponse(getTeams);
}
