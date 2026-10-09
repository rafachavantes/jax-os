import { NextResponse, type NextRequest } from "next/server";
import { collectorResponse } from "@/server/api";
import { getTokensUsage, RANGES, type RangeKey } from "@/server/collectors/tokens-usage";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const range = req.nextUrl.searchParams.get("range") ?? "7d";
  if (!Object.prototype.hasOwnProperty.call(RANGES, range)) {
    return NextResponse.json({ ok: false, error: "bad range" }, { status: 400 });
  }
  return collectorResponse(() => getTokensUsage(range as RangeKey));
}
