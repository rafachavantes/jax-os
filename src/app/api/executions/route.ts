import { NextResponse } from "next/server";
import { collectorResponse } from "@/server/api";
import { getDb } from "@/server/db";
import { queryExecutions } from "@/server/db/workflows";
import { decodeExecutionCursor, parseExecutionSearch } from "@/lib/executions";

export const dynamic = "force-dynamic";

// Durable history only: no project scan, manifest, Git, tmux or process inspection is awaited
// here, so a stalled inspection source can never delay this response or the home cards.
export async function GET(req: Request) {
  const u = new URL(req.url);
  const parsed = parseExecutionSearch(u.searchParams);
  if (!parsed.ok) return NextResponse.json({ ok: false, error: parsed.error });
  let cursor = null;
  if (u.searchParams.has("cursor")) {
    cursor = decodeExecutionCursor(u.searchParams.get("cursor") ?? "");
    if (cursor === null) return NextResponse.json({ ok: false, error: "invalid cursor" });
  }
  return collectorResponse(() => queryExecutions(getDb(), parsed.filters, cursor));
}
