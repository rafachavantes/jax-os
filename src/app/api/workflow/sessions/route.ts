import { NextResponse } from "next/server";
import { LIMITS } from "@/lib/workflow";
import { collectorResponse, readJsonCapped, requireSameOrigin } from "@/server/api";
import { parseSessionBody } from "@/server/collectors/workflow-events";
import { getDb } from "@/server/db";
import { upsertSession } from "@/server/db/workflows";

// Lead/adhoc-session registry: (pane, tmux_incarnation) → project/session/role
// (spec §3.1b, §4.2). Format validation only; the answer route (Phase C)
// revalidates against live tmux before every injection. The A′-window legacy
// {project, tmux_target} body is no longer accepted (Task B6, cutover contract
// item 3) — it now falls through to the unknown-key rejection.
export async function POST(req: Request) {
  const guard = requireSameOrigin(req);
  if (guard) return guard;
  const read = await readJsonCapped(req, LIMITS.requestBytes);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const parsed = parseSessionBody(read.value);
  if (!parsed.ok) return NextResponse.json({ ok: false, error: `invalid session: ${parsed.error}` });
  return collectorResponse(() => {
    upsertSession(getDb(), parsed.session);
    return parsed.session;
  });
}
