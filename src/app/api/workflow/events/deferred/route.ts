import { collectorResponse } from "../../../../../server/api";
import { getDb } from "../../../../../server/db";
import { listDeferred } from "../../../../../server/db/workflows";

export const dynamic = "force-dynamic";

// The rung-7 queue. Returns the message text the classifier needs, which is why this route
// exists rather than reusing the timeline projection — it is the one sanctioned reader of
// message_tail, and it is loopback-only like every other workflow route.
export async function GET() {
  return collectorResponse(() => listDeferred(getDb(), 20));
}
