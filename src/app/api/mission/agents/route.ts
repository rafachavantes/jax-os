import { collectorResponse } from "@/server/api";
import { getAgents } from "@/server/collectors/tmux";

export const dynamic = "force-dynamic";

export async function GET() {
  return collectorResponse(getAgents);
}
