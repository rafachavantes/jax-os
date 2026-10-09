import { collectorResponse } from "@/server/api";
import { getAlerts } from "@/server/collectors/alerts";

export const dynamic = "force-dynamic";

export async function GET() {
  return collectorResponse(getAlerts);
}
