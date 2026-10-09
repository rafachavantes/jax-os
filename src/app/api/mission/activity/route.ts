import { collectorResponse } from "@/server/api";
import { getActivity } from "@/server/collectors/activity";

export const dynamic = "force-dynamic";

export async function GET() {
  return collectorResponse(getActivity);
}
