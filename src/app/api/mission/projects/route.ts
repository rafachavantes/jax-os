import { collectorResponse } from "../../../../server/api";
import { getProjects } from "../../../../server/collectors/projects";

export const dynamic = "force-dynamic";

export async function GET() {
  return collectorResponse(getProjects);
}
