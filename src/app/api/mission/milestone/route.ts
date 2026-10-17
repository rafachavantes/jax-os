import { handleMissionMilestone } from "./handler";
export function POST(req: Request) {
  return handleMissionMilestone(req);
}
