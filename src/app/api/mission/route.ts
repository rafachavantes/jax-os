import { handleMissionStart } from "./handler";
export function POST(req: Request) {
  return handleMissionStart(req);
}
