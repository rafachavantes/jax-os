import { handleMissionStatus } from "./handler";
export function POST(req: Request) {
  return handleMissionStatus(req);
}
