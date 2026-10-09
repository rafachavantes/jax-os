import { handleMissionFinish } from "./handler";
export function POST(req: Request) {
  return handleMissionFinish(req);
}
