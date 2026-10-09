import { handleMissionCurrent } from "./handler";
export function GET(req: Request) {
  return handleMissionCurrent(req);
}
