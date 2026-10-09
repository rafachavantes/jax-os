import { handleSettingsGet, handleSettingsPost } from "./handler";
export function GET(req: Request) {
  return handleSettingsGet(req);
}
export function POST(req: Request) {
  return handleSettingsPost(req);
}
