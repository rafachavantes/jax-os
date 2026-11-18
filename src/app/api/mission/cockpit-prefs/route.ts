import { handleCockpitPrefsGet, handleCockpitPrefsPost } from "./handler";

export function GET() {
  return handleCockpitPrefsGet();
}

export function POST(req: Request) {
  return handleCockpitPrefsPost(req);
}
