import { handlePrefsPost } from "./handler";
export function POST(req: Request) {
  return handlePrefsPost(req);
}
