import { handleDispatchPost } from "./handler";
export function POST(req: Request) {
  return handleDispatchPost(req);
}
