import { handleCancelPost } from "./handler";
// Round-1 F1: a one-argument wrapper — a second parameter would be read as Next's route context.
export function POST(req: Request) {
  return handleCancelPost(req);
}
