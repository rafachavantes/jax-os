import { handleAnswerPost } from "./handler";

export function POST(req: Request) {
  return handleAnswerPost(req);
}
