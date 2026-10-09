import { handleAfkGet, handleAfkPost } from "./handler";

// Next validates a route export's first parameter against `Request | NextRequest`, so neither
// handler can be re-exported directly: `handleAfkGet`'s request argument is optional (tests call
// it with no request at all). `tsc --noEmit` does not run that check — only `next build` does.
export function GET(req: Request) {
  return handleAfkGet(req);
}

export function POST(req: Request) {
  return handleAfkPost(req);
}
