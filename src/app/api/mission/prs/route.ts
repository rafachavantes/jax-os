import { handlePrsGet } from "./handler";

export const dynamic = "force-dynamic";

// One scoped result per request; anything other than the explicit "all" is the Mine default.
export async function GET(req: Request) {
  const scope = new URL(req.url).searchParams.get("scope") === "all" ? "all" : "mine";
  return handlePrsGet(undefined, scope);
}
