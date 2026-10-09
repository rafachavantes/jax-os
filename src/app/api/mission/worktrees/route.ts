import { handleWorktreesGet } from "./handler";

export const dynamic = "force-dynamic";

export async function GET() {
  return handleWorktreesGet();
}
