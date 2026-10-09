import { handlePanesTailGet } from "./handler";

export const dynamic = "force-dynamic";

export async function GET() {
  return handlePanesTailGet();
}
