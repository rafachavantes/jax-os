import { after } from "next/server";
import { handleTokensUsage } from "./handler";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  return handleTokensUsage(req, { after: (callback) => after(callback) });
}
