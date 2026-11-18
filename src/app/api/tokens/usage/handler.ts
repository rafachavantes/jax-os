import type Database from "better-sqlite3";
import { NextResponse } from "next/server";
import { getDb } from "@/server/db";
import { defaultNativeCollectorDeps, runNativeUsagePass } from "@/server/collectors/native-usage";
import { getTokensUsage, isRange, type UsageParams } from "@/server/collectors/tokens-usage";

export type UsageRouteDeps = {
  db: () => Database.Database;
  after: (callback: () => void | Promise<void>) => void;
  collect: (db: Database.Database) => Promise<void>;
};

const MAX_FILTER_LENGTH = 256;

function validFilter(value: string | null): string | null | undefined {
  if (value === null || value === "") return null;
  if (value.length > MAX_FILTER_LENGTH || /[\u0000-\u001f\u007f]/.test(value)) return undefined;
  return value;
}

// Reads the persisted snapshot synchronously, then registers one shared bounded
// collection pass after the response. Native work never runs on the response path.
export async function handleTokensUsage(
  req: Request,
  overrides: Partial<UsageRouteDeps> = {},
): Promise<Response> {
  const deps: UsageRouteDeps = {
    db: getDb,
    after: () => undefined,
    collect: async (db) => runNativeUsagePass(defaultNativeCollectorDeps(db)),
    ...overrides,
  };
  const url = new URL(req.url);
  const rawRange = url.searchParams.get("range") ?? "7d";
  if (!isRange(rawRange)) {
    return NextResponse.json({ ok: false, error: "bad range" }, { status: 400 });
  }
  const agent = validFilter(url.searchParams.get("agent"));
  if (agent === undefined) {
    return NextResponse.json({ ok: false, error: "bad agent filter" }, { status: 400 });
  }
  const model = validFilter(url.searchParams.get("model"));
  if (model === undefined) {
    return NextResponse.json({ ok: false, error: "bad model filter" }, { status: 400 });
  }

  let db: Database.Database;
  try {
    db = deps.db();
  } catch {
    return NextResponse.json({ ok: false, error: "usage unavailable" });
  }
  const params: UsageParams = { range: rawRange, agent, model };
  try {
    const data = getTokensUsage(db, params);
    deps.after(async () => {
      try {
        await deps.collect(db);
      } catch {
        /* every failure is already a safe source status */
      }
    });
    return NextResponse.json({ ok: true, data });
  } catch {
    return NextResponse.json({ ok: false, error: "usage unavailable" });
  }
}
