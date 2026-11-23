import type Database from "better-sqlite3";
import { NextResponse } from "next/server";
import { getDb } from "@/server/db";
import { getSourceStatuses } from "@/server/db/native-usage";
import { defaultNativeCollectorDeps, runNativeUsagePass } from "@/server/collectors/native-usage";

export type CollectDeps = {
  db: () => Database.Database;
  pass: (db: Database.Database) => Promise<void>;
  now: () => number; // milliseconds
  deadlineMs: number;
};

const LOOPBACK_FORWARD_HEADERS = ["x-forwarded-for", "x-forwarded-host", "origin", "sec-fetch-site"];

function isLoopbackCaller(req: Request): boolean {
  const host = req.headers.get("host");
  if (!host) return false;
  const hostname = host.replace(/:\d+$/, "").toLowerCase();
  if (hostname !== "127.0.0.1" && hostname !== "localhost") return false;
  return !LOOPBACK_FORWARD_HEADERS.some((header) => req.headers.has(header));
}

// One bounded catch-up run for the systemd timer: POST only, loopback only. Passes repeat
// until no source is indexing or the deadline passes; a failed pass just stops this tick
// (the source statuses already carry its safe warning) and the next tick resumes.
export async function handleUsageCollect(req: Request, overrides: Partial<CollectDeps> = {}): Promise<Response> {
  const deps: CollectDeps = {
    db: getDb,
    pass: async (db) => runNativeUsagePass(await defaultNativeCollectorDeps(db)),
    now: Date.now,
    deadlineMs: 30_000,
    ...overrides,
  };
  if (!isLoopbackCaller(req)) {
    return NextResponse.json({ ok: false, error: "loopback only" }, { status: 403 });
  }
  let db: Database.Database;
  try {
    db = deps.db();
  } catch {
    return NextResponse.json({ ok: false, error: "usage unavailable" });
  }

  const start = deps.now();
  let passes = 0;
  let sources: { source: string; state: string; lastRefresh: number | null }[] = [];
  let caughtUp = false;
  try {
    for (;;) {
      await deps.pass(db);
      passes += 1;
      sources = getSourceStatuses(db).map(({ source, state, lastRefresh }) => ({ source, state, lastRefresh }));
      if (!sources.some((source) => source.state === "indexing")) {
        caughtUp = true;
        break;
      }
      if (deps.now() - start >= deps.deadlineMs) break;
    }
  } catch {
    /* a failed pass is already a safe source warning; report partial progress */
  }
  return NextResponse.json({
    ok: true,
    data: { passes, caughtUp, sources },
  });
}
