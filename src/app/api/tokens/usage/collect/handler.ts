import type Database from "better-sqlite3";
import { NextResponse } from "next/server";
import { getDb } from "../../../../../server/db";
import { getSourceStatuses } from "../../../../../server/db/native-usage";
import { defaultNativeCollectorDeps, runNativeUsagePass } from "../../../../../server/collectors/native-usage";

export type CollectDeps = {
  db: () => Database.Database;
  pass: (db: Database.Database) => Promise<void>;
  now: () => number; // milliseconds
  deadlineMs: number;
};

// One bounded catch-up run for the systemd timer: POST only. No caller guard: collection is idempotent internal state and the app binds 127.0.0.1
// (Next adds x-forwarded-* headers itself, so header checks would only break the timer). Passes repeat
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
