// M5 Task 1: the ONLY own-DB read the Environment diagnostics is allowed. Read-only, no
// migrations, no foreign DB: the caller passes the app's own better-sqlite3 handle.
import type Database from "better-sqlite3";

export type OwnDbCheck = {
  ok: boolean;
  schemaVersion: number | null;
  error: "database-unavailable" | null;
};

export function checkOwnDatabase(db: Database.Database): OwnDbCheck {
  try {
    db.prepare("SELECT 1").get();
    const version = db.pragma("user_version", { simple: true });
    return { ok: true, schemaVersion: typeof version === "number" ? version : null, error: null };
  } catch {
    return { ok: false, schemaVersion: null, error: "database-unavailable" };
  }
}
