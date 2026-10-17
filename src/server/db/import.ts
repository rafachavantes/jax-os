import { existsSync, readFileSync, renameSync } from "node:fs";
import type Database from "better-sqlite3";
import { insertMutation } from "./mutations";

export function parseLogLines(raw: string): { entries: Record<string, unknown>[]; skipped: number } {
  const entries: Record<string, unknown>[] = [];
  let skipped = 0;
  for (const line of raw.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const e: unknown = JSON.parse(line);
      if (typeof e === "object" && e !== null && !Array.isArray(e)) entries.push(e as Record<string, unknown>);
      else skipped++;
    } catch {
      skipped++;
    }
  }
  return { entries, skipped };
}

// One-shot import of the pre-DB JSONL log. Double guard (marker file absent
// AND empty table): insert-then-rename isn't atomic — a crash between them
// must not re-import duplicates on the next boot. Untested fs glue; the
// parser above is the tested surface.
export function runImportIfNeeded(db: Database.Database, logPath: string): void {
  const marker = `${logPath}.imported`;
  if (existsSync(marker) || !existsSync(logPath)) return;
  const count = (db.prepare("SELECT COUNT(*) AS c FROM mutations").get() as { c: number }).c;
  if (count > 0) return;
  const { entries, skipped } = parseLogLines(readFileSync(logPath, "utf8"));
  db.transaction(() => {
    for (const e of entries) insertMutation(db, e);
  })();
  renameSync(logPath, marker);
  if (skipped > 0) console.warn(`[jaxos-db] mutations.log import: ${skipped} corrupt line(s) skipped`);
}
