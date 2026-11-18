import type Database from "better-sqlite3";
import { insertMutation } from "./mutations";
import {
  isPrScope, mergeSavedLayout, parseLayoutEntries,
  type CockpitSnapshot, type LayoutEntry, type PrScope,
} from "../../lib/cockpitLayout";

const DEFAULT_REVISION = 1;

type Row = { revision: number; layout: string | null; prScope: string };

// Absent row = documented defaults (no layout override, Mine, revision 1). A corrupt stored
// layout throws instead of being silently replaced: the route turns that into a visible source
// warning, and a later save still has the conflict check protecting the (unreadable) revision.
export function getCockpitPrefs(db: Database.Database): CockpitSnapshot {
  const row = db
    .prepare("SELECT layout_revision AS revision, layout, pr_scope AS prScope FROM cockpit_prefs WHERE id = 1")
    .get() as Row | undefined;
  if (!row) return { revision: DEFAULT_REVISION, layout: null, prScope: "mine" };
  const layout = row.layout === null ? null : parseLayoutEntries(JSON.parse(row.layout));
  if (row.layout !== null && layout === null) throw new Error("cockpit prefs layout is malformed");
  return { revision: row.revision, layout, prScope: row.prScope === "all" ? "all" : "mine" };
}

function writeSnapshot(db: Database.Database, snapshot: CockpitSnapshot): void {
  db.prepare(
    "INSERT INTO cockpit_prefs (id, layout_revision, layout, pr_scope) VALUES (1, ?, ?, ?) " +
    "ON CONFLICT(id) DO UPDATE SET layout_revision = excluded.layout_revision, layout = excluded.layout, pr_scope = excluded.pr_scope",
  ).run(snapshot.revision, snapshot.layout === null ? null : JSON.stringify(snapshot.layout), snapshot.prScope);
}

export type LayoutWriteResult =
  | { ok: true; snapshot: CockpitSnapshot }
  | { ok: false; conflict: true; snapshot: CockpitSnapshot };

// Optimistic save: the expected revision is checked INSIDE the immediate transaction, the merged
// arrangement is validated in full, then row + audit commit together. A conflict changes nothing.
export function saveCockpitLayout(
  db: Database.Database,
  expectedRevision: number,
  submitted: LayoutEntry[],
  now = new Date(),
): LayoutWriteResult {
  return db.transaction((): LayoutWriteResult => {
    const current = getCockpitPrefs(db);
    if (current.revision !== expectedRevision) return { ok: false, conflict: true, snapshot: current };
    const merged = parseLayoutEntries(mergeSavedLayout(current.layout, submitted));
    if (merged === null) throw new Error("layout exceeds the bounded identity/width rules");
    const next: CockpitSnapshot = { revision: current.revision + 1, layout: merged, prScope: current.prScope };
    writeSnapshot(db, next);
    insertMutation(db, { ts: now.toISOString(), kind: "cockpit-layout-save", revision: next.revision, layout: merged, previous: current });
    return { ok: true, snapshot: next };
  }).immediate();
}

// Reset clears ONLY the layout override; PR scope (and every project_prefs/session preference in
// their own tables) is left untouched.
export function resetCockpitLayout(db: Database.Database, expectedRevision: number, now = new Date()): LayoutWriteResult {
  return db.transaction((): LayoutWriteResult => {
    const current = getCockpitPrefs(db);
    if (current.revision !== expectedRevision) return { ok: false, conflict: true, snapshot: current };
    const next: CockpitSnapshot = { revision: current.revision + 1, layout: null, prScope: current.prScope };
    writeSnapshot(db, next);
    insertMutation(db, { ts: now.toISOString(), kind: "cockpit-layout-reset", revision: next.revision, previous: current });
    return { ok: true, snapshot: next };
  }).immediate();
}

// Audited, no layout revision bump and no layout overwrite.
export function setCockpitPrScope(db: Database.Database, scope: PrScope, now = new Date()): CockpitSnapshot {
  if (!isPrScope(scope)) throw new Error("invalid PR scope");
  return db.transaction((): CockpitSnapshot => {
    const current = getCockpitPrefs(db);
    const next: CockpitSnapshot = { ...current, prScope: scope };
    writeSnapshot(db, next);
    insertMutation(db, { ts: now.toISOString(), kind: "cockpit-pr-scope", scope, previous: current.prScope });
    return next;
  }).immediate();
}
