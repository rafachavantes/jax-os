import { describe, expect, it } from "vitest";
import Database from "better-sqlite3";
import { MIGRATIONS, migrate, openDb } from "./index";
import { getCockpitPrefs, resetCockpitLayout, saveCockpitLayout, setCockpitPrScope } from "./cockpit-prefs";
import type { LayoutEntry } from "../../lib/cockpitLayout";

const A: LayoutEntry = { project: "Alpha", width: "third" };
const B: LayoutEntry = { project: "Beta", width: "half" };
const C: LayoutEntry = { project: "Gamma", width: "full" };

function mutationKinds(db: Database.Database): string[] {
  return (db.prepare("SELECT kind FROM mutations ORDER BY id").all() as { kind: string }[]).map((r) => r.kind);
}

describe("cockpit prefs migration (plan Task 1)", () => {
  it("migrates a previous-version db to 13 with an empty singleton and defaults", () => {
    const db = new Database(":memory:");
    for (const sql of MIGRATIONS.slice(0, 12)) db.exec(sql);
    db.pragma("user_version = 12");
    migrate(db);
    expect(db.pragma("user_version", { simple: true })).toBe(13);
    const cols = (db.prepare("PRAGMA table_info(cockpit_prefs)").all() as { name: string }[]).map((c) => c.name);
    expect(cols).toEqual(["id", "layout_revision", "layout", "pr_scope"]);
    expect(db.prepare("SELECT COUNT(*) AS n FROM cockpit_prefs").get()).toEqual({ n: 0 });
    expect(getCockpitPrefs(db)).toEqual({ revision: 1, layout: null, prScope: "mine" });
    db.close();
  });
});

describe("cockpit prefs storage (plan Task 1)", () => {
  it("save merges absent identities, bumps only the revision and audits", () => {
    const db = openDb(":memory:");
    const first = saveCockpitLayout(db, 1, [A, B], new Date("2026-11-18T12:00:00Z"));
    expect(first.ok && first.snapshot).toMatchObject({ revision: 2, layout: [A, B], prScope: "mine" });
    const second = saveCockpitLayout(db, 2, [C], new Date("2026-11-18T12:01:00Z"));
    expect(second.ok && second.snapshot.layout).toEqual([C, A, B]);
    expect(getCockpitPrefs(db).layout).toEqual([C, A, B]);
    expect(mutationKinds(db)).toEqual(["cockpit-layout-save", "cockpit-layout-save"]);
    db.close();
  });

  it("a stale revision is a typed conflict that writes nothing", () => {
    const db = openDb(":memory:");
    saveCockpitLayout(db, 1, [A]);
    const stale = saveCockpitLayout(db, 1, [B]);
    expect(stale).toEqual({ ok: false, conflict: true, snapshot: { revision: 2, layout: [A], prScope: "mine" } });
    expect(getCockpitPrefs(db)).toEqual({ revision: 2, layout: [A], prScope: "mine" });
    expect(mutationKinds(db)).toEqual(["cockpit-layout-save"]);
    db.close();
  });

  it("two browsers: A's stale save after B's save conflicts with B's snapshot", () => {
    const db = openDb(":memory:");
    const a = getCockpitPrefs(db);
    const b = getCockpitPrefs(db);
    saveCockpitLayout(db, b.revision, [A]);
    const aSave = saveCockpitLayout(db, a.revision, [B]);
    expect(aSave.ok).toBe(false);
    expect(aSave.ok === false && aSave.snapshot.layout).toEqual([A]);
    // Cancel/Reload baselines read the latest snapshot, never the stale fetch.
    expect(getCockpitPrefs(db)).toEqual({ revision: 2, layout: [A], prScope: "mine" });
    db.close();
  });

  it("reset clears only the layout override", () => {
    const db = openDb(":memory:");
    setCockpitPrScope(db, "all");
    saveCockpitLayout(db, 1, [A, B]);
    db.prepare("INSERT INTO project_prefs (project, pinned, hidden_at) VALUES ('Alpha', 1, NULL)").run();
    const reset = resetCockpitLayout(db, 2);
    expect(reset.ok && reset.snapshot).toEqual({ revision: 3, layout: null, prScope: "all" });
    expect(db.prepare("SELECT project, pinned FROM project_prefs").all()).toEqual([{ project: "Alpha", pinned: 1 }]);
    expect(mutationKinds(db)).toEqual(["cockpit-pr-scope", "cockpit-layout-save", "cockpit-layout-reset"]);
    db.close();
  });

  it("pr scope writes are audited without bumping the revision or touching layout", () => {
    const db = openDb(":memory:");
    saveCockpitLayout(db, 1, [A]);
    const next = setCockpitPrScope(db, "all");
    expect(next).toEqual({ revision: 2, layout: [A], prScope: "all" });
    expect(getCockpitPrefs(db)).toEqual(next);
    expect(mutationKinds(db)).toEqual(["cockpit-layout-save", "cockpit-pr-scope"]);
    db.close();
  });

  it("an audit failure rolls the whole save back", () => {
    const db = openDb(":memory:");
    saveCockpitLayout(db, 1, [A]);
    db.exec("DROP TABLE mutations");
    expect(() => saveCockpitLayout(db, 2, [B])).toThrow();
    expect(getCockpitPrefs(db)).toEqual({ revision: 2, layout: [A], prScope: "mine" });
    db.close();
  });

  it("rejects overflow past 500 identities without evicting saved choices", () => {
    const db = openDb(":memory:");
    const full = Array.from({ length: 500 }, (_, i) => ({ project: `p${i}`, width: "third" as const }));
    saveCockpitLayout(db, 1, full);
    expect(() => saveCockpitLayout(db, 2, [{ project: "one-more", width: "third" }])).toThrow();
    expect(getCockpitPrefs(db).layout).toHaveLength(500);
    db.close();
  });

  it("a corrupt stored layout surfaces as a source error, not a quiet default", () => {
    const db = openDb(":memory:");
    db.prepare("INSERT INTO cockpit_prefs (id, layout_revision, layout, pr_scope) VALUES (1, 4, '{', 'mine')").run();
    expect(() => getCockpitPrefs(db)).toThrow();
    db.close();
  });
});
