import type Database from "better-sqlite3";
import { insertMutation } from "./mutations";
import { sourceRevision } from "../collectors/rules";
import { norm, type RuleDocs } from "../../lib/rules";

export type RuleSlot = keyof RuleDocs;

export type SetRuleDocOutcome = { ok: true; revision: string } | { ok: false; reason: "stale" };

// Fails CLOSED when any slot row is missing (seed never ran): defaulting a
// missing row to "" would let the apply route compose "\n" and overwrite a
// real global rules file with a one-LF document. The throw bubbles to the
// GET route's collectorResponse ({ok:false} visible warning) and to the POST
// routes' 500 handler — no destructive path stays open on a failed seed.
export function getRuleDocs(db: Database.Database): RuleDocs {
  const rows = db.prepare("SELECT slot, content FROM rule_docs").all() as { slot: RuleSlot; content: string }[];
  const map = Object.fromEntries(rows.map((r) => [r.slot, r.content])) as Partial<RuleDocs>;
  const missing = (["global", "claude", "codex", "opencode"] as const).filter((slot) => map[slot] === undefined);
  if (missing.length > 0) throw new Error(`rule_docs not initialized (seed never ran — missing: ${missing.join(", ")})`);
  return map as RuleDocs;
}

// Transactional edit (spec §5): the doc UPDATE and the mutation INSERT
// commit or roll back together — an edit can never exist unlogged (spec §7
// test 8). Throws (and rolls back) when the slot row doesn't exist, which
// only happens if the seeder never ran — a real bug, not a normal path.
//
// Task 1: when expectedRevision is given, the CURRENT slot is loaded inside
// this same transaction and its hash compared before the update. A stale
// revision is a typed refusal (no throw), so the route can answer 409 while
// the DB stays untouched. The revision check and the write share one
// transaction, so a second writer cannot slip between compare and update.
export function setRuleDoc(
  db: Database.Database,
  slot: RuleSlot,
  content: string,
  mutation: Record<string, unknown>,
  expectedRevision?: string,
): SetRuleDocOutcome {
  return db.transaction((): SetRuleDocOutcome => {
    const current = db.prepare("SELECT content FROM rule_docs WHERE slot = ?").get(slot) as { content: string } | undefined;
    if (!current) throw new Error(`rule_docs: unknown slot "${slot}" (seed never ran)`);
    if (expectedRevision !== undefined && sourceRevision(current.content) !== expectedRevision) {
      return { ok: false, reason: "stale" };
    }
    const result = db
      .prepare("UPDATE rule_docs SET content = ?, updated_at = ? WHERE slot = ?")
      .run(content, new Date().toISOString(), slot);
    if (result.changes === 0) throw new Error(`rule_docs: unknown slot "${slot}" (seed never ran)`);
    insertMutation(db, mutation);
    return { ok: true, revision: sourceRevision(content) };
  })();
}

// Pure seed computation (spec D2). `global` is the starter canonical,
// optionally followed by an "imported" block holding whatever the installer's
// own CLAUDE.md/AGENTS.md already contained (claudeRaw wins when both exist —
// deterministic, and Claude is jax-os's own primary harness). The exception
// slots are always the starter exception files, never derived from the
// installer's files. Never fails closed: the starter ships with the app, so
// there is no mismatch left to detect. Exported for direct unit testing.
export function computeSeed(
  starter: RuleDocs,
  existing: { claudeRaw: string | null; codexRaw: string | null },
): RuleDocs {
  const global = norm(starter.global);
  const imported = existing.claudeRaw !== null ? existing.claudeRaw : existing.codexRaw;
  const withImport = imported !== null ? `${global}\n\n## Your existing rules (imported)\n\n${norm(imported)}` : global;
  // "" stays the no-exception sentinel (spec §3, same rule as the exception
  // route): norm("") would produce "\n", which is NOT the sentinel.
  const exception = (t: string) => (t === "" ? "" : norm(t));
  return {
    global: withImport,
    claude: exception(starter.claude),
    codex: exception(starter.codex),
    opencode: exception(starter.opencode),
  };
}

// The real filesystem reader (readSeedSources) lives in
// src/server/collectors/rules.ts — hard rule 2: ALL external system access
// goes through collectors. This DB module only ever receives the strings.
export type SeedSources = {
  starter: RuleDocs;
  existing: { claudeRaw: string | null; codexRaw: string | null };
};

// One-shot seeder (SP1 import-hook precedent, src/server/db/import.ts):
// fills rule_docs ONLY when the table is empty, in ONE transaction; never
// overwrites a nonempty table. The starter files are bundled with the app, so
// the only remaining "leave the table empty" case is a genuinely unexpected
// read error, which still isn't seeded through; the next boot retries
// automatically since table emptiness is the only guard, no marker file.
export function seedRuleDocsIfNeeded(db: Database.Database, readSources: () => SeedSources): void {
  const count = (db.prepare("SELECT COUNT(*) AS c FROM rule_docs").get() as { c: number }).c;
  if (count > 0) return;
  const { starter, existing } = readSources();
  const seed = computeSeed(starter, existing);
  const now = new Date().toISOString();
  db.transaction(() => {
    for (const slot of ["global", "claude", "codex", "opencode"] as const) {
      db.prepare("INSERT INTO rule_docs (slot, content, updated_at) VALUES (?, ?, ?)").run(slot, seed[slot], now);
    }
  })();
}
