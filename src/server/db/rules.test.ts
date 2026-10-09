import { describe, expect, it, vi } from "vitest";
import { openDb } from "./index";
import { computeSeed, getRuleDocs, seedRuleDocsIfNeeded, setRuleDoc } from "./rules";
import type { RuleDocs } from "../../lib/rules";

const STARTER: RuleDocs = {
  global: "starter canonical\n",
  claude: "starter claude\n",
  codex: "starter codex\n",
  opencode: "starter opencode\n",
};
const IMPORT_HEAD = "## Your existing rules (imported)";

function memDb() {
  return openDb(":memory:");
}

const noExisting = () => ({ starter: STARTER, existing: { claudeRaw: null, codexRaw: null } });

describe("computeSeed (D2)", () => {
  it("no existing files: global is the starter canonical alone, no imported block", () => {
    expect(computeSeed(STARTER, { claudeRaw: null, codexRaw: null })).toEqual(STARTER);
  });

  it("claudeRaw present: imported block built from it, codexRaw ignored", () => {
    const seed = computeSeed(STARTER, { claudeRaw: "my old rules\n", codexRaw: "unrelated\n" });
    expect(seed.global).toBe(`starter canonical\n\n\n${IMPORT_HEAD}\n\nmy old rules\n`);
    expect(seed.global).not.toContain("unrelated");
  });

  it("claudeRaw null + codexRaw present: imported block built from codexRaw", () => {
    const seed = computeSeed(STARTER, { claudeRaw: null, codexRaw: "my codex rules\n" });
    expect(seed.global).toBe(`starter canonical\n\n\n${IMPORT_HEAD}\n\nmy codex rules\n`);
  });

  it("exceptions always equal the starter exception files regardless of existing content", () => {
    const seed = computeSeed(STARTER, { claudeRaw: "my old rules\n", codexRaw: "unrelated\n" });
    expect({ claude: seed.claude, codex: seed.codex, opencode: seed.opencode }).toEqual({
      claude: STARTER.claude,
      codex: STARTER.codex,
      opencode: STARTER.opencode,
    });
  });

  it("keeps \"\" as the no-exception sentinel instead of norm-ing it to \"\\n\"", () => {
    const emptyExceptions: RuleDocs = { ...STARTER, claude: "", codex: "", opencode: "" };
    expect(computeSeed(emptyExceptions, { claudeRaw: null, codexRaw: null })).toEqual(emptyExceptions);
  });

  it("never returns null", () => {
    const seed = computeSeed(STARTER, { claudeRaw: null, codexRaw: null });
    expect(seed).toBeTruthy();
    expect(seed).toEqual(STARTER);
  });

  it("normalizes the imported text like every canonical write (no trailing newline / repeated trailing newlines)", () => {
    const noTrailing = computeSeed(STARTER, { claudeRaw: "my old rules", codexRaw: null });
    const repeated = computeSeed(STARTER, { claudeRaw: "my old rules\n\n\n", codexRaw: null });
    expect(noTrailing.global).toBe(`starter canonical\n\n\n${IMPORT_HEAD}\n\nmy old rules\n`);
    expect(repeated.global).toBe(noTrailing.global);
  });
});

describe("seedRuleDocsIfNeeded", () => {
  it("fills an empty table in one shot from the new seed shape", () => {
    const db = memDb();
    seedRuleDocsIfNeeded(db, noExisting);
    expect(getRuleDocs(db)).toEqual(STARTER);
  });

  it("never overwrites a nonempty table, even when called again", () => {
    const db = memDb();
    seedRuleDocsIfNeeded(db, noExisting);
    setRuleDoc(db, "global", "hand-edited\n", { ts: "t", kind: "rules-edit", ok: true });
    seedRuleDocsIfNeeded(db, noExisting);
    expect(getRuleDocs(db).global).toBe("hand-edited\n");
  });
});

describe("setRuleDoc (§7 test 8)", () => {
  it("updates the slot and inserts the mutation in the same transaction", () => {
    const db = memDb();
    seedRuleDocsIfNeeded(db, noExisting);
    setRuleDoc(db, "global", "new content\n", {
      ts: "t",
      kind: "rules-edit",
      ok: true,
      slot: "global",
      previous: STARTER.global,
      content: "new content\n",
    });
    expect(getRuleDocs(db).global).toBe("new content\n");
    const row = db.prepare("SELECT payload FROM mutations WHERE kind = 'rules-edit'").get() as { payload: string };
    expect(row).toBeTruthy();
    // spec §7 test 8: the audited payload carries {slot, previous, content}
    expect(JSON.parse(row.payload)).toMatchObject({ slot: "global", previous: STARTER.global, content: "new content\n" });
  });

  it("rolls back the doc update when the mutation insert fails (injected failure, :memory: db — §7 test 8)", () => {
    const db = memDb();
    seedRuleDocsIfNeeded(db, noExisting);
    const realPrepare = db.prepare.bind(db);
    vi.spyOn(db, "prepare").mockImplementation((sql: string) => {
      if (sql.startsWith("INSERT INTO mutations")) throw new Error("injected mutation insert failure");
      return realPrepare(sql);
    });
    try {
      expect(() =>
        setRuleDoc(db, "global", "should not stick\n", { ts: "t", kind: "rules-edit" }),
      ).toThrow("injected mutation insert failure");
    } finally {
      vi.restoreAllMocks(); // finally: a failed assertion must not leak the prepare mock into later tests
    }
    expect(getRuleDocs(db).global).toBe(STARTER.global); // rolled back — still the seeded value, not the failed edit
    const count = db.prepare("SELECT COUNT(*) AS c FROM mutations WHERE kind = 'rules-edit'").get() as { c: number };
    expect(count.c).toBe(0); // no mutation row survived the rollback either
  });

  it("throws when the slot row doesn't exist yet (seed never ran) instead of silently no-op-ing", () => {
    const db = memDb(); // NOT seeded
    expect(() => setRuleDoc(db, "global", "x\n", { ts: "t", kind: "rules-edit" })).toThrow(/unknown slot/);
  });
});

describe("getRuleDocs fails closed on an unseeded table (spec §3 fail-closed)", () => {
  it("throws instead of defaulting missing slots to empty documents", () => {
    const db = memDb(); // NOT seeded
    expect(() => getRuleDocs(db)).toThrow(/not initialized/);
  });
});
