import { describe, expect, it } from "vitest";
import { checkOwnDatabase } from "./environment";
import { openDb } from "./index";

describe("checkOwnDatabase (M5)", () => {
  it("selects and reads the current migration version from the own DB", () => {
    const db = openDb(":memory:");
    const result = checkOwnDatabase(db);
    expect(result.ok).toBe(true);
    expect(result.schemaVersion).toBe(15);
    expect(result.error).toBeNull();
    db.close();
  });

  it("a closed/unusable handle is unavailable, never a throw", () => {
    const db = openDb(":memory:");
    db.close();
    expect(checkOwnDatabase(db)).toEqual({ ok: false, schemaVersion: null, error: "database-unavailable" });
  });
});
