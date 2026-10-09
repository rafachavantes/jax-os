import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";

let testDb: Database.Database;
vi.mock("./db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./db")>();
  return { ...actual, getDb: () => testDb };
});

import { openDb } from "./db";
import { MutationRejected } from "../lib/mutationOutcome";
import { fileEffect, runMutation } from "./mutations";

const ENTRY = { ts: "2026-09-09T12:00:00.000Z", kind: "file-edit", root: "repos", rel: "a.txt" };

type Row = { id: number; ts: string; kind: string; ok: number | null; error: string | null; payload: string };

function allRows(): Row[] {
  return testDb.prepare("SELECT id, ts, kind, ok, error, payload FROM mutations").all() as Row[];
}

describe("runMutation", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
  });
  afterEach(() => {
    testDb.close();
  });

  it("never calls the effect when the initial insert fails", async () => {
    testDb.exec("CREATE TRIGGER fail_ins BEFORE INSERT ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    let calls = 0;
    const result = await runMutation(ENTRY, () => {
      calls++;
      return "ok";
    });
    expect(calls).toBe(0);
    expect(allRows()).toEqual([]);
    expect(result).toEqual({
      ok: false,
      code: "audit-unavailable",
      effect: "not-applied",
      audit: "unavailable",
      error: "audit unavailable; action not performed",
    });
  });

  it("inserts one pending row visible to the effect, then finalizes the same id", async () => {
    let seen: Row[] = [];
    const result = await runMutation(ENTRY, () => {
      seen = allRows();
      return { hash: "ab" };
    });
    expect(result).toEqual({ ok: true, value: { hash: "ab" } });
    expect(seen).toHaveLength(1);
    expect(seen[0].ok).toBeNull();
    expect(JSON.parse(seen[0].payload)).toMatchObject({
      ...ENTRY, outcome: "pending",
    });
    const done = allRows();
    expect(done).toHaveLength(1);
    expect(done[0].id).toBe(seen[0].id);
    expect(done[0].ok).toBe(1);
    expect(JSON.parse(done[0].payload)).toMatchObject({
      ...ENTRY, outcome: "done", ok: true, error: null,
    });
  });

  it("records a typed rejection as failed/not-applied and preserves the safe error", async () => {
    let calls = 0;
    const result = await runMutation(ENTRY, () => {
      calls++;
      throw new MutationRejected("already exists", 200);
    });
    expect(calls).toBe(1);
    expect(result).toMatchObject({
      ok: false,
      code: "mutation-rejected",
      effect: "not-applied",
      audit: "recorded",
      error: "already exists",
      status: 200,
    });
    const rows = allRows();
    expect(rows).toHaveLength(1);
    expect(rows[0].ok).toBe(0);
    expect(JSON.parse(rows[0].payload)).toMatchObject({
      ...ENTRY, outcome: "failed", ok: false, error: "already exists",
    });
  });

  it("records a generic throw as abandoned/unconfirmed without retrying", async () => {
    let calls = 0;
    const result = await runMutation(ENTRY, () => {
      calls++;
      throw new Error("ETIMEDOUT");
    });
    expect(calls).toBe(1);
    expect(result).toEqual({
      ok: false,
      code: "mutation-unconfirmed",
      effect: "unconfirmed",
      audit: "recorded",
      error: "action outcome unconfirmed; check current state before retrying",
    });
    const rows = allRows();
    expect(rows).toHaveLength(1);
    expect(JSON.parse(rows[0].payload)).toMatchObject({
      ...ENTRY, outcome: "abandoned", ok: false,
    });
  });

  it("keeps the pending row when the effect succeeds but UPDATE fails", async () => {
    testDb.exec("CREATE TRIGGER fail_upd BEFORE UPDATE ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    let calls = 0;
    const result = await runMutation(ENTRY, () => {
      calls++;
      return { hash: "ab" };
    });
    expect(calls).toBe(1);
    expect(result).toMatchObject({
      ok: false,
      code: "audit-finalization-failed",
      effect: "applied",
      audit: "pending",
      error: "action applied but audit recording failed; check current state before retrying",
      value: { hash: "ab" },
    });
    const rows = allRows();
    expect(rows).toHaveLength(1);
    expect(rows[0].ok).toBeNull();
    expect(JSON.parse(rows[0].payload).outcome).toBe("pending");
  });

  it("keeps the pending row when a typed rejection cannot be finalized", async () => {
    testDb.exec("CREATE TRIGGER fail_upd BEFORE UPDATE ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const result = await runMutation(ENTRY, () => {
      throw new MutationRejected("changed on disk", 200);
    });
    expect(result).toMatchObject({
      ok: false,
      code: "audit-finalization-failed",
      effect: "not-applied",
      audit: "pending",
      error: "action rejected; audit recording failed",
      status: 200,
    });
    expect(allRows()[0].ok).toBeNull();
  });

  it("keeps the pending row when a generic throw cannot be finalized", async () => {
    testDb.exec("CREATE TRIGGER fail_upd BEFORE UPDATE ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const result = await runMutation(ENTRY, () => {
      throw new Error("EIO");
    });
    expect(result).toMatchObject({
      ok: false,
      code: "audit-finalization-failed",
      effect: "unconfirmed",
      audit: "pending",
      error: "action outcome unconfirmed; audit recording failed",
    });
    expect(allRows()[0].ok).toBeNull();
  });

  it("does not treat a finalization throw as an effect failure", async () => {
    testDb.exec("CREATE TRIGGER fail_upd BEFORE UPDATE ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const result = await runMutation(ENTRY, () => "ok");
    expect(result.ok).toBe(false);
    if (result.ok) throw new Error("expected failure");
    expect(result.code).toBe("audit-finalization-failed");
    expect(result.effect).toBe("applied");
    expect(result.audit).toBe("pending");
  });

  it("writes details from the success callback onto the same row", async () => {
    const result = await runMutation(
      { ts: ENTRY.ts, kind: "rules-apply", app: "claude" },
      () => ({ bytes: 9, backup: ".bak-y" }),
      (value) => ({ bytes: value.bytes, backup: value.backup }),
    );
    expect(result).toEqual({ ok: true, value: { bytes: 9, backup: ".bak-y" } });
    expect(JSON.parse(allRows()[0].payload)).toMatchObject({
      kind: "rules-apply", app: "claude", bytes: 9, backup: ".bak-y", outcome: "done",
    });
  });

  it("merges failureDetails into the row on a rejected AND an unconfirmed effect, details on success", async () => {
    const tails = () => ({ exitCode: 2, stderrTail: "already-finished\n" });
    const rejected = await runMutation(ENTRY, () => { throw new MutationRejected("already-finished"); }, undefined, tails);
    expect(rejected.ok).toBe(false);
    expect(JSON.parse(allRows()[0].payload)).toMatchObject({ outcome: "failed", exitCode: 2, stderrTail: "already-finished\n" });

    const unconfirmed = await runMutation(ENTRY, () => { throw new Error("ETIMEDOUT"); }, undefined, () => ({ exitCode: null, stderrTail: "" }));
    expect(unconfirmed.ok).toBe(false);
    expect(JSON.parse(allRows()[1].payload)).toMatchObject({ outcome: "abandoned", exitCode: null, stderrTail: "" });

    const done = await runMutation(ENTRY, () => "v", (v) => ({ got: v, exitCode: 0 }), () => ({ never: true }));
    expect(done).toEqual({ ok: true, value: "v" });
    expect(JSON.parse(allRows()[2].payload)).toMatchObject({ outcome: "done", got: "v", exitCode: 0 });
    expect(JSON.parse(allRows()[2].payload)).not.toHaveProperty("never");
  });
});

describe("fileEffect", () => {
  it("maps a symlink-source Error to a MutationRejected with the exact message", () => {
    expect(() => fileEffect(() => { throw new Error("symlink source"); })).toThrow(MutationRejected);
    try {
      fileEffect(() => { throw new Error("symlink source"); });
      throw new Error("expected fileEffect to throw");
    } catch (e) {
      expect(e).toBeInstanceOf(MutationRejected);
      expect((e as MutationRejected).message).toBe("symlink source");
    }
  });
});
