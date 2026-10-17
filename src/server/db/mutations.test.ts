import { describe, expect, it } from "vitest";
import { openDb } from "./index";
import {
  decodeMutationCursor, encodeMutationCursor, extractMutation, finishMutation, insertMutation, queryMutations,
} from "./mutations";

const APPROVAL = { ts: "2026-07-06T22:23:22.226Z", kind: "approval", project: "jax-os", gate: "build", ok: true };
const TAKE = { ts: "2026-07-07T00:36:27.391Z", kind: "take-control", session: "jax-scratch" };
const EDIT_FAIL = { ts: "2026-07-11T12:48:36.570Z", kind: "file-edit", root: "repos", rel: "x.md", ok: false, error: "changed on disk" };

describe("extractMutation", () => {
  it("maps ok to 1/0/null and defaults kind", () => {
    expect(extractMutation(APPROVAL).ok).toBe(1);
    expect(extractMutation(EDIT_FAIL)).toMatchObject({ ok: 0, error: "changed on disk", kind: "file-edit" });
    expect(extractMutation(TAKE).ok).toBeNull();
    expect(extractMutation({ ts: "2026-01-01T00:00:00.000Z" }).kind).toBe("unknown");
  });
  it("keeps the full entry as JSON payload", () => {
    expect(JSON.parse(extractMutation(TAKE).payload)).toEqual(TAKE);
  });
});

describe("insert + query roundtrip", () => {
  function seeded() {
    const db = openDb(":memory:");
    for (const e of [APPROVAL, TAKE, EDIT_FAIL]) insertMutation(db, e);
    return db;
  }

  it("returns newest first with parsed payload, kinds, total", () => {
    const db = seeded();
    const page = queryMutations(db, { limit: 50 });
    expect(page.total).toBe(3);
    expect(page.rows.map((r) => r.kind)).toEqual(["file-edit", "take-control", "approval"]);
    expect(page.rows[0]).toMatchObject({ ok: false, error: "changed on disk" });
    expect(page.rows[1].ok).toBeNull();
    expect(page.rows[2].payload).toEqual(APPROVAL);
    expect(page.kinds).toEqual(["approval", "file-edit", "take-control"]);
    db.close();
  });

  it("filters by kind and date range; paginates", () => {
    const db = seeded();
    expect(queryMutations(db, { kind: "approval", limit: 50 }).total).toBe(1);
    expect(queryMutations(db, { from: "2026-07-07", limit: 50 }).total).toBe(2);
    expect(queryMutations(db, { from: "2026-07-07", to: "2026-07-08", limit: 50 }).total).toBe(1);
    const p1 = queryMutations(db, { limit: 1 });
    expect(p1.rows[0].kind).toBe("file-edit");
    expect(p1.nextCursor).toBe(encodeMutationCursor(p1.rows[0].ts, p1.rows[0].id));
    const p2 = queryMutations(db, { limit: 1, cursor: decodeMutationCursor(p1.nextCursor!)! });
    expect(p2.rows).toHaveLength(1);
    expect(p2.rows[0].kind).toBe("take-control");
    expect(p2.total).toBe(3);
    db.close();
  });

  it("returns a positive id and finishMutation updates ok, error, and outcome", () => {
    const db = openDb(":memory:");
    const id = insertMutation(db, {
      ts: "2026-08-18T12:00:00.000Z", kind: "workflow-answer", project: "p1",
      target: "jax-p1-lead:1.1", question_event_id: 7, tool_use_id: "t1", answer_count: 1, outcome: "pending",
    });
    expect(id).toBeGreaterThan(0);
    finishMutation(db, id, true, null, "done");
    const row = db.prepare("SELECT ok, error, payload FROM mutations WHERE id = ?").get(id) as {
      ok: number | null; error: string | null; payload: string;
    };
    expect(row.ok).toBe(1);
    expect(row.error).toBeNull();
    expect(JSON.parse(row.payload)).toMatchObject({
      kind: "workflow-answer", project: "p1", target: "jax-p1-lead:1.1",
      question_event_id: 7, tool_use_id: "t1", answer_count: 1,
      outcome: "done", ok: true, error: null,
    });
    db.close();
  });

  it("merges optional details before ok/error/outcome", () => {
    const db = openDb(":memory:");
    const id = insertMutation(db, {
      ts: "2026-09-09T00:00:00.000Z", kind: "rules-apply", app: "claude", outcome: "pending",
    });
    finishMutation(db, id, true, null, "done", { bytes: 12, backup: ".bak-x" });
    const row = db.prepare("SELECT payload FROM mutations WHERE id = ?").get(id) as { payload: string };
    expect(JSON.parse(row.payload)).toMatchObject({
      kind: "rules-apply", app: "claude", bytes: 12, backup: ".bak-x",
      outcome: "done", ok: true, error: null,
    });
    db.close();
  });
});

describe("mutation cursor codec", () => {
  it("roundtrips a stored timestamp without reformatting", () => {
    const ts = "2026-07-06T22:23:22.226Z";
    const raw = encodeMutationCursor(ts, 12);
    expect(decodeMutationCursor(raw)).toEqual({ ts, id: 12 });
  });

  it("rejects oversized, noncanonical, wrong-shape, and illegal tuple values", () => {
    expect(decodeMutationCursor("a".repeat(1025))).toBeNull();
    const ok = encodeMutationCursor("2026-01-01T00:00:00.000Z", 1);
    expect(decodeMutationCursor(ok + "=")).toBeNull();
    expect(decodeMutationCursor(Buffer.from(JSON.stringify({ ts: "t", id: 1 }), "utf8").toString("base64url"))).toBeNull();
    expect(decodeMutationCursor(Buffer.from(JSON.stringify(["", 1]), "utf8").toString("base64url"))).toBeNull();
    expect(decodeMutationCursor(Buffer.from(JSON.stringify(["a\nb", 1]), "utf8").toString("base64url"))).toBeNull();
    expect(decodeMutationCursor(Buffer.from(JSON.stringify(["é".repeat(129), 1]), "utf8").toString("base64url"))).toBeNull();
    expect(decodeMutationCursor(Buffer.from(JSON.stringify(["2026-01-01T00:00:00.000Z", 0]), "utf8").toString("base64url"))).toBeNull();
    expect(decodeMutationCursor(Buffer.from(JSON.stringify(["2026-01-01T00:00:00.000Z", 1.5]), "utf8").toString("base64url"))).toBeNull();
  });
});

describe("queryMutations cursor pagination", () => {
  it("orders equal timestamps by id and keeps older boundary rows after a newer insert", () => {
    const db = openDb(":memory:");
    const ts = "2026-07-06T22:23:22.226Z";
    const older = "2026-07-05T00:00:00.000Z";
    insertMutation(db, { ts, kind: "a" });
    insertMutation(db, { ts, kind: "b" });
    insertMutation(db, { ts: older, kind: "c" });
    const page1 = queryMutations(db, { limit: 2 });
    expect(page1.rows.map((r) => r.kind)).toEqual(["b", "a"]);
    expect(page1.nextCursor).toBeTruthy();
    const cursor = decodeMutationCursor(page1.nextCursor!);
    expect(cursor).toEqual({ ts, id: page1.rows[1].id });
    insertMutation(db, { ts: "2026-07-07T00:00:00.000Z", kind: "newer" });
    const page2 = queryMutations(db, { limit: 2, cursor: cursor! });
    expect(page2.rows.map((r) => r.kind)).toEqual(["c"]);
    expect(page2.nextCursor).toBeNull();
    expect(page2.rows.map((r) => r.id)).toEqual([expect.any(Number)]);
    db.close();
  });

  it("isolates filters from the cursor and reports total without the cursor", () => {
    const db = openDb(":memory:");
    insertMutation(db, { ts: "2026-07-08T00:00:00.000Z", kind: "keep" });
    insertMutation(db, { ts: "2026-07-07T00:00:00.000Z", kind: "keep" });
    insertMutation(db, { ts: "2026-07-06T00:00:00.000Z", kind: "drop" });
    const page1 = queryMutations(db, { kind: "keep", limit: 1 });
    expect(page1.total).toBe(2);
    expect(page1.rows[0].kind).toBe("keep");
    const page2 = queryMutations(db, { kind: "keep", limit: 1, cursor: decodeMutationCursor(page1.nextCursor!)! });
    expect(page2.total).toBe(2);
    expect(page2.rows).toHaveLength(1);
    expect(page2.rows[0].kind).toBe("keep");
    expect(page2.nextCursor).toBeNull();
    db.close();
  });
});
