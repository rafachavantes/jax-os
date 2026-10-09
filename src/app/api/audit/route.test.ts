import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({ getDb: vi.fn() }));
vi.mock("@/server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/server/db")>();
  return { ...actual, getDb: () => harness.getDb() };
});

import { openDb } from "@/server/db";
import { encodeMutationCursor, insertMutation } from "@/server/db/mutations";
import { GET } from "./route";

function req(qs: string) {
  return new Request(`http://127.0.0.1/api/audit${qs}`);
}

describe("GET /api/audit", () => {
  beforeEach(() => {
    const db = openDb(":memory:");
    insertMutation(db, { ts: "2026-07-08T00:00:00.000Z", kind: "keep", ok: true });
    insertMutation(db, { ts: "2026-07-07T00:00:00.000Z", kind: "keep", ok: true });
    harness.getDb.mockReset();
    harness.getDb.mockReturnValue(db);
  });

  it("rejects a legacy offset before touching the db", async () => {
    const res = await GET(req("?offset=0"));
    expect(await res.json()).toEqual({ ok: false, error: "offset is not supported" });
    expect(harness.getDb).not.toHaveBeenCalled();
  });

  it("rejects a malformed cursor before touching the db", async () => {
    const res = await GET(req("?cursor=%%%"));
    expect(await res.json()).toEqual({ ok: false, error: "invalid cursor" });
    expect(harness.getDb).not.toHaveBeenCalled();
  });

  it("rejects over-cap kind and bad from/to before touching the db", async () => {
    expect((await (await GET(req(`?kind=${"é".repeat(129)}`))).json()).ok).toBe(false);
    expect((await (await GET(req("?from=not-a-date"))).json()).ok).toBe(false);
    expect((await (await GET(req("?to=nope"))).json()).ok).toBe(false);
    expect(harness.getDb).not.toHaveBeenCalled();
  });

  it("clamps limit and serves a cursor page", async () => {
    const zero = await (await GET(req("?limit=0"))).json();
    expect(zero.ok).toBe(true);
    expect(zero.data.rows).toHaveLength(2);
    const body = await (await GET(req("?limit=1"))).json();
    expect(body.data.rows).toHaveLength(1);
    expect(body.data.nextCursor).toBe(encodeMutationCursor(body.data.rows[0].ts, body.data.rows[0].id));
    const next = await (await GET(req(`?limit=1&cursor=${body.data.nextCursor}`))).json();
    expect(next.data.rows).toHaveLength(1);
    expect(next.data.nextCursor).toBeNull();
    expect(harness.getDb).toHaveBeenCalled();
  });

  it("accepts YYYY-MM-DD and ISO from/to", async () => {
    expect((await (await GET(req("?from=2026-07-07"))).json()).ok).toBe(true);
    expect((await (await GET(req("?from=2026-07-07T00:00:00.000Z"))).json()).ok).toBe(true);
    expect(harness.getDb).toHaveBeenCalled();
  });
});
