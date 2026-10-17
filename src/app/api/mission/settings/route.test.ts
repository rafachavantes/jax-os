import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";

let testDb: Database.Database;
vi.mock("../../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});

import { openDb } from "../../../../server/db";
import { handleSettingsGet, handleSettingsPost } from "./handler";

describe("GET/POST /api/mission/settings", () => {
  beforeEach(() => { testDb = openDb(":memory:"); });
  afterEach(() => { testDb.close(); });

  it("GET returns the default 14 with no row", async () => {
    expect(await (await handleSettingsGet()).json()).toEqual({ ok: true, data: { archiveAfterDays: 14 } });
  });

  it("POST writes a bounded value and GET reflects it", async () => {
    const req = new Request("http://localhost/api/mission/settings", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ archiveAfterDays: 7 }),
    });
    expect(await (await handleSettingsPost(req)).json()).toEqual({ ok: true, data: { archiveAfterDays: 7 } });
    expect(await (await handleSettingsGet()).json()).toEqual({ ok: true, data: { archiveAfterDays: 7 } });
  });

  it("POST refuses a value outside 1-30 without writing, as a quiet 200 {ok:false}", async () => {
    const req = new Request("http://localhost/api/mission/settings", {
      method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ archiveAfterDays: 90 }),
    });
    const res = await handleSettingsPost(req);
    expect(res.status).toBe(200);
    expect((await res.json()).ok).toBe(false);
    expect(await (await handleSettingsGet()).json()).toEqual({ ok: true, data: { archiveAfterDays: 14 } });
  });
});
