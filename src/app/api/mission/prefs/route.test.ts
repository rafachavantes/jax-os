import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";

let testDb: Database.Database;
vi.mock("../../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});

import { openDb } from "../../../../server/db";
import { getProjectPrefs } from "../../../../server/db/workflows";
import { handlePrefsPost } from "./handler";

function post(body: unknown, headers: Record<string, string> = {}) {
  return new Request("http://localhost/api/mission/prefs", {
    method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body),
  });
}

describe("POST /api/mission/prefs", () => {
  beforeEach(() => { testDb = openDb(":memory:"); });
  afterEach(() => { testDb.close(); });

  it("sets hidden and pinned, readable back via getProjectPrefs", async () => {
    const r1 = await handlePrefsPost(post({ project: "p1", hidden: true }));
    expect(await r1.json()).toEqual({ ok: true, data: { pinned: false, hiddenAt: expect.any(String) } });
    const r2 = await handlePrefsPost(post({ project: "p1", pinned: true }));
    expect((await r2.json()).data.pinned).toBe(true);
    expect(getProjectPrefs(testDb).p1.pinned).toBe(true);
  });

  it("refuses a malformed project, a non-boolean field, or an empty patch", async () => {
    for (const body of [{}, { project: "../x", hidden: true }, { project: "p1", hidden: "yes" }, { project: "p1" }]) {
      expect(await (await handlePrefsPost(post(body))).json()).toEqual({ ok: false, error: "invalid payload" });
    }
  });

  it("blocks a cross-site request", async () => {
    expect(await (await handlePrefsPost(post({ project: "p1", hidden: true }, { "sec-fetch-site": "cross-site" }))).json())
      .toEqual({ ok: false, error: "cross-site request blocked" });
  });
});
