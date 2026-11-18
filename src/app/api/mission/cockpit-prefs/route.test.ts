import { describe, expect, it } from "vitest";
import { openDb } from "../../../../server/db";
import { getCockpitPrefs, resetCockpitLayout, saveCockpitLayout, setCockpitPrScope } from "../../../../server/db/cockpit-prefs";
import { handleCockpitPrefsGet, handleCockpitPrefsPost, type CockpitPrefsDeps } from "./handler";

function deps(db = openDb(":memory:")): CockpitPrefsDeps {
  return {
    getDb: () => db,
    getCockpitPrefs,
    saveCockpitLayout: (d, rev, projects, now) => saveCockpitLayout(d, rev, projects, now),
    resetCockpitLayout: (d, rev, now) => resetCockpitLayout(d, rev, now),
    setCockpitPrScope: (d, scope, now) => setCockpitPrScope(d, scope, now),
  };
}

function post(body: unknown, headers: Record<string, string> = {}): Request {
  return new Request("http://localhost/api/mission/cockpit-prefs", {
    method: "POST",
    headers: { "content-type": "application/json", ...headers },
    body: typeof body === "string" ? body : JSON.stringify(body),
  });
}

describe("GET /api/mission/cockpit-prefs", () => {
  it("returns the documented defaults", async () => {
    const body = await (await handleCockpitPrefsGet(deps())).json();
    expect(body).toEqual({ ok: true, data: { revision: 1, layout: null, prScope: "mine" } });
  });

  it("a storage exception is a safe source warning, never a throw", async () => {
    const d = deps();
    const body = await (await handleCockpitPrefsGet({ ...d, getCockpitPrefs: () => { throw new Error("boom"); } })).json();
    expect(body).toEqual({ ok: false, error: "boom" });
  });
});

describe("POST /api/mission/cockpit-prefs", () => {
  it("layout-save returns the new snapshot, then a stale revision is a typed conflict", async () => {
    const d = deps();
    const saved = await (await handleCockpitPrefsPost(post({
      action: "layout-save", expectedRevision: 1,
      projects: [{ project: "Alpha", width: "full" }, { project: "nested/Beta", width: "half" }],
    }), d)).json();
    expect(saved).toEqual({ ok: true, data: { conflict: false, snapshot: { revision: 2, layout: [{ project: "Alpha", width: "full" }, { project: "nested/Beta", width: "half" }], prScope: "mine" } } });

    const conflictBody = await (await handleCockpitPrefsPost(post({
      action: "layout-save", expectedRevision: 1, projects: [{ project: "Other", width: "third" }],
    }), d)).json();
    expect(conflictBody).toEqual({ ok: true, data: { conflict: true, snapshot: saved.data.snapshot } });
  });

  it("layout-reset is layout-only and revision checked", async () => {
    const db = openDb(":memory:");
    const d = deps(db);
    await handleCockpitPrefsPost(post({ action: "pr-scope", scope: "all" }), d);
    await handleCockpitPrefsPost(post({ action: "layout-save", expectedRevision: 1, projects: [{ project: "Alpha", width: "half" }] }), d);
    const reset = await (await handleCockpitPrefsPost(post({ action: "layout-reset", expectedRevision: 2 }), d)).json();
    expect(reset).toEqual({ ok: true, data: { conflict: false, snapshot: { revision: 3, layout: null, prScope: "all" } } });
  });

  it("pr-scope writes are audited and do not bump the layout revision", async () => {
    const body = await (await handleCockpitPrefsPost(post({ action: "pr-scope", scope: "all" }), deps())).json();
    expect(body).toEqual({ ok: true, data: { revision: 1, layout: null, prScope: "all" } });
  });

  it("rejects malformed JSON, unknown actions, extra keys, bad widths, duplicate identities, bad revisions and oversized bodies", async () => {
    const d = deps();
    const cases: unknown[] = [
      "{",
      { action: "nope" },
      { action: "layout-save", expectedRevision: 1, projects: [], extra: true },
      { action: "pr-scope", scope: "team" },
      { action: "layout-save", expectedRevision: 1.5, projects: [] },
      { action: "layout-save", expectedRevision: 1, projects: [{ project: "A", width: "wide" }] },
      { action: "layout-save", expectedRevision: 1, projects: [{ project: "A", width: "third" }, { project: "A", width: "half" }] },
      { action: "layout-save", expectedRevision: 1, projects: [{ project: "../escape", width: "third" }] },
      { action: "layout-reset", expectedRevision: -1 },
    ];
    for (const body of cases) {
      const res = await (await handleCockpitPrefsPost(post(body), d)).json();
      expect(res).toEqual({ ok: false, error: expect.any(String) });
    }
    const big = JSON.stringify({ action: "pr-scope", scope: "all", pad: "x".repeat(200 * 1024) });
    expect(await (await handleCockpitPrefsPost(post(big), d)).json()).toMatchObject({ ok: false });
  });

  it("blocks a cross-site request", async () => {
    const res = await handleCockpitPrefsPost(post({ action: "pr-scope", scope: "all" }, { "sec-fetch-site": "cross-site" }), deps());
    expect(await res.json()).toEqual({ ok: false, error: "cross-site request blocked" });
  });

  it("a storage exception is reported without a false success", async () => {
    const d: CockpitPrefsDeps = { ...deps(), saveCockpitLayout: () => { throw new Error("disk"); } };
    const body = await (await handleCockpitPrefsPost(post({ action: "layout-save", expectedRevision: 1, projects: [] }), d)).json();
    expect(body).toEqual({ ok: false, error: "disk" });
  });
});
