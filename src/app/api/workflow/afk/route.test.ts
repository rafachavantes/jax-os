import { describe, expect, it } from "vitest";
import { openDb } from "../../../../server/db";
import { getAfkEnabled, getForwardTypes, setAfk, setForwardTypes } from "../../../../server/db/workflows";
import { DEFAULT_FORWARD_TYPES } from "../../../../lib/workflow";
import { handleAfkGet, handleAfkPost, type AfkRouteDeps } from "./handler";

// route.test.ts was listed as a plan deliverable but the plan gave no test content for it — this
// file exercises the origin guard, the error contract, and real GET/POST round-tripping against
// an injected `:memory:` db (never the real ~/.jax-os/jaxos.db).
function depsWithDb(): AfkRouteDeps {
  const db = openDb(":memory:");
  return { getDb: () => db, getAfkEnabled, setAfk, getForwardTypes, setForwardTypes };
}

function postRequest(body: unknown, headers: Record<string, string> = { "content-type": "application/json" }): Request {
  return new Request("http://localhost/api/workflow/afk", { method: "POST", headers, body: JSON.stringify(body) });
}

describe("handleAfkGet", () => {
  it("defaults to disabled with the default forward types, and reflects toggles made directly on the db", async () => {
    const deps = depsWithDb();
    expect(await (await handleAfkGet(undefined, deps)).json()).toEqual({ ok: true, data: { enabled: false, forwardTypes: DEFAULT_FORWARD_TYPES } });
    setAfk(deps.getDb(), true);
    setForwardTypes(deps.getDb(), ["run-finished"]);
    expect(await (await handleAfkGet(undefined, deps)).json()).toEqual({ ok: true, data: { enabled: true, forwardTypes: ["run-finished"] } });
  });

  it("a DB failure returns {ok:false,error}, never a crash (Finding 9, house rule 3)", async () => {
    const base = depsWithDb();
    const deps: AfkRouteDeps = {
      getDb: base.getDb,
      getAfkEnabled: () => { throw new Error("db unavailable"); },
      setAfk,
      getForwardTypes: base.getForwardTypes,
      setForwardTypes,
    };
    const res = await handleAfkGet(undefined, deps);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "db unavailable" });
  });
});

describe("handleAfkPost", () => {
  it("blocks same-site and cross-site browser requests before opening the db", async () => {
    const deps = depsWithDb();
    for (const fetchSite of ["same-site", "cross-site"]) {
      const req = postRequest({ enabled: true }, { "sec-fetch-site": fetchSite, "content-type": "application/json" });
      expect(await (await handleAfkPost(req, deps)).json()).toEqual({ ok: false, error: "cross-site request blocked" });
    }
    expect(getAfkEnabled(deps.getDb())).toBe(false);
  });

  it("toggles enabled on, then off, and audits each as a mutation", async () => {
    const deps = depsWithDb();
    expect(await (await handleAfkPost(postRequest({ enabled: true }), deps)).json()).toEqual({
      ok: true, data: { enabled: true, forwardTypes: DEFAULT_FORWARD_TYPES },
    });
    expect(getAfkEnabled(deps.getDb())).toBe(true);
    expect(await (await handleAfkPost(postRequest({ enabled: false }), deps)).json()).toEqual({
      ok: true, data: { enabled: false, forwardTypes: DEFAULT_FORWARD_TYPES },
    });
    expect(getAfkEnabled(deps.getDb())).toBe(false);
    const rows = deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'workflow-afk-toggle'").get();
    expect(rows).toEqual({ n: 2 });
  });

  it("rejects a body with neither key, without touching the db", async () => {
    const deps = depsWithDb();
    for (const body of [{}, null, [], "x", 5]) {
      const res = await handleAfkPost(postRequest(body), deps);
      expect(await res.json()).toEqual({ ok: false, error: "enabled or forwardTypes required" });
    }
    expect(getAfkEnabled(deps.getDb())).toBe(false);
    expect(getForwardTypes(deps.getDb())).toEqual(DEFAULT_FORWARD_TYPES);
  });

  it("rejects a non-boolean enabled without touching the db", async () => {
    const deps = depsWithDb();
    for (const body of [{ enabled: "true" }, { enabled: 1 }, { enabled: null }]) {
      const res = await handleAfkPost(postRequest(body), deps);
      expect(await res.json()).toEqual({ ok: false, error: "enabled must be a boolean" });
    }
    expect(getAfkEnabled(deps.getDb())).toBe(false);
  });

  it("a DB failure returns {ok:false,error}, never a crash (Finding 9, house rule 3)", async () => {
    const base = depsWithDb();
    const deps: AfkRouteDeps = {
      getDb: base.getDb,
      getAfkEnabled,
      setAfk: () => { throw new Error("db unavailable"); },
      getForwardTypes: base.getForwardTypes,
      setForwardTypes: base.setForwardTypes,
    };
    const res = await handleAfkPost(postRequest({ enabled: true }), deps);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "db unavailable" });
  });

  it("the mutation payload carries the previous state, not just the new one (Finding 2)", async () => {
    const deps = depsWithDb();
    await handleAfkPost(postRequest({ enabled: true }), deps);
    await handleAfkPost(postRequest({ enabled: false }), deps);
    const rows = deps.getDb().prepare(
      "SELECT payload FROM mutations WHERE kind = 'workflow-afk-toggle' ORDER BY id",
    ).all() as { payload: string }[];
    expect(rows.map((r) => JSON.parse(r.payload))).toEqual([
      expect.objectContaining({ enabled: true, previous: false }),
      expect.objectContaining({ enabled: false, previous: true }),
    ]);
  });

  it("accepts forwardTypes alone, with no enabled key", async () => {
    const deps = depsWithDb();
    expect(await (await handleAfkPost(postRequest({ forwardTypes: ["question", "merge-approved"] }), deps)).json()).toEqual({
      ok: true, data: { enabled: false, forwardTypes: ["question", "merge-approved"] },
    });
    expect(getForwardTypes(deps.getDb())).toEqual(["question", "merge-approved"]);
  });

  it("a request with both keys produces two separately audited mutation rows", async () => {
    const deps = depsWithDb();
    await handleAfkPost(postRequest({ enabled: true, forwardTypes: ["run-finished"] }), deps);
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'workflow-afk-toggle'").get()).toEqual({ n: 1 });
    expect(deps.getDb().prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'workflow-forward-types'").get()).toEqual({ n: 1 });
  });

  it("rejects an invalid forwardTypes element, a non-array value, or a duplicate, without ever calling setForwardTypes (round F2)", async () => {
    const deps = depsWithDb();
    let calls = 0;
    const spiedDeps: AfkRouteDeps = { ...deps, setForwardTypes: (db, types, now) => { calls += 1; setForwardTypes(db, types, now); } };
    for (const forwardTypes of [["not-a-type"], "question", 5, ["question", "question"]]) {
      const res = await handleAfkPost(postRequest({ forwardTypes }), spiedDeps);
      expect(await res.json()).toEqual({ ok: false, error: "forwardTypes must be an array of forwardable event types with no duplicates" });
    }
    expect(calls).toBe(0);
    expect(getForwardTypes(deps.getDb())).toEqual(DEFAULT_FORWARD_TYPES);
  });
});
