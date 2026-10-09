import { beforeEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";

// route.ts calls the real singleton getDb() (../../../../server/db) — this test
// swaps ONLY that export for an injected ":memory:" db, never touching
// ~/.jax-os/jaxos.db, while keeping every other export (openDb, MIGRATIONS, ...)
// real, so insertEvent's actual terminal-claim SQL runs for real (spec §2.6
// cold review G1, §7.2 #33's route-level assertion).
let testDb: Database.Database;
vi.mock("../../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});

// route.ts also calls notificationTargetFromEnv()/forwardNotification() on every successful
// insert. Stubbed so the real module (and its real `fetch`) is never loaded — this
// test never reads or sets env vars, so this mock alone is what keeps it hermetic,
// even when NOTIFICATION_WEBHOOK_URL/SECRET happen to be set in the shell.
//
// Both stubs read their module-level bindings LAZILY, at call time — the same pattern
// `getDb: () => testDb` above uses. This matters: `vi.mock`'s factory runs while the
// mocked module is being imported, which is BEFORE this file's own `let`/`const`
// initializers run, so the factory must not READ a module-level binding, only close
// over one. `forwardCalls` is a plain counter for the same reason (and because a spy
// would buy nothing here).
let hermesTarget: { url: string; secret: string } | null = null;
let forwardOk = true;
let forwardCalls = 0;
vi.mock("../../../../server/collectors/workflow-webhook", () => ({
  notificationTargetFromEnv: () => hermesTarget,
  forwardNotification: async () => {
    forwardCalls++;
    return forwardOk ? { ok: true, status: 200 } : { ok: false, error: "stub failure" };
  },
}));

// MOA-498: insertEvent's delivery ladder now reads integrations.webhook via
// readGeneralSettings. Force it on (delegating to the real reader for every other field, so
// import-time consumers like reposRoot()/REPOS_ROOT keep their real values) so this route's
// forward-path tests (AFK on + forward types enabled) keep exercising the same behavior they
// always did. The gate itself is covered in server/db/workflows.test.ts.
// MOA-502 Decision 1: same pattern for integrations.classifier — the classify-route tests
// here settle rows inserted through the real derive.ts rung 7, which now gates queueing on it.
vi.mock("../../../../server/settings", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/settings")>();
  return {
    ...actual,
    readGeneralSettings: (...args: Parameters<typeof actual.readGeneralSettings>) => {
      const r = actual.readGeneralSettings(...args);
      return r.ok ? { ...r, data: { ...r.data, integrations: { ...r.data.integrations, webhook: true, classifier: true } } } : r;
    },
  };
});

import { openDb } from "../../../../server/db";
import { insertEvent, setAfk, setForwardTypes } from "../../../../server/db/workflows";
import { FORWARDABLE_EVENT_TYPES } from "../../../../lib/workflow";
import { POST } from "./route";
import { POST as classifyPost } from "./classify/route";

function postRequest(body: unknown): Request {
  return new Request("http://localhost/api/workflow/events", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

const RUN_STARTED_BODY = {
  project: "p1", run_id: "r1", role: "builder", type: "run-started",
  source: "deterministic", emitter: "wrapper",
  payload: {
    phase: "B", runtime: "opencode-deepseek", kind: "plan", target: "/home/rafa/repos/jax-os/.local/docs/plans/example.md",
    caller: "claude", caller_session: "sess-abc123", model: "grok-4.6", effort: "high",
    session: "jax-p1-builder-r1", repo: "/home/rafa/repos/jax-os",
  },
};
const RUN_FINISHED_BODY = {
  project: "p1", run_id: "r1", role: "builder", type: "run-finished",
  source: "deterministic", emitter: "wrapper",
  payload: {
    phase: "B", exit_code: 0, contract_status: "ok", report_path: "/home/rafa/repos/jax-os/.local/reports/r1.md",
    summary: "done", head_sha: "a".repeat(40), result: "success",
  },
};

describe("POST /api/workflow/events", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
    hermesTarget = null;
    forwardOk = true;
    forwardCalls = 0;
  });

  it("persists run-started then run-finished, both 200 {ok:true}", async () => {
    const started = await POST(postRequest(RUN_STARTED_BODY));
    expect(started.status).toBe(200);
    expect((await started.json()).ok).toBe(true);
    const finished = await POST(postRequest(RUN_FINISHED_BODY));
    expect(finished.status).toBe(200);
    expect((await finished.json()).ok).toBe(true);
  });

  it("maps a second run-finished for the same run_id to 409 (spec §2.6 cold review G1)", async () => {
    await POST(postRequest(RUN_STARTED_BODY));
    const first = await POST(postRequest(RUN_FINISHED_BODY));
    expect(first.status).toBe(200);
    const second = await POST(postRequest(RUN_FINISHED_BODY));
    expect(second.status).toBe(409);
    expect(await second.json()).toEqual({ ok: false, error: "run already finished" });
  });

  it("keeps the plain 200 {ok:false} shape for every other insert/validation failure", async () => {
    const res = await POST(postRequest({ ...RUN_STARTED_BODY, payload: { ...RUN_STARTED_BODY.payload, kind: "nonsense" } }));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.ok).toBe(false);
    expect(json.error).toMatch(/kind not allowed/);
  });

  it("accepts opencode-builder run-started and still rejects an extra payload key", async () => {
    const simple = {
      ...RUN_STARTED_BODY, run_id: "r-ob",
      payload: { ...RUN_STARTED_BODY.payload, runtime: "opencode-builder" },
    };
    const okSimple = await POST(postRequest(simple));
    expect(okSimple.status).toBe(200);
    expect((await okSimple.json()).ok).toBe(true);
    const body = {
      ...RUN_STARTED_BODY, run_id: "r-ob2",
      payload: {
        ...RUN_STARTED_BODY.payload, runtime: "opencode-builder", kind: "build", verify: "true",
        requested_profile: "default", root_build_run_id: "aabbccddeeff",
      },
    };
    const ok = await POST(postRequest(body));
    expect(ok.status).toBe(200);
    expect((await ok.json()).ok).toBe(true);
    const bad = await POST(postRequest({ ...body, payload: { ...body.payload, extra: 1 } }));
    expect(bad.status).toBe(200);
    const json = await bad.json();
    expect(json.ok).toBe(false);
    expect(json.error).toMatch(/unknown key extra/);
  });
});

describe("POST /api/workflow/events/classify", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
  });

  it.each([
    ["both shapes at once", { id: 1, capsule_status: "done", failed: true }],
    ["neither field",       { id: 1 }],
    ["failed: false",       { id: 1, failed: false }],
    ["an extra key",        { id: 1, capsule_status: "done", note: "hi" }],
    ["a non-integer id",    { id: "1", capsule_status: "done" }],
    ["capsule_status unknown", { id: 1, capsule_status: "unknown" }],
    ["a status outside the enum", { id: 1, capsule_status: "sleepy" }],
    ["indeterminate: false", { id: 1, indeterminate: false }],
    ["indeterminate combined with capsule_status", { id: 1, capsule_status: "done", indeterminate: true }],
    ["merge_ask out of range",  { id: 1, capsule_status: "done", merge_ask: 1.5 }],
    ["merge_ask non-numeric",   { id: 1, capsule_status: "done", merge_ask: "x" }],
    ["merge_ask with failed",   { id: 1, failed: true, merge_ask: 0.5 }],
  ])("refuses %s", async (_name, body) => {
    const res = await classifyPost(new Request("http://127.0.0.1:3100/api/workflow/events/classify", {
      method: "POST", headers: { "Content-Type": "application/json", origin: "http://127.0.0.1:3100" },
      body: JSON.stringify(body),
    }));
    expect(await res.json()).toMatchObject({ ok: false });
  });

  it("accepts {id, indeterminate: true}, settles the row unknown/abandoned, and is a no-op on replay", async () => {
    // F2 (review 73891ba1c1b1): an empty db + `{ok: true}` alone would still pass with a route
    // that accepts the shape but never calls `classifyDeferred` — seed a real deferred row (same
    // insertEvent shape workflows.test.ts's classifyDeferred tests use to reach rung 7) and check
    // the actual DB transition, not just the envelope.
    const row = insertEvent(testDb, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: "234790:1787586213",
      payload: { message_tail: "ambiguous prose" },
    });
    const post = () => classifyPost(new Request("http://127.0.0.1:3100/api/workflow/events/classify", {
      method: "POST", headers: { "Content-Type": "application/json", origin: "http://127.0.0.1:3100" },
      body: JSON.stringify({ id: row.id, indeterminate: true }),
    }));
    const res = await post();
    expect(await res.json()).toMatchObject({ ok: true, data: { settled: true } });
    const stored = testDb.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(row.id) as { payload: string };
    const payload = JSON.parse(stored.payload);
    expect(payload.capsule_status).toBe("unknown");
    expect(payload.capsule_rule).toBe("abandoned");
    expect(payload.capsule_attempts).toBe(0); // unchanged from the deferred insert — indeterminate never touches the counter
    const second = await post();
    expect(await second.json()).toMatchObject({ ok: true, data: { settled: false } }); // replay is a no-op
  });

  it("accepts {id, capsule_status, merge_ask}, stores merge_ask, and still accepts the plain shape unchanged", async () => {
    const withMergeAsk = insertEvent(testDb, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: "234790:1787586213",
      payload: { message_tail: "posso mergear feat/x em main?\nAguardo seu retorno." },
    });
    const res = await classifyPost(new Request("http://127.0.0.1:3100/api/workflow/events/classify", {
      method: "POST", headers: { "Content-Type": "application/json", origin: "http://127.0.0.1:3100" },
      body: JSON.stringify({ id: withMergeAsk.id, capsule_status: "needs_input", merge_ask: 0.82 }),
    }));
    expect(await res.json()).toMatchObject({ ok: true, data: { settled: true } });
    const storedA = testDb.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(withMergeAsk.id) as { payload: string };
    expect(JSON.parse(storedA.payload)).toMatchObject({ capsule_status: "needs_input", capsule_rule: "classified", merge_ask: 0.82 });

    const plain = insertEvent(testDb, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%2", tmux_incarnation: "234790:1787586213",
      payload: { message_tail: "ambiguous prose" },
    });
    const res2 = await classifyPost(new Request("http://127.0.0.1:3100/api/workflow/events/classify", {
      method: "POST", headers: { "Content-Type": "application/json", origin: "http://127.0.0.1:3100" },
      body: JSON.stringify({ id: plain.id, capsule_status: "done" }),
    }));
    expect(await res2.json()).toMatchObject({ ok: true, data: { settled: true } });
    const storedB = testDb.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(plain.id) as { payload: string };
    expect(JSON.parse(storedB.payload)).not.toHaveProperty("merge_ask");
  });

  it.each([0, 1])("accepts and stores the exact boundary merge_ask value %s (cold review round 1 F3)", async (boundary) => {
    const row = insertEvent(testDb, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: "234790:1787586213",
      payload: { message_tail: "posso mergear feat/x em main?\nAguardo seu retorno." },
    });
    const res = await classifyPost(new Request("http://127.0.0.1:3100/api/workflow/events/classify", {
      method: "POST", headers: { "Content-Type": "application/json", origin: "http://127.0.0.1:3100" },
      body: JSON.stringify({ id: row.id, capsule_status: "needs_input", merge_ask: boundary }),
    }));
    expect(await res.json()).toMatchObject({ ok: true, data: { settled: true } });
    const stored = testDb.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(row.id) as { payload: string };
    expect(JSON.parse(stored.payload)).toMatchObject({ merge_ask: boundary });
  });
});

// spec §7.2 #39 (MOA-453). The DB-level suppression is covered in workflows.test.ts; what is
// pinned HERE is that the route answers 200 (never 409 — the retry still has a push to reach)
// and that the suppression path adds NO forwarding rule of its own: the returned stored row
// meets route.ts's existing `delivery !== "pending"` early return, and that alone decides.
const MERGE_APPROVED_BODY = {
  project: "p1", role: "lead", type: "merge-approved",
  source: "deterministic", emitter: "wrapper",
  payload: {
    phase: "Phase C", branch: "feat/x", sha: "a".repeat(40), target: "main",
    approved_by: "rafa", merge_sha: "c".repeat(40),
  },
};

describe("POST /api/workflow/events — duplicate merge-approved (spec §2.6, MOA-453)", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
    hermesTarget = null;
    forwardOk = true;
    forwardCalls = 0;
  });

  it("answers 200 with the SAME row id, never 409", async () => {
    const first = await POST(postRequest(MERGE_APPROVED_BODY));
    const second = await POST(postRequest(MERGE_APPROVED_BODY));
    expect(first.status).toBe(200);
    expect(second.status).toBe(200);
    const a = await first.json();
    const b = await second.json();
    expect(b.ok).toBe(true);
    expect(b.data.id).toBe(a.data.id);
    expect(testDb.prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 1 });
  });

  it("does not forward a second time when the first row was already delivered", async () => {
    setAfk(testDb, true);
    setForwardTypes(testDb, [...FORWARDABLE_EVENT_TYPES]);
    hermesTarget = { url: "http://127.0.0.1:9/hook", secret: "s" };
    const first = await POST(postRequest(MERGE_APPROVED_BODY));
    expect((await first.json()).forwarded).toBe(true);
    expect(forwardCalls).toBe(1);
    const second = await POST(postRequest(MERGE_APPROVED_BODY));
    expect((await second.json()).forwarded).toBe(false);
    expect(forwardCalls).toBe(1); // the delivered row takes the early return
  });

  it("DOES forward the duplicate when the first forward failed and left the row pending", async () => {
    setAfk(testDb, true);
    setForwardTypes(testDb, [...FORWARDABLE_EVENT_TYPES]);
    hermesTarget = { url: "http://127.0.0.1:9/hook", secret: "s" };
    forwardOk = false;
    const first = await POST(postRequest(MERGE_APPROVED_BODY));
    expect((await first.json()).forwarded).toBe(false);
    forwardOk = true;
    const second = await POST(postRequest(MERGE_APPROVED_BODY));
    const b = await second.json();
    expect(b.forwarded).toBe(true); // that delivery never happened; retrying it is correct
    expect(b.data.delivery).toBe("delivered");
    expect(forwardCalls).toBe(2);
  });

  it("forwards neither the first nor the duplicate with AFK off (delivery: local)", async () => {
    hermesTarget = { url: "http://127.0.0.1:9/hook", secret: "s" }; // AFK left OFF
    const first = await POST(postRequest(MERGE_APPROVED_BODY));
    expect((await first.json()).data.delivery).toBe("local");
    const second = await POST(postRequest(MERGE_APPROVED_BODY));
    expect((await second.json()).forwarded).toBe(false);
    expect(forwardCalls).toBe(0);
  });
});

describe("POST /api/workflow/events — turn-stopped is local telemetry (MOA-469 live corrections)", () => {
  const STOP_BODY = (over: Record<string, unknown> = {}) => ({
    project: "p1", role: "lead", type: "turn-stopped", source: "deterministic", emitter: "claude-stop",
    pane: "%1", tmux_incarnation: "100:1000",
    payload: { capsule_status: "done", capsule_rule: "tag" },
    ...over,
  });

  beforeEach(() => {
    testDb = openDb(":memory:");
    hermesTarget = null;
    forwardOk = true;
    forwardCalls = 0;
  });

  it("a valid stop persists with delivery local and invokes no Hermes forward, even with AFK on and a Hermes target configured", async () => {
    setAfk(testDb, true);
    setForwardTypes(testDb, [...FORWARDABLE_EVENT_TYPES]);
    hermesTarget = { url: "http://127.0.0.1:9/hook", secret: "s" };
    const res = await POST(postRequest(STOP_BODY()));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json).toMatchObject({ ok: true, forwarded: false, data: { delivery: "local" } });
    expect(forwardCalls).toBe(0);
    const stored = testDb.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE type = 'turn-stopped'").get() as { n: number };
    expect(stored.n).toBe(1); // persisted, not dropped
  });

  it("a native codex stop likewise persists without forwarding", async () => {
    setAfk(testDb, true);
    setForwardTypes(testDb, [...FORWARDABLE_EVENT_TYPES]);
    hermesTarget = { url: "http://127.0.0.1:9/hook", secret: "s" };
    const res = await POST(postRequest(STOP_BODY({
      emitter: "codex-stop", source: "behavioral", harness_session: "0191f0aa-d1ce-7000-8000-0000000000d1",
      pane: undefined, tmux_incarnation: undefined, payload: { message_tail: "prose" },
    })));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json).toMatchObject({ ok: true, forwarded: false, data: { delivery: "local" } });
    expect(forwardCalls).toBe(0);
  });

  it("run-finished still forwards while a stop never does (forwarding tests kept)", async () => {
    setAfk(testDb, true);
    setForwardTypes(testDb, [...FORWARDABLE_EVENT_TYPES]);
    hermesTarget = { url: "http://127.0.0.1:9/hook", secret: "s" };
    await POST(postRequest(STOP_BODY()));
    const finished = await POST(postRequest(RUN_FINISHED_BODY));
    expect((await finished.json()).forwarded).toBe(true);
    expect(forwardCalls).toBe(1);
  });
});
