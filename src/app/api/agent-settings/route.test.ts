import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import type Database from "better-sqlite3";
import { BOOTSTRAP_DRAFT, SETTINGS_CAP } from "../../../lib/agent-settings";
import { HelperRefusal } from "../../../server/collectors/agent-settings";
import { openDb } from "../../../server/db";
import sample from "../../../../workflow/fixtures/agent-settings-v1.json";
import { GET, POST, dynamic, type AgentSettingsRouteDeps } from "./route";

function handleGet(req: Request, deps: AgentSettingsRouteDeps) {
  Object.defineProperty(req, "jaxDeps", { value: deps });
  return GET(req);
}

function handlePost(req: Request, deps: AgentSettingsRouteDeps) {
  Object.defineProperty(req, "jaxDeps", { value: deps });
  return POST(req);
}

let testDb: Database.Database;
vi.mock("../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});

const REV = { settings: "absent", sources: { "opencode.json": "absent", "opencode.jsonc": "absent" } };
const OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";
const SENTINEL = "sk-leak-test-value-do-not-keep";
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

function post(payload: unknown, headers: Record<string, string> = {}) {
  return new Request("http://127.0.0.1/api/agent-settings", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "content-type": "application/json", ...headers },
  });
}

function rawPost(body: BodyInit, headers: Record<string, string> = {}) {
  return new Request("http://127.0.0.1/api/agent-settings", {
    method: "POST",
    body,
    headers: { "content-type": "application/json", ...headers },
  });
}

// Mocked variant of AgentSettingsRouteDeps: each field is a vi.fn() mock, so tests can read
// `.mock.calls` off it — the interface itself stays a plain function signature (route.ts is
// production code, untouched here).
type AgentSettingsDepsMock = {
  snapshot: Mock<AgentSettingsRouteDeps["snapshot"]>;
  preview: Mock<AgentSettingsRouteDeps["preview"]>;
  apply: Mock<AgentSettingsRouteDeps["apply"]>;
  sanitize: Mock<AgentSettingsRouteDeps["sanitize"]>;
};

function deps(over: Partial<AgentSettingsDepsMock> = {}): AgentSettingsDepsMock {
  return {
    snapshot: vi.fn(async () => ({
      editor_revision: REV,
      settings: null,
      connections: [{ id: "xai", apiKey: SENTINEL }],
    })),
    preview: vi.fn(async () => ({ affected: ["opencode.jsonc"], aliases: { default: "xai" } })),
    apply: vi.fn(async () => ({ effect: "published", editor_revision: REV, settings: sample })),
    sanitize: vi.fn(() => [{ id: "xai", auth: "api-key" }]),
    ...over,
  };
}

function rows() {
  return testDb.prepare("SELECT id, ok, payload FROM mutations").all() as {
    id: number; ok: number | null; payload: string;
  }[];
}

async function jsonOf(res: Response) {
  return { status: res.status, cache: res.headers.get("cache-control"), body: await res.json() };
}

const draft = { reviewers: sample.reviewers, builders: sample.builders };
const previewBody = { action: "preview", expected: REV, ...draft };
const saveBody = { action: "save", expected: REV, ...draft };

describe("agent-settings route", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
  });
  afterEach(() => {
    testDb.close();
  });

  it("is force-dynamic", () => {
    expect(dynamic).toBe("force-dynamic");
  });

  it("GET returns bootstrap draft, revision, sanitized connections, no-store", async () => {
    const d = deps();
    const res = await handleGet(new Request("http://127.0.0.1/api/agent-settings"), d);
    const got = await jsonOf(res);
    expect(got.status).toBe(200);
    expect(got.cache).toBe("no-store");
    expect(got.body).toEqual({
      ok: true,
      data: {
        editor_revision: REV,
        settings: null,
        draft: BOOTSTRAP_DRAFT,
        connections: [{ id: "xai", auth: "api-key" }],
      },
    });
    expect(JSON.stringify(got.body)).not.toContain(SENTINEL);
    expect(d.sanitize).toHaveBeenCalled();
    expect(d.snapshot.mock.calls[0]).toEqual([]);
  });

  it("GET response no longer has a health field (MOA-498 D5)", async () => {
    const res = await handleGet(new Request("http://127.0.0.1/api/agent-settings"), deps());
    const json = await res.json();
    expect(json.data).not.toHaveProperty("health");
  });

  it("GET helper refusal is 200 ok:false with the code", async () => {
    const res = await handleGet(new Request("http://127.0.0.1/api/agent-settings"), deps({
      snapshot: vi.fn(async () => { throw new HelperRefusal("native-config-malformed"); }),
    }));
    const got = await jsonOf(res);
    expect(got.status).toBe(200);
    expect(got.cache).toBe("no-store");
    expect(got.body).toEqual({ ok: false, error: "native-config-malformed" });
  });

  it("POST preview has no mutation and returns affected/aliases", async () => {
    const d = deps();
    const got = await jsonOf(await handlePost(post(previewBody), d));
    expect(got).toMatchObject({ status: 200, cache: "no-store", body: { ok: true, data: { affected: ["opencode.jsonc"], aliases: { default: "xai" } } } });
    expect(d.preview).toHaveBeenCalledOnce();
    expect(d.preview.mock.calls[0]).toHaveLength(2);
    expect(d.apply).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);
  });

  it("refuses unknown action, extra keys, bad content-type, same-site, prototype extras, UTF-8, oversized body", async () => {
    const d = deps();
    const cases: Request[] = [
      post({ ...previewBody, action: "drop-tables" }),
      post({ ...previewBody, extra: 1 }),
      post(previewBody, { "content-type": "text/plain" }),
      post(previewBody, { "sec-fetch-site": "same-site" }),
      post(previewBody, { "sec-fetch-site": "cross-site" }),
      rawPost(`{"action":"preview","expected":${JSON.stringify(REV)},"reviewers":${JSON.stringify(sample.reviewers)},"builders":${JSON.stringify(sample.builders)},"__proto__":{"x":1}}`),
      rawPost(new Uint8Array([0xff, 0xfe])),
      rawPost("x".repeat(SETTINGS_CAP + 1), { "content-length": String(SETTINGS_CAP + 1) }),
      post({ action: "save", expected: REV, ...draft, operation_id: "not-a-uuid" }),
    ];
    for (const req of cases) {
      const res = await handlePost(req, d);
      expect(res.status).toBe(200);
      expect((await res.json()).ok).toBe(false);
    }
    expect(d.preview).not.toHaveBeenCalled();
    expect(d.apply).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);
  });

  it("accepts same-origin and no-header preview", async () => {
    const d = deps();
    expect((await (await handlePost(post(previewBody, { "sec-fetch-site": "same-origin" }), d)).json()).ok).toBe(true);
    expect((await (await handlePost(post(previewBody), d)).json()).ok).toBe(true);
    expect(rows()).toEqual([]);
  });

  it("POST save publishes, audits without secrets, generates operation_id", async () => {
    const d = deps();
    const got = await jsonOf(await handlePost(post(saveBody), d));
    expect(got.status).toBe(200);
    expect(got.body).toEqual({
      ok: true,
      data: { effect: "published", editor_revision: REV, settings: sample },
    });
    expect(d.apply).toHaveBeenCalledOnce();
    expect(d.apply.mock.calls[0]).toHaveLength(3);
    const [, submittedRevision, operationId] = vi.mocked(d.apply).mock.calls[0];
    expect(submittedRevision).toEqual(REV);
    expect(vi.mocked(d.apply).mock.calls[0][0]).toEqual({
      kind: "save-settings",
      reviewers: sample.reviewers,
      builders: sample.builders,
    });
    expect(operationId).toMatch(UUID_RE);
    expect(d.preview).not.toHaveBeenCalled();
    const payload = JSON.parse(rows()[0].payload) as Record<string, unknown>;
    expect(payload).toMatchObject({ kind: "agent-settings-save", action: "save", operation_id: operationId, outcome: "done" });
    expect(payload).not.toHaveProperty("reviewers");
    expect(payload).not.toHaveProperty("builders");
    expect(JSON.stringify(payload)).not.toContain(SENTINEL);
    expect(rows()[0].ok).toBe(1);
  });

  it("a save apply result without a revision is surfaced without inventing one", async () => {
    const d = deps({ apply: vi.fn(async () => ({ effect: "published", settings: sample })) });
    const body = await (await handlePost(post(saveBody), d)).json();
    expect(body.ok).toBe(true);
    expect(body.data).not.toHaveProperty("editor_revision");
  });

  it("activation-pending is not ok:true", async () => {
    const d = deps({
      apply: vi.fn(async () => ({ effect: "activation-pending", editor_revision: REV, settings: sample })),
    });
    const body = await (await handlePost(post({ ...saveBody, operation_id: OP }), d)).json();
    expect(body).toMatchObject({
      ok: false, code: "activation-pending", effect: "activation-pending", audit: "recorded",
    });
    expect(body.ok).toBe(false);
    expect(JSON.parse(rows()[0].payload).outcome).toBe("done");
  });

  it("stale revision is a recorded rejection", async () => {
    const d = deps({
      apply: vi.fn(async () => { throw new HelperRefusal("agent-settings-source-changed"); }),
    });
    const body = await (await handlePost(post({ ...saveBody, operation_id: OP }), d)).json();
    expect(body).toEqual({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded",
      error: "agent-settings-source-changed",
    });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("failed");
  });

  it("audit-unavailable skips apply", async () => {
    testDb.exec("CREATE TRIGGER fail_ins BEFORE INSERT ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const d = deps();
    const body = await (await handlePost(post(saveBody), d)).json();
    expect(body).toMatchObject({ ok: false, code: "audit-unavailable", effect: "not-applied" });
    expect(d.apply).not.toHaveBeenCalled();
  });

  it("uncertain apply throw is unconfirmed", async () => {
    const d = deps({ apply: vi.fn(async () => { throw new Error("timeout after accept"); }) });
    const body = await (await handlePost(post({ ...saveBody, operation_id: OP }), d)).json();
    expect(body).toMatchObject({ ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded" });
    expect(JSON.parse(rows()[0].payload).outcome).toBe("abandoned");
  });

  it("applied-but-audit-pending is not a generic failed save", async () => {
    testDb.exec("CREATE TRIGGER fail_upd BEFORE UPDATE ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const d = deps();
    const body = await (await handlePost(post({ ...saveBody, operation_id: OP }), d)).json();
    expect(body).toMatchObject({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
    });
    expect(body).not.toHaveProperty("status");
    expect(body).not.toHaveProperty("value");
    expect(body.data).toMatchObject({ effect: "published" });
    expect(rows()[0].ok).toBeNull();
  });

  it("POST reconcile applies current settings and audits", async () => {
    const d = deps({
      snapshot: vi.fn(async () => ({ editor_revision: REV, settings: sample, connections: [] })),
    });
    const body = await (await handlePost(post({ action: "reconcile", expected: REV, operation_id: OP }), d)).json();
    expect(body).toMatchObject({ ok: true, data: { effect: "published" } });
    expect(d.apply).toHaveBeenCalledWith(
      { kind: "save-settings", reviewers: sample.reviewers, builders: sample.builders },
      REV,
      OP,
    );
    expect(JSON.parse(rows()[0].payload)).toMatchObject({ kind: "agent-settings-reconcile", action: "reconcile" });
  });

  it("GET no longer reads provider-credential mutations (MOA-498 D4 — the async BWS reconciliation this fed is gone)", async () => {
    const mod = await import("node:fs");
    expect(mod.existsSync(new URL("../../../server/db/provider-operations.ts", import.meta.url))).toBe(false);
  });
});
