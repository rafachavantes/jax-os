import { spawn, spawnSync } from "node:child_process";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi, type Mock } from "vitest";
import type Database from "better-sqlite3";
import { modelInputPayload, SETTINGS_CAP, type ModelChoice } from "../../../lib/agent-settings";
import { HelperRefusal, runSettingsHelper, type SpawnImpl } from "../../../server/collectors/agent-settings";
import {
  applyProviders,
  previewProviders,
  sanitizeConnections,
  snapshotProviders,
} from "../../../server/collectors/opencode-providers";
import { openDb } from "../../../server/db";
import { GET, POST, dynamic, type ProvidersRouteDeps } from "./route";
import {
  GET as SettingsGet,
  POST as SettingsPost,
  type AgentSettingsRouteDeps,
} from "../agent-settings/route";

function handleGet(req: Request, deps: ProvidersRouteDeps) {
  Object.defineProperty(req, "jaxDeps", { value: deps });
  return GET(req);
}

function handlePost(req: Request, deps: ProvidersRouteDeps) {
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
const OP2 = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff";
const SENTINEL = "sk-leak-test-value-do-not-keep";
const CONN = { id: "acme", adapter: "openai-compatible", base_url: "https://api.example.com" };

function post(payload: unknown, headers: Record<string, string> = {}) {
  return new Request("http://127.0.0.1/api/opencode-providers", {
    method: "POST",
    body: JSON.stringify(payload),
    headers: { "content-type": "application/json", ...headers },
  });
}

function rawPost(body: BodyInit, headers: Record<string, string> = {}) {
  return new Request("http://127.0.0.1/api/opencode-providers", {
    method: "POST",
    body,
    headers: { "content-type": "application/json", ...headers },
  });
}

// Mocked variant of ProvidersRouteDeps: each field is a vi.fn() mock, so tests can read
// `.mock.calls` off it — the interface itself stays a plain function signature (route.ts is
// production code, untouched here).
type ProvidersDepsMock = {
  snapshot: Mock<ProvidersRouteDeps["snapshot"]>;
  preview: Mock<ProvidersRouteDeps["preview"]>;
  apply: Mock<ProvidersRouteDeps["apply"]>;
  sanitize: Mock<ProvidersRouteDeps["sanitize"]>;
};

function deps(over: Partial<ProvidersDepsMock> = {}): ProvidersDepsMock {
  return {
    snapshot: vi.fn(async () => ({
      editor_revision: REV,
      connections: [{ id: "acme", apiKey: SENTINEL }],
    })),
    preview: vi.fn(async () => ({ affected: ["opencode.jsonc"], aliases: {} })),
    apply: vi.fn(async () => ({ effect: "published", editor_revision: REV, settings: null })),
    sanitize: vi.fn(() => [{ id: "acme", auth: "api-key" }]),
    ...over,
  };
}

function rows() {
  return testDb.prepare("SELECT id, ok, payload FROM mutations").all() as {
    id: number; ok: number | null; payload: string;
  }[];
}

describe("opencode-providers route", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
  });
  afterEach(() => {
    testDb.close();
  });

  it("is force-dynamic", () => {
    expect(dynamic).toBe("force-dynamic");
  });

  it("GET sanitizes connections and sets no-store", async () => {
    const d = deps();
    const res = await handleGet(new Request("http://127.0.0.1/api/opencode-providers"), d);
    expect(res.headers.get("cache-control")).toBe("no-store");
    const body = await res.json();
    expect(body).toEqual({
      ok: true,
      data: { editor_revision: REV, connections: [{ id: "acme", auth: "api-key" }] },
    });
    expect(JSON.stringify(body)).not.toContain(SENTINEL);
    expect(d.snapshot.mock.calls[0]).toEqual([]);
  });

  it("GET helper refusal stays 200", async () => {
    const res = await handleGet(new Request("http://127.0.0.1/api/opencode-providers"), deps({
      snapshot: vi.fn(async () => { throw new HelperRefusal("native-config-malformed"); }),
    }));
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ ok: false, error: "native-config-malformed" });
  });

  it("preview save-connection does not mutate", async () => {
    const d = deps();
    const body = await (await handlePost(post({
      action: "preview", expected: REV, kind: "save-connection", connection: CONN,
    }), d)).json();
    expect(body).toEqual({ ok: true, data: { affected: ["opencode.jsonc"], aliases: {} } });
    expect(d.preview.mock.calls[0]).toHaveLength(2);
    expect(d.apply).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);
  });

  it("refuses unsafe URLs, newline ids, jaxflow aliases, extra keys, unknown action, UTF-8, cap, same-site", async () => {
    const d = deps();
    const badUrls = [
      "http://api.example.com",
      "https://localhost",
      "https://127.0.0.1",
      "https://10.0.0.1",
      "https://user:pass@api.example.com",
      "https://api.example.com?x=1",
      "https://api.example.com#x",
    ];
    const cases: Request[] = [
      ...badUrls.map((base_url) => post({
        action: "save-connection", expected: REV, connection: { ...CONN, base_url },
      })),
      post({ action: "save-connection", expected: REV, connection: { ...CONN, id: "x\ny" } }),
      post({
        action: "save-model", expected: REV, connection: "acme",
        model: { id: "jaxflow-builder-x" },
      }),
      post({ action: "preview", expected: REV, kind: "save-connection", connection: CONN, extra: 1 }),
      post({ action: "drop-tables", expected: REV, connection: CONN }),
      post({ action: "preview", expected: REV, kind: "save-connection", connection: CONN }, { "sec-fetch-site": "same-site" }),
      rawPost(new Uint8Array([0xff])),
      rawPost("x".repeat(SETTINGS_CAP + 1), { "content-length": String(SETTINGS_CAP + 1) }),
    ];
    for (const req of cases) {
      expect((await (await handlePost(req, d)).json()).ok).toBe(false);
    }
    expect(d.preview).not.toHaveBeenCalled();
    expect(d.apply).not.toHaveBeenCalled();
    expect(rows()).toEqual([]);
  });

  it("save-connection / save-model / remove-model / remove-connection audit without spreading bodies", async () => {
    const d = deps();
    const actions = [
      { action: "save-connection", expected: REV, connection: CONN, operation_id: OP },
      {
        action: "save-model",
        expected: REV,
        connection: "acme",
        model: {
          id: "foo",
          limit: { context: 1000, output: 100 },
          reasoning: true,
          tool_call: true,
          effort_template: "reasoning",
          efforts: ["high"],
        },
        operation_id: OP,
      },
      { action: "remove-model", expected: REV, connection: "acme", model: "foo", operation_id: OP },
      { action: "remove-connection", expected: REV, connection: "acme", operation_id: OP },
    ];
    for (const payload of actions) {
      testDb.close();
      testDb = openDb(":memory:");
      const body = await (await handlePost(post(payload), d)).json();
      expect(body).toMatchObject({ ok: true, data: { effect: "published" } });
      expect(d.apply.mock.calls.at(-1)).toHaveLength(3);
      const row = JSON.parse(rows()[0].payload) as Record<string, unknown>;
      expect(row).toMatchObject({
        kind: `opencode-providers-${payload.action}`,
        action: payload.action,
        operation_id: OP,
        outcome: "done",
      });
      expect(row).not.toHaveProperty("connection");
      expect(row).not.toHaveProperty("model");
    }
  });

  it("both-profile removal and stale revision are helper codes", async () => {
    const conflict = deps({
      preview: vi.fn(async () => { throw new HelperRefusal("agent-profile-conflict"); }),
      apply: vi.fn(async () => { throw new HelperRefusal("agent-profile-conflict"); }),
    });
    expect(await (await handlePost(post({
      action: "preview", expected: REV, kind: "remove-connection", connection: "acme",
    }), conflict)).json()).toEqual({ ok: false, error: "agent-profile-conflict" });
    expect(rows()).toEqual([]);

    const stale = deps({
      apply: vi.fn(async () => { throw new HelperRefusal("agent-settings-source-changed"); }),
    });
    expect(await (await handlePost(post({
      action: "remove-connection", expected: REV, connection: "acme", operation_id: OP,
    }), stale)).json()).toMatchObject({
      code: "mutation-rejected", error: "agent-settings-source-changed", effect: "not-applied",
    });
  });

  it("activation-pending, audit-unavailable, uncertain, applied-but-audit-pending", async () => {
    const pending = deps({
      apply: vi.fn(async () => ({ effect: "activation-pending", editor_revision: REV, settings: null })),
    });
    expect(await (await handlePost(post({
      action: "save-connection", expected: REV, connection: CONN, operation_id: OP,
    }), pending)).json()).toMatchObject({
      ok: false, code: "activation-pending", effect: "activation-pending",
    });

    testDb.close();
    testDb = openDb(":memory:");
    testDb.exec("CREATE TRIGGER fail_ins BEFORE INSERT ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const skip = deps();
    expect(await (await handlePost(post({
      action: "save-connection", expected: REV, connection: CONN, operation_id: OP,
    }), skip)).json()).toMatchObject({ code: "audit-unavailable" });
    expect(skip.apply).not.toHaveBeenCalled();

    testDb.close();
    testDb = openDb(":memory:");
    const boom = deps({ apply: vi.fn(async () => { throw new Error("disk full"); }) });
    expect(await (await handlePost(post({
      action: "save-connection", expected: REV, connection: CONN, operation_id: OP,
    }), boom)).json()).toMatchObject({ code: "mutation-unconfirmed", effect: "unconfirmed" });

    testDb.close();
    testDb = openDb(":memory:");
    testDb.exec("CREATE TRIGGER fail_upd BEFORE UPDATE ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const fin = deps();
    const body = await (await handlePost(post({
      action: "save-connection", expected: REV, connection: CONN, operation_id: OP,
    }), fin)).json();
    expect(body).toMatchObject({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
    });
    expect(body.data).toMatchObject({ effect: "published" });
  });

  it("GET no longer reads provider-credential mutations (MOA-498 D4 — the async BWS reconciliation this fed is gone)", async () => {
    // Regression pin: importing the route module must not pull in server/db/provider-operations
    // at all, since that file no longer exists after this task. Path is relative to THIS test
    // file (src/app/api/opencode-providers/route.test.ts) — 3 levels up reaches src/, not 4.
    const mod = await import("node:fs");
    expect(mod.existsSync(new URL("../../../server/db/provider-operations.ts", import.meta.url))).toBe(false);
  });
});

describe("route-to-helper model form", () => {
  it("passes the exact shared form payload to the helper runner", async () => {
    const seen: { op: string; body: { intent?: unknown } }[] = [];
    const run: import("../../../server/collectors/agent-settings").HelperRunner = (op, body) => {
      seen.push({ op, body });
      if (op === "preview") return Promise.resolve({ affected: ["opencode.jsonc"], aliases: {} });
      return Promise.resolve({ effect: "published", editor_revision: REV, settings: null });
    };
    const d = deps({
      apply: vi.fn((intent, expected, op) => applyProviders(intent, expected, op, {}, run)),
    });
    testDb = openDb(":memory:");
    const model = {
      id: "reasoning-model",
      label: "Reasoning",
      limit: { context: 4096, output: 512 },
      reasoning: true,
      tool_call: true,
      effort_template: "reasoning" as const,
      efforts: ["high", "medium"],
    };
    const res = await handlePost(post({
      action: "save-model",
      expected: REV,
      connection: "acme",
      model,
      operation_id: OP,
    }), d);
    expect(res.status).toBe(200);
    expect((await res.json()).ok).toBe(true);
    const applyCall = seen.find((c) => c.op === "apply");
    expect(applyCall).toBeDefined();
    expect((applyCall!.body as { intent?: { model?: unknown } }).intent).toMatchObject({ model });
  });

  it("route parsing must match every rejected exact shared payload", async () => {
    const d = deps();
    const badModels = [
      { id: "x", effort_template: "reasoning" },
      { id: "x", limit: { context: 0, output: 1 }, tool_call: true, effort_template: "none" },
      { id: "x", reasoning: true, tool_call: true, effort_template: "reasoning", efforts: [] },
      { id: "x", extra: 1 },
      { id: "x", efforts: ["high", "high"], reasoning: true, tool_call: true, effort_template: "reasoning", limit: { context: 1, output: 1 } },
    ];
    for (const model of badModels) {
      testDb.close();
      testDb = openDb(":memory:");
      const res = await handlePost(post({
        action: "save-model", expected: REV, connection: "acme", model, operation_id: OP,
      }), d);
      expect((await res.json()).ok).toBe(false);
    }
    expect(d.apply).not.toHaveBeenCalled();
  });
});

describe("production boundary fixture against a temporary HOME", { timeout: 60_000 }, () => {
  let home: string;
  const OPENROUTER = "@openrouter/ai-sdk-provider";
  const CONN = "fixture";
  const MODEL = "wire-reasoning-model";
  const SOURCE = `{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "${CONN}": {
      "npm": "${OPENROUTER}",
      "models": {
        "${MODEL}": {
          "id": "${MODEL}",
          "name": "{env:LANG}",
          "limit": {"context": 32000, "output": 4096},
          "variants": {"high": {"reasoning": {"effort": "high"}}}
        }
      },
      "options": { "apiKey": "sk-fixture-native-key" },
      "whitelist": ["${MODEL}"]
    }
  },
  "unrelated": { "keep": true }
}
`;

  function providerDeps(): ProvidersRouteDeps {
    const run = helperRun();
    return {
      snapshot: () => snapshotProviders({}, run),
      preview: (intent, expected) => previewProviders(intent, expected, {}, run),
      apply: (intent, expected, operationId) => applyProviders(intent, expected, operationId, {}, run),
      sanitize: sanitizeConnections,
    };
  }

  function childEnv(): Record<string, string> {
    const xdgRoot = join(home, "xdg");
    return {
      HOME: home,
      // Ambient values of these three survive the {...process.env, ...childEnv()} spread and
      // win over HOME, so pin them inside this test's own home fixture.
      // JAXOS_HOME is the state dir itself (jaxflow_env.jaxos_home(): JAXOS_HOME overrides the
      // default ~/.jax-os), and this fixture keeps its state in <home>/.jax-os (beforeEach and
      // the persisted-file assertion below), so it must point there, not at <home>.
      JAXOS_HOME: join(home, ".jax-os"),
      CODEX_HOME: home,
      CLAUDE_CONFIG_DIR: home,
      XDG_CONFIG_HOME: join(home, ".config"),
      XDG_DATA_HOME: join(xdgRoot, "data"),
      XDG_CACHE_HOME: join(xdgRoot, "cache"),
      XDG_STATE_HOME: join(xdgRoot, "state"),
      OPENCODE_CONFIG: join(home, ".config", "opencode", "opencode.json"),
      OPENCODE_DISABLE_DEFAULT_PLUGINS: "1",
      OPENCODE_DISABLE_EXTERNAL_SKILLS: "1",
      OPENCODE_DISABLE_CLAUDE_CODE_SKILLS: "1",
      NO_COLOR: "1",
      TERM: "dumb",
      LANG: "C.UTF-8",
    };
  }

  function helperRun(): import("../../../server/collectors/agent-settings").HelperRunner {
    const env = { ...process.env, ...childEnv() };
    const spawnImpl = ((file: string, args: string[], options: { stdio: ["pipe", "pipe", "pipe"] }) =>
      spawn(file, args, { ...options, env })) as unknown as SpawnImpl;
    return (op, body) => runSettingsHelper(op, body, spawnImpl);
  }

  function spawnPython(script: string): { code: number; stdout: string; stderr: string } {
    const repo = join(dirname(fileURLToPath(import.meta.url)), "../../../..");
    const res = spawnSync("python3", ["-c", script], {
      cwd: repo,
      env: { ...process.env, ...childEnv(), PYTHONPATH: join(repo, "scripts") },
      encoding: "utf8",
      timeout: 30_000,
    });
    return { code: res.status ?? -1, stdout: res.stdout ?? "", stderr: res.stderr ?? "" };
  }

  function snapshotBody() {
    return handleGet(new Request("http://127.0.0.1/api/opencode-providers"), providerDeps()).then((r) => r.json());
  }

  beforeEach(() => {
    testDb = openDb(":memory:");
    home = mkdtempSync(join(tmpdir(), "jaxos-boundary-"));
    mkdirSync(join(home, ".config", "opencode"), { recursive: true });
    mkdirSync(join(home, ".jax-os"), { recursive: true });
    writeFileSync(join(home, ".config", "opencode", "opencode.json"), SOURCE);
  });
  afterEach(() => {
    rmSync(home, { recursive: true, force: true });
    testDb.close();
  });

  it("uses the real helper spawn to discover native registration and publish a reasoning model", async () => {
    const res = await handleGet(new Request("http://127.0.0.1/api/opencode-providers"), providerDeps());
    const body = await res.json();
    expect(body.ok).toBe(true);
    const rows = body.data.connections as Array<{ id: string; adapter: string; auth: string; health: string; models: unknown[]; editable: boolean; reason: string | null }>;
    const conn = rows.find((r) => r.id === CONN);
    expect(conn).toBeTruthy();
    expect(conn!.adapter).toBe("openrouter");
    expect(conn!.auth).toBe("api-key");
    expect(conn!.health).toBe("registered");
    expect(conn!.editable).toBe(true);
    expect(conn!.reason).toBeNull();
    expect(JSON.stringify(conn)).not.toContain("sk-fixture-native-key");
    expect(JSON.stringify(conn!.models)).toContain(MODEL);
    expect(conn!.models.some((m) => (m as { id: string }).id === MODEL)).toBe(true);
  });

  it("save-model through the real helper writes localized native state and shows the new choice", async () => {
    const model = {
      id: "reasoning-plus",
      label: "Reasoning Plus",
      limit: { context: 64000, output: 8192 },
      reasoning: true,
      tool_call: true,
      effort_template: "reasoning" as const,
      efforts: ["high", "medium"],
    };
    const initial = (await snapshotBody()) as { ok: boolean; data?: { editor_revision?: unknown } };
    const expected = initial.data!.editor_revision;
    const saveRes = await handlePost(post({
      action: "save-model",
      expected,
      connection: CONN,
      model,
      operation_id: OP,
    }), providerDeps());
    expect(saveRes.status).toBe(200);
    expect((await saveRes.json()).ok).toBe(true);
    const edited = readFileSync(join(home, ".config", "opencode", "opencode.json"), "utf8");
    expect(edited).toContain("reasoning-plus");
    const after = (await snapshotBody()) as {
      ok: boolean;
      data?: { connections: Array<{ models: ModelChoice[] }> };
    };
    expect(JSON.stringify(after.data!.connections)).not.toContain("sk-fixture-native-key");
    const conn = after.data!.connections.find((row) => row.models.some((m) => m.id === "reasoning-plus"));
    expect(conn?.models.map((m) => m.id)).toContain("reasoning-plus");
    const added = conn?.models.find((m) => m.id === "reasoning-plus");
    expect(added?.context).toBe(64000);
    expect(added?.effort_template).toBe("reasoning");
    expect(JSON.stringify(added)).not.toContain("8192");
  });

  it("provider preview runs through the real helper without mutating native state", async () => {
    const initial = (await snapshotBody()) as { ok: boolean; data?: { editor_revision?: unknown } };
    const expected = initial.data!.editor_revision;
    const res = await handlePost(post({
      action: "preview",
      expected,
      kind: "save-model",
      connection: CONN,
      model: {
        id: "preview-only",
        limit: { context: 4096, output: 512 },
        reasoning: true,
        tool_call: true,
        effort_template: "reasoning" as const,
        efforts: ["high"],
      },
    }), providerDeps());
    const body = (await res.json()) as { ok: boolean; data?: { affected?: string[] } };
    expect(body.ok).toBe(true);
    expect(body.data?.affected).toContain("opencode.json");
    expect(readFileSync(join(home, ".config", "opencode", "opencode.json"), "utf8")).not.toContain("preview-only");
  });

  it("saves both profiles through the agent-settings route and reads them back with the shared resolver", async () => {
    const run = helperRun();
    const settingsDeps: AgentSettingsRouteDeps = {
      snapshot: () => import("../../../server/collectors/agent-settings").then((m) => m.snapshotAgentSettings({}, run)),
      preview: (intent, expected) => import("../../../server/collectors/agent-settings").then((m) => m.previewAgentSettings(intent, expected, {}, run)),
      apply: (intent, expected, op) => import("../../../server/collectors/agent-settings").then((m) => m.applyAgentSettings(intent, expected, op, {}, run)),
      sanitize: sanitizeConnections,
    };

    const withDeps = (init?: RequestInit) => {
      const req = new Request("http://127.0.0.1/api/agent-settings", init);
      (req as Request & { jaxDeps?: unknown }).jaxDeps = settingsDeps;
      return req;
    };
    const boot = await SettingsGet(withDeps());
    const bootBody = (await boot.json()) as {
      ok: boolean;
      data: { editor_revision?: unknown; connections?: Array<{ id: string; models: Array<{ id: string }> }> };
    };
    expect(bootBody.ok).toBe(true);
    const revision = bootBody.data.editor_revision;

    const routingOr = { sort: "price" as const, allow_fallbacks: true };
    const draft = {
      reviewers: { claude: { model: "sonnet", effort: "xhigh" }, codex: { model: "gpt-5.6-luna", effort: "xhigh" } },
      builders: {
        default: { connection: CONN, model: MODEL, effort: "high", credential: { kind: "native" }, routing: routingOr },
        fallback: { connection: CONN, model: MODEL, effort: "high", credential: { kind: "native" }, routing: routingOr },
      },
    };

    const previewRes = await SettingsPost(withDeps({
      method: "POST",
      body: JSON.stringify({ action: "preview", expected: revision, reviewers: draft.reviewers, builders: draft.builders }),
      headers: { "content-type": "application/json" },
    }));
    const previewBody = (await previewRes.json()) as { ok: boolean; data?: { affected?: string[]; aliases?: { after?: Array<{ effort?: string | null; model?: string }> } } };
    expect(previewBody.ok).toBe(true);
    expect(previewBody.data?.affected).toContain("opencode.json");

    const saveRes = await SettingsPost(withDeps({
      method: "POST",
      body: JSON.stringify({ action: "save", expected: revision, reviewers: draft.reviewers, builders: draft.builders, operation_id: OP2 }),
      headers: { "content-type": "application/json" },
    }));
    expect((await saveRes.json()).ok).toBe(true);

    const after = await SettingsGet(withDeps());
    const afterBody = (await after.json()) as {
      ok: boolean;
      data?: { settings?: { builders?: Record<string, { model: string; effort: string | null }> } | null; editor_revision?: { settings?: string } };
    };
    expect(afterBody.ok).toBe(true);
    expect(afterBody.data?.settings?.builders?.default.model).toBe(MODEL);
    expect(afterBody.data?.settings?.builders?.fallback.model).toBe(MODEL);
    expect(afterBody.data?.settings?.builders?.default.effort).toBe("high");
    expect(afterBody.data?.editor_revision?.settings).toMatch(/^[0-9a-f]{32}$/);

    const persisted = JSON.parse(readFileSync(join(home, ".jax-os", "agent-settings.json"), "utf8")) as { builders: unknown };
    expect(persisted.builders).toBeTruthy();
    expect(JSON.stringify(afterBody)).not.toContain("sk-fixture-native-key");
  });

  it("publishes an uncatalogued model as a builder profile through the real helper (F1)", async () => {
    const initial = (await snapshotBody()) as { ok: boolean; data?: { editor_revision?: unknown } };
    expect(initial.ok).toBe(true);
    const saveRes = await handlePost(post({
      action: "save-model",
      expected: initial.data!.editor_revision,
      connection: CONN,
      model: { id: "custom-uncatalogued", label: "Custom Uncatalogued", context: 5000 },
      operation_id: OP,
    }), providerDeps());
    expect((await saveRes.json()).ok).toBe(true);

    const persisted = JSON.parse(readFileSync(join(home, ".config", "opencode", "opencode.json"), "utf8")) as {
      provider: Record<string, { models: Record<string, { id?: unknown; limit?: unknown }> }>;
    };
    const entry = persisted.provider[CONN].models["custom-uncatalogued"];
    expect(entry.id).toBe("custom-uncatalogued");
    expect(entry.limit).toEqual({ context: 5000, output: 0 });

    const afterModel = (await snapshotBody()) as {
      ok: boolean;
      data?: {
        editor_revision?: unknown;
        connections?: Array<{ id: string; models: Array<{ id: string; effort_template: string | null; no_effort: boolean }> }>;
      };
    };
    expect(afterModel.ok).toBe(true);
    const chosen = afterModel.data!.connections!.find((r) => r.id === CONN)!.models.find((m) => m.id === "custom-uncatalogued")!;
    expect(chosen.effort_template).toBeNull();
    expect(chosen.no_effort).toBe(true);

    const routing = { sort: "price" as const, allow_fallbacks: true };
    const draft = {
      reviewers: { claude: { model: "sonnet", effort: "xhigh" }, codex: { model: "gpt-5.6-luna", effort: "xhigh" } },
      builders: {
        default: { connection: CONN, model: "custom-uncatalogued", effort: null, credential: { kind: "native" }, routing },
        fallback: { connection: CONN, model: MODEL, effort: "high", credential: { kind: "native" }, routing },
      },
    };
    const run = helperRun();
    const settingsDeps: AgentSettingsRouteDeps = {
      snapshot: () => import("../../../server/collectors/agent-settings").then((m) => m.snapshotAgentSettings({}, run)),
      preview: (intent, expected) => import("../../../server/collectors/agent-settings").then((m) => m.previewAgentSettings(intent, expected, {}, run)),
      apply: (intent, expected, op) => import("../../../server/collectors/agent-settings").then((m) => m.applyAgentSettings(intent, expected, op, {}, run)),
      sanitize: sanitizeConnections,
    };
    const withDeps = (init?: RequestInit) => {
      const req = new Request("http://127.0.0.1/api/agent-settings", init);
      (req as Request & { jaxDeps?: unknown }).jaxDeps = settingsDeps;
      return req;
    };
    const saveProfiles = await SettingsPost(withDeps({
      method: "POST",
      body: JSON.stringify({
        action: "save",
        expected: afterModel.data!.editor_revision,
        reviewers: draft.reviewers,
        builders: draft.builders,
        operation_id: OP2,
      }),
      headers: { "content-type": "application/json" },
    }));
    expect((await saveProfiles.json()).ok).toBe(true);

    const py = spawnPython(`import json
from pathlib import Path
import os
import jaxflow_settings as settings_io
from jaxflow import _pure_config_run

settings = settings_io.read_settings(Path.home() / ".jax-os" / "agent-settings.json")
effective = settings_io.read_effective_opencode_config(run=_pure_config_run, env=dict(os.environ), repo=Path.cwd())
rows = []
for name in ("default", "fallback"):
    res = settings_io.resolve_profile(settings, name, effective_config=effective)
    rows.append({"profile": name, "runtime_model": res["runtime_model"], "model": res["model"], "effort": res["effort"]})
print(json.dumps(rows))`);
    expect(py.code).toBe(0);
    const resolved = JSON.parse(py.stdout.trim().split("\n").pop()!) as Array<{
      profile: string; runtime_model: string; model: string; effort: string | null;
    }>;
    const def = resolved.find((r) => r.profile === "default")!;
    expect(def.runtime_model).toBe(`${CONN}/jaxflow-builder-default`);
    expect(def.model).toBe(`${CONN}/custom-uncatalogued`);
    expect(def.effort).toBeNull();
    const fallback = resolved.find((r) => r.profile === "fallback")!;
    expect(fallback.runtime_model).toBe(`${CONN}/jaxflow-builder-fallback`);
    expect(fallback.model).toBe(`${CONN}/${MODEL}`);
    expect(fallback.effort).toBe("high");
  });

  it("native-CLI discovery, shared model payload and Part 1 resolution close F2/F4", async () => {
    const cliEnv = { ...process.env, ...childEnv() };
    const configOut = spawnSync("opencode", ["debug", "config", "--pure"], { env: cliEnv, encoding: "utf8", timeout: 30_000 });
    expect(configOut.status).toBe(0);
    expect(configOut.stdout).toContain(`"${CONN}"`);

    const modelsOut = spawnSync("opencode", ["models", CONN, "--verbose", "--pure"], { env: cliEnv, encoding: "utf8", timeout: 30_000 });
    expect(modelsOut.status).toBe(0);
    const verbose = modelsOut.stdout;
    expect(verbose).toContain("wire-reasoning-model");
    expect(verbose).toContain("capabilities");
    expect(verbose).toContain("toolcall");

    const resolvedConfig = JSON.parse(configOut.stdout) as { provider: Record<string, { models: Record<string, { name?: unknown }> }> };
    const resolvedName = resolvedConfig.provider[CONN].models[MODEL].name;
    expect(resolvedName).toBe("C.UTF-8");
    const sourceModel = (JSON.parse(SOURCE) as { provider: Record<string, { models: Record<string, { name?: unknown }> }> })
      .provider[CONN].models[MODEL];
    expect(sourceModel.name).toBe("{env:LANG}");
    const catalogStart = (await snapshotBody()) as { data?: { editor_revision?: unknown } };
    const catalogSave = await handlePost(post({
      action: "save-model", expected: catalogStart.data!.editor_revision, connection: CONN,
      model: { id: MODEL, label: resolvedName }, operation_id: "cccccccc-dddd-4eee-8fff-000000000000",
    }), providerDeps());
    expect((await catalogSave.json()).ok).toBe(true);
    const persistedCatalog = JSON.parse(readFileSync(join(home, ".config", "opencode", "opencode.json"), "utf8")) as {
      provider: Record<string, { models: Record<string, { name?: unknown; variants?: unknown; limit?: unknown }> }>;
    };
    // The real CLI normalization crosses the helper boundary as the client-supplied label,
    // and the narrow edit replaces only the name without dropping native metadata.
    expect(persistedCatalog.provider[CONN].models[MODEL].name).toBe(resolvedName);
    expect(persistedCatalog.provider[CONN].models[MODEL].name).not.toBe(sourceModel.name);
    expect(persistedCatalog.provider[CONN].models[MODEL].variants).toEqual({ high: { reasoning: { effort: "high" } } });
    expect(persistedCatalog.provider[CONN].models[MODEL].limit).toEqual({ context: 32000, output: 4096 });
    const catalogView = (await snapshotBody()) as { data?: { connections?: Array<{ id: string; models: Array<{ id: string; label?: string }> }> } };
    expect(catalogView.data!.connections!.find((r) => r.id === CONN)!.models.find((m) => m.id === MODEL)!.label).toBe("C.UTF-8");

    const initial = (await snapshotBody()) as {
      ok: boolean;
      data?: { editor_revision?: unknown; connections?: Array<{ id: string; adapter: string; auth: string; health: string; models: Array<{ id: string }>; reason: string | null }> };
    };
    expect(initial.ok).toBe(true);
    const conn = initial.data!.connections!.find((r) => r.id === CONN);
    expect(conn).toBeTruthy();
    expect(conn!.adapter).toBe("openrouter");
    expect(conn!.auth).toBe("api-key");
    expect(conn!.health).toBe("registered");
    expect(JSON.stringify(conn!.models)).not.toContain("jaxflow-builder-");
    expect(conn!.models.map((m) => m.id)).toEqual(expect.arrayContaining(["wire-reasoning-model"]));

    const model = modelInputPayload({
      id: "reasoning-cli",
      label: "Reasoning CLI",
      context: 4096,
      output: 512,
      reasoning: true,
      toolCall: true,
      effort: "reasoning",
      efforts: ["high", "medium"],
    });
    expect(model).not.toBeNull();
    const saveModel = await handlePost(post({
      action: "save-model", expected: initial.data!.editor_revision, connection: CONN,
      model: model!,
      operation_id: OP,
    }), providerDeps());
    expect((await saveModel.json()).ok).toBe(true);

    const afterModel = (await snapshotBody()) as {
      ok: boolean;
      data?: { editor_revision?: unknown; connections?: Array<{ id: string; models: Array<{ id: string }> }> };
    };
    expect(afterModel.ok).toBe(true);
    const connAfter = afterModel.data!.connections!.find((r) => r.id === CONN);
    expect(connAfter!.models.map((m) => m.id)).toContain("reasoning-cli");

    const routingDefault = { sort: "price" as const, allow_fallbacks: true };
    const routingFallback = { sort: "price" as const, allow_fallbacks: true, max_price: { prompt: 2 } };
    const revision = afterModel.data!.editor_revision;
    const draft = {
      reviewers: { claude: { model: "sonnet", effort: "xhigh" }, codex: { model: "gpt-5.6-luna", effort: "xhigh" } },
      builders: {
        default: { connection: CONN, model: "reasoning-cli", effort: "high", credential: { kind: "native" }, routing: routingDefault },
        fallback: { connection: CONN, model: "reasoning-cli", effort: "medium", credential: { kind: "native" }, routing: routingFallback },
      },
    };
    const run = helperRun();
    const settingsDeps: AgentSettingsRouteDeps = {
      snapshot: () => import("../../../server/collectors/agent-settings").then((m) => m.snapshotAgentSettings({}, run)),
      preview: (intent, expected) => import("../../../server/collectors/agent-settings").then((m) => m.previewAgentSettings(intent, expected, {}, run)),
      apply: (intent, expected, op) => import("../../../server/collectors/agent-settings").then((m) => m.applyAgentSettings(intent, expected, op, {}, run)),
      sanitize: sanitizeConnections,
    };
    const withDeps = (init?: RequestInit) => {
      const req = new Request("http://127.0.0.1/api/agent-settings", init);
      (req as Request & { jaxDeps?: unknown }).jaxDeps = settingsDeps;
      return req;
    };
    const previewRes = await SettingsPost(withDeps({
      method: "POST",
      body: JSON.stringify({ action: "preview", expected: revision, reviewers: draft.reviewers, builders: draft.builders }),
      headers: { "content-type": "application/json" },
    }));
    expect((await previewRes.json()).ok).toBe(true);
    const saveRes = await SettingsPost(withDeps({
      method: "POST",
      body: JSON.stringify({ action: "save", expected: revision, reviewers: draft.reviewers, builders: draft.builders, operation_id: OP2 }),
      headers: { "content-type": "application/json" },
    }));
    const saveBody = await saveRes.json();
    expect(saveBody.ok).toBe(true);

    const py = spawnPython(`
import json
from pathlib import Path
import os
import jaxflow_settings as settings_io
from jaxflow import _pure_config_run

settings = settings_io.read_settings(Path.home() / ".jax-os" / "agent-settings.json")
effective = settings_io.read_effective_opencode_config(run=_pure_config_run, env=dict(os.environ), repo=Path.cwd())
rows = []
for name in ("default", "fallback"):
    res = settings_io.resolve_profile(settings, name, effective_config=effective)
    conn = settings["builders"][name]["connection"]
    provider = effective["provider"][conn]
    native_model = provider["models"].get("jaxflow-builder-" + name)
    routing = None
    if native_model is not None and isinstance(native_model.get("options"), dict):
        routing = native_model["options"].get("provider")
    rows.append({
        "profile": name,
        "runtime_model": res["runtime_model"],
        "model": res["model"],
        "effort": res["effort"],
        "routing": routing,
        "saved_routing": settings["builders"][name]["routing"],
    })
print(json.dumps(rows))
`);
    expect(py.code).toBe(0);
    const rows = JSON.parse(py.stdout.trim().split("\n").pop()!) as Array<{
      profile: string; runtime_model: string; model: string; effort: string | null; routing: unknown; saved_routing: unknown;
    }>;
    const def = rows.find((r) => r.profile === "default")!;
    expect(def.runtime_model).toBe(`${CONN}/jaxflow-builder-default`);
    expect(def.model).toBe(`${CONN}/reasoning-cli`);
    expect(def.effort).toBe("high");
    expect(def.routing).toEqual(routingDefault);
    expect(def.saved_routing).toEqual(routingDefault);
    const fallback = rows.find((r) => r.profile === "fallback")!;
    expect(fallback.runtime_model).toBe(`${CONN}/jaxflow-builder-fallback`);
    expect(fallback.model).toBe(`${CONN}/reasoning-cli`);
    expect(fallback.effort).toBe("medium");
    expect(fallback.routing).toEqual(routingFallback);
    expect(fallback.saved_routing).toEqual(routingFallback);
  });

  it("snapshot reports the new env-kind binding for an env-referenced apiKey (MOA-498 D4)", async () => {
    const withEnvRef = JSON.parse(SOURCE) as { provider: Record<string, { options?: Record<string, unknown> }> };
    withEnvRef.provider[CONN].options = { apiKey: "{env:JAX_PROVIDER_FIXTURE_API_KEY}" };
    writeFileSync(join(home, ".config", "opencode", "opencode.json"), JSON.stringify(withEnvRef, null, 2));
    const saved = process.env.JAX_PROVIDER_FIXTURE_API_KEY;
    process.env.JAX_PROVIDER_FIXTURE_API_KEY = "sk-fixture-env-value";
    try {
      const body = (await snapshotBody()) as {
        ok: boolean;
        data?: { connections?: Array<{ id: string; health: string; credential?: unknown }> };
      };
      expect(body.ok).toBe(true);
      const row = body.data!.connections!.find((r) => r.id === CONN);
      expect(row!.credential).toEqual({ kind: "env", env: "JAX_PROVIDER_FIXTURE_API_KEY" });
      expect(row!.health).toBe("registered");
      expect(JSON.stringify(body)).not.toContain("sk-fixture-env-value");
    } finally {
      if (saved === undefined) delete process.env.JAX_PROVIDER_FIXTURE_API_KEY;
      else process.env.JAX_PROVIDER_FIXTURE_API_KEY = saved;
    }
  });

  it("bind-credential validates {connection, credential} and reaches deps.apply (MOA-498 D4/12)", async () => {
    const d = deps();
    const res = await handlePost(post({
      action: "bind-credential",
      expected: REV,
      connection: "acme",
      credential: { kind: "env", env: "JAX_PROVIDER_ACME_API_KEY" },
    }), d);
    expect(d.apply).toHaveBeenCalledWith(
      { kind: "bind-credential", connection: "acme", credential: { kind: "env", env: "JAX_PROVIDER_ACME_API_KEY" } },
      REV,
      expect.any(String),
    );
    expect(await res.json()).toMatchObject({ ok: true });
  });

  it("bind-credential refuses the retired bws-env shape and a malformed connection id (MOA-498 D4/12)", async () => {
    const d = deps();
    const res1 = await handlePost(post({
      action: "bind-credential", expected: REV, connection: "acme",
      credential: { kind: "bws-env", env: "X", secret_id: "y" },
    }), d);
    expect(await res1.json()).toMatchObject({ ok: false });
    const res2 = await handlePost(post({
      action: "bind-credential", expected: REV, connection: "not valid!",
      credential: { kind: "native" },
    }), d);
    expect(await res2.json()).toMatchObject({ ok: false });
    expect(d.apply).not.toHaveBeenCalled();
  });

  describe("GET ?credential-env (MOA-498 D4/12 — configured/missing status, never a value)", () => {
    it("refuses an invalid env name before any file read", async () => {
      const res = await GET(new Request("http://127.0.0.1/api/opencode-providers?credential-env=not%20valid"));
      expect(res.headers.get("cache-control")).toBe("no-store");
      expect(await res.json()).toEqual({ ok: false, error: "invalid-env-name" });
    });

    it("reports configured/missing from $JAXOS_HOME/.env", async () => {
      const dir = mkdtempSync(join(tmpdir(), "jax-cred-env-"));
      writeFileSync(join(dir, ".env"), "JAX_PROVIDER_ACME_API_KEY=sk-live\n", { mode: 0o600 });
      const saved = process.env.JAXOS_HOME;
      process.env.JAXOS_HOME = dir;
      try {
        const res = await GET(new Request("http://127.0.0.1/api/opencode-providers?credential-env=JAX_PROVIDER_ACME_API_KEY"));
        expect(await res.json()).toEqual({ ok: true, data: { state: "configured" } });
        const res2 = await GET(new Request("http://127.0.0.1/api/opencode-providers?credential-env=JAX_PROVIDER_OTHER_API_KEY"));
        expect(await res2.json()).toEqual({ ok: true, data: { state: "missing" } });
      } finally {
        if (saved === undefined) delete process.env.JAXOS_HOME;
        else process.env.JAXOS_HOME = saved;
        rmSync(dir, { recursive: true, force: true });
      }
    });
  });

});
