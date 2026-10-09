import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { execFile } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type Database from "better-sqlite3";

let testDb: Database.Database;
vi.mock("../../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});
vi.mock("../../../../server/collectors/agent-settings", () => ({ snapshotAgentSettings: vi.fn() }));
vi.mock("../../../../server/db/provider-operations", () => ({ readCredentialBindings: vi.fn(() => []) }));

import { openDb } from "../../../../server/db";
import { snapshotAgentSettings } from "../../../../server/collectors/agent-settings";
import {
  applyInventory,
  getInventoryData,
  inventoryRevision,
  prepareInventory,
  previewInventory,
  reconcileInventory,
  runPython,
  type HelperRun,
  type InventoryDeps,
} from "../../../../server/collectors/inventory";
import { GET, POST } from "./route";
import { readInventoryOperation, readInventoryOperationPayload } from "../../../../server/db/inventory-operations";
import type { InventoryData, InventoryItem } from "../../../../server/collectors/inventory";

const OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";
const OP2 = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff";
const ITEM_ID = "a".repeat(64);
const REV = "b".repeat(64);
const CAND = "c".repeat(64);

function item(overrides: Partial<InventoryItem> = {}): InventoryItem {
  return {
    id: ITEM_ID,
    executor: "claude",
    kind: "plugin",
    registration: "alpha@mp",
    name: "alpha",
    description: "",
    path: "/c/settings.json",
    scope: "user",
    origin: "settings",
    parent: null,
    canonicalTarget: null,
    state: "enabled",
    capabilities: {},
    ...overrides,
  };
}

function data(overrides: Partial<InventoryItem> = {}): InventoryData {
  return {
    executors: {
      claude: { ok: true, items: [item(overrides)] },
      codex: { ok: true, items: [] },
      opencode: { ok: true, items: [] },
    },
  };
}

function request(method: string, payload?: unknown, deps?: unknown, url = "http://127.0.0.1/api/tools/inventory") {
  const req = new Request(url, {
    method,
    ...(payload !== undefined ? { body: JSON.stringify(payload), headers: { "content-type": "application/json" } } : {}),
  });
  if (deps) Object.defineProperty(req, "jaxDeps", { value: deps });
  return req;
}

function deps(overrides: Record<string, unknown> = {}) {
  return {
    getData: vi.fn(async () => data()),
    preview: vi.fn(async () => ({ itemId: ITEM_ID, change: "disable", revision: CAND, targets: [], effect: "disable plugin alpha", recovery: null, requiresNewSession: true })),
    prepare: vi.fn(async () => ({
      itemId: ITEM_ID,
      change: "disable",
      beforeDigest: REV,
      candidateDigest: CAND,
      provenance: { prior: true },
      settingsExpected: null,
      settingsCandidate: null,
      executor: "claude",
      selector: "alpha@mp",
      target: "/c/settings.json",
    })),
    apply: vi.fn(async () => ({ effect: "applied", candidateDigest: CAND, provenance: { prior: true }, requiresNewSession: true })),
    reconcile: vi.fn(async () => ({ effect: "published" })),
    revision: vi.fn(async () => ({ revision: CAND, settings: null })),
    ...overrides,
  };
}

describe("inventory route", () => {
  beforeEach(() => {
    testDb = openDb(":memory:");
    vi.mocked(snapshotAgentSettings).mockResolvedValue(undefined);
  });
  afterEach(() => {
    testDb.close();
    vi.clearAllMocks();
  });

  it("GET returns the inventory envelope", async () => {
    const res = await GET(request("GET", undefined, deps()));
    expect(res.status).toBe(200);
    expect((await res.json()).ok).toBe(true);
  });

  it("GET ?operation validates and never writes", async () => {
    expect((await GET(request("GET", undefined, deps(), "http://127.0.0.1/api/tools/inventory?operation=nope"))).status).toBe(400);
    expect((await GET(request("GET", undefined, deps(), `http://127.0.0.1/api/tools/inventory?operation=${OP}`))).status).toBe(404);
  });

  it("GET joins an unresolved operation onto its item (reload recovery)", async () => {
    await POST(
      request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP }, deps({
        apply: vi.fn(async () => ({ effect: "activation-pending", candidateDigest: CAND, editor_revision: { settings: "absent", sources: { "opencode.jsonc": "tok" } } })),
      })),
    );
    const res = await GET(request("GET", undefined, deps()));
    const body = await res.json();
    expect(body.data.executors.claude.items[0].operation).toMatchObject({ operationId: OP, stage: "activation-pending" });
  });

  it("POST preview is read-only and enforces exact keys/format", async () => {
    const bad = await POST(request("POST", { action: "preview", itemId: "nothex", change: "disable" }, deps()));
    expect(bad.status).toBe(400);
    const extra = await POST(request("POST", { action: "preview", itemId: ITEM_ID, change: "disable", path: "/x" }, deps()));
    expect(extra.status).toBe(400);
    const ok = await POST(request("POST", { action: "preview", itemId: ITEM_ID, change: "disable" }, deps()));
    expect(await ok.json()).toMatchObject({ ok: true, data: { revision: CAND } });
  });

  it("POST apply reserves audit before the effect, persists the candidate digest and settles done", async () => {
    const d = deps();
    const res = await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP }, d));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data.operationId).toBe(OP);
    expect(d.apply).toHaveBeenCalledOnce();
    const view = readInventoryOperation(testDb, OP);
    expect(view?.itemId).toBe(ITEM_ID);
    expect(view?.ok).toBe(true);
    const payload = JSON.parse((testDb.prepare("SELECT payload FROM mutations").get() as { payload: string }).payload);
    expect(payload).toMatchObject({ candidateDigest: CAND, stage: "applied", beforeDigest: REV });
  });

  it("POST apply with an existing operation ID returns its status without a new effect", async () => {
    const d = deps();
    await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP }, d));
    const again = await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP }, d));
    expect(await again.json()).toMatchObject({ ok: true, data: { operationId: OP, ok: true } });
    expect(d.apply).toHaveBeenCalledOnce();
  });

  it("POST recheck settles on the RECORDED candidate/before digests, not a coarse state", async () => {
    await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP }, deps()));
    const applied = await POST(
      request("POST", { action: "recheck", operationId: OP }, deps({ revision: vi.fn(async () => ({ revision: CAND, settings: null })) })),
    );
    expect(await applied.json()).toMatchObject({ ok: true, data: { stage: "applied" } });

    await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP2 }, deps()));
    const notApplied = await POST(
      request("POST", { action: "recheck", operationId: OP2 }, deps({ revision: vi.fn(async () => ({ revision: REV, settings: null })) })),
    );
    expect(await notApplied.json()).toMatchObject({ ok: true, data: { stage: "not-applied" } });

    const OP3 = "cccccccc-dddd-4eee-8fff-000000000000";
    await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP3 }, deps()));
    const conflict = await POST(
      request("POST", { action: "recheck", operationId: OP3 }, deps({ revision: vi.fn(async () => ({ revision: "d".repeat(64), settings: null })) })),
    );
    expect(await conflict.json()).toMatchObject({ ok: true, data: { stage: "conflict" } });
  });

  it("POST recheck returns a settled row without rewriting it from a failed readback", async () => {
    const d = deps();
    await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP }, d));
    const settled = await POST(request("POST", { action: "recheck", operationId: OP }, deps({ revision: vi.fn(async () => ({ revision: CAND, settings: null })) })));
    expect((await settled.json()).data).toMatchObject({ stage: "applied", ok: true });
    const again = await POST(request("POST", { action: "recheck", operationId: OP }, deps({ revision: vi.fn(async () => { throw new Error("source down"); }) })));
    expect((await again.json()).data).toMatchObject({ stage: "applied", ok: true });
    expect(readInventoryOperationPayload(testDb, OP)).toMatchObject({ settled: true, stage: "applied" });
  });

  it("POST reconcile refuses an operation that is not a recorded partial publication", async () => {
    await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP }, deps()));
    const res = await POST(request("POST", { action: "reconcile", operationId: OP, expectedRevision: REV }, deps()));
    expect(res.status).toBe(409);
  });

  it("POST preview-reconcile + reconcile use the recorded candidate, not a fabricated post-effect revision", async () => {
    const d = deps({
      prepare: vi.fn(async () => ({
        itemId: ITEM_ID,
        change: "disable",
        beforeDigest: REV,
        candidateDigest: CAND,
        provenance: { prior: true },
        settingsExpected: { settings: "s".repeat(64), sources: { "opencode.jsonc": "pre" } },
        settingsCandidate: { settings: "a".repeat(64), sources: { "opencode.jsonc": "candidate" } },
        executor: "opencode",
        selector: "writer",
        target: "/x/opencode.jsonc",
      })),
      // a partial publisher returns only its effect: no editor_revision to be
      // mistaken for a recorded candidate.
      apply: vi.fn(async () => ({ effect: "activation-pending", candidateDigest: CAND })),
      reconcile: vi.fn(async () => ({ effect: "published" })),
      revision: vi.fn(async () => ({ revision: CAND, settings: { settings: "s".repeat(64), sources: { "opencode.jsonc": "pre" } } })),
    });
    const applyRes = await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP }, d));
    expect(applyRes.status).toBe(200);
    const confirmation = await POST(request("POST", { action: "preview-reconcile", operationId: OP }, d));
    const revision = (await confirmation.json()).data.revision as string;
    expect(revision).toMatch(/^[0-9a-f]{64}$/);
    const ok = await POST(request("POST", { action: "reconcile", operationId: OP, expectedRevision: revision }, d));
    expect(await ok.json()).toMatchObject({ ok: true, data: { effect: "published" } });
    expect(d.reconcile).toHaveBeenCalledWith(
      { settings: "s".repeat(64), sources: { "opencode.jsonc": "candidate" } },
      OP,
    );
    // a stale/other confirmation revision is refused
    const stale = await POST(request("POST", { action: "reconcile", operationId: OP, expectedRevision: REV }, d));
    expect(stale.status).toBe(409);
  });

  it("rejects unknown actions and mixed/oversized input", async () => {
    expect((await POST(request("POST", { action: "nope" }, deps()))).status).toBe(400);
    expect((await POST(request("POST", { action: "apply", itemId: ITEM_ID, change: "disable", expectedRevision: REV, operationId: OP, query: "x" }, deps()))).status).toBe(400);
  });

  // Real boundary: route → collector → actual helper subprocess → temporary
  // filesystem → actual SQLite mutation row. Uses the worktree's isolated
  // helper venv when present.
  it("applies a real Claude toggle end-to-end over a temporary root", async () => {
    const venv = join(process.cwd(), ".local", "inventory-venv", "bin", "python");
    if (!existsSync(venv)) return;
    const root = mkdtempSync(join(tmpdir(), "jax-inv-route-"));
    const claude = join(root, "claude");
    mkdirSync(claude, { recursive: true });
    const settingsPath = join(claude, "settings.json");
    writeFileSync(settingsPath, JSON.stringify({ enabledPlugins: { "alpha@market": true } }), "utf8");
    const roots = { claude_dir: claude, codex_dir: join(root, "codex"), opencode_dir: join(root, "oc"), jax_os: join(root, "jax") };
    const realRun: HelperRun = (action, payload, python) => runPython(python, action, payload);
    const realDeps: InventoryDeps = { run: realRun, roots, venvPython: venv };
    const routeDeps = {
      getData: () => getInventoryData(realDeps),
      preview: (itemId: string, change: "disable", provenance?: unknown) => previewInventory(realDeps, itemId, change, provenance),
      prepare: (itemId: string, change: "disable", expectedRevision: string, provenance?: unknown) =>
        prepareInventory(realDeps, itemId, change, expectedRevision, provenance),
      apply: (itemId: string, change: "disable", revision: string, operationId: string, provenance?: unknown) =>
        applyInventory(realDeps, itemId, change, revision, operationId, provenance),
      reconcile: () => Promise.resolve({ effect: "published" }),
      revision: (itemId: string, executor?: string) => inventoryRevision(realDeps, itemId, executor),
    };
    const data = await routeDeps.getData();
    const item = (data.executors.claude.ok ? data.executors.claude.items : []).find((i) => i.registration === "alpha@market");
    expect(item).toBeDefined();
    const prev = await routeDeps.preview(item!.id, "disable");
    const res = await POST(
      request("POST", { action: "apply", itemId: item!.id, change: "disable", expectedRevision: prev.revision, operationId: OP }, routeDeps),
    );
    expect(res.status).toBe(200);
    expect(JSON.parse(readFileSync(settingsPath, "utf8")).enabledPlugins["alpha@market"]).toBe(false);
    expect(readInventoryOperation(testDb, OP)).toMatchObject({ itemId: item!.id, ok: true, stage: "applied" });
    // after the managed disable the served capability offers the inverse (F1)
    const after = await GET(request("GET", undefined, routeDeps));
    const afterItem = (await after.json()).data.executors.claude.items.find((i: InventoryItem) => i.id === item!.id);
    expect(afterItem.capabilities.enable).toEqual({ available: true, reason: null });
  });

  it("applies a real Codex toggle (tomlkit writer) end-to-end", async () => {
    const venv = join(process.cwd(), ".local", "inventory-venv", "bin", "python");
    if (!existsSync(venv)) return;
    const root = mkdtempSync(join(tmpdir(), "jax-inv-codex-"));
    const codex = join(root, "codex");
    mkdirSync(codex, { recursive: true });
    const configPath = join(codex, "config.toml");
    writeFileSync(configPath, '# keep this comment\n[plugins.alpha]\nenabled = true\nother = "x"\n', "utf8");
    const roots = { claude_dir: join(root, "claude"), codex_dir: codex, opencode_dir: join(root, "oc"), jax_os: join(root, "jax") };
    const realDeps: InventoryDeps = {
      run: (action, payload, python) => runPython(python, action, payload),
      roots,
      venvPython: venv,
    };
    const routeDeps = {
      getData: () => getInventoryData(realDeps),
      preview: (itemId: string, change: "disable", provenance?: unknown) => previewInventory(realDeps, itemId, change, provenance),
      prepare: (itemId: string, change: "disable", expectedRevision: string, provenance?: unknown) =>
        prepareInventory(realDeps, itemId, change, expectedRevision, provenance),
      apply: (itemId: string, change: "disable", revision: string, operationId: string, provenance?: unknown) =>
        applyInventory(realDeps, itemId, change, revision, operationId, provenance),
      reconcile: () => Promise.resolve({ effect: "published" }),
      revision: (itemId: string, executor?: string) => inventoryRevision(realDeps, itemId, executor),
    };
    const data = await routeDeps.getData();
    const item = (data.executors.codex.ok ? data.executors.codex.items : []).find((i) => i.registration === "alpha");
    const prev = await routeDeps.preview(item!.id, "disable");
    const res = await POST(
      request("POST", { action: "apply", itemId: item!.id, change: "disable", expectedRevision: prev.revision, operationId: OP }, routeDeps),
    );
    expect(res.status).toBe(200);
    const text = readFileSync(configPath, "utf8");
    expect(text).toContain("# keep this comment");
    expect(text).toContain('other = "x"');
    expect(text).toContain("enabled = false");
    expect(readInventoryOperation(testDb, OP)).toMatchObject({ ok: true, stage: "applied" });
  });

  // Joined managed-restore path: route → collector → actual helper subprocess →
  // temporary native files → actual SQLite. Disable persists the restoration
  // record before the effect; recheck settles from the recorded candidate; the
  // enable reuses the recorded provenance and restores the exact prior policy.
  it("recovers a real Claude skill managed restore through disable → recheck → enable", async () => {
    const venv = join(process.cwd(), ".local", "inventory-venv", "bin", "python");
    if (!existsSync(venv)) return;
    const root = mkdtempSync(join(tmpdir(), "jax-inv-managed-"));
    const claude = join(root, "claude");
    const skill = join(claude, "skills", "solo");
    mkdirSync(skill, { recursive: true });
    writeFileSync(join(skill, "SKILL.md"), "# solo\n", "utf8");
    const settingsPath = join(claude, "settings.json");
    writeFileSync(settingsPath, JSON.stringify({ skillOverrides: { solo: "name-only" } }), "utf8");
    const roots = { claude_dir: claude, codex_dir: join(root, "codex"), opencode_dir: join(root, "oc"), jax_os: join(root, "jax") };
    const realDeps: InventoryDeps = {
      run: (action, payload, python) => runPython(python, action, payload),
      roots,
      venvPython: venv,
    };
    const routeDeps = {
      getData: () => getInventoryData(realDeps),
      preview: (itemId: string, change: "disable", provenance?: unknown) => previewInventory(realDeps, itemId, change, provenance),
      prepare: (itemId: string, change: "disable", expectedRevision: string, provenance?: unknown) =>
        prepareInventory(realDeps, itemId, change, expectedRevision, provenance),
      apply: (itemId: string, change: "disable", revision: string, operationId: string, provenance?: unknown) =>
        applyInventory(realDeps, itemId, change, revision, operationId, provenance),
      reconcile: () => Promise.resolve({ effect: "published" }),
      revision: (itemId: string, executor?: string) => inventoryRevision(realDeps, itemId, executor),
    };
    const data = await routeDeps.getData();
    const item = (data.executors.claude.ok ? data.executors.claude.items : []).find((i) => i.registration === skill);
    expect(item).toBeDefined();
    const prev = await routeDeps.preview(item!.id, "disable");
    const disabled = await POST(
      request("POST", { action: "apply", itemId: item!.id, change: "disable", expectedRevision: prev.revision, operationId: OP }, routeDeps),
    );
    expect(disabled.status).toBe(200);
    // the restoration record is persisted BEFORE the recheck, not only in the response
    const payload = JSON.parse((testDb.prepare("SELECT payload FROM mutations ORDER BY id DESC LIMIT 1").get() as { payload: string }).payload);
    expect(payload).toMatchObject({ candidateDigest: expect.any(String), beforeDigest: expect.any(String) });
    expect(payload.provenance).toMatchObject({ selector: "solo", prior: "name-only" });
    // recheck reads the native candidate directly and settles applied
    const rechecked = await POST(request("POST", { action: "recheck", operationId: OP }, routeDeps));
    expect((await rechecked.json()).data).toMatchObject({ stage: "applied" });
    expect(readInventoryOperation(testDb, OP)).toMatchObject({ ok: true, stage: "applied" });
    expect(JSON.parse(readFileSync(settingsPath, "utf8")).skillOverrides.solo).toBe("off");

    // the served capability offers the managed inverse, and enable restores name-only
    const after = await POST(request("POST", { action: "preview", itemId: item!.id, change: "enable" }, routeDeps));
    const enableRevision = (await after.json()).data.revision as string;
    const enabled = await POST(
      request("POST", { action: "apply", itemId: item!.id, change: "enable", expectedRevision: enableRevision, operationId: OP2 }, routeDeps),
    );
    expect(enabled.status).toBe(200);
    expect(JSON.parse(readFileSync(settingsPath, "utf8")).skillOverrides.solo).toBe("name-only");
  });

  // ---- Finish plan checkpoint 2: real OpenCode joined recovery ----

  const OPENCODE_JSONC = `{
  // keep this comment
  "provider": {
    "fixture": {
      "npm": "@openrouter/ai-sdk-provider",
      "models": {
        "jaxflow-builder-default": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "allow_fallbacks": true } }
        },
        "jaxflow-builder-fallback": {
          "id": "wire-real-model",
          "variants": { "high": { "reasoning": { "effort": "high" } } },
          "options": { "provider": { "sort": "price", "allow_fallbacks": false } }
        }
      }
    }
  },
  "plugin": ["pkg-a@1.0.0", "pkg-b@2.0.0"],
  "permission": { "skill": { "other": "allow" } }
}
`;

  function openCodeRoute(root: string, fault: { value: string | null }, venv: string, timeout = { value: false }) {
    const wrapper = join(root, "inv_wrapper.py");
    writeFileSync(
      wrapper,
      'import os, sys\nsys.path.insert(0, os.environ["PYTHONPATH"])\nimport jaxflow_settings_io as sio\nsio._FAULT = os.environ.get("JAX_INV_FAULT") or None\nimport jaxflow_inventory_io as inv\nraise SystemExit(inv.main(sys.argv))\n',
      "utf8",
    );
    const run = (action: string, payload: Record<string, unknown>) =>
      new Promise<{ ok: true; data: unknown } | { ok: false; error: string }>((resolve) => {
        const child = execFile(
          venv,
          [wrapper, action],
          {
            timeout: 20000,
            maxBuffer: 8 * 1024 * 1024,
            env: { ...process.env, PYTHONPATH: join(process.cwd(), "scripts"), JAX_INV_FAULT: fault.value ?? "" },
          },
          (err, stdout) => {
            if (err) {
              resolve({ ok: false, error: (err as { killed?: boolean }).killed ? "inventory-unconfirmed" : "inventory-helper-failed" });
              return;
            }
            try {
              const parsed = JSON.parse(stdout) as { ok?: unknown; data?: unknown; error?: unknown };
              // A process killed AFTER the real effect: the native write happened,
              // but the caller cannot confirm it.
              if (timeout.value && action === "apply") {
                resolve({ ok: false, error: "inventory-unconfirmed" });
                return;
              }
              resolve(parsed.ok === true ? { ok: true, data: parsed.data } : { ok: false, error: typeof parsed.error === "string" ? parsed.error : "inventory-helper-failed" });
            } catch {
              resolve({ ok: false, error: "inventory-helper-failed" });
            }
          },
        );
        child.stdin?.end(JSON.stringify(payload));
      });
    const roots = { claude_dir: join(root, "claude"), codex_dir: join(root, "codex"), opencode_dir: join(root, "config", "opencode"), jax_os: join(root, "jax-os") };
    for (const dir of Object.values(roots)) mkdirSync(dir, { recursive: true });
    mkdirSync(join(roots.opencode_dir, "skills", "writer"), { recursive: true });
    writeFileSync(join(roots.opencode_dir, "skills", "writer", "SKILL.md"), "# writer\n", "utf8");
    writeFileSync(join(roots.opencode_dir, "opencode.jsonc"), OPENCODE_JSONC, "utf8");
    const settingsPath = join(roots.jax_os, "agent-settings.json");
    writeFileSync(settingsPath, readFileSync(join(process.cwd(), "workflow", "fixtures", "agent-settings-v1.json"), "utf8"), "utf8");
    chmodSync(settingsPath, 0o600);
    const realDeps: InventoryDeps = { run, roots, venvPython: venv };
    const routeDeps = {
      getData: () => getInventoryData(realDeps),
      preview: (itemId: string, change: "disable" | "enable" | "remove-registration", provenance?: unknown) => previewInventory(realDeps, itemId, change, provenance),
      prepare: (itemId: string, change: "disable" | "enable" | "remove-registration", expectedRevision: string, provenance?: unknown, operationId?: string) =>
        prepareInventory(realDeps, itemId, change, expectedRevision, provenance, operationId),
      apply: (itemId: string, change: "disable" | "enable" | "remove-registration", revision: string, operationId: string, provenance?: unknown) =>
        applyInventory(realDeps, itemId, change, revision, operationId, provenance),
      reconcile: (expected: unknown, operationId: string) => reconcileInventory(realDeps, expected, operationId),
      revision: (itemId: string, executor?: string) => inventoryRevision(realDeps, itemId, executor),
    };
    return { routeDeps, roots, settingsPath, sourcePath: join(roots.opencode_dir, "opencode.jsonc") };
  }

  it("recovers a real OpenCode managed skill through a native-only publication failure, reload, recheck and settings-only reconcile", async () => {
    const venv = join(process.cwd(), ".local", "inventory-venv", "bin", "python");
    if (!existsSync(venv)) return;
    const root = mkdtempSync(join(tmpdir(), "jax-inv-oc-skill-"));
    const fault = { value: null as string | null };
    const { routeDeps, settingsPath, sourcePath } = openCodeRoute(root, fault, venv);

    const data = await routeDeps.getData();
    const bucket = data.executors.opencode;
    const item = (bucket.ok ? bucket.items : []).find((i) => i.registration.includes("skills/writer"));
    expect(item).toBeDefined();
    const prev = await routeDeps.preview(item!.id, "disable");
    const settingsBefore = JSON.parse(readFileSync(settingsPath, "utf8"));

    // an actual publication fault after the native write and before the settings write
    fault.value = "before-settings";
    const applied = await POST(
      request("POST", { action: "apply", itemId: item!.id, change: "disable", expectedRevision: prev.revision, operationId: OP }, routeDeps),
    );
    expect(applied.status).toBe(200);
    expect((await applied.json()).data.effect).toBe("activation-pending");
    fault.value = null;

    // native edit published, settings still pre-operation
    expect(readFileSync(sourcePath, "utf8")).toContain('"writer": "deny"');
    expect(JSON.parse(readFileSync(settingsPath, "utf8")).source_revisions).toEqual(settingsBefore.source_revisions);

    // prepared facts were persisted BEFORE the effect, not recovered from apply
    const payload = readInventoryOperationPayload(testDb, OP)!;
    expect(payload.settingsExpected).toMatchObject({ settings: settingsBefore.revision });
    expect(payload.settingsCandidate).toMatchObject({ settings: OP.replace(/-/g, "") });
    expect(payload.executor).toBe("opencode");
    expect(payload.selector).toContain("writer");
    expect(typeof payload.candidateDigest).toBe("string");

    // reload: a fresh server read joins the unresolved operation onto the item
    const reloaded = await POST(request("POST", { action: "recheck", operationId: OP }, routeDeps));
    expect((await reloaded.json()).data).toMatchObject({ stage: "activation-pending" });
    expect(readInventoryOperation(testDb, OP)?.stage).toBe("activation-pending");

    // confirmation comes from the recorded operation, never a live item preview
    const confirmation = await POST(request("POST", { action: "preview-reconcile", operationId: OP }, routeDeps));
    const confirmationBody = await confirmation.json();
    expect(confirmationBody.data).toMatchObject({ effect: "settings-only", settingsOnly: true });
    expect(readFileSync(sourcePath, "utf8")).toContain("keep this comment");

    const nativeBefore = readFileSync(sourcePath, "utf8");
    const reconciled = await POST(
      request("POST", { action: "reconcile", operationId: OP, expectedRevision: confirmationBody.data.revision }, routeDeps),
    );
    expect(reconciled.status).toBe(200);
    expect(readFileSync(sourcePath, "utf8")).toBe(nativeBefore);
    const settingsAfter = JSON.parse(readFileSync(settingsPath, "utf8"));
    expect(settingsAfter.source_revisions["opencode.jsonc"]).not.toBe(settingsBefore.source_revisions["opencode.jsonc"]);
    expect(settingsAfter.builders).toEqual(settingsBefore.builders);
    expect(settingsAfter.reviewers).toEqual(settingsBefore.reviewers);
    // a fresh artifact UUID for the reconcile: the first attempt's OP backup
    // plus a second, distinct one from the republish
    const backups = readdirSync(join(root, "jax-os")).filter((n) => n.startsWith("agent-settings.json.bak-"));
    expect(backups.some((n) => n.endsWith(OP))).toBe(true);
    expect(backups.some((n) => n.endsWith(OP) === false && /[0-9a-f-]{36}$/.test(n))).toBe(true);

    const settled = await POST(request("POST", { action: "recheck", operationId: OP }, routeDeps));
    expect((await settled.json()).data).toMatchObject({ stage: "applied" });
    expect(readInventoryOperation(testDb, OP)).toMatchObject({ ok: true, stage: "applied" });
  });

  it("recovers a removed OpenCode config-array plugin whose registration is already gone", async () => {
    const venv = join(process.cwd(), ".local", "inventory-venv", "bin", "python");
    if (!existsSync(venv)) return;
    const root = mkdtempSync(join(tmpdir(), "jax-inv-oc-plugin-"));
    const fault = { value: null as string | null };
    const { routeDeps, settingsPath, sourcePath } = openCodeRoute(root, fault, venv);

    const data = await routeDeps.getData();
    const bucket = data.executors.opencode;
    const item = (bucket.ok ? bucket.items : []).find((i) => i.registration === "pkg-a@1.0.0");
    expect(item).toBeDefined();
    const prev = await routeDeps.preview(item!.id, "remove-registration");

    fault.value = "before-settings";
    const applied = await POST(
      request("POST", { action: "apply", itemId: item!.id, change: "remove-registration", expectedRevision: prev.revision, operationId: OP }, routeDeps),
    );
    expect(applied.status).toBe(200);
    expect((await applied.json()).data.effect).toBe("activation-pending");
    fault.value = null;
    expect(readFileSync(sourcePath, "utf8")).not.toContain("pkg-a@1.0.0");

    const payload = readInventoryOperationPayload(testDb, OP)!;
    expect(payload).toMatchObject({ executor: "opencode", selector: "pkg-a@1.0.0", candidateDigest: "absent" });
    expect(payload.settingsCandidate).toMatchObject({ sources: expect.any(Object) });

    // the item is gone: GET keeps a SEPARATE recovery row (not an item join)
    const list = await GET(request("GET", undefined, routeDeps));
    const listBody = await list.json();
    expect(listBody.data.recovery).toEqual(
      expect.arrayContaining([expect.objectContaining({ operationId: OP, executor: "opencode", unavailable: false })]),
    );
    const rechecked = await POST(request("POST", { action: "recheck", operationId: OP }, routeDeps));
    expect((await rechecked.json()).data).toMatchObject({ stage: "activation-pending" });

    const confirmation = await POST(request("POST", { action: "preview-reconcile", operationId: OP }, routeDeps));
    const confirmationBody = await confirmation.json();
    const nativeBefore = readFileSync(sourcePath, "utf8");
    const reconciled = await POST(
      request("POST", { action: "reconcile", operationId: OP, expectedRevision: confirmationBody.data.revision }, routeDeps),
    );
    expect(reconciled.status).toBe(200);
    expect(readFileSync(sourcePath, "utf8")).toBe(nativeBefore);
    expect(JSON.parse(readFileSync(settingsPath, "utf8")).source_revisions["opencode.jsonc"]).not.toBe("absent");

    const settled = await POST(request("POST", { action: "recheck", operationId: OP }, routeDeps));
    expect((await settled.json()).data).toMatchObject({ stage: "applied" });
    expect(readInventoryOperation(testDb, OP)).toMatchObject({ ok: true, stage: "applied" });
  });

  it("refuses a changed preview before reserving any mutation or touching native state", async () => {
    const venv = join(process.cwd(), ".local", "inventory-venv", "bin", "python");
    if (!existsSync(venv)) return;
    const root = mkdtempSync(join(tmpdir(), "jax-inv-oc-stale-"));
    const { routeDeps, sourcePath } = openCodeRoute(root, { value: null }, venv);
    const data = await routeDeps.getData();
    const item = (data.executors.opencode.ok ? data.executors.opencode.items : []).find((i) => i.registration.includes("skills/writer"))!;
    const before = readFileSync(sourcePath, "utf8");
    const res = await POST(
      request("POST", { action: "apply", itemId: item.id, change: "disable", expectedRevision: "0".repeat(64), operationId: OP }, routeDeps),
    );
    expect(res.status).toBe(409);
    expect(readFileSync(sourcePath, "utf8")).toBe(before);
    expect(testDb.prepare("SELECT COUNT(*) AS c FROM mutations").get()).toMatchObject({ c: 0 });
  });

  it("reuses one operation UUID instead of minting a new effect, and keeps the native result on audit-finalization failure", async () => {
    const venv = join(process.cwd(), ".local", "inventory-venv", "bin", "python");
    if (!existsSync(venv)) return;
    const root = mkdtempSync(join(tmpdir(), "jax-inv-oc-dupe-"));
    const { routeDeps, sourcePath } = openCodeRoute(root, { value: null }, venv);
    const data = await routeDeps.getData();
    const item = (data.executors.opencode.ok ? data.executors.opencode.items : []).find((i) => i.registration.includes("skills/writer"))!;
    const prev = await routeDeps.preview(item.id, "disable");
    const body = { action: "apply", itemId: item.id, change: "disable", expectedRevision: prev.revision, operationId: OP };
    expect((await POST(request("POST", body, routeDeps)).then((r) => r.json())).ok).toBe(true);
    const rowsAfterFirst = (testDb.prepare("SELECT COUNT(*) AS c FROM mutations").get() as { c: number }).c;
    const second = await POST(request("POST", body, routeDeps));
    expect(await second.json()).toMatchObject({ ok: true, data: { operationId: OP, itemId: item.id } });
    expect((testDb.prepare("SELECT COUNT(*) AS c FROM mutations").get() as { c: number }).c).toBe(rowsAfterFirst);

    // a finalization update failure keeps the applied native result and stays pending
    const root2 = mkdtempSync(join(tmpdir(), "jax-inv-oc-final-"));
    testDb.close();
    testDb = openDb(":memory:");
    const second2 = openCodeRoute(root2, { value: null }, venv);
    const data2 = await second2.routeDeps.getData();
    const item2 = (data2.executors.opencode.ok ? data2.executors.opencode.items : []).find((i) => i.registration.includes("skills/writer"))!;
    const prev2 = await second2.routeDeps.preview(item2.id, "disable");
    testDb.exec("CREATE TRIGGER fail_fin BEFORE UPDATE ON mutations WHEN NEW.ok IS NOT NULL BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const fin = await POST(
      request("POST", { action: "apply", itemId: item2.id, change: "disable", expectedRevision: prev2.revision, operationId: OP2 }, second2.routeDeps),
    );
    const finBody = await fin.json();
    expect(finBody).toMatchObject({ ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending" });
    expect(finBody.data).toMatchObject({ effect: "published" });
    expect(readFileSync(second2.sourcePath, "utf8")).toContain('"writer": "deny"');
    expect((testDb.prepare("SELECT ok FROM mutations WHERE json_extract(payload, '$.operationId') = ?").get(OP2) as { ok: number | null }).ok).toBeNull();

    // a timeout reported AFTER the real native effect is unconfirmed, and recheck recovers it
    const root3 = mkdtempSync(join(tmpdir(), "jax-inv-oc-timeout-"));
    testDb.close();
    testDb = openDb(":memory:");
    const timeout = { value: true };
    const third = openCodeRoute(root3, { value: null }, venv, timeout);
    const data3 = await third.routeDeps.getData();
    const item3 = (data3.executors.opencode.ok ? data3.executors.opencode.items : []).find((i) => i.registration.includes("skills/writer"))!;
    const prev3 = await third.routeDeps.preview(item3.id, "disable");
    const late = await POST(
      request("POST", { action: "apply", itemId: item3.id, change: "disable", expectedRevision: prev3.revision, operationId: OP }, third.routeDeps),
    );
    timeout.value = false;
    expect(await late.json()).toMatchObject({ ok: false, code: "mutation-unconfirmed", effect: "unconfirmed" });
    expect(readFileSync(third.sourcePath, "utf8")).toContain('"writer": "deny"');
    expect(readInventoryOperation(testDb, OP)?.stage).toBe("unconfirmed");
    const recovered = await POST(request("POST", { action: "recheck", operationId: OP }, third.routeDeps));
    expect((await recovered.json()).data).toMatchObject({ stage: "applied" });
    expect(readInventoryOperation(testDb, OP)).toMatchObject({ ok: true, stage: "applied" });
    void sourcePath;
  });
});
