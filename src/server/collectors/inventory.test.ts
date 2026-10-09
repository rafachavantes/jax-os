import { describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  applyInventory,
  getInventoryData,
  inventoryRevision,
  prepareInventory,
  previewInventory,
  reconcileInventory,
  runPython,
  type InventoryData,
  type InventoryDeps,
} from "./inventory";

const EMPTY: InventoryData = {
  executors: {
    claude: { ok: true, items: [] },
    codex: { ok: true, items: [] },
    opencode: { ok: true, items: [] },
  },
};

function deps(overrides: Partial<InventoryDeps> = {}): InventoryDeps {
  return {
    run: vi.fn(async () => ({ ok: true as const, data: EMPTY })),
    roots: { claude_dir: "/tmp/c", codex_dir: "/tmp/x", opencode_dir: "/tmp/o", jax_os: "/tmp/j" },
    venvPython: "/tmp/venv/bin/python",
    ...overrides,
  };
}

describe("inventory collector", () => {
  it("parses the helper's snapshot envelope and always passes the fixed roots", async () => {
    const run = vi.fn(async () => ({ ok: true as const, data: EMPTY }));
    const data = await getInventoryData(deps({ run }));
    expect(data).toEqual(EMPTY);
    expect(run).toHaveBeenCalledWith("snapshot", { roots: expect.any(Object) }, "python3");
  });

  it("maps a helper failure to a thrown error (visible warning, never a crash page)", async () => {
    const run = vi.fn(async () => ({ ok: false as const, error: "inventory-format-unsupported" }));
    await expect(getInventoryData(deps({ run }))).rejects.toThrow("inventory-format-unsupported");
  });

  it("preview is read-only and returns the helper's revision-bearing envelope", async () => {
    const run = vi.fn(async () => ({
      ok: true as const,
      data: { itemId: "a".repeat(64), change: "disable", revision: "b".repeat(64), targets: [], effect: "x", recovery: null, requiresNewSession: true },
    }));
    const preview = await previewInventory(deps({ run }), "a".repeat(64), "disable");
    expect(preview.revision).toBe("b".repeat(64));
    expect(run).toHaveBeenCalledWith("preview", expect.objectContaining({ change: "disable" }), "python3");
  });

  it("apply requires the pinned write interpreter and uses it (never the system python)", async () => {
    await expect(applyInventory(deps({ venvPython: null }), "a".repeat(64), "disable", "b".repeat(64), "op")).rejects.toThrow(
      "inventory-setup-required",
    );
    const run = vi.fn(async () => ({ ok: true as const, data: { effect: "applied" } }));
    await applyInventory(deps({ run }), "a".repeat(64), "disable", "b".repeat(64), "op");
    expect(run).toHaveBeenCalledWith("apply", expect.objectContaining({ expectedRevision: "b".repeat(64) }), "/tmp/venv/bin/python");
  });

  it("reconcile is a distinct confirmed call, also on the pinned interpreter", async () => {
    const run = vi.fn(async () => ({ ok: true as const, data: { effect: "published" } }));
    await reconcileInventory(deps({ run }), { settings: "absent", sources: {} }, "op");
    expect(run).toHaveBeenCalledWith("reconcile", expect.objectContaining({ operationId: "op" }), "/tmp/venv/bin/python");
  });

  it("prepare is read-only but runs on the pinned interpreter and forwards the expected revision", async () => {
    const prep = {
      itemId: "a".repeat(64),
      change: "disable",
      beforeDigest: "b".repeat(64),
      candidateDigest: "c".repeat(64),
      provenance: { selector: "solo" },
      settingsExpected: null,
      settingsCandidate: null,
    };
    const run = vi.fn(async () => ({ ok: true as const, data: prep }));
    const out = await prepareInventory(deps({ run }), "a".repeat(64), "disable", "b".repeat(64));
    expect(out.candidateDigest).toBe("c".repeat(64));
    expect(run).toHaveBeenCalledWith(
      "prepare",
      expect.objectContaining({ expectedRevision: "b".repeat(64), change: "disable" }),
      "/tmp/venv/bin/python",
    );
    await expect(prepareInventory(deps({ venvPython: null }), "a".repeat(64), "disable", "b".repeat(64))).rejects.toThrow(
      "inventory-setup-required",
    );
  });

  it("revision reads the current state without preview eligibility", async () => {
    const run = vi.fn(async () => ({ ok: true as const, data: { revision: "d".repeat(64), settings: null } }));
    expect(await inventoryRevision(deps({ run }), "a".repeat(64), "opencode")).toEqual({ revision: "d".repeat(64), settings: null });
    expect(run).toHaveBeenCalledWith("revision", expect.objectContaining({ itemId: "a".repeat(64), executor: "opencode" }), "python3");
  });

  // Real boundary: collector → actual Python helper subprocess → temporary
  // filesystem. Snapshot is stdlib-only (no tomlkit), so the system
  // interpreter is enough.
  it("reads a real temporary native root through the actual helper subprocess", async () => {
    const root = mkdtempSync(join(tmpdir(), "jax-inv-integration-"));
    const claude = join(root, "claude");
    mkdirSync(claude, { recursive: true });
    writeFileSync(join(claude, "settings.json"), JSON.stringify({ enabledPlugins: { "alpha@market": true } }), "utf8");
    const data = await getInventoryData({
      run: (action, payload, python) => runPython(python, action, payload),
      roots: { claude_dir: claude, codex_dir: join(root, "codex"), opencode_dir: join(root, "oc"), jax_os: join(root, "jax") },
      venvPython: null,
    });
    const claudeBucket = data.executors.claude;
    expect(claudeBucket.ok).toBe(true);
    const items = claudeBucket.ok ? claudeBucket.items : [];
    expect(items.some((i) => i.registration === "alpha@market" && i.state === "enabled")).toBe(true);
    expect(data.executors.opencode.ok).toBe(true);
  });
});
