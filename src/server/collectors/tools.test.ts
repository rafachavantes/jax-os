import { mkdirSync, mkdtempSync, realpathSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  codexGroup,
  getToolsData,
  localPluginName,
  normalizeSpecifier,
  type AppError,
  type AppTools,
} from "./tools";
import type { InventoryData, InventoryExecutor, InventoryExecutorBucket, InventoryItem } from "./inventory";

function item(overrides: Partial<InventoryItem> & Pick<InventoryItem, "kind">): InventoryItem {
  return {
    id: "i".repeat(64),
    executor: "claude",
    registration: "x",
    name: "x",
    description: "",
    path: "/p/x",
    scope: "user",
    origin: "independent",
    parent: null,
    canonicalTarget: null,
    state: "enabled",
    capabilities: {},
    ...overrides,
  };
}

function data(items: Partial<Record<InventoryExecutor, InventoryItem[]>>, failed: InventoryExecutor[] = []): InventoryData {
  const bucket = (executor: InventoryExecutor): InventoryExecutorBucket =>
    failed.includes(executor)
      ? { ok: false, error: "inventory-format-unsupported", items: [] }
      : { ok: true, items: items[executor] ?? [] };
  return { executors: { claude: bucket("claude"), codex: bucket("codex"), opencode: bucket("opencode") } };
}

describe("pure opencode specifier rules (spec §3)", () => {
  it("splits a scoped package at the LAST @ and keeps git/absolute specs version-less", () => {
    expect(normalizeSpecifier("@scope/name@1.2.3")).toEqual({ name: "@scope/name", version: "1.2.3" });
    expect(normalizeSpecifier("@scope/name")).toEqual({ name: "@scope/name", version: null });
    expect(normalizeSpecifier("pkg@git+https://x/y")).toEqual({ name: "pkg", version: null });
    expect(normalizeSpecifier("/abs/plugin.js")).toEqual({ name: "plugin.js", version: null });
  });

  it("local plugin names drop the js/ts/tsx extension; codex group follows the openai- prefix", () => {
    expect(localPluginName("mine.ts")).toBe("mine");
    expect(localPluginName("keep.jsx")).toBe("keep.jsx");
    expect(codexGroup("openai-official")).toBe("system");
    expect(codexGroup("community")).toBe("yours");
    expect(codexGroup(null)).toBe("yours");
  });
});

describe("getToolsData projects the native snapshot (single production read)", () => {
  it("projects Claude configured plugins with their selected version and bundled skills", () => {
    const items = [
      item({ kind: "plugin", registration: "alpha@market", name: "alpha@market", origin: "settings", state: "enabled", installedVersion: "1.2.0" }),
      item({ kind: "plugin", registration: "cached@market", name: "cached@market", origin: "cache", state: "unknown" }),
      item({ kind: "skill", registration: "/x/alpha/skills/child", name: "child", origin: "independent", parent: "alpha@market" }),
    ];
    const tools = getToolsData(data({ claude: items }), "/nope/registry");
    const claude = tools.claude as AppTools;
    expect(claude.plugins).toEqual([
      { name: "alpha", marketplace: "market", version: "1.2.0", enabled: true, group: "yours", skills: ["child"] },
      { name: "cached", marketplace: "market", version: null, enabled: false, group: "yours", skills: [] },
    ]);
  });

  it("distinguishes dir, registry and broken skill sources from the native facts", () => {
    const registry = realpathSync(mkdtempSync(join(tmpdir(), "jax-registry-")));
    const shared = join(registry, "shared");
    mkdirSync(shared);
    const link = join(tmpdir(), `jax-link-${Date.now()}`);
    symlinkSync(shared, link);
    const items = [
      item({ kind: "skill", registration: "/x/dir", name: "dir", origin: "independent", state: "enabled" }),
      item({ kind: "skill", registration: link, name: "registry", origin: "registration-link", state: "enabled", canonicalTarget: shared }),
      item({ kind: "skill", registration: "/x/gone", name: "gone", origin: "registration-link", state: "broken", canonicalTarget: "/missing" }),
    ];
    const claude = getToolsData(data({ claude: items }), registry).claude as AppTools;
    expect(claude.skills).toEqual([
      { name: "dir", source: "dir" },
      { name: "registry", source: "registry" },
      { name: "gone", source: "broken" },
    ]);
  });

  it("keeps agent descriptions and the opencode loading sources; Codex agents stay empty", () => {
    const claudeAgent = item({ kind: "agent", executor: "claude", registration: "/a/reviewer.md", name: "reviewer", description: "does reviews" });
    const codexAgent = item({ kind: "agent", executor: "codex", registration: "/a/x.md", name: "x", description: "" });
    const tools = getToolsData(data({ claude: [claudeAgent], codex: [codexAgent] }), "/nope");
    expect((tools.claude as AppTools).agents).toEqual([{ name: "reviewer", description: "does reviews" }]);
    expect((tools.codex as AppTools).agents).toEqual([{ name: "x", description: null }]);
  });

  it("projects OpenCode configured specs and local plugin files", () => {
    const items = [
      item({ kind: "plugin", executor: "opencode", registration: "pkg-a@1.0.0", name: "pkg-a@1.0.0", origin: "config-array", state: "enabled" }),
      item({ kind: "plugin", executor: "opencode", registration: "/x/plugin/mine.ts", name: "mine.ts", origin: "local-file", state: "enabled" }),
    ];
    const opencode = getToolsData(data({ opencode: items }), "/nope").opencode as AppTools;
    expect(opencode.plugins).toEqual([
      { name: "pkg-a", marketplace: null, version: "1.0.0", enabled: true, group: "yours", skills: [] },
      { name: "mine", marketplace: null, version: null, enabled: true, group: "yours", skills: [] },
    ]);
  });

  it("exposes loadingSources from the opencode bucket and isolates a failed bucket", () => {
    const withSources: InventoryData = {
      executors: {
        claude: { ok: true, items: [] },
        codex: { ok: false, error: "inventory-format-unsupported", items: [] },
        opencode: { ok: true, items: [], loadingSources: ["opencode.jsonc"] },
      },
    } as unknown as InventoryData;
    const tools = getToolsData(withSources, "/nope");
    expect((tools.codex as AppError).error).toBe("inventory-format-unsupported");
    expect((tools.opencode as AppTools).loadingSources).toEqual(["opencode.jsonc"]);
    expect((tools.claude as AppTools).plugins).toEqual([]);
  });

  it("preserves same-name plugins from different executors as separate rows", () => {
    const items: InventoryItem[] = [
      item({ kind: "plugin", executor: "claude", registration: "shared@market", name: "shared@market" }),
      item({ kind: "plugin", executor: "codex", registration: "shared", name: "shared" }),
    ];
    const tools = getToolsData(data({ claude: [items[0]], codex: [items[1]] }), "/nope");
    expect((tools.claude as AppTools).plugins[0].name).toBe("shared");
    expect((tools.codex as AppTools).plugins[0].name).toBe("shared");
    expect((tools.codex as AppTools).plugins[0].marketplace).toBeNull();
  });
});
