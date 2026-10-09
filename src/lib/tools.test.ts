import { describe, expect, it } from "vitest";
import { deriveInventoryView, labelKey, stageKey } from "./tools";
import type { InventoryData, InventoryItem } from "../server/collectors/inventory";

function item(partial: Partial<InventoryItem> & Pick<InventoryItem, "id" | "executor" | "kind" | "name">): InventoryItem {
  return {
    registration: partial.name,
    description: "",
    path: "/x",
    scope: "user",
    origin: "native",
    parent: null,
    canonicalTarget: null,
    state: "enabled",
    capabilities: {},
    ...partial,
  };
}

const items: InventoryItem[] = [
  item({ id: "a", executor: "claude", kind: "plugin", name: "superpowers", origin: "settings" }),
  item({ id: "b", executor: "codex", kind: "plugin", name: "superpowers", origin: "config" }),
  item({ id: "c", executor: "opencode", kind: "skill", name: "writer", description: "writes docs", origin: "permission-map" }),
];

function inventoryData(overrides: Partial<InventoryData["executors"]> = {}): InventoryData {
  return {
    executors: {
      claude: { ok: true, items: items.filter((i) => i.executor === "claude") },
      codex: { ok: true, items: items.filter((i) => i.executor === "codex") },
      opencode: { ok: true, items: items.filter((i) => i.executor === "opencode") },
      ...overrides,
    } as InventoryData["executors"],
  };
}

describe("deriveInventoryView (spec §7.1)", () => {
  it("search covers name, description and origin; executor/kind filters apply", () => {
    expect(deriveInventoryView(inventoryData(), { query: "writes", executor: "all", kind: "all" }).rows.map((r) => r.id)).toEqual(["c"]);
    expect(deriveInventoryView(inventoryData(), { query: "permission", executor: "all", kind: "all" }).rows.map((r) => r.id)).toEqual(["c"]);
    expect(deriveInventoryView(inventoryData(), { query: "", executor: "claude", kind: "all" }).rows.map((r) => r.id)).toEqual(["a"]);
    expect(deriveInventoryView(inventoryData(), { query: "", executor: "all", kind: "skill" }).rows.map((r) => r.id)).toEqual(["c"]);
  });

  it("reports filtered-versus-known counts and keeps same-name distinct rows", () => {
    const view = deriveInventoryView(inventoryData(), { query: "superpowers", executor: "all", kind: "all" });
    expect(view.known).toBe(3);
    expect(view.filtered).toBe(2);
    expect(view.rows.map((r) => r.executor).sort()).toEqual(["claude", "codex"]);
  });

  it("a partial source failure is reported, not turned into zero installed items", () => {
    const view = deriveInventoryView(inventoryData({ codex: { ok: false, error: "boom", items: [] } }), { query: "", executor: "all", kind: "all" });
    expect(view.partial).toBe(true);
    expect(view.anyError).toBe(true);
    expect(view.known).toBe(2); // the failed source contributes no rows but the others survive
  });

  it("distinguishes an empty search from an unavailable source", () => {
    const empty = deriveInventoryView(inventoryData(), { query: "zzz", executor: "all", kind: "all" });
    expect(empty.empty).toBe(true);
    expect(empty.partial).toBe(false);
    const unavailable = deriveInventoryView(null, { query: "", executor: "all", kind: "all" });
    expect(unavailable.anyError).toBe(true);
  });
});

describe("typed display labels (F9)", () => {
  it("maps a known stage to its locale key and an unknown stage to the fallback", () => {
    expect(stageKey("activation-pending")).toBe("activation-pending");
    expect(stageKey("weird")).toBe("unknown");
  });

  it("maps known kind/origin values to locale keys and passes unknown source data through", () => {
    expect(labelKey("kindLabel", "plugin")).toBe("kindLabel.plugin");
    expect(labelKey("originLabel", "config-array")).toBe("originLabel.config-array");
    expect(labelKey("originLabel", "/home/x/custom")).toBe("/home/x/custom");
  });
});
