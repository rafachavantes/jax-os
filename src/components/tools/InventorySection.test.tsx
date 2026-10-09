import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  query: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown, refetch: () => Promise.resolve() },
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({ useQuery: () => harness.query }));
vi.mock("./InventoryActionDialog", () => ({ InventoryActionDialog: () => null }));

import { InventorySection } from "./InventorySection";
import type { InventoryData, InventoryItem } from "@/server/collectors/inventory";

function item(overrides: Partial<InventoryItem> = {}): InventoryItem {
  return {
    id: "a".repeat(64),
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
    capabilities: { disable: { available: true, reason: null } },
    ...overrides,
  };
}

function payload(overrides: Partial<InventoryData["executors"]> = {}): InventoryData {
  return {
    executors: {
      claude: { ok: true, items: [item()] },
      codex: { ok: true, items: [item({ id: "b".repeat(64), executor: "codex", name: "beta", origin: "config" })] },
      opencode: { ok: true, items: [] },
      ...overrides,
    },
  };
}

describe("InventorySection — SSR", () => {
  beforeEach(() => {
    harness.query = { data: { ok: true, data: payload() }, isError: false, dataUpdatedAt: 0, error: undefined, refetch: () => Promise.resolve() };
  });

  it("renders the toolbar, filtered/known counts and rows", () => {
    const html = renderToStaticMarkup(createElement(InventorySection));
    expect(html).toContain("heading");
    expect(html).toContain("search");
    expect(html).toContain("count");
    expect(html).toContain("alpha");
    expect(html).toContain("beta");
    expect(html).toContain("selectItem");
  });

  it("keeps an unavailable source as a warning, not as zero installed items", () => {
    harness.query = {
      data: { ok: true, data: payload({ codex: { ok: false, error: "boom", items: [] } }) },
      isError: false,
      dataUpdatedAt: 0,
      error: undefined,
      refetch: () => Promise.resolve(),
    };
    const html = renderToStaticMarkup(createElement(InventorySection));
    expect(html).toContain("partial");
    expect(html).toContain("alpha");
  });

  it("renders a visible warning when the whole source failed", () => {
    harness.query = { data: undefined, isError: true, dataUpdatedAt: 0, error: new Error("down"), refetch: () => Promise.resolve() };
    const html = renderToStaticMarkup(createElement(InventorySection));
    expect(html).toContain("unavailable");
  });

  it("localizes stage/origin/kind through typed fields and distinguishes unavailable from removed recovery rows", () => {
    const recoveryPayload: InventoryData = {
      executors: {
        claude: { ok: true, items: [item({ operation: { operationId: "o1", change: "disable", stage: "activation-pending", effect: null } })] },
        codex: { ok: false, error: "boom", items: [] },
        opencode: { ok: true, items: [] },
      },
      recovery: [
        { operationId: "r".repeat(36), itemId: "b".repeat(64), change: "remove-registration", stage: "activation-pending", effect: null, executor: "opencode", unavailable: true },
        { operationId: "q".repeat(36), itemId: "c".repeat(64), change: "remove-registration", stage: "activation-pending", effect: null, executor: "opencode", unavailable: false },
      ],
    };
    harness.query = { data: { ok: true, data: recoveryPayload }, isError: false, dataUpdatedAt: 0, error: undefined, refetch: () => Promise.resolve() };
    const html = renderToStaticMarkup(createElement(InventorySection));
    // typed machine values are translated, never shown verbatim
    expect(html).toContain("stage.activation-pending");
    expect(html).toContain("originLabel.settings");
    expect(html).toContain("sourceUnavailable");
    expect(html).toContain("removed");
    expect(html).toContain("reconcile");
  });
});
