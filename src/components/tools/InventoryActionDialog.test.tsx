import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

const stateFixture = vi.hoisted(() => ({ forceConfirmed: false }));
vi.mock("react", async () => {
  const actual = await vi.importActual<typeof import("react")>("react");
  return { ...actual, useState: (initial: boolean) => (stateFixture.forceConfirmed ? [true, () => {}] : actual.useState(initial)) };
});
afterEach(() => { stateFixture.forceConfirmed = false; });

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));

import { InventoryActionDialog } from "./InventoryActionDialog";
import type { InventoryItem, InventoryPreview } from "@/server/collectors/inventory";

const item: InventoryItem = {
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
};

const preview: InventoryPreview = {
  itemId: item.id,
  change: "disable",
  revision: "b".repeat(64),
  targets: ["/c/settings.json"],
  effect: "disable plugin alpha",
  recovery: null,
  requiresNewSession: true,
};

describe("InventoryActionDialog", () => {
  it("names the item, effect and target and requires an explicit confirmation", () => {
    const html = renderToStaticMarkup(
      createElement(InventoryActionDialog, {
        item,
        preview,
        status: null,
        error: null,
        pending: false,
        onApply: () => {},
        onRecheck: () => {},
        onClose: () => {},
      }),
    );
    expect(html).toContain("<dialog");
    expect(html).toContain("dialogTitle");
    // the helper's raw English effect is never rendered: the typed change maps
    // to a locale key with the item name
    expect(html).toContain("effectText.disable");
    expect(html).not.toContain("disable plugin alpha");
    expect(html).toContain("/c/settings.json");
    expect(html).toContain("confirm");
    expect(html).toContain('type="checkbox"');
    expect(html).toContain("recheck");
    expect(html).toMatch(/primaryButton[^>]*disabled/);
  });

  it("renders the refused/uncertain error in an accessible live region", () => {
    const html = renderToStaticMarkup(
      createElement(InventoryActionDialog, {
        item,
        preview,
        status: { tone: "error", text: "writeUnconfirmed" },
        error: "inventory-unconfirmed",
        pending: false,
        onApply: () => {},
        onRecheck: () => {},
        onClose: () => {},
      }),
    );
    expect(html).toContain('role="status"');
    expect(html).toContain("applyRefused");
  });

  it("disables Apply when stale but keeps Close enabled", () => {
    stateFixture.forceConfirmed = true;
    const fresh = renderToStaticMarkup(
      createElement(InventoryActionDialog, {
        item,
        preview,
        status: null,
        error: null,
        pending: false,
        stale: false,
        onApply: () => {},
        onRecheck: () => {},
        onClose: () => {},
      }),
    );
    const html = renderToStaticMarkup(
      createElement(InventoryActionDialog, {
        item,
        preview,
        status: null,
        error: null,
        pending: false,
        stale: true,
        onApply: () => {},
        onRecheck: () => {},
        onClose: () => {},
      }),
    );
    expect(fresh).toMatch(/primaryButton(?![^>]*disabled)[^>]*>apply</);
    expect(html).toMatch(/primaryButton[^>]*disabled/);
    expect(html.match(/<button[^>]*dialogClose[^>]*>close<\/button>/)?.[0]).not.toMatch(/\bdisabled\b/);
  });
});
