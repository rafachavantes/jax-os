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
}));

import { InventoryReconcileDialog } from "./InventoryReconcileDialog";

describe("InventoryReconcileDialog", () => {
  it("confirms a settings-only reconciliation for a removed item without any live preview", () => {
    const html = renderToStaticMarkup(
      createElement(InventoryReconcileDialog, {
        name: "abc123…",
        pending: false,
        error: null,
        onConfirm: () => {},
        onCancel: () => {},
      }),
    );
    expect(html).toContain("<dialog");
    expect(html).toContain("reconcileTitle");
    expect(html).toContain("reconcileBody");
    expect(html).toContain("effectText.settings");
    expect(html).toContain("reconcileConfirm");
    expect(html).toContain('type="checkbox"');
    expect(html).toContain("cancel");
    expect(html).toMatch(/primaryButton[^>]*disabled/);
  });

  it("shows a refused reconciliation error in an accessible live region", () => {
    const html = renderToStaticMarkup(
      createElement(InventoryReconcileDialog, {
        name: "abc123…",
        pending: false,
        error: "native state changed since the operation",
        onConfirm: () => {},
        onCancel: () => {},
      }),
    );
    expect(html).toContain('role="status"');
    expect(html).toContain("applyRefused");
  });

  it("disables confirmation when stale but keeps cancellation enabled", () => {
    stateFixture.forceConfirmed = true;
    const fresh = renderToStaticMarkup(
      createElement(InventoryReconcileDialog, {
        name: "abc123…",
        pending: false,
        stale: false,
        error: null,
        onConfirm: () => {},
        onCancel: () => {},
      }),
    );
    const html = renderToStaticMarkup(
      createElement(InventoryReconcileDialog, {
        name: "abc123…",
        pending: false,
        stale: true,
        error: null,
        onConfirm: () => {},
        onCancel: () => {},
      }),
    );
    expect(fresh).toMatch(/primaryButton(?![^>]*disabled)[^>]*>reconcile</);
    expect(html).toMatch(/primaryButton[^>]*disabled/);
    expect(html.match(/<button[^>]*compareButton[^>]*>cancel<\/button>/)?.[0]).not.toMatch(/\bdisabled\b/);
  });
});
