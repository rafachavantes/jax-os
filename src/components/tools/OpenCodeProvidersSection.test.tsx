import { createElement, type ComponentType } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import { BOOTSTRAP_DRAFT, type ConnectionChoice, type EditorRevision } from "@/lib/agent-settings";
import { ProviderRemovalDialog } from "./ProviderRemovalDialog";
import en from "../../../messages/en-US.json";
import pt from "../../../messages/pt-BR.json";
import type { ToolsDraftApi } from "./ToolsDraftProvider";

const harness = vi.hoisted(() => ({
  api: {} as unknown,
  query: { data: undefined as unknown, isError: false, dataUpdatedAt: 0, error: undefined as unknown },
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => harness.query,
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));
vi.mock("./ToolsDraftProvider", async (importOriginal) => {
  const actual = await importOriginal<typeof import("./ToolsDraftProvider")>();
  return { ...actual, useToolsDraft: () => harness.api as never };
});

import { OpenCodeProvidersSection } from "./OpenCodeProvidersSection";

const REV: EditorRevision = {
  settings: "a".repeat(32),
  sources: { "opencode.json": "absent", "opencode.jsonc": "absent" },
};

function connection(over: Partial<ConnectionChoice> & { id: string }): ConnectionChoice {
  return {
    label: over.id,
    adapter: "xai",
    base_url: null,
    auth: "api-key",
    health: "registered",
    credential: { kind: "native" },
    editable: true,
    reason: null,
    used_by: [],
    models: [
      { id: "m/1", label: "M", origin: "catalog", efforts: [], no_effort: true, compatible: true, reason: null, context: 1000, effort_template: null },
    ],
    ...over,
  };
}

function render(connections: ConnectionChoice[], busy = false) {
  harness.api = {
    dirty: false,
    conflict: false,
    draft: JSON.parse(JSON.stringify(BOOTSTRAP_DRAFT)),
    baseline: JSON.parse(JSON.stringify(BOOTSTRAP_DRAFT)),
    revision: REV,
    connections,
    settingsPresent: true,
    busy,
    resetEpoch: 0,
    setBusy: vi.fn(),
    setDraft: vi.fn(),
    discard: vi.fn(),
    applyServer: vi.fn(),
    markSaved: vi.fn(),
    credential: null,
    openCredential: vi.fn(),
    closeCredential: vi.fn(),
  } satisfies ToolsDraftApi;
  harness.query = {
    data: {
      ok: true,
      data: {
        editor_revision: REV,
        connections,
      },
    },
    isError: false,
    dataUpdatedAt: 0,
    error: undefined,
  };
  // The component's props param has a `= {}` default, which defeats createElement's generic
  // prop inference (falls back to the no-props `Attributes` overload) — cast the reference so
  // TS keeps the real prop type.
  const Section = OpenCodeProvidersSection as ComponentType<{ onOpenAgents?: () => void }>;
  return renderToStaticMarkup(createElement(Section, {
    onOpenAgents: () => {},
  }));
}

describe("OpenCodeProvidersSection", () => {
  it("offers model edit/removal through the approved dialogs, not confirm/alert", () => {
    const html = render([connection({ id: "acme" })]);
    expect(html).toContain("editModel");
    expect(html).toContain("removeModel");
    expect(html).toContain("editConnection");
    expect(html).toContain("viewAgents");
    expect(html).toContain("m/1");
  });

  it("offers one Edit credential button for an uncredentialed API-key connection", () => {
    const html = render([connection({ id: "fresh", credential: null })]);
    expect(button(html, "editCredential")).toContain("editCredential");
    expect(button(html, "editCredential")).not.toContain('disabled=""');
    expect(button(html, "addCredential")).toBe("");
    expect(button(html, "linkCredential")).toBe("");
    expect(button(html, "replaceCredential")).toBe("");
  });

  it("opens the credential dialog with the connection's current binding on the one Edit credential button (MOA-498 D4/12)", () => {
    const html = render([connection({ id: "acme", credential: { kind: "env", env: "JAX_PROVIDER_ACME_API_KEY" } })]);
    expect(button(html, "editCredential")).toContain("editCredential");
    expect(html).not.toContain("addCredential");
    expect(html).not.toContain("linkCredential");
    expect(html).not.toContain("replaceCredential");
  });

  it("hides every provider mutation for an OAuth connection and explains why", () => {
    const html = render([connection({ id: "oauth-conn", auth: "oauth", health: "oauth-managed", credential: { kind: "native" }, editable: false, reason: "oauth-managed" })]);
    for (const label of ["addModel", "editModel", "removeModel", "editCredential", "editConnection", "removeConnection"]) {
      expect(button(html, label)).toBe("");
    }
    expect(html).toContain("oauthReadOnly");
    expect(html).toContain("viewAgents");
    expect(html).toContain("searchModels");
  });

  it("disables provider mutations for unavailable or unsafe native connections", () => {
    const html = render([connection({ id: "unsafe", auth: "api-key", editable: false, reason: "unsupported-address" })]);
    for (const label of ["addModel", "editModel", "removeModel", "editCredential", "editConnection", "removeConnection"]) {
      expect(button(html, label)).toContain('disabled=""');
    }
  });

  it("disables provider mutations while the shared draft is busy", () => {
    const html = render([connection({ id: "acme" })], true);
    for (const label of ["addModel", "editModel", "removeModel", "editCredential", "editConnection", "removeConnection"]) {
      expect(button(html, label)).toContain('disabled=""');
    }
  });

  it("places the connection actions above the model list in mockup order", () => {
    const html = render([connection({ id: "acme" })]);
    const actions = html.indexOf("providerActions");
    expect(actions).toBeGreaterThan(-1);
    expect(actions).toBeLessThan(html.indexOf("modelHead"));
    const block = html.slice(actions, html.indexOf("modelHead"));
    const order = ["editConnection", "editCredential", "removeConnection"].map((label) => block.indexOf(label));
    expect(order.every((at) => at > -1)).toBe(true);
    expect(order).toEqual([...order].sort((a, b) => a - b));
  });

  it("marks Add connection as the primary brand action", () => {
    const html = render([connection({ id: "acme" })]);
    expect(button(html, "addConnection")).toContain("bg-brand");
  });

  it("shows a localized context and provider-default effort summary", () => {
    const html = render([connection({ id: "acme" })]);
    expect(html).toContain("modelContext");
    expect(html).toContain("1,000");
    expect(html).toContain("modelEffort: effortDefault");
  });

  it("lists supported efforts and says when the default is also allowed", () => {
    const html = render([connection({
      id: "acme",
      models: [
        { id: "m/1", label: "M", origin: "catalog", efforts: ["low", "high"], no_effort: false, compatible: true, reason: null, context: 50000, effort_template: "reasoning_effort" },
        { id: "m/2", label: "N", origin: "catalog", efforts: ["low"], no_effort: true, compatible: true, reason: null, context: null, effort_template: null },
      ],
    })]);
    expect(html).toContain("modelEfforts: low, high");
    expect(html).toContain("50,000");
    expect(html).toContain("modelEfforts: low · effortDefault");
    expect(html).toContain("modelContext: modelContextUnknown");
  });

  it.each([false, true])("uses universal connection copy, never an editability-derived catalog hint (%s)", (catalog) => {
    const target = { kind: "connection" as const, id: "acme", catalog };
    const html = renderToStaticMarkup(createElement(ProviderRemovalDialog, {
      target, references: [], pending: false, onConfirm: () => {}, onClose: () => {},
    }));
    expect(html).toContain("removeConnectionConfirm");
    expect(html).not.toContain("removeConnectionCatalog");
  });

  it("keeps a referenced connection blocked under the universal confirmation", () => {
    const html = renderToStaticMarkup(createElement(ProviderRemovalDialog, {
      target: { kind: "connection", id: "acme" },
      references: ["default"], pending: false, onConfirm: () => {}, onClose: () => {},
    }));
    expect(html).toContain("removeConnectionConfirm");
    expect(html).not.toContain("removeConnectionCatalog");
    expect(html).toContain("removeBlocked");
    expect(button(html, "remove")).toContain('disabled=""');
  });

  it.each([false, true])("preserves the real model-origin distinction (%s)", (catalog) => {
    const html = renderToStaticMarkup(createElement(ProviderRemovalDialog, {
      target: { kind: "model", id: "acme", model: "m/1", catalog },
      references: ["default"], pending: false, onConfirm: () => {}, onClose: () => {},
    }));
    expect(html).toContain(catalog ? "removeModelCatalogConfirm" : "removeModelConfirm");
    expect(html).toContain("removeBlocked");
    expect(button(html, "remove")).toContain('disabled=""');
  });

  it("removes obsolete connection-only catalog translations and explains both outcomes", () => {
    for (const messages of [en.tools, pt.tools]) {
      expect(messages).not.toHaveProperty("removeConnectionCatalog");
      expect(messages).not.toHaveProperty("removeConnectionCatalogConfirm");
    }
    expect(en.tools.removeConnectionConfirm).toContain("local definition");
    expect(en.tools.removeConnectionConfirm).toContain("hidden from the catalog");
    expect(pt.tools.removeConnectionConfirm).toContain("definição local");
    expect(pt.tools.removeConnectionConfirm).toContain("ocultada do catálogo");
  });

  it("does not invent a default for a model with no supported efforts", () => {
    const html = render([connection({
      id: "acme",
      models: [
        { id: "m/1", label: "M", origin: "catalog", efforts: [], no_effort: false, compatible: true, reason: null, context: 1000, effort_template: null },
      ],
    })]);
    expect(html).toContain("modelEffort: modelEffortUnknown");
    expect(html).not.toContain("modelEffort: effortDefault");
  });
});

function button(html: string, label: string): string {
  return html.match(new RegExp(`<button[^>]*>${label}</button>`))?.[0] ?? "";
}
