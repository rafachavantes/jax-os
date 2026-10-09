import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
const harness = vi.hoisted(() => ({ settings: undefined as unknown }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts?: { queryKey?: unknown[] }) => ({
    data: opts?.queryKey?.[0] === "settings" ? harness.settings : undefined,
    isError: false, dataUpdatedAt: 0, error: undefined,
  }),
  useQueryClient: () => ({ invalidateQueries: vi.fn() }),
}));
vi.mock("@/components/rules/RulesSection", () => ({ RulesSection: () => createElement("div", null, "rules") }));
vi.mock("@/components/tools/InventorySection", () => ({ InventorySection: () => createElement("div", null, "inventory") }));

import en from "../../../messages/en-US.json";
import pt from "../../../messages/pt-BR.json";
import SettingsPage from "./page";

describe("SettingsPage tabs", () => {
  it("keeps the mockup tab order and mounts every panel", () => {
    const html = renderToStaticMarkup(createElement(SettingsPage));
    const order = ["agents", "providers", "rules", "inventory"].map((id) => html.indexOf(`tools-tab-${id}`));
    expect(order.every((index) => index >= 0)).toBe(true);
    expect([...order].sort((a, b) => a - b)).toEqual(order);
    for (const id of ["agents", "providers", "rules", "inventory"]) {
      expect(html).toContain(`id="tools-panel-${id}"`);
    }
  });

  it("renders a general tab button first, before agents", () => {
    const html = renderToStaticMarkup(createElement(SettingsPage));
    const generalIdx = html.indexOf('id="tools-tab-general"');
    const agentsIdx = html.indexOf('id="tools-tab-agents"');
    expect(generalIdx).toBeGreaterThan(-1);
    expect(generalIdx).toBeLessThan(agentsIdx);
  });

  it("renders the approved page heading, subtitle and tab icons", () => {
    const html = renderToStaticMarkup(createElement(SettingsPage));
    expect(html).toContain("heading");
    expect(html).toContain("headingSubtitle");
    for (const id of ["agents", "providers", "rules", "inventory"]) {
      expect(html).toContain(`tool-icon-${id}`);
    }
  });

  // The render mock above returns message KEYS, so the visible copy itself can only be checked
  // at the message files (this repo's convention for copy-only changes: mission.test.ts and the
  // tools sections' own tests import both locales' JSON the same way).
  it("shows the settings title/subtitle, not the stale agents-and-tools copy (visual fix)", () => {
    expect(en.tools.heading).toBe(en.title.tools);
    expect(en.tools.headingSubtitle).toBe(en.subtitle.tools);
    expect(pt.tools.heading).toBe(pt.title.tools);
    expect(pt.tools.headingSubtitle).toBe(pt.subtitle.tools);
    expect(en.tools.heading).not.toBe("Agents and tools");
    expect(pt.tools.heading).not.toBe("Agentes e ferramentas");
  });
});

const settingsWith = (opencode: boolean) => ({
  ok: true, webhookConfigured: false, envFileState: "ok",
  data: {
    ownerName: "", locale: "en-US", reposRoot: "/x", vaultPath: null, monitoredUnits: { user: [], system: [] },
    integrations: { linear: false, ttyd: false, webhook: false, hermesTokens: false,
      agents: { claude: true, codex: true, opencode }, classifier: false, github: false, vault: false },
  },
});

describe("SettingsPage — Providers tab follows OpenCode (MOA-504 D10)", () => {
  afterEach(() => { harness.settings = undefined; });
  it("OpenCode off: no providers tab button and no providers panel; the other tabs stay", () => {
    harness.settings = settingsWith(false);
    const html = renderToStaticMarkup(createElement(SettingsPage));
    expect(html).not.toContain('id="tools-tab-providers"');
    expect(html).not.toContain('id="tools-panel-providers"');
    for (const id of ["general", "agents", "rules", "inventory"]) expect(html).toContain(`id="tools-tab-${id}"`);
  });
  it("OpenCode on, or settings still unknown (fail open): the providers tab is there", () => {
    harness.settings = settingsWith(true);
    expect(renderToStaticMarkup(createElement(SettingsPage))).toContain('id="tools-tab-providers"');
    harness.settings = undefined;
    expect(renderToStaticMarkup(createElement(SettingsPage))).toContain('id="tools-tab-providers"');
  });
});
