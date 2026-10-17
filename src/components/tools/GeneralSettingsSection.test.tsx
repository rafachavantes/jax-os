import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import en from "../../../messages/en-US.json";
import pt from "../../../messages/pt-BR.json";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string, vars?: Record<string, unknown>) => (vars ? `${key}:${JSON.stringify(vars)}` : key) }));

// vi.hoisted, not a plain const — a vi.mock factory runs before any of this file's own top-level
// `const`s initialize, so the closure below must close over a binding vi.mock's own hoisting
// already guarantees exists (same fix as F2's route-test harness).
const { useQueryMock } = vi.hoisted(() => ({ useQueryMock: vi.fn() }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => useQueryMock(),
  useQueryClient: () => ({ invalidateQueries: vi.fn(), setQueryData: vi.fn(), getQueryData: () => undefined }),
}));

import { GeneralSettingsSection } from "./GeneralSettingsSection";

const BASE_SETTINGS = {
  ownerName: "", locale: "en-US", reposRoot: "/home/x/repos", vaultPath: null,
  monitoredUnits: { user: [], system: [] },
  integrations: { linear: false, ttyd: false, webhook: false, hermesTokens: false,
    agents: { claude: false, codex: false, opencode: false }, classifier: false, github: false, vault: false },
};
useQueryMock.mockReturnValue({ data: { ok: true, data: BASE_SETTINGS, webhookConfigured: false }, isLoading: false, isError: false });

describe("GeneralSettingsSection — fresh install (every toggle off)", () => {
  it("renders every integration row, all unchecked, discoverable even though off (spec: Geral's own toggles stay rendered)", () => {
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    for (const flag of ["linear", "ttyd", "hermesTokens", "agentClaude", "agentCodex", "agentOpenCode", "classifier", "github", "vault"]) {
      expect(html).toContain(`data-integration="${flag}"`);
    }
    expect(html).not.toContain('aria-checked="true"'); // no toggle pre-checked on a fresh install
  });

  it("renders the owner/locale/paths/monitoring/notifications sections", () => {
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).toContain('name="ownerName"');
    expect(html).toContain('name="reposRoot"');
    expect(html).toContain('name="vaultPath"');
  });

  it("the notifications row keeps the webhook toggle editable AND shows a read-only missing status (F1)", () => {
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).toContain('data-integration="webhook"');
    expect(html).toContain('data-webhook-status="missing"');
    expect(html).not.toContain("NOTIFICATION_WEBHOOK"); // no URL/secret field anywhere (Non-goals)
  });
});

describe("GeneralSettingsSection — webhook env vars both set", () => {
  it("shows the configured status while the toggle stays independently editable", () => {
    useQueryMock.mockReturnValueOnce({ data: { ok: true, data: BASE_SETTINGS, webhookConfigured: true }, isLoading: false, isError: false });
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).toContain('data-webhook-status="configured"');
  });

  it("shows the env file status next to the webhook status", () => {
    useQueryMock.mockReturnValueOnce({
      data: { ok: true, data: BASE_SETTINGS, webhookConfigured: true, envFileState: "wrong-owner" },
      isLoading: false, isError: false,
    });
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).toContain('data-env-file-status="wrong-owner"');
  });
});

describe("GeneralSettingsSection — native credential status (diff review e52d9e6dc555 F3)", () => {
  it("shows configured/missing status for the three native credentials (F3)", () => {
    useQueryMock.mockReturnValueOnce({
      data: {
        ok: true, data: BASE_SETTINGS, webhookConfigured: false,
        credentialsConfigured: { claude: true, codex: false, opencodeGo: false },
      },
      isLoading: false, isError: false,
    });
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).toContain('data-credential-status="configured"'); // claude row
    expect(html).toContain('data-credential-status="missing"'); // codex + opencodeGo rows
  });
});

describe("GeneralSettingsSection — toggles render as switches", () => {
  it("uses role=switch (the app-wide toggle look), not a raw checkbox", () => {
    useQueryMock.mockReturnValueOnce({ data: { ok: true, data: BASE_SETTINGS, webhookConfigured: false }, isLoading: false, isError: false });
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).not.toContain('type="checkbox"');
    expect(html.match(/role="switch"/g)?.length).toBe(10); // 9 integration rows + webhook
  });
});

describe("GeneralSettingsSection — settings-malformed", () => {
  it("renders the form backed by defaults plus a visible warning (repairable via Save)", () => {
    useQueryMock.mockReturnValueOnce({ data: { ok: false, error: "settings-malformed" }, isLoading: false, isError: false });
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).toContain('name="ownerName"');
    expect(html).toContain("general.settingsMalformed");
    // Save is enabled even with no local edits — no `disabled` HTML attribute on its <button>.
    expect(html).not.toMatch(/<button[^>]*\bdisabled\b(?!:)[^>]*>save</);
  });
});

describe("GeneralSettingsSection — Agents block (MOA-504 D5)", () => {
  it("renders the Agents heading + hint above Integrations, three agent rows, and no subs* rows", () => {
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).toContain("general.agentsHeading");
    expect(html).toContain("general.agentsHint");
    expect(html.indexOf("general.agentsHeading")).toBeLessThan(html.indexOf("general.integrationsHeading"));
    for (const flag of ["agentClaude", "agentCodex", "agentOpenCode"]) expect(html).toContain(`data-integration="${flag}"`);
    expect(html).not.toContain("subsClaude");
    expect(html).not.toContain("general.subs");
  });

  it("checks the switch of an agent that is on", () => {
    useQueryMock.mockReturnValueOnce({
      data: { ok: true, data: { ...BASE_SETTINGS, integrations: { ...BASE_SETTINGS.integrations, agents: { claude: true, codex: false, opencode: false } } }, webhookConfigured: false },
      isLoading: false, isError: false,
    });
    expect(renderToStaticMarkup(createElement(GeneralSettingsSection)).match(/aria-checked="true"/g)?.length).toBe(1);
  });

  it("has the new keys in both locales and no stale subs* keys", () => {
    for (const m of [en, pt]) {
      for (const k of ["agentsHeading", "agentsHint", "agentClaudeLabel", "agentCodexLabel", "agentOpenCodeLabel", "meterConfiguredStatus", "meterMissingStatus"]) {
        expect((m.tools.general as unknown as Record<string, string>)[k]).toBeTruthy();
      }
      expect(m.tools.general).not.toHaveProperty("subsClaudeLabel");
    }
  });
});

describe("GeneralSettingsSection — settings-unreadable", () => {
  it("renders a refusal message with no form", () => {
    useQueryMock.mockReturnValueOnce({ data: { ok: false, error: "settings-unreadable" }, isLoading: false, isError: false });
    const html = renderToStaticMarkup(createElement(GeneralSettingsSection));
    expect(html).not.toContain('name="ownerName"');
    expect(html).toContain("general.settingsUnreadable");
  });
});
