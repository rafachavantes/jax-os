import { afterEach, describe, expect, it, vi } from "vitest";
import { bannerLabelKey } from "../lib/settingsBanner";

describe("settings warning banner (spec: ONE app-wide warning banner)", () => {
  it("names the malformed key on settings-malformed", () => {
    expect(bannerLabelKey({ ok: false, error: "settings-malformed" })).toBe("settings.malformed");
  });
  it("names the unreadable key on settings-unreadable", () => {
    expect(bannerLabelKey({ ok: false, error: "settings-unreadable" })).toBe("settings.unreadable");
  });
  it("is null (no banner) when settings are valid or absent", () => {
    expect(bannerLabelKey({ ok: true, data: {} as never })).toBeNull();
  });
});

import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";

const harness = vi.hoisted(() => ({
  settingsResult: { ok: true, data: {} } as { ok: true; data: unknown } | { ok: false; error: "settings-malformed" | "settings-unreadable" },
}));

vi.mock("next-intl", () => ({
  NextIntlClientProvider: ({ children }: { children: React.ReactNode }) => children,
}));
vi.mock("next-intl/server", () => ({
  getLocale: async () => "en-US",
  getTranslations: async () => (key: string) => key,
}));
vi.mock("next/headers", () => ({
  cookies: async () => ({ get: () => undefined }),
}));
vi.mock("@/server/settings", () => ({
  readGeneralSettings: () => harness.settingsResult,
}));
vi.mock("@/components/Shell", () => ({
  Shell: ({ children }: { children: React.ReactNode }) => createElement("div", { "data-testid": "shell" }, children),
}));
vi.mock("./Providers", () => ({ Providers: ({ children }: { children: React.ReactNode }) => children }));
vi.mock("./fonts", () => ({ fontVariables: "" }));
vi.mock("./globals.css", () => ({}));
vi.mock("@/components/AgentsBanner", () => ({ AgentsBanner: () => createElement("div", { "data-testid": "agents-banner" }) }));

const { default: RootLayout } = await import("./layout");

afterEach(() => { harness.settingsResult = { ok: true, data: {} }; });

describe("RootLayout settings warning banner — actual wiring, not just the key (spec: ONE app-wide warning banner)", () => {
  it("renders the translated warning when settings are malformed", async () => {
    harness.settingsResult = { ok: false, error: "settings-malformed" };
    const html = renderToStaticMarkup(await RootLayout({ children: createElement("div") }));
    expect(html).toContain("settings.malformed");
  });
  it("renders the translated warning when settings are unreadable", async () => {
    harness.settingsResult = { ok: false, error: "settings-unreadable" };
    const html = renderToStaticMarkup(await RootLayout({ children: createElement("div") }));
    expect(html).toContain("settings.unreadable");
  });
  it("renders no banner when settings are absent (ok:true, default data)", async () => {
    harness.settingsResult = { ok: true, data: {} };
    const html = renderToStaticMarkup(await RootLayout({ children: createElement("div") }));
    expect(html).not.toContain("settings.malformed");
    expect(html).not.toContain("settings.unreadable");
  });
  it("renders no banner when settings are valid (ok:true, real data)", async () => {
    harness.settingsResult = { ok: true, data: { reposRoot: "/tmp/x", vaultPath: null } };
    const html = renderToStaticMarkup(await RootLayout({ children: createElement("div") }));
    expect(html).not.toContain("settings.malformed");
    expect(html).not.toContain("settings.unreadable");
  });
  it("mounts the client AgentsBanner inside the shell", async () => {
    const html = renderToStaticMarkup(await RootLayout({ children: createElement("div") }));
    expect(html).toContain('data-testid="agents-banner"');
  });
});

describe("RootLayout metadata (MOA-501 D3)", () => {
  it("describes the product neutrally and registers no manual icons", async () => {
    const { metadata } = await import("./layout");
    expect(metadata.title).toBe("Jax OS");
    expect(metadata.description).toBe("Self-hosted work cockpit. You watch; your agents do the work.");
    expect(metadata).not.toHaveProperty("icons");
  });
});
