import { createElement } from "react";
import { NextIntlClientProvider } from "next-intl";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import en from "../../messages/en-US.json";
import pt from "../../messages/pt-BR.json";

const harness = vi.hoisted(() => ({ query: { data: undefined as unknown } }));
vi.mock("@tanstack/react-query", () => ({ useQuery: () => harness.query }));

import { AgentsBanner } from "./AgentsBanner";

// renderToStaticMarkup HTML-escapes the ">" of "Settings -> General".
const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
const agents = (claude: boolean, codex: boolean, opencode: boolean) =>
  ({ data: { ok: true, data: { integrations: { agents: { claude, codex, opencode } } } } });

function render(locale: "en-US" | "pt-BR", query: { data: unknown }) {
  harness.query = query;
  return renderToStaticMarkup(createElement(NextIntlClientProvider, {
    locale, messages: locale === "pt-BR" ? pt : en, timeZone: "UTC", children: createElement(AgentsBanner),
  }));
}

describe.each([["en-US", en], ["pt-BR", pt]] as const)("AgentsBanner in %s", (locale, m) => {
  it("all off: shows noAgents ONLY", () => {
    const html = render(locale, agents(false, false, false));
    expect(html).toContain(esc(m.settings.noAgents));
    expect(html).not.toContain(esc(m.settings.noReviewer));
  });
  it("OpenCode only: shows noReviewer (review cannot run, builds still work) and never both", () => {
    const html = render(locale, agents(false, false, true));
    expect(html).toContain(esc(m.settings.noReviewer));
    expect(html).not.toContain(esc(m.settings.noAgents));
  });
  it("renders nothing with a reviewer agent on, while loading, and on a settings error", () => {
    expect(render(locale, agents(true, false, false))).toBe("");
    expect(render(locale, { data: undefined })).toBe("");
    expect(render(locale, { data: { ok: false, error: "settings-malformed" } })).toBe("");
  });
});
