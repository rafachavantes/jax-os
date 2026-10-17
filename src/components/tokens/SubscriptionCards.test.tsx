import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  query: { data: undefined as unknown, isError: false, dataUpdatedAt: 0 },
  claudeEnabled: true,
  codexEnabled: true,
  opencodeGoEnabled: true,
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
const useTokensApiSpy = vi.hoisted(() => vi.fn());
vi.mock("@/lib/useTokensApi", () => ({
  useTokensApi: (...args: unknown[]) => {
    useTokensApiSpy(...args);
    return harness.query;
  },
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: unknown[] }) =>
    opts.queryKey[0] === "settings"
      ? {
          data: {
            ok: true,
            data: {
              integrations: {
                agents: {
                  claude: harness.claudeEnabled,
                  codex: harness.codexEnabled,
                  opencode: harness.opencodeGoEnabled,
                },
              },
            },
          },
        }
      : { data: undefined },
}));

import { SubscriptionCards } from "./SubscriptionCards";

describe("SubscriptionCards", () => {
  it("shows last provider usage under a section error", () => {
    harness.query = {
      data: {
        ok: true,
        data: {
          anthropic: { ok: false, error: "expired" },
          codex: { ok: true, data: { provider: "codex", plan: "plus", fiveHour: null, weekly: null, models: [] } },
          opencode: { ok: true, data: { provider: "opencode", plan: null, fiveHour: null, weekly: null, monthly: null, models: [] } },
        },
      },
      isError: false,
      dataUpdatedAt: 8,
    };
    const html = renderToStaticMarkup(createElement(SubscriptionCards));
    expect(html).toContain("unavailable");
    expect(html).toContain("plus");
  });

  it("shows the missing-credentials hint for OpenCode GO and renders its monthly window", () => {
    harness.query = {
      data: {
        ok: true,
        data: {
          anthropic: { ok: true, data: { provider: "anthropic", plan: "max", fiveHour: null, weekly: null, models: [] } },
          codex: { ok: true, data: { provider: "codex", plan: "plus", fiveHour: null, weekly: null, models: [] } },
          opencode: { ok: false, error: "missing-credentials" },
        },
      },
      isError: false,
      dataUpdatedAt: 8,
    };
    const html = renderToStaticMarkup(createElement(SubscriptionCards));
    expect(html).toContain("errors.missing-credentials.opencode");
  });

  it("enabled:false when all three subscription flags are off (F1)", () => {
    harness.claudeEnabled = false;
    harness.codexEnabled = false;
    harness.opencodeGoEnabled = false;
    try {
      renderToStaticMarkup(createElement(SubscriptionCards));
      expect(useTokensApiSpy).toHaveBeenCalledWith("subscriptions", 60_000, false);
    } finally {
      harness.claudeEnabled = true;
      harness.codexEnabled = true;
      harness.opencodeGoEnabled = true;
    }
  });

  it("enabled:true when at least one flag is on (regression)", () => {
    harness.claudeEnabled = false;
    harness.codexEnabled = true;
    harness.opencodeGoEnabled = false;
    try {
      renderToStaticMarkup(createElement(SubscriptionCards));
      expect(useTokensApiSpy).toHaveBeenCalledWith("subscriptions", 60_000, true);
    } finally {
      harness.claudeEnabled = true;
      harness.codexEnabled = true;
      harness.opencodeGoEnabled = true;
    }
  });

  it.each([
    { flag: "claudeEnabled", hidden: "Anthropic" },
    { flag: "codexEnabled", hidden: "OpenAI Codex" },
    { flag: "opencodeGoEnabled", hidden: "OpenCode GO" },
  ] as const)("omits the $hidden card when its subscription flag is off, keeps the other two (spec per-integration table)", ({ flag, hidden }) => {
    harness[flag] = false;
    try {
      harness.query = {
        data: {
          ok: true,
          data: {
            anthropic: { ok: true, data: { provider: "anthropic", plan: "max", fiveHour: null, weekly: null, models: [] } },
            codex: { ok: true, data: { provider: "codex", plan: "plus", fiveHour: null, weekly: null, models: [] } },
            opencode: { ok: true, data: { provider: "opencode", plan: "go", fiveHour: null, weekly: null, monthly: null, models: [] } },
          },
        },
        isError: false,
        dataUpdatedAt: 8,
      };
      const html = renderToStaticMarkup(createElement(SubscriptionCards));
      expect(html).not.toContain(hidden);
      for (const name of ["Anthropic", "OpenAI Codex", "OpenCode GO"]) {
        if (name !== hidden) expect(html).toContain(name);
      }
    } finally {
      harness[flag] = true;
    }
  });
});
