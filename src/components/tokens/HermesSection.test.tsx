import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";
import type { HermesMeta } from "@/server/collectors/hermes-meta";

const harness = vi.hoisted(() => ({
  query: { data: undefined as { ok: true; data: HermesMeta } | undefined, isError: false, dataUpdatedAt: 0 },
  hermesTokensEnabled: true,
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
      ? { data: { ok: true, data: { integrations: { hermesTokens: harness.hermesTokensEnabled } } } }
      : { data: undefined },
}));

import { HermesSection } from "./HermesSection";

const budget = {
  platform: "cli",
  model: "x",
  components: [{ key: "systemPrompt" as const, bytes: 10 }],
  totalBytes: 10,
  toolCount: 1,
};

describe("HermesSection", () => {
  it("maps stable prompt errors without raw text", () => {
    harness.query = {
      data: {
        ok: true,
        data: {
          prompt: { ok: false, error: "timeout" },
          skills: { ok: false, error: "SECRET=1" },
        },
      },
      isError: false,
      dataUpdatedAt: 20,
    };
    const html = renderToStaticMarkup(createElement(HermesSection));
    expect(html).toContain("promptTimeout");
    expect(html).not.toContain("SECRET=1");
    expect(html).toContain("skillsUnavailable");
    // no fabricated bars from a failed section payload
    expect(html).not.toContain("systemPrompt");
  });

  it("renders a successful prompt section", () => {
    harness.query = {
      data: {
        ok: true,
        data: {
          prompt: { ok: true, data: budget },
          skills: { ok: true, data: [{ name: "a", useCount: 1, lastUsedAt: null, flagged: false }] },
        },
      },
      isError: false,
      dataUpdatedAt: 10,
    };
    expect(renderToStaticMarkup(createElement(HermesSection))).toContain("systemPrompt");
  });

  it("transport failure keeps the last valid sections visible under a stale warning", () => {
    // TanStack keeps the last successful response after a transport failure, so
    // the retained envelope must still render prompt/skills below the warning.
    harness.query = {
      data: {
        ok: true,
        data: {
          prompt: { ok: true, data: budget },
          skills: { ok: true, data: [{ name: "a", useCount: 1, lastUsedAt: null, flagged: false }] },
        },
      },
      isError: true,
      dataUpdatedAt: 30,
    };
    const html = renderToStaticMarkup(createElement(HermesSection));
    expect(html).toContain("promptUnavailable");
    expect(html).toContain("systemPrompt");
    expect(html).toContain("skillsTitle");
  });

  it("passes enabled:false to useTokensApi when hermesTokens is off (F1)", () => {
    harness.hermesTokensEnabled = false;
    try {
      renderToStaticMarkup(createElement(HermesSection));
      expect(useTokensApiSpy).toHaveBeenCalledWith("hermes", 300_000, false);
    } finally {
      harness.hermesTokensEnabled = true;
    }
  });

  it("passes enabled:true when hermesTokens is on (regression)", () => {
    harness.hermesTokensEnabled = true;
    renderToStaticMarkup(createElement(HermesSection));
    expect(useTokensApiSpy).toHaveBeenCalledWith("hermes", 300_000, true);
  });

  it("renders nothing when hermesTokens is disabled (spec per-integration table)", () => {
    harness.hermesTokensEnabled = false;
    try {
      harness.query = {
        data: {
          ok: true,
          data: {
            prompt: { ok: true, data: budget },
            skills: { ok: true, data: [{ name: "a", useCount: 1, lastUsedAt: null, flagged: false }] },
          },
        },
        isError: false,
        dataUpdatedAt: 10,
      };
      expect(renderToStaticMarkup(createElement(HermesSection))).toBe("");
    } finally {
      harness.hermesTokensEnabled = true;
    }
  });
});
