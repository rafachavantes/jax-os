import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  query: { data: undefined as unknown, isError: false, dataUpdatedAt: 0 },
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@/lib/useTokensApi", () => ({
  useTokensApi: () => harness.query,
}));

import { UsageSection } from "./UsageSection";

describe("UsageSection", () => {
  it("shows last usage under a warning", () => {
    harness.query = {
      data: { ok: true, data: { totalTokens: 10, totalCost: 0, daily: [], byModel: [], byAgent: [], skipped: [] } },
      isError: true,
      dataUpdatedAt: 9,
    };
    const html = renderToStaticMarkup(createElement(UsageSection));
    expect(html).toContain("unavailable");
    expect(html).toContain("dailyBurn");
  });
});
