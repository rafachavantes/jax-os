import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  query: { data: undefined as { ok: true; data: { enabled: boolean } } | undefined, isError: false },
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("@tanstack/react-query", () => ({
  useQuery: () => harness.query,
  useQueryClient: () => ({ setQueryData: vi.fn() }),
}));

import { AfkToggle } from "./AfkToggle";

describe("AfkToggle", () => {
  it("renders a switch with checked state, a live status only while loading", () => {
    harness.query = { data: { ok: true, data: { enabled: true } }, isError: false };
    const html = renderToStaticMarkup(createElement(AfkToggle));
    expect(html).toContain('role="switch"');
    expect(html).toContain('aria-checked="true"');
    // AFK on shows no caption: the switch alone says it (the on text lives only in the tooltip).
    expect(html).not.toContain('role="status"');
    harness.query = { data: undefined, isError: false };
    expect(renderToStaticMarkup(createElement(AfkToggle))).toContain('role="status"');
  });

  it("disables the switch until the value is known", () => {
    harness.query = { data: undefined, isError: false };
    expect(renderToStaticMarkup(createElement(AfkToggle))).toContain("disabled");
    harness.query = { data: { ok: true, data: { enabled: false } }, isError: false };
    expect(renderToStaticMarkup(createElement(AfkToggle))).toContain('aria-checked="false"');
  });
});