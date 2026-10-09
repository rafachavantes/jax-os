import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

import { TabBar } from "./TabBar";

describe("TabBar", () => {
  it("keeps dirty-tab close on the keyboard focus path", () => {
    const html = renderToStaticMarkup(createElement(TabBar, {
      tabs: [{ root: "repos", rel: "a.ts", pinned: true }],
      activeIdx: 0,
      dirtyKeys: new Set(["repos:a.ts"]),
      onSelect: () => {},
      onPin: () => {},
      onClose: () => {},
    }));
    expect(html).toContain("closeTab");
    expect(html).toContain("focus-visible:opacity-100");
    expect(html).toContain("unsaved");
    expect(html).not.toContain("hidden group-hover:block");
  });

  it("shows the disambiguated label, not the bare basename, for two tabs sharing a name", () => {
    const html = renderToStaticMarkup(createElement(TabBar, {
      tabs: [
        { root: "repos", rel: "dirA/x.ts", pinned: true },
        { root: "repos", rel: "dirB/x.ts", pinned: true },
      ],
      activeIdx: 0,
      dirtyKeys: new Set<string>(),
      onSelect: () => {},
      onPin: () => {},
      onClose: () => {},
    }));
    expect(html).toContain(">x.ts<");
    expect(html).toContain(">dirA<");
    expect(html).toContain(">dirB<");
  });
});
