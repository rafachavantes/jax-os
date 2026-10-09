import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key, useLocale: () => "en-US" }));
vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh: () => {} }) }));
vi.mock("./mission/AfkToggle", () => ({ AfkToggle: () => createElement("div", { "data-testid": "afk-toggle" }) }));

import { MobileNavAfkRow, MobileNavBrand, MobileNavControls } from "./MobileNav";

// MobileNav's drawer itself only mounts once `open` (internal useState) is true, which
// a static render can't simulate — MobileNavControls is the exact drawer footer, exported
// so the moved controls can be asserted directly (Topbar.test.tsx covers the topbar side).
describe("MobileNavControls — the drawer renders both controls moved off the topbar", () => {
  it("renders the language selector and the theme toggle", () => {
    const html = renderToStaticMarkup(createElement(MobileNavControls, { initialTheme: "dark" }));
    expect(html).toContain(">PT<");
    expect(html).toContain(">EN<");
    expect(html).toContain('aria-label="toggleTheme"');
  });
});

// Same exported-for-testability seam as MobileNavControls — MobileNavAfkRow is the exact
// drawer row, rendered on every route.
describe("MobileNavAfkRow — the AFK toggle moved off InboxStrip into the drawer", () => {
  it("renders the AFK toggle", () => {
    const html = renderToStaticMarkup(createElement(MobileNavAfkRow));
    expect(html).toContain('data-testid="afk-toggle"');
  });
});

// Same exported-for-testability seam: the drawer header (lockup + close control) can be asserted
// without simulating the open click.
describe("MobileNavBrand — the drawer header carries the full lockup and keeps the close control", () => {
  it("renders the lockup pair at fixed size and the close button with its label", () => {
    const html = renderToStaticMarkup(createElement(MobileNavBrand, { onClose: () => {} }));
    expect(html).toContain('src="/brand/logo-dark.svg"');
    expect(html).toContain('src="/brand/logo-light.svg"');
    expect(html).toContain('width="149" height="30"');
    expect(html).toContain('aria-label="closeMenu"');
    expect(html).not.toContain("font-display");
    expect(html).not.toContain(">JAX OS<");
    expect(html).not.toContain("rc-" + "mark");
  });
});
