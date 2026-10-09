import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("./NavList", () => ({ NavList: () => null }));

import { Sidebar } from "./Sidebar";

const OLD_MARK = "rc-" + "mark"; // built so the retired name never appears literally
const render = (mode: "expanded" | "collapsed") => renderToStaticMarkup(createElement(Sidebar, { mode, onToggle: () => {} }));
const imgs = (html: string) => html.match(/<img\b/g)?.length ?? 0;

describe("Sidebar brand surfaces (MOA-511)", () => {
  it("expanded: full lockup pair in the header, Relay pair in the footer, fixed dimensions, no old chrome", () => {
    const html = render("expanded");
    expect(imgs(html)).toBe(4); // lockup dark+light, mark dark+light
    for (const f of ["logo-dark", "logo-light", "mark-dark", "mark-light"]) expect(html).toContain(`src="/brand/${f}.svg"`);
    expect(html).toContain('width="149" height="30"');
    expect(html).toContain('width="29" height="26"');
    expect(html).not.toContain(OLD_MARK);
    expect(html).not.toContain("jax-ring");
    expect(html).not.toContain(">JAX OS<");
    expect(html).not.toContain(">J<");
    expect(html).not.toContain("font-display");
    expect(html).toContain("tagline");
    expect(html).toContain('aria-label="collapse"');
    expect(html).toContain("profileName");
    expect(html).toContain("profileStatus");
    expect(html).toContain('alt=""'); // the footer Relay is decorative next to the profile text
  });
  it("collapsed: Relay pair only (no lockup), footer toggle unchanged, no header control added", () => {
    const html = render("collapsed");
    expect(imgs(html)).toBe(4); // header mark dark+light, footer mark dark+light
    expect(html).not.toContain("/brand/logo-");
    expect(html).toContain('src="/brand/mark-dark.svg"');
    expect(html).toContain('title="profileName"');
    expect(html).toContain('aria-label="hide"');
    expect(html).not.toContain('aria-label="collapse"');
    expect(html).not.toContain(OLD_MARK);
    expect((html.match(/<button\b/g) ?? []).length).toBe(1);
  });
  it("hidden renders nothing", () => {
    expect(renderToStaticMarkup(createElement(Sidebar, { mode: "hidden", onToggle: () => {} }))).toBe("");
  });
});

describe("brand pair theme switching (globals.css)", () => {
  const css = readFileSync(fileURLToPath(new URL("../app/globals.css", import.meta.url)), "utf8");
  it("shows exactly one variant from the server-rendered data-theme, with no OS-theme query", () => {
    expect(css).toContain(".brand-light { display: none; }");
    expect(css).toContain('[data-theme="light"] .brand-dark { display: none; }');
    expect(css).toContain('[data-theme="light"] .brand-light { display: inline; }');
    expect(css).not.toContain("jax-ring");
  });
});

describe("retired assets", () => {
  it("no longer ships the retired RC mark image", () => {
    expect(existsSync(fileURLToPath(new URL(`../../public/${OLD_MARK}.png`, import.meta.url)))).toBe(false);
  });
});
