import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key }));
vi.mock("next/navigation", () => ({ usePathname: () => "/" }));
vi.mock("next/link", () => ({
  default: ({ href, children, className, title }: { href: string; children: React.ReactNode; className?: string; title?: string }) =>
    createElement("a", { href, className, title }, children),
}));
vi.mock("@/components/files/FilesWorkspaceProvider", () => ({ useFilesEditWorkspace: () => ({ dirty: false }) }));

const harness = vi.hoisted(() => ({ linearEnabled: true }));
vi.mock("@/lib/settingsQuery", () => ({
  useGeneralSettings: () => ({ data: { ok: true, data: { integrations: { linear: harness.linearEnabled } } } }),
}));

import { NavList } from "./NavList";

describe("NavList — hidden nav items (round-1 F9)", () => {
  it("omits /health and /audit in both expanded and collapsed mode, expanded keeping every other route", () => {
    const expanded = renderToStaticMarkup(createElement(NavList, { mode: "expanded" }));
    expect(expanded).not.toContain('href="/health"');
    expect(expanded).not.toContain('href="/audit"');
    for (const href of ["/", "/kanban", "/tmux", "/files", "/tokens", "/settings"]) expect(expanded).toContain(`href="${href}"`);

    const collapsed = renderToStaticMarkup(createElement(NavList, { mode: "collapsed" }));
    expect(collapsed).not.toContain('href="/health"');
    expect(collapsed).not.toContain('href="/audit"');
    expect(collapsed).toContain('href="/"');
  });

  it("no longer imports the combined useFilesWorkspace hook (spec §3 — must use the narrow edit hook)", () => {
    const src = readFileSync(fileURLToPath(new URL("./NavList.tsx", import.meta.url)), "utf8");
    expect(src).not.toMatch(/\buseFilesWorkspace\b/);
    expect(src).toContain("useFilesEditWorkspace");
  });

  it("omits /kanban when linear is disabled (spec per-integration table)", () => {
    harness.linearEnabled = false;
    try {
      expect(renderToStaticMarkup(createElement(NavList, { mode: "expanded" }))).not.toContain('href="/kanban"');
      expect(renderToStaticMarkup(createElement(NavList, { mode: "collapsed" }))).not.toContain('href="/kanban"');
    } finally {
      harness.linearEnabled = true;
    }
  });

  it("keeps /kanban when linear is enabled (regression)", () => {
    expect(renderToStaticMarkup(createElement(NavList, { mode: "expanded" }))).toContain('href="/kanban"');
  });
});
