import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
  useLocale: () => "en-US",
}));
vi.mock("./files/FilesWorkspaceProvider", () => ({
  useFilesEditWorkspace: () => ({ dirty: false }),
  UnsavedFilesNotice: () => null,
}));
vi.mock("./Sidebar", () => ({ Sidebar: () => null }));
vi.mock("./Topbar", () => ({ Topbar: () => null }));

import { Shell } from "./Shell";

describe("Shell", () => {
  it("provides a skip link to a focusable main landmark", () => {
    const html = renderToStaticMarkup(
      createElement(Shell, { initialSidebar: "expanded", initialTheme: "dark", children: null }),
    );
    expect(html).toContain('href="#main-content"');
    expect(html).toContain("skipToContent");
    expect(html).toContain('id="main-content"');
    expect(html).toContain('tabindex="-1"');
    expect(html).toContain("<main");
  });

  it("no longer imports the combined useFilesWorkspace hook (spec §3 — must use the narrow edit hook)", () => {
    const src = readFileSync(fileURLToPath(new URL("./Shell.tsx", import.meta.url)), "utf8");
    expect(src).not.toMatch(/\buseFilesWorkspace\b/);
    expect(src).toContain("useFilesEditWorkspace");
  });
});