import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

vi.mock("next/link", () => ({
  default: ({ href, children }: { href: string; children: React.ReactNode }) =>
    createElement("a", { href }, children),
}));

import { FilesWorkspaceProvider, UnsavedFilesNotice } from "./FilesWorkspaceProvider";

describe("UnsavedFilesNotice", () => {
  it("announces politely and links back to Files", () => {
    const html = renderToStaticMarkup(createElement(UnsavedFilesNotice));
    expect(html).toContain("aria-live=\"polite\"");
    expect(html).toContain("unsavedNotice");
    expect(html).toContain("returnToFiles");
    expect(html).toContain('href="/files"');
  });
});

describe("FilesWorkspaceProvider", () => {
  it("renders children without wrapping a notice above the shell", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    const html = renderToStaticMarkup(
      createElement(QueryClientProvider, { client },
        createElement(FilesWorkspaceProvider, null, createElement("main", null, "page")),
      ),
    );
    expect(html).toContain("page");
    expect(html).not.toContain("unsavedNotice");
    expect(html).not.toContain("returnToFiles");
  });

  it("persists only tree-shaped UI state so content edits do not serialize", () => {
    const src = readFileSync(fileURLToPath(new URL("./FilesWorkspaceProvider.tsx", import.meta.url)), "utf8");
    expect(src).toContain("}, [treeSnap]);");
  });

  it("splits the context into a tree half and an edit half, each memoized off its own snapshot (spec §3)", () => {
    const src = readFileSync(fileURLToPath(new URL("./FilesWorkspaceProvider.tsx", import.meta.url)), "utf8");
    expect(src).toContain("<FilesTreeContext.Provider value={treeValue}>");
    expect(src).toContain("<FilesEditContext.Provider value={editValue}>");
    const treeMemo = src.match(/const treeValue[^=]*=\s*useMemo\(\s*\(\)\s*=>\s*\(\{[\s\S]*?\}\),\s*\[([^\]]*)\]\s*\);/);
    expect(treeMemo).not.toBeNull();
    const treeDeps = treeMemo![1];
    expect(treeDeps).toContain("treeSnap");
    expect(treeDeps).not.toMatch(/editSnap|dirty|dirtyKeys/);
    const editMemo = src.match(/const editValue[^=]*=\s*useMemo\(\s*\(\)\s*=>\s*\(\{[\s\S]*?\}\),\s*\[([^\]]*)\]\s*\);/);
    expect(editMemo).not.toBeNull();
    expect(editMemo![1]).toContain("editSnap");
  });
});
