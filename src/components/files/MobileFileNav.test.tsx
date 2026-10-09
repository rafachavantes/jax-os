import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  scope: { kind: "repo", name: "jax-os" } as { kind: "repo"; name: string } | { kind: "vault" } | { kind: "all" },
  selected: null as { root: "repos" | "vault"; rel: string; isDir: boolean } | null,
  tabs: [] as { root: "repos" | "vault"; rel: string; pinned: boolean }[],
  activeIdx: null as number | null,
  setSelected: vi.fn(), setScope: vi.fn(), openFile: vi.fn(), goBack: vi.fn(),
  treeQuery: { data: undefined as unknown, isError: false, isLoading: false },
  repoNamesQuery: { data: undefined as unknown, isError: false },
  vaultPath: "/tmp/vault" as string | null,
}));

vi.mock("next-intl", () => ({ useTranslations: () => (k: string) => k, useLocale: () => "en-US" }));
vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: unknown[] }) =>
    opts.queryKey[0] === "settings"
      ? { data: { ok: true, data: { integrations: { vault: true }, vaultPath: harness.vaultPath } } }
      : opts.queryKey[1] === "repoNames" ? harness.repoNamesQuery : harness.treeQuery,
}));
vi.mock("./FilesWorkspaceProvider", () => ({ useFilesTreeWorkspace: () => harness }));

import { MobileFileNav } from "./MobileFileNav";

function backButtonTag(html: string): string {
  const tag = html.match(/<button[^>]*aria-label="backToFolders"[^>]*>/)?.[0] ?? "";
  // Strip the class list: the DS `disabled:` variants contain the literal word "disabled" even on
  // an enabled button, which would defeat the substring assertions below.
  return tag.replace(/ class(?:Name)?="[^"]*"/, "");
}

describe("MobileFileNav — 'all' scope (spec §4.3-4.5)", () => {
  it("renders the flat repo+vault picker, disables Back, shows only the fixed Arquivos root crumb", () => {
    harness.scope = { kind: "all" };
    harness.repoNamesQuery = { data: { ok: true, data: { dirs: [{ name: "jax-os" }], files: [] } }, isError: false };
    const html = renderToStaticMarkup(createElement(MobileFileNav));
    expect(html).toContain("jax-os");
    expect(html).toContain("scopeVault");
    expect(html).toContain("filesRoot");
    expect(backButtonTag(html)).toContain("disabled");
  });
  it("hides the vault row when vaultPath is null even though the flag is on (F2)", () => {
    harness.scope = { kind: "all" };
    harness.vaultPath = null;
    try {
      harness.repoNamesQuery = { data: { ok: true, data: { dirs: [{ name: "jax-os" }], files: [] } }, isError: false };
      const html = renderToStaticMarkup(createElement(MobileFileNav));
      expect(html).toContain("jax-os");
      expect(html).not.toContain("scopeVault");
    } finally {
      harness.vaultPath = "/tmp/vault";
    }
  });

  it("enables Back at a repo's own top level (spec §4.5 — was disabled before this plan)", () => {
    harness.scope = { kind: "repo", name: "jax-os" };
    harness.selected = null;
    harness.treeQuery = { data: { ok: true, data: { dirs: [], files: [] } }, isError: false, isLoading: false };
    const html = renderToStaticMarkup(createElement(MobileFileNav));
    expect(backButtonTag(html)).not.toContain("disabled");
  });
});
