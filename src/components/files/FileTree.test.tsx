import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it, vi } from "vitest";

const harness = vi.hoisted(() => ({
  opNotes: [] as { id: string; tone: string; text: string; sticky: boolean; undo?: unknown }[],
  locked: false,
  inline: null as unknown,
  items: [] as unknown[],
  treeSweep: null as { version: number; root: string; rel: string } | null,
  scope: { kind: "repo", name: "jax-os" } as { kind: "repo"; name: string } | { kind: "vault" } | { kind: "all" },
  selected: null as { root: string; rel: string; isDir: boolean } | null,
  setSelected: vi.fn(),
  setScope: vi.fn(),
  repoNamesQuery: { data: undefined as unknown, isError: false },
  vaultPath: "/tmp/vault" as string | null,
  beginCreate: vi.fn(),
  beginRename: vi.fn(),
  cancelInline: vi.fn(),
  commitInline: vi.fn(),
  remove: vi.fn(),
  undoTrash: vi.fn(),
  downloadZip: vi.fn(),
  uploadMany: vi.fn(),
  isLocked: (root: string, rel: string) => harness.locked && root === "repos" && rel === "",
}));

vi.mock("next-intl", () => ({
  useTranslations: () => (key: string) => key,
}));

const fakeTree = {
  getContainerProps: () => ({ role: "tree" }),
  getItems: () => harness.items ?? [],
};
// Captures the REAL config passed to useTree() — a mock ignoring it would pass even
// with the data loader/expansion bridge/onPrimaryAction ripped out (round 1 F5).
let capturedConfig: Record<string, unknown> | null = null;
vi.mock("@headless-tree/react", () => ({
  useTree: (config: Record<string, unknown>) => { capturedConfig = config; return fakeTree; },
}));
vi.mock("@headless-tree/core", () => ({ asyncDataLoaderFeature: {}, hotkeysCoreFeature: {}, selectionFeature: {} }));

vi.mock("./FilesWorkspaceProvider", () => ({
  useFilesTreeWorkspace: () => harness,
}));
// Item 1/2 (spec): FileTree now renders its own sticky RepoSwitcher+breadcrumb header. RepoSwitcher
// fetches via useQuery (no QueryClientProvider in this harness) — stubbed out, same as fakeTree
// keeps @headless-tree out of scope; this suite tests FileTree's OWN row/toolbar rendering only.
vi.mock("./RepoSwitcher", () => ({ RepoSwitcher: () => null }));

vi.mock("@tanstack/react-query", () => ({
  useQuery: (opts: { queryKey: unknown[] }) =>
    opts.queryKey[0] === "settings"
      ? { data: { ok: true, data: { integrations: { vault: true }, vaultPath: harness.vaultPath } } }
      : harness.repoNamesQuery,
}));
vi.mock("../../lib/fileSearch", () => ({
  fetchRepoNames: () => Promise.resolve(harness.repoNamesQuery.data),
  repoNamesFromListing: (listing: { dirs: { name: string }[] }) => listing.dirs.map((d) => d.name),
}));

import { buildTreeChildren, commonItems, diffExpanded, dirMenuItems, fetchDirChildren, FileTree, gitStatusTint, iconForEntry, inlineRowProps, invalidateRootOnScopeChange, pickActiveMenu, refreshExpandedDirectories, reloadAffectedDirectory, revealAncestorsSequential, scopeRootId, undoHandler } from "./FileTree";
import { File, FileArchive, FileAudio, FileCode, FileImage, FileJson, FileVideo, Folder, FolderOpen } from "lucide-react";
import { ContextMenu } from "./ContextMenu";
import ptBR from "../../../messages/pt-BR.json";

type RenderOverrides = Partial<{ expanded: Set<string>; onToggle: (key: string) => void; onOpenFile: (root: "repos" | "vault", rel: string) => void; onOpenFilePinned: (root: "repos" | "vault", rel: string) => void }>;
function render(activeFile: { root: "repos" | "vault"; rel: string } | null = null, overrides: RenderOverrides = {}) {
  return renderToStaticMarkup(createElement(FileTree, {
    expanded: overrides.expanded ?? new Set<string>(),
    onToggle: overrides.onToggle ?? (() => {}),
    onOpenFile: overrides.onOpenFile ?? (() => {}),
    onOpenFilePinned: overrides.onOpenFilePinned ?? (() => {}),
    activeFile,
  }));
}

function fakeItem(data: ReturnType<typeof mkNode>, level: number, expanded: boolean, loading = false) {
  return {
    getId: () => data.id,
    getItemData: () => data,
    getItemMeta: () => ({ level }),
    isExpanded: () => expanded,
    isLoading: () => loading,
    getProps: () => ({ "aria-label": data.name }),
    registerElement: () => {},
  };
}
function mkNode(over: Partial<{ id: string; root: "repos" | "vault"; rel: string; name: string; isDir: boolean; gitStatus: "modified" | "added" | "deleted" | "untracked" | null; parentRel: string }>) {
  return { id: "repos:a.ts", root: "repos" as const, rel: "a.ts", name: "a.ts", isDir: false, gitStatus: null, parentRel: "", ...over };
}

describe("FileTree", () => {
  it("shows pending, confirmed, and error tree status and a '···' trigger on every row", () => {
    harness.opNotes = [
      { id: "repos:a.ts", tone: "pending", text: "opPending", sticky: false },
      { id: "repos:b.ts", tone: "ok", text: "opConfirmed", sticky: false },
      { id: "repos:c.ts", tone: "error", text: "opFailed", sticky: false },
    ];
    harness.locked = true;
    harness.items = [fakeItem(mkNode({}), 0, false)];
    const html = render();
    expect(html).toContain("opPending");
    expect(html).toContain("opConfirmed");
    expect(html).toContain("opFailed");
    expect(html).toContain("moreActions");
    expect(html).toContain("disabled");
  });

  it("tints a modified file with the warning tone class", () => {
    harness.opNotes = [];
    harness.locked = false;
    harness.items = [fakeItem(mkNode({ gitStatus: "modified" }), 1, false)];
    expect(render()).toContain("text-warning");
  });

  it("renders an Undo action when an opNote carries a trash undo payload", () => {
    harness.opNotes = [
      { id: "repos:a.ts", tone: "ok", text: "deletedUndo", sticky: false, undo: { root: "repos", trashRel: ".jax-trash/T/a.ts" } },
    ];
    harness.locked = false;
    harness.items = [];
    const html = render();
    expect(html).toContain("deletedUndo");
    expect(html).toContain("undo");
  });

  it("keeps the inline create row visible for an open, matching directory", () => {
    harness.opNotes = [];
    harness.locked = false;
    harness.inline = { kind: "create", root: "repos", dirRel: "", fileKind: "file" };
    harness.items = [fakeItem(mkNode({ id: "repos:", rel: "", name: "repos", isDir: true }), 0, true)];
    expect(render()).toContain("min-w-0");
  });

  // Round 1 F6: no DOM library in this repo's harness (vitest's default "node" environment, no
  // jsdom/testing-library — confirmed via vitest.config.ts and package.json), so a real
  // right-click/click can't be simulated. Call the extracted pure functions directly instead.
  it("wires context-menu items to the exact beginCreate/beginRename/remove/downloadZip calls", () => {
    const ws = { beginCreate: vi.fn(), beginRename: vi.fn(), remove: vi.fn(), downloadZip: vi.fn() };
    const t = (k: string) => k;

    const common = commonItems(t, ws, "repos", "dir/a.ts", "dir", "a.ts");
    common.find((item) => item.id === "rename")!.onSelect();
    common.find((item) => item.id === "delete")!.onSelect();
    expect(ws.beginRename).toHaveBeenCalledExactlyOnceWith("repos", "dir/a.ts", "dir", "a.ts");
    expect(ws.remove).toHaveBeenCalledExactlyOnceWith("repos", "dir/a.ts", "dir", "a.ts");

    const onUpload = vi.fn();
    const dirItems = dirMenuItems(t, ws, "repos", "dir", "", "dir", false, onUpload);
    dirItems.find((item) => item.id === "newFile")!.onSelect();
    dirItems.find((item) => item.id === "newFolder")!.onSelect();
    dirItems.find((item) => item.id === "zip")!.onSelect();
    expect(ws.beginCreate).toHaveBeenNthCalledWith(1, "repos", "dir", "file");
    expect(ws.beginCreate).toHaveBeenNthCalledWith(2, "repos", "dir", "folder");
    expect(ws.downloadZip).toHaveBeenCalledExactlyOnceWith("repos", "dir", "dir");
    expect(onUpload).not.toHaveBeenCalled();
  });

  // FI-4/5/6 (2026-09-19 blueprint parity audit): mockup order/labels/caption, real pt-BR strings
  // (not the `t = k => k` identity used above) so the rendered text is actually checked.
  it("matches the mockup's context-menu order, the Excluir label, and its dimmed hint", () => {
    const t = (k: string) => (ptBR.files as unknown as Record<string, string>)[k];
    const ws = { beginCreate: vi.fn(), beginRename: vi.fn(), remove: vi.fn(), downloadZip: vi.fn() };
    const items = dirMenuItems(t, ws, "repos", "dir", "", "dir", false, vi.fn());

    expect(items.map((i) => i.label)).toEqual([
      "Novo arquivo", "Nova pasta", "Enviar arquivo", "Renomear", "Baixar pasta (.zip)", "Copiar caminho", "Excluir",
    ]);

    const html = renderToStaticMarkup(createElement(ContextMenu, { pos: { x: 0, y: 0 }, items, onClose: () => {} }));
    expect(html).toContain("Excluir");
    expect(html).not.toContain("Deletar");
    expect(html).toContain("vai para a lixeira");
  });

  // Round 3 F5: upload was only a root-row toolbar button — every directory needs the menu action too.
  it("gives every directory an upload action wired to its own rel, root or not", () => {
    const ws = { beginCreate: vi.fn(), beginRename: vi.fn(), remove: vi.fn(), downloadZip: vi.fn() };
    const t = (k: string) => k;
    const onUpload = vi.fn();

    const nonRootItems = dirMenuItems(t, ws, "repos", "dir/sub", "dir", "sub", false, onUpload);
    nonRootItems.find((item) => item.id === "upload")!.onSelect();
    expect(onUpload).toHaveBeenCalledExactlyOnceWith("repos", "dir/sub");

    onUpload.mockClear();
    const rootItems = dirMenuItems(t, ws, "vault", "", "", "vault", true, onUpload);
    rootItems.find((item) => item.id === "upload")!.onSelect();
    expect(onUpload).toHaveBeenCalledExactlyOnceWith("vault", "");
  });

  it("wires the inline commit/cancel row and the Undo action to the exact controller calls", () => {
    const ws = { commitInline: vi.fn(), cancelInline: vi.fn() };
    const props = inlineRowProps(ws);
    props.onCommit("newname.ts");
    props.onCancel();
    expect(ws.commitInline).toHaveBeenCalledExactlyOnceWith("newname.ts");
    expect(ws.cancelInline).toHaveBeenCalledOnce();

    const undoTrash = vi.fn();
    const note = {
      id: "repos:a.ts", tone: "ok" as const, text: "deletedUndo", sticky: false,
      undo: { root: "repos" as const, trashRel: ".jax-trash/T/a.ts" },
    };
    undoHandler(undoTrash, note)();
    expect(undoTrash).toHaveBeenCalledExactlyOnceWith("repos:a.ts", "repos", ".jax-trash/T/a.ts");
  });

  it("no longer imports the combined useFilesWorkspace hook (spec §3 — must use the narrow tree hook)", () => {
    const src = readFileSync(fileURLToPath(new URL("./FileTree.tsx", import.meta.url)), "utf8");
    expect(src).not.toMatch(/\buseFilesWorkspace\b/);
    expect(src).toContain("useFilesTreeWorkspace");
  });
});

describe("FileTree wiring (the config actually passed to useTree — round 1 F1/F5)", () => {
  it("treats the synthetic root as a folder and requests the scope's ONE root node as its only child, for a repo scope and a vault scope", async () => {
    harness.items = [];
    harness.scope = { kind: "repo", name: "jax-os" };
    render();
    const cfg = capturedConfig!;
    const isItemFolder = cfg.isItemFolder as (item: { getItemData: () => unknown }) => boolean;
    const dataLoader = cfg.dataLoader as {
      getItem: (id: string) => { isDir: boolean };
      getChildrenWithData: (id: string) => Promise<{ id: string; data: { root: string; rel: string; isDir: boolean; name: string } }[]>;
    };
    expect(isItemFolder({ getItemData: () => dataLoader.getItem("@root") })).toBe(true);
    let children = await dataLoader.getChildrenWithData("@root");
    expect(children.map((c) => c.id)).toEqual(["repos:jax-os"]);
    expect(children[0].data).toMatchObject({ root: "repos", rel: "jax-os", isDir: true, name: "jax-os" });

    harness.scope = { kind: "vault" };
    render();
    children = await (capturedConfig!.dataLoader as typeof dataLoader).getChildrenWithData("@root");
    expect(children.map((c) => c.id)).toEqual(["vault:"]);
    expect(children[0].data).toMatchObject({ rel: "", name: "roots.vault" });
  });

  it("bridges headless-tree's setExpandedItems into the exact added/removed onToggle calls", () => {
    const onToggle = vi.fn();
    harness.items = [];
    render(null, { expanded: new Set(["repos:"]), onToggle });
    const setExpandedItems = capturedConfig!.setExpandedItems as (v: string[]) => void;
    setExpandedItems(["@root", "repos:", "repos:src"]);
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(onToggle).toHaveBeenCalledWith("repos:src");
  });

  it("routes onPrimaryAction to onOpenFile for a file row only, never for a directory", () => {
    const onOpenFile = vi.fn();
    harness.items = [];
    render(null, { onOpenFile });
    const onPrimaryAction = capturedConfig!.onPrimaryAction as (item: { getItemData: () => ReturnType<typeof mkNode> }) => void;
    onPrimaryAction({ getItemData: () => mkNode({ isDir: false, rel: "a.ts" }) });
    expect(onOpenFile).toHaveBeenCalledWith("repos", "a.ts");
    onPrimaryAction({ getItemData: () => mkNode({ isDir: true, rel: "dir" }) });
    expect(onOpenFile).toHaveBeenCalledTimes(1);
  });
});

describe("iconForEntry", () => {
  it("returns Folder/FolderOpen for a directory depending on the open flag", () => {
    expect(iconForEntry("src", true, false)).toBe(Folder);
    expect(iconForEntry("src", true, true)).toBe(FolderOpen);
  });

  it("maps a known extension to its icon group, case-insensitively", () => {
    expect(iconForEntry("a.TS", false, false)).toBe(FileCode);
    expect(iconForEntry("a.json", false, false)).toBe(FileJson);
    expect(iconForEntry("a.png", false, false)).toBe(FileImage);
    expect(iconForEntry("a.mp3", false, false)).toBe(FileAudio);
    expect(iconForEntry("a.mp4", false, false)).toBe(FileVideo);
    expect(iconForEntry("a.zip", false, false)).toBe(FileArchive);
  });

  it("falls back to the generic File icon for an unknown extension or none at all", () => {
    expect(iconForEntry("weird.xyz", false, false)).toBe(File);
    expect(iconForEntry("Makefile", false, false)).toBe(File);
  });
});

describe("buildTreeChildren", () => {
  it("maps dirs and files into id/data pairs keyed by root:childRel, carrying gitStatus through", () => {
    const listing = {
      dirs: [{ name: "sub", isDir: true as const, mtime: "2026-01-01T00:00:00.000Z", gitStatus: "modified" as const }],
      files: [{ name: "a.ts", isDir: false as const, mtime: "2026-01-01T00:00:00.000Z", size: 10, gitStatus: null }],
    };
    const children = buildTreeChildren(listing, "repos", "dir");
    expect(children).toEqual([
      { id: "repos:dir/sub", data: { id: "repos:dir/sub", root: "repos", rel: "dir/sub", name: "sub", isDir: true, gitStatus: "modified", parentRel: "dir" } },
      { id: "repos:dir/a.ts", data: { id: "repos:dir/a.ts", root: "repos", rel: "dir/a.ts", name: "a.ts", isDir: false, gitStatus: null, parentRel: "dir" } },
    ]);
  });

  it("joins at the root (empty rel) without a leading slash", () => {
    const listing = { dirs: [], files: [{ name: "x.md", isDir: false as const, mtime: "2026-01-01T00:00:00.000Z", size: 1, gitStatus: null }] };
    expect(buildTreeChildren(listing, "vault", "").map((c) => c.id)).toEqual(["vault:x.md"]);
  });
});

describe("fetchDirChildren (F2 error contract)", () => {
  it("returns ok:true with an empty list on a successful, quietly empty listing", async () => {
    const fetchImpl = vi.fn().mockResolvedValue({ json: async () => ({ ok: true, data: { dirs: [], files: [] } }) });
    await expect(fetchDirChildren("repos", "dir", fetchImpl as unknown as typeof fetch)).resolves.toEqual({ ok: true, children: [] });
  });

  it("returns ok:false on an {ok:false} envelope, never an empty list", async () => {
    const fetchImpl = vi.fn().mockResolvedValue({ json: async () => ({ ok: false, error: "boom" }) });
    await expect(fetchDirChildren("repos", "dir", fetchImpl as unknown as typeof fetch)).resolves.toEqual({ ok: false });
  });

  it("returns ok:false when the fetch itself rejects", async () => {
    const fetchImpl = vi.fn().mockRejectedValue(new Error("network down"));
    await expect(fetchDirChildren("repos", "dir", fetchImpl as unknown as typeof fetch)).resolves.toEqual({ ok: false });
  });

  it("returns ok:false when the response body isn't valid JSON", async () => {
    const fetchImpl = vi.fn().mockResolvedValue({ json: async () => { throw new Error("bad json"); } });
    await expect(fetchDirChildren("repos", "dir", fetchImpl as unknown as typeof fetch)).resolves.toEqual({ ok: false });
  });
});

describe("gitStatusTint", () => {
  it("maps each status to its DS tone class, null to no class", () => {
    expect(gitStatusTint("modified")).toBe("text-warning");
    expect(gitStatusTint("added")).toBe("text-success");
    expect(gitStatusTint("deleted")).toBe("text-danger");
    expect(gitStatusTint("untracked")).toBe("text-muted");
    expect(gitStatusTint(null)).toBe("");
  });
});

describe("pickActiveMenu", () => {
  it("returns exactly the selected row's own items and position, never another row's", () => {
    const itemsA = [{ id: "rename", label: "Rename A", onSelect: () => {} }];
    const itemsB = [{ id: "rename", label: "Rename B", onSelect: () => {} }];
    const rows = [
      { id: "repos:a.ts", items: itemsA },
      { id: "repos:b.ts", items: itemsB },
    ];
    expect(pickActiveMenu(rows, { id: "repos:b.ts", x: 10, y: 20 })).toEqual({
      pos: { x: 10, y: 20 },
      items: itemsB,
    });
  });

  it("returns null when no row is selected, or the selected row no longer exists", () => {
    expect(pickActiveMenu([], null)).toBeNull();
    expect(pickActiveMenu([{ id: "repos:a.ts", items: [] }], { id: "repos:gone", x: 0, y: 0 })).toBeNull();
  });
});

describe("reloadAffectedDirectory", () => {
  it("invalidates exactly the swept directory's children (F2)", () => {
    const invalidateChildrenIds = vi.fn();
    const getItemInstance = vi.fn().mockReturnValue({ invalidateChildrenIds });
    reloadAffectedDirectory({ getItemInstance }, { root: "repos" as const, rel: "src" });
    expect(getItemInstance).toHaveBeenCalledExactlyOnceWith("repos:src");
    expect(invalidateChildrenIds).toHaveBeenCalledOnce();
  });

  it("does nothing when nothing was swept", () => {
    const getItemInstance = vi.fn();
    reloadAffectedDirectory({ getItemInstance }, null);
    expect(getItemInstance).not.toHaveBeenCalled();
  });
});

describe("scopeRootId", () => {
  it("keys a repo scope as 'repos:<name>' and a vault scope as 'vault:'", () => {
    expect(scopeRootId({ kind: "repo", name: "jax-os" })).toBe("repos:jax-os");
    expect(scopeRootId({ kind: "vault" })).toBe("vault:");
  });
});

describe("invalidateRootOnScopeChange (round-1 F1/F2 — a scope switch must invalidate the SYNTHETIC ROOT, never the new scope's own root id)", () => {
  it("invalidates exactly ROOT_ITEM_ID ('@root'), the same duck-typed spy reloadAffectedDirectory already uses", () => {
    const invalidateChildrenIds = vi.fn();
    const getItemInstance = vi.fn().mockReturnValue({ invalidateChildrenIds });
    invalidateRootOnScopeChange({ getItemInstance });
    expect(getItemInstance).toHaveBeenCalledExactlyOnceWith("@root");
    expect(invalidateChildrenIds).toHaveBeenCalledOnce();
  });
});

describe("refreshExpandedDirectories", () => {
  it("invalidates the scope root and only expanded keys UNDER it (never a different scope's key, never itself twice); a vault scope's empty root rel means every expanded vault key qualifies", async () => {
    const calls: string[] = [];
    const getItemInstance = vi.fn((id: string) => ({ invalidateChildrenIds: () => { calls.push(id); } }));
    await refreshExpandedDirectories({ getItemInstance }, "repos:jax-os", new Set(["repos:jax-os/src", "repos:other-repo/lib", "repos:jax-os"]));
    expect(calls).toEqual(["repos:jax-os", "repos:jax-os/src"]);
    calls.length = 0;
    await refreshExpandedDirectories({ getItemInstance }, "vault:", new Set(["vault:daily", "repos:jax-os/src"]));
    expect(calls).toEqual(["vault:", "vault:daily"]);
  });
});

describe("diffExpanded", () => {
  it("reports keys present in next but not prev as added, and vice versa as removed", () => {
    const prev = new Set(["repos:", "repos:src"]);
    expect(diffExpanded(prev, ["repos:", "repos:src", "repos:src/lib"])).toEqual({ added: ["repos:src/lib"], removed: [] });
    expect(diffExpanded(prev, ["repos:"])).toEqual({ added: [], removed: ["repos:src"] });
  });

  it("always ignores the synthetic super-root id", () => {
    expect(diffExpanded(new Set(), ["@root", "repos:"])).toEqual({ added: ["repos:"], removed: [] });
  });

  it("reports no change when the sets are equal", () => {
    expect(diffExpanded(new Set(["repos:"]), ["@root", "repos:"])).toEqual({ added: [], removed: [] });
  });
});

describe("revealAncestorsSequential", () => {
  it("toggles and awaits loadChildrenIds in strict root-to-leaf order, level N awaited before level N+1 fires (interleaved, not just both called)", async () => {
    const order: string[] = [];
    let resolveFirst!: () => void;
    const loadChildrenIds = vi.fn((id: string) => {
      if (id === "repos:") return new Promise<string[]>((res) => { resolveFirst = () => { order.push("load:repos: resolved"); res([]); }; });
      order.push(`load:${id}`);
      return Promise.resolve([]);
    });
    const onToggle = vi.fn((key: string) => order.push(`toggle:${key}`));
    const p = revealAncestorsSequential({ loadChildrenIds }, new Set(), onToggle, "repos", "src/a/b.ts", false);
    await Promise.resolve();
    expect(order).toEqual(["toggle:repos:"]); // "load:repos:" pending -> "toggle:repos:src" must NOT have fired yet
    resolveFirst();
    await p;
    expect(order).toEqual([
      "toggle:repos:", "load:repos: resolved",
      "toggle:repos:src", "load:repos:src",
      "toggle:repos:src/a", "load:repos:src/a",
    ]);
  });

  it("skips an already-expanded PREFIX, revealing only the not-yet-expanded suffix — no redundant toggle or load", async () => {
    const calls: string[] = [];
    const loadChildrenIds = vi.fn((id: string) => { calls.push(`load:${id}`); return Promise.resolve([]); });
    const onToggle = vi.fn((key: string) => calls.push(`toggle:${key}`));
    await revealAncestorsSequential({ loadChildrenIds }, new Set(["repos:"]), onToggle, "repos", "src/a/b.ts", false);
    expect(calls).toEqual(["toggle:repos:src", "load:repos:src", "toggle:repos:src/a", "load:repos:src/a"]);

    calls.length = 0;
    await revealAncestorsSequential({ loadChildrenIds }, new Set(["repos:", "repos:src", "repos:src/a"]), onToggle, "repos", "src/a/b.ts", false);
    expect(calls).toEqual([]); // fully expanded already -> no toggle, no load
  });
});

describe("FileTree — 'all' scope renders a flat repo+vault picker, not the tree (spec §4.4)", () => {
  it("renders every repo name plus one vault row, and does not mount the tree container", () => {
    harness.scope = { kind: "all" };
    harness.repoNamesQuery = { data: { ok: true, data: { dirs: [{ name: "jax-os" }, { name: "other-repo" }], files: [] } }, isError: false };
    const html = render();
    expect(html).toContain("jax-os");
    expect(html).toContain("other-repo");
    expect(html).toContain("scopeVault");
    expect(html).not.toContain('role="tree"');
  });
  it("wires rows to ws.setScope and fetches via the RepoSwitcher/ScopeChip query key", () => {
    const src = readFileSync(fileURLToPath(new URL("./FileTree.tsx", import.meta.url)), "utf8");
    expect(src).toContain('queryKey: ["files", "repoNames"]');
    expect(src).toContain('ws.setScope({ kind: "repo", name })');
    expect(src).toContain('ws.setScope({ kind: "vault" })');
  });
  it("hides the vault row when vaultPath is null even though the flag is on (F2)", () => {
    harness.scope = { kind: "all" };
    harness.vaultPath = null;
    try {
      harness.repoNamesQuery = { data: { ok: true, data: { dirs: [{ name: "jax-os" }], files: [] } }, isError: false };
      const html = render();
      expect(html).toContain("jax-os");
      expect(html).not.toContain("scopeVault");
    } finally {
      harness.vaultPath = "/tmp/vault";
    }
  });
});
