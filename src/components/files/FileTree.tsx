"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useTree } from "@headless-tree/react";
import { asyncDataLoaderFeature, hotkeysCoreFeature, selectionFeature } from "@headless-tree/core";
import {
  ChevronDown, ChevronRight, File, FileArchive, FileAudio, FileCode, FileImage, FileJson, FileText,
  FileVideo, Folder, FolderOpen, ListCollapse, MoreHorizontal, RefreshCw, ScanSearch, Upload,
} from "lucide-react";
import { useTranslations } from "next-intl";
import type { Envelope } from "@/lib/api";
import type { DirListing, RepoRef, Root } from "@/server/collectors/files";
import type { FileGitStatus } from "@/server/collectors/gitStatus";
import type { InlineEdit, OpNote } from "@/lib/filesWorkspace";
import { affectedPath, relativeToScope, scopedRel, serializeScope } from "../../lib/filesWorkspace";
import { breadcrumbSegments } from "../../lib/breadcrumb";
import { fetchRepoNames, repoNamesFromListing } from "../../lib/fileSearch";
import { DEFAULT_SCOPE } from "../../lib/filesState";
import { useGeneralSettings, vaultUsable } from "../../lib/settingsQuery";
import { ancestorKeys } from "./FileEditor";
import { ContextMenu, useContextMenu, type MenuItem } from "./ContextMenu";
import { useFilesTreeWorkspace } from "./FilesWorkspaceProvider";
import { SourceWarning } from "../mission/SourceWarning";
import { RepoSwitcher } from "./RepoSwitcher";
import { Breadcrumb } from "./Breadcrumb";

const ROOT_ITEM_ID = "@root";

function stem(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 ? name.slice(0, dot) : name;
}

// One inline text row shared by "new file/folder" and "rename" (AC6, unchanged from Build B2).
function InlineNameInput({ initial, selectStem, onCommit, onCancel }: {
  initial: string; selectStem: boolean; onCommit: (name: string) => void; onCancel: () => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.focus();
    if (selectStem) el.setSelectionRange(0, stem(initial).length);
    else el.select();
  }, [initial, selectStem]);
  return (
    <input
      ref={ref}
      defaultValue={initial}
      onKeyDown={(e) => {
        if (e.key === "Enter") onCommit((e.target as HTMLInputElement).value);
        else if (e.key === "Escape") onCancel();
      }}
      onBlur={(e) => onCommit(e.target.value)}
      className="min-w-0 flex-1 rounded border border-line bg-surface px-1 py-0.5 text-[13px] text-body-ink"
    />
  );
}

function RowMenuTrigger({ label, onOpen }: { label: string; onOpen: (x: number, y: number) => void }) {
  return (
    <button
      type="button"
      title={label}
      aria-label={label}
      onClick={(e) => { e.stopPropagation(); const r = e.currentTarget.getBoundingClientRect(); onOpen(r.left, r.bottom); }}
      className="flex-none rounded p-1 text-muted transition hover:bg-surface-3 hover:text-body-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand"
    >
      <MoreHorizontal className="h-3.5 w-3.5" />
    </button>
  );
}

// Unchanged from Build B2 (round 1 F6) — rename/delete/copy-path, identical for a dir or file row.
export function commonItems(
  t: (k: string) => string,
  ws: Pick<ReturnType<typeof useFilesTreeWorkspace>, "beginRename" | "remove">,
  root: Root, rel: string, parentRel: string, name: string,
): MenuItem[] {
  return [
    { id: "rename", label: t("rename"), onSelect: () => ws.beginRename(root, rel, parentRel, name) },
    { id: "copyPath", label: t("copyPath"), onSelect: () => { void navigator.clipboard.writeText(`${root}/${rel}`); } },
    { id: "delete", label: t("delete"), danger: true, hint: t("deleteHint"), onSelect: () => ws.remove(root, rel, parentRel, name) },
  ];
}

// round 3 F5: upload is otherwise only a root-row toolbar button (below) — every directory needs a
// way in, so the menu carries it too, reusing the same upload input/handler via `onUpload`.
export function dirMenuItems(
  t: (k: string) => string,
  ws: Pick<ReturnType<typeof useFilesTreeWorkspace>, "beginCreate" | "beginRename" | "remove" | "downloadZip">,
  root: Root, rel: string, parentRel: string, name: string, isRoot: boolean,
  onUpload: (root: Root, rel: string) => void,
): MenuItem[] {
  const create = [
    { id: "newFile", label: t("newFile"), onSelect: () => ws.beginCreate(root, rel, "file") },
    { id: "newFolder", label: t("newFolder"), onSelect: () => ws.beginCreate(root, rel, "folder") },
    { id: "upload", label: t("upload"), onSelect: () => onUpload(root, rel) },
  ];
  const zip = { id: "zip", label: t("downloadZip"), onSelect: () => void ws.downloadZip(root, rel, name) };
  // Mockup order (FI-4): newFile, newFolder, rename, zip, copyPath, delete — upload has no
  // mockup slot, kept grouped with the other creation actions.
  if (isRoot) return [...create, zip, { id: "copyPath", label: t("copyPath"), onSelect: () => { void navigator.clipboard.writeText(`${root}/${rel}`); } }];
  const [rename, copyPath, del] = commonItems(t, ws, root, rel, parentRel, name);
  return [...create, rename, zip, copyPath, del];
}

// Unchanged from Build B2 (round 1 F6).
export function inlineRowProps(ws: Pick<ReturnType<typeof useFilesTreeWorkspace>, "commitInline" | "cancelInline">) {
  return { onCommit: ws.commitInline, onCancel: ws.cancelInline };
}

// Unchanged from Build B2 (round 1 F6).
export function undoHandler(undoTrash: ReturnType<typeof useFilesTreeWorkspace>["undoTrash"], note: OpNote) {
  return () => undoTrash(note.id, note.undo!.root, note.undo!.trashRel);
}

// Per-extension icon (spec §5 item 9 "per-extension icons"). Falls through to a generic File icon —
// ponytail: one flat lookup, not a MIME-sniffing service; matches viewerFor's own extension groups
// (Task 8) without importing from it (that file is not safe to import from a tree row for the same
// client/server boundary reason mediaMime isn't imported directly, see Task 8's own note).
const EXT_ICON: [ext: string[], icon: typeof FileCode][] = [
  [["ts", "tsx", "js", "jsx", "py", "sh", "css", "scss"], FileCode],
  [["json", "yaml", "yml", "toml"], FileJson],
  [["md", "markdown", "html", "htm", "txt"], FileText],
  [["png", "jpg", "jpeg", "gif", "webp", "svg"], FileImage],
  [["mp3", "wav", "ogg", "m4a", "aac"], FileAudio],
  [["mp4", "webm", "mov"], FileVideo],
  [["zip", "tar", "gz"], FileArchive],
];
export function iconForEntry(name: string, isDir: boolean, open: boolean): typeof FileCode {
  if (isDir) return open ? FolderOpen : Folder;
  const ext = name.split(".").pop()?.toLowerCase() ?? "";
  for (const [exts, icon] of EXT_ICON) if (exts.includes(ext)) return icon;
  return File;
}

// Git-status tint (spec §5 item 9 "git-status tinting"). One DS tone per status — matches the
// PRIORITY order gitStatus.ts's parseGitStatusPorcelain already uses server-side.
export const GIT_TINT: Record<FileGitStatus, string> = {
  modified: "text-warning", added: "text-success", deleted: "text-danger", untracked: "text-muted",
};
export function gitStatusTint(status: FileGitStatus | null): string {
  return status ? GIT_TINT[status] : "";
}

export type TreeNode = { id: string; root: Root; rel: string; name: string; isDir: boolean; gitStatus: FileGitStatus | null; parentRel: string };

// Maps one /api/files/tree response into headless-tree's {id, data}[] shape (spec §5 item 9's data
// source: the EXISTING lazy per-directory call, unchanged — only its consumer changes, Part 1's own
// note). Pure and fixture-tested directly (no fetch, no DOM).
export function buildTreeChildren(listing: DirListing, root: Root, rel: string): { id: string; data: TreeNode }[] {
  const mk = (name: string, isDir: boolean, gitStatus: FileGitStatus | null) => {
    const childRel = rel ? `${rel}/${name}` : name;
    return { id: `${root}:${childRel}`, data: { id: `${root}:${childRel}`, root, rel: childRel, name, isDir, gitStatus, parentRel: rel } };
  };
  return [
    ...listing.dirs.map((d) => mk(d.name, true, d.gitStatus)),
    ...listing.files.map((f) => mk(f.name, false, f.gitStatus)),
  ];
}

// Sequential reveal (spec §5 #9, fixing FI-2): awaits each level's loadChildrenIds before toggling
// the next, deeper ancestor — the tree's own cache (and cache.current below) is only correct for a
// key once its PARENT's children resolve; toggling every ancestor in one tick (the old
// revealAncestors) left every key but the shallowest a placeholder, so only the root expanded.
export async function revealAncestorsSequential(
  tree: { loadChildrenIds: (itemId: string) => Promise<string[]> },
  expanded: Set<string>,
  onToggle: (key: string) => void,
  root: Root,
  rel: string,
  targetIsDir: boolean,
): Promise<void> {
  for (const key of ancestorKeys(root, rel, targetIsDir)) {
    if (expanded.has(key)) continue;
    onToggle(key);
    await tree.loadChildrenIds(key);
  }
}

export type DirFetchOutcome = { ok: true; children: { id: string; data: TreeNode }[] } | { ok: false };

// Error contract (F2): an `{ok:false}` envelope AND a thrown fetch/JSON failure both mean the
// directory is unavailable, distinct from a successful `{ok:true, data:[]}` empty listing. Pure
// aside from the injected fetch, so both failure paths and the quiet-empty path are spy-tested
// without mounting a tree.
export async function fetchDirChildren(root: Root, rel: string, fetchImpl: typeof fetch = fetch): Promise<DirFetchOutcome> {
  try {
    const envelope: Envelope<DirListing> = await (await fetchImpl(`/api/files/tree?root=${root}&rel=${encodeURIComponent(rel)}`)).json();
    if (!envelope.ok) return { ok: false };
    return { ok: true, children: buildTreeChildren(envelope.data, root, rel) };
  } catch {
    return { ok: false };
  }
}

// Diffs headless-tree's own next expandedItems array against our externally-owned `prev` Set (the
// SAME Set ws.expanded already is), ignoring the synthetic super-root id — never part of the
// external contract. Pure so the sync logic below is testable without mounting the tree at all.
export function diffExpanded(prev: Set<string>, next: readonly string[]): { added: string[]; removed: string[] } {
  const nextSet = new Set(next.filter((k) => k !== ROOT_ITEM_ID));
  const added: string[] = [];
  const removed: string[] = [];
  for (const k of nextSet) if (!prev.has(k)) added.push(k);
  for (const k of prev) if (!nextSet.has(k)) removed.push(k);
  return { added, removed };
}

// Given every row's own computed menu items and which row (if any) currently has the menu open,
// picks the ONE row to render (round 3 F1). Pure so a no-jsdom harness can prove exactly one row's
// items come back, never a mix of several rows' items for one shared position.
export function pickActiveMenu(
  rows: { id: string; items: MenuItem[] }[],
  selection: { id: string; x: number; y: number } | null,
): { pos: { x: number; y: number }; items: MenuItem[] } | null {
  if (!selection) return null;
  const row = rows.find((r) => r.id === selection.id);
  return row ? { pos: { x: selection.x, y: selection.y }, items: row.items } : null;
}

function splitKey(key: string): [Root, string] {
  const i = key.indexOf(":");
  return [key.slice(0, i) as Root, key.slice(i + 1)];
}

export function scopeRootId(scope: RepoRef): string {
  return scope.kind === "vault" ? "vault:" : `repos:${scope.name}`;
}

// Scope switch (spec §5, see Global Constraints, round-1 F1): the async data loader caches
// ROOT_ITEM_ID's OWN children mapping — i.e. which single child id "@root" currently resolves to —
// independently of the `dataLoader` closure's own identity. Invalidating the NEW scope root's own
// children does nothing, because ROOT_ITEM_ID itself still has the OLD scope's rootId cached as its
// one child: ROOT_ITEM_ID must be invalidated, never scopeRootId(ws.scope). Extracted as a pure
// function, the same way reloadAffectedDirectory/refreshExpandedDirectories already are, so the
// invalidation TARGET is asserted directly instead of trusted.
export function invalidateRootOnScopeChange(tree: { getItemInstance: (id: string) => { invalidateChildrenIds: () => void | Promise<void> } }): void {
  void tree.getItemInstance(ROOT_ITEM_ID).invalidateChildrenIds();
}

// Refresh button (spec §5): invalidates the scope root AND every ws.expanded key actually under it
// (ws.expanded is global/unscoped — a different scope's stale key must never be touched). Reuses
// affectedPath, the same "under that path" predicate rename/delete reconciliation already uses.
export function refreshExpandedDirectories(
  tree: { getItemInstance: (id: string) => { invalidateChildrenIds: () => void | Promise<void> } },
  scopeRootId: string,
  expanded: Set<string>,
): Promise<void> {
  const [scopeRoot, scopeRel] = splitKey(scopeRootId);
  const under = [...expanded].filter((key) => {
    if (key === scopeRootId) return false;
    const [root, rel] = splitKey(key);
    return affectedPath({ root, rel }, scopeRoot, scopeRel);
  });
  return Promise.all([scopeRootId, ...under].map((id) => tree.getItemInstance(id).invalidateChildrenIds())).then(() => {});
}

// F2: asks headless-tree to reload exactly the swept directory's children (invalidateChildrenIds
// clears its cache for that id and refetches via the SAME dataLoader above). Pure over a duck-typed
// tree so the wiring — the right id, called exactly once — is spy-testable without mounting a tree.
export function reloadAffectedDirectory(
  tree: { getItemInstance: (id: string) => { invalidateChildrenIds: () => void } },
  sweep: { root: Root; rel: string } | null,
): void {
  if (!sweep) return;
  tree.getItemInstance(`${sweep.root}:${sweep.rel}`).invalidateChildrenIds();
}

type FileTreeProps = {
  expanded: Set<string>;
  onToggle: (key: string) => void;
  onOpenFile: (root: Root, rel: string) => void;
  onOpenFilePinned: (root: Root, rel: string) => void;
  activeFile: { root: Root; rel: string } | null;
};

export function FileTree({ expanded, onToggle, onOpenFile, onOpenFilePinned, activeFile }: FileTreeProps) {
  const t = useTranslations("files");
  const { data: settings } = useGeneralSettings();
  const vaultEnabled = vaultUsable(settings);
  const ws = useFilesTreeWorkspace();
  // Global Constraints move 3: every ws.scope read below assumes a concrete repo/vault; in the
  // "all" scope the tree body renders the flat picker instead, so this fallback is never observed.
  const treeScope: RepoRef = ws.scope.kind === "all" ? DEFAULT_SCOPE : ws.scope;
  const repoNamesQuery = useQuery({ queryKey: ["files", "repoNames"], queryFn: () => fetchRepoNames(), staleTime: 60_000 });
  const repoNames = repoNamesQuery.data?.ok ? repoNamesFromListing(repoNamesQuery.data.data) : [];
  const repoNamesFailed = repoNamesQuery.isError || (repoNamesQuery.data != null && !repoNamesQuery.data.ok);
  const menu = useContextMenu();
  const uploadRef = useRef<HTMLInputElement>(null);
  const uploadTargetRef = useRef<{ root: Root; rel: string } | null>(null);
  const cache = useRef(new Map<string, TreeNode>());
  const itemRefs = useRef(new Map<string, HTMLElement>());
  // Directories whose last fetch was unavailable (F2) — rendered as a warning row instead of
  // the quiet empty list `getChildrenWithData` still has to return to headless-tree.
  const [failedDirs, setFailedDirs] = useState<Set<string>>(new Set());
  const [refreshing, setRefreshing] = useState(false);

  const currentScopeRootId = scopeRootId(treeScope);
  const dataLoader = useMemo(() => {
    const rootId = scopeRootId(treeScope);
    const rootRoot: Root = treeScope.kind === "vault" ? "vault" : "repos";
    const rootRel = treeScope.kind === "vault" ? "" : treeScope.name;
    const rootLabel = treeScope.kind === "vault" ? t("roots.vault") : treeScope.name;
    const scopeRoot: TreeNode = { id: rootId, root: rootRoot, rel: rootRel, name: rootLabel, isDir: true, gitStatus: null, parentRel: "" };
    return {
      // Root must report isDir:true (pre-existing F1). ONE root keyed by ws.scope (spec §5) —
      // children come from the SAME /api/files/tree?root=repos&rel=<name> call, unchanged endpoint.
      getItem: (itemId: string): TreeNode =>
        itemId === ROOT_ITEM_ID ? { id: ROOT_ITEM_ID, root: "repos", rel: "", name: "", isDir: true, gitStatus: null, parentRel: "" }
          : itemId === rootId ? scopeRoot
            : cache.current.get(itemId) ?? { id: itemId, root: "repos", rel: "", name: "", isDir: false, gitStatus: null, parentRel: "" },
      getChildrenWithData: async (itemId: string) => {
        if (itemId === ROOT_ITEM_ID) return [{ id: rootId, data: scopeRoot }];
        const [root, rel] = splitKey(itemId);
        const outcome = await fetchDirChildren(root, rel);
        setFailedDirs((prev) => {
          if (outcome.ok === !prev.has(itemId)) return prev;
          const next = new Set(prev);
          if (outcome.ok) next.delete(itemId); else next.add(itemId);
          return next;
        });
        if (!outcome.ok) return [];
        for (const c of outcome.children) cache.current.set(c.id, c.data);
        return outcome.children;
      },
    };
  }, [t, treeScope]);

  // A fresh array here re-renders forever: useTree folds `state` into the tree on every render and a
  // new reference counts as a change (React error 301 in production, 2026-09-19). Memoize on the Set.
  const expandedItems = useMemo(() => [ROOT_ITEM_ID, ...expanded], [expanded]);
  const selectedItems = useMemo(() => (ws.selected ? [`${ws.selected.root}:${ws.selected.rel}`] : []), [ws.selected]);
  const tree = useTree<TreeNode>({
    rootItemId: ROOT_ITEM_ID,
    getItemName: (item) => item.getItemData().name,
    isItemFolder: (item) => item.getItemData().isDir,
    dataLoader,
    state: { expandedItems, selectedItems },
    setExpandedItems: (updaterOrValue) => {
      const prevArr = [ROOT_ITEM_ID, ...expanded];
      const next = typeof updaterOrValue === "function" ? updaterOrValue(prevArr) : updaterOrValue;
      const { added, removed } = diffExpanded(expanded, next);
      for (const k of added) onToggle(k);
      for (const k of removed) onToggle(k);
    },
    setSelectedItems: (updaterOrValue) => {
      const next = typeof updaterOrValue === "function" ? updaterOrValue(selectedItems) : updaterOrValue;
      const id = next[next.length - 1]; // capped at one — last wins, mirrors setExpandedItems's own diffing style
      if (!id) return;
      const [root, rel] = splitKey(id);
      const data = cache.current.get(id) ?? (id === currentScopeRootId ? dataLoader.getItem(id) : undefined);
      ws.setSelected(root, rel, data?.isDir ?? false);
    },
    onPrimaryAction: (item) => {
      const d = item.getItemData();
      if (!d.isDir) onOpenFile(d.root, d.rel);
    },
    features: [hotkeysCoreFeature, asyncDataLoaderFeature, selectionFeature],
  });

  // Scope switch (spec §5, see Global Constraints, round-1 F1): forces a refetch of the new scope's
  // root by invalidating the SYNTHETIC root, not the new scope's own root (invalidateRootOnScopeChange
  // above). Skipped on the FIRST run (mount already fetched it) to avoid a visibly-flickering refetch.
  const scopeMountRef = useRef(true);
  useEffect(() => {
    if (scopeMountRef.current) { scopeMountRef.current = false; return; }
    invalidateRootOnScopeChange(tree);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `tree` is this mount's stable instance; only an actual scope change should re-trigger
  }, [serializeScope(ws.scope)]);

  useEffect(() => {
    reloadAffectedDirectory(tree, ws.treeSweep);
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `tree` is this mount's stable instance; only a new treeSweep should re-trigger a reload
  }, [ws.treeSweep]);

  useEffect(() => {
    if (!ws.revealRequest) return;
    const { root, rel, targetIsDir, version } = ws.revealRequest;
    // One-shot channel (review F4): clear this request once its own walk finishes, so a remounted
    // tree doesn't replay it. consumeReveal is version-guarded — a completion for an older request
    // can't clobber a newer one that raced ahead of it.
    void revealAncestorsSequential(tree, expanded, onToggle, root, rel, targetIsDir).then(() => {
      ws.consumeReveal(version);
      requestAnimationFrame(() => itemRefs.current.get(`${root}:${rel}`)?.scrollIntoView({ block: "nearest" }));
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `tree`/`expanded`/`onToggle` are this mount's stable/prop values; only a new revealRequest should re-trigger a reveal
  }, [ws.revealRequest]);

  const inline = ws.inline as InlineEdit | null;
  // Filled during the row map below, read once after it — one pass, no separate lookup pass.
  const menuRows: { id: string; items: MenuItem[] }[] = [];
  async function handleRefresh() {
    setRefreshing(true);
    await refreshExpandedDirectories(tree, currentScopeRootId, expanded);
    setRefreshing(false);
  }

  // Item 2 (spec): switcher + breadcrumb are pinned above the tree, not scrolling with it —
  // navigateScope mirrors page.tsx's own helper (same ws.setSelected/requestReveal pair) since the
  // breadcrumb now renders here, next to the tree it scrolls independently from.
  const scopeRoot: Root = ws.scope.kind === "vault" ? "vault" : "repos";
  function navigateScope(rel: string, isDir: boolean) {
    const fullRel = scopedRel(treeScope, rel);
    ws.setSelected(scopeRoot, fullRel, isDir);
    ws.requestReveal(scopeRoot, fullRel, isDir);
  }

  return (
    <div className="flex h-full flex-col">
      <div className="flex-none space-y-2 bg-surface pb-2">
        <div className="flex items-center gap-2">
          <RepoSwitcher scope={ws.scope} onChange={ws.setScope} />
          {ws.scope.kind !== "all" ? (
            <>
          <button
            type="button"
            title={t("revealActive")}
            aria-label={t("revealActive")}
            disabled={!activeFile}
            onClick={() => activeFile && ws.requestReveal(activeFile.root, activeFile.rel, false)}
            className="flex h-10 w-10 flex-none items-center justify-center rounded-lg border border-line-strong bg-surface-2 text-muted transition hover:bg-surface-3 hover:text-body-ink disabled:cursor-not-allowed disabled:opacity-40"
          >
            <ScanSearch className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            title={t("collapseAll")}
            aria-label={t("collapseAll")}
            onClick={() => { for (const k of expanded) onToggle(k); }}
            className="flex h-10 w-10 flex-none items-center justify-center rounded-lg border border-line-strong bg-surface-2 text-muted transition hover:bg-surface-3 hover:text-body-ink"
          >
            <ListCollapse className="h-3.5 w-3.5" />
          </button>
          <button type="button" title={t("refreshTree")} aria-label={t("refreshTree")} disabled={refreshing} onClick={() => void handleRefresh()}
            className="flex h-10 w-10 flex-none items-center justify-center rounded-lg border border-line-strong bg-surface-2 text-body-ink transition hover:bg-surface-3 disabled:cursor-not-allowed disabled:opacity-40">
            <RefreshCw className={`h-4 w-4 ${refreshing ? "animate-spin" : ""}`} />
          </button>
            </>
          ) : null}
        </div>
        <div className="min-w-0 px-1">
          <Breadcrumb rootLabel={t("filesRoot")} rootClickable={ws.scope.kind !== "all"} ariaLabel={t("breadcrumbLabel")} finalClickable={false}
            segments={ws.scope.kind === "all" ? [] : [
              { label: ws.scope.kind === "vault" ? t("scopeVault") : ws.scope.name, rel: "", isDir: true, collapsed: false },
              ...breadcrumbSegments(ws.selected ? relativeToScope(ws.scope, ws.selected.rel) : "", ws.selected?.isDir ?? false),
            ]}
            onRootClick={() => ws.setScope({ kind: "all" })} onSegmentClick={navigateScope} />
        </div>
      </div>
      <div className="jax-scroll min-h-0 flex-1 overflow-y-auto">
      {ws.scope.kind === "all" ? (
        repoNamesFailed ? (
          <div className="p-2"><SourceWarning label={t("unavailable")} /></div>
        ) : (
          <div className="flex flex-col gap-0.5 px-1 pb-1 pt-1">
            {repoNames.map((name) => (
              <button key={name} type="button" onClick={() => ws.setScope({ kind: "repo", name })}
                className="flex h-9 w-full items-center gap-1.5 rounded-md px-2 text-left text-[13px] text-body-ink hover:bg-surface-2">
                <Folder className="h-3.5 w-3.5 flex-none text-warning" /><span className="truncate">{name}</span>
              </button>
            ))}
            {vaultEnabled ? (
              <button type="button" onClick={() => ws.setScope({ kind: "vault" })}
                className="flex h-9 w-full items-center gap-1.5 rounded-md px-2 text-left text-[13px] text-body-ink hover:bg-surface-2">
                <Folder className="h-3.5 w-3.5 flex-none text-warning" /><span className="truncate">{t("scopeVault")}</span>
              </button>
            ) : null}
          </div>
        )
      ) : (
        <>
      {ws.opNotes.map((note) => (
        <p
          key={note.id}
          role="status"
          aria-live="polite"
          className={`px-2 py-1 text-[12px] ${
            note.tone === "pending" ? "text-muted"
              : note.tone === "ok" ? "text-body-ink"
                : note.tone === "warning" ? "text-warning"
                  : "text-danger"
          }`}
        >
          {note.text}
          {note.undo ? (
            <button type="button" onClick={undoHandler(ws.undoTrash, note)} className="ml-2 underline">
              {t("undo")}
            </button>
          ) : null}
        </p>
      ))}
      {refreshing ? <p role="status" aria-live="polite" className="px-2 py-1 text-[12px] text-muted">{t("refreshingTree")}</p> : null}
      <div {...tree.getContainerProps(t("treeLabel"))} className="flex flex-col gap-0.5 px-1 pb-1 pt-1">
        {tree.getItems().map((item) => {
          const d = item.getItemData();
          const level = item.getItemMeta().level;
          const isRoot = item.getId() === currentScopeRootId;
          const isSelected = !!ws.selected && ws.selected.root === d.root && ws.selected.rel === d.rel;
          const isOpen = !!activeFile && activeFile.root === d.root && activeFile.rel === d.rel;
          const open = item.isExpanded();
          const locked = ws.isLocked(d.root, d.rel);
          const renamingHere = inline?.kind === "rename" && inline.root === d.root && inline.entryRel === d.rel;
          const creatingHere = d.isDir && open && inline?.kind === "create" && inline.root === d.root && inline.dirRel === d.rel;
          const Icon = iconForEntry(d.name, d.isDir, open);
          const items: MenuItem[] = d.isDir
            ? dirMenuItems(t, ws, d.root, d.rel, d.parentRel, d.name, isRoot, (uRoot, uRel) => {
                uploadTargetRef.current = { root: uRoot, rel: uRel };
                uploadRef.current?.click();
              })
            : [
                { id: "download", label: t("download"), onSelect: () => { window.location.href = `/api/files/download?root=${d.root}&rel=${encodeURIComponent(d.rel)}`; } },
                ...commonItems(t, ws, d.root, d.rel, d.parentRel, d.name),
              ];
          menuRows.push({ id: item.getId(), items });
          const guides = Array.from({ length: level }, (_, i) => (
            <span key={i} className="w-4 flex-none self-stretch border-l border-line/60" />
          ));
          // Item 4 (spec): an open editor tab wins over a plain tree selection — surface-3 + sage
          // (accent) left bar, vs. selection's brand-soft + brand left bar. Both keep the SAME 3px
          // rail width as the mockup's `base` (transparent) state, so rows never shift on select.
          const rowTone = isOpen ? "border-accent bg-surface-3" : isSelected ? "border-brand bg-brand-soft" : "border-transparent";
          // Item 4 (spec): the mockup's mk() gives BOTH 'selected' and 'active' the same ink+bold
          // label — only the row background/rail and the separate "aberto" tag carry the color.
          const labelTone = isOpen || isSelected ? "font-semibold text-ink" : !d.isDir ? gitStatusTint(d.gitStatus) : "";
          return (
            <div key={item.getId()}>
              <div
                className={`group flex items-stretch rounded-md border-l-[3px] hover:bg-surface-2 ${rowTone}`}
                onContextMenu={(e) => { e.preventDefault(); menu.open(item.getId(), e.clientX, e.clientY); }}
              >
                {guides}
                {renamingHere && inline ? (
                  <div className="flex flex-1 items-center gap-1.5 py-1 pr-2">
                    <Icon className={`h-3.5 w-3.5 flex-none ${d.isDir ? "text-warning" : "text-muted"}`} />
                    <InlineNameInput initial={inline.kind === "rename" ? inline.curName : ""} selectStem={!d.isDir} {...inlineRowProps(ws)} />
                  </div>
                ) : (
                  <button
                    {...item.getProps()}
                    ref={(el) => {
                      item.registerElement(el);
                      if (el) itemRefs.current.set(item.getId(), el); else itemRefs.current.delete(item.getId());
                    }}
                    onDoubleClick={() => { if (!d.isDir) onOpenFilePinned(d.root, d.rel); }}
                    disabled={locked}
                    className={`flex flex-1 items-center gap-1.5 py-1 pr-2 text-left text-[13px] text-body-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-brand disabled:cursor-not-allowed disabled:opacity-50 ${
                      level === 0 ? "font-semibold" : ""
                    }`}
                  >
                    {d.isDir ? (open ? <ChevronDown className="h-3.5 w-3.5 flex-none text-muted" /> : <ChevronRight className="h-3.5 w-3.5 flex-none text-muted" />) : (
                      <span className="h-3.5 w-3.5 flex-none" />
                    )}
                    <Icon className={`h-3.5 w-3.5 flex-none ${d.isDir ? "text-warning" : gitStatusTint(d.gitStatus) || "text-muted"}`} />
                    <span className={`truncate ${labelTone}`}>{d.name}</span>
                    {isOpen ? <span className="ml-auto flex-none text-[10px] font-semibold text-brand">{t("openTag")}</span> : null}
                  </button>
                )}
                {isRoot ? (
                  <>
                    <button
                      type="button"
                      title={t("upload")}
                      aria-label={t("upload")}
                      onClick={() => { uploadTargetRef.current = { root: d.root, rel: d.rel }; uploadRef.current?.click(); }}
                      disabled={locked}
                      className="flex-none rounded p-1 text-muted opacity-0 transition hover:bg-surface-3 hover:text-body-ink group-hover:opacity-100 [@media(hover:none)]:opacity-100 focus-visible:opacity-100 disabled:cursor-not-allowed disabled:hover:bg-transparent disabled:hover:text-muted"
                    >
                      <Upload className="h-3.5 w-3.5" />
                    </button>
                  </>
                ) : null}
                <RowMenuTrigger label={t("moreActions")} onOpen={(x, y) => menu.open(item.getId(), x, y)} />
              </div>
              {creatingHere && inline ? (
                <div className="flex items-center gap-1.5 py-1 pr-2">
                  {Array.from({ length: level + 1 }, (_, i) => (
                    <span key={i} className="w-4 flex-none self-stretch border-l border-line/60" />
                  ))}
                  {inline.kind === "create" && inline.fileKind === "folder" ? (
                    <Folder className="h-3.5 w-3.5 flex-none text-warning" />
                  ) : (
                    <File className="h-3.5 w-3.5 flex-none text-muted" />
                  )}
                  <InlineNameInput initial="" selectStem={false} {...inlineRowProps(ws)} />
                </div>
              ) : null}
              {d.isDir && open && item.isLoading() ? (
                <div className="py-1 pr-2" style={{ paddingLeft: (level + 1) * 16 + 8 }}>
                  <div className="h-4 animate-pulse rounded bg-surface-2" />
                </div>
              ) : null}
              {d.isDir && open && !item.isLoading() && failedDirs.has(item.getId()) ? (
                <div className="py-1 pr-2" style={{ paddingLeft: (level + 1) * 16 + 8 }}>
                  <SourceWarning label={t("unavailable")} />
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
        </>
      )}
      </div>
      {(() => {
        const activeMenu = pickActiveMenu(menuRows, menu.selection);
        return activeMenu ? <ContextMenu pos={activeMenu.pos} items={activeMenu.items} onClose={menu.close} /> : null;
      })()}
      <input
        ref={uploadRef}
        type="file"
        multiple
        hidden
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          e.target.value = "";
          const target = uploadTargetRef.current;
          if (files.length > 0 && target) ws.uploadMany(target.root, target.rel, files);
        }}
      />
    </div>
  );
}
