"use client";

import { useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useTranslations } from "next-intl";
import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import type { Root, RepoRef, SearchScope } from "@/server/collectors/files";
import {
  normalizeState,
  parseRecentRepos,
  parseRecents,
  pushRecent,
  pushRecentRepo,
  RECENT_REPOS_KEY,
  RECENTS_KEY,
  STORAGE_KEY,
  type EditPatch,
  type EditState,
  type SelectedEntry,
  type Tab,
} from "@/lib/filesState";
import {
  consumedLineRevealRequest,
  consumedRevealRequest,
  createFilesController,
  dirtyKeys as dirtyKeySet,
  filesQueryAffected,
  isDirty,
  nextLineRevealRequest,
  nextRevealRequest,
  nextTreeSweep,
  repoOf,
  revealScopeSwitch,
  serializeScope,
  uiStorage,
  type InlineEdit,
  type LineRevealRequest,
  type OpNote,
  type RevealRequest,
  type TreeSweep,
} from "@/lib/filesWorkspace";

type FilesTreeWorkspaceApi = {
  expanded: Set<string>;
  tabs: Tab[];
  activeIdx: number | null;
  hydrated: boolean;
  scope: SearchScope;
  selected: SelectedEntry | null;
  opNotes: OpNote[];
  retiredModels: string[];
  treeSweep: TreeSweep | null;
  revealRequest: RevealRequest | null;
  requestReveal: (root: Root, rel: string, targetIsDir: boolean) => void;
  consumeReveal: (version: number) => void;
  lineRevealRequest: LineRevealRequest | null;
  consumeLineReveal: (version: number) => void;
  toggle: (key: string) => void;
  setSelected: (root: Root, rel: string, isDir: boolean) => void;
  setScope: (scope: SearchScope) => void;
  goBack: () => void;
  openFile: (root: Root, rel: string, line?: number) => void;
  openFilePinned: (root: Root, rel: string, line?: number) => void;
  close: (idx: number) => void;
  setActiveIdx: (idx: number | null) => void;
  pin: (idx: number) => void;
  onUserEdit: () => void;
  isLocked: (root: Root, rel: string) => boolean;
  inline: InlineEdit | null;
  beginCreate: (root: Root, dirRel: string, kind: "file" | "folder") => void;
  beginRename: (root: Root, entryRel: string, parentRel: string, curName: string) => void;
  cancelInline: () => void;
  commitInline: (name: string) => void;
  remove: (root: Root, entryRel: string, parentRel: string, name: string) => void;
  undoTrash: (id: string, root: Root, trashRel: string) => void;
  downloadZip: (root: Root, rel: string, name: string) => Promise<void>;
  uploadMany: (root: Root, dirRel: string, files: File[]) => void;
};

type FilesEditWorkspaceApi = {
  edits: Record<string, EditState>;
  dirty: boolean;
  dirtyKeys: Set<string>;
  onSeed: (key: string, content: string, baseHash: string) => void;
  onEdit: (key: string, id: symbol, patch: EditPatch) => void;
  getEdit: (key: string) => EditState | undefined;
};

export type FilesWorkspaceApi = FilesTreeWorkspaceApi & FilesEditWorkspaceApi;

const FilesTreeContext = createContext<FilesTreeWorkspaceApi | null>(null);
const FilesEditContext = createContext<FilesEditWorkspaceApi | null>(null);

export function useFilesTreeWorkspace(): FilesTreeWorkspaceApi {
  const value = useContext(FilesTreeContext);
  if (!value) throw new Error("FilesWorkspaceProvider required");
  return value;
}
export function useFilesEditWorkspace(): FilesEditWorkspaceApi {
  const value = useContext(FilesEditContext);
  if (!value) throw new Error("FilesWorkspaceProvider required");
  return value;
}
export function useFilesWorkspace(): FilesWorkspaceApi {
  return { ...useFilesTreeWorkspace(), ...useFilesEditWorkspace() };
}

export function UnsavedFilesNotice() {
  const t = useTranslations("files");
  return (
    <p aria-live="polite" className="border-b border-line bg-warning-soft px-4 py-2 text-[12px] text-body-ink">
      {t("unsavedNotice")}{" "}
      <Link href="/files" className="font-semibold text-brand underline">
        {t("returnToFiles")}
      </Link>
    </p>
  );
}

export function FilesWorkspaceProvider({ children }: { children: React.ReactNode }) {
  const t = useTranslations("files");
  const qc = useQueryClient();
  const [, bump] = useState(0);
  const [treeSweep, setTreeSweep] = useState<TreeSweep | null>(null);
  const [revealRequest, setRevealRequest] = useState<RevealRequest | null>(null);
  const [lineRevealRequest, setLineRevealRequest] = useState<LineRevealRequest | null>(null);
  const depsRef = useRef({ t, qc });
  depsRef.current = { t, qc };
  const ctl = useRef(createFilesController({
    getDeps: () => ({
      t: depsRef.current.t,
      confirm: (message) => window.confirm(message),
      fetchJson: (url, init) => fetch(url, init).then((r) => r.json()).catch((e) => ({ ok: false, error: String(e) })),
      fetchRaw: (url) => fetch(url),
      saveBlob: (blob, filename) => {
        const objectUrl = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = objectUrl;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(objectUrl);
      },
      sweep: (root, rel, parent) => {
        const client = depsRef.current.qc;
        client.cancelQueries({ predicate: (query) => filesQueryAffected(query.queryKey, root, rel) });
        client.removeQueries({
          predicate: (query) =>
            filesQueryAffected(query.queryKey, root, rel) && query.queryKey[1] !== "search",
        });
        client.invalidateQueries({ queryKey: ["files", "tree", root, parent] });
        client.invalidateQueries({ queryKey: ["files", "search"] });
        setTreeSweep((prev) => nextTreeSweep(prev, root, parent));
      },
      recordRecent: (root, rel) => {
        try {
          const current = parseRecents(JSON.parse(localStorage.getItem(RECENTS_KEY) ?? "null"));
          localStorage.setItem(RECENTS_KEY, JSON.stringify(pushRecent(current, { root, rel })));
          const currentRepos = parseRecentRepos(JSON.parse(localStorage.getItem(RECENT_REPOS_KEY) ?? "null"));
          localStorage.setItem(RECENT_REPOS_KEY, JSON.stringify(pushRecentRepo(currentRepos, serializeScope(repoOf({ root, rel })))));
        } catch {
          // quota/private mode — best-effort, same as the tabs/expanded persistence a few lines below
        }
      },
      revealLine: (root, rel, line) => setLineRevealRequest((prev) => nextLineRevealRequest(prev, root, rel, line)),
      recordRecentRepo: (scope) => {
        try {
          const current = parseRecentRepos(JSON.parse(localStorage.getItem(RECENT_REPOS_KEY) ?? "null"));
          localStorage.setItem(RECENT_REPOS_KEY, JSON.stringify(pushRecentRepo(current, serializeScope(scope))));
        } catch {
          // quota/private mode — best-effort, same as recordRecent above
        }
      },
    }),
  })).current;
  ctl.setEmit(() => bump((n) => n + 1));
  const treeSnap = ctl.treeSnapshot();
  const editSnap = ctl.editSnapshot();

  useEffect(() => {
    try {
      ctl.loadUi(normalizeState(JSON.parse(localStorage.getItem(STORAGE_KEY) ?? "null")));
    } catch {
      // corrupt/absent → clean state
    }
    ctl.setHydrated(true);
  }, [ctl]);

  useEffect(() => {
    if (!treeSnap.hydrated) return;
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(uiStorage(treeSnap)));
    } catch {
      // quota/private mode — persistence is best-effort
    }
  }, [treeSnap]);

  const dirty = isDirty(editSnap.edits);

  useEffect(() => {
    if (!dirty) return;
    const onUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", onUnload);
    return () => window.removeEventListener("beforeunload", onUnload);
  }, [dirty]);

  const opNotes = ctl.notes();
  const retiredModels = ctl.retired();
  const inline = ctl.inline();

  const treeValue: FilesTreeWorkspaceApi = useMemo(() => ({
    expanded: treeSnap.expanded,
    tabs: treeSnap.tabs,
    activeIdx: treeSnap.activeIdx,
    hydrated: treeSnap.hydrated,
    scope: treeSnap.scope,
    selected: treeSnap.selected,
    opNotes,
    retiredModels,
    treeSweep,
    revealRequest,
    requestReveal: (root, rel, targetIsDir) => {
      const switchTo = revealScopeSwitch(root, rel, ctl.treeSnapshot().scope);
      if (switchTo) ctl.setScope(switchTo);
      setRevealRequest((prev) => nextRevealRequest(prev, root, rel, targetIsDir));
    },
    consumeReveal: (version) => setRevealRequest((prev) => consumedRevealRequest(prev, version)),
    lineRevealRequest,
    consumeLineReveal: (version) => setLineRevealRequest((prev) => consumedLineRevealRequest(prev, version)),
    toggle: ctl.toggle,
    setSelected: ctl.setSelected,
    setScope: ctl.setScope,
    goBack: ctl.goBack,
    openFile: ctl.openFile,
    openFilePinned: ctl.openFilePinned,
    close: ctl.close,
    setActiveIdx: ctl.setActiveIdx,
    pin: ctl.pin,
    onUserEdit: ctl.onUserEdit,
    isLocked: ctl.isLocked,
    inline,
    beginCreate: ctl.beginCreate,
    beginRename: ctl.beginRename,
    cancelInline: ctl.cancelInline,
    commitInline: ctl.commitInline,
    remove: ctl.remove,
    undoTrash: ctl.undoTrash,
    downloadZip: ctl.downloadZip,
    uploadMany: ctl.uploadMany,
  }), [treeSnap, opNotes, retiredModels, treeSweep, revealRequest, lineRevealRequest, inline, ctl]);

  const editValue: FilesEditWorkspaceApi = useMemo(() => ({
    edits: editSnap.edits,
    dirty,
    dirtyKeys: dirtyKeySet(editSnap.edits),
    onSeed: ctl.onSeed,
    onEdit: ctl.onEdit,
    getEdit: ctl.getEdit,
  }), [editSnap, dirty, ctl]);

  return (
    <FilesTreeContext.Provider value={treeValue}>
      <FilesEditContext.Provider value={editValue}>
        {children}
      </FilesEditContext.Provider>
    </FilesTreeContext.Provider>
  );
}
