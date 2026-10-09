import type { Root, RepoRef, SearchScope } from "@/server/collectors/files";
import {
  closeTab,
  createEdit,
  openPreview,
  patchEdit,
  pinTab,
  tabKey,
  type EditPatch,
  type EditState,
  type SelectedEntry,
  type Tab,
} from "./filesState";

export type FilesTreeState = {
  expanded: Set<string>;
  tabs: Tab[];
  activeIdx: number | null;
  hydrated: boolean;
  scope: SearchScope;
  selected: SelectedEntry | null;
};

export type FilesEditState = {
  edits: Record<string, EditState>;
};

export type FilesWorkspaceState = FilesTreeState & FilesEditState;

export function isDirty(edits: Record<string, EditState>): boolean {
  return Object.values(edits).some((edit) => edit.content !== edit.savedContent);
}

export function dirtyKeys(edits: Record<string, EditState>): Set<string> {
  return new Set(Object.keys(edits).filter((key) => edits[key].content !== edits[key].savedContent));
}

export function shouldConfirmLeave(pathname: string, href: string, dirty: boolean): boolean {
  return dirty && pathname === "/files" && href !== "/files";
}

export function uiStorage(state: { expanded: Iterable<string>; tabs: Tab[]; activeIdx: number | null; scope: SearchScope; selected: SelectedEntry | null }) {
  return { expanded: [...state.expanded], tabs: state.tabs, activeIdx: state.activeIdx, scope: state.scope, selected: state.selected };
}

export type TreeSweep = { version: number; root: Root; rel: string };

// F2: the tree renders through headless-tree's own data loader (a plain fetch cached outside
// TanStack Query), so `sweep`'s query-cache invalidation below never reaches it. The provider
// bumps this counter on every sweep and republishes the swept directory; FileTree's effect reacts
// to the new object identity and reloads exactly that directory via headless-tree's own
// invalidation API. Pure so the bump itself is testable without mounting the provider.
export function nextTreeSweep(prev: TreeSweep | null, root: Root, rel: string): TreeSweep {
  return { version: (prev?.version ?? 0) + 1, root, rel };
}

export type RevealRequest = { version: number; root: Root; rel: string; targetIsDir: boolean };

// Sibling to nextTreeSweep, same reasoning (F2/§5 #10): bumping a version on every request gives a
// fresh object identity even for a repeat click on the same file, so FileTree's effect always fires.
export function nextRevealRequest(prev: RevealRequest | null, root: Root, rel: string, targetIsDir: boolean): RevealRequest {
  return { version: (prev?.version ?? 0) + 1, root, rel, targetIsDir };
}

// One-shot channel (review F4): FileTree calls this once its reveal walk for `version` finishes, so
// a remounted tree doesn't replay a stale request. Guarded by version so a completion for an OLDER
// request can never clear a NEWER one that raced ahead of it.
export function consumedRevealRequest(current: RevealRequest | null, version: number): RevealRequest | null {
  return current && current.version === version ? null : current;
}

// §4: the shared scope vocabulary's two pure resolvers.
export function repoOf(file: { root: Root; rel: string }): RepoRef {
  return file.root === "vault" ? { kind: "vault" } : { kind: "repo", name: file.rel.split("/")[0] };
}

export function resolveCurrentRepo(
  active: { root: Root; rel: string } | null, selected: RepoRef | null, treeScope: SearchScope | null,
): SearchScope {
  if (active) return repoOf(active);
  if (selected) return selected;
  if (treeScope) return treeScope;
  return { kind: "all" };
}

export function serializeScope(scope: SearchScope): string {
  if (scope.kind === "all") return "all";
  if (scope.kind === "vault") return "vault";
  return `repo:${scope.name}`;
}

// requestReveal's auto-switch-scope decision (spec §5 "opening a file outside the current scope",
// round-1 F3): the reveal target's own scope when it differs from `currentScope`, else null. Keys
// off the TARGET's own repo (repoOf), never the active tab's. A repo target's OWN root reveal
// always carries rel === scope.name (never ""; see displayedTarget/scopedRel's own repo-root
// handling) — root==="repos" && rel==="" never legitimately occurs from any caller in this plan, so
// it is treated as "no switch" here rather than computing repoOf's degenerate
// {kind:"repo",name:""}. A VAULT root reveal, by contrast, IS carried as rel==="" (vault has no name
// prefix) and DOES need the switch — FileEditor's root-crumb click for a vault file (Task 4) and the
// tree-panel breadcrumb's root click (Task 11) for a vault scope both produce exactly this shape.
export function revealScopeSwitch(root: Root, rel: string, currentScope: SearchScope): RepoRef | null {
  if (root === "repos" && rel === "") return null;
  const fileScope = repoOf({ root, rel });
  return serializeScope(fileScope) !== serializeScope(currentScope) ? fileScope : null;
}

// Shared "open from search" handler (round-2 F1, downgraded HIGH->MEDIUM: the file DOES open today,
// only the tree fails to follow). FileSearch's name/content results and QuickOpen all resolve a
// target that may be outside the current tree scope; calling `open` alone (ws.openFile/
// ws.openFilePinned) only ever sets `selected` — it never switches scope or reveals the tree. Every
// search/Quick Open call site (page.tsx, Task 11) routes through this instead of calling `open`
// directly. `reveal` MUST run first — see this function's own Interfaces note above for why the
// order also fixes where `selected` ends up.
export function openFromSearch(
  reveal: (root: Root, rel: string, targetIsDir: boolean) => void,
  open: (root: Root, rel: string, line?: number) => void,
  root: Root,
  rel: string,
  line?: number,
): void {
  reveal(root, rel, false);
  open(root, rel, line);
}

// Mirrors RevealRequest/nextRevealRequest/consumedRevealRequest above exactly, for a DIFFERENT
// concern: jumping the OPEN EDITOR to a line (a content-search result), not scrolling the tree.
export type LineRevealRequest = { version: number; root: Root; rel: string; line: number };
export function nextLineRevealRequest(prev: LineRevealRequest | null, root: Root, rel: string, line: number): LineRevealRequest {
  return { version: (prev?.version ?? 0) + 1, root, rel, line };
}
export function consumedLineRevealRequest(current: LineRevealRequest | null, version: number): LineRevealRequest | null {
  return current && current.version === version ? null : current;
}

// Round-1 F1: FileEditor is the sole consumer, and only one tab is ever open at a time — a pending
// request must not fire against whichever file happens to be mounted, only the one it targets.
// `activeTabKey` is the same `tabKey(file)` string FileEditor already computes for `ws.getEdit`.
export function shouldRevealLine(request: LineRevealRequest | null, activeTabKey: string | null): boolean {
  return !!request && !!activeTabKey && tabKey({ root: request.root, rel: request.rel }) === activeTabKey;
}

export function affectedPath(candidate: { root: Root; rel: string }, root: Root, rel: string): boolean {
  if (candidate.root !== root) return false;
  if (rel === "") return true;
  return candidate.rel === rel || candidate.rel.startsWith(rel + "/");
}

export function remapRel(rel: string, from: string, to: string): string {
  if (rel === from) return to;
  if (from !== "" && rel.startsWith(from + "/")) return to + rel.slice(from.length);
  return rel;
}

function parseEditKey(key: string): { root: Root; rel: string } | null {
  if (key.startsWith("repos:")) return { root: "repos", rel: key.slice(6) };
  if (key.startsWith("vault:")) return { root: "vault", rel: key.slice(6) };
  return null;
}

type PathPieces = {
  tabs: Tab[];
  activeIdx: number | null;
  edits: Record<string, EditState>;
  expanded: Iterable<string>;
  selected: SelectedEntry | null;
};

export function monacoPath(root: Root, rel: string): string {
  return `${root}/${rel}`;
}

export function retiredMonacoPaths(tabs: Tab[], root: Root, rel: string): string[] {
  return tabs.filter((tab) => affectedPath(tab, root, rel)).map((tab) => monacoPath(tab.root, tab.rel));
}

export function renameCollision(
  tabs: Tab[],
  root: Root,
  from: string,
  to: string,
  edits: Record<string, EditState> = {},
): boolean {
  const taken = new Set<string>();
  const mark = (item: { root: Root; rel: string }) => {
    if (item.root === root && !affectedPath(item, root, from)) taken.add(item.rel);
  };
  for (const tab of tabs) mark(tab);
  for (const key of Object.keys(edits)) {
    const parsed = parseEditKey(key);
    if (parsed) mark(parsed);
  }
  const destOf = (rel: string) => remapRel(rel, from, to);
  if (tabs.some((tab) => tab.root === root && affectedPath(tab, root, from) && taken.has(destOf(tab.rel)))) return true;
  return Object.keys(edits).some((key) => {
    const parsed = parseEditKey(key);
    return !!parsed && parsed.root === root && affectedPath(parsed, root, from) && taken.has(destOf(parsed.rel));
  });
}

export function applyRename(state: PathPieces, root: Root, from: string, to: string): Omit<PathPieces, "expanded"> & { expanded: string[] } {
  const tabs = state.tabs.map((tab) =>
    affectedPath(tab, root, from) ? { ...tab, rel: remapRel(tab.rel, from, to) } : tab,
  );
  const edits: Record<string, EditState> = {};
  for (const [key, edit] of Object.entries(state.edits)) {
    const parsed = parseEditKey(key);
    if (!parsed || !affectedPath(parsed, root, from)) {
      edits[key] = edit;
      continue;
    }
    edits[tabKey({ root, rel: remapRel(parsed.rel, from, to) })] = { ...edit, id: Symbol() };
  }
  const expanded = [...state.expanded].map((key) => {
    const parsed = parseEditKey(key);
    if (!parsed || !affectedPath(parsed, root, from)) return key;
    return `${root}:${remapRel(parsed.rel, from, to)}`;
  });
  const selected = state.selected && affectedPath(state.selected, root, from)
    ? { ...state.selected, rel: remapRel(state.selected.rel, from, to) }
    : state.selected;
  return { tabs, activeIdx: state.activeIdx, edits, expanded, selected };
}

export function applyDelete(state: PathPieces, root: Root, rel: string): Omit<PathPieces, "expanded"> & { expanded: string[] } {
  let tabs = state.tabs;
  let activeIdx = state.activeIdx;
  for (let i = tabs.length - 1; i >= 0; i--) {
    if (affectedPath(tabs[i], root, rel)) {
      const next = closeTab(tabs, activeIdx, i);
      tabs = next.tabs;
      activeIdx = next.activeIdx;
    }
  }
  const edits: Record<string, EditState> = {};
  for (const [key, edit] of Object.entries(state.edits)) {
    const parsed = parseEditKey(key);
    if (parsed && affectedPath(parsed, root, rel)) continue;
    edits[key] = edit;
  }
  const expanded = [...state.expanded].filter((key) => {
    const parsed = parseEditKey(key);
    return !parsed || !affectedPath(parsed, root, rel);
  });
  const selected = state.selected && affectedPath(state.selected, root, rel)
    ? (parentRel(rel) === "" ? null : { root, rel: parentRel(rel), isDir: true as const })
    : state.selected;
  return { tabs, activeIdx, edits, expanded, selected };
}

export function overlappingLock(locks: { root: Root; rel: string }[], root: Root, rel: string): boolean {
  return locks.some((lock) => affectedPath(lock, root, rel) || affectedPath({ root, rel }, lock.root, lock.rel));
}

export function affectedBusy(edits: Record<string, EditState>, root: Root, rel: string): boolean {
  return Object.entries(edits).some(([key, edit]) => {
    const parsed = parseEditKey(key);
    return !!parsed && affectedPath(parsed, root, rel) && (edit.saving || edit.reloading);
  });
}

export function dirtyDescendantCount(edits: Record<string, EditState>, root: Root, rel: string): number {
  return Object.entries(edits).filter(([key, edit]) => {
    const parsed = parseEditKey(key);
    return parsed && affectedPath(parsed, root, rel) && edit.content !== edit.savedContent;
  }).length;
}

export function classifyMutation(res: { ok?: boolean; error?: string; effect?: string; code?: string; audit?: string } | null | undefined): {
  kind: "ok" | "applied-unrecorded" | "unconfirmed" | "refused";
  error?: string;
} {
  if (!res || typeof res !== "object") return { kind: "unconfirmed" };
  if (res.ok) return { kind: "ok" };
  if (res.effect === "applied") return { kind: "applied-unrecorded" };
  if (res.effect === "unconfirmed") return { kind: "unconfirmed" };
  if (res.effect === "not-applied") return { kind: "refused", error: res.error };
  return { kind: "unconfirmed" };
}

export function filesQueryAffected(queryKey: readonly unknown[], root: Root, rel: string): boolean {
  if (queryKey[0] !== "files") return false;
  if (queryKey[1] === "search") return true;
  if (queryKey[1] !== "read" && queryKey[1] !== "tree") return false;
  const qRoot = queryKey[2];
  const qRel = queryKey[3];
  return qRoot === root && typeof qRel === "string" && affectedPath({ root, rel: qRel }, root, rel);
}

export function parentRel(rel: string): string {
  const slash = rel.lastIndexOf("/");
  return slash >= 0 ? rel.slice(0, slash) : "";
}

// §7: the folder the phone nav currently lists — a deterministic function of `selected`.
export function displayedDir(selected: { rel: string; isDir: boolean } | null): string {
  return selected === null ? "" : selected.isDir ? selected.rel : parentRel(selected.rel);
}

// §7 round-1 F1: displayedDir alone is ambiguous for a repo scope with no selection ("" is the
// WHOLE repos root, not this one repo) — falls back to the repo's own root; a real selection's
// rel is already name-prefixed, so displayedDir's value is used as-is, never re-joined with name.
export function displayedTarget(scope: RepoRef, selected: { rel: string; isDir: boolean } | null): { root: Root; rel: string } {
  if (scope.kind === "vault") return { root: "vault", rel: displayedDir(selected) };
  return { root: "repos", rel: displayedDir(selected) || scope.name };
}

// Strips/restores a repo scope's own name prefix for the tree-panel/phone breadcrumbs. Vault has
// no such prefix — both are the identity there.
export function relativeToScope(scope: RepoRef, rel: string): string {
  if (scope.kind === "vault") return rel;
  if (rel === scope.name) return "";
  return rel.startsWith(`${scope.name}/`) ? rel.slice(scope.name.length + 1) : rel;
}
export function scopedRel(scope: RepoRef, relative: string): string {
  if (scope.kind === "vault") return relative;
  return relative === "" ? scope.name : `${scope.name}/${relative}`;
}

const UPLOAD_CAP = 5 * 1024 * 1024;

export type OpNote = {
  id: string; tone: "pending" | "ok" | "warning" | "error"; text: string; sticky: boolean;
  undo?: { root: Root; trashRel: string };
};
export type InlineEdit =
  | { kind: "create"; root: Root; dirRel: string; fileKind: "file" | "folder" }
  | { kind: "rename"; root: Root; entryRel: string; parentRel: string; curName: string };

export type WorkspaceDeps = {
  t: (key: string, values?: Record<string, string | number>) => string;
  confirm: (message: string) => boolean;
  fetchJson: (url: string, init?: RequestInit) => Promise<unknown>;
  fetchRaw: (url: string) => Promise<Response>;
  saveBlob: (blob: Blob, filename: string) => void;
  sweep: (root: Root, rel: string, parent: string) => void;
  recordRecent: (root: Root, rel: string) => void;
  recordRecentRepo: (scope: SearchScope) => void;
  revealLine: (root: Root, rel: string, line: number) => void;
};

export type FilesController = ReturnType<typeof createFilesController>;

export function createFilesController(opts: { getDeps: () => WorkspaceDeps }) {
  let treeSnap: FilesTreeState = { expanded: new Set(), tabs: [], activeIdx: null, hydrated: false, scope: { kind: "repo", name: "jax-os" }, selected: null };
  let editSnap: FilesEditState = { edits: {} };
  let locks: { root: Root; rel: string }[] = [];
  let notes: OpNote[] = [];
  let retired: string[] = [];
  let inline: InlineEdit | null = null;
  let emit = () => {};

  const deps = () => opts.getDeps();
  const publish = () => emit();

  const setNote = (id: string, tone: OpNote["tone"], text: string, sticky: boolean, undo?: OpNote["undo"]) => {
    notes = [...notes.filter((note) => note.id !== id), { id, tone, text, sticky, undo }];
    publish();
  };

  const lockEdits = (root: Root, rels: string[], locked: boolean) => {
    let edits = editSnap.edits;
    for (const [key, edit] of Object.entries(edits)) {
      const parsed = parseEditKey(key);
      if (!parsed || !rels.some((rel) => affectedPath(parsed, root, rel))) continue;
      edits = patchEdit(edits, key, edit.id, { pathLocked: locked });
    }
    editSnap = { edits };
  };

  async function runLocked(root: Root, rels: string[], parent: string, work: () => Promise<string | void>) {
    const id = `${root}:${rels[0] ?? ""}`;
    if (rels.some((rel) => affectedBusy(editSnap.edits, root, rel))) {
      setNote(id, "warning", deps().t("waitPending"), false);
      return;
    }
    if (rels.some((rel) => overlappingLock(locks, root, rel))) return;
    const added = rels.map((rel) => ({ root, rel }));
    locks = [...locks, ...added];
    lockEdits(root, rels, true);
    setNote(id, "pending", deps().t("opPending"), false);
    let extra = rels[0] ?? "";
    try {
      extra = (await work()) || extra;
    } finally {
      const drop = new Set(added);
      locks = locks.filter((lock) => !drop.has(lock));
      lockEdits(root, extra !== rels[0] ? [...rels, extra] : rels, false);
      deps().sweep(root, rels[0] ?? "", parent);
      if (extra && extra !== rels[0]) deps().sweep(root, extra, parent);
      publish();
    }
  }

  function applyOutcome(id: string, res: unknown, onApplied: () => void) {
    const classified = classifyMutation(res as { ok?: boolean; error?: string; effect?: string });
    if (classified.kind === "ok") {
      onApplied();
      setNote(id, "ok", deps().t("opConfirmed"), false);
      return;
    }
    if (classified.kind === "applied-unrecorded") {
      onApplied();
      setNote(id, "warning", deps().t("appliedUnrecorded"), true);
      return;
    }
    if (classified.kind === "unconfirmed") {
      setNote(id, "warning", deps().t("checkCurrentState"), true);
      return;
    }
    const error = classified.error;
    const text = error === "directory not empty" ? deps().t("dirNotEmpty")
      : error === "already exists" ? deps().t("nameExists")
        : deps().t("opFailed");
    setNote(id, "error", text, false);
  }

  function dropReplacedPreview(root: Root, rel: string) {
    const key = tabKey({ root, rel });
    if (treeSnap.tabs.some((tab) => tabKey(tab) === key)) return;
    const prev = treeSnap.tabs.find((tab) => !tab.pinned);
    if (!prev) return;
    retired = [monacoPath(prev.root, prev.rel)]; // spec §4 item 1
    const drop = tabKey(prev);
    if (!editSnap.edits[drop]) return;
    const { [drop]: _removed, ...rest } = editSnap.edits;
    editSnap = { edits: rest };
  }

  return {
    setEmit(fn: () => void) { emit = fn; },
    snapshot: () => ({ ...treeSnap, ...editSnap }),
    treeSnapshot: () => treeSnap,
    editSnapshot: () => editSnap,
    locks: () => locks,
    notes: () => notes,
    retired: () => retired,
    inline: () => inline,
    getEdit: (key: string) => editSnap.edits[key],
    isLocked: (root: Root, rel: string) => overlappingLock(locks, root, rel),
    loadUi(next: { expanded: string[]; tabs: Tab[]; activeIdx: number | null; scope: SearchScope; selected: SelectedEntry | null }) {
      treeSnap = { ...treeSnap, expanded: new Set(next.expanded), tabs: next.tabs, activeIdx: next.activeIdx, scope: next.scope, selected: next.selected };
      publish();
    },
    setHydrated(hydrated: boolean) {
      treeSnap = { ...treeSnap, hydrated };
      publish();
    },
    toggle(key: string) {
      const expanded = new Set(treeSnap.expanded);
      if (expanded.has(key)) expanded.delete(key);
      else expanded.add(key);
      treeSnap = { ...treeSnap, expanded };
      publish();
    },
    openFile(root: Root, rel: string, line?: number) {
      if (overlappingLock(locks, root, rel)) return;
      dropReplacedPreview(root, rel);
      const r = openPreview(treeSnap.tabs, { root, rel });
      treeSnap = { ...treeSnap, tabs: r.tabs, activeIdx: r.activeIdx, selected: { root, rel, isDir: false } };
      deps().recordRecent(root, rel);
      deps().recordRecentRepo(repoOf({ root, rel }));
      if (line !== undefined) deps().revealLine(root, rel, line);
      publish();
    },
    openFilePinned(root: Root, rel: string, line?: number) {
      if (overlappingLock(locks, root, rel)) return;
      dropReplacedPreview(root, rel);
      const r = openPreview(treeSnap.tabs, { root, rel });
      treeSnap = { ...treeSnap, tabs: pinTab(r.tabs, r.activeIdx), activeIdx: r.activeIdx, selected: { root, rel, isDir: false } };
      deps().recordRecent(root, rel);
      deps().recordRecentRepo(repoOf({ root, rel }));
      if (line !== undefined) deps().revealLine(root, rel, line);
      publish();
    },
    close(idx: number) {
      const tab = treeSnap.tabs[idx];
      if (!tab) return;
      const k = tabKey(tab);
      const edit = editSnap.edits[k];
      if (edit?.pathLocked || overlappingLock(locks, tab.root, tab.rel)) return;
      if (edit && edit.content !== edit.savedContent) {
        const name = tab.rel.split("/").pop() ?? tab.rel;
        if (!deps().confirm(deps().t("closeDirtyConfirm", { name }))) return;
      }
      retired = [monacoPath(tab.root, tab.rel)]; // spec §4 item 1
      const { [k]: _removed, ...rest } = editSnap.edits;
      editSnap = { edits: rest };
      const r = closeTab(treeSnap.tabs, treeSnap.activeIdx, idx);
      treeSnap = { ...treeSnap, tabs: r.tabs, activeIdx: r.activeIdx };
      publish();
    },
    setActiveIdx(idx: number | null) {
      treeSnap = { ...treeSnap, activeIdx: idx };
      publish();
    },
    pin(idx: number) {
      treeSnap = { ...treeSnap, tabs: pinTab(treeSnap.tabs, idx) };
      publish();
    },
    setSelected(root: Root, rel: string, isDir: boolean) {
      treeSnap = { ...treeSnap, selected: { root, rel, isDir } };
      publish();
    },
    setScope(scope: SearchScope) {
      treeSnap = { ...treeSnap, scope, selected: null };
      deps().recordRecentRepo(scope);
      publish();
    },
    // Phone nav Back (spec §7): relativeToScope's "" check covers both the null-selection root
    // state and the {rel:scope.name,isDir:true} root state goBack itself produces — repeated
    // presses at the root are a true no-op. At a repo/vault's own root (spec §4.5) it now steps
    // out to the Files root instead of no-oping.
    goBack() {
      if (treeSnap.scope.kind === "all") return; // true no-op at the true top
      const dir = displayedDir(treeSnap.selected);
      if (relativeToScope(treeSnap.scope, dir) === "") {
        treeSnap = { ...treeSnap, scope: { kind: "all" }, selected: null };
        publish();
        return;
      }
      const root: Root = treeSnap.scope.kind === "vault" ? "vault" : "repos";
      treeSnap = { ...treeSnap, selected: { root, rel: parentRel(dir), isDir: true } };
      publish();
    },
    onSeed(key: string, content: string, baseHash: string) {
      const parsed = parseEditKey(key);
      if (parsed && overlappingLock(locks, parsed.root, parsed.rel)) return;
      if (editSnap.edits[key]) return;
      editSnap = { edits: { ...editSnap.edits, [key]: createEdit(content, baseHash) } };
      publish();
    },
    onEdit(key: string, id: symbol, patch: EditPatch) {
      const current = editSnap.edits[key];
      const mutating = patch.content !== undefined || patch.savedContent !== undefined || patch.baseHash !== undefined;
      if (current?.pathLocked && mutating) return;
      const parsed = parseEditKey(key);
      if (parsed && overlappingLock(locks, parsed.root, parsed.rel) && mutating) return;
      editSnap = { edits: patchEdit(editSnap.edits, key, id, patch) };
      publish();
    },
    onUserEdit() {
      const idx = treeSnap.activeIdx;
      if (idx === null) return;
      const tab = treeSnap.tabs[idx];
      if (!tab || tab.pinned) return;
      treeSnap = { ...treeSnap, tabs: pinTab(treeSnap.tabs, idx) };
      publish();
    },
    beginCreate(root: Root, dirRel: string, kind: "file" | "folder") {
      // Round 1 F2: force the target directory open, otherwise a create triggered from a
      // collapsed row (or the root) leaves the inline input mounted where nothing is visible.
      const key = `${root}:${dirRel}`;
      if (!treeSnap.expanded.has(key)) treeSnap = { ...treeSnap, expanded: new Set(treeSnap.expanded).add(key) };
      inline = { kind: "create", root, dirRel, fileKind: kind };
      publish();
    },
    beginRename(root: Root, entryRel: string, parentRel: string, curName: string) {
      inline = { kind: "rename", root, entryRel, parentRel, curName };
      publish();
    },
    cancelInline() {
      inline = null;
      publish();
    },
    commitInline(name: string) {
      const op = inline;
      if (!op) return;
      const trimmed = name.trim();
      inline = null;
      if (!trimmed) { publish(); return; }
      if (op.kind === "create") {
        const rel = op.dirRel ? `${op.dirRel}/${trimmed}` : trimmed;
        void runLocked(op.root, [rel], op.dirRel, async () => {
          const res = await deps().fetchJson("/api/files/create", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ root: op.root, relParentDir: op.dirRel, basename: trimmed, kind: op.fileKind }),
          });
          applyOutcome(`${op.root}:${rel}`, res, () => {});
        });
      } else {
        if (trimmed === op.curName) { publish(); return; }
        const to = op.parentRel ? `${op.parentRel}/${trimmed}` : trimmed;
        if (renameCollision(treeSnap.tabs, op.root, op.entryRel, to, editSnap.edits)) {
          setNote(`${op.root}:${op.entryRel}`, "warning", deps().t("closeTargetFirst"), false);
          publish();
          return;
        }
        void runLocked(op.root, [op.entryRel, to], op.parentRel, async () => {
          const res = await deps().fetchJson("/api/files/rename", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ root: op.root, relFrom: op.entryRel, relTo: to }),
          });
          applyOutcome(`${op.root}:${op.entryRel}`, res, () => {
            if (renameCollision(treeSnap.tabs, op.root, op.entryRel, to, editSnap.edits)) {
              setNote(`${op.root}:${op.entryRel}`, "warning", deps().t("closeTargetFirst"), false);
              return;
            }
            retired = retiredMonacoPaths(treeSnap.tabs, op.root, op.entryRel);
            const next = applyRename({
              tabs: treeSnap.tabs,
              activeIdx: treeSnap.activeIdx,
              edits: editSnap.edits,
              expanded: treeSnap.expanded,
              selected: treeSnap.selected,
            }, op.root, op.entryRel, to);
            treeSnap = { ...treeSnap, tabs: next.tabs, activeIdx: next.activeIdx, expanded: new Set(next.expanded), selected: next.selected };
            editSnap = { edits: next.edits };
          });
          return to;
        });
      }
      publish();
    },
    remove(root: Root, entryRel: string, parent: string, name: string) {
      const id = `${root}:${entryRel}`;
      void runLocked(root, [entryRel], parent, async () => {
        // Non-blocking (delete is reversible now — §5 item 6); overwrites runLocked's own
        // synchronous "pending" note with this one before any paint, exactly like the pending
        // note itself does — both writes land in the same microtask.
        const count = dirtyDescendantCount(editSnap.edits, root, entryRel);
        if (count > 0) setNote(id, "warning", deps().t("deleteDirtyNote", { name, count }), false);
        const res = await deps().fetchJson("/api/files/delete", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ root, rel: entryRel }),
        });
        const classified = classifyMutation(res as { ok?: boolean; error?: string; effect?: string });
        // round 2 F2: an audit-write failure after an applied delete ("applied-unrecorded") is still
        // an applied delete — treat it the same as "ok" here, exactly like undoTrash already does for
        // its own "ok" | "applied-unrecorded" pair, so the entry leaves the tree and Undo still works.
        if (classified.kind === "ok" || classified.kind === "applied-unrecorded") {
          retired = retiredMonacoPaths(treeSnap.tabs, root, entryRel);
          const next = applyDelete({
            tabs: treeSnap.tabs,
            activeIdx: treeSnap.activeIdx,
            edits: editSnap.edits,
            expanded: treeSnap.expanded,
            selected: treeSnap.selected,
          }, root, entryRel);
          treeSnap = { ...treeSnap, tabs: next.tabs, activeIdx: next.activeIdx, expanded: new Set(next.expanded), selected: next.selected };
          editSnap = { edits: next.edits };
          const trashRel = (res as { trashRel?: string }).trashRel;
          if (trashRel) setNote(id, "ok", deps().t("deletedUndo"), false, { root, trashRel });
          else setNote(id, "ok", deps().t("opConfirmed"), false);
        } else {
          applyOutcome(id, res, () => {});
        }
      });
    },
    undoTrash(id: string, root: Root, trashRel: string) {
      void (async () => {
        const res = await deps().fetchJson("/api/files/trash/restore", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ root, trashRel }),
        });
        const classified = classifyMutation(res as { ok?: boolean; error?: string; effect?: string });
        if (classified.kind === "ok" || classified.kind === "applied-unrecorded") {
          // Round 1 F3: undoTrash bypasses runLocked (it isn't reserving a path), so it never got
          // runLocked's automatic sweep. Sweep the RESTORED path (from the response), not trashRel —
          // trashRel is a `.jax-trash/...` implementation path the tree never shows.
          const restoredRel = (res as { restoredRel?: string }).restoredRel;
          if (restoredRel) deps().sweep(root, restoredRel, parentRel(restoredRel));
          setNote(id, "ok", deps().t("undoRestored"), false);
        } else {
          setNote(id, "error", deps().t("undoFailed"), false);
        }
      })();
    },
    async downloadZip(root: Root, rel: string, name: string) {
      const id = `${root}:${rel}`;
      let res: Response;
      try {
        res = await deps().fetchRaw(`/api/files/zip?root=${root}&rel=${encodeURIComponent(rel)}`);
      } catch {
        setNote(id, "error", deps().t("opFailed"), false);
        return;
      }
      if (!(res.headers.get("content-type") ?? "").startsWith("application/zip")) {
        const body = (await res.json().catch(() => null)) as { error?: string } | null;
        setNote(id, "error", body?.error === "zip-too-large" ? deps().t("zipTooLarge") : deps().t("opFailed"), false);
        return;
      }
      const excluded = Number(res.headers.get("X-Secrets-Excluded") ?? "0");
      deps().saveBlob(await res.blob(), `${name}.zip`);
      if (excluded > 0) setNote(id, "warning", deps().t("secretsExcluded", { count: excluded }), false);
      else setNote(id, "ok", deps().t("opConfirmed"), false);
    },
    uploadMany(root: Root, dirRel: string, files: File[]) {
      const rels = files.map((file) => (dirRel ? `${dirRel}/${file.name}` : file.name));
      const noteId = `${root}:${rels[0] ?? dirRel}`;
      void runLocked(root, rels.length > 0 ? rels : [dirRel], dirRel, async () => {
        const failed: string[] = [];
        for (const file of files) {
          if (file.size > UPLOAD_CAP) {
            failed.push(`${file.name}: ${deps().t("opFailed")}`);
            continue;
          }
          const fd = new FormData();
          fd.append("root", root);
          fd.append("relParentDir", dirRel);
          fd.append("file", file);
          const res = await deps().fetchJson("/api/files/upload", { method: "POST", body: fd });
          const classified = classifyMutation(res as { ok?: boolean; effect?: string });
          if (classified.kind === "ok") continue;
          if (classified.kind === "applied-unrecorded") failed.push(`${file.name}: ${deps().t("appliedUnrecorded")}`);
          else if (classified.kind === "unconfirmed") failed.push(`${file.name}: ${deps().t("checkCurrentState")}`);
          else failed.push(`${file.name}: ${deps().t("opFailed")}`);
        }
        if (failed.length > 0) {
          setNote(noteId, "warning", deps().t("uploadFailedSome", {
            failed: failed.length, total: files.length, names: failed.join(" · "),
          }), true);
        } else {
          setNote(noteId, "ok", deps().t("opConfirmed"), false);
        }
      });
    },
  };
}
