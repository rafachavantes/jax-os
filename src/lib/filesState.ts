import type { Root, RepoRef, SearchScope } from "@/server/collectors/files"; // type-only: erased at build, safe in client code

export type Tab = { root: Root; rel: string; pinned: boolean };
export type EditState = {
  id: symbol;
  revision: number;
  content: string;
  savedContent: string;
  baseHash: string;
  saving: boolean;
  reloading: boolean;
  stale: boolean;
  writeErr: boolean;
  justSaved: boolean;
  reloadKept: boolean;
  pathLocked: boolean;
  auditWarning: boolean;
  unconfirmed: boolean;
  auditUnavailable: boolean;
};
export type EditPatch = Partial<Omit<EditState, "id" | "revision">>;
export function createEdit(content: string, baseHash: string): EditState {
  return { id: Symbol(), revision: 0, content, savedContent: content, baseHash,
    saving: false, reloading: false, stale: false, writeErr: false, justSaved: false,
    reloadKept: false, pathLocked: false, auditWarning: false, unconfirmed: false, auditUnavailable: false };
}
export function patchEdit(edits: Record<string, EditState>, key: string, id: symbol, patch: EditPatch): Record<string, EditState> {
  const current = edits[key];
  if (!current || current.id !== id) return edits;
  const next = { ...current, ...patch };
  const changed =
    (patch.content !== undefined && patch.content !== current.content)
    || (patch.savedContent !== undefined && patch.savedContent !== current.savedContent)
    || (patch.baseHash !== undefined && patch.baseHash !== current.baseHash);
  if (changed) next.revision = current.revision + 1;
  return { ...edits, [key]: next };
}
export type SelectedEntry = { root: Root; rel: string; isDir: boolean };
export type FilesUiState = { expanded: string[]; tabs: Tab[]; activeIdx: number | null; scope: SearchScope; selected: SelectedEntry | null };

// §5's fixed first-run pick — a simple, deterministic default; the switcher is one click away.
export const DEFAULT_SCOPE: RepoRef = { kind: "repo", name: "jax-os" };

function normalizeScope(raw: unknown): SearchScope {
  if (typeof raw !== "object" || raw === null) return DEFAULT_SCOPE;
  const { kind, name } = raw as Record<string, unknown>;
  if (kind === "all") return { kind: "all" };
  if (kind === "vault") return { kind: "vault" };
  if (kind === "repo" && typeof name === "string" && name !== "") return { kind: "repo", name };
  return DEFAULT_SCOPE;
}

function normalizeSelected(raw: unknown): SelectedEntry | null {
  if (typeof raw !== "object" || raw === null) return null;
  const { root, rel, isDir } = raw as Record<string, unknown>;
  if ((root !== "repos" && root !== "vault") || typeof rel !== "string" || rel === "" || typeof isDir !== "boolean") return null;
  return { root, rel, isDir };
}

export const STORAGE_KEY = "jax-os.files.v1";

export function tabKey(t: { root: Root; rel: string }): string {
  return `${t.root}:${t.rel}`;
}

// spec §4 item 4: one label per open tab, escalating only as far as needed to stay unique.
// segmentsUp clamps at the rel's own length, so a shallow path (fewer than n+1 segments) just
// repeats its shortest available label instead of throwing — harmless, since a repeat only
// matters if it still collides, and depth 3 (tabKey) always resolves that case.
function segmentsUp(rel: string, n: number): string {
  const parts = rel.split("/");
  return parts.slice(Math.max(0, parts.length - 1 - n)).join("/");
}

export function disambiguateTabLabels(tabs: Tab[]): string[] {
  const labelAt = (i: number, depth: number): string =>
    depth >= 3 ? tabKey(tabs[i]) : segmentsUp(tabs[i].rel, depth);
  const labels = new Array<string>(tabs.length);
  // Resolve per collision group, independently (round-1 F2): a colliding pair escalates only ITS
  // OWN depth — an unrelated tab whose label is already unique at this depth is never pulled along.
  const resolve = (indices: number[], depth: number) => {
    const groups = new Map<string, number[]>();
    for (const i of indices) {
      const label = labelAt(i, depth);
      const group = groups.get(label) ?? [];
      group.push(i);
      groups.set(label, group);
    }
    for (const group of groups.values()) {
      if (group.length === 1 || depth >= 3) {
        for (const i of group) labels[i] = labelAt(i, depth);
      } else {
        resolve(group, depth + 1);
      }
    }
  };
  resolve(tabs.map((_, i) => i), 0);
  return labels;
}

// Defensive load of persisted UI state: junk/corrupt/hand-edited localStorage
// must yield a clean state, never a crash. Also enforces the invariants the
// UI assumes: at most ONE preview (pinned:false) tab, no duplicate tabs,
// activeIdx in range.
export function normalizeState(raw: unknown): FilesUiState {
  const empty: FilesUiState = { expanded: [], tabs: [], activeIdx: null, scope: DEFAULT_SCOPE, selected: null };
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) return empty;
  const o = raw as Record<string, unknown>;
  const expanded = Array.isArray(o.expanded)
    ? o.expanded.filter((k): k is string => typeof k === "string")
    : [];
  const tabs: Tab[] = [];
  let previewSeen = false;
  for (const t of Array.isArray(o.tabs) ? o.tabs : []) {
    if (typeof t !== "object" || t === null) continue;
    const { root, rel, pinned } = t as Record<string, unknown>;
    if ((root !== "repos" && root !== "vault") || typeof rel !== "string" || rel === "") continue;
    if (tabs.some((x) => x.root === root && x.rel === rel)) continue;
    let p = pinned === true;
    if (!p) {
      if (previewSeen) p = true; // second preview → force-pin
      else previewSeen = true;
    }
    tabs.push({ root, rel, pinned: p });
  }
  const i = o.activeIdx;
  const activeIdx =
    typeof i === "number" && Number.isInteger(i) && i >= 0 && i < tabs.length
      ? i
      : tabs.length > 0
        ? 0
        : null;
  return { expanded, tabs, activeIdx, scope: normalizeScope(o.scope), selected: normalizeSelected(o.selected) };
}

// Zed model: single click targets the one preview slot (replaced in place);
// an already-open file is just activated.
export function openPreview(tabs: Tab[], file: { root: Root; rel: string }): { tabs: Tab[]; activeIdx: number } {
  const key = tabKey(file);
  const existing = tabs.findIndex((t) => tabKey(t) === key);
  if (existing >= 0) return { tabs, activeIdx: existing };
  const previewIdx = tabs.findIndex((t) => !t.pinned);
  if (previewIdx >= 0) {
    const next = tabs.slice();
    next[previewIdx] = { root: file.root, rel: file.rel, pinned: false };
    return { tabs: next, activeIdx: previewIdx };
  }
  return { tabs: [...tabs, { root: file.root, rel: file.rel, pinned: false }], activeIdx: tabs.length };
}

export function pinTab(tabs: Tab[], idx: number): Tab[] {
  const t = tabs[idx];
  if (!t || t.pinned) return tabs;
  const next = tabs.slice();
  next[idx] = { ...t, pinned: true };
  return next;
}

export function closeTab(tabs: Tab[], activeIdx: number | null, idx: number): { tabs: Tab[]; activeIdx: number | null } {
  const next = tabs.filter((_, i) => i !== idx);
  if (next.length === 0 || activeIdx === null) return { tabs: next, activeIdx: next.length ? Math.min(activeIdx ?? 0, next.length - 1) : null };
  let a = activeIdx;
  if (idx < a) a -= 1;
  else if (idx === a) a = Math.max(0, a - 1);
  return { tabs: next, activeIdx: Math.min(a, next.length - 1) };
}

// A renamed dir changes its rel (and its descendants'), so "root:rel" keys no
// longer match and the branch would collapse on refetch. Re-key self + descendants.
// (Moved from FileTree.tsx so the page-owned expanded state can persist.)
export function remapExpandedKeys(expanded: string[], root: Root, oldRel: string, newRel: string): string[] {
  const oldKey = `${root}:${oldRel}`;
  const prefix = `${oldKey}/`;
  return expanded.map((k) =>
    k === oldKey ? `${root}:${newRel}` : k.startsWith(prefix) ? `${root}:${newRel}/${k.slice(prefix.length)}` : k,
  );
}

// Recents (§6 localStorage, §8 cold review F10): a sibling key to STORAGE_KEY, holding up to
// RECENTS_CAP entries, most-recent-first; an already-present entry moves to the front instead of
// duplicating. Same defensive-normalize style as normalizeState — junk/corrupt storage yields [].
export const RECENTS_KEY = "jax-os.files.recents.v1";
export const RECENTS_CAP = 20;
export type RecentEntry = { root: Root; rel: string };

function parseRecentsGeneric<T>(raw: unknown, isValid: (item: unknown) => item is T, keyOf: (item: T) => string, cap: number): T[] {
  if (!Array.isArray(raw)) return [];
  const out: T[] = [];
  const seen = new Set<string>();
  for (const item of raw) {
    if (!isValid(item)) continue;
    const key = keyOf(item);
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(item);
  }
  return out.slice(0, cap);
}
function pushRecentGeneric<T>(list: T[], entry: T, keyOf: (item: T) => string, cap: number): T[] {
  const key = keyOf(entry);
  return [entry, ...list.filter((item) => keyOf(item) !== key)].slice(0, cap);
}
function isRecentEntry(item: unknown): item is RecentEntry {
  if (typeof item !== "object" || item === null) return false;
  const { root, rel } = item as Record<string, unknown>;
  return (root === "repos" || root === "vault") && typeof rel === "string" && rel !== "";
}
export function parseRecents(raw: unknown): RecentEntry[] {
  return parseRecentsGeneric(raw, isRecentEntry, (e) => `${e.root}:${e.rel}`, RECENTS_CAP);
}
export function pushRecent(list: RecentEntry[], entry: RecentEntry): RecentEntry[] {
  return pushRecentGeneric(list, entry, (e) => `${e.root}:${e.rel}`, RECENTS_CAP);
}

// §9: a sibling recents key for the ScopeChip's own recents row — same shape/cap/dedupe as file
// recents above, generalized into the two helpers those now share. Entries are serialized scope
// strings ("vault" / "repo:<name>"), never "all".
export const RECENT_REPOS_KEY = "jax-os.files.recentRepos.v1";
function isRecentRepoScope(item: unknown): item is string {
  return typeof item === "string" && item.length > 0;
}
export function parseRecentRepos(raw: unknown): string[] {
  return parseRecentsGeneric(raw, isRecentRepoScope, (s) => s, RECENTS_CAP);
}
export function pushRecentRepo(list: string[], entry: string): string[] {
  return pushRecentGeneric(list, entry, (s) => s, RECENTS_CAP);
}
