import { describe, expect, it, vi } from "vitest";
import type { Root, RepoRef, SearchScope } from "@/server/collectors/files";
import { createEdit, tabKey } from "./filesState";
import {
  affectedBusy,
  affectedPath,
  applyDelete,
  applyRename,
  classifyMutation,
  consumedLineRevealRequest,
  consumedRevealRequest,
  createFilesController,
  dirtyDescendantCount,
  dirtyKeys,
  displayedDir,
  displayedTarget,
  filesQueryAffected,
  isDirty,
  nextLineRevealRequest,
  nextRevealRequest,
  nextTreeSweep,
  openFromSearch,
  overlappingLock,
  relativeToScope,
  renameCollision,
  repoOf,
  resolveCurrentRepo,
  revealScopeSwitch,
  scopedRel,
  serializeScope,
  shouldConfirmLeave,
  shouldRevealLine,
  uiStorage,
} from "./filesWorkspace";

describe("isDirty / dirtyKeys", () => {
  it("detects unsaved text only", () => {
    const clean = createEdit("A", "ha");
    const dirty = { ...createEdit("B", "ha"), content: "C" };
    expect(isDirty({ a: clean })).toBe(false);
    expect(isDirty({ a: clean, b: dirty })).toBe(true);
    expect([...dirtyKeys({ a: clean, b: dirty })]).toEqual(["b"]);
  });
});

describe("shouldConfirmLeave", () => {
  it("prompts only when leaving Files while dirty", () => {
    expect(shouldConfirmLeave("/files", "/kanban", true)).toBe(true);
    expect(shouldConfirmLeave("/files", "/files", true)).toBe(false);
    expect(shouldConfirmLeave("/files", "/kanban", false)).toBe(false);
    expect(shouldConfirmLeave("/kanban", "/", true)).toBe(false);
  });
});

describe("uiStorage", () => {
  it("serializes only UI fields", () => {
    expect(uiStorage({
      expanded: new Set(["repos:a"]),
      tabs: [{ root: "repos", rel: "a.md", pinned: true }],
      activeIdx: 0,
      scope: { kind: "vault" },
      selected: null,
    })).toEqual({
      expanded: ["repos:a"],
      tabs: [{ root: "repos", rel: "a.md", pinned: true }],
      activeIdx: 0,
      scope: { kind: "vault" },
      selected: null,
    });
  });
});

describe("nextTreeSweep", () => {
  it("bumps the version from null and carries the swept root/rel (F2)", () => {
    const first = nextTreeSweep(null, "repos", "src");
    expect(first).toEqual({ version: 1, root: "repos", rel: "src" });
    expect(nextTreeSweep(first, "vault", "notes")).toEqual({ version: 2, root: "vault", rel: "notes" });
  });
});

describe("nextRevealRequest", () => {
  it("bumps the version from null and carries the requested root/rel/targetIsDir", () => {
    const first = nextRevealRequest(null, "repos", "src/a.ts", false);
    expect(first).toEqual({ version: 1, root: "repos", rel: "src/a.ts", targetIsDir: false });
    expect(nextRevealRequest(first, "vault", "notes", true)).toEqual({ version: 2, root: "vault", rel: "notes", targetIsDir: true });
  });
});

describe("consumedRevealRequest (one-shot channel, review F4)", () => {
  it("clears the request once its own version is consumed", () => {
    const req = nextRevealRequest(null, "repos", "src/a.ts", false);
    expect(consumedRevealRequest(req, req.version)).toBeNull();
  });

  it("never clears null", () => {
    expect(consumedRevealRequest(null, 1)).toBeNull();
  });

  it("keeps a newer request that raced ahead of a stale completion for an older version", () => {
    const first = nextRevealRequest(null, "repos", "src/a.ts", false);
    const second = nextRevealRequest(first, "vault", "notes", true);
    expect(consumedRevealRequest(second, first.version)).toBe(second);
  });
});

describe("repoOf (spec §4)", () => {
  it("a vault file maps to {kind:'vault'}; a repos file maps to {kind:'repo', name: first segment}", () => {
    expect(repoOf({ root: "vault", rel: "daily/a.md" })).toEqual({ kind: "vault" });
    expect(repoOf({ root: "repos", rel: "jax-os/src/a.ts" })).toEqual({ kind: "repo", name: "jax-os" });
    expect(repoOf({ root: "repos", rel: "jax-os" })).toEqual({ kind: "repo", name: "jax-os" });
  });
});

describe("resolveCurrentRepo (spec §4's four-priority table)", () => {
  it("priority 1 (active tab) wins over 2/3; priority 2 (selected) wins over 3; priority 3 (treeScope) wins over the default; nothing given defaults to {kind:'all'}", () => {
    const active = { root: "repos" as const, rel: "jax-os/a.ts" };
    expect(resolveCurrentRepo(active, { kind: "vault" }, { kind: "vault" })).toEqual({ kind: "repo", name: "jax-os" });
    const selected: RepoRef = { kind: "repo", name: "other" };
    expect(resolveCurrentRepo(null, selected, { kind: "vault" })).toEqual(selected);
    const treeScope: RepoRef = { kind: "vault" };
    expect(resolveCurrentRepo(null, null, treeScope)).toEqual(treeScope);
    expect(resolveCurrentRepo(null, null, null)).toEqual({ kind: "all" }); // this phase's own call sites (Task 9/10) always pass null/null
    expect(resolveCurrentRepo(null, null, { kind: "all" })).toEqual({ kind: "all" });
  });
});

describe("serializeScope", () => {
  it("round-trips the same three forms parseScope (files.ts) parses", () => {
    expect(serializeScope({ kind: "all" })).toBe("all");
    expect(serializeScope({ kind: "vault" })).toBe("vault");
    expect(serializeScope({ kind: "repo", name: "jax-os" })).toBe("repo:jax-os");
  });
});

describe("displayedDir / displayedTarget (spec §7, round-1 F1)", () => {
  it("displayedDir: null -> ''; a folder selection -> its own rel; a file selection -> its parent rel", () => {
    expect(displayedDir(null)).toBe("");
    expect(displayedDir({ rel: "jax-os/src", isDir: true })).toBe("jax-os/src");
    expect(displayedDir({ rel: "jax-os/src/a.ts", isDir: false })).toBe("jax-os/src");
  });
  it("displayedTarget: vault's null targets the vault root, a selection as-is; repo's null targets the repo's OWN root (never the global 'all repos' root), a nested selection as-is, never re-joined with the name a second time", () => {
    expect(displayedTarget({ kind: "vault" }, null)).toEqual({ root: "vault", rel: "" });
    expect(displayedTarget({ kind: "vault" }, { rel: "daily", isDir: true })).toEqual({ root: "vault", rel: "daily" });
    const scope = { kind: "repo" as const, name: "jax-os" };
    expect(displayedTarget(scope, null)).toEqual({ root: "repos", rel: "jax-os" });
    expect(displayedTarget(scope, { rel: "jax-os/src", isDir: true })).toEqual({ root: "repos", rel: "jax-os/src" });
    expect(displayedTarget(scope, { rel: "jax-os/src/a.ts", isDir: false })).toEqual({ root: "repos", rel: "jax-os/src" });
  });
});

describe("relativeToScope / scopedRel (round-trip helpers for the tree-panel and phone breadcrumbs)", () => {
  it("relativeToScope strips a repo's own name prefix, '' for the repo root itself; a vault rel passes through unchanged", () => {
    const scope = { kind: "repo" as const, name: "jax-os" };
    expect(relativeToScope(scope, "jax-os/src/a.ts")).toBe("src/a.ts");
    expect(relativeToScope(scope, "jax-os")).toBe("");
    expect(relativeToScope({ kind: "vault" }, "daily/a.md")).toBe("daily/a.md");
  });
  it("scopedRel is relativeToScope's inverse", () => {
    const scope = { kind: "repo" as const, name: "jax-os" };
    expect(scopedRel(scope, "src/a.ts")).toBe("jax-os/src/a.ts");
    expect(scopedRel(scope, "")).toBe("jax-os");
    expect(scopedRel({ kind: "vault" }, "daily/a.md")).toBe("daily/a.md");
    expect(scopedRel(scope, relativeToScope(scope, "jax-os/src/a.ts"))).toBe("jax-os/src/a.ts");
  });
});

describe("applyRename / applyDelete — selected reconciliation (round-1 F2, spec §5)", () => {
  const base = { tabs: [], activeIdx: null, edits: {}, expanded: [] };

  it("applyRename remaps a selected path that IS the renamed entry, and one that is a descendant of a renamed ancestor; leaves an unrelated or null selection untouched", () => {
    const s1 = { ...base, selected: { root: "repos" as const, rel: "dir/a.ts", isDir: false } };
    expect(applyRename(s1, "repos", "dir/a.ts", "dir/b.ts").selected).toEqual({ root: "repos", rel: "dir/b.ts", isDir: false });
    const s2 = { ...base, selected: { root: "repos" as const, rel: "dir/sub/a.ts", isDir: false } };
    expect(applyRename(s2, "repos", "dir", "renamed").selected).toEqual({ root: "repos", rel: "renamed/sub/a.ts", isDir: false });
    const s3 = { ...base, selected: { root: "repos" as const, rel: "other.ts", isDir: false } };
    expect(applyRename(s3, "repos", "dir/a.ts", "dir/b.ts").selected).toEqual(s3.selected);
    expect(applyRename({ ...base, selected: null }, "repos", "dir/a.ts", "dir/b.ts").selected).toBeNull();
  });

  it("applyDelete moves a selected path (or a selected descendant) to the nearest surviving parent, or clears it when the deleted path was already at the scope root; leaves an unrelated selection untouched", () => {
    const nested = { ...base, selected: { root: "repos" as const, rel: "dir/sub/a.ts", isDir: false } };
    expect(applyDelete(nested, "repos", "dir/sub").selected).toEqual({ root: "repos", rel: "dir", isDir: true });
    expect(applyDelete(nested, "repos", "dir/sub/a.ts").selected).toEqual({ root: "repos", rel: "dir/sub", isDir: true });
    const atRoot = { ...base, selected: { root: "repos" as const, rel: "top", isDir: true } };
    expect(applyDelete(atRoot, "repos", "top").selected).toBeNull();
    const unrelated = { ...base, selected: { root: "repos" as const, rel: "elsewhere.ts", isDir: false } };
    expect(applyDelete(unrelated, "repos", "dir/sub").selected).toEqual(unrelated.selected);
  });
});

describe("uiStorage — carries scope/selected through (spec §6)", () => {
  it("round-trips scope and selected unchanged", () => {
    const scope = { kind: "repo" as const, name: "jax-os" };
    const selected = { root: "repos" as const, rel: "jax-os/a.ts", isDir: false };
    expect(uiStorage({ expanded: new Set(["repos:jax-os"]), tabs: [], activeIdx: null, scope, selected })).toEqual({
      expanded: ["repos:jax-os"], tabs: [], activeIdx: null, scope, selected,
    });
  });
});

describe("revealScopeSwitch (round-1 F3 — requestReveal's auto-switch-scope decision)", () => {
  it("switches to a DIFFERENT repo target's own scope; no-ops when the target is already in scope", () => {
    const current = { kind: "repo" as const, name: "jax-os" };
    expect(revealScopeSwitch("repos", "other-repo/a.ts", current)).toEqual({ kind: "repo", name: "other-repo" });
    expect(revealScopeSwitch("repos", "jax-os/src/a.ts", current)).toBeNull();
    expect(revealScopeSwitch("repos", "other-repo/a.ts", { kind: "all" })).toEqual({ kind: "repo", name: "other-repo" }); // a target always switches you away from "all"
  });
  it("a VAULT ROOT reveal (rel === '', root === 'vault') switches scope to vault even from a repo scope — the vault has no name prefix, so rel === '' is its normal root target, not a no-op signal", () => {
    expect(revealScopeSwitch("vault", "", { kind: "repo", name: "jax-os" })).toEqual({ kind: "vault" });
    expect(revealScopeSwitch("vault", "", { kind: "vault" })).toBeNull();
  });
  it("root==='repos' with rel==='' never legitimately occurs (no caller produces it — a repo's own root reveal always carries rel===scope.name) — treated as a no-op rather than repoOf's degenerate {kind:'repo',name:''}", () => {
    expect(revealScopeSwitch("repos", "", { kind: "vault" })).toBeNull();
  });
});

describe("openFromSearch (round-2 F1 MEDIUM — every search/Quick Open open reveals+switches scope THEN opens, per the plan's live acceptance)", () => {
  it("calls reveal before open; reveal's targetIsDir is fixed false, open's line is passed through as given", () => {
    const calls: string[] = [];
    const reveal = vi.fn((root: string, rel: string, targetIsDir: boolean) => calls.push(`reveal:${root}:${rel}:${targetIsDir}`));
    const open = vi.fn((root: string, rel: string, line?: number) => calls.push(`open:${root}:${rel}:${line}`));
    openFromSearch(reveal, open, "repos", "other-repo/a.ts", 12);
    expect(calls).toEqual(["reveal:repos:other-repo/a.ts:false", "open:repos:other-repo/a.ts:12"]);
  });

  it("passes an absent line through as undefined, unchanged", () => {
    const reveal = vi.fn();
    const open = vi.fn();
    openFromSearch(reveal, open, "vault", "daily/a.md");
    expect(reveal).toHaveBeenCalledWith("vault", "daily/a.md", false);
    expect(open).toHaveBeenCalledWith("vault", "daily/a.md", undefined);
  });
});

describe("nextLineRevealRequest / consumedLineRevealRequest (one-shot, mirrors nextRevealRequest/consumedRevealRequest)", () => {
  it("bumps version on every request (even a repeat for the same line); consumed clears only a matching version, never a newer one that raced ahead", () => {
    const first = nextLineRevealRequest(null, "repos", "a.ts", 10);
    const second = nextLineRevealRequest(first, "repos", "a.ts", 10);
    expect([first.version, second.version]).toEqual([1, 2]);
    expect(consumedLineRevealRequest(first, first.version)).toBeNull();
    const newer = nextLineRevealRequest(second, "repos", "b.ts", 20);
    expect(consumedLineRevealRequest(newer, first.version)).toBe(newer);
  });
});

describe("shouldRevealLine (round-1 F1 — the editor consumer's pure gate)", () => {
  it("true only when a pending request's file matches the active tab; false for no request, no active tab, or a different file", () => {
    const request = nextLineRevealRequest(null, "repos", "a.ts", 10);
    expect(shouldRevealLine(request, tabKey({ root: "repos", rel: "a.ts" }))).toBe(true);
    expect(shouldRevealLine(request, tabKey({ root: "repos", rel: "b.ts" }))).toBe(false);
    expect(shouldRevealLine(null, tabKey({ root: "repos", rel: "a.ts" }))).toBe(false);
    expect(shouldRevealLine(request, null)).toBe(false);
  });
});

describe("affectedPath", () => {
  it("matches root and segment boundaries", () => {
    expect(affectedPath({ root: "repos", rel: "dir/a" }, "repos", "dir")).toBe(true);
    expect(affectedPath({ root: "repos", rel: "dir" }, "repos", "dir")).toBe(true);
    expect(affectedPath({ root: "repos", rel: "directory/a" }, "repos", "dir")).toBe(false);
    expect(affectedPath({ root: "vault", rel: "dir/a" }, "repos", "dir")).toBe(false);
    expect(affectedPath({ root: "repos", rel: "a.ts" }, "repos", "")).toBe(true);
    expect(affectedPath({ root: "vault", rel: "a.ts" }, "repos", "")).toBe(false);
  });
});

describe("applyRename / applyDelete", () => {
  const dirTab = { root: "repos" as const, rel: "dir/a.ts", pinned: true };
  const otherRoot = { root: "vault" as const, rel: "dir/a.ts", pinned: true };
  const near = { root: "repos" as const, rel: "directory/x.ts", pinned: false };
  const sibling = { root: "repos" as const, rel: "keep.ts", pinned: true };

  it("renames descendants, preserves pin/dirty text, and issues a new identity", () => {
    const old = createEdit("draft", "ha");
    old.content = "dirty";
    const state = applyRename({
      tabs: [dirTab, otherRoot, near],
      activeIdx: 0,
      edits: { "repos:dir/a.ts": old, "vault:dir/a.ts": createEdit("v", "hv") },
      expanded: ["repos:dir", "repos:dir/sub", "repos:directory", "vault:dir"],
      selected: null,
    }, "repos", "dir", "moved");
    expect(state.tabs[0]).toEqual({ root: "repos", rel: "moved/a.ts", pinned: true });
    expect(state.tabs[1]).toEqual(otherRoot);
    expect(state.tabs[2]).toEqual(near);
    expect(state.activeIdx).toBe(0);
    expect(state.edits["repos:moved/a.ts"].content).toBe("dirty");
    expect(state.edits["repos:moved/a.ts"].id).not.toBe(old.id);
    expect(state.edits["repos:dir/a.ts"]).toBeUndefined();
    expect(state.edits["vault:dir/a.ts"].content).toBe("v");
    expect(state.expanded).toEqual(["repos:moved", "repos:moved/sub", "repos:directory", "vault:dir"]);
  });

  it("blocks renaming onto an unrelated open destination tab", () => {
    expect(renameCollision([dirTab, sibling], "repos", "dir/a.ts", "keep.ts")).toBe(true);
    expect(renameCollision([dirTab], "repos", "dir/a.ts", "keep.ts")).toBe(false);
  });

  it("blocks a directory rename whose remapped descendant hits an unrelated open tab or draft", () => {
    const src = { root: "repos" as const, rel: "src/a.ts", pinned: true };
    const dest = { root: "repos" as const, rel: "dest/a.ts", pinned: true };
    expect(renameCollision([src, dest], "repos", "src", "dest")).toBe(true);
    expect(renameCollision(
      [src],
      "repos",
      "src",
      "dest",
      { "repos:dest/a.ts": createEdit("other", "hx") },
    )).toBe(true);
    expect(renameCollision([src], "repos", "src", "dest")).toBe(false);
  });

  it("deletes affected tabs and selects the nearest remaining tab", () => {
    const state = applyDelete({
      tabs: [sibling, dirTab, near],
      activeIdx: 1,
      edits: { "repos:dir/a.ts": createEdit("x", "h"), "repos:keep.ts": createEdit("k", "hk") },
      expanded: ["repos:dir", "repos:keep.ts"],
      selected: null,
    }, "repos", "dir");
    expect(state.tabs.map((t) => t.rel)).toEqual(["keep.ts", "directory/x.ts"]);
    expect(state.activeIdx).toBe(0);
    expect(state.edits["repos:dir/a.ts"]).toBeUndefined();
    expect(state.edits["repos:keep.ts"]).toBeDefined();
    expect(state.expanded).toEqual(["repos:keep.ts"]);
  });
});

describe("locks and mutation classification", () => {
  it("treats overlapping paths as busy/locked and ignores the other root", () => {
    const edits = {
      "repos:dir/a.ts": { ...createEdit("A", "ha"), saving: true },
      "vault:dir/a.ts": createEdit("B", "hb"),
    };
    expect(affectedBusy(edits, "repos", "dir")).toBe(true);
    expect(affectedBusy(edits, "vault", "dir")).toBe(false);
    expect(overlappingLock([{ root: "repos", rel: "dir" }], "repos", "dir/a.ts")).toBe(true);
    expect(overlappingLock([{ root: "repos", rel: "dir/a.ts" }], "repos", "dir")).toBe(true);
    expect(overlappingLock([{ root: "repos", rel: "dir" }], "repos", "directory")).toBe(false);
    expect(overlappingLock([{ root: "repos", rel: "" }], "repos", "a.ts")).toBe(true);
    expect(overlappingLock([{ root: "repos", rel: "" }], "vault", "a.ts")).toBe(false);
  });

  it("classifies applied-unrecorded, unconfirmed, and refused outcomes", () => {
    expect(classifyMutation({ ok: true })).toEqual({ kind: "ok" });
    expect(classifyMutation({
      ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending", error: "x",
    })).toEqual({ kind: "applied-unrecorded" });
    expect(classifyMutation({
      ok: false, code: "mutation-unconfirmed", effect: "unconfirmed", audit: "recorded", error: "x",
    })).toEqual({ kind: "unconfirmed" });
    expect(classifyMutation({
      ok: false, code: "mutation-rejected", effect: "not-applied", audit: "recorded", error: "directory not empty",
    })).toEqual({ kind: "refused", error: "directory not empty" });
    expect(classifyMutation({ ok: false, error: "TypeError: Failed to fetch" })).toEqual({ kind: "unconfirmed" });
    expect(classifyMutation(undefined)).toEqual({ kind: "unconfirmed" });
    expect(classifyMutation({ ok: false })).toEqual({ kind: "unconfirmed" });
  });

  it("counts dirty descendants for delete confirmation", () => {
    const edits = {
      "repos:dir/a.ts": { ...createEdit("A", "ha"), content: "d" },
      "repos:dir/b.ts": createEdit("B", "hb"),
      "repos:directory/c.ts": { ...createEdit("C", "hc"), content: "d" },
    };
    expect(dirtyDescendantCount(edits, "repos", "dir")).toBe(1);
  });
});

describe("createFilesController production path", () => {
  function deferred<T>() {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((yes) => { resolve = yes; });
    return { promise, resolve };
  }

  function make(fetchJson = vi.fn(), fetchRaw = vi.fn(), recordRecent = vi.fn(), revealLine = vi.fn(), recordRecentRepo = vi.fn()) {
    const saveBlob = vi.fn();
    const sweep = vi.fn();
    const ctl = createFilesController({
      getDeps: () => ({
        t: (key) => key,
        confirm: () => true,
        fetchJson,
        fetchRaw,
        saveBlob,
        sweep,
        recordRecent,
        revealLine,
        recordRecentRepo,
      }),
    });
    return { ctl, fetchJson, fetchRaw, saveBlob, sweep, recordRecent, revealLine, recordRecentRepo };
  }

  it("reserves synchronously so a second same-turn rename does not fetch", () => {
    const pending = deferred<unknown>();
    const fetchJson = vi.fn(() => pending.promise);
    const { ctl } = make(fetchJson);
    ctl.openFile("repos", "src");
    ctl.beginRename("repos", "src", "", "src");
    ctl.commitInline("dest");
    ctl.beginRename("repos", "src", "", "src");
    ctl.commitInline("dest");
    expect(fetchJson).toHaveBeenCalledOnce();
    pending.resolve({ ok: true });
  });

  it("sees an immediate edit in getEdit before any rerender", () => {
    const { ctl } = make();
    ctl.onSeed("repos:a.ts", "A", "ha");
    const id = ctl.getEdit("repos:a.ts")!.id;
    ctl.onEdit("repos:a.ts", id, { content: "B", justSaved: false });
    expect(ctl.getEdit("repos:a.ts")).toMatchObject({ content: "B", revision: 1 });
  });

  it("refuses rename when a same-turn save is already pending", () => {
    const fetchJson = vi.fn();
    const { ctl } = make(fetchJson);
    ctl.onSeed("repos:a.ts", "A", "ha");
    const id = ctl.getEdit("repos:a.ts")!.id;
    ctl.onEdit("repos:a.ts", id, { saving: true });
    ctl.beginRename("repos", "a.ts", "", "a.ts");
    ctl.commitInline("dest");
    expect(fetchJson).not.toHaveBeenCalled();
    expect(ctl.notes().some((note) => note.text === "waitPending")).toBe(true);
  });

  it("keeps an unconfirmed note after an unrelated confirmed create", async () => {
    const fetchJson = vi.fn()
      .mockResolvedValueOnce({ ok: false, error: "offline" })
      .mockResolvedValueOnce({ ok: true });
    const { ctl } = make(fetchJson);
    ctl.beginRename("repos", "a.ts", "", "a.ts");
    ctl.commitInline("dest");
    await vi.waitFor(() => {
      expect(ctl.notes().some((note) => note.sticky && note.text === "checkCurrentState")).toBe(true);
    });
    ctl.beginCreate("repos", "", "file");
    ctl.commitInline("newfile.txt");
    await vi.waitFor(() => {
      expect(ctl.notes().some((note) => note.text === "opConfirmed")).toBe(true);
    });
    expect(ctl.notes().some((note) => note.sticky && note.text === "checkCurrentState")).toBe(true);
  });

  it.each([
    [{ ok: true }, "ok", "opConfirmed"],
    [{ ok: false, effect: "not-applied", error: "denied" }, "warning", "uploadFailedSome"],
    [{ ok: false, effect: "unconfirmed" }, "warning", "uploadFailedSome"],
  ])("keeps one upload note id from pending through %j", async (res, tone, text) => {
    const pending = deferred<unknown>();
    const fetchJson = vi.fn(() => pending.promise);
    const { ctl } = make(fetchJson);
    ctl.uploadMany("repos", "dir", [new File(["x"], "a.txt")]);
    expect(ctl.notes()).toEqual([
      expect.objectContaining({ id: "repos:dir/a.txt", tone: "pending", text: "opPending" }),
    ]);
    pending.resolve(res);
    await vi.waitFor(() => {
      expect(ctl.notes().some((note) => note.tone === "pending")).toBe(false);
    });
    expect(ctl.notes()).toEqual([
      expect.objectContaining({ id: "repos:dir/a.txt", tone, text }),
    ]);
  });

  it("uses the parent as the upload note id when the file list is empty", async () => {
    const { ctl } = make(vi.fn());
    ctl.uploadMany("repos", "dir", []);
    await vi.waitFor(() => {
      expect(ctl.notes()).toEqual([
        expect.objectContaining({ id: "repos:dir", tone: "ok", text: "opConfirmed" }),
      ]);
    });
  });

  it("blocks opening and seeding a reserved path including files without EditState", () => {
    const pending = deferred<unknown>();
    const fetchJson = vi.fn(() => pending.promise);
    const { ctl } = make(fetchJson);
    ctl.beginRename("repos", "shot.png", "", "shot.png");
    ctl.commitInline("dest.png");
    ctl.openFile("repos", "shot.png");
    ctl.onSeed("repos:shot.png", "", "h");
    expect(ctl.snapshot().tabs).toEqual([]);
    expect(ctl.getEdit("repos:shot.png")).toBeUndefined();
    expect(ctl.isLocked("repos", "shot.png")).toBe(true);
    pending.resolve({ ok: true });
  });

  it("commitInline never fetches for an empty/whitespace name, a same-name rename, or no active inline op", () => {
    const fetchJson = vi.fn();
    const { ctl } = make(fetchJson);
    ctl.commitInline("anything"); // no beginCreate/beginRename first
    ctl.beginCreate("repos", "", "file");
    ctl.commitInline("   ");
    expect(ctl.inline()).toBeNull();
    ctl.beginRename("repos", "a.ts", "", "a.ts");
    ctl.commitInline("");
    ctl.beginRename("repos", "a.ts", "", "a.ts");
    ctl.commitInline("a.ts");
    expect(fetchJson).not.toHaveBeenCalled();
  });

  it("beginCreate expands the target directory even if it was collapsed (round 1 F2)", () => {
    const { ctl } = make();
    expect(ctl.snapshot().expanded.has("repos:docs")).toBe(false);
    ctl.beginCreate("repos", "docs", "file");
    expect(ctl.snapshot().expanded.has("repos:docs")).toBe(true);
    expect(ctl.inline()).toEqual({ kind: "create", root: "repos", dirRel: "docs", fileKind: "file" });
  });

  it("remove() no longer confirms, captures trashRel for an undo toast, and warns (non-blocking) on dirty descendants", async () => {
    const fetchJson = vi.fn().mockResolvedValueOnce({ ok: true, trashRel: ".jax-trash/T/a.ts" });
    const { ctl } = make(fetchJson);
    ctl.remove("repos", "a.ts", "", "a.ts");
    await vi.waitFor(() => {
      expect(ctl.notes().some((note) => note.undo?.trashRel === ".jax-trash/T/a.ts")).toBe(true);
    });
    expect(fetchJson).toHaveBeenCalledWith("/api/files/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root: "repos", rel: "a.ts" }),
    }); // round 1 F5: exact request, not just the method

    const hung = vi.fn(() => new Promise(() => {})); // never resolves — assert only the pre-fetch note
    const { ctl: ctl2 } = make(hung);
    ctl2.onSeed("repos:dir/a.ts", "A", "ha");
    const id = ctl2.getEdit("repos:dir/a.ts")!.id;
    ctl2.onEdit("repos:dir/a.ts", id, { content: "dirty" });
    ctl2.remove("repos", "dir", "", "dir");
    expect(ctl2.notes().some((note) => note.tone === "warning" && note.id === "repos:dir")).toBe(true);
    expect(hung).toHaveBeenCalledOnce();
  });

  it("remove() treats an applied-unrecorded response as applied too (round 2 F2): entry leaves the tree, Undo note carries trashRel", async () => {
    const fetchJson = vi.fn().mockResolvedValueOnce({ ok: false, effect: "applied", trashRel: ".jax-trash/T/b.ts" });
    const { ctl } = make(fetchJson);
    ctl.onSeed("repos:b.ts", "B", "hb");
    ctl.remove("repos", "b.ts", "", "b.ts");
    await vi.waitFor(() => {
      expect(ctl.getEdit("repos:b.ts")).toBeUndefined();
    });
    expect(ctl.notes().some((note) => note.undo?.trashRel === ".jax-trash/T/b.ts")).toBe(true);
  });

  it("remove() applied-unrecorded without trashRel still confirms the delete without crashing", async () => {
    const fetchJson = vi.fn().mockResolvedValueOnce({ ok: false, effect: "applied" });
    const { ctl } = make(fetchJson);
    ctl.onSeed("repos:c.ts", "C", "hc");
    ctl.remove("repos", "c.ts", "", "c.ts");
    await vi.waitFor(() => {
      expect(ctl.getEdit("repos:c.ts")).toBeUndefined();
    });
    expect(ctl.notes().some((note) => note.id === "repos:c.ts" && note.tone === "ok")).toBe(true);
  });

  it("undoTrash treats an applied-unrecorded restore as applied too: sweeps restoredRel and its parent", async () => {
    const fetchJson = vi.fn().mockResolvedValueOnce({ ok: false, effect: "applied", restoredRel: "dir/b.ts" });
    const { ctl, sweep } = make(fetchJson);
    ctl.undoTrash("repos:dir/b.ts", "repos", ".jax-trash/T/dir/b.ts");
    await vi.waitFor(() => {
      expect(ctl.notes()).toEqual([expect.objectContaining({ id: "repos:dir/b.ts", tone: "ok", text: "undoRestored" })]);
    });
    expect(sweep).toHaveBeenCalledExactlyOnceWith("repos", "dir/b.ts", "dir");
  });

  it("undoTrash applied-unrecorded without restoredRel still shows the ok note without crashing", async () => {
    const fetchJson = vi.fn().mockResolvedValueOnce({ ok: false, effect: "applied" });
    const { ctl, sweep } = make(fetchJson);
    ctl.undoTrash("repos:c.ts", "repos", ".jax-trash/T/c.ts");
    await vi.waitFor(() => {
      expect(ctl.notes()).toEqual([expect.objectContaining({ id: "repos:c.ts", tone: "ok", text: "undoRestored" })]);
    });
    expect(sweep).not.toHaveBeenCalled();
  });

  it("undoTrash posts root+trashRel to /trash/restore, sweeps the restored path, and marks the note ok or failed", async () => {
    const fetchJson = vi.fn()
      .mockResolvedValueOnce({ ok: true, restoredRel: "dir/a.ts" })
      .mockResolvedValueOnce({ ok: false, effect: "not-applied", error: "already exists" });
    const { ctl, sweep } = make(fetchJson);
    ctl.undoTrash("repos:dir/a.ts", "repos", ".jax-trash/T/dir/a.ts");
    await vi.waitFor(() => {
      expect(ctl.notes()).toEqual([expect.objectContaining({ id: "repos:dir/a.ts", tone: "ok", text: "undoRestored" })]);
    });
    expect(fetchJson).toHaveBeenCalledWith("/api/files/trash/restore", expect.objectContaining({
      method: "POST", body: JSON.stringify({ root: "repos", trashRel: ".jax-trash/T/dir/a.ts" }),
    }));
    // round 1 F3: undoTrash must sweep the restored path's parent (using the response's restoredRel,
    // not the request's trashRel) so the tree actually shows the restored entry.
    expect(sweep).toHaveBeenCalledExactlyOnceWith("repos", "dir/a.ts", "dir");
    ctl.undoTrash("repos:b.ts", "repos", ".jax-trash/T/b.ts");
    await vi.waitFor(() => {
      expect(ctl.notes()).toContainEqual(expect.objectContaining({ id: "repos:b.ts", tone: "error", text: "undoFailed" }));
    });
    expect(sweep).toHaveBeenCalledOnce(); // unchanged — a failed restore has no restoredRel to sweep
  });

  it("downloadZip saves the blob and toasts excluded secrets, or errors cleanly on zip-too-large", async () => {
    const blob = new Blob(["x"]);
    const zipRes = {
      headers: new Headers({ "content-type": "application/zip", "X-Secrets-Excluded": "2" }),
      blob: () => Promise.resolve(blob),
    } as unknown as Response;
    const tooLargeRes = {
      headers: new Headers({ "content-type": "application/json" }),
      json: () => Promise.resolve({ ok: false, error: "zip-too-large" }),
    } as unknown as Response;
    const fetchRaw = vi.fn().mockResolvedValueOnce(zipRes).mockResolvedValueOnce(tooLargeRes);
    const { ctl, saveBlob } = make(vi.fn(), fetchRaw);
    await ctl.downloadZip("repos", "docs", "docs");
    expect(fetchRaw).toHaveBeenCalledWith("/api/files/zip?root=repos&rel=docs");
    // round 1 F4: assert the actual download, not just the note text.
    expect(saveBlob).toHaveBeenCalledExactlyOnceWith(blob, "docs.zip");
    expect(ctl.notes()).toContainEqual(expect.objectContaining({ id: "repos:docs", tone: "warning", text: "secretsExcluded" }));
    await ctl.downloadZip("repos", "big", "big");
    expect(ctl.notes()).toContainEqual(expect.objectContaining({ id: "repos:big", tone: "error", text: "zipTooLarge" }));
    expect(saveBlob).toHaveBeenCalledOnce(); // unchanged — never called for the zip-too-large failure
  });

  it("records a recent on openFile/openFilePinned, but not for an open blocked by a lock", () => {
    const { ctl, recordRecent } = make();
    ctl.openFile("repos", "a.ts");
    expect(recordRecent).toHaveBeenCalledWith("repos", "a.ts");
    ctl.openFilePinned("repos", "b.ts");
    expect(recordRecent).toHaveBeenCalledWith("repos", "b.ts");

    const pending = deferred<unknown>();
    const { ctl: ctl2, recordRecent: recordRecent2 } = make(vi.fn(() => pending.promise));
    ctl2.beginRename("repos", "locked.ts", "", "locked.ts");
    ctl2.commitInline("dest.ts"); // reserves "locked.ts" synchronously, before the fetch resolves
    ctl2.openFile("repos", "locked.ts");
    expect(recordRecent2).not.toHaveBeenCalled();
    pending.resolve({ ok: true });
  });

  it("calls revealLine only when a line is given, for both openFile and openFilePinned", () => {
    const { ctl, revealLine } = make();
    ctl.openFile("repos", "a.ts");
    expect(revealLine).not.toHaveBeenCalled();
    ctl.openFile("repos", "a.ts", 42);
    expect(revealLine).toHaveBeenCalledWith("repos", "a.ts", 42);
    ctl.openFilePinned("repos", "b.ts", 7);
    expect(revealLine).toHaveBeenCalledWith("repos", "b.ts", 7);
  });

  describe("createFilesController — scope/selected (spec §5)", () => {
    it("starts with the default scope and no selection; setSelected sets the given entry directly; openFile/openFilePinned also select the opened file and record its repo as recent", () => {
      const { ctl, recordRecentRepo } = make();
      expect(ctl.snapshot().scope).toEqual({ kind: "repo", name: "jax-os" });
      expect(ctl.snapshot().selected).toBeNull();
      ctl.setSelected("repos", "jax-os/src", true);
      expect(ctl.snapshot().selected).toEqual({ root: "repos", rel: "jax-os/src", isDir: true });
      ctl.openFile("repos", "jax-os/a.ts");
      expect(ctl.snapshot().selected).toEqual({ root: "repos", rel: "jax-os/a.ts", isDir: false });
      expect(recordRecentRepo).toHaveBeenCalledWith({ kind: "repo", name: "jax-os" });
      ctl.openFilePinned("vault", "daily/a.md");
      expect(ctl.snapshot().selected).toEqual({ root: "vault", rel: "daily/a.md", isDir: false });
      expect(recordRecentRepo).toHaveBeenCalledWith({ kind: "vault" });
    });

    it("setScope switches scope, clears the current selection, and records the new scope as recent", () => {
      const { ctl, recordRecentRepo } = make();
      ctl.setSelected("repos", "jax-os/src", true);
      ctl.setScope({ kind: "vault" });
      expect(ctl.snapshot().scope).toEqual({ kind: "vault" });
      expect(ctl.snapshot().selected).toBeNull();
      expect(recordRecentRepo).toHaveBeenCalledWith({ kind: "vault" });
    });

    it("goBack walks folder -> nested folder -> open a file -> back to repo root -> back to Files root "
      + "-> a further back is a true no-op; a fresh vault scope with no selection also reaches the Files "
      + "root in one step (spec §4.5)", () => {
      const { ctl } = make();
      ctl.setSelected("repos", "jax-os/src", true);
      ctl.setSelected("repos", "jax-os/src/lib", true);
      ctl.openFile("repos", "jax-os/src/lib/a.ts");
      ctl.goBack();
      expect(ctl.snapshot().selected).toEqual({ root: "repos", rel: "jax-os/src", isDir: true });
      ctl.goBack();
      expect(ctl.snapshot().selected).toEqual({ root: "repos", rel: "jax-os", isDir: true });
      ctl.goBack(); // was a no-op here; now reaches the Files root
      expect(ctl.snapshot().scope).toEqual({ kind: "all" });
      expect(ctl.snapshot().selected).toBeNull();
      ctl.goBack(); // true no-op at the true top
      expect(ctl.snapshot().scope).toEqual({ kind: "all" });
      ctl.setScope({ kind: "vault" });
      ctl.goBack();
      expect(ctl.snapshot().scope).toEqual({ kind: "all" });
      expect(ctl.snapshot().selected).toBeNull();
    });
  });

  describe("close/dropReplacedPreview retire the affected Monaco model (spec §4 item 1)", () => {
    it("close(idx) on a dirty-free tab retires that tab's Monaco path", () => {
      const { ctl } = make();
      ctl.openFilePinned("repos", "a.ts");
      ctl.openFilePinned("repos", "b.ts");
      ctl.close(0);
      expect(ctl.retired()).toEqual(["repos/a.ts"]);
    });

    it("replacing the preview tab (via openFile) retires the dropped preview's Monaco path", () => {
      const { ctl } = make();
      ctl.openFile("repos", "a.ts"); // preview
      ctl.openFile("repos", "b.ts"); // replaces the preview slot -> drops a.ts
      expect(ctl.retired()).toEqual(["repos/a.ts"]);
    });

    it("a later close/dropReplacedPreview overwrites retired, never accumulates (matches rename/delete)", () => {
      const { ctl } = make();
      ctl.openFilePinned("repos", "a.ts");
      ctl.openFilePinned("repos", "b.ts");
      ctl.close(0);
      expect(ctl.retired()).toEqual(["repos/a.ts"]);
      ctl.close(0); // b.ts is now at index 0
      expect(ctl.retired()).toEqual(["repos/b.ts"]);
    });
  });

  describe("treeSnapshot identity across an edit-only vs. a tree-shaped change (spec §3)", () => {
    it("stays the SAME object across onSeed/onEdit; a NEW object after a tree-shaped mutation", () => {
      const { ctl } = make();
      ctl.openFile("repos", "a.ts");
      ctl.onSeed("repos:a.ts", "hello", "hash1");
      const before = ctl.treeSnapshot();
      const id = ctl.getEdit("repos:a.ts")!.id;
      ctl.onEdit("repos:a.ts", id, { content: "hello!" });
      expect(ctl.treeSnapshot()).toBe(before);
      ctl.toggle("repos:src");
      expect(ctl.treeSnapshot()).not.toBe(before);
    });
  });
});

describe("files query predicates", () => {
  it("matches read/tree path boundaries and all search keys", () => {
    expect(filesQueryAffected(["files", "read", "repos", "dir/a.ts"], "repos", "dir")).toBe(true);
    expect(filesQueryAffected(["files", "tree", "repos", "directory"], "repos", "dir")).toBe(false);
    expect(filesQueryAffected(["files", "search", "q"], "repos", "dir")).toBe(true);
    expect(filesQueryAffected(["files", "read", "vault", "dir/a.ts"], "repos", "dir")).toBe(false);
  });
});
