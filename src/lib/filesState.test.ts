import { describe, expect, it } from "vitest";
import { closeTab, createEdit, disambiguateTabLabels, normalizeState, openPreview, parseRecentRepos, parseRecents, patchEdit, pinTab, pushRecent, pushRecentRepo, remapExpandedKeys, RECENT_REPOS_KEY, RECENTS_CAP, tabKey, type EditState, type RecentEntry, type Tab } from "./filesState";

const A: Tab = { root: "repos", rel: "a.md", pinned: true };
const B: Tab = { root: "repos", rel: "dir/b.ts", pinned: false };

describe("normalizeState", () => {
  it("returns clean state for junk input", () => {
    for (const junk of [null, undefined, 42, "x", [], { tabs: "no" }]) {
      expect(normalizeState(junk)).toEqual({ expanded: [], tabs: [], activeIdx: null, scope: { kind: "repo", name: "jax-os" }, selected: null });
    }
  });
  it("keeps valid entries, drops invalid ones, dedupes, enforces one preview", () => {
    const s = normalizeState({
      expanded: ["repos:x", 7, "vault:y"],
      tabs: [A, { root: "nope", rel: "z" }, B, { root: "repos", rel: "c.md", pinned: false }, A],
      activeIdx: 1,
    });
    expect(s.expanded).toEqual(["repos:x", "vault:y"]);
    expect(s.tabs).toEqual([A, B, { root: "repos", rel: "c.md", pinned: true }]); // 2nd preview force-pinned, dup A dropped
    expect(s.activeIdx).toBe(1);
  });
  it("clamps out-of-range activeIdx", () => {
    expect(normalizeState({ expanded: [], tabs: [A], activeIdx: 9 }).activeIdx).toBe(0);
    expect(normalizeState({ expanded: [], tabs: [], activeIdx: 0 }).activeIdx).toBeNull();
  });
});

describe("normalizeState — scope/selected (spec §6)", () => {
  it("defaults scope to {kind:'repo',name:'jax-os'} and selected to null when the field is absent or the wrong shape", () => {
    expect(normalizeState(null).scope).toEqual({ kind: "repo", name: "jax-os" });
    expect(normalizeState(null).selected).toBeNull();
    expect(normalizeState({ scope: "bogus", selected: "bogus" }).scope).toEqual({ kind: "repo", name: "jax-os" });
    expect(normalizeState({ scope: { kind: "repo" } }).scope).toEqual({ kind: "repo", name: "jax-os" }); // missing name
    expect(normalizeState({ scope: { kind: "all" } }).scope).toEqual({ kind: "all" }); // spec §4.1
  });

  it("accepts a valid persisted vault scope, a valid persisted repo scope, and a valid persisted selected entry", () => {
    const state = normalizeState({ scope: { kind: "vault" }, selected: { root: "vault", rel: "daily/a.md", isDir: false } });
    expect(state.scope).toEqual({ kind: "vault" });
    expect(state.selected).toEqual({ root: "vault", rel: "daily/a.md", isDir: false });
    expect(normalizeState({ scope: { kind: "repo", name: "other-repo" } }).scope).toEqual({ kind: "repo", name: "other-repo" });
  });

  it("drops a selected entry with a bad root, empty rel, or non-boolean isDir", () => {
    expect(normalizeState({ selected: { root: "bogus", rel: "a", isDir: true } }).selected).toBeNull();
    expect(normalizeState({ selected: { root: "repos", rel: "", isDir: true } }).selected).toBeNull();
    expect(normalizeState({ selected: { root: "repos", rel: "a", isDir: "yes" } }).selected).toBeNull();
    expect(normalizeState({ selected: null }).selected).toBeNull();
  });
});

describe("openPreview / pinTab", () => {
  it("reuses the single preview slot", () => {
    const r1 = openPreview([A, B], { root: "vault", rel: "n.md" });
    expect(r1.tabs).toEqual([A, { root: "vault", rel: "n.md", pinned: false }]);
    expect(r1.activeIdx).toBe(1);
  });
  it("appends a preview when none exists", () => {
    const r = openPreview([A], { root: "vault", rel: "n.md" });
    expect(r.tabs).toHaveLength(2);
    expect(r.activeIdx).toBe(1);
  });
  it("activates an already-open tab without changes", () => {
    const r = openPreview([A, B], { root: "repos", rel: "a.md" });
    expect(r.tabs).toEqual([A, B]);
    expect(r.activeIdx).toBe(0);
  });
  it("pinTab pins in place and is a no-op on pinned/missing", () => {
    expect(pinTab([B], 0)[0].pinned).toBe(true);
    const tabs = [A];
    expect(pinTab(tabs, 0)).toBe(tabs);
    expect(pinTab(tabs, 5)).toBe(tabs);
  });
});

describe("closeTab", () => {
  it("closing the active tab activates the left neighbor", () => {
    expect(closeTab([A, B], 1, 1)).toEqual({ tabs: [A], activeIdx: 0 });
  });
  it("closing before the active tab shifts the index", () => {
    expect(closeTab([A, B], 1, 0)).toEqual({ tabs: [B], activeIdx: 0 });
  });
  it("closing the last tab yields null", () => {
    expect(closeTab([A], 0, 0)).toEqual({ tabs: [], activeIdx: null });
  });
});

describe("remapExpandedKeys", () => {
  it("re-keys the renamed dir and its descendants only", () => {
    expect(remapExpandedKeys(["repos:old", "repos:old/sub", "repos:other", "vault:old"], "repos", "old", "new"))
      .toEqual(["repos:new", "repos:new/sub", "repos:other", "vault:old"]);
  });
});

describe("tabKey", () => {
  it("is root:rel", () => {
    expect(tabKey(A)).toBe("repos:a.md");
  });
});

describe("createEdit / patchEdit revision", () => {
  it("starts at 0 and increments on text then undo, not on status-only patches", () => {
    const created = createEdit("A", "ha");
    expect(created.revision).toBe(0);
    let edits: Record<string, EditState> = { x: created };
    edits = patchEdit(edits, "x", created.id, { content: "B" });
    expect(edits.x.revision).toBe(1);
    edits = patchEdit(edits, "x", created.id, { content: "A" });
    expect(edits.x.revision).toBe(2);
    edits = patchEdit(edits, "x", created.id, { saving: true, stale: true, justSaved: true });
    expect(edits.x.revision).toBe(2);
    expect(edits.x.content).toBe("A");
  });

  it("increments when saved baseline or hash changes", () => {
    const created = createEdit("A", "ha");
    let edits: Record<string, EditState> = { x: created };
    edits = patchEdit(edits, "x", created.id, { savedContent: "A", baseHash: "hb" });
    expect(edits.x.revision).toBe(1);
  });

  it("keeps identity and ignores a mismatched id", () => {
    const created = createEdit("A", "ha");
    const edits = patchEdit({ x: created }, "x", Symbol(), { content: "B" });
    expect(edits.x).toBe(created);
    expect(edits.x.id).toBe(created.id);
  });
});

describe("parseRecents", () => {
  it("returns [] for junk input", () => {
    for (const junk of [null, undefined, 42, "x", {}]) expect(parseRecents(junk)).toEqual([]);
  });

  it("keeps valid entries, drops invalid ones, caps at RECENTS_CAP", () => {
    const raw = [
      { root: "repos", rel: "a.ts" },
      { root: "nope", rel: "b.ts" },
      { root: "vault", rel: "" },
      { root: "vault", rel: "c.md" },
    ];
    expect(parseRecents(raw)).toEqual([{ root: "repos", rel: "a.ts" }, { root: "vault", rel: "c.md" }]);
    const long = Array.from({ length: 30 }, (_, i) => ({ root: "repos", rel: `f${i}.ts` }));
    expect(parseRecents(long)).toHaveLength(RECENTS_CAP);
  });

  it("dedupes duplicate root:rel entries before applying the cap (F3)", () => {
    const raw = [
      { root: "repos", rel: "a.ts" },
      { root: "repos", rel: "a.ts" },
      { root: "vault", rel: "c.md" },
    ];
    expect(parseRecents(raw)).toEqual([{ root: "repos", rel: "a.ts" }, { root: "vault", rel: "c.md" }]);
    // duplicates must not crowd out unique entries under the cap
    const dupHeavy = [
      ...Array.from({ length: RECENTS_CAP }, () => ({ root: "repos", rel: "dup.ts" })),
      { root: "vault", rel: "unique.md" },
    ];
    const result = parseRecents(dupHeavy);
    expect(result).toContainEqual({ root: "vault", rel: "unique.md" });
    expect(result).toHaveLength(2);
  });
});

describe("pushRecent", () => {
  it("moves an already-present entry to the front instead of duplicating", () => {
    const list: RecentEntry[] = [{ root: "repos", rel: "a.ts" }, { root: "repos", rel: "b.ts" }];
    expect(pushRecent(list, { root: "repos", rel: "b.ts" })).toEqual([
      { root: "repos", rel: "b.ts" },
      { root: "repos", rel: "a.ts" },
    ]);
  });

  it("prepends a new entry and caps at RECENTS_CAP, dropping the oldest", () => {
    const list = Array.from({ length: RECENTS_CAP }, (_, i) => ({ root: "repos" as const, rel: `f${i}.ts` }));
    const next = pushRecent(list, { root: "vault", rel: "new.md" });
    expect(next).toHaveLength(RECENTS_CAP);
    expect(next[0]).toEqual({ root: "vault", rel: "new.md" });
    expect(next.at(-1)).toEqual({ root: "repos", rel: `f${RECENTS_CAP - 2}.ts` });
  });
});

describe("disambiguateTabLabels (spec §4 item 4)", () => {
  it("returns the bare basename when no other open tab shares it", () => {
    const tabs: Tab[] = [
      { root: "repos", rel: "a/x.ts", pinned: true },
      { root: "repos", rel: "b/y.ts", pinned: true },
    ];
    expect(disambiguateTabLabels(tabs)).toEqual(["x.ts", "y.ts"]);
  });

  it("qualifies with the parent folder when the basename collides", () => {
    const tabs: Tab[] = [
      { root: "repos", rel: "dirA/x.ts", pinned: true },
      { root: "repos", rel: "dirB/x.ts", pinned: true },
    ];
    expect(disambiguateTabLabels(tabs)).toEqual(["dirA/x.ts", "dirB/x.ts"]);
  });

  it("qualifies with grandparent/parent when the parent-qualified label still collides", () => {
    const tabs: Tab[] = [
      { root: "repos", rel: "root1/mid/x.ts", pinned: true },
      { root: "repos", rel: "root2/mid/x.ts", pinned: true },
    ];
    expect(disambiguateTabLabels(tabs)).toEqual(["root1/mid/x.ts", "root2/mid/x.ts"]);
  });

  it("falls back to the full tabKey when basename, parent, AND grandparent all collide (pathological)", () => {
    const tabs: Tab[] = [
      { root: "repos", rel: "same/mid/x.ts", pinned: true },
      { root: "vault", rel: "same/mid/x.ts", pinned: true },
    ];
    expect(disambiguateTabLabels(tabs)).toEqual(["repos:same/mid/x.ts", "vault:same/mid/x.ts"]);
  });

  it("preserves order and length even with 3+ tabs, some colliding and some not", () => {
    const tabs: Tab[] = [
      { root: "repos", rel: "z.ts", pinned: true },
      { root: "repos", rel: "a/dup.ts", pinned: true },
      { root: "repos", rel: "b/dup.ts", pinned: true },
    ];
    expect(disambiguateTabLabels(tabs)).toEqual(["z.ts", "a/dup.ts", "b/dup.ts"]);
  });

  it("resolves collisions per group, independently — an unrelated collision never qualifies an already-unique basename (round-1 F2)", () => {
    const tabs: Tab[] = [
      { root: "repos", rel: "src/unique.ts", pinned: true },
      { root: "repos", rel: "a/x.ts", pinned: true },
      { root: "repos", rel: "b/x.ts", pinned: true },
    ];
    expect(disambiguateTabLabels(tabs)).toEqual(["unique.ts", "a/x.ts", "b/x.ts"]);
  });
});

describe("parseRecentRepos / pushRecentRepo (generalized recents, spec §9)", () => {
  it("parses valid string entries, drops junk, dedupes, caps at RECENTS_CAP — same shape as parseRecents", () => {
    for (const junk of [null, undefined, 42, {}]) expect(parseRecentRepos(junk)).toEqual([]);
    expect(parseRecentRepos(["repo:a", "", 5, "vault", "repo:a"])).toEqual(["repo:a", "vault"]);
    expect(parseRecentRepos(Array.from({ length: RECENTS_CAP + 5 }, (_, i) => `repo:r${i}`))).toHaveLength(RECENTS_CAP);
  });
  it("pushRecentRepo prepends and dedupes, capping at RECENTS_CAP", () => {
    expect(pushRecentRepo(["repo:a", "vault"], "repo:b")).toEqual(["repo:b", "repo:a", "vault"]);
    expect(pushRecentRepo(["repo:a", "vault"], "repo:a")).toEqual(["repo:a", "vault"]); // re-pick moves to front, no dup
    expect(pushRecentRepo(Array.from({ length: RECENTS_CAP }, (_, i) => `repo:r${i}`), "vault")).toHaveLength(RECENTS_CAP);
  });
  it("RECENT_REPOS_KEY is its own storage key", () => expect(RECENT_REPOS_KEY).toBe("jax-os.files.recentRepos.v1"));
});
