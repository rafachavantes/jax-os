import { describe, expect, it, vi } from "vitest";
import type { Root } from "@/server/collectors/files"; // type-only: erased at build, safe here (value @/ imports fail in vitest)
import { parseFileSelection, preferredActive, serializeFileSelection, shouldPush, syncFromUrl, syncToUrl } from "./filesUrl";
import { openFromSearch } from "./filesWorkspace";

describe("parseFileSelection", () => {
  it("is absent when both fields are missing", () => {
    expect(parseFileSelection(new URLSearchParams("q=keep"))).toEqual({ status: "absent" });
  });

  it("accepts unicode paths and does not decode twice", () => {
    const params = new URLSearchParams();
    params.set("root", "repos");
    params.set("rel", "dir/caf%C3%A9.md");
    expect(parseFileSelection(params)).toEqual({ status: "valid", root: "repos", rel: "dir/caf%C3%A9.md" });
    const encoded = new URLSearchParams("root=vault&rel=dir%2F%E6%96%87.txt");
    expect(parseFileSelection(encoded)).toEqual({ status: "valid", root: "vault", rel: "dir/文.txt" });
  });

  it("rejects malformed, absolute, traversal, NUL, and overlong fields", () => {
    expect(parseFileSelection(new URLSearchParams("root=nope&rel=a.ts"))).toEqual({ status: "invalid" });
    expect(parseFileSelection(new URLSearchParams("root=repos"))).toEqual({ status: "invalid" });
    expect(parseFileSelection(new URLSearchParams("root=repos&rel="))).toEqual({ status: "invalid" });
    expect(parseFileSelection(new URLSearchParams("root=repos&rel=/etc/passwd"))).toEqual({ status: "invalid" });
    expect(parseFileSelection(new URLSearchParams("root=repos&rel=../secret"))).toEqual({ status: "invalid" });
    expect(parseFileSelection(new URLSearchParams("root=repos&rel=a/../b"))).toEqual({ status: "invalid" });
    const nul = new URLSearchParams();
    nul.set("root", "repos");
    nul.set("rel", "a\0b");
    expect(parseFileSelection(nul)).toEqual({ status: "invalid" });
    const long = "x".repeat(4097);
    expect(parseFileSelection(new URLSearchParams(`root=repos&rel=${long}`))).toEqual({ status: "invalid" });
  });
});

describe("preferredActive", () => {
  it("lets a valid URL override stored active and ignores stored on invalid URL", () => {
    const stored = { root: "vault" as const, rel: "old.md" };
    expect(preferredActive({ status: "valid", root: "repos", rel: "new.ts" }, stored))
      .toEqual({ status: "ok", file: { root: "repos", rel: "new.ts" } });
    expect(preferredActive({ status: "absent" }, stored)).toEqual({ status: "ok", file: stored });
    expect(preferredActive({ status: "invalid" }, stored)).toEqual({ status: "invalid" });
  });
});

describe("serializeFileSelection", () => {
  it("preserves unrelated fields and round-trips a valid selection", () => {
    const current = new URLSearchParams("q=keep&root=old&rel=gone");
    const next = serializeFileSelection(current, { root: "repos", rel: "a.ts" });
    expect(next.get("q")).toBe("keep");
    expect(next.get("root")).toBe("repos");
    expect(next.get("rel")).toBe("a.ts");
    expect(parseFileSelection(next)).toEqual({ status: "valid", root: "repos", rel: "a.ts" });
  });

  it("deletes root and rel when no tab is selected", () => {
    const next = serializeFileSelection(new URLSearchParams("root=repos&rel=a.ts&x=1"), null);
    expect(next.get("root")).toBeNull();
    expect(next.get("rel")).toBeNull();
    expect(next.get("x")).toBe("1");
  });
});

describe("shouldPush", () => {
  it("pushes only for a pinned open of a different file; replaces otherwise", () => {
    const pinnedA = { root: "repos" as const, rel: "a.ts", pinned: true };
    const previewA = { root: "repos" as const, rel: "a.ts", pinned: false };
    const pinnedB = { root: "repos" as const, rel: "b.ts", pinned: true };
    // first pinned open of the session: push (no prior file to compare against)
    expect(shouldPush(null, pinnedA)).toBe(true);
    // pinned open of a DIFFERENT file than the one previously active: push
    expect(shouldPush(pinnedA, pinnedB)).toBe(true);
    // a preview (single-click) open never pushes, regardless of prior state
    expect(shouldPush(null, previewA)).toBe(false);
    // same-file promotion (preview -> pinned, no navigation): replace, never push
    // (cold review round 2 F1 — a forced push here breaks Back)
    expect(shouldPush(previewA, pinnedA)).toBe(false);
    // no active tab: replace
    expect(shouldPush(pinnedA, null)).toBe(false);
  });
});

describe("syncFromUrl", () => {
  // cold review round 1 F1: the old guard returned before openFilePinned could
  // run whenever root+rel matched, so an already-open but merely-previewed
  // tab never got promoted; an already-PINNED same-file tab is still a no-op.
  it("opens pinned for a new/different file, flags invalid selections, and promotes a same-file PREVIEW (F1)", () => {
    const a = { setUrlError: vi.fn(), openFilePinned: vi.fn() };
    expect(syncFromUrl(new URLSearchParams("root=repos&rel=a.ts"), null, null, a)).toBe(true);
    expect(a.openFilePinned).toHaveBeenCalledWith("repos", "a.ts");
    const activeA = { root: "repos" as const, rel: "a.ts", pinned: true };
    expect(syncFromUrl(new URLSearchParams("root=repos&rel=b.ts"), activeA, null, a)).toBe(true);
    expect(a.openFilePinned).toHaveBeenCalledWith("repos", "b.ts");
    expect(syncFromUrl(new URLSearchParams("root=repos&rel=/etc/passwd"), null, null, a)).toBe(false);
    expect(a.setUrlError).toHaveBeenCalledWith(true);
    a.openFilePinned.mockClear();
    const preview = { root: "repos" as const, rel: "a.ts", pinned: false };
    expect(syncFromUrl(new URLSearchParams("root=repos&rel=a.ts"), preview, null, a)).toBe(true);
    expect(a.openFilePinned).toHaveBeenCalledWith("repos", "a.ts");
    a.openFilePinned.mockClear();
    expect(syncFromUrl(new URLSearchParams("root=repos&rel=a.ts"), { ...preview, pinned: true }, null, a)).toBe(false);
    expect(a.openFilePinned).not.toHaveBeenCalled();
  });
});

describe("syncFromUrl deep-link open (round-3 F1 — page.tsx's action composes openFromSearch)", () => {
  // page.tsx's URL->tabs effect can't be unit-tested directly, so this proves the composition
  // it uses for its openFilePinned action: reveal (scope switch) MUST run before open, or a
  // cross-repo deep link opens the file before requestReveal switches scope and clears `selected`.
  it("reveals/switches scope before opening a deep-linked file from a different repo", () => {
    const calls: string[] = [];
    const reveal = vi.fn((root: string, rel: string, targetIsDir: boolean) => calls.push(`reveal:${root}:${rel}:${targetIsDir}`));
    const open = vi.fn((root: string, rel: string) => calls.push(`open:${root}:${rel}`));
    const actions = {
      setUrlError: vi.fn(),
      openFilePinned: (root: Root, rel: string) => openFromSearch(reveal, open, root, rel),
    };
    expect(syncFromUrl(new URLSearchParams("root=repos&rel=other-repo/a.ts"), null, null, actions)).toBe(true);
    expect(calls).toEqual(["reveal:repos:other-repo/a.ts:false", "open:repos:other-repo/a.ts"]);
  });
});

describe("syncToUrl", () => {
  // F1's fixture, write-direction half: serializeFileSelection never encodes
  // `pinned`, so a same-file promotion leaves the URL matching already — no
  // call fires at all (spec §6), same as when the URL already matches.
  it("replaces a preview open, pushes only a pinned open of a different file, and makes no call for a same-file promotion or an unchanged URL", () => {
    const a = { push: vi.fn(), replace: vi.fn() };
    const preview = { root: "repos" as const, rel: "a.ts", pinned: false };
    syncToUrl(new URLSearchParams(""), "/files", preview, null, a);
    expect(a.replace).toHaveBeenCalledWith("/files?root=repos&rel=a.ts");
    expect(a.push).not.toHaveBeenCalled();
    const prev = { root: "repos" as const, rel: "a.ts", pinned: true };
    const pinnedB = { root: "repos" as const, rel: "b.ts", pinned: true };
    syncToUrl(new URLSearchParams("root=repos&rel=a.ts"), "/files", pinnedB, prev, a);
    expect(a.push).toHaveBeenCalledWith("/files?root=repos&rel=b.ts");
    a.push.mockClear();
    a.replace.mockClear();
    const promoted = { root: "repos" as const, rel: "a.ts", pinned: true };
    syncToUrl(new URLSearchParams("root=repos&rel=a.ts"), "/files", promoted, preview, a);
    syncToUrl(new URLSearchParams("root=repos&rel=a.ts"), "/files", promoted, promoted, a);
    expect(a.push).not.toHaveBeenCalled();
    expect(a.replace).not.toHaveBeenCalled();
  });
});

describe("syncFromUrl + syncToUrl (round 2 F1)", () => {
  // 011eca3d7f72 F1: syncToUrl's own replace() must not re-promote the preview
  // it just wrote when that URL change reruns syncFromUrl; a later external
  // nav to the same URL still promotes it (round 1 F1).
  it("no-ops on its own replace() echo, but still promotes a genuine external nav to the same URL", () => {
    const write = { push: vi.fn(), replace: vi.fn() };
    const prevActive = { root: "repos" as const, rel: "a.ts", pinned: true };
    const preview = { root: "repos" as const, rel: "b.ts", pinned: false };
    const written = syncToUrl(new URLSearchParams("root=repos&rel=a.ts"), "/files", preview, prevActive, write);
    expect(written).toBe("root=repos&rel=b.ts");
    expect(write.replace).toHaveBeenCalledWith("/files?root=repos&rel=b.ts");
    const echoedParams = new URLSearchParams("root=repos&rel=b.ts");
    const echo = { setUrlError: vi.fn(), openFilePinned: vi.fn() };
    expect(syncFromUrl(echoedParams, preview, written, echo)).toBe(true);
    expect(echo.openFilePinned).not.toHaveBeenCalled();
    const external = { setUrlError: vi.fn(), openFilePinned: vi.fn() };
    expect(syncFromUrl(echoedParams, preview, null, external)).toBe(true);
    expect(external.openFilePinned).toHaveBeenCalledWith("repos", "b.ts");
  });
});
