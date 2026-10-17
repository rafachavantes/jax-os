import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const actualFs = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");

vi.mock("node:fs/promises", async (importOriginal) => {
  const fs = await importOriginal<typeof import("node:fs/promises")>();
  return { ...fs, opendir: vi.fn(fs.opendir), realpath: vi.fn(fs.realpath) };
});

import { opendir, realpath } from "node:fs/promises";
import {
  SEARCH_CAP,
  SEARCH_MAX_DEPTH,
  SEARCH_TIME_MS,
  SEARCH_VISIT_CAP,
  searchByName,
  searchRoots,
} from "./files";

type Kind = "file" | "dir" | "symlink" | "other";
type Tracker = { opened: number; closed: number; closeAttempts: string[]; pending: number; maxPending: number; processed: number };
type FakeEnt = { name: string; kind: Kind };

function tracker(): Tracker {
  return { opened: 0, closed: 0, closeAttempts: [], pending: 0, maxPending: 0, processed: 0 };
}

function trackIO<T>(t: Tracker, work: Promise<T>): Promise<T> {
  t.pending++;
  t.maxPending = Math.max(t.maxPending, t.pending);
  return work.finally(() => { t.pending--; });
}

function fsError(code: string): NodeJS.ErrnoException {
  const err = new Error(code) as NodeJS.ErrnoException;
  err.code = code;
  return err;
}

function dirent(t: Tracker, name: string, kind: Kind) {
  let seen = false;
  const mark = () => { if (!seen) { seen = true; t.processed++; } };
  return {
    get name() { mark(); return name; },
    isDirectory: () => { mark(); return kind === "dir"; },
    isFile: () => { mark(); return kind === "file"; },
    isSymbolicLink: () => { mark(); return kind === "symlink"; },
  };
}

function fakeDir(t: Tracker, path: string, entries: FakeEnt[], onRead?: (i: number) => void, closeError?: Error) {
  let i = 0;
  let closed = false;
  return {
    read() {
      if (closed) return Promise.reject(new Error("read after close"));
      return trackIO(t, Promise.resolve().then(() => {
        onRead?.(i);
        const next = entries[i++];
        return next ? dirent(t, next.name, next.kind) : null;
      }));
    },
    close() {
      return trackIO(t, Promise.resolve().then(() => {
        t.closeAttempts.push(path);
        if (closed) throw new Error("double close");
        if (closeError) throw closeError;
        closed = true;
        t.closed++;
      }));
    },
  };
}

function files(prefix: string, n: number): FakeEnt[] {
  return Array.from({ length: n }, (_, i) => ({ name: `${prefix}${i}.txt`, kind: "file" as const }));
}

async function withTracker(run: (t: Tracker) => Promise<void>) {
  const t = tracker();
  try {
    await run(t);
  } finally {
    expect(t.opened).toBe(t.closed);
    expect(t.maxPending).toBeLessThanOrEqual(1);
  }
}

function installTree(
  t: Tracker,
  tree: Record<string, FakeEnt[] | Error>,
  onRead?: (path: string, i: number) => void,
  closeErrors?: Record<string, Error>,
) {
  vi.mocked(realpath).mockImplementation(async (p) => String(p));
  vi.mocked(opendir).mockImplementation(async (p) => {
    const key = String(p);
    const node = tree[key];
    if (node instanceof Error) throw node;
    if (!node) throw fsError("ENOENT");
    t.opened++;
    return fakeDir(t, key, node, (i) => onRead?.(key, i), closeErrors?.[key]) as unknown as Awaited<ReturnType<typeof actualFs.opendir>>;
  });
}

describe("searchRoots", () => {
  beforeEach(() => {
    vi.mocked(opendir).mockImplementation(actualFs.opendir);
    vi.mocked(realpath).mockImplementation(actualFs.realpath);
    vi.mocked(opendir).mockClear();
    vi.mocked(realpath).mockClear();
  });
  afterEach(() => {
    vi.restoreAllMocks();
    vi.mocked(opendir).mockImplementation(actualFs.opendir);
    vi.mocked(realpath).mockImplementation(actualFs.realpath);
  });

  describe("real fixtures", () => {
    let base: string;
    let outside: string;
    let deepBase: string;
    beforeAll(() => {
      base = mkdtempSync(join(tmpdir(), "jax-search-"));
      outside = mkdtempSync(join(tmpdir(), "jax-search-out-"));
      deepBase = mkdtempSync(join(tmpdir(), "jax-search-deep-"));
      mkdirSync(join(base, "deep", "er"), { recursive: true });
      writeFileSync(join(base, "deep", "er", "FindMe.ts"), "");
      mkdirSync(join(base, "node_modules"));
      writeFileSync(join(base, "node_modules", "hidden.ts"), "");
      mkdirSync(join(base, "sub"));
      writeFileSync(join(base, "sub", "a.txt"), "hi");
      writeFileSync(join(outside, "secret.txt"), "SECRET");
      symlinkSync(outside, join(base, "escapedir"));
      let deep = deepBase;
      for (let i = 1; i <= SEARCH_MAX_DEPTH; i++) {
        deep = join(deep, `n${i}`);
        mkdirSync(deep);
      }
      writeFileSync(join(deep, "leaf.ts"), "");
      mkdirSync(join(deep, "n13"));
      writeFileSync(join(deep, "n13", "too-deep.ts"), "");
    });
    afterAll(() => {
      rmSync(base, { recursive: true, force: true });
      rmSync(outside, { recursive: true, force: true });
      rmSync(deepBase, { recursive: true, force: true });
    });
    it("matches case-insensitively on path, excludes heavy dirs, files only", async () => {
      const found = await searchRoots([{ root: "repos", path: base }], "findme");
      expect(found.truncated).toBe(false);
      expect(found.hits.some((h) => h.name === "FindMe.ts")).toBe(true);
      expect(found.hits.some((h) => h.rel.includes("node_modules"))).toBe(false);
      const deep = await searchRoots([{ root: "repos", path: base }], "deep");
      expect(deep.hits.every((h) => h.rel !== "deep")).toBe(true);
      expect(deep.hits.some((h) => h.rel === "deep/er/FindMe.ts")).toBe(true);
    });
    it("returns empty complete for a blank query without filesystem work", async () => {
      expect(await searchRoots([{ root: "repos", path: base }], "   ")).toEqual({ hits: [], truncated: false });
      expect(opendir).not.toHaveBeenCalled();
      expect(await searchByName("", { kind: "all" })).toEqual({ hits: [], truncated: false });
      expect(opendir).not.toHaveBeenCalled();
    });
    it("includes a depth-12 file and does not open the deeper subtree", async () => {
      const rel = Array.from({ length: SEARCH_MAX_DEPTH }, (_, i) => `n${i + 1}`).join("/");
      const result = await searchRoots([{ root: "repos", path: deepBase }], "leaf");
      expect(result.hits.some((h) => h.rel === `${rel}/leaf.ts`)).toBe(true);
      const deeper = await searchRoots([{ root: "repos", path: deepBase }], "too-deep");
      expect(deeper.hits).toEqual([]);
      expect(deeper.truncated).toBe(true);
    });
    it("does not traverse a directory symlink or emit escaped hits", async () => {
      const result = await searchRoots([{ root: "repos", path: base }], "secret");
      expect(result.hits).toEqual([]);
    });
    it("completes a small successful search", async () => {
      const result = await searchRoots([{ root: "repos", path: base }], "a.txt");
      expect(result).toEqual({
        hits: [{ root: "repos", rel: "sub/a.txt", name: "a.txt" }],
        truncated: false,
      });
    });
  });

  it("splits the 200-hit cap in half across two roots, each capped independently (§5 #7 — no reclaiming, §5 #5)", async () => {
    await withTracker(async (t) => {
      installTree(t, {
        "/a": files("hit-a-", 50),
        "/b": files("hit-b-", 200),
      });
      const result = await searchRoots(
        [{ root: "repos", path: "/a" }, { root: "vault", path: "/b" }],
        "hit",
      );
      expect(result.hits).toHaveLength(150);
      expect(result.truncated).toBe(true);
      expect(result.hits.filter((h) => h.root === "repos")).toHaveLength(50); // well under its own 100-hit half
      expect(result.hits.filter((h) => h.root === "vault")).toHaveLength(100); // capped by its OWN half, not reclaiming repos's unused 50
    });
  });

  it("stops before processing the 20001st entry across roots", async () => {
    await withTracker(async (t) => {
      installTree(t, {
        "/a": files("x", 15000),
        "/b": files("y", 15000),
      });
      const result = await searchRoots(
        [{ root: "repos", path: "/a" }, { root: "vault", path: "/b" }],
        "nope",
      );
      expect(result.hits).toEqual([]);
      expect(result.truncated).toBe(true);
      expect(t.processed).toBe(SEARCH_VISIT_CAP);
    });
  });

  it("keeps earlier hits when time expires on the second root", async () => {
    await withTracker(async (t) => {
      let now = 0;
      vi.spyOn(performance, "now").mockImplementation(() => now);
      installTree(t, {
        "/a": [{ name: "keep.ts", kind: "file" }],
        "/b": [{ name: "late.ts", kind: "file" }],
      });
      vi.mocked(realpath).mockImplementation(async (p) => {
        const path = String(p);
        if (path === "/b") now = SEARCH_TIME_MS;
        return path;
      });
      const result = await searchRoots(
        [{ root: "repos", path: "/a" }, { root: "vault", path: "/b" }],
        ".ts",
      );
      expect(result.hits).toEqual([{ root: "repos", rel: "keep.ts", name: "keep.ts" }]);
      expect(result.truncated).toBe(true);
    });
  });

  it("rejects an initial abort without opening directories", async () => {
    await withTracker(async (t) => {
      installTree(t, { "/a": files("z", 1) });
      const controller = new AbortController();
      const reason = new Error("stop");
      controller.abort(reason);
      await expect(searchRoots([{ root: "repos", path: "/a" }], "z", controller.signal)).rejects.toBe(reason);
      expect(opendir).not.toHaveBeenCalled();
    });
  });

  it("rejects abort during child iteration and closes handles", async () => {
    await withTracker(async (t) => {
      const controller = new AbortController();
      const reason = new Error("mid");
      installTree(t, {
        "/a": [{ name: "sub", kind: "dir" }, { name: "keep.ts", kind: "file" }],
        "/a/sub": [{ name: "child.ts", kind: "file" }],
      }, (path) => {
        if (path === "/a/sub") controller.abort(reason);
      });
      await expect(searchRoots([{ root: "repos", path: "/a" }], "child", controller.signal)).rejects.toBe(reason);
    });
  });

  it("throws a fixed source error when the second root fails after hits", async () => {
    await withTracker(async (t) => {
      installTree(t, {
        "/a": [{ name: "keep.ts", kind: "file" }],
        "/b": fsError("EACCES"),
      });
      await expect(searchRoots(
        [{ root: "repos", path: "/a" }, { root: "vault", path: "/b" }],
        "keep",
      )).rejects.toThrow("file search unavailable");
    });
  });

  it("keeps hits and marks truncated on a child read failure", async () => {
    await withTracker(async (t) => {
      installTree(t, {
        "/a": [{ name: "keep.ts", kind: "file" }, { name: "locked", kind: "dir" }],
        "/a/locked": fsError("EACCES"),
      });
      const result = await searchRoots([{ root: "repos", path: "/a" }], "keep");
      expect(result.hits).toEqual([{ root: "repos", rel: "keep.ts", name: "keep.ts" }]);
      expect(result.truncated).toBe(true);
    });
  });

    it("skips a child replaced by a symlink and marks incomplete", async () => {
      await withTracker(async (t) => {
        installTree(t, {
          "/a": [{ name: "trap", kind: "dir" }, { name: "ok.ts", kind: "file" }],
          "/etc": [{ name: "passwd", kind: "file" }],
        });
        vi.mocked(realpath).mockImplementation(async (p) => String(p) === "/a/trap" ? "/etc" : String(p));
        const result = await searchRoots([{ root: "repos", path: "/a" }], "passwd");
        expect(result.hits).toEqual([]);
        expect(result.truncated).toBe(true);
      });
    });

    it("keeps first-root hits when a child close fails in the second root", async () => {
      const t = tracker();
      installTree(t, {
        "/a": [{ name: "keep.ts", kind: "file" }],
        "/b": [{ name: "sub", kind: "dir" }],
        "/b/sub": [{ name: "x.ts", kind: "file" }],
      }, undefined, { "/b/sub": new Error("child close") });
      const result = await searchRoots(
        [{ root: "repos", path: "/a" }, { root: "vault", path: "/b" }],
        "keep",
      );
      expect(result.hits).toEqual([{ root: "repos", rel: "keep.ts", name: "keep.ts" }]);
      expect(result.truncated).toBe(true);
      expect(t.closeAttempts).toContain("/b/sub");
      expect(t.closeAttempts).toContain("/a");
      expect(t.closeAttempts).toContain("/b");
      expect(t.closed).toBe(2);
    });

    it("treats root close failure as a source error", async () => {
      const t = tracker();
      installTree(t, {
        "/a": [{ name: "keep.ts", kind: "file" }],
      }, undefined, { "/a": new Error("root close") });
      await expect(searchRoots([{ root: "repos", path: "/a" }], "keep")).rejects.toThrow("file search unavailable");
      expect(t.closeAttempts).toEqual(["/a"]);
      expect(t.closed).toBe(0);
    });

    it("rejects abort over a failing child close and still closes ancestors", async () => {
      const t = tracker();
      const controller = new AbortController();
      const reason = new Error("mid");
      installTree(t, {
        "/a": [{ name: "sub", kind: "dir" }, { name: "keep.ts", kind: "file" }],
        "/a/sub": [{ name: "child.ts", kind: "file" }],
      }, (path) => {
        if (path === "/a/sub") controller.abort(reason);
      }, { "/a/sub": new Error("child close") });
      await expect(searchRoots([{ root: "repos", path: "/a" }], "child", controller.signal)).rejects.toBe(reason);
      // Discovery must read "/a"'s full top-level listing (to size its slices) before any
      // subdirectory slice is walked — so "/a"'s own handle now closes FIRST, not last (§5 #14).
      expect(t.closeAttempts).toEqual(["/a", "/a/sub"]);
      expect(t.closed).toBe(1);
    });

    describe("fairness (spec 2026-09-19 §5 #1/#7/#13/#14 — per-slice budget, FI-1)", () => {
    it("matches a hit in the small sibling + the root's own loose file within a root, and splits the budget across two roots so an oversized child in one cannot exhaust the other's (§5 #7)", async () => {
      await withTracker(async (t) => {
        installTree(t, {
          "/a": [{ name: "small", kind: "dir" }, { name: "huge", kind: "dir" }, { name: "loose-hit.txt", kind: "file" }],
          "/a/small": [{ name: "hit.txt", kind: "file" }],
          "/a/huge": files("hit-huge-", SEARCH_CAP),
        });
        const oneRoot = await searchRoots([{ root: "repos", path: "/a" }], "hit");
        expect(oneRoot.hits.some((h) => h.rel === "small/hit.txt")).toBe(true);
        expect(oneRoot.hits.some((h) => h.rel === "loose-hit.txt")).toBe(true);
        expect(oneRoot.hits.some((h) => h.rel.startsWith("huge/"))).toBe(true);
        expect(oneRoot.truncated).toBe(true); // huge's own slice exhausts its share
      });
      await withTracker(async (t) => {
        installTree(t, {
          "/a": files("hit-a-", SEARCH_CAP), // flat — would have consumed the WHOLE old shared cap alone
          "/b": [{ name: "small", kind: "dir" }],
          "/b/small": [{ name: "hit.txt", kind: "file" }],
        });
        const twoRoots = await searchRoots([{ root: "repos", path: "/a" }, { root: "vault", path: "/b" }], "hit");
        expect(twoRoots.hits.some((h) => h.root === "vault" && h.rel === "small/hit.txt")).toBe(true);
        expect(twoRoots.hits.filter((h) => h.root === "repos")).toHaveLength(SEARCH_CAP / 2);
        expect(twoRoots.truncated).toBe(true);
      });
    });

    it("returns the existing empty, complete result for an empty roots list instead of dividing by zero (round 1 F1)", async () => {
      const result = await searchRoots([], "hit");
      expect(result).toEqual({ hits: [], truncated: false });
    });
    });
  });
