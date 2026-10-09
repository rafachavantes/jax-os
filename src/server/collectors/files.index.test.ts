import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.hoisted(() => {
  const { mkdtempSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const dir = mkdtempSync(join(tmpdir(), "jaxos-files-index-test-"));
  const vaultDir = mkdtempSync(join(tmpdir(), "jaxos-files-index-test-vault-"));
  writeFileSync(join(dir, "settings.json"), JSON.stringify({ vaultPath: vaultDir }), "utf8");
  process.env.JAXOS_HOME = dir;
});

vi.mock("node:fs/promises", async (importOriginal) => {
  const fs = await importOriginal<typeof import("node:fs/promises")>();
  return { ...fs, opendir: vi.fn(fs.opendir), realpath: vi.fn(fs.realpath) };
});

import { opendir, realpath } from "node:fs/promises";
import { buildIndex, buildIndexUncached, INDEX_PATHS_CAP, INDEX_TIME_MS, INDEX_VISIT_CAP, ROOTS } from "./files";

const VAULT = ROOTS.vault as string;

type FakeEnt = { name: string; kind: "file" | "dir" | "symlink" };
function fakeDir(entries: FakeEnt[]) {
  let i = 0;
  return {
    async read() {
      const next = entries[i++];
      if (!next) return null;
      return {
        name: next.name,
        isDirectory: () => next.kind === "dir",
        isFile: () => next.kind === "file",
        isSymbolicLink: () => next.kind === "symlink",
      };
    },
    async close() {},
  };
}
function manyFiles(prefix: string, n: number): FakeEnt[] {
  return Array.from({ length: n }, (_, i) => ({ name: `${prefix}${i}.txt`, kind: "file" as const }));
}
function manyDirs(prefix: string, n: number): FakeEnt[] {
  return Array.from({ length: n }, (_, i) => ({ name: `${prefix}${i}`, kind: "dir" as const }));
}
function installTree(tree: Record<string, FakeEnt[]>) {
  vi.mocked(realpath).mockImplementation(async (p) => String(p));
  vi.mocked(opendir).mockImplementation(async (p) => {
    const node = tree[String(p)];
    if (!node) throw Object.assign(new Error("ENOENT"), { code: "ENOENT" });
    return fakeDir(node) as unknown as Awaited<ReturnType<typeof opendir>>;
  });
}

describe("buildIndexUncached (own per-root budget — cold review round 2 F3)", () => {
  beforeEach(() => {
    vi.mocked(opendir).mockReset();
    vi.mocked(realpath).mockReset();
  });
  afterEach(() => vi.restoreAllMocks());

  it("truncates a root over INDEX_PATHS_CAP without shrinking the other root's own budget", async () => {
    installTree({ [ROOTS.repos]: manyFiles("r", INDEX_PATHS_CAP + 500), [VAULT]: manyFiles("v", 10) });
    const result = await buildIndexUncached({ kind: "all" });
    expect(result.truncatedRoots).toEqual(["repos"]);
    expect(result.truncated).toBe(true);
    expect(result.data.filter((h) => h.root === "repos")).toHaveLength(INDEX_PATHS_CAP);
    expect(result.data.filter((h) => h.root === "vault")).toHaveLength(10);
  });

  it("truncates a root over INDEX_VISIT_CAP, resetting the counter per root", async () => {
    installTree({ [ROOTS.repos]: manyFiles("r", INDEX_VISIT_CAP + 100), [VAULT]: manyFiles("v", 5) });
    const result = await buildIndexUncached({ kind: "all" });
    expect(result.truncatedRoots).toEqual(["repos"]);
    expect(result.data.filter((h) => h.root === "vault")).toHaveLength(5);
  });

  it("resets the time budget per root", async () => {
    let now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    installTree({ [ROOTS.repos]: [{ name: "slow.txt", kind: "file" }], [VAULT]: [{ name: "fast.txt", kind: "file" }] });
    vi.mocked(realpath).mockImplementation(async (p) => {
      const path = String(p);
      if (path === ROOTS.repos) now = 10_000; // already over INDEX_TIME_MS once repos starts walking
      return path;
    });
    const result = await buildIndexUncached({ kind: "all" });
    expect(result.truncatedRoots).toContain("repos");
    expect(result.data.some((h) => h.name === "fast.txt")).toBe(true);
  });

  it("excludes heavy dirs, same predicate as searchRoots", async () => {
    installTree({ [ROOTS.repos]: [{ name: "node_modules", kind: "dir" }, { name: "keep.ts", kind: "file" }], [VAULT]: [] });
    const result = await buildIndexUncached({ kind: "all" });
    expect(result.data.filter((h) => h.root === "repos").map((h) => h.name)).toEqual(["keep.ts"]);
  });

  it("throws when a root's own realpath fails, instead of masking it as a truncated budget cap (diff review F2)", async () => {
    installTree({ [ROOTS.repos]: [{ name: "keep.ts", kind: "file" }], [VAULT]: [] });
    vi.mocked(realpath).mockImplementation(async (p) => {
      if (String(p) === ROOTS.repos) throw Object.assign(new Error("EACCES"), { code: "EACCES" });
      return String(p);
    });
    await expect(buildIndexUncached({ kind: "all" })).rejects.toThrow();
  });

  it("throws when a root's own opendir fails, instead of masking it as a truncated budget cap (diff review F1)", async () => {
    installTree({ [VAULT]: [] }); // repos is absent from the tree, so opendir(ROOTS.repos) throws ENOENT
    await expect(buildIndexUncached({ kind: "all" })).rejects.toThrow();
  });

  it("truncates (not throws) when a NESTED directory's opendir fails, still indexing the rest", async () => {
    installTree({
      [ROOTS.repos]: [{ name: "sub", kind: "dir" }, { name: "keep.ts", kind: "file" }],
      // "sub" itself is absent from the tree, so opendir on it throws ENOENT one level down
      [VAULT]: [{ name: "v.txt", kind: "file" }],
    });
    const result = await buildIndexUncached({ kind: "all" });
    expect(result.truncatedRoots).toEqual(["repos"]);
    expect(result.data.filter((h) => h.root === "repos").map((h) => h.name)).toEqual(["keep.ts"]);
    expect(result.data.filter((h) => h.root === "vault").map((h) => h.name)).toEqual(["v.txt"]);
  });

  it("skips a symlink unconditionally, whether it points inside the tree or escapes the root (cold review round 1 F4)", async () => {
    installTree({
      [ROOTS.repos]: [
        { name: "inside-link", kind: "symlink" },
        { name: "escape-link", kind: "symlink" },
        { name: "keep.ts", kind: "file" },
      ],
      [VAULT]: [],
    });
    const result = await buildIndexUncached({ kind: "all" });
    // Both links are skipped by the same entry.isSymbolicLink() guard before any realpath check —
    // the walker never follows a symlink to inspect its target, inside-tree or escaping alike.
    expect(result.data.filter((h) => h.root === "repos").map((h) => h.name)).toEqual(["keep.ts"]);
  });
});

describe("walkAllPaths fairness (spec 2026-09-19 §5 #1/#3/#13/#14 — per-slice budget, FI-1)", () => {
  function fairnessTree(order: readonly ["small", "huge"] | readonly ["huge", "small"]) {
    const dirs = order.map((name) => ({ name, kind: "dir" as const }));
    installTree({
      [ROOTS.repos]: [...dirs, { name: "loose.txt", kind: "file" }],
      [`${ROOTS.repos}/small`]: manyFiles("s", 3),
      [`${ROOTS.repos}/huge`]: manyFiles("h", INDEX_PATHS_CAP),
      [VAULT]: [],
    });
  }

  it.each([
    ["small first", ["small", "huge"] as const],
    ["huge first", ["huge", "small"] as const],
  ])("indexes the small sibling in full and the root's own loose file, regardless of readdir order (%s)", async (_label, order) => {
    fairnessTree(order);
    const result = await buildIndexUncached({ kind: "all" });
    const repos = result.data.filter((h) => h.root === "repos");
    expect(repos.filter((h) => h.rel.startsWith("small/"))).toHaveLength(3);
    expect(repos.some((h) => h.rel === "loose.txt")).toBe(true);
    expect(repos.some((h) => h.rel.startsWith("huge/"))).toBe(true);
    expect(repos.filter((h) => h.rel.startsWith("huge/")).length).toBeLessThan(INDEX_PATHS_CAP);
    expect(result.truncatedRoots).toContain("repos");
  });

  it("per-slice truncation rule (§5 #6): never truncated when every slice stays within its own cap; truncated only for an oversized sibling, others finish fully", async () => {
    installTree({
      [ROOTS.repos]: [{ name: "a", kind: "dir" }, { name: "b", kind: "dir" }, { name: "c", kind: "dir" }],
      [`${ROOTS.repos}/a`]: manyFiles("a", 100),
      [`${ROOTS.repos}/b`]: manyFiles("b", 100),
      [`${ROOTS.repos}/c`]: manyFiles("c", 100),
      [VAULT]: [],
    });
    const evenResult = await buildIndexUncached({ kind: "all" });
    expect(evenResult.truncatedRoots).toEqual([]);
    expect(evenResult.data.filter((h) => h.root === "repos")).toHaveLength(300);

    installTree({
      [ROOTS.repos]: [{ name: "a", kind: "dir" }, { name: "big", kind: "dir" }],
      [`${ROOTS.repos}/a`]: manyFiles("a", 50),
      [`${ROOTS.repos}/big`]: manyFiles("big", INDEX_PATHS_CAP),
      [VAULT]: [],
    });
    const skewedResult = await buildIndexUncached({ kind: "all" });
    const repos = skewedResult.data.filter((h) => h.root === "repos");
    expect(repos.filter((h) => h.rel.startsWith("a/"))).toHaveLength(50); // fully indexed, its own slice never hit its cap
    expect(repos.filter((h) => h.rel.startsWith("big/")).length).toBeLessThan(INDEX_PATHS_CAP);
    expect(skewedResult.truncatedRoots).toEqual(["repos"]);
  });

  it("behaves identically to a flat root when there is only one top-level child (boundary, spec §8)", async () => {
    installTree({
      [ROOTS.repos]: [{ name: "only", kind: "dir" }],
      [`${ROOTS.repos}/only`]: manyFiles("o", 25),
      [VAULT]: [],
    });
    const result = await buildIndexUncached({ kind: "all" });
    expect(result.data.filter((h) => h.root === "repos")).toHaveLength(25);
    expect(result.truncatedRoots).toEqual([]);
  });

  it("gives a single, oversized top-level child the WHOLE cap, not half (cap-boundary, round 1 F2)", async () => {
    installTree({
      [ROOTS.repos]: [{ name: "only", kind: "dir" }],
      [`${ROOTS.repos}/only`]: manyFiles("o", INDEX_PATHS_CAP + 500),
      [VAULT]: [],
    });
    const result = await buildIndexUncached({ kind: "all" });
    // Unsliced: the child's own slice gets the full INDEX_PATHS_CAP, exactly like a flat walk over
    // the same tree would — not INDEX_PATHS_CAP / 2 from a root-bucket/child split.
    expect(result.data.filter((h) => h.root === "repos")).toHaveLength(INDEX_PATHS_CAP);
    expect(result.truncatedRoots).toEqual(["repos"]);
  });
});

describe("root bucket shares ONE budget between discovery and loose files (spec §5 #13, review F2)", () => {
  it("truncates and skips every loose file once discovery alone exhausts the root bucket's share", async () => {
    // 49 top-level subdirs -> sliceCount 50 -> root bucket gets floor(INDEX_VISIT_CAP/50) visits.
    // 1200 loose files pushes discovery's own entry count (49 + 1200) past that share on its own,
    // before a single loose file is even considered.
    installTree({
      [ROOTS.repos]: [...manyDirs("d", 49), ...manyFiles("loose", 1200)],
      [VAULT]: [],
    });
    const result = await buildIndexUncached({ kind: "all" });
    const repos = result.data.filter((h) => h.root === "repos");
    expect(repos.some((h) => h.name.startsWith("loose"))).toBe(false);
    expect(result.truncatedRoots).toContain("repos");
  });
});

describe("a slow top-level realpath still marks the slice truncated (spec §5 #6, review F3)", () => {
  afterEach(() => vi.restoreAllMocks());
  it("marks truncated when overBudget() trips right at the realpath gate, before walk() ever runs", async () => {
    let now = 0;
    vi.spyOn(performance, "now").mockImplementation(() => now);
    installTree({
      [ROOTS.repos]: [{ name: "child", kind: "dir" }],
      [`${ROOTS.repos}/child`]: [{ name: "f.txt", kind: "file" }],
      [VAULT]: [],
    });
    vi.mocked(realpath).mockImplementation(async (p) => {
      const path = String(p);
      if (path === `${ROOTS.repos}/child`) now = INDEX_TIME_MS + 1000; // blows the slice's own clock before walk() starts
      return path;
    });
    const result = await buildIndexUncached({ kind: "all" });
    const repos = result.data.filter((h) => h.root === "repos");
    expect(repos).toHaveLength(0); // walk() never ran — the gate skipped it
    expect(result.truncatedRoots).toContain("repos"); // but that skip must still be reported
  });
});

describe("buildIndex cache", () => {
  beforeEach(() => {
    vi.mocked(opendir).mockReset();
    vi.mocked(realpath).mockReset();
    installTree({ [ROOTS.repos]: [], [VAULT]: [] });
  });
  afterEach(() => vi.restoreAllMocks());

  it("serves a cached result within the 30s TTL, then rebuilds after it expires", async () => {
    const first = await buildIndex({ kind: "all" }, 0);
    installTree({ [ROOTS.repos]: [{ name: "new.ts", kind: "file" }], [VAULT]: [] });
    const stillCached = await buildIndex({ kind: "all" }, 1000);
    expect(stillCached.data).toEqual(first.data);
    const fresh = await buildIndex({ kind: "all" }, 31_000);
    expect(fresh.data.some((h) => h.name === "new.ts")).toBe(true);
  });
});
