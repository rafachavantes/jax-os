import { describe, expect, it, beforeAll, afterAll, vi } from "vitest";
vi.hoisted(() => {
  const { mkdtempSync, mkdirSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const dir = mkdtempSync(join(tmpdir(), "jaxos-files-test-"));
  const vaultDir = mkdtempSync(join(tmpdir(), "jaxos-files-test-vault-"));
  // A plain, empty mkdtempSync vault fixture (portable — CI has no `/home/rafa/obsidian-vault`
  // and must never depend on it). Empty is enough: every existing "vault" assertion below is
  // satisfied by an empty directory as-is — "walks only the vault root" checks `.every(...)`
  // over the hit list (vacuously true on zero hits), the nonexistent-query search already
  // expects zero hits, and the per-scope cache test only needs two calls against the SAME
  // scope to return the identical object. No fixture files are required for any of them.
  // A portable repos fixture too: the repo-scope tests assert a `jax-os` repo and a
  // `package.json` inside it; without this, ROOTS.repos fell back to `$HOME/repos`
  // (nonexistent on a fresh runner, and an unstated dependency on the owner's real checkout).
  const reposDir = mkdtempSync(join(tmpdir(), "jaxos-files-test-repos-"));
  mkdirSync(join(reposDir, "jax-os"));
  writeFileSync(join(reposDir, "jax-os", "package.json"), '{"name":"jax-os"}', "utf8");
  writeFileSync(join(dir, "settings.json"), JSON.stringify({ vaultPath: vaultDir, reposRoot: reposDir }), "utf8");
  process.env.JAXOS_HOME = dir;
});
import { mkdtempSync, mkdirSync, writeFileSync, symlinkSync, rmSync, existsSync, realpathSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { resolveWithin, resolveNewWithin, isBinary, hashBytes, listDirAt, readFileAt, writeFileAt, READ_CAP, createEntryAt, renameEntryAt, deleteEntryAt, resolveExisting, createEntry, renameEntry, deleteEntry, uploadFile, imageMime, mediaMime, ROOTS, relativeWithin, planSlices, splitTotalCaps, parseScope, listRepoNames, searchByName, buildIndex, buildIndexUncached, gateVaultRoot, gateVaultScope } from "./files";

it("ROOTS.repos is never null; ROOTS.vault is a string when settings configure one", () => {
  expect(typeof ROOTS.repos).toBe("string");
  expect(typeof ROOTS.vault).toBe("string");
});

let base: string, outside: string;
beforeAll(() => {
  base = mkdtempSync(join(tmpdir(), "jax-files-"));
  outside = mkdtempSync(join(tmpdir(), "jax-outside-"));
  mkdirSync(join(base, "sub"));
  writeFileSync(join(base, "sub", "a.txt"), "hi");
  writeFileSync(join(outside, "secret.txt"), "SECRET");
  symlinkSync(join(outside, "secret.txt"), join(base, "escape"));      // symlink out of base
  symlinkSync(outside, join(base, "escapedir"));                       // dir symlink out of base
  mkdirSync(join(base, "node_modules"));
  writeFileSync(join(base, "b.md"), "# hi");
  writeFileSync(join(base, "bin"), Buffer.from([1, 0, 2]));
  writeFileSync(join(base, "big.bin"), Buffer.alloc(READ_CAP + 1024));
});
afterAll(() => { rmSync(base, { recursive: true, force: true }); rmSync(outside, { recursive: true, force: true }); });

describe("resolveWithin", () => {
  it("allows an in-tree file and the root itself", () => {
    expect(resolveWithin(base, "sub/a.txt")).toBe(join(base, "sub", "a.txt"));
    expect(resolveWithin(base, "")).toBe(base);
  });
  it("rejects ../ traversal", () => {
    expect(() => resolveWithin(base, "../jax-outside-x/secret.txt")).toThrow();
    expect(() => resolveWithin(base, "sub/../../etc/passwd")).toThrow();
    expect(() => resolveWithin(base, "..")).toThrow();   // resolves to an EXISTING dir (/tmp) → exercises the allowlist branch, not just ENOENT
  });
  it("neutralizes an absolute rel via join (does not escape)", () => {
    // join(base, "/etc/passwd") === base/etc/passwd, which doesn't exist → realpath throws (still not an escape)
    expect(() => resolveWithin(base, "/etc/passwd")).toThrow();
  });
  it("rejects a symlink escaping the root (file and dir)", () => {
    expect(() => resolveWithin(base, "escape")).toThrow();
    expect(() => resolveWithin(base, "escapedir/secret.txt")).toThrow();
  });
});

describe("resolveNewWithin", () => {
  it("returns parent+basename for a valid new name", () => {
    expect(resolveNewWithin(base, "sub", "new.txt")).toBe(join(base, "sub", "new.txt"));
  });
  it("rejects bad basenames", () => {
    for (const bad of ["", "a/b", "..", "x\0y"]) expect(() => resolveNewWithin(base, "sub", bad)).toThrow();
  });
  it("rejects when the parent dir is a symlink out of the root", () => {
    expect(() => resolveNewWithin(base, "escapedir", "x.txt")).toThrow();
  });
});

describe("isBinary / hashBytes", () => {
  it("flags a NUL byte as binary", () => {
    expect(isBinary(Buffer.from([1, 0, 2]))).toBe(true);
    expect(isBinary(Buffer.from("hello"))).toBe(false);
  });
  it("hashes stably", () => {
    expect(hashBytes(Buffer.from("hi"))).toBe(hashBytes(Buffer.from("hi")));
  });
});

describe("listDirAt", () => {
  it("excludes heavy dirs and sorts dirs-first", async () => {
    const { dirs, files } = await listDirAt(base, async () => null);
    expect(dirs.map((d) => d.name)).not.toContain("node_modules");
    expect(files.some((f) => f.name === "b.md")).toBe(true);
  });

  it("hides .jax-trash the same way it hides node_modules", async () => {
    mkdirSync(join(base, ".jax-trash"));
    expect((await listDirAt(base, async () => null)).dirs.map((d) => d.name)).not.toContain(".jax-trash");
  });

  it("populates mtime on every entry and size on files only", async () => {
    const { dirs, files } = await listDirAt(base, async () => null);
    const bmd = files.find((f) => f.name === "b.md");
    expect(bmd?.size).toBeGreaterThan(0);
    expect(bmd?.mtime).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    const sub = dirs.find((d) => d.name === "sub") as { size?: number } | undefined;
    expect(sub?.size).toBeUndefined();
  });

  it("maps a git-status result onto the matching entry by name, defaulting to null otherwise", async () => {
    const gitMap = new Map([["b.md", "modified" as const]]);
    const { files } = await listDirAt(base, async () => gitMap);
    expect(files.find((f) => f.name === "b.md")?.gitStatus).toBe("modified");
    expect(files.find((f) => f.name === "bin")?.gitStatus).toBeNull();
  });

  it("does not throw when the directory contains a dangling symlink (cold review round 1 F1)", async () => {
    symlinkSync(join(base, "does-not-exist"), join(base, "broken-link"));
    const { files } = await listDirAt(base, async () => null);
    // lstatSync succeeds on the link itself even though its target is missing — the entry surfaces
    // (with the link's own metadata) instead of throwing and aborting the rest of the listing.
    expect(files.some((f) => f.name === "broken-link")).toBe(true);
    expect(files.some((f) => f.name === "b.md")).toBe(true);
  });
});

describe("readFileAt", () => {
  it("reads text with a hash", () => {
    const r = readFileAt(join(base, "b.md"));
    expect(r.binary).toBe(false);
    expect(r.content).toBe("# hi");
    expect(r.hash.length).toBe(64);
  });
  it("flags binary content", () => {
    expect(readFileAt(join(base, "bin")).binary).toBe(true);
  });
  it("short-circuits an over-cap file without reading its bytes", () => {
    const r = readFileAt(join(base, "big.bin"));
    expect(r.binary).toBe(true);
    expect(r.content).toBe(null);
    expect(r.size).toBe(READ_CAP + 1024);
  });
});

describe("writeFileAt", () => {
  it("writes when baseHash matches and rejects when stale", () => {
    const p = join(base, "sub", "w.txt"); writeFileSync(p, "one");
    const h = readFileAt(p).hash;
    expect(writeFileAt(p, "two", h).hash).toBe(readFileAt(p).hash);   // fresh write ok
    expect(() => writeFileAt(p, "three", h)).toThrow("changed on disk"); // h is now stale
  });
});

describe("createEntryAt / renameEntryAt / deleteEntryAt (write mutations)", () => {
  // cold-review must-fix: creating through a symlinked basename must NOT follow
  // the link (O_EXCL) — base/escape → outside/secret.txt, so wx must EEXIST.
  it("createEntryAt refuses a basename that is an existing symlink out of root (O_EXCL)", () => {
    expect(() => createEntryAt(base, "escape", "file")).toThrow();
  });
  it("deleteEntryAt refuses a non-empty directory (ENOTEMPTY)", () => {
    expect(() => deleteEntryAt(join(base, "sub"))).toThrow();   // sub contains a.txt
  });
  it("creates, renames, then deletes a file", () => {
    createEntryAt(base, "t11.txt", "file");
    expect(existsSync(join(base, "t11.txt"))).toBe(true);
    renameEntryAt(join(base, "t11.txt"), base, "t11b.txt");
    expect(existsSync(join(base, "t11.txt"))).toBe(false);
    expect(existsSync(join(base, "t11b.txt"))).toBe(true);
    deleteEntryAt(join(base, "t11b.txt"));
    expect(existsSync(join(base, "t11b.txt"))).toBe(false);
  });
  it("createEntryAt rejects a bad basename and a duplicate name", () => {
    for (const bad of ["", "a/b", "..", "x\0y"]) expect(() => createEntryAt(base, bad, "file")).toThrow();
    expect(() => createEntryAt(base, "b.md", "file")).toThrow();   // EEXIST
  });
  it("renameEntryAt refuses to clobber an existing dest (no follow)", () => {
    expect(() => renameEntryAt(join(base, "b.md"), base, "bin")).toThrow();
  });
});

// The Root wrappers delegate to the *At cores through the allowlist gate. Their
// happy path is covered by the *At tests above (real fixture); here we prove the
// gate: a rel that escapes the root is refused BEFORE any fs mutation runs.
describe("Root wrappers gate the real allowlist", () => {
  it("resolveExisting canonicalizes a root itself", () => {
    expect(resolveExisting("repos", "")).toBe(realpathSync(ROOTS.repos));
  });
  it("createEntry refuses a parent dir that escapes the root", () => {
    expect(() => createEntry("repos", "../../../etc", "jax-x.txt", "file")).toThrow();
  });
  it("uploadFile refuses a parent dir that escapes the root", () => {
    expect(() => uploadFile("repos", "../../../etc", "jax-x", Buffer.from("x"))).toThrow();
  });
  it("deleteEntry refuses a target outside the root", () => {
    expect(() => deleteEntry("repos", "../../../etc")).toThrow();
  });
  it("renameEntry refuses a source outside the root", () => {
    expect(() => renameEntry("repos", "../../../etc", "x")).toThrow();
  });
});

describe("imageMime", () => {
  it("maps image extensions case-insensitively and rejects everything else", () => {
    expect(imageMime("shot.png")).toBe("image/png");
    expect(imageMime("dir/photo.JPG")).toBe("image/jpeg");
    expect(imageMime("a.jpeg")).toBe("image/jpeg");
    expect(imageMime("anim.gif")).toBe("image/gif");
    expect(imageMime("pic.webp")).toBe("image/webp");
    expect(imageMime("icon.svg")).toBe("image/svg+xml");
    expect(imageMime("notes.md")).toBeNull();
    expect(imageMime("archive.png.zip")).toBeNull();
    expect(imageMime("noext")).toBeNull();
  });
});

describe("mediaMime", () => {
  it("maps pdf/audio/video extensions case-insensitively and rejects everything else", () => {
    expect(mediaMime("doc.pdf")).toEqual({ mime: "application/pdf", kind: "pdf" });
    expect(mediaMime("doc.PDF")).toEqual({ mime: "application/pdf", kind: "pdf" });
    expect(mediaMime("song.mp3")).toEqual({ mime: "audio/mpeg", kind: "audio" });
    expect(mediaMime("song.m4a")).toEqual({ mime: "audio/mp4", kind: "audio" });
    expect(mediaMime("song.wav")).toEqual({ mime: "audio/wav", kind: "audio" });
    expect(mediaMime("song.ogg")).toEqual({ mime: "audio/ogg", kind: "audio" });
    expect(mediaMime("song.aac")).toEqual({ mime: "audio/aac", kind: "audio" });
    expect(mediaMime("clip.mp4")).toEqual({ mime: "video/mp4", kind: "video" });
    expect(mediaMime("clip.webm")).toEqual({ mime: "video/webm", kind: "video" });
    expect(mediaMime("clip.mov")).toEqual({ mime: "video/quicktime", kind: "video" });
    expect(mediaMime("notes.md")).toBeNull();
    expect(mediaMime("page.html")).toBeNull(); // html never touches raw — spec §5 item 10
    expect(mediaMime("shot.png")).toBeNull(); // images stay on imageMime, not mediaMime
  });
});

describe("relativeWithin (Phase 3 spec §11, Decision 18 — the inverse of resolveWithin)", () => {
  it("an absolute path inside root returns the root-relative rel; the root itself returns \"\"", () => {
    const root = mkdtempSync(join(tmpdir(), "jax-relwithin-"));
    mkdirSync(join(root, "a", "b"), { recursive: true });
    writeFileSync(join(root, "a", "b", "c.md"), "x");
    expect(relativeWithin(root, join(root, "a", "b", "c.md"))).toBe(join("a", "b", "c.md"));
    expect(relativeWithin(root, root)).toBe("");
  });

  it("a same-prefix sibling directory (repos-evil) is never mistaken for inside root", () => {
    const parent = mkdtempSync(join(tmpdir(), "jax-relwithin-"));
    const root = join(parent, "repos");
    const evil = join(parent, "repos-evil");
    mkdirSync(root);
    mkdirSync(evil);
    writeFileSync(join(evil, "x.md"), "x");
    expect(relativeWithin(root, join(evil, "x.md"))).toBeNull();
  });

  it("a symlink escaping root resolves outside and returns null; a missing path returns null", () => {
    const parent = mkdtempSync(join(tmpdir(), "jax-relwithin-"));
    const root = join(parent, "repos");
    const outside = join(parent, "outside");
    mkdirSync(root);
    mkdirSync(outside);
    writeFileSync(join(outside, "secret.md"), "x");
    symlinkSync(join(outside, "secret.md"), join(root, "link.md"));
    expect(relativeWithin(root, join(root, "link.md"))).toBeNull();
    expect(relativeWithin(root, join(root, "nope.md"))).toBeNull();
  });
});

describe("splitTotalCaps", () => {
  it("floor-divides evenly with no remainder, and count 1 returns the whole total unchanged", () => {
    expect(splitTotalCaps({ pathsCap: 100, visitCap: 200, timeMs: 40 }, 4)).toEqual([
      { pathsCap: 25, visitCap: 50, timeMs: 10 },
      { pathsCap: 25, visitCap: 50, timeMs: 10 },
      { pathsCap: 25, visitCap: 50, timeMs: 10 },
      { pathsCap: 25, visitCap: 50, timeMs: 10 },
    ]);
    const total = { pathsCap: 20_000, visitCap: 50_000, timeMs: 2_000 };
    expect(splitTotalCaps(total, 1)).toEqual([total]);
  });

  it("folds the remainder into the first share only, and clamps every slice to a minimum of 1", () => {
    const result = splitTotalCaps({ pathsCap: 10, visitCap: 10, timeMs: 10 }, 3);
    expect(result[0]).toEqual({ pathsCap: 4, visitCap: 4, timeMs: 4 }); // floor(10/3)=3, remainder=1 -> 3+1
    expect(result[1]).toEqual({ pathsCap: 3, visitCap: 3, timeMs: 3 });
    expect(result[2]).toEqual({ pathsCap: 3, visitCap: 3, timeMs: 3 });

    const tiny = splitTotalCaps({ pathsCap: 2, visitCap: 2, timeMs: 2 }, 5);
    expect(tiny).toHaveLength(5);
    expect(tiny.every((c) => c.pathsCap >= 1 && c.visitCap >= 1 && c.timeMs >= 1)).toBe(true);
  });
});

describe("planSlices", () => {
  it("gives the root bucket the remainder share and every subdir the same base share", () => {
    const plan = planSlices(["a", "b", "c"], { pathsCap: 10, visitCap: 10, timeMs: 10 });
    // sliceCount = 4 -> floor(10/4)=2, remainder=2 -> root bucket 2+2=4
    expect(plan.rootBucket).toEqual({ pathsCap: 4, visitCap: 4, timeMs: 4 });
    expect(plan.subdirs.get("a")).toEqual({ pathsCap: 2, visitCap: 2, timeMs: 2 });
    expect(plan.subdirs.get("b")).toEqual({ pathsCap: 2, visitCap: 2, timeMs: 2 });
    expect(plan.subdirs.get("c")).toEqual({ pathsCap: 2, visitCap: 2, timeMs: 2 });
  });

  it("degenerates to the whole total for a root with no subdirectories (boundary, spec §8)", () => {
    const total = { pathsCap: 20_000, visitCap: 50_000, timeMs: 2_000 };
    const plan = planSlices([], total);
    expect(plan.rootBucket).toEqual(total);
    expect(plan.subdirs.size).toBe(0);
  });

  it("does NOT slice for exactly one top-level subdirectory — root bucket AND the one child both get the whole, unhalved total (round 1 F2)", () => {
    const total = { pathsCap: 20_000, visitCap: 50_000, timeMs: 2_000 };
    const plan = planSlices(["only"], total);
    expect(plan.rootBucket).toEqual(total);
    expect(plan.subdirs.get("only")).toEqual(total);
  });
});

describe("listRepoNames (spec §5)", () => {
  it("lists top-level repo directories, excluding EXCLUDES entries", () => {
    const names = listRepoNames();
    expect(names).toContain("jax-os"); // this checkout — always present
    expect(names).not.toContain("node_modules");
    expect(names.every((n) => typeof n === "string" && n.length > 0)).toBe(true);
  });
});

describe("parseScope (spec §5)", () => {
  it("defaults null/empty/'all' to {kind:'all'}", () => {
    expect(parseScope(null)).toEqual({ kind: "all" });
    expect(parseScope("")).toEqual({ kind: "all" });
    expect(parseScope("all")).toEqual({ kind: "all" });
  });
  it("'vault' parses to {kind:'vault'}", () => expect(parseScope("vault")).toEqual({ kind: "vault" }));
  it("'repo:<name>' parses to {kind:'repo', name} for an existing, non-excluded top-level dir", () => {
    expect(parseScope("repo:jax-os")).toEqual({ kind: "repo", name: "jax-os" });
  });
  it("rejects every traversal shape BEFORE any filesystem call (round-4 F3)", () => {
    for (const bad of ["repo:foo/..", "repo:..", "repo:a/b", "repo:.", "repo:", "repo:a\\b"]) {
      expect(parseScope(bad)).toBeNull();
    }
  });
  it("rejects a well-shaped name that is not an existing top-level repo dir", () => {
    expect(parseScope("repo:definitely-not-a-real-repo-xyz")).toBeNull();
  });
  it("rejects any other unrecognized string", () => expect(parseScope("bogus")).toBeNull());
});

describe("searchByName (scope-aware, spec §6)", () => {
  it("'all' still searches both roots — the fair-walk path is untouched by this refactor", async () => {
    const result = await searchByName("package.json", { kind: "all" }, undefined);
    expect(result.hits.some((h) => h.root === "repos")).toBe(true);
  });
  it("'vault' walks only the vault root", async () => {
    const result = await searchByName("nonexistent-query-xyz", { kind: "vault" }, undefined);
    expect(result).toEqual({ hits: [], truncated: false });
  });
  it("'repo:<name>' walks only that repo, hits' rel prefixed '<name>/'", async () => {
    const result = await searchByName("package.json", { kind: "repo", name: "jax-os" }, undefined);
    expect(result.hits.length).toBeGreaterThan(0);
    expect(result.hits.every((h) => h.root === "repos" && h.rel.startsWith("jax-os/"))).toBe(true);
    expect(result.hits.some((h) => h.rel === "jax-os/package.json")).toBe(true);
  });
});

describe("gateVaultRoot / gateVaultScope (spec decision 16)", () => {
  it("refuses only root=vault when the vault is unusable, passes every other root", () => {
    expect(gateVaultRoot("vault", false)).toBe(true);
    expect(gateVaultRoot("repos", false)).toBe(false);
    expect(gateVaultRoot("vault", true)).toBe(false);
  });
  it("refuses scope=vault, narrows scope=all to reposOnly, passes a repo scope through (either way)", () => {
    expect(gateVaultScope({ kind: "vault" }, false)).toBe("refuse");
    expect(gateVaultScope({ kind: "vault" }, true)).toEqual({ kind: "vault" });
    expect(gateVaultScope({ kind: "all" }, false)).toEqual({ kind: "reposOnly" });
    expect(gateVaultScope({ kind: "all" }, true)).toEqual({ kind: "all" });
    expect(gateVaultScope({ kind: "repo", name: "jax-os" }, false)).toEqual({ kind: "repo", name: "jax-os" });
  });
});

describe("buildIndexUncached — reposOnly scope (spec decision 16, F2)", () => {
  it("walks ROOTS.repos only, never touching ROOTS.vault (the repos half 'all' already uses)", async () => {
    const result = await buildIndexUncached({ kind: "reposOnly" });
    expect(result.data.every((h) => h.root === "repos")).toBe(true);
    expect(result.truncatedRoots).not.toContain("vault");
  });
});

describe("buildIndex / buildIndexUncached (scope-aware, per-scope cache, spec §6)", () => {
  it("'repo:<name>' walks only that repo at the full index budget, hits prefixed", async () => {
    const result = await buildIndexUncached({ kind: "repo", name: "jax-os" });
    expect(result.data.length).toBeGreaterThan(0);
    expect(result.data.every((h) => h.root === "repos" && h.rel.startsWith("jax-os/"))).toBe(true);
  });
  it("'vault' walks only the vault root", async () => {
    expect((await buildIndexUncached({ kind: "vault" })).data.every((h) => h.root === "vault")).toBe(true);
  });
  it("caches per scope key — a cached 'vault' result is NOT reused for 'all' but IS reused for a repeat 'vault' call within TTL", async () => {
    const now = Date.now();
    const vault1 = await buildIndex({ kind: "vault" }, now);
    expect(await buildIndex({ kind: "vault" }, now + 1000)).toBe(vault1); // identity check
    const all1 = await buildIndex({ kind: "all" }, now + 1000);
    expect(all1).not.toBe(vault1);
    expect(all1.data.some((h) => h.root === "repos")).toBe(true);
  });
});
