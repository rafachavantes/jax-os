import { describe, expect, it } from "vitest";
import { beforeEach, afterEach } from "vitest";
import { existsSync, mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { TRASH_DIR, compactTimestamp, parseTrashRel, strictRel, trashEntryAt, restoreTrashEntryAt, sweepTrashAt, removeExpiredTrashEntryAt } from "./files";

describe("compactTimestamp", () => {
  it("strips hyphens, colons, and milliseconds from an ISO date into the compact UTC grammar", () => {
    const d = new Date("2026-09-18T18:30:00.123Z");
    expect(compactTimestamp(d)).toBe("20260918T183000Z");
    expect(compactTimestamp(d)).toMatch(/^\d{8}T\d{6}Z$/);
  });
});

describe("strictRel", () => {
  it("rejects empty/./../leading-slash/backslash/NUL segments and accepts a normal multi-segment rel (round 1 plan review F2)", () => {
    for (const bad of ["a/../b", "/abs", "a\\b", "a//b", ".", "..", "", "a/./b", "a\0b"]) {
      expect(strictRel(bad)).toBe(false);
    }
    expect(strictRel("docs/plans/old-plan.md")).toBe(true);
    expect(strictRel("", true)).toBe(true); // zip's "whole root" case allows empty
  });
});

describe("parseTrashRel", () => {
  it("parses a well-formed trash path into its timestamp and original rel", () => {
    expect(parseTrashRel(`${TRASH_DIR}/20260918T183000Z/docs/plans/old-plan.md`)).toEqual({
      timestamp: "20260918T183000Z",
      rel: "docs/plans/old-plan.md",
    });
    expect(parseTrashRel(`${TRASH_DIR}/20260918T183000Z/a.txt`)).toEqual({
      timestamp: "20260918T183000Z",
      rel: "a.txt",
    });
  });

  it("accepts a collision-suffixed timestamp and rejects a zero or non-numeric suffix (round 4 F2)", () => {
    expect(parseTrashRel(`${TRASH_DIR}/20260919T060000Z-1/a.txt`)).toEqual({
      timestamp: "20260919T060000Z-1",
      rel: "a.txt",
    });
    expect(parseTrashRel(`${TRASH_DIR}/20260919T060000Z-0/a.txt`)).toBeNull();
    expect(parseTrashRel(`${TRASH_DIR}/20260919T060000Z-x/a.txt`)).toBeNull();
  });

  it("rejects every malformed shape without touching the filesystem", () => {
    const bad = [
      "",
      `${TRASH_DIR}/20260918T183000Z`,                      // round 3 F2: no original-path segment
      `${TRASH_DIR}//a.txt`,                                 // empty segment
      `${TRASH_DIR}/./a.txt`,                                 // "." segment (too short anyway, but exercise the guard)
      `${TRASH_DIR}/20260918T183000Z/./a.txt`,                // "." segment
      `${TRASH_DIR}/20260918T183000Z/../a.txt`,               // ".." segment
      `/${TRASH_DIR}/20260918T183000Z/a.txt`,                 // leading slash
      `${TRASH_DIR}/20260918T183000Z/a.txt\0`,                // NUL byte
      `${TRASH_DIR}/20260918T183000Z/a\\b.txt`,               // backslash
      `not-jax-trash/20260918T183000Z/a.txt`,                 // wrong first segment
      `${TRASH_DIR}/2026-09-18T18-30-00-000Z/a.txt`,          // round 3 F1: old hyphenated shape rejected
      `${TRASH_DIR}/20260918T1830Z/a.txt`,                    // malformed timestamp (too short)
    ];
    for (const value of bad) expect(parseTrashRel(value)).toBeNull();
  });
});

describe("trashEntryAt / restoreTrashEntryAt", () => {
  let base: string, outside: string;
  const fixed = new Date("2026-09-18T18:30:00.000Z");

  beforeEach(() => {
    base = mkdtempSync(join(tmpdir(), "jax-trash-"));
    outside = mkdtempSync(join(tmpdir(), "jax-trash-outside-"));
    mkdirSync(join(base, "docs", "plans"), { recursive: true });
    writeFileSync(join(base, "docs", "plans", "old-plan.md"), "keep me");
    writeFileSync(join(base, "a.txt"), "hi");
    mkdirSync(join(base, "full"));
    writeFileSync(join(base, "full", "child.txt"), "nested");
    writeFileSync(join(outside, "secret.txt"), "SECRET");
    symlinkSync(join(outside, "secret.txt"), join(base, "escape.txt"));
    symlinkSync(outside, join(base, "escapedir"));
  });
  afterEach(() => {
    rmSync(base, { recursive: true, force: true });
    rmSync(outside, { recursive: true, force: true });
  });

  it("renames a nested file into .jax-trash/<timestamp>/<rel>, then restores it byte-identical", () => {
    const { trashRel } = trashEntryAt(base, "docs/plans/old-plan.md", fixed);
    expect(trashRel).toBe(`${TRASH_DIR}/20260918T183000Z/docs/plans/old-plan.md`);
    expect(existsSync(join(base, "docs", "plans", "old-plan.md"))).toBe(false);
    expect(readFileSync(join(base, trashRel), "utf8")).toBe("keep me");
    const { restoredRel } = restoreTrashEntryAt(base, trashRel);
    expect(restoredRel).toBe("docs/plans/old-plan.md");
    expect(readFileSync(join(base, "docs", "plans", "old-plan.md"), "utf8")).toBe("keep me");
  });

  it("renames a non-empty folder whole (O(1) — no recursive copy) and restores it byte-identical", () => {
    const { trashRel } = trashEntryAt(base, "full", fixed);
    expect(existsSync(join(base, "full"))).toBe(false);
    expect(readFileSync(join(base, trashRel, "child.txt"), "utf8")).toBe("nested");
    const { restoredRel } = restoreTrashEntryAt(base, trashRel);
    expect(restoredRel).toBe("full");
    expect(readFileSync(join(base, "full", "child.txt"), "utf8")).toBe("nested");
  });

  it("rejects a/../b, /abs, a\\b, a//b, ., and empty before any fs access (round 1 plan review F2)", () => {
    for (const bad of ["a/../b", "/abs", "a\\b", "a//b", ".", ""]) {
      expect(() => trashEntryAt(base, bad, fixed)).toThrow("outside allowlist");
    }
    expect(existsSync(join(base, "a.txt"))).toBe(true); // untouched — rejected before any rename
  });

  it("refuses a symlinked file or a symlinked directory as a delete source — never renamed-by-target", () => {
    expect(() => trashEntryAt(base, "escape.txt", fixed)).toThrow("symlink source");
    expect(() => trashEntryAt(base, "escapedir", fixed)).toThrow("symlink source");
    expect(existsSync(join(base, "escape.txt"))).toBe(true);
    expect(existsSync(join(outside, "secret.txt"))).toBe(true);
  });

  it("refuses a symlink placed directly inside .jax-trash on restore — never restored-by-target", () => {
    const tsDir = join(base, TRASH_DIR, "20260918T183000Z");
    mkdirSync(tsDir, { recursive: true });
    symlinkSync(join(outside, "secret.txt"), join(tsDir, "sneaky.txt"));
    expect(() => restoreTrashEntryAt(base, `${TRASH_DIR}/20260918T183000Z/sneaky.txt`)).toThrow("symlink source");
  });

  it("gives two deletes of the same rel within the same second distinct trash entries, both restorable (round 4 F2)", () => {
    const { trashRel: first } = trashEntryAt(base, "a.txt", fixed);
    expect(first).toBe(`${TRASH_DIR}/20260918T183000Z/a.txt`);
    writeFileSync(join(base, "a.txt"), "second version");
    const { trashRel: second } = trashEntryAt(base, "a.txt", fixed);
    expect(second).toBe(`${TRASH_DIR}/20260918T183000Z-1/a.txt`);
    expect(readFileSync(join(base, first), "utf8")).toBe("hi");
    expect(readFileSync(join(base, second), "utf8")).toBe("second version");

    const { restoredRel: r1 } = restoreTrashEntryAt(base, first);
    expect(r1).toBe("a.txt");
    expect(readFileSync(join(base, "a.txt"), "utf8")).toBe("hi");
    rmSync(join(base, "a.txt"));
    const { restoredRel: r2 } = restoreTrashEntryAt(base, second);
    expect(r2).toBe("a.txt");
    expect(readFileSync(join(base, "a.txt"), "utf8")).toBe("second version");
  });

  it("refuses a trashRel failing parseTrashRel before any fs call, and refuses to clobber an existing destination", () => {
    expect(() => restoreTrashEntryAt(base, "not-a-trash-path")).toThrow("outside allowlist");
    const { trashRel } = trashEntryAt(base, "a.txt", fixed);
    writeFileSync(join(base, "a.txt"), "someone recreated me");
    expect(() => restoreTrashEntryAt(base, trashRel)).toThrow("destination exists");
  });
});

describe("sweepTrashAt", () => {
  it("lists only entries whose .jax-trash timestamp dir is past the 7-day TTL, one entry per top-level child, WITHOUT touching the filesystem (round 1 plan review F1 — sweepTrashAt is a pure lister)", () => {
    const base2 = mkdtempSync(join(tmpdir(), "jax-sweep-"));
    try {
      const trashDir = join(base2, TRASH_DIR);
      mkdirSync(join(trashDir, "20200101T000000Z", "old"), { recursive: true });
      writeFileSync(join(trashDir, "20200101T000000Z", "old", "a.txt"), "x");
      writeFileSync(join(trashDir, "20200101T000000Z", "b.txt"), "y");
      const recentTs = compactTimestamp(new Date());
      mkdirSync(join(trashDir, recentTs), { recursive: true });
      writeFileSync(join(trashDir, recentTs, "keep.txt"), "z");

      const expired = sweepTrashAt(base2, new Date());
      expect(expired.map((s) => s.trashRel).sort()).toEqual([
        `${TRASH_DIR}/20200101T000000Z/b.txt`,
        `${TRASH_DIR}/20200101T000000Z/old`,
      ]);
      expect(expired.every((s) => typeof s.absPath === "string" && s.absPath.startsWith(trashDir))).toBe(true);
      // Nothing removed yet — listing is pure; removal is the caller's job via removeExpiredTrashEntryAt.
      expect(existsSync(join(trashDir, "20200101T000000Z", "b.txt"))).toBe(true);
      expect(existsSync(join(trashDir, recentTs, "keep.txt"))).toBe(true);
    } finally { rmSync(base2, { recursive: true, force: true }); }
  });

  it("ages a collision-suffixed entry by its stamp, stripping the suffix (round 4 F2)", () => {
    const base5 = mkdtempSync(join(tmpdir(), "jax-sweep-suffix-"));
    try {
      const trashDir = join(base5, TRASH_DIR);
      mkdirSync(join(trashDir, "20200101T000000Z-1"), { recursive: true });
      writeFileSync(join(trashDir, "20200101T000000Z-1", "b.txt"), "y");

      const expired = sweepTrashAt(base5, new Date());
      expect(expired.map((s) => s.trashRel)).toEqual([`${TRASH_DIR}/20200101T000000Z-1/b.txt`]);
    } finally { rmSync(base5, { recursive: true, force: true }); }
  });

  it("returns an empty array when the root has no .jax-trash directory yet", () => {
    const base3 = mkdtempSync(join(tmpdir(), "jax-sweep-empty-"));
    try {
      expect(sweepTrashAt(base3, new Date())).toEqual([]);
    } finally { rmSync(base3, { recursive: true, force: true }); }
  });
});

describe("removeExpiredTrashEntryAt", () => {
  it("removes one expired entry, and its now-empty parent timestamp dir once every sibling is gone (round 1 plan review F1)", () => {
    const base4 = mkdtempSync(join(tmpdir(), "jax-sweep-remove-"));
    try {
      const tsDir = join(base4, TRASH_DIR, "20200101T000000Z");
      mkdirSync(tsDir, { recursive: true });
      writeFileSync(join(tsDir, "a.txt"), "x");
      writeFileSync(join(tsDir, "b.txt"), "y");
      removeExpiredTrashEntryAt(join(tsDir, "a.txt"));
      expect(existsSync(join(tsDir, "a.txt"))).toBe(false);
      expect(existsSync(tsDir)).toBe(true); // b.txt still there — ENOTEMPTY swallowed
      removeExpiredTrashEntryAt(join(tsDir, "b.txt"));
      expect(existsSync(tsDir)).toBe(false); // last sibling gone — parent cleaned up too
    } finally { rmSync(base4, { recursive: true, force: true }); }
  });
});
