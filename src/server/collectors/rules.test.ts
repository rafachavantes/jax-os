import { existsSync, lstatSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from "vitest";
import { ABSENT_HASH, type RuleDocs } from "../../lib/rules";
import { _applyBusyForTests, applyRuleFile, getRulesData, hashOrAbsent, readRuleFileRaw, readSeedSources, rulePaths } from "./rules";

describe("rulePaths (env overrides, D3)", () => {
  it("defaults to ~/.claude, ~/.codex, ~/.config/opencode when no env vars are set", () => {
    expect(rulePaths({} as unknown as NodeJS.ProcessEnv)).toEqual({
      claude: join(homedir(), ".claude", "CLAUDE.md"),
      codex: join(homedir(), ".codex", "AGENTS.md"),
      opencode: join(homedir(), ".config", "opencode", "AGENTS.md"),
    });
  });

  it("CLAUDE_CONFIG_DIR overrides only the claude path", () => {
    const paths = rulePaths({ CLAUDE_CONFIG_DIR: "/tmp/x" } as unknown as NodeJS.ProcessEnv);
    expect(paths.claude).toBe("/tmp/x/CLAUDE.md");
    expect(paths.codex).toBe(join(homedir(), ".codex", "AGENTS.md"));
    expect(paths.opencode).toBe(join(homedir(), ".config", "opencode", "AGENTS.md"));
  });

  it("CODEX_HOME overrides only the codex path", () => {
    const paths = rulePaths({ CODEX_HOME: "/tmp/y" } as unknown as NodeJS.ProcessEnv);
    expect(paths.codex).toBe("/tmp/y/AGENTS.md");
    expect(paths.claude).toBe(join(homedir(), ".claude", "CLAUDE.md"));
    expect(paths.opencode).toBe(join(homedir(), ".config", "opencode", "AGENTS.md"));
  });
});

describe("readSeedSources", () => {
  it("reads the four bundled starter files from workflow/templates/global-rules", () => {
    const empty = mkdtempSync(join(tmpdir(), "jax-rules-starter-"));
    try {
      const sources = readSeedSources({ CLAUDE_CONFIG_DIR: empty, CODEX_HOME: empty } as unknown as NodeJS.ProcessEnv);
      const dir = join(process.cwd(), "workflow", "templates", "global-rules");
      expect(sources.starter.global).toBe(readFileSync(join(dir, "canonical.md"), "utf8"));
      expect(sources.starter.claude).toBe(readFileSync(join(dir, "claude.md"), "utf8"));
      expect(sources.starter.codex).toBe(readFileSync(join(dir, "codex.md"), "utf8"));
      expect(sources.starter.opencode).toBe(readFileSync(join(dir, "opencode.md"), "utf8"));
    } finally {
      rmSync(empty, { recursive: true, force: true });
    }
  });

  it("still returns null for claudeRaw/codexRaw when the existing files are absent (ENOENT)", () => {
    const dir = mkdtempSync(join(tmpdir(), "jax-rules-seed-"));
    try {
      const sources = readSeedSources({ CLAUDE_CONFIG_DIR: dir, CODEX_HOME: dir } as unknown as NodeJS.ProcessEnv);
      expect(sources.existing.claudeRaw).toBeNull();
      expect(sources.existing.codexRaw).toBeNull();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("hashOrAbsent (§7 test 2)", () => {
  it("absent sentinel for null disk", () => {
    expect(hashOrAbsent(null)).toBe(ABSENT_HASH);
  });
  it("real sha256 hex for a real (even empty) string, distinct from the absent sentinel", () => {
    const h = hashOrAbsent("");
    expect(h).not.toBe(ABSENT_HASH);
    expect(h).toMatch(/^[0-9a-f]{64}$/);
  });
  it("two different disk contents hash differently", () => {
    expect(hashOrAbsent("a\n")).not.toBe(hashOrAbsent("b\n"));
  });
  it("hashes RAW bytes, not normalized text — trailing-newline differences change the hash (spec §4: 'no normalization')", () => {
    expect(hashOrAbsent("a\n")).not.toBe(hashOrAbsent("a\n\n"));
  });
});

describe("readRuleFileRaw — lstat-first missing/symlink/error distinction (§7 tests 2, 5)", () => {
  let base: string;
  beforeAll(() => {
    base = mkdtempSync(join(tmpdir(), "jax-rules-read-"));
    writeFileSync(join(base, "real.md"), "hello\n");
    symlinkSync(join(base, "does-not-exist.md"), join(base, "dangling.md"));
    symlinkSync(join(base, "real.md"), join(base, "valid-symlink.md"));
  });
  afterAll(() => rmSync(base, { recursive: true, force: true }));

  it("missing file: raw null, not an error", () => {
    expect(readRuleFileRaw(join(base, "nope.md"))).toEqual({ raw: null, buf: null });
  });
  it("real file: raw content", () => {
    expect(readRuleFileRaw(join(base, "real.md"))).toEqual({ raw: "hello\n", buf: Buffer.from("hello\n") });
  });
  it("dangling symlink at the target: error, never missing (spec §4)", () => {
    const r = readRuleFileRaw(join(base, "dangling.md"));
    expect("error" in r).toBe(true);
  });
  it("a valid (non-dangling) symlink at the target is ALSO an error, never silently followed", () => {
    const r = readRuleFileRaw(join(base, "valid-symlink.md"));
    expect("error" in r).toBe(true);
  });
  it("a missing PARENT directory is also just 'missing', not an error", () => {
    expect(readRuleFileRaw(join(base, "no-such-dir", "f.md"))).toEqual({ raw: null, buf: null });
  });
  it("invalid UTF-8 on disk: diskHash is the hash of the RAW bytes, not of the re-encoded string", () => {
    const bad = Buffer.from([0x68, 0x69, 0xff, 0xfe, 0x0a]); // "hi" + invalid UTF-8 + LF
    writeFileSync(join(base, "bad-utf8.md"), bad);
    const r = readRuleFileRaw(join(base, "bad-utf8.md"));
    if ("error" in r) throw new Error("expected a read, got error");
    expect(hashOrAbsent(r.buf)).toBe(hashOrAbsent(bad));
    expect(hashOrAbsent(r.buf)).not.toBe(hashOrAbsent(r.raw)); // the decoded string re-encodes differently
  });
});

describe("getRulesData — per-app isolation + status wiring (§7 tests 3, 11; §4)", () => {
  const docs: RuleDocs = { global: "canon\n", claude: "", codex: "", opencode: "" };

  it("one app's read error leaves the other two apps' data intact (route isolation)", () => {
    const data = getRulesData(docs, {
      readRuleFile: (app) => (app === "codex" ? { error: "boom" } : { raw: "canon\n", buf: Buffer.from("canon\n") }),
    });
    expect(data.apps.codex).toMatchObject({ error: "boom" });
    expect(data.apps.claude).toMatchObject({ status: "synced" });
    expect(data.apps.opencode).toMatchObject({ status: "synced" });
    expect(data.canonical).toBe("canon\n");
  });

  it("empty file is drifted or synced by content, never missing — diskHash differs from the absent sentinel", () => {
    const data = getRulesData(docs, { readRuleFile: () => ({ raw: "", buf: Buffer.alloc(0) }) });
    if ("error" in data.apps.claude) throw new Error("expected a read, got error");
    expect(data.apps.claude.status).not.toBe("missing"); // "" normalizes to "\n"
    expect(data.apps.claude.diskHash).not.toBe("absent");
  });

  it("missing file reports status missing, disk null, diskHash 'absent'", () => {
    const data = getRulesData(docs, { readRuleFile: () => ({ raw: null, buf: null }) });
    expect(data.apps.claude).toMatchObject({ status: "missing", disk: null, diskHash: "absent" });
  });

  it("exceptions object mirrors the injected docs, keyed per app", () => {
    const withExc: RuleDocs = { global: "canon\n", claude: "", codex: "exc\n", opencode: "" };
    const data = getRulesData(withExc, { readRuleFile: () => ({ raw: null, buf: null }) });
    expect(data.exceptions).toEqual({ claude: "", codex: "exc\n", opencode: "" });
  });
});

describe("applyRuleFile — race guard + write (§7 tests 3, 4, 6)", () => {
  let base: string;
  beforeEach(() => {
    base = mkdtempSync(join(tmpdir(), "jax-rules-apply-"));
  });
  afterEach(() => rmSync(base, { recursive: true, force: true }));

  it("stale diskHash: refused, file untouched", () => {
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "old\n");
    const outcome = applyRuleFile(target, "new\n", "wrong-hash");
    expect(outcome).toEqual({ ok: false, reason: "stale" });
    expect(readFileSync(target, "utf8")).toBe("old\n");
  });

  it("'absent' is accepted only when the file is really missing — creates it, including the parent dir", () => {
    const target = join(base, "new-dir", "AGENTS.md"); // parent doesn't exist yet
    const outcome = applyRuleFile(target, "content\n", ABSENT_HASH);
    expect(outcome).toMatchObject({ ok: true, backup: null });
    expect(readFileSync(target, "utf8")).toBe("content\n");
  });

  it("'absent' is refused when the file actually exists (stale)", () => {
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "already here\n");
    const outcome = applyRuleFile(target, "new\n", ABSENT_HASH);
    expect(outcome).toEqual({ ok: false, reason: "stale" });
  });

  it("existing file: exclusive backup created with the logged name, content atomic-replaced", () => {
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "old\n");
    const outcome = applyRuleFile(target, "new\n", hashOrAbsent("old\n"));
    expect(outcome.ok).toBe(true);
    if (!outcome.ok) throw new Error("unreachable");
    expect(outcome.backup).toMatch(/^\.bak-jaxrules-.+/);
    expect(readFileSync(join(base, outcome.backup as string), "utf8")).toBe("old\n");
    expect(readFileSync(target, "utf8")).toBe("new\n");
  });

  it("a pre-existing file at the computed backup name is never overwritten — collision refuses the apply", () => {
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "old\n");
    const fixedNow = () => new Date("2026-08-30T00:00:00.000Z");
    const fixedRandom = () => "aaaaaa";
    const collisionName = `.bak-jaxrules-${fixedNow().toISOString()}-aaaaaa`;
    writeFileSync(join(base, collisionName), "pre-existing, must survive\n");
    const outcome = applyRuleFile(target, "new\n", hashOrAbsent("old\n"), { now: fixedNow, randomHex: fixedRandom });
    expect(outcome.ok).toBe(false);
    expect(readFileSync(join(base, collisionName), "utf8")).toBe("pre-existing, must survive\n");
    expect(readFileSync(target, "utf8")).toBe("old\n"); // original also untouched
  });

  it("no backup file for a missing target", () => {
    const target = join(base, "AGENTS.md");
    const outcome = applyRuleFile(target, "content\n", ABSENT_HASH);
    expect(outcome).toMatchObject({ ok: true, backup: null });
    expect(readdirSync(base).some((n) => n.startsWith(".bak-jaxrules-"))).toBe(false);
  });

  it("temp file is cleaned up when rename fails AFTER the temp was created (injected rename failure)", () => {
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "old\n");
    const outcome = applyRuleFile(target, "new\n", hashOrAbsent("old\n"), {
      now: () => new Date("2026-08-30T12:00:00.000Z"),
      randomHex: () => "abc123",
      rename: () => {
        throw new Error("injected rename failure");
      },
    });
    expect(outcome.ok).toBe(false);
    expect(readdirSync(base).some((n) => n.startsWith(".tmp-jaxrules-"))).toBe(false); // temp removed
    expect(readFileSync(target, "utf8")).toBe("old\n"); // target untouched
    expect(readdirSync(base).some((n) => n.startsWith(".bak-jaxrules-"))).toBe(true); // backup (made before temp) remains
  });

  it("a PRE-EXISTING file at the temp name: wx refuses AND the colliding file is NOT deleted", () => {
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "old\n");
    const tempName = join(base, ".tmp-jaxrules-abc123");
    writeFileSync(tempName, "someone else's file\n");
    const outcome = applyRuleFile(target, "new\n", hashOrAbsent("old\n"), {
      now: () => new Date("2026-08-30T12:00:00.000Z"),
      randomHex: () => "abc123", // fixed random forces the collision
    });
    expect(outcome.ok).toBe(false);
    expect(readFileSync(tempName, "utf8")).toBe("someone else's file\n"); // NOT unlinked — we never created it
    expect(readFileSync(target, "utf8")).toBe("old\n");
  });

  it("a PRE-EXISTING SYMLINK at the temp name: wx refuses and the symlink survives", () => {
    const victim = join(base, "victim.md");
    writeFileSync(victim, "victim\n");
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "old\n");
    const tempName = join(base, ".tmp-jaxrules-abc123");
    symlinkSync(victim, tempName);
    const outcome = applyRuleFile(target, "new\n", hashOrAbsent("old\n"), {
      now: () => new Date("2026-08-30T12:00:00.000Z"),
      randomHex: () => "abc123",
    });
    expect(outcome.ok).toBe(false);
    expect(lstatSync(tempName).isSymbolicLink()).toBe(true); // survives
    expect(readFileSync(victim, "utf8")).toBe("victim\n");
  });

  it("applying twice with the same original diskHash — the second call is refused by the STALE-HASH guard (sequential; the lock branch has its own test below)", () => {
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "old\n");
    const hash = hashOrAbsent("old\n");
    const first = applyRuleFile(target, "new\n", hash);
    expect(first.ok).toBe(true);
    const second = applyRuleFile(target, "new-again\n", hash); // same (now-stale) hash
    expect(second).toEqual({ ok: false, reason: "stale" });
  });

  it("re-entering the same target while the lock is held is refused AT THE LOCK BRANCH (§7 test 6)", () => {
    const target = join(base, "AGENTS.md");
    writeFileSync(target, "old\n");
    _applyBusyForTests.add(target); // simulate a concurrent in-flight apply on this target
    try {
      const outcome = applyRuleFile(target, "new\n", hashOrAbsent("old\n"));
      expect(outcome).toEqual({ ok: false, reason: "stale" });
      expect(readFileSync(target, "utf8")).toBe("old\n"); // nothing touched
    } finally {
      _applyBusyForTests.delete(target);
    }
  });
});

describe("applyRuleFile — symlink policy (§7 test 5)", () => {
  let base: string;
  beforeEach(() => {
    base = mkdtempSync(join(tmpdir(), "jax-rules-symlink-"));
  });
  afterEach(() => rmSync(base, { recursive: true, force: true }));

  it("a symlinked target: refused, no write, no backup", () => {
    const real = join(base, "real.md");
    writeFileSync(real, "old\n");
    const target = join(base, "link.md");
    symlinkSync(real, target);
    const outcome = applyRuleFile(target, "new\n", hashOrAbsent("old\n"));
    expect(outcome).toEqual({ ok: false, reason: "symlink" });
    expect(readFileSync(real, "utf8")).toBe("old\n");
    expect(readdirSync(base).some((n) => n.startsWith(".bak-jaxrules-"))).toBe(false);
  });

  it("a symlinked PARENT dir: refused before any write", () => {
    const realDir = join(base, "real-dir");
    mkdirSync(realDir);
    const linkedDir = join(base, "linked-dir");
    symlinkSync(realDir, linkedDir);
    const target = join(linkedDir, "AGENTS.md");
    const outcome = applyRuleFile(target, "new\n", ABSENT_HASH);
    expect(outcome).toEqual({ ok: false, reason: "symlink" });
    expect(existsSync(join(realDir, "AGENTS.md"))).toBe(false);
  });

  it("a dangling symlink at the target: refused, same as a live symlink", () => {
    const target = join(base, "dangling.md");
    symlinkSync(join(base, "does-not-exist.md"), target);
    const outcome = applyRuleFile(target, "new\n", ABSENT_HASH);
    expect(outcome).toEqual({ ok: false, reason: "symlink" });
  });

  it("a non-directory path component (ENOTDIR from lstat): fs-error, fail closed, nothing written", () => {
    const plainFile = join(base, "plain-file");
    writeFileSync(plainFile, "x\n");
    const target = join(plainFile, "nested", "AGENTS.md"); // lstat on plain-file/nested → ENOTDIR
    const outcome = applyRuleFile(target, "new\n", ABSENT_HASH);
    expect(outcome.ok).toBe(false);
    expect(outcome).toMatchObject({ reason: "fs-error" });
  });
});
