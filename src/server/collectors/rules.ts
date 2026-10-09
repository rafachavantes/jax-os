// Jax Rules collector (spec C.2). Shared pure normalization/composition/
// status logic lives in ../../lib/rules.ts (zero Node imports — needed
// client-side too). This file holds the server-only pieces: the SHA-256
// disk-hash helper (Task 1) below, then the impure disk read for GET
// (Task 3) and the impure disk write for apply (Task 4) are appended below
// in their own sections, matching src/server/collectors/tools.ts's layout.
import { closeSync, fchmodSync, lstatSync, mkdirSync, openSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { randomBytes } from "node:crypto";
import { homedir } from "node:os";
import { dirname, join, sep } from "node:path";
import { hashBytes } from "./files";
import { ABSENT_HASH, APP_KEYS, compose, computeStatus, type AppKey, type RuleDocs, type RuleSourceRevisions, type RuleStatus } from "../../lib/rules";
import type { SeedSources } from "../db/rules";

export type AppRuleInfo =
  | { path: string; status: RuleStatus; disk: string | null; diskHash: string }
  | { path: string; error: string };

export type RulesData = {
  canonical: string;
  exceptions: Record<AppKey, string>;
  apps: Record<AppKey, AppRuleInfo>;
  // Per-stored-source revision (Task 1 §4): SHA-256 over each exact DB string,
  // so the client can bind a save/preview to the source it actually read.
  sourceRevisions: RuleSourceRevisions;
};

// Raw-bytes hash (spec §4: "no normalization; same helper in GET and
// apply"). Reuses files.ts's sha256 helper — ponytail: don't reimplement
// hashing, files.ts already has exactly this.
// Accepts the Buffer read straight from disk (the ONLY correct hash input:
// invalid UTF-8 re-encodes differently after a string round-trip) or a
// string for test fixtures/valid-UTF-8 convenience.
export function hashOrAbsent(diskRaw: Buffer | string | null): string {
  if (diskRaw === null) return ABSENT_HASH;
  return hashBytes(typeof diskRaw === "string" ? Buffer.from(diskRaw, "utf8") : diskRaw);
}

// SHA-256 over the exact stored source string (Task 1 §4). Shared by the DB
// write path (setRuleDoc) and the GET response's per-slot revisions so a
// revision round-trips only when the bytes match.
export function sourceRevision(content: string): string {
  return hashBytes(Buffer.from(content, "utf8"));
}

// ---- Seed-source reader (Task 2; hard rule 2: fs access lives in collectors) ----

function readSeedOrNull(path: string): string | null {
  try {
    return readFileSync(path, "utf8");
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code === "ENOENT") return null;
    throw e;
  }
}

// The bundled starter rules that ship with the app (D1/D2). Resolved from the
// repo root, same "app-bundled asset" assumption workflow-spawn.ts /
// inventory.ts already make for their scripts.
const STARTER_DIR = join(process.cwd(), "workflow", "templates", "global-rules");

// The bundle (always present — an ENOENT here is a packaging bug) plus the
// installer's own files, still optional (D3 paths). Passed into
// seedRuleDocsIfNeeded by getDb(); tests inject fixture strings instead.
export function readSeedSources(env: NodeJS.ProcessEnv = process.env): SeedSources {
  const paths = rulePaths(env);
  return {
    starter: {
      global: readFileSync(join(STARTER_DIR, "canonical.md"), "utf8"),
      claude: readFileSync(join(STARTER_DIR, "claude.md"), "utf8"),
      codex: readFileSync(join(STARTER_DIR, "codex.md"), "utf8"),
      opencode: readFileSync(join(STARTER_DIR, "opencode.md"), "utf8"),
    },
    existing: {
      claudeRaw: readSeedOrNull(paths.claude),
      codexRaw: readSeedOrNull(paths.codex),
    },
  };
}

// ---- Impure: disk read for GET (Task 3) ----

function isEnoent(e: unknown): boolean {
  return typeof e === "object" && e !== null && (e as NodeJS.ErrnoException).code === "ENOENT";
}
function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

// The three target paths (never client-supplied, never derived from a
// request body; the apply route's `app` field only selects among these three
// fixed shapes). `env` mirrors codexSocketPath(env): a test seam only, never
// client input — reads CLAUDE_CONFIG_DIR/CODEX_HOME exactly like Codex does.
export function rulePaths(env: NodeJS.ProcessEnv = process.env): Record<AppKey, string> {
  const claudeHome = env.CLAUDE_CONFIG_DIR || join(homedir(), ".claude");
  const codexHome = env.CODEX_HOME || join(homedir(), ".codex");
  return {
    claude: join(claudeHome, "CLAUDE.md"),
    codex: join(codexHome, "AGENTS.md"),
    opencode: join(homedir(), ".config", "opencode", "AGENTS.md"),
  };
}

// Display form for the GET response (spec §4: "path: string (display, ~-prefixed)").
export function displayPath(absPath: string): string {
  const home = homedir();
  return absPath === home || absPath.startsWith(home + sep) ? "~" + absPath.slice(home.length) : absPath;
}

export type RawRead = { raw: string | null; buf: Buffer | null } | { error: string };

// lstat-first read (spec §4): ENOENT ⇒ missing (raw: null). A symlink at the
// target — dangling or not — is an ERROR, never "missing": GET must not
// silently follow a symlink and report a misleading synced/drifted status
// for content that isn't really the managed file. Any other stat/read
// failure (EACCES…) is also an error, isolated per app by the caller.
export function readRuleFileRaw(absPath: string): RawRead {
  let st;
  try {
    st = lstatSync(absPath);
  } catch (e) {
    if (isEnoent(e)) return { raw: null, buf: null };
    return { error: errMsg(e) };
  }
  if (st.isSymbolicLink()) return { error: "target is a symlink" };
  try {
    const buf = readFileSync(absPath); // Buffer — raw disk bytes (the hash/backup input); decoded separately for the UI
    return { raw: buf.toString("utf8"), buf };
  } catch (e) {
    return { error: errMsg(e) };
  }
}

export type RulesFsDeps = { readRuleFile: (app: AppKey) => RawRead };
const systemRulesFsDeps: RulesFsDeps = { readRuleFile: (app) => readRuleFileRaw(rulePaths()[app]) };

// Assembles the §4 GET shape from already-fetched DB slots (docs, read by
// the route from src/server/db/rules.ts — this collector never touches
// better-sqlite3) plus a live disk read per app, isolating each app's read
// failure independently (spec §4 per-app isolation, same pattern as
// tools.ts's getToolsData/safeBuild).
export function getRulesData(docs: RuleDocs, deps: RulesFsDeps = systemRulesFsDeps): RulesData {
  const canonical = docs.global;
  const exceptions: Record<AppKey, string> = { claude: docs.claude, codex: docs.codex, opencode: docs.opencode };
  const apps = {} as Record<AppKey, AppRuleInfo>;
  const sourceRevisions: RuleSourceRevisions = {
    global: sourceRevision(docs.global),
    claude: sourceRevision(docs.claude),
    codex: sourceRevision(docs.codex),
    opencode: sourceRevision(docs.opencode),
  };
  for (const app of APP_KEYS) {
    const expected = compose(exceptions[app], canonical);
    const path = displayPath(rulePaths()[app]);
    const result = deps.readRuleFile(app);
    if ("error" in result) {
      apps[app] = { path, error: result.error };
      continue;
    }
    apps[app] = {
      path,
      status: computeStatus(result.raw, expected),
      disk: result.raw,
      diskHash: hashOrAbsent(result.buf),
    };
  }
  return { canonical, exceptions, apps, sourceRevisions };
}

// ---- Impure: disk write for apply (Task 4) ----

class SymlinkRefusedError extends Error {}

// Symlink policy (spec §5.2, no-follow): walk the target's FULL absolute
// path, lstat-ing every path component that currently exists. Any existing
// component that is a symlink — the target itself or a parent dir — refuses
// the whole apply; a component that doesn't exist yet is fine (it gets
// created below). ponytail: this walks from the filesystem root rather than
// scoping strictly to $HOME as spec §5.2's prose describes — checking a
// handful of extra always-real system dirs (/, /home, /home/rafa) costs
// nothing, is a strict superset of the required check, and keeps this
// function fully path-injectable for tmp-dir tests instead of hardcoding
// $HOME into the walk itself.
function assertNoSymlinkInPath(absPath: string): void {
  const parts = absPath.split(sep).filter(Boolean);
  let current: string = sep;
  for (const part of parts) {
    current = join(current, part);
    let st;
    try {
      st = lstatSync(current);
    } catch (e) {
      if (isEnoent(e)) continue; // component doesn't exist yet — created below
      throw e; // EACCES/ENOTDIR/…: fail CLOSED — never skip a component the no-follow check couldn't inspect
    }
    if (st.isSymbolicLink()) throw new SymlinkRefusedError(current);
  }
}

// `rename` is injectable (optional, defaults to renameSync) so the cleanup
// test can force a failure AFTER the temp file exists — the only way to
// exercise the unlink path without defeating the wx protections.
export type ApplyDeps = { now: () => Date; randomHex: () => string; rename?: (from: string, to: string) => void };
const systemApplyDeps: ApplyDeps = { now: () => new Date(), randomHex: () => randomBytes(3).toString("hex") };

export type ApplyOutcome =
  | { ok: true; bytes: number; backup: string | null }
  | { ok: false; reason: "stale" }
  | { ok: false; reason: "symlink" }
  | { ok: false; reason: "fs-error"; error: string };

// Per-target reentrancy guard (spec §5.3: "module-level per-app lock").
// Node's single-threaded, no-await-inside-this-function execution model
// already prevents true interleaving, but the explicit flag documents and
// enforces the invariant directly rather than relying on that implicit
// guarantee — cheap, and it's exactly what spec §7 test 6 exercises ("at the
// locked-function boundary").
const busy = new Set<string>();
// Test-only seam for the lock branch (§7 test 6): pre-adding a target here
// is the only way to observe the re-entrancy refusal from fully synchronous code.
export const _applyBusyForTests = busy;

// The whole guard→backup→temp→rename sequence (spec §5.2-5.6), fully
// synchronous end to end — no await anywhere in this function. `targetPath`
// is an explicit parameter (not looked up internally from rulePaths()) so
// this function stays fully testable against tmp dirs; the ROUTE is what
// maps the validated `app` enum to rulePaths()[app] (D4 — never client input).
export function applyRuleFile(
  targetPath: string,
  expectedContent: string,
  clientHash: string,
  deps: ApplyDeps = systemApplyDeps,
): ApplyOutcome {
  if (busy.has(targetPath)) return { ok: false, reason: "stale" };
  busy.add(targetPath);
  try {
    try {
      assertNoSymlinkInPath(targetPath);
    } catch (e) {
      if (e instanceof SymlinkRefusedError) return { ok: false, reason: "symlink" };
      return { ok: false, reason: "fs-error", error: errMsg(e) };
    }

    let currentBuf: Buffer | null;
    try {
      currentBuf = readFileSync(targetPath); // Buffer — raw bytes, same hash input as GET
    } catch (e) {
      if (!isEnoent(e)) return { ok: false, reason: "fs-error", error: errMsg(e) };
      currentBuf = null;
    }
    // Race guard (spec §5.4): re-read + re-hash inside this synchronous
    // section, compare to the client's last-seen hash. Mismatch → refused,
    // nothing touched yet (no mkdir/backup/write has happened above this line).
    if (hashOrAbsent(currentBuf) !== clientHash) return { ok: false, reason: "stale" };

    const dir = dirname(targetPath);
    const now = deps.now();
    const randomHex = deps.randomHex();

    // Preserve the target's existing mode when it exists; a brand-new rules
    // file is owner-only. Backups/temp are always 0600.
    let targetMode = 0o600;
    if (currentBuf !== null) {
      try {
        targetMode = lstatSync(targetPath).mode & 0o777;
      } catch {
        targetMode = 0o600; // raced away between read and here: fail closed to owner-only
      }
    }

    try {
      mkdirSync(dir, { recursive: true }); // missing file → parent dir created too (spec §5.6)

      let backup: string | null = null;
      if (currentBuf !== null) {
        backup = `.bak-jaxrules-${now.toISOString()}-${randomHex}`;
        writeFileSync(join(dir, backup), currentBuf, { flag: "wx", mode: 0o600 }); // O_EXCL, owner-only; Buffer preserves raw bytes
      }

      const tempPath = join(dir, `.tmp-jaxrules-${randomHex}`);
      let tempCreated = false;
      try {
        // Exclusive open FIRST, ownership marked immediately: if the write
        // itself fails afterwards (I/O, ENOSPC), the temp is still ours to
        // clean up. writeFileSync on a raw "wx" path would set ownership too
        // late for that window.
        const fd = openSync(tempPath, "wx", 0o600);
        tempCreated = true;
        try {
          writeFileSync(fd, expectedContent);
          fchmodSync(fd, targetMode); // never widen a restrictive target mode
        } finally {
          closeSync(fd);
        }
        (deps.rename ?? renameSync)(tempPath, targetPath);
      } catch (e) {
        // Only unlink a temp THIS invocation created: an EEXIST from the "wx"
        // create means the colliding name is someone else's file (or symlink) —
        // deleting it would defeat the exclusive-create protection.
        if (tempCreated) {
          try {
            unlinkSync(tempPath);
          } catch {
            /* best-effort cleanup */
          }
        }
        throw e;
      }

      return { ok: true, bytes: Buffer.byteLength(expectedContent, "utf8"), backup };
    } catch (e) {
      return { ok: false, reason: "fs-error", error: errMsg(e) };
    }
  } finally {
    busy.delete(targetPath);
  }
}
