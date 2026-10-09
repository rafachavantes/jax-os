import {
  closeSync, constants, fchmodSync, fstatSync, fsyncSync, lstatSync, mkdirSync, openSync,
  readdirSync, readSync, realpathSync, renameSync, rmdirSync, rmSync, statSync, unlinkSync, writeFileSync,
  type Stats,
} from "node:fs";
import { opendir, realpath } from "node:fs/promises";
import { crc32 } from "node:zlib";
import { createHash, randomUUID } from "node:crypto";
import { dirname, join, sep } from "node:path";
// relative import: vitest resolves no @/ alias for value imports in tested server modules
import { reposRoot, vaultPath } from "../reposRoot";
import { MutationRejected } from "../../lib/mutationOutcome";
import { defaultGitStatusForDir, type FileGitStatus } from "./gitStatus";
import { cacheGet, cachePut } from "./linear";

// Frozen once, at this module's first import — matches how the old hardcoded constant behaved
// and how every downstream consumer of projects.ts's REPOS_ROOT already treats it
// (Global Constraints: "restart applies" in the literal sense, here only).
export const ROOTS: { repos: string; vault: string | null } = { repos: reposRoot(), vault: vaultPath() };
export type Root = keyof typeof ROOTS;

// A generic root-taking operation dereferencing a null (unconfigured) vault throws "disabled"
// — collectorResponse's existing catch (src/server/api.ts) turns that into exactly
// {ok:false,error:"disabled"}, status 200, no new response shape. The `integrations.vault`
// on/off UI gate is MOA-496's job (ownership matrix); this is just "vault not configured".
export function rootPath(root: Root): string {
  const path = ROOTS[root];
  if (path === null) throw new Error("disabled");
  return path;
}
// scope=all is a legitimate repos-only request too when vault is unconfigured — no refusal,
// just narrowing (spec: Null vault, "the same {kind:'reposOnly'} narrowing").
export function activeRoots(): { root: Root; path: string }[] {
  const entries: { root: Root; path: string }[] = [{ root: "repos", path: ROOTS.repos }];
  if (ROOTS.vault !== null) entries.push({ root: "vault", path: ROOTS.vault });
  return entries;
}

export type Entry = { name: string; isDir: boolean; mtime: string; size?: number; gitStatus: FileGitStatus | null };
export type DirListing = { dirs: Entry[]; files: Entry[] };
export type FileRead =
  | { binary: false; content: string; hash: string; size: number }
  | { binary: true; content: null; hash: string; size: number };

// ".jax-trash" is a literal here, not the TRASH_DIR constant defined later — a module-level const
// can't forward-reference one declared further down (TDZ); duplicating one short literal is
// cheaper than reordering every function this plan appends below EXCLUDES.
export const EXCLUDES: ReadonlySet<string> = new Set(["node_modules", ".git", ".next", "dist", "build", ".jax-trash"]);
export const READ_CAP = 2 * 1024 * 1024;
export const UPLOAD_CAP = 5 * 1024 * 1024;

// Image preview gate for /api/files/raw — extension → mime, null = not servable.
const IMAGE_MIME: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  gif: "image/gif",
  webp: "image/webp",
  svg: "image/svg+xml",
};
export function imageMime(rel: string): string | null {
  return IMAGE_MIME[rel.split(".").pop()?.toLowerCase() ?? ""] ?? null;
}

export const RAW_PDF_CAP = 20 * 1024 * 1024;
export const RAW_MEDIA_CAP = 64 * 1024 * 1024;
const PDF_MIME: Record<string, string> = { pdf: "application/pdf" };
const AUDIO_MIME: Record<string, string> = { mp3: "audio/mpeg", m4a: "audio/mp4", wav: "audio/wav", ogg: "audio/ogg", aac: "audio/aac" };
const VIDEO_MIME: Record<string, string> = { mp4: "video/mp4", webm: "video/webm", mov: "video/quicktime" };
// Separate from imageMime (never merged into one "rawMime") so raw/route.ts's existing image path,
// and its existing test's mock, stay byte-for-byte unchanged — this only adds new extensions.
export function mediaMime(rel: string): { mime: string; kind: "pdf" | "audio" | "video" } | null {
  const ext = rel.split(".").pop()?.toLowerCase() ?? "";
  if (PDF_MIME[ext]) return { mime: PDF_MIME[ext], kind: "pdf" };
  if (AUDIO_MIME[ext]) return { mime: AUDIO_MIME[ext], kind: "audio" };
  if (VIDEO_MIME[ext]) return { mime: VIDEO_MIME[ext], kind: "video" };
  return null;
}

export function isBinary(buf: Buffer): boolean {
  const n = Math.min(buf.length, 8192);
  for (let i = 0; i < n; i++) if (buf[i] === 0) return true;
  return false;
}

export function hashBytes(buf: Buffer): string {
  return createHash("sha256").update(buf).digest("hex");
}

export async function listDirAt(
  absDir: string,
  gitStatusFor: (absDir: string) => Promise<Map<string, FileGitStatus> | null> = defaultGitStatusForDir,
): Promise<DirListing> {
  const gitMap = await gitStatusFor(absDir);
  const dirs: Entry[] = [], files: Entry[] = [];
  for (const d of readdirSync(absDir, { withFileTypes: true })) {
    const abs = join(absDir, d.name);
    // cold review round 1 F1: statSync FOLLOWS a symlink and throws ENOENT for a dangling one; a
    // real ENOENT also fires for an entry removed between readdirSync and this line (race). Either
    // must skip that one entry, never abort the whole listing. lstatSync (which does NOT follow the
    // link) is used for a symlink so a dangling target can't throw at all.
    let stat;
    try {
      stat = d.isSymbolicLink() ? lstatSync(abs) : statSync(abs);
    } catch {
      continue;
    }
    const mtime = stat.mtime.toISOString();
    const gitStatus = gitMap?.get(d.name) ?? null;
    if (d.isDirectory()) { if (!EXCLUDES.has(d.name)) dirs.push({ name: d.name, isDir: true, mtime, gitStatus }); }
    else files.push({ name: d.name, isDir: false, mtime, size: stat.size, gitStatus });
  }
  const byName = (a: Entry, b: Entry) => a.name.localeCompare(b.name);
  return { dirs: dirs.sort(byName), files: files.sort(byName) };
}

const OPEN_READ = constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK;

function readBoundedAt(absPath: string, cap: number): { oversize: true; size: number } | { oversize: false; bytes: Buffer } {
  const fd = openSync(absPath, OPEN_READ);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile()) throw new Error("not a regular file");
    if (stat.size > cap) return { oversize: true, size: stat.size };
    const buffer = Buffer.alloc(cap + 1);
    let used = 0;
    while (used < buffer.length) {
      const n = readSync(fd, buffer, used, buffer.length - used, used);
      if (n === 0) break;
      used += n;
    }
    const sizeNow = fstatSync(fd).size;
    if (used > cap || sizeNow > cap) return { oversize: true, size: Math.max(used, sizeNow) };
    return { oversize: false, bytes: buffer.subarray(0, used) };
  } finally {
    closeSync(fd);
  }
}

export function readFileAt(absPath: string): FileRead {
  const result = readBoundedAt(absPath, READ_CAP);
  if (result.oversize) return { binary: true, content: null, hash: "", size: result.size };
  const hash = hashBytes(result.bytes);
  if (isBinary(result.bytes)) return { binary: true, content: null, hash, size: result.bytes.length };
  return { binary: false, content: result.bytes.toString("utf8"), hash, size: result.bytes.length };
}

export function readRawPreviewAt(absPath: string): Buffer {
  const result = readBoundedAt(absPath, UPLOAD_CAP);
  if (result.oversize) throw new Error("too large");
  return result.bytes;
}

function snapshotRegularFile(absPath: string): { stat: Stats; bytes: Buffer } {
  const listed = lstatSync(absPath);
  if (!listed.isFile()) throw new MutationRejected("not a regular file");
  const fd = openSync(absPath, constants.O_RDWR | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const stat = fstatSync(fd);
    if (!stat.isFile()) throw new MutationRejected("not a regular file");
    if (stat.size > READ_CAP) throw new MutationRejected("file too large");
    const buffer = Buffer.alloc(READ_CAP + 1);
    let used = 0;
    while (used < buffer.length) {
      const n = readSync(fd, buffer, used, buffer.length - used, used);
      if (n === 0) break;
      used += n;
    }
    if (used > READ_CAP || fstatSync(fd).size > READ_CAP)
      throw new MutationRejected("file too large");
    return { stat, bytes: buffer.subarray(0, used) };
  } finally {
    closeSync(fd);
  }
}

export function writeFileAt(absPath: string, content: string, baseHash: string): { hash: string } {
  const replacement = Buffer.from(content, "utf8");
  if (replacement.length > READ_CAP) throw new MutationRejected("file too large");
  const replacementHash = hashBytes(replacement);

  let tempOwned = false;
  let tempFd: number | undefined;
  let parentFd: number | undefined;
  let anchoredTemp: string | undefined;
  let tempDev: number | undefined;
  let tempIno: number | undefined;
  let failed = false;
  let failure: unknown;
  let cleanupFailed = false;
  let committed = false;
  try {
    if (realpathSync(absPath) !== absPath) throw new MutationRejected("changed on disk");
    const parentPath = dirname(absPath);
    const parent = statSync(parentPath);
    const initial = snapshotRegularFile(absPath);
    const initialHash = hashBytes(initial.bytes);
    if (initialHash !== baseHash) throw new MutationRejected("changed on disk");

    parentFd = openSync(parentPath, constants.O_RDONLY | constants.O_DIRECTORY | constants.O_NOFOLLOW);
    const openedParent = fstatSync(parentFd);
    if (!openedParent.isDirectory() || openedParent.dev !== parent.dev || openedParent.ino !== parent.ino)
      throw new MutationRejected("changed on disk");
    const tempName = ".jax-save-" + randomUUID();
    anchoredTemp = `/proc/self/fd/${parentFd}/${tempName}`;

    tempFd = openSync(anchoredTemp, "wx", 0o600);
    tempOwned = true;
    const tempStat = fstatSync(tempFd);
    tempDev = tempStat.dev;
    tempIno = tempStat.ino;
    writeFileSync(tempFd, replacement);
    fchmodSync(tempFd, initial.stat.mode & 0o7777);
    fsyncSync(tempFd);
    const closeTemp = tempFd;
    tempFd = undefined;
    closeSync(closeTemp);

    if (realpathSync(parentPath) !== parentPath) throw new MutationRejected("changed on disk");
    const parentNow = statSync(parentPath);
    if (parentNow.dev !== parent.dev || parentNow.ino !== parent.ino)
      throw new MutationRejected("changed on disk");

    let current: { stat: Stats; bytes: Buffer };
    try {
      current = snapshotRegularFile(absPath);
    } catch {
      throw new MutationRejected("changed on disk");
    }
    if (
      current.stat.dev !== initial.stat.dev ||
      current.stat.ino !== initial.stat.ino ||
      current.stat.mode !== initial.stat.mode ||
      hashBytes(current.bytes) !== initialHash
    ) {
      throw new MutationRejected("changed on disk");
    }
    let destListed: Stats;
    try {
      destListed = lstatSync(absPath);
    } catch {
      throw new MutationRejected("changed on disk");
    }
    if (!destListed.isFile() || destListed.dev !== initial.stat.dev || destListed.ino !== initial.stat.ino)
      throw new MutationRejected("changed on disk");
    try {
      if (realpathSync(parentPath) !== parentPath) throw new MutationRejected("changed on disk");
      const parentFinal = statSync(parentPath);
      if (parentFinal.dev !== parent.dev || parentFinal.ino !== parent.ino)
        throw new MutationRejected("changed on disk");
    } catch {
      throw new MutationRejected("changed on disk");
    }

    let tempListed: Stats;
    try {
      tempListed = lstatSync(anchoredTemp);
    } catch {
      throw new MutationRejected("changed on disk");
    }
    if (!tempListed.isFile() || tempListed.dev !== tempDev || tempListed.ino !== tempIno)
      throw new MutationRejected("changed on disk");

    renameSync(anchoredTemp, absPath);
    committed = true;
    tempOwned = false;
  } catch (e) {
    failed = true;
    failure = e;
  } finally {
    if (tempFd !== undefined) {
      const fd = tempFd;
      tempFd = undefined;
      try { closeSync(fd); } catch { cleanupFailed = true; }
    }
    if (tempOwned && anchoredTemp) {
      try {
        const listed = lstatSync(anchoredTemp);
        if (!listed.isFile() || listed.dev !== tempDev || listed.ino !== tempIno) cleanupFailed = true;
        else unlinkSync(anchoredTemp);
      } catch { cleanupFailed = true; }
    }
    if (parentFd !== undefined) {
      const fd = parentFd;
      parentFd = undefined;
      try { closeSync(fd); }
      catch {
        if (!committed) cleanupFailed = true;
        else console.error("[files] atomic save completed; parent descriptor close failed");
      }
    }
  }
  if (cleanupFailed)
    throw new MutationRejected("atomic save failed; original file was not replaced; temporary cleanup failed");
  if (failed) {
    if (failure instanceof MutationRejected) throw failure;
    throw new MutationRejected("atomic save failed; original file was not replaced");
  }
  return { hash: replacementHash };
}

// Canonicalize an EXISTING target and assert it stays within `base`.
// realpathSync resolves symlinks + `..` on both sides → a symlink escaping the
// root is rejected. join(base, rel) neutralizes an absolute `rel`.
export function resolveWithin(base: string, rel: string): string {
  const canonicalBase = realpathSync(base);
  const canonical = realpathSync(join(base, rel));
  if (canonical !== canonicalBase && !canonical.startsWith(canonicalBase + sep)) {
    throw new Error("outside allowlist");
  }
  return canonical;
}

// Inverse of resolveWithin (spec §11, Decision 18): canonicalizes an absolute path and returns it
// as a root-relative string ONLY if it still resolves inside `root` — same realpathSync +
// startsWith(base + sep) containment check, run backwards. Never throws: a missing file, an
// outside path, a same-prefix sibling ("repos-evil"), or a symlink escape all return null
// (best-effort enrichment, AGENTS.md error contract — never a thrown poll).
export function relativeWithin(root: string, absolutePath: string): string | null {
  let canonicalRoot: string;
  let canonical: string;
  try {
    canonicalRoot = realpathSync(root);
    canonical = realpathSync(absolutePath);
  } catch {
    return null;
  }
  if (canonical === canonicalRoot) return "";
  if (!canonical.startsWith(canonicalRoot + sep)) return null;
  return canonical.slice(canonicalRoot.length + sep.length);
}

// For a NEW target: canonicalize the (existing) parent, validate the basename.
// Callers writing here MUST use flag:"wx" (O_EXCL) so a symlink at `basename`
// is refused, not followed.
export function resolveNewWithin(base: string, relParentDir: string, basename: string): string {
  const parent = resolveWithin(base, relParentDir);
  assertBasename(basename);
  return join(parent, basename);
}

export function resolveExisting(root: Root, rel: string): string {
  return resolveWithin(rootPath(root), rel);
}
export function resolveNew(root: Root, relParentDir: string, basename: string): string {
  return resolveNewWithin(rootPath(root), relParentDir, basename);
}

// Validate a user-supplied basename before it touches the fs (no path parts,
// no traversal, no NUL). Shared by create + rename.
function assertBasename(basename: string): void {
  if (!basename || basename.includes("/") || basename.includes("..") || basename.includes("\0")) {
    throw new Error("bad name");
  }
}

// Create a NEW entry under an already-resolved (in-allowlist) parent dir.
// O_EXCL semantics: mkdirSync / writeFileSync{flag:"wx"} throw EEXIST on an
// existing name — INCLUDING a symlink, which is refused (never followed).
export function createEntryAt(absDir: string, basename: string, kind: "file" | "folder"): void {
  assertBasename(basename);
  const target = join(absDir, basename);
  if (kind === "folder") mkdirSync(target);
  else writeFileSync(target, "", { flag: "wx" });
}

// Delete an already-resolved (in-allowlist) entry. rmdirSync throws ENOTEMPTY
// on a non-empty dir (surfaced to the UI as "directory not empty").
export function deleteEntryAt(absPath: string): void {
  if (statSync(absPath).isDirectory()) rmdirSync(absPath);
  else unlinkSync(absPath);
}

// Rename within the allowlist (source + dest parent both pre-resolved). lstat
// (not stat) so a symlink at the dest counts as "exists" — no clobber, no follow.
export function renameEntryAt(absFrom: string, absToParent: string, basename: string): void {
  assertBasename(basename);
  const dest = join(absToParent, basename);
  let exists = true;
  try { lstatSync(dest); } catch { exists = false; }
  if (exists) throw new Error("destination exists");
  renameSync(absFrom, dest);
}

export function createEntry(root: Root, relParentDir: string, basename: string, kind: "file" | "folder"): void {
  createEntryAt(resolveExisting(root, relParentDir), basename, kind);
}
// Write uploaded bytes as a NEW file. resolveNew canonicalizes the parent +
// validates the basename; flag:"wx" (O_EXCL) refuses an existing/symlinked
// target (EEXIST) — never follows, never clobbers.
export function uploadFile(root: Root, relParentDir: string, basename: string, bytes: Buffer): void {
  writeFileSync(resolveNew(root, relParentDir, basename), bytes, { flag: "wx" });
}
export function deleteEntry(root: Root, rel: string): void {
  deleteEntryAt(resolveExisting(root, rel));
}
// relTo = "parentRel/basename": resolve source + dest PARENT (both in-allowlist).
export function renameEntry(root: Root, relFrom: string, relTo: string): void {
  const slash = relTo.lastIndexOf("/");
  const parentRel = slash >= 0 ? relTo.slice(0, slash) : "";
  const basename = slash >= 0 ? relTo.slice(slash + 1) : relTo;
  renameEntryAt(resolveExisting(root, relFrom), resolveExisting(root, parentRel), basename);
}

export function listDir(root: Root, rel: string): Promise<DirListing> { return listDirAt(resolveExisting(root, rel)); }
export function readFile(root: Root, rel: string): FileRead { return readFileAt(resolveExisting(root, rel)); }
export function writeFile(root: Root, rel: string, content: string, baseHash: string): { hash: string } {
  return writeFileAt(resolveExisting(root, rel), content, baseHash);
}

export type SearchHit = { root: Root; rel: string; name: string };
export type SearchResult = { hits: SearchHit[]; truncated: boolean };
export const SEARCH_CAP = 200;
export const SEARCH_MAX_DEPTH = 12;
export const SEARCH_VISIT_CAP = 20000;
export const SEARCH_TIME_MS = 5000;

export type RepoRef = { kind: "repo"; name: string } | { kind: "vault" };
export type SearchScope = RepoRef | { kind: "all" };

// Decision 16's narrowing target: the routes hand the collectors the scope AFTER gateVaultScope,
// which may be the internal-only `reposOnly` (vault disabled + scope=all). Never part of the
// public SearchScope client-facing consumers (src/lib/filesWorkspace.ts) already use.
export type GatedSearchScope = SearchScope | { kind: "reposOnly" };

// Decision 16: root=vault/scope=vault are refused outright when the vault is unusable (off OR
// unconfigured — both share the "disabled" answer rootPath already throws); scope=all simply
// narrows to reposOnly, a legitimate request even with no vault (decision 6: a request that
// would have worked anyway never paints a false warning). One helper pair, reused by all 14
// Files routes, never reimplemented per route.
export function gateVaultRoot(root: string, vaultUsable: boolean): boolean {
  return root === "vault" && !vaultUsable;
}

export function gateVaultScope(scope: SearchScope, vaultUsable: boolean): GatedSearchScope | "refuse" {
  if (vaultUsable) return scope;
  if (scope.kind === "vault") return "refuse";
  if (scope.kind === "all") return { kind: "reposOnly" };
  return scope;
}

// §5: dirs only, EXCLUDES-filtered — same predicate listDirAt (line 90) applies to a directory
// listing; no status.md read (unlike getProjects, projects.ts:264 — this is a name list, not a
// project scan).
export function listRepoNames(): string[] {
  return readdirSync(ROOTS.repos, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !EXCLUDES.has(d.name))
    .map((d) => d.name);
}

// §5: shape checked BEFORE any filesystem call, existence checked after — closes the gap
// resolveWithin alone leaves open (resolveWithin(ROOTS.repos, "foo/..") canonicalizes to
// ROOTS.repos itself, which does not throw). listRepoNames() already excludes EXCLUDES entries,
// so containment in it also covers "not in EXCLUDES" in one check.
export function parseScope(raw: string | null): SearchScope | null {
  if (raw === null || raw === "" || raw === "all") return { kind: "all" };
  if (raw === "vault") return { kind: "vault" };
  if (!raw.startsWith("repo:")) return null;
  const name = raw.slice(5);
  if (!name || name.includes("/") || name.includes("\\") || name === "." || name === "..") return null;
  if (!listRepoNames().includes(name)) return null;
  return { kind: "repo", name };
}

// --- Fairness slicing (spec 2026-09-19 §5 #1-#14) ---------------------------------------------
// One shared formula for two splits this fix needs: within a root (root bucket + N subdirectory
// slices, §5 #14) and across roots for searchRoots (§5 #7) — same floor+remainder rule at both
// levels, only the count differs. Index 0 of the result always carries the remainder (the "first"
// share — the root bucket within a root, the first root across roots).
export type SliceCaps = { pathsCap: number; visitCap: number; timeMs: number };

export function splitTotalCaps(total: SliceCaps, count: number): SliceCaps[] {
  const div = (n: number) => Math.floor(n / count);
  const base: SliceCaps = { pathsCap: div(total.pathsCap), visitCap: div(total.visitCap), timeMs: div(total.timeMs) };
  const remainder: SliceCaps = {
    pathsCap: total.pathsCap - base.pathsCap * count,
    visitCap: total.visitCap - base.visitCap * count,
    timeMs: total.timeMs - base.timeMs * count,
  };
  const clamp = (c: SliceCaps): SliceCaps =>
    ({ pathsCap: Math.max(1, c.pathsCap), visitCap: Math.max(1, c.visitCap), timeMs: Math.max(1, c.timeMs) });
  const first = clamp({
    pathsCap: base.pathsCap + remainder.pathsCap,
    visitCap: base.visitCap + remainder.visitCap,
    timeMs: base.timeMs + remainder.timeMs,
  });
  return [first, ...Array.from({ length: count - 1 }, () => clamp(base))];
}

export type SlicePlan = { rootBucket: SliceCaps; subdirs: Map<string, SliceCaps> };

// Pure planner (§5 #1/#14): subdirNames is the caller's already-filtered top-level listing
// (non-excluded, non-symlink directories only — discovered by the walker itself; this function
// never touches the filesystem). Slice count = subdirNames.length + 1 (the root bucket), which
// always gets index 0's remainder-carrying share from splitTotalCaps.
//
// N <= 1 (zero or one top-level subdirectory) is NOT sliced — there is no sibling to protect from
// starvation, so splitting would only shrink the one child's cap for no reason (round 1 F2:
// splitTotalCaps(total, 2) used to hand the single child half the total, truncating a large child
// earlier than today's flat walk does). Root bucket and the (at most one) child both get the whole,
// unhalved `total` instead.
export function planSlices(subdirNames: readonly string[], total: SliceCaps): SlicePlan {
  if (subdirNames.length <= 1) {
    return { rootBucket: total, subdirs: new Map(subdirNames.map((name) => [name, total] as const)) };
  }
  const [rootBucket, ...subdirCaps] = splitTotalCaps(total, subdirNames.length + 1);
  const subdirs = new Map(subdirNames.map((name, i) => [name, subdirCaps[i]] as const));
  return { rootBucket, subdirs };
}

export function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  throw signal.reason !== undefined ? signal.reason : new DOMException("The operation was aborted.", "AbortError");
}

export async function searchRoots(
  roots: readonly { root: Root; path: string }[],
  query: string,
  signal?: AbortSignal,
): Promise<SearchResult> {
  const q = query.trim().toLowerCase();
  if (!q) return { hits: [], truncated: false };
  throwIfAborted(signal);
  // round 1 F1: an empty roots list has nothing to split a budget across — splitTotalCaps(total, 0)
  // divides by zero (Infinity/NaN caps). Same empty, complete result the old flat implementation
  // returned (its loop over `roots` simply never ran).
  if (roots.length === 0) return { hits: [], truncated: false };

  // §5 #7: split the call's total budget equally across roots FIRST (remainder to the first root —
  // same splitTotalCaps used within a root, one level up), so one root can no longer starve another.
  const perRoot = splitTotalCaps({ pathsCap: SEARCH_CAP, visitCap: SEARCH_VISIT_CAP, timeMs: SEARCH_TIME_MS }, roots.length);
  const hits: SearchHit[] = [];
  let truncated = false;
  const match = (rel: string) => rel.toLowerCase().includes(q);

  for (let i = 0; i < roots.length; i++) {
    const item = roots[i];
    throwIfAborted(signal);
    const rootBucketStart = performance.now(); // before THIS root's own realpath — mirrors walkAllPaths
    let realRoot: string;
    try {
      realRoot = await realpath(item.path);
    } catch {
      throwIfAborted(signal);
      throw new Error("file search unavailable");
    }
    throwIfAborted(signal);
    try {
      const result = await walkRootFair(item.root, realRoot, perRoot[i], rootBucketStart, match, signal);
      hits.push(...result.hits);
      if (result.truncated) truncated = true;
    } catch (err) {
      throwIfAborted(signal);
      throw new Error("file search unavailable");
    }
  }
  return { hits, truncated };
}

export function searchByName(query: string, scope: GatedSearchScope, signal?: AbortSignal): Promise<SearchResult> {
  if (scope.kind === "all") {
    return searchRoots(activeRoots(), query, signal);
  }
  if (scope.kind === "reposOnly") return searchRoots([{ root: "repos", path: ROOTS.repos }], query, signal);
  if (scope.kind === "vault") return searchRoots([{ root: "vault", path: rootPath("vault") }], query, signal);
  return searchRoots([{ root: "repos", path: resolveWithin(ROOTS.repos, scope.name) }], query, signal)
    .then((r) => ({ hits: r.hits.map((h) => ({ ...h, rel: `${scope.name}/${h.rel}` })), truncated: r.truncated }));
}

// --- Trash (§5 item 6) ---------------------------------------------------
export const TRASH_DIR = ".jax-trash";
// "-<n> collision suffix" (diff review 0e652b904cea F2) — appended only when the same stamp+rel
// destination is already occupied (two deletes of the same rel within one second).
const TRASH_TS_RE = /^\d{8}T\d{6}Z(-[1-9]\d*)?$/;

// "2026-09-18T18:30:00.123Z" -> "20260918T183000Z" — never the hyphenated extended form (round 3 F1).
export function compactTimestamp(d: Date = new Date()): string {
  return d.toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z");
}

// Strict segment validator (round 1 plan review F2) — rejects empty/"."/".."/leading-slash/
// backslash/NUL segments. validRel (src/server/inputLimits.ts) only bounds length and rejects NUL;
// it stays in place for that bound, this adds the segment-shape guarantee validRel never made.
// Defined once here, next to parseTrashRel, and reused by it below, by trashEntryAt/planZipAt, and
// by the delete/zip routes — every rel this build accepts is rejected at the segment level before
// any filesystem access, regardless of entry point.
export function strictRel(value: unknown, allowEmpty = false): value is string {
  if (typeof value !== "string" || value.includes("\0") || value.includes("\\") || value.startsWith("/"))
    return false;
  if (value === "") return allowEmpty;
  return value.split("/").every((s) => s !== "" && s !== "." && s !== "..");
}

// Dedicated parser (round 2 F2) — never validRel alone, which would accept a traversal-shaped rel.
// strictRel gates the segment shape; at least 3 segments; segment 1 must match the compact UTC
// grammar (round 3 F1/F2).
export function parseTrashRel(value: string): { timestamp: string; rel: string } | null {
  if (!strictRel(value)) return null;
  const segments = value.split("/");
  if (segments.length < 3) return null;
  if (segments[0] !== TRASH_DIR) return null;
  if (!TRASH_TS_RE.test(segments[1])) return null;
  return { timestamp: segments[1], rel: segments.slice(2).join("/") };
}

// Split a rel into parent+basename — same shape renameEntry() inlines once (files.ts:322-324),
// named here since Task 2 needs it three more times.
function splitParentBasename(rel: string): { parent: string; basename: string } {
  const slash = rel.lastIndexOf("/");
  return slash >= 0 ? { parent: rel.slice(0, slash), basename: rel.slice(slash + 1) } : { parent: "", basename: rel };
}

function existsAt(absPath: string): boolean {
  try { lstatSync(absPath); return true; }
  catch { return false; }
}

function ensureTrashRootAt(base: string): void {
  const trashAbs = join(base, TRASH_DIR);
  try { mkdirSync(trashAbs); }
  catch (e) { if ((e as NodeJS.ErrnoException).code !== "EEXIST") throw e; }
  // EEXIST-tolerant lazy init (unlike createEntryAt's deliberate wx/EEXIST-throws); still lstat to
  // refuse a pre-existing symlink here.
  if (!lstatSync(trashAbs).isDirectory()) throw new Error("outside allowlist");
}

// Delete -> trash. rel is gated by strictRel FIRST (round 1 plan review F2) — without it, a shape
// like "." or "a/.." would resolve (via splitParentBasename + resolveWithin's own join-normalization)
// to the ROOT itself, trash-renaming the whole allowlisted directory. The source is then lstat'd on
// its UN-resolved path (parent via resolveWithin, basename lstat'd directly) so a symlink at `rel` is
// refused, never renamed-by-target. A rename is O(1) regardless of folder size — no recursive copy.
export function trashEntryAt(base: string, rel: string, now: Date = new Date()): { trashRel: string } {
  if (!strictRel(rel)) throw new Error("outside allowlist");
  const { parent, basename } = splitParentBasename(rel);
  const parentAbs = resolveWithin(base, parent);
  const sourceAbs = join(parentAbs, basename);
  let sourceLstat: Stats;
  try { sourceLstat = lstatSync(sourceAbs); }
  catch { throw new Error("outside allowlist"); }
  if (sourceLstat.isSymbolicLink()) throw new Error("symlink source");

  ensureTrashRootAt(base);
  const stamp = compactTimestamp(now);
  let timestamp = stamp;
  let trashAbs = join(base, TRASH_DIR, timestamp, rel);
  // Same rel trashed twice within one second would otherwise reuse this exact destination and
  // renameSync would silently overwrite the earlier entry (diff review 0e652b904cea F2). Bump a
  // "-<n>" suffix on the stamp segment until the destination is free — first free wins.
  for (let n = 1; existsAt(trashAbs); n++) {
    timestamp = `${stamp}-${n}`;
    trashAbs = join(base, TRASH_DIR, timestamp, rel);
  }
  const trashRel = `${TRASH_DIR}/${timestamp}/${rel}`;
  mkdirSync(dirname(trashAbs), { recursive: true });
  renameSync(sourceAbs, trashAbs);
  return { trashRel };
}
export function trashEntry(root: Root, rel: string): { trashRel: string } {
  return trashEntryAt(rootPath(root), rel);
}

// Restore = rename back; original path recomputed from the trash path's OWN suffix — no manifest,
// no DB row. trashRel is rebuilt ONLY from parseTrashRel's parsed pieces, never the raw input.
export function restoreTrashEntryAt(base: string, rawTrashRel: string): { restoredRel: string } {
  const parsed = parseTrashRel(rawTrashRel);
  if (!parsed) throw new Error("outside allowlist");
  const trashRel = `${TRASH_DIR}/${parsed.timestamp}/${parsed.rel}`;

  const { parent: srcParentRel, basename: srcBasename } = splitParentBasename(trashRel);
  const srcParentAbs = resolveWithin(base, srcParentRel);
  let srcLstat: Stats;
  try { srcLstat = lstatSync(join(srcParentAbs, srcBasename)); }
  catch { throw new Error("outside allowlist"); }
  if (srcLstat.isSymbolicLink()) throw new Error("symlink source");

  const sourceAbs = resolveExisting_forRoot(base, trashRel);
  const { parent: destParentRel, basename: destBasename } = splitParentBasename(parsed.rel);
  // Dest parent must already exist — same resolveWithin renameEntry() uses (files.ts:325);
  // renameEntryAt validates destBasename itself, so resolveNewWithin here would be redundant.
  const destParentAbs = resolveWithin(base, destParentRel);
  renameEntryAt(sourceAbs, destParentAbs, destBasename); // "destination exists" if already occupied
  return { restoredRel: parsed.rel };
}
export function restoreTrashEntry(root: Root, trashRel: string): { restoredRel: string } {
  return restoreTrashEntryAt(rootPath(root), trashRel);
}

// Named alias so the call below reads as "resolveExisting (trash source)" per the guard matrix.
function resolveExisting_forRoot(base: string, rel: string): string {
  return resolveWithin(base, rel);
}

export const TRASH_TTL_MS = 7 * 24 * 60 * 60 * 1000;

// Age is read from the TIMESTAMP ENCODED IN THE DIRECTORY NAME, not its mtime — deterministic and
// exactly what a test can back-date.
function parseCompactTimestamp(ts: string): Date | null {
  if (!TRASH_TS_RE.test(ts)) return null;
  const stamp = ts.slice(0, ts.indexOf("-") === -1 ? ts.length : ts.indexOf("-")); // strip "-<n>" suffix
  const iso = `${stamp.slice(0, 4)}-${stamp.slice(4, 6)}-${stamp.slice(6, 8)}T${stamp.slice(9, 11)}:${stamp.slice(11, 13)}:${stamp.slice(13, 15)}Z`;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

// Opportunistic (no cron) — called after a successful trash-delete in the SAME root. PURE: only
// LISTS what's past the 7-day TTL, no fs mutation (round 1 plan review F1 — the old version deleted
// here and audited after the fact, so an audit-insert failure left a delete unrecorded). The caller
// removes each listed entry inside its OWN runMutation effect via removeExpiredTrashEntryAt below,
// so the pending audit row always lands before the rm. "entry" = one top-level child of a stale
// <timestamp> dir — the only granularity reachable without a manifest.
export function sweepTrashAt(base: string, now: Date = new Date()): { trashRel: string; absPath: string }[] {
  const trashAbs = join(base, TRASH_DIR);
  let tsDirs: string[];
  try {
    tsDirs = readdirSync(trashAbs, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name);
  } catch {
    return [];
  }
  const expired: { trashRel: string; absPath: string }[] = [];
  for (const ts of tsDirs) {
    const tsDate = parseCompactTimestamp(ts);
    if (tsDate === null || now.getTime() - tsDate.getTime() <= TRASH_TTL_MS) continue;
    const tsAbs = join(trashAbs, ts);
    for (const child of readdirSync(tsAbs)) {
      expired.push({ trashRel: `${TRASH_DIR}/${ts}/${child}`, absPath: join(tsAbs, child) });
    }
  }
  return expired;
}
export function sweepTrash(root: Root, now: Date = new Date()): { trashRel: string; absPath: string }[] {
  return sweepTrashAt(rootPath(root), now);
}

// The actual removal — wrapped in runMutation by the caller (round 1 plan review F1), never called
// bare. Also removes the parent <timestamp> dir once it's empty, best-effort: ENOTEMPTY (siblings not
// yet processed in this same pass) is swallowed, and a future sweep or a later entry in this same
// loop finishes the job. ponytail: no Root-based sibling — absPath always comes pre-resolved from
// sweepTrash's own directory scan, never from user input, so there is no rel left to resolve.
export function removeExpiredTrashEntryAt(absPath: string): void {
  rmSync(absPath, { recursive: true, force: true });
  try { rmdirSync(dirname(absPath)); } catch { /* siblings remain — a later removal or sweep finishes it */ }
}

// --- Zip (§5 item 7) -------------------------------------------------------
export const ZIP_ENTRY_CAP = 5_000;
export const ZIP_BYTES_CAP = 200 * 1024 * 1024;
const ZIP_CHUNK = 64 * 1024;

// Case-insensitive, basename-only — one small predicate reused for files and directories alike.
const SECRET_BASENAME_RE = /^\.env(\..+)?$|\.(pem|key|p12|pfx|jks)$|^id_rsa|^id_ed25519|^credentials|^secrets/i;
export function isSecretName(basename: string): boolean {
  return SECRET_BASENAME_RE.test(basename);
}

export type ZipEntry = { rel: string; absPath: string; size: number };
export type ZipPlan =
  | { ok: true; entries: ZipEntry[]; secretsExcluded: number }
  | { ok: false; error: "zip-too-large" };

// Lazy, metadata-only walk under a PRUNED secret directory: counts files for X-Secrets-Excluded
// without ever opening one for content. Doesn't re-verify containment like the main walk does —
// the only consequence of an inaccurate count is a slightly-off toast number.
async function countPrunedFiles(absDir: string): Promise<number> {
  let count = 0;
  let dir;
  try { dir = await opendir(absDir); } catch { return count; }
  try {
    for await (const entry of dir) {
      if (entry.isSymbolicLink()) continue;
      const childAbs = join(absDir, entry.name);
      if (entry.isDirectory()) count += await countPrunedFiles(childAbs);
      else if (entry.isFile()) count++;
    }
  } finally {
    await dir.close().catch(() => undefined);
  }
  return count;
}

export async function planZipAt(base: string, rel: string): Promise<ZipPlan> {
  if (!strictRel(rel, true)) throw new Error("outside allowlist"); // round 1 plan review F2
  const rootAbs = resolveWithin(base, rel);
  if (!statSync(rootAbs).isDirectory()) throw new Error("not a directory");

  const entries: ZipEntry[] = [];
  let secretsExcluded = 0;
  let totalBytes = 0;
  let overCap = false;

  async function walk(absDir: string, relDir: string): Promise<void> {
    if (overCap) return;
    let dir;
    try { dir = await opendir(absDir); } catch { return; }
    try {
      for await (const entry of dir) {
        if (overCap) return;
        if (entry.isSymbolicLink()) continue;
        const childRel = relDir ? `${relDir}/${entry.name}` : entry.name;
        const childAbs = join(absDir, entry.name);
        if (entry.isDirectory()) {
          if (EXCLUDES.has(entry.name)) continue; // covers ".jax-trash" too — belt-and-suspenders
          if (isSecretName(entry.name)) { secretsExcluded += await countPrunedFiles(childAbs); continue; }
          let childReal: string;
          try { childReal = await realpath(childAbs); } catch { continue; }
          if (childReal !== childAbs) continue; // mirrors searchRoots's own escape check
          await walk(childReal, childRel);
        } else if (entry.isFile()) {
          if (isSecretName(entry.name)) { secretsExcluded++; continue; }
          let stat;
          try { stat = statSync(childAbs); } catch { continue; }
          entries.push({ rel: childRel, absPath: childAbs, size: stat.size });
          totalBytes += stat.size;
          if (entries.length > ZIP_ENTRY_CAP || totalBytes > ZIP_BYTES_CAP) { overCap = true; return; }
        }
      }
    } finally {
      await dir.close().catch(() => undefined);
    }
  }

  await walk(rootAbs, "");
  if (overCap) return { ok: false, error: "zip-too-large" };
  return { ok: true, entries, secretsExcluded };
}
export function planZip(root: Root, rel: string): Promise<ZipPlan> {
  return planZipAt(rootPath(root), rel);
}

function dosDateTime(d: Date): { time: number; date: number } {
  const time = (d.getHours() << 11) | (d.getMinutes() << 5) | (d.getSeconds() >> 1);
  const date = ((d.getFullYear() - 1980) << 9) | ((d.getMonth() + 1) << 5) | d.getDate();
  return { time, date };
}

function zipLocalHeader(nameBytes: Buffer, date: number, time: number): Buffer {
  const b = Buffer.alloc(30);
  b.writeUInt32LE(0x04034b50, 0);
  b.writeUInt16LE(20, 4);
  b.writeUInt16LE(0x0008, 6);  // general purpose flag bit 3: streaming — crc/sizes follow in a data descriptor
  b.writeUInt16LE(0, 8);       // method 0 = store (no compression, per §5 item 7)
  b.writeUInt16LE(time, 10);
  b.writeUInt16LE(date, 12);
  b.writeUInt32LE(0, 14);
  b.writeUInt32LE(0, 18);
  b.writeUInt32LE(0, 22);
  b.writeUInt16LE(nameBytes.length, 26);
  b.writeUInt16LE(0, 28);
  return Buffer.concat([b, nameBytes]);
}
function zipDataDescriptor(crc: number, size: number): Buffer {
  const b = Buffer.alloc(16);
  b.writeUInt32LE(0x08074b50, 0);
  b.writeUInt32LE(crc >>> 0, 4);
  b.writeUInt32LE(size, 8);
  b.writeUInt32LE(size, 12);
  return b;
}
function zipCentralHeader(nameBytes: Buffer, date: number, time: number, crc: number, size: number, localOffset: number): Buffer {
  const b = Buffer.alloc(46);
  b.writeUInt32LE(0x02014b50, 0);
  b.writeUInt16LE(20, 4);
  b.writeUInt16LE(20, 6);
  b.writeUInt16LE(0x0008, 8);
  b.writeUInt16LE(0, 10);
  b.writeUInt16LE(time, 12);
  b.writeUInt16LE(date, 14);
  b.writeUInt32LE(crc >>> 0, 16);
  b.writeUInt32LE(size, 20);
  b.writeUInt32LE(size, 24);
  b.writeUInt16LE(nameBytes.length, 28);
  b.writeUInt16LE(0, 30);
  b.writeUInt16LE(0, 32);
  b.writeUInt16LE(0, 34);
  b.writeUInt16LE(0, 36);
  b.writeUInt32LE(0, 38);
  b.writeUInt32LE(localOffset, 42);
  return Buffer.concat([b, nameBytes]);
}
function zipEndOfCentralDirectory(count: number, centralSize: number, centralOffset: number): Buffer {
  const b = Buffer.alloc(22);
  b.writeUInt32LE(0x06054b50, 0);
  b.writeUInt16LE(0, 4);
  b.writeUInt16LE(0, 6);
  b.writeUInt16LE(count, 8);
  b.writeUInt16LE(count, 10);
  b.writeUInt32LE(centralSize, 12);
  b.writeUInt32LE(centralOffset, 16);
  b.writeUInt16LE(0, 20);
  return b;
}

// Streamed STORE-only writer. Reads each file once, computing CRC32 incrementally per chunk
// (zlib.crc32 takes a running value) while yielding the same bytes — sizes are known upfront from
// planZipAt's stat, CRC only after reading, hence the streaming-flag + trailing data-descriptor
// technique (never buffers or re-reads a file). One shared archive-creation timestamp for all
// entries — ponytail: source mtimes not preserved, upgrade if Rafa asks.
export async function* zipStream(entries: ZipEntry[]): AsyncGenerator<Uint8Array> {
  let offset = 0;
  const central: Buffer[] = [];
  const { time, date } = dosDateTime(new Date());
  for (const entry of entries) {
    const nameBytes = Buffer.from(entry.rel, "utf8");
    const header = zipLocalHeader(nameBytes, date, time);
    yield header;
    const localOffset = offset;
    offset += header.length;

    let crc = 0;
    let size = 0;
    const fd = openSync(entry.absPath, OPEN_READ);
    try {
      const buf = Buffer.alloc(ZIP_CHUNK);
      for (;;) {
        const n = readSync(fd, buf, 0, buf.length, null);
        if (n === 0) break;
        const chunk = Buffer.from(buf.subarray(0, n));
        crc = crc32(chunk, crc);
        size += n;
        yield chunk;
        offset += n;
      }
    } finally {
      closeSync(fd);
    }

    const descriptor = zipDataDescriptor(crc, size);
    yield descriptor;
    offset += descriptor.length;
    central.push(zipCentralHeader(nameBytes, date, time, crc, size, localOffset));
  }
  const centralStart = offset;
  const centralBuf = Buffer.concat(central);
  yield centralBuf;
  yield zipEndOfCentralDirectory(entries.length, centralBuf.length, centralStart);
}

// --- Quick-open index (§5 item 8) -----------------------------------------
export type IndexHit = { root: Root; rel: string; name: string };
export type IndexResult = { data: IndexHit[]; truncated: boolean; truncatedRoots: Root[]; builtAt: string };
export const INDEX_PATHS_CAP = 20_000;
export const INDEX_VISIT_CAP = 50_000;
export const INDEX_TIME_MS = 2_000;
export const INDEX_TTL_MS = 30_000;

// Shared by walkAllPaths (below, no query filter/signal) and searchRoots (Task 3, query filter +
// abort signal): discovery builds this root's top-level listing, bounded by rootTotal since slice
// sizes are only known once it completes (§5 #13/#14); planSlices then divides rootTotal into a
// root bucket (loose files) + one slice per subdirectory, each walked to completion or its own
// exhaustion, so no sibling starves another (§5 #1/#3, fixing FI-1). `match` filters a hit's rel
// path; `signal` is optional — `throwIfAborted` no-ops when it is undefined.
async function walkRootFair(
  root: Root,
  realRoot: string,
  rootTotal: SliceCaps,
  rootBucketStart: number,
  match: (rel: string) => boolean,
  signal?: AbortSignal,
): Promise<{ hits: IndexHit[]; truncated: boolean }> {
  const subdirNames: string[] = [];
  const looseFileNames: string[] = [];
  let truncated = false;
  let discoveryVisited = 0;
  const discoveryDir = await opendir(realRoot);
  try {
    for (;;) {
      throwIfAborted(signal);
      if (discoveryVisited >= rootTotal.visitCap || performance.now() - rootBucketStart >= rootTotal.timeMs) {
        truncated = true;
        break;
      }
      const entry = await discoveryDir.read();
      throwIfAborted(signal);
      if (!entry) break;
      discoveryVisited++;
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory()) { if (!EXCLUDES.has(entry.name)) subdirNames.push(entry.name); }
      else if (entry.isFile()) looseFileNames.push(entry.name);
    }
  } finally {
    await discoveryDir.close();
  }

  const plan = planSlices(subdirNames, rootTotal);
  const hits: IndexHit[] = [];

  {
    const budget = plan.rootBucket;
    // Discovery's own readdir entries and the loose-file scan draw from the SAME root-bucket
    // budget (§5 #13) — carry discoveryVisited forward instead of resetting the counter, so a
    // discovery scan that alone approaches the bucket's cap leaves no loose files reachable
    // (cold review F2).
    let visited = discoveryVisited;
    if (visited >= budget.visitCap || performance.now() - rootBucketStart >= budget.timeMs) truncated = true;
    for (const name of looseFileNames) {
      throwIfAborted(signal);
      if (visited >= budget.visitCap || hits.length >= budget.pathsCap || performance.now() - rootBucketStart >= budget.timeMs) {
        truncated = true;
        break;
      }
      visited++;
      if (match(name)) hits.push({ root, rel: name, name });
    }
  }

  for (const name of subdirNames) {
    const budget = plan.subdirs.get(name)!;
    const sliceHits: IndexHit[] = [];
    let visited = 0;
    let sliceTruncated = false;
    const start = performance.now(); // fresh per-slice clock — no starvation from the root bucket's own timing
    const overBudget = () =>
      sliceHits.length >= budget.pathsCap || visited >= budget.visitCap || performance.now() - start >= budget.timeMs;

    const childAbs = join(realRoot, name);
    let childReal: string | null = null;
    try {
      childReal = await realpath(childAbs);
      throwIfAborted(signal);
      if (childReal !== childAbs) { sliceTruncated = true; childReal = null; }
    } catch {
      throwIfAborted(signal);
      sliceTruncated = true;
    }

    const walk = async (absDir: string, rel: string, depth: number): Promise<void> => {
      throwIfAborted(signal);
      if (overBudget()) { sliceTruncated = true; return; }
      let dir;
      try {
        dir = await opendir(absDir);
      } catch {
        throwIfAborted(signal);
        sliceTruncated = true;
        return;
      }
      try {
        for (;;) {
          throwIfAborted(signal);
          if (overBudget()) { sliceTruncated = true; return; }
          let entry;
          try {
            entry = await dir.read();
          } catch {
            throwIfAborted(signal);
            sliceTruncated = true;
            return;
          }
          throwIfAborted(signal);
          if (overBudget()) { sliceTruncated = true; return; }
          if (!entry) break;
          visited++;
          if (visited > budget.visitCap) { sliceTruncated = true; return; }
          const childRel = rel ? `${rel}/${entry.name}` : entry.name;
          if (entry.isSymbolicLink()) continue;
          if (entry.isDirectory()) {
            if (EXCLUDES.has(entry.name)) continue;
            if (depth >= SEARCH_MAX_DEPTH) { sliceTruncated = true; continue; }
            const grandAbs = join(absDir, entry.name);
            let grandReal: string;
            try {
              grandReal = await realpath(grandAbs);
            } catch {
              throwIfAborted(signal);
              sliceTruncated = true;
              continue;
            }
            throwIfAborted(signal);
            if (overBudget()) { sliceTruncated = true; return; }
            if (grandReal !== grandAbs) { sliceTruncated = true; continue; }
            await walk(grandReal, childRel, depth + 1);
            if (overBudget()) { sliceTruncated = true; return; }
          } else if (entry.isFile() && match(childRel)) {
            sliceHits.push({ root, rel: childRel, name: entry.name });
            if (sliceHits.length >= budget.pathsCap) { sliceTruncated = true; return; }
          }
        }
      } finally {
        try { await dir.close(); }
        catch { throwIfAborted(signal); sliceTruncated = true; }
        throwIfAborted(signal);
      }
    };

    if (childReal !== null) {
      // overBudget() can already be true here (a slow realpath ate the slice's own clock) — the
      // walk never runs, but that must still count as a truncation, not a silent skip (review F3).
      if (overBudget()) sliceTruncated = true;
      else await walk(childReal, name, 1);
    }
    hits.push(...sliceHits);
    if (sliceTruncated) truncated = true;
  }

  return { hits, truncated };
}

// Captured before any fs call so a root with no subdirectories (root bucket = the WHOLE budget,
// §5 #14's N=0 case) still times out from the same instant the old whole-root walk did.
async function walkAllPaths(root: Root, absRoot: string): Promise<{ hits: IndexHit[]; truncated: boolean }> {
  const rootBucketStart = performance.now();
  const realRoot = await realpath(absRoot);
  const rootTotal: SliceCaps = { pathsCap: INDEX_PATHS_CAP, visitCap: INDEX_VISIT_CAP, timeMs: INDEX_TIME_MS };
  return walkRootFair(root, realRoot, rootTotal, rootBucketStart, () => true);
}

export async function buildIndexUncached(scope: GatedSearchScope): Promise<IndexResult> {
  const targets: { root: Root; path: string }[] =
    scope.kind === "all" ? activeRoots()
    : scope.kind === "reposOnly" ? [{ root: "repos", path: ROOTS.repos }]
    : scope.kind === "vault" ? [{ root: "vault", path: rootPath("vault") }]
    : [{ root: "repos", path: resolveWithin(ROOTS.repos, scope.name) }];
  const data: IndexHit[] = [];
  const truncatedRoots: Root[] = [];
  for (const target of targets) {
    const { hits, truncated } = await walkAllPaths(target.root, target.path);
    data.push(...(scope.kind === "repo" ? hits.map((h) => ({ ...h, rel: `${scope.name}/${h.rel}` })) : hits));
    if (truncated) truncatedRoots.push(target.root);
  }
  return { data, truncated: truncatedRoots.length > 0, truncatedRoots, builtAt: new Date().toISOString() };
}

const indexCacheStore = new Map<string, { at: number; json: unknown }>();
export async function buildIndex(scope: GatedSearchScope, now: number = Date.now()): Promise<IndexResult> {
  const key = scope.kind === "repo" ? `repo:${scope.name}` : scope.kind;
  const cached = cacheGet(indexCacheStore, key, now, INDEX_TTL_MS);
  if (cached) return cached as IndexResult;
  const result = await buildIndexUncached(scope);
  cachePut(indexCacheStore, key, result, now);
  return result;
}
