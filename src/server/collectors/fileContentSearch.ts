// Content search (spec §8): spawns system `rg` (argv array, never a shell), streams its NDJSON
// stdout (never execFile's buffered maxBuffer), and converts each hit's path back through
// relativeWithin — the SAME allowlist mechanism the rest of files.ts uses.
// relative import: vitest resolves no @/ alias for value imports in tested server modules
import { join } from "node:path";
import { spawn } from "node:child_process";
import { activeRoots, EXCLUDES, isSecretName, READ_CAP, relativeWithin, resolveWithin, rootPath, ROOTS, throwIfAborted, type GatedSearchScope, type Root } from "./files";

export type ContentHit = { root: Root; rel: string; line: number; snippet: string };
export type ContentSearchResult = { hits: ContentHit[]; truncated: boolean };
export const CONTENT_SEARCH_CAP = 200; // matches SEARCH_CAP
export const CONTENT_SEARCH_TIMEOUT_MS = 5000; // matches SEARCH_TIME_MS
export const PER_FILE_CAP = 5;

export type SpawnImpl = (file: string, args: string[], opts: { cwd: string; stdio: ["ignore", "pipe", "pipe"] }) => {
  stdout: { on: (ev: "data", fn: (chunk: Buffer) => void) => void } | null;
  on: (ev: "error" | "close", fn: (arg: unknown) => void) => void;
  kill: (signal?: NodeJS.Signals) => boolean;
};
const systemSpawn = spawn as unknown as SpawnImpl;

export function buildRgArgs(query: string): string[] {
  const excludeGlobs = [...EXCLUDES].flatMap((name) => ["--glob", `!${name}`]);
  return ["--json", "-i", "--fixed-strings", "--max-filesize", String(READ_CAP), "-m", String(PER_FILE_CAP), ...excludeGlobs, "--", query];
}

export type RgHit = { path: string; line: number; snippet: string };
export function parseRgLine(line: string): RgHit | null {
  if (!line) return null;
  let obj: unknown;
  try { obj = JSON.parse(line); } catch { return null; }
  const m = obj as { type?: string; data?: { path?: { text?: string }; line_number?: number; lines?: { text?: string } } };
  if (!m || m.type !== "match" || !m.data?.path?.text || typeof m.data.line_number !== "number") return null;
  return { path: m.data.path.text, line: m.data.line_number, snippet: (m.data.lines?.text ?? "").replace(/\n$/, "") };
}

// §8: EXCLUDES + isSecretName checked on every path SEGMENT — the same functions the zip walk
// (planZipAt) already applies, no second exclusion list. Mandatory, never just an rg --glob hint.
export function isExcludedRel(rel: string): boolean {
  return rel.split("/").some((s) => EXCLUDES.has(s) || isSecretName(s));
}

function abortError(signal?: AbortSignal): Error {
  const reason = signal?.reason;
  return reason instanceof Error ? reason : (new DOMException("The operation was aborted.", "AbortError") as unknown as Error);
}

type Shared = { count: number };

// Spawns ONE rg process, settling on whichever bound is hit first (shared cap, timeout, abort, or
// exit). `kill()` settles THIS stream immediately, never waiting on a real close event.
// `filter` runs on every raw hit BEFORE it counts toward `cap` (round-2 F2): an excluded or
// missing hit must never crowd out a valid one, so filtering can't wait until after the process
// exits (the OLD shape — filter the whole array once streaming finished) the way the round-1 draft
// did. Default is the identity — `runRg` below stays ROOTS-independent for its own real-rg test.
function startRg(
  cwd: string, query: string, signal: AbortSignal | undefined, spawnImpl: SpawnImpl,
  shared: Shared, cap: number, killSibling: () => void,
  filter: (raw: RgHit) => RgHit | null = (raw) => raw,
): { result: Promise<{ hits: RgHit[]; truncated: boolean }>; kill: () => void } {
  let finishOk = (_truncated: boolean) => {};
  let finishErr = (_err: Error) => {};
  const result = new Promise<{ hits: RgHit[]; truncated: boolean }>((resolve, reject) => {
    throwIfAborted(signal);
    const child = spawnImpl("rg", buildRgArgs(query), { cwd, stdio: ["ignore", "pipe", "pipe"] });
    const hits: RgHit[] = [];
    let buf = "";
    let settled = false;
    const cleanup = () => { clearTimeout(timer); signal?.removeEventListener("abort", onAbort); };
    finishOk = (truncated) => {
      if (settled) return;
      settled = true;
      cleanup();
      try { child.kill("SIGTERM"); } catch { /* already exited */ }
      resolve({ hits, truncated });
    };
    finishErr = (err) => {
      if (settled) return;
      settled = true;
      cleanup();
      try { child.kill("SIGTERM"); } catch { /* already exited */ }
      reject(err);
    };
    const onAbort = () => finishErr(abortError(signal));
    signal?.addEventListener("abort", onAbort, { once: true });
    const timer = setTimeout(() => finishOk(true), CONTENT_SEARCH_TIMEOUT_MS);
    child.stdout?.on("data", (chunk: Buffer) => {
      buf += chunk.toString("utf8");
      const parts = buf.split("\n");
      buf = parts.pop() ?? "";
      for (const line of parts) {
        const raw = parseRgLine(line);
        if (!raw) continue;
        const m = filter(raw);
        if (!m) continue; // excluded or missing — dropped BEFORE it can count toward the cap (round-2 F2)
        hits.push(m);
        shared.count++;
        if (shared.count >= cap) { finishOk(true); killSibling(); return; }
      }
    });
    child.on("error", (err: unknown) => {
      const code = (err as NodeJS.ErrnoException | null)?.code;
      finishErr(new Error(code === "ENOENT" ? "rg is required and was not found" : "content search unavailable"));
    });
    child.on("close", (code: unknown) => {
      if (code === 0 || code === 1) finishOk(false); // 1 = no matches, not an error
      else finishErr(new Error("content search unavailable"));
    });
  });
  return { result, kill: () => finishOk(true) };
}

// Exported for a direct, ROOTS-independent test against a real temp dir — a lone stream with its
// own private cap, no sibling to kill, and no filter (raw paths, not yet converted to a `rel`).
export function runRg(cwd: string, query: string, signal?: AbortSignal, spawnImpl: SpawnImpl = systemSpawn): Promise<{ hits: RgHit[]; truncated: boolean }> {
  return startRg(cwd, query, signal, spawnImpl, { count: 0 }, CONTENT_SEARCH_CAP, () => {}).result;
}

// relativeWithin's ROOT is always ROOTS[root] (never `cwd`) — for a repo scope this naturally
// yields "<name>/<hitPath>" with no manual prefix, and rejects a symlink-escape (§8's table). Runs
// PER HIT, inside `startRg`'s stream handler, not after the process exits (round-2 F2) — a raw
// hit's `path` field is overwritten with the final, allowlist-checked `rel`, or dropped.
function scopedFilter(root: Root, cwd: string): (raw: RgHit) => RgHit | null {
  return (raw) => {
    const rel = relativeWithin(rootPath(root), join(cwd, raw.path));
    if (rel === null || isExcludedRel(rel)) return null;
    return { ...raw, path: rel };
  };
}
function startScopedRg(
  root: Root, cwd: string, query: string, signal: AbortSignal | undefined, spawnImpl: SpawnImpl,
  shared: Shared, cap: number, killSibling: () => void,
): ReturnType<typeof startRg> {
  return startRg(cwd, query, signal, spawnImpl, shared, cap, killSibling, scopedFilter(root, cwd));
}
function toContentHits(root: Root, raws: RgHit[]): ContentHit[] {
  return raws.map((raw) => ({ root, rel: raw.path, line: raw.line, snippet: raw.snippet }));
}

async function runAll(query: string, signal: AbortSignal | undefined, spawnImpl: SpawnImpl) {
  const shared: Shared = { count: 0 };
  const runners: { root: Root; run: ReturnType<typeof startRg> }[] = [];
  for (const { root, path } of activeRoots()) {
    runners.push({
      root,
      run: startScopedRg(root, path, query, signal, spawnImpl, shared, CONTENT_SEARCH_CAP, () => {
        for (const other of runners) if (other.root !== root) other.run.kill();
      }),
    });
  }
  try {
    const settled = await Promise.all(runners.map(async ({ root, run }) => [root, await run.result] as const));
    return Object.fromEntries(settled) as Partial<Record<Root, { hits: RgHit[]; truncated: boolean }>>;
  } catch (err) {
    // Round-2 (d4876572f596) F5: `killSibling` above only fires on a shared-cap hit, never on a
    // spawn error or bad exit code — without this, a one-sided rejection would leave the OTHER
    // rg process running until its own CONTENT_SEARCH_TIMEOUT_MS. `kill()` is a no-op on the side
    // that already settled (its `finishOk`/`finishErr` already ran); the caller is rejecting
    // regardless, so the still-open side's now-discarded resolution doesn't matter.
    for (const { run } of runners) run.kill();
    throw err;
  }
}

export async function searchContent(scope: GatedSearchScope, query: string, signal?: AbortSignal, spawnImpl: SpawnImpl = systemSpawn): Promise<ContentSearchResult> {
  throwIfAborted(signal);
  if (scope.kind === "vault") {
    const { hits, truncated } = await startScopedRg("vault", rootPath("vault"), query, signal, spawnImpl, { count: 0 }, CONTENT_SEARCH_CAP, () => {}).result;
    return { hits: toContentHits("vault", hits), truncated };
  }
  // Decision 16: the repos-only narrowing of "all" — ONE rg over ROOTS.repos itself (exactly
  // "all"'s repos-half target), never the vault sibling runAll would spawn.
  if (scope.kind === "reposOnly") {
    const { hits, truncated } = await startScopedRg("repos", ROOTS.repos, query, signal, spawnImpl, { count: 0 }, CONTENT_SEARCH_CAP, () => {}).result;
    return { hits: toContentHits("repos", hits), truncated };
  }
  if (scope.kind === "repo") {
    const cwd = resolveWithin(rootPath("repos"), scope.name);
    const { hits, truncated } = await startScopedRg("repos", cwd, query, signal, spawnImpl, { count: 0 }, CONTENT_SEARCH_CAP, () => {}).result;
    return { hits: toContentHits("repos", hits), truncated };
  }
  const perRoot = await runAll(query, signal, spawnImpl);
  const roots = activeRoots();
  return {
    hits: roots.flatMap(({ root }) => toContentHits(root, perRoot[root]!.hits)),
    truncated: roots.some(({ root }) => perRoot[root]!.truncated),
  };
}
