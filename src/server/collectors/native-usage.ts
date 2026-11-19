import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { mkdir, readFile, stat as nodeStatAsync, realpath as nodeRealpathAsync, opendir as nodeOpendirAsync, open as nodeOpenAsync, writeFile, access } from "node:fs/promises";
import { homedir } from "node:os";
import { isAbsolute, join, sep } from "node:path";
import { promisify } from "node:util";
import type Database from "better-sqlite3";
import {
  addPendingDir,
  applyFileBatch,
  countRegisteredFiles,
  ensureDiscoveryCycle,
  finishDiscoveryCycle,
  getFileCheckpoint,
  getOpenCodeCursor,
  getRegistryCounts,
  getSourceStatuses,
  hasPendingDirs,
  markDirDone,
  markFileCurrent,
  markFileFailure,
  nextPendingDir,
  persistOpenCodeBatch,
  pruneNativeUsage,
  registerNativeFile,
  selectIngestionCandidates,
  setDiscoveryCurrentDir,
  setSourceStatus,
  type FileCheckpoint,
  type NativeContextRow,
  type OpenCodeCursorState,
} from "../db/native-usage";
import { readGeneralSettings, type SettingsResult } from "../settings";
import { jaxosHome } from "../env";

export const HARNESSES = ["claude", "codex", "opencode"] as const;
export type Harness = (typeof HARNESSES)[number];

export type UsageCost = {
  amount: number;
  provenance: "source-reported";
  currency: string | null;
};

export type NativeUsageItem = {
  harness: Harness;
  itemId: string;
  sessionId: string;
  childId: string | null;
  parentId: string | null;
  isChild: boolean;
  provenance: string;
  agent: string;
  model: string | null;
  modelConflict: boolean;
  identityConflict: boolean;
  ts: number; // earliest valid occurrence, unix seconds
  updatedTs: number; // latest valid update timestamp, unix seconds
  day: string; // UTC calendar day of ts
  allInput: number;
  allOutput: number;
  details: {
    input: number | null;
    output: number | null;
    cacheRead: number | null;
    cacheWrite: number | null;
    reasoning: number | null;
  };
  cost: UsageCost | null;
  projection: "v1" | "v2" | null;
  fileIdentity: string | null;
  turnId: string | null;
  rootTurnId: string | null;
};

export type SourceState = "available" | "empty" | "indexing" | "unavailable" | "disabled" | "incompatible";
export type SourceCoverage = {
  source: Harness;
  state: SourceState;
  lastRefresh: number | null;
  progress: { filesDone: number; filesTotal: number } | null;
  warnings: string[];
};

export type ParseOutcome<T> = { item: T } | { skip: string };

export const MAX_ID_CODE_POINTS = 256;

const isObj = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v);
const count = (v: unknown): number | null =>
  typeof v === "number" && Number.isSafeInteger(v) && v >= 0 ? v : null;
const unicodeLength = (v: string) => [...v].length;
const validId = (v: unknown): v is string =>
  typeof v === "string" && v.length > 0 && unicodeLength(v) <= MAX_ID_CODE_POINTS;
const validIdOrNull = (v: unknown): boolean => v === undefined || v === null || validId(v);

export const utcDay = (epochSeconds: number) =>
  new Date(epochSeconds * 1000).toISOString().slice(0, 10);

function parseTimestamp(v: unknown, now: number): number | null {
  if (typeof v !== "string" && typeof v !== "number") return null;
  const ms = typeof v === "number" ? (v >= 1e11 ? v : v * 1000) : Date.parse(v);
  if (!Number.isFinite(ms) || ms <= 0) return null;
  const seconds = Math.floor(ms / 1000);
  if (seconds > now) return null;
  return seconds;
}

function modelFrom(a: unknown, b: unknown): { model: string | null; modelConflict: boolean } {
  const left = typeof a === "string" && a.length > 0 ? a : null;
  const right = typeof b === "string" && b.length > 0 ? b : null;
  if (left !== null && right !== null && left !== right) return { model: null, modelConflict: true };
  return { model: left ?? right, modelConflict: false };
}

export function parseClaudeLine(
  line: string,
  opts: { child: boolean; now: number },
): ParseOutcome<NativeUsageItem> {
  let raw: unknown;
  try {
    raw = JSON.parse(line);
  } catch {
    return { skip: "malformed" };
  }
  if (!isObj(raw) || !isObj(raw.message)) return { skip: "not-assistant" };
  const message = raw.message;
  if (raw.type !== undefined && raw.type !== "assistant") return { skip: "not-assistant" };
  if (message.role !== undefined && message.role !== "assistant") return { skip: "not-assistant" };
  if (!validId(message.id) || !validId(raw.sessionId ?? raw.session_id)) return { skip: "bad-identity" };
  const sessionId = (raw.sessionId ?? raw.session_id) as string;

  let childId: string | null = null;
  if (opts.child) {
    const agentId = raw.agentId ?? message.agentId;
    if (!validId(agentId)) return { skip: "bad-child-identity" };
    childId = agentId;
  }

  const ts = parseTimestamp(raw.timestamp, opts.now);
  if (ts === null) return { skip: "bad-timestamp" };
  if (!isObj(message.usage)) return { skip: "no-usage" };
  const usage = message.usage;
  const input = count(usage.input_tokens);
  const output = count(usage.output_tokens);
  const cacheRead = usage.cache_read_input_tokens === undefined ? 0 : count(usage.cache_read_input_tokens);
  const cacheWrite = usage.cache_creation_input_tokens === undefined ? 0 : count(usage.cache_creation_input_tokens);
  if (input === null || output === null || cacheRead === null || cacheWrite === null) return { skip: "bad-counter" };
  const allInput = input + cacheRead + cacheWrite;
  if (!Number.isSafeInteger(allInput)) return { skip: "bad-counter" };

  const { model, modelConflict } = modelFrom(message.model, raw.requestedModel);
  return {
    item: {
      harness: "claude",
      itemId: message.id,
      sessionId,
      childId,
      parentId: null,
      isChild: opts.child,
      provenance: opts.child ? "child" : "lead",
      agent: "Claude Code",
      model,
      modelConflict,
      identityConflict: false,
      ts,
      updatedTs: ts,
      day: utcDay(ts),
      allInput,
      allOutput: output,
      details: { input, output, cacheRead, cacheWrite, reasoning: null },
      cost: null,
      projection: null,
      fileIdentity: null,
      turnId: null,
      rootTurnId: null,
    },
  };
}

type NativeContext = { fileIdentity: string; turnId: string; rootTurnId: string; model: string };

export function parseCodexLine(
  line: string,
  opts: { fileIdentity: string; now: number },
): ParseOutcome<NativeUsageItem> | { context: NativeContext } {
  let raw: unknown;
  try {
    raw = JSON.parse(line);
  } catch {
    return { skip: "malformed" };
  }
  if (!isObj(raw)) return { skip: "unsupported" };
  const payload = isObj(raw.payload) ? raw.payload : null;
  if (!payload) return { skip: "unsupported" };

  if (raw.type === "turn_context" || payload.type === "turn_context") {
    const turnId = payload.turn_id ?? payload.turnId;
    const rootTurnId = payload.root_turn_id ?? payload.rootTurnId ?? turnId;
    const model = payload.model;
    if (!validId(turnId) || !validId(rootTurnId) || typeof model !== "string" || model.length === 0)
      return { skip: "unsupported" };
    return { context: { fileIdentity: opts.fileIdentity, turnId, rootTurnId, model } };
  }

  let usage: Record<string, unknown> | null = null;
  if (isObj(payload.usage)) usage = payload.usage;
  else if (payload.type === "token_count" && isObj(payload.info) && isObj(payload.info.last_token_usage))
    usage = payload.info.last_token_usage;
  if (!usage) return { skip: "unsupported" };

  const sessionId = payload.session_id;
  const threadId = payload.thread_id;
  const responseId = payload.response_id;
  if (!validId(sessionId) || !validId(threadId) || !validId(responseId)) return { skip: "bad-identity" };
  if (!validIdOrNull(payload.turn_id) || !validIdOrNull(payload.root_turn_id)) return { skip: "bad-identity" };

  const ts = parseTimestamp(raw.timestamp, opts.now);
  if (ts === null) return { skip: "bad-timestamp" };

  const input = count(usage.input_tokens);
  const output = count(usage.output_tokens);
  const total = count(usage.total_tokens);
  const cached = usage.cached_input_tokens === undefined ? 0 : count(usage.cached_input_tokens);
  const reasoning = usage.reasoning_output_tokens === undefined ? 0 : count(usage.reasoning_output_tokens);
  const cacheWrite = usage.cache_write_input_tokens === undefined ? 0 : count(usage.cache_write_input_tokens);
  if (input === null || output === null || total === null || cached === null || reasoning === null || cacheWrite === null)
    return { skip: "bad-counter" };
  if (total !== input + output || cached > input || reasoning > output) return { skip: "bad-constraint" };

  const turnId = typeof payload.turn_id === "string" ? payload.turn_id : null;
  const rootTurnId = typeof payload.root_turn_id === "string" ? payload.root_turn_id : turnId;
  return {
    item: {
      harness: "codex",
      itemId: `${threadId}|${responseId}`,
      sessionId: threadId,
      childId: null,
      parentId: null,
      isChild: false,
      provenance: sessionId,
      agent: "Codex",
      model: null,
      modelConflict: false,
      identityConflict: false,
      ts,
      updatedTs: ts,
      day: utcDay(ts),
      allInput: input,
      allOutput: output,
      details: { input, output, cacheRead: cached, cacheWrite, reasoning },
      cost: null,
      projection: null,
      fileIdentity: opts.fileIdentity,
      turnId,
      rootTurnId,
    },
  };
}

export function parseOpenCodeRow(raw: unknown, opts: { now: number }): ParseOutcome<NativeUsageItem> {
  if (!isObj(raw)) return { skip: "malformed" };
  const projection = raw.projection;
  if (projection !== "v1" && projection !== "v2") return { skip: "unsupported" };
  if (typeof raw.session_id === "string" && unicodeLength(raw.session_id) > MAX_ID_CODE_POINTS)
    return { skip: "id-too-long" };
  if (typeof raw.message_id === "string" && unicodeLength(raw.message_id) > MAX_ID_CODE_POINTS)
    return { skip: "id-too-long" };
  if (!validId(raw.session_id) || !validId(raw.message_id)) return { skip: "bad-identity" };
  if (!validIdOrNull(raw.parent_session_id)) return { skip: "bad-identity" };

  const ts = parseTimestamp(raw.created, opts.now);
  if (ts === null) return { skip: "bad-timestamp" };
  const updated = parseTimestamp(raw.updated, opts.now);

  const input = count(raw.input);
  const output = count(raw.output);
  const reasoning = raw.reasoning === undefined ? 0 : count(raw.reasoning);
  const cacheRead = raw.cache_read === undefined ? 0 : count(raw.cache_read);
  const cacheWrite = raw.cache_write === undefined ? 0 : count(raw.cache_write);
  if (input === null || output === null || reasoning === null || cacheRead === null || cacheWrite === null)
    return { skip: "bad-counter" };
  const allInput = input + cacheRead + cacheWrite;
  const allOutput = output + reasoning;
  if (!Number.isSafeInteger(allInput) || !Number.isSafeInteger(allOutput)) return { skip: "bad-counter" };

  const parentId = typeof raw.parent_session_id === "string" && raw.parent_session_id.length > 0
    ? raw.parent_session_id
    : null;
  const costAmount =
    typeof raw.cost === "number" && Number.isFinite(raw.cost) && raw.cost >= 0 ? raw.cost : null;
  const model = typeof raw.model_id === "string" && raw.model_id.length > 0 ? raw.model_id : null;
  const agent = typeof raw.agent === "string" && raw.agent.length > 0 ? raw.agent : "OpenCode";
  const sessionId = raw.session_id;

  return {
    item: {
      harness: "opencode",
      itemId: `${sessionId}|${raw.message_id}|${projection}`,
      sessionId,
      childId: parentId ? sessionId : null,
      parentId,
      isChild: parentId !== null,
      provenance: projection,
      agent,
      model,
      modelConflict: false,
      identityConflict: false,
      ts,
      updatedTs: Math.max(ts, updated ?? ts),
      day: utcDay(ts),
      allInput,
      allOutput,
      details: { input, output, cacheRead, cacheWrite, reasoning },
      cost: costAmount === null ? null : { amount: costAmount, provenance: "source-reported", currency: null },
      projection,
      fileIdentity: null,
      turnId: null,
      rootTurnId: null,
    },
  };
}

function counterTuple(item: NativeUsageItem): number[] {
  return [
    item.details.output ?? 0,
    item.details.input ?? 0,
    item.details.cacheRead ?? 0,
    item.details.cacheWrite ?? 0,
    item.details.reasoning ?? 0,
  ];
}

function compareTuples(a: number[], b: number[]): number {
  for (let i = 0; i < Math.max(a.length, b.length); i += 1) {
    const diff = (a[i] ?? 0) - (b[i] ?? 0);
    if (diff !== 0) return diff;
  }
  return 0;
}

function attributionKey(item: NativeUsageItem): [number, string, string, string] {
  return [item.ts, item.sessionId, item.childId ?? "", item.isChild ? "child" : "lead"];
}

function earliestAttribution(a: NativeUsageItem, b: NativeUsageItem): NativeUsageItem {
  const ka = attributionKey(a);
  const kb = attributionKey(b);
  for (let i = 0; i < ka.length; i += 1) {
    if (ka[i] < kb[i]) return a;
    if (ka[i] > kb[i]) return b;
  }
  return a;
}

// Deterministic, order-independent merge for a repeated native identity.
export function mergeNativeItem(a: NativeUsageItem, b: NativeUsageItem): NativeUsageItem {
  if (a.harness !== b.harness || a.itemId !== b.itemId) {
    throw new Error("merge requires the same native identity");
  }
  const counterWinner =
    a.updatedTs !== b.updatedTs
      ? a.updatedTs > b.updatedTs ? a : b
      : compareTuples(counterTuple(a), counterTuple(b)) >= 0 ? a : b;
  const attribution = earliestAttribution(a, b);
  const ts = Math.min(a.ts, b.ts);
  const modelConflict =
    a.modelConflict ||
    b.modelConflict ||
    (a.model !== null && b.model !== null && a.model !== b.model);
  return {
    ...counterWinner,
    sessionId: attribution.sessionId,
    childId: attribution.childId,
    parentId: attribution.parentId,
    isChild: attribution.isChild,
    provenance: attribution.provenance,
    ts,
    day: utcDay(ts),
    model: modelConflict ? null : a.model ?? b.model,
    modelConflict,
    identityConflict:
      a.identityConflict || b.identityConflict ||
      a.sessionId !== b.sessionId || (a.childId ?? "") !== (b.childId ?? "") || a.isChild !== b.isChild,
  };
}

export function canonicalOpenCodeItems(items: NativeUsageItem[]): {
  items: NativeUsageItem[];
  excludedV1: number;
  mixedSessions: number;
} {
  const sessionsWithV2 = new Set(
    items.filter((i) => i.harness === "opencode" && i.projection === "v2").map((i) => i.sessionId),
  );
  const included: NativeUsageItem[] = [];
  let excludedV1 = 0;
  let mixedSessions = 0;
  for (const item of items) {
    if (item.harness !== "opencode") {
      included.push(item);
      continue;
    }
    if (sessionsWithV2.has(item.sessionId)) {
      if (item.projection === "v2") included.push(item);
      else excludedV1 += 1;
    } else {
      included.push(item);
    }
  }
  for (const session of sessionsWithV2) {
    if (items.some((i) => i.harness === "opencode" && i.sessionId === session && i.projection === "v1"))
      mixedSessions += 1;
  }
  return { items: included, excludedV1, mixedSessions };
}

export function accountingSummary(items: NativeUsageItem[]): {
  allInput: number;
  allOutput: number;
  headline: number;
} {
  let allInput = 0;
  let allOutput = 0;
  for (const item of items) {
    allInput += item.allInput;
    allOutput += item.allOutput;
    if (!Number.isSafeInteger(allInput) || !Number.isSafeInteger(allOutput)) {
      throw new Error("aggregate-overflow");
    }
  }
  const headline = allInput + allOutput;
  if (!Number.isSafeInteger(headline)) throw new Error("aggregate-overflow");
  return { allInput, allOutput, headline };
}

// ---------------------------------------------------------------------------
// Bounded native discovery and incremental collection.
// System access is injected and asynchronous: tests never read real transcript
// roots, run the installed OpenCode binary, or perform synchronous native IO in
// the request-triggered after() callback.

export const OPENCODE_EXECUTABLE = "/home/rafa/.opencode/bin/opencode";
export const OPENCODE_EXECUTABLE_SHA256 =
  "9ca0b9953d49997601655e54f846a3efa464f237e47c6f1b04716d0f2e64c4c2";

const JSONL_BUDGET = {
  maxFilesPerPass: 6,
  maxBytesPerSource: 16 * 1024 * 1024,
  stepBytes: 2 * 1024 * 1024,
  chunkBytes: 32 * 1024,
  softDeadlineMs: 200,
  startLineDeadlineMs: 150,
  wallDeadlineMs: 1200,
  lineDeadlineMs: 1000,
  maxLineBytes: 1024 * 1024,
  discoveryEntriesPerPass: 64,
  discoveryMsPerPass: 50,
  refreshMs: 60_000,
  failureRetryMs: 60_000,
};
const OPENCODE_BUDGET = { timeoutMs: 2000, maxReplyBytes: 256 * 1024 };

export type StatLike = {
  dev: number | bigint;
  ino: number | bigint;
  size: number;
  mtimeMs: number;
  isFile(): boolean;
  isDirectory(): boolean;
  isSymbolicLink(): boolean;
};
export type DirEntryLike = {
  name: string;
  isFile(): boolean;
  isDirectory(): boolean;
  isSymbolicLink(): boolean;
};
export type DirHandleLike = {
  read(): Promise<DirEntryLike | null>;
  close(): Promise<void>;
};
export type FileHandleLike = {
  read(buffer: Buffer, offset: number, length: number, position: number): Promise<{ bytesRead: number }>;
  close(): Promise<void>;
  stat?(): Promise<StatLike>;
};
export type NativeFs = {
  stat(path: string): Promise<StatLike>;
  realpath(path: string): Promise<string>;
  opendir(path: string): Promise<DirHandleLike>;
  open(path: string): Promise<FileHandleLike>;
};
export type NativeExecResult = { stdout: string };
export type NativeExec = (
  file: string,
  args: string[],
  opts: { cwd: string; env: Record<string, string>; timeout: number; maxBuffer: number },
) => Promise<NativeExecResult>;

export type NativeCollectorDeps = {
  db: Database.Database;
  now: () => number;
  nowMs: () => number;
  settings: () => SettingsResult;
  fs: NativeFs;
  roots: { claude: string; codex: string; opencodeDataRoot: string; opencodeDb: string };
  opencode: {
    executable: string;
    cwd: string;
    configPath: string;
    xdgConfigHome: string;
    ensureAdapter?: () => Promise<void>;
    exec: NativeExec;
    stat: (path: string) => Promise<StatLike | null>;
    hash: (path: string) => Promise<string | null>;
  };
};

type DiscoveryCandidate = {
  identity: string;
  child: boolean;
};

const base = (p: string) => (p.endsWith(sep) ? p : p + sep);

function insideRealRoot(realRoot: string, real: string): boolean {
  return real === realRoot || real.startsWith(base(realRoot));
}

function warningBag(): { add(code: string): void; values: string[] } {
  const set = new Set<string>();
  return {
    add: (code) => set.add(code),
    get values() {
      return [...set].sort();
    },
  };
}

function errorCode(error: unknown): string | null {
  const code = (error as { code?: unknown } | null)?.code;
  return typeof code === "string" ? code : null;
}

class IoTimeout extends Error {
  constructor() {
    super("io-timeout");
  }
}

function isTimeout(error: unknown): boolean {
  return error instanceof IoTimeout;
}

async function boundedIo<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  if (!(timeoutMs > 0)) throw new IoTimeout();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new IoTimeout()), timeoutMs);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

// Per-source single-flight runtime: at most one directory iterator and one JSONL
// handle, plus the latch that retains ownership of a timed-out unabortable op.
type OwnedOp = { done: boolean };
type SourceRuntime = {
  iterator: { dir: DirHandleLike; path: string } | null;
  pending: OwnedOp | null;
  discoveryFirstNext: boolean;
};

const runtimes = new WeakMap<Database.Database, Map<"claude" | "codex", SourceRuntime>>();

function runtimeFor(db: Database.Database, source: "claude" | "codex"): SourceRuntime {
  let map = runtimes.get(db);
  if (!map) {
    map = new Map();
    runtimes.set(db, map);
  }
  let runtime = map.get(source);
  if (!runtime) {
    runtime = { iterator: null, pending: null, discoveryFirstNext: false };
    map.set(source, runtime);
  }
  return runtime;
}

function ownPending(runtime: SourceRuntime, promise: Promise<unknown>, cleanup?: () => Promise<void> | void): void {
  const record: OwnedOp = { done: false };
  runtime.pending = record;
  promise
    .then(async () => {
      if (cleanup) await cleanup();
    })
    .catch(() => undefined)
    .finally(() => {
      record.done = true;
      if (runtime.pending === record) runtime.pending = null;
    });
}

const joinRel = (a: string, b: string) => (a === "." ? b : `${a}/${b}`);
const relDepth = (rel: string) => (rel === "." ? 0 : rel.split("/").length);
const absOf = (root: string, rel: string) => (rel === "." ? root : join(root, ...rel.split("/")));

async function containReal(
  fs: NativeFs,
  realRoot: string,
  absPath: string,
  timeoutMs: number,
): Promise<string | null> {
  try {
    const real = await boundedIo(fs.realpath(absPath), timeoutMs);
    return insideRealRoot(realRoot, real) ? real : null;
  } catch (error) {
    if (error instanceof IoTimeout) throw error;
    return null;
  }
}

type RootState = { ok: true; realRoot: string } | { ok: false; missing: boolean };

async function resolveRoot(deps: NativeCollectorDeps, root: string, timeoutMs: number): Promise<RootState> {
  try {
    const info = await boundedIo(deps.fs.stat(root), timeoutMs);
    if (!info.isDirectory()) return { ok: false, missing: false };
  } catch (error) {
    if (errorCode(error) === "ENOENT") return { ok: false, missing: true };
    return { ok: false, missing: false };
  }
  try {
    const realRoot = await boundedIo(deps.fs.realpath(root), timeoutMs);
    return { ok: true, realRoot };
  } catch (error) {
    if (errorCode(error) === "ENOENT") return { ok: false, missing: true };
    return { ok: false, missing: false };
  }
}

type DiscoveryOutcome = {
  rootUnavailable: boolean;
  warnings: string[];
};

type DiscoveryContext = {
  deps: NativeCollectorDeps;
  runtime: SourceRuntime;
  source: "claude" | "codex";
  realRoot: string;
  cycle: number;
  warnings: ReturnType<typeof warningBag>;
};

async function registerEntry(
  ctx: DiscoveryContext,
  dirPath: string,
  entry: DirEntryLike,
  dirChild: boolean,
  remainingMs: number,
): Promise<void> {
  const { deps, source } = ctx;
  const rel = joinRel(dirPath, entry.name);
  const abs = absOf(deps.roots[source], rel);
  const depth = relDepth(rel);
  const safeDir = async (): Promise<boolean> => {
    if (!entry.isDirectory()) return false;
    const real = await containReal(deps.fs, ctx.realRoot, abs, remainingMs);
    return real !== null;
  };
  const safeFile = async (): Promise<boolean> => {
    const real = await containReal(deps.fs, ctx.realRoot, abs, remainingMs);
    if (real === null) {
      ctx.warnings.add("symlink-escape");
      return false;
    }
    return true;
  };

  if (source === "claude") {
    if (depth === 1) {
      if (await safeDir()) addPendingDir(deps.db, source, rel, false, ctx.cycle, deps.now());
      else ctx.warnings.add("unsupported-layout");
      return;
    }
    if (depth === 2) {
      if (entry.isDirectory()) {
        if (await safeDir()) addPendingDir(deps.db, source, `${rel}/subagents`, true, ctx.cycle, deps.now());
        else ctx.warnings.add("unsupported-layout");
        return;
      }
      if (entry.isFile() || entry.isSymbolicLink()) {
        if (!entry.name.endsWith(".jsonl")) {
          ctx.warnings.add("unsupported-layout");
          return;
        }
        if (await safeFile()) registerNativeFile(deps.db, source, rel, false, deps.now());
        return;
      }
      ctx.warnings.add("unsupported-layout");
      return;
    }
    if (depth === 4 && dirChild) {
      if ((entry.isFile() || entry.isSymbolicLink()) && entry.name.endsWith(".jsonl")) {
        if (await safeFile()) registerNativeFile(deps.db, source, rel, true, deps.now());
        return;
      }
      ctx.warnings.add("unsupported-layout");
      return;
    }
    ctx.warnings.add("unsupported-layout");
    return;
  }

  if (depth <= 3) {
    if (!entry.isDirectory()) {
      ctx.warnings.add("unsupported-layout");
      return;
    }
    const okName = depth === 1 ? /^\d{4}$/.test(entry.name) : /^\d{2}$/.test(entry.name);
    if (!okName || !(await safeDir())) {
      ctx.warnings.add("unsupported-layout");
      return;
    }
    addPendingDir(deps.db, source, rel, false, ctx.cycle, deps.now());
    return;
  }
  if (depth === 4) {
    if (
      (entry.isFile() || entry.isSymbolicLink()) &&
      entry.name.startsWith("rollout-") &&
      entry.name.endsWith(".jsonl")
    ) {
      if (await safeFile()) registerNativeFile(deps.db, source, rel, false, deps.now());
      return;
    }
  }
  ctx.warnings.add("unsupported-layout");
}

async function discoveryPass(
  deps: NativeCollectorDeps,
  runtime: SourceRuntime,
  source: "claude" | "codex",
  realRoot: string,
  budgetMs: number,
  warnings: ReturnType<typeof warningBag>,
): Promise<boolean> {
  const state = ensureDiscoveryCycle(deps.db, source, deps.now(), JSONL_BUDGET.refreshMs / 1000);
  if (state.state === "complete") return false;
  const ctx: DiscoveryContext = {
    deps, runtime, source, realRoot, cycle: state.cycle, warnings,
  };
  const start = deps.nowMs();
  let entries = 0;
  let rootUnavailable = false;
  let activeDir: string | null = state.currentDir;

  while (entries < JSONL_BUDGET.discoveryEntriesPerPass && deps.nowMs() - start < budgetMs) {
    const current = activeDir ?? nextPendingDir(deps.db, source);
    if (current === null) {
      finishDiscoveryCycle(deps.db, source, deps.now());
      break;
    }
    if (!runtime.iterator || runtime.iterator.path !== current) {
      if (runtime.iterator) {
        try {
          await boundedIo(runtime.iterator.dir.close(), JSONL_BUDGET.wallDeadlineMs);
        } catch {
          /* the replaced iterator is released best-effort */
        }
        runtime.iterator = null;
      }
      const openPromise = deps.fs.opendir(absOf(deps.roots[source], current));
      try {
        const dir = await boundedIo(openPromise, JSONL_BUDGET.wallDeadlineMs);
        runtime.iterator = { dir, path: current };
        setDiscoveryCurrentDir(deps.db, source, current, deps.now());
      } catch (error) {
        if (current === ".") {
          if (errorCode(error) === "ENOENT") {
            finishDiscoveryCycle(deps.db, source, deps.now());
            break;
          }
          if (isTimeout(error)) {
            warnings.add("io-timeout");
            ownPending(runtime, openPromise.then((dir) => dir.close()).catch(() => undefined));
          } else {
            warnings.add("root-unavailable");
          }
          rootUnavailable = true;
          break;
        }
        if (errorCode(error) === "ENOENT" && current.endsWith("/subagents")) {
          markDirDone(deps.db, source, current, deps.now());
          continue;
        }
        if (isTimeout(error)) {
          warnings.add("io-timeout");
          ownPending(runtime, openPromise.then((dir) => dir.close()).catch(() => undefined));
          return rootUnavailable;
        }
        warnings.add("subdir-failed");
        markDirDone(deps.db, source, current, deps.now());
        continue;
      }
    }
    const iterator = runtime.iterator;
    if (!iterator) break;
    const dirChild = current !== "." && current.endsWith("/subagents");
    let exhausted = false;
    while (entries < JSONL_BUDGET.discoveryEntriesPerPass && deps.nowMs() - start < budgetMs) {
      const readPromise = iterator.dir.read();
      let entry: DirEntryLike | null;
      try {
        entry = await boundedIo(readPromise, JSONL_BUDGET.wallDeadlineMs);
      } catch (error) {
        ownPending(runtime, readPromise, () => iterator.dir.close());
        runtime.iterator = null;
        setDiscoveryCurrentDir(deps.db, source, current, deps.now());
        warnings.add(isTimeout(error) ? "io-timeout" : "discovery-failed");
        return rootUnavailable;
      }
      if (entry === null) {
        exhausted = true;
        break;
      }
      entries += 1;
      try {
        await registerEntry(ctx, current, entry, dirChild, JSONL_BUDGET.wallDeadlineMs);
      } catch (error) {
        if (isTimeout(error)) {
          warnings.add("io-timeout");
          setDiscoveryCurrentDir(deps.db, source, current, deps.now());
          return rootUnavailable;
        }
        warnings.add("unsupported-layout");
      }
    }
    if (exhausted) {
      const closePromise = iterator.dir.close();
      try {
        await boundedIo(closePromise, JSONL_BUDGET.wallDeadlineMs);
      } catch {
        ownPending(runtime, closePromise);
      }
      runtime.iterator = null;
      markDirDone(deps.db, source, current, deps.now());
      activeDir = null;
      continue;
    }
    setDiscoveryCurrentDir(deps.db, source, current, deps.now());
    return rootUnavailable;
  }
  if (!hasPendingDirs(deps.db, source) && runtime.iterator === null) {
    finishDiscoveryCycle(deps.db, source, deps.now());
  }
  return rootUnavailable;
}

type IngestionOutcome = {
  filesProcessed: number;
  items: number;
  bytes: number;
  overranReserve: boolean;
  rootUnavailable: boolean;
};

async function ingestFile(
  deps: NativeCollectorDeps,
  runtime: SourceRuntime,
  candidate: DiscoveryCandidate,
  source: "claude" | "codex",
  realRoot: string,
  passStart: number,
  maxSourceBytes: number,
  warnings: ReturnType<typeof warningBag>,
): Promise<{ processed: boolean; items: number; bytes: number }> {
  const cp = getFileCheckpoint(deps.db, source, candidate.identity);
  const abs = absOf(deps.roots[source], candidate.identity);
  let real: string | null;
  try {
    real = await containReal(deps.fs, realRoot, abs, JSONL_BUDGET.wallDeadlineMs);
  } catch (error) {
    if (isTimeout(error)) {
      warnings.add("io-timeout");
      ownPending(runtime, deps.fs.realpath(abs));
      return { processed: false, items: 0, bytes: 0 };
    }
    real = null;
  }
  if (real === null) {
    warnings.add("symlink-escape");
    markFileFailure(deps.db, source, candidate.identity, deps.now() + JSONL_BUDGET.failureRetryMs / 1000, deps.now());
    return { processed: false, items: 0, bytes: 0 };
  }
  let info: StatLike;
  try {
    info = await boundedIo(deps.fs.stat(real), JSONL_BUDGET.wallDeadlineMs);
  } catch (error) {
    if (isTimeout(error)) {
      warnings.add("io-timeout");
      ownPending(runtime, deps.fs.stat(real));
      return { processed: false, items: 0, bytes: 0 };
    }
    warnings.add("missing-file");
    markFileFailure(deps.db, source, candidate.identity, deps.now() + JSONL_BUDGET.failureRetryMs / 1000, deps.now());
    return { processed: false, items: 0, bytes: 0 };
  }
  if (!info.isFile()) {
    warnings.add("missing-file");
    markFileFailure(deps.db, source, candidate.identity, deps.now() + JSONL_BUDGET.failureRetryMs / 1000, deps.now());
    return { processed: false, items: 0, bytes: 0 };
  }
  const generation = `${info.dev}:${info.ino}`;
  const sameGeneration = cp !== undefined && cp.generation === generation;
  const mtime = Math.floor(info.mtimeMs / 1000);
  const current = sameGeneration && cp!.done && !cp!.discarding && cp!.size === info.size && cp!.mtime === mtime;
  if (current) {
    markFileCurrent(deps.db, source, candidate.identity, info.size, mtime, deps.now());
    return { processed: false, items: 0, bytes: 0 };
  }

  const startOffset = sameGeneration && cp!.offset <= info.size ? cp!.offset : 0;
  let offset = startOffset;
  let lineStart = startOffset;
  let carry = Buffer.alloc(0);
  let discarding = sameGeneration && cp!.offset <= info.size ? cp!.discarding === true : false;
  if (discarding) warnings.add("line-too-long");
  let done = false;
  let fileBytes = 0;
  let sourceBytes = 0;
  let deadlineHit = false;
  const items: NativeUsageItem[] = [];
  const contexts: NativeContextRow[] = [];
  let handle: FileHandleLike | null = null;
  let handleOwnedByPending = false;
  const openPromise = deps.fs.open(real);
  try {
    handle = await boundedIo(openPromise, JSONL_BUDGET.wallDeadlineMs);
  } catch (error) {
    if (isTimeout(error)) {
      warnings.add("io-timeout");
      ownPending(runtime, openPromise.then((opened) => opened.close()).catch(() => undefined));
      return { processed: false, items: 0, bytes: 0 };
    }
    warnings.add("read-failed");
    markFileFailure(deps.db, source, candidate.identity, deps.now() + JSONL_BUDGET.failureRetryMs / 1000, deps.now());
    return { processed: false, items: 0, bytes: 0 };
  }
  if (handle.stat) {
    try {
      const opened = await boundedIo(handle.stat(), JSONL_BUDGET.wallDeadlineMs);
      if (`${opened.dev}:${opened.ino}` !== generation) {
        warnings.add("file-replaced");
        await handle.close().catch(() => undefined);
        return { processed: false, items: 0, bytes: 0 };
      }
    } catch {
      /* identity is re-checked before commit */
    }
  }

  try {
    while (fileBytes < JSONL_BUDGET.stepBytes && sourceBytes < maxSourceBytes) {
      const elapsed = deps.nowMs() - passStart;
      if (deadlineHit) break;
      if (carry.length === 0 && elapsed >= JSONL_BUDGET.startLineDeadlineMs) {
        deadlineHit = true;
        break;
      }
      if (elapsed >= JSONL_BUDGET.softDeadlineMs) {
        deadlineHit = true;
        break;
      }
      const remainingLineMs = JSONL_BUDGET.lineDeadlineMs;
      const bounded = Math.max(
        1,
        Math.min(JSONL_BUDGET.chunkBytes, JSONL_BUDGET.stepBytes - fileBytes, maxSourceBytes - sourceBytes),
      );
      const chunk = Buffer.alloc(bounded);
      const activeHandle = handle;
      const readPromise = activeHandle.read(chunk, 0, bounded, offset);
      let read: { bytesRead: number };
      try {
        read = await boundedIo(readPromise, remainingLineMs);
      } catch (error) {
        if (isTimeout(error)) {
          warnings.add("io-timeout");
          handleOwnedByPending = true;
          ownPending(runtime, readPromise, () => activeHandle.close());
          return { processed: false, items: 0, bytes: sourceBytes };
        }
        warnings.add("read-failed");
        break;
      }
      if (read.bytesRead <= 0) {
        done = true;
        break;
      }
      offset += read.bytesRead;
      fileBytes += read.bytesRead;
      sourceBytes += read.bytesRead;
      carry = Buffer.concat([carry, chunk.subarray(0, read.bytesRead)]);
      for (;;) {
        const newline = carry.indexOf(10);
        if (newline < 0) {
          if (carry.length > JSONL_BUDGET.maxLineBytes) {
            warnings.add("line-too-long");
            carry = Buffer.alloc(0);
            lineStart = offset;
            discarding = true;
          }
          break;
        }
        const nextLineStart = offset - carry.length + newline + 1;
        if (discarding) {
          discarding = false;
          carry = carry.subarray(newline + 1);
          lineStart = nextLineStart;
          continue;
        }
        const lineBytes = carry.subarray(0, newline);
        carry = carry.subarray(newline + 1);
        lineStart = nextLineStart;
        if (lineBytes.length > JSONL_BUDGET.maxLineBytes) {
          warnings.add("line-too-long");
        } else {
          const text = lineBytes.toString("utf8");
          for (const line of text.split("\n")) {
            if (line.trim() === "") continue;
            if (source === "claude") {
              const out = parseClaudeLine(line, { child: candidate.child, now: deps.now() });
              if ("item" in out) items.push(out.item);
              else if (out.skip !== "not-assistant") warnings.add(out.skip);
            } else {
              const out = parseCodexLine(line, { fileIdentity: candidate.identity, now: deps.now() });
              if ("item" in out) items.push(out.item);
              else if ("context" in out) contexts.push(out.context);
              else warnings.add(out.skip);
            }
          }
        }
        if (deps.nowMs() - passStart >= JSONL_BUDGET.startLineDeadlineMs) {
          deadlineHit = true;
          break;
        }
      }
      if (deadlineHit) break;
    }
  } finally {
    if (handle && !handleOwnedByPending) {
      try {
        await boundedIo(handle.close(), JSONL_BUDGET.wallDeadlineMs);
      } catch {
        /* the process exit path releases the descriptor */
      }
    }
  }

  if (done) {
    if (discarding) {
      // EOF inside an oversized line: keep the discard marker so a later append
      // continues discarding instead of parsing a fragment as a new line.
      lineStart = offset;
    } else if (carry.length > 0) {
      done = false;
    }
  }
  const consumed = done ? offset : discarding ? offset : lineStart;
  const completeAtEof = done && carry.length === 0;

  const expectedOffset = sameGeneration ? cp!.offset : 0;
  const expectNewGeneration = cp !== undefined && !sameGeneration;
  const checkpoint: FileCheckpoint = {
    source,
    identity: candidate.identity,
    child: candidate.child,
    generation,
    dev: String(info.dev),
    ino: String(info.ino),
    size: info.size,
    mtime,
    offset: consumed,
    discarding,
    done: completeAtEof,
    processedAt: deps.now(),
  };
  try {
    const verify = await boundedIo(deps.fs.stat(real), JSONL_BUDGET.wallDeadlineMs);
    if (`${verify.dev}:${verify.ino}` !== generation) {
      warnings.add("file-replaced");
      return { processed: false, items: 0, bytes: sourceBytes };
    }
  } catch {
    warnings.add("read-failed");
    return { processed: false, items: 0, bytes: sourceBytes };
  }
  try {
    applyFileBatch(deps.db, {
      checkpoint,
      expectedOffset,
      expectNewGeneration,
      items,
      contexts,
      now: deps.now(),
    });
  } catch {
    warnings.add("checkpoint-conflict");
    return { processed: false, items: 0, bytes: sourceBytes };
  }
  return { processed: true, items: items.length, bytes: sourceBytes };
}

async function ingestionPass(
  deps: NativeCollectorDeps,
  source: "claude" | "codex",
  realRoot: string,
  passStart: number,
  warnings: ReturnType<typeof warningBag>,
): Promise<IngestionOutcome> {
  const outcome: IngestionOutcome = {
    filesProcessed: 0,
    items: 0,
    bytes: 0,
    overranReserve: false,
    rootUnavailable: false,
  };
  const now = deps.now();
  const today = utcDay(now);
  const todayStart = Date.parse(`${today}T00:00:00Z`) / 1000;
  const candidates = selectIngestionCandidates(deps.db, source, todayStart, now, JSONL_BUDGET.maxFilesPerPass);
  for (const candidate of candidates) {
    if (outcome.filesProcessed >= JSONL_BUDGET.maxFilesPerPass) break;
    if (outcome.bytes >= JSONL_BUDGET.maxBytesPerSource) break;
    const elapsed = deps.nowMs() - passStart;
    if (elapsed >= JSONL_BUDGET.startLineDeadlineMs) {
      outcome.overranReserve = true;
      break;
    }
    const result = await ingestFile(
      deps,
      runtimeFor(deps.db, source),
      candidate,
      source,
      realRoot,
      passStart,
      JSONL_BUDGET.maxBytesPerSource - outcome.bytes,
      warnings,
    );
    outcome.filesProcessed += result.processed ? 1 : 0;
    outcome.items += result.items;
    outcome.bytes += result.bytes;
  }
  return outcome;
}

function stateFor(
  source: "claude" | "codex",
  db: Database.Database,
  now: number,
  coverageFailure: boolean,
): SourceState {
  const counts = getRegistryCounts(db, source, now);
  const discovery = db
    .prepare("SELECT state FROM native_usage_discovery WHERE source = ?")
    .get(source) as { state: "active" | "complete" } | undefined;
  const discoveryActive = discovery?.state === "active";
  if (discoveryActive || counts.pending > 0) return "indexing";
  if (counts.total === 0) return coverageFailure ? "unavailable" : "empty";
  return "available";
}

const coverageFailureCodes = new Set(["subdir-failed", "discovery-failed", "io-timeout"]);

async function runJsonlSourcePass(
  deps: NativeCollectorDeps,
  source: "claude" | "codex",
): Promise<{
  items: number;
  filesProcessed: number;
  filesTotal: number;
  filesDone: number;
  warnings: string[];
  state: SourceState;
  stalled: boolean;
}> {
  const root = deps.roots[source];
  const warnings = warningBag();
  const runtime = runtimeFor(deps.db, source);
  const passStart = deps.nowMs();

  if (runtime.pending) {
    warnings.add("io-pending");
    const counts = getRegistryCounts(deps.db, source, deps.now());
    return {
      items: 0,
      filesProcessed: 0,
      filesTotal: counts.total,
      filesDone: counts.done,
      warnings: warnings.values,
      state: "indexing",
      stalled: true,
    };
  }

  const rootState = await resolveRoot(deps, root, Math.max(1, JSONL_BUDGET.wallDeadlineMs));
  if (!rootState.ok) {
    if (rootState.missing) {
      const counts = getRegistryCounts(deps.db, source, deps.now());
      return {
        items: 0,
        filesProcessed: 0,
        filesTotal: counts.total,
        filesDone: counts.done,
        warnings: warnings.values,
        state: counts.total === 0 ? "empty" : "indexing",
        stalled: false,
      };
    }
    return {
      items: 0,
      filesProcessed: 0,
      filesTotal: countRegisteredFiles(deps.db, source),
      filesDone: getRegistryCounts(deps.db, source, deps.now()).done,
      warnings: ["root-unavailable"],
      state: "unavailable",
      stalled: false,
    };
  }

  const knownCount = countRegisteredFiles(deps.db, source);
  const discoveryFirst = runtime.discoveryFirstNext || knownCount === 0;
  runtime.discoveryFirstNext = false;
  let rootUnavailable = false;
  let ingestion: IngestionOutcome | null = null;

  const runDiscovery = async (budget: number) => {
    const unavailable = await discoveryPass(deps, runtime, source, rootState.realRoot, Math.max(0, budget), warnings);
    if (unavailable) rootUnavailable = true;
  };

  if (discoveryFirst) {
    await runDiscovery(JSONL_BUDGET.discoveryMsPerPass);
    ingestion = await ingestionPass(deps, source, rootState.realRoot, passStart, warnings);
    if (ingestion.overranReserve) runtime.discoveryFirstNext = true;
  } else {
    ingestion = await ingestionPass(deps, source, rootState.realRoot, passStart, warnings);
    if (ingestion.overranReserve) runtime.discoveryFirstNext = true;
    // The final 50 ms of the scheduling window is reserved for discovery.
    const elapsed = deps.nowMs() - passStart;
    const remaining = JSONL_BUDGET.softDeadlineMs - elapsed;
    if (remaining > 0) await runDiscovery(Math.min(JSONL_BUDGET.discoveryMsPerPass, remaining));
  }

  const counts = getRegistryCounts(deps.db, source, deps.now());
  const coverageFailure = warnings.values.some((code) => coverageFailureCodes.has(code));
  const state = rootUnavailable
    ? "unavailable"
    : stateFor(source, deps.db, deps.now(), coverageFailure);
  return {
    items: ingestion?.items ?? 0,
    filesProcessed: ingestion?.filesProcessed ?? 0,
    filesTotal: counts.total,
    filesDone: counts.done,
    warnings: warnings.values,
    state,
    stalled: false,
  };
}

// ---------------------------------------------------------------------------
// Fixed OpenCode metadata helper. Raw rowid pages are selected BEFORE
// assistant/type/JSON filtering; BLOBs are fetched one bounded row at a time.

export const OPENCODE_HELPER_SOURCE = String.raw`(function(){
  function emit(value){ process.stdout.write(JSON.stringify(value)); }
  function fail(code){ emit({v:1,ok:false,error:code}); return; }
  var argv = process.argv || [];
  if (argv.length < 2) { fail("bad-argv"); return; }
  var dbPath = argv[argv.length - 2];
  var stateRaw = argv[argv.length - 1];
  if (typeof dbPath !== "string" || dbPath.indexOf("\u0000") !== -1 || Buffer.byteLength(dbPath, "utf8") > 4096) { fail("bad-path"); return; }
  if (typeof stateRaw !== "string" || Buffer.byteLength(stateRaw, "utf8") > 16384) { fail("bad-state"); return; }
  var state;
  try { state = JSON.parse(stateRaw); } catch (e) { fail("bad-state"); return; }
  if (!state || state.v !== 1) { fail("bad-state"); return; }
  var MIN_ROWID = "-9223372036854775808";
  function rowId(v){ if (typeof v !== "string" || !/^-?\d{1,20}$/.test(v)) return null; try { var n = BigInt(v); if (n < BigInt(MIN_ROWID) || n > BigInt("9223372036854775807")) return null; return v; } catch (e) { return null; } }
  function cursorOf(c){ if (!c || typeof c !== "object") return null; var after = c.after === null || c.after === undefined ? null : rowId(c.after); if (c.after !== null && c.after !== undefined && after === null) return null; var ceiling = c.ceiling === null || c.ceiling === undefined ? null : rowId(c.ceiling); if (c.ceiling !== null && c.ceiling !== undefined && ceiling === null) return null; if (after !== null && ceiling !== null && BigInt(after) > BigInt(ceiling)) return null; return { after: after, ceiling: ceiling, done: c.done === true }; }
  var v1 = cursorOf(state.v1), v2 = cursorOf(state.v2);
  if (!v1 || !v2) { fail("bad-state"); return; }
  var Database;
  try { Database = require("bun:sqlite").Database; } catch (e) { fail("runtime-fail"); return; }
  var db;
  try { db = new Database(dbPath, { readonly: true, strict: true }); } catch (e) { fail("open-failed"); return; }
  var tables;
  try { tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table'").all().map(function(r){ return r.name; }); }
  catch (e) { fail("schema-fail-closed"); return; }
  var required = ["message", "migration", "session", "session_message"];
  for (var ti = 0; ti < required.length; ti += 1) { if (tables.indexOf(required[ti]) < 0) { fail("schema-fail-closed"); return; } }
  var need = {
    session: ["id", "parent_id"],
    message: ["id", "session_id", "time_created", "time_updated", "data"],
    session_message: ["id", "session_id", "type", "time_created", "time_updated", "data"]
  };
  for (var nt in need) {
    var cols;
    try { cols = db.prepare("PRAGMA table_info(" + nt + ")").all().map(function(r){ return r.name; }); }
    catch (e) { fail("schema-fail-closed"); return; }
    for (var ci = 0; ci < need[nt].length; ci += 1) { if (cols.indexOf(need[nt][ci]) < 0) { fail("schema-fail-closed"); return; } }
  }
  var MAX_ROWS = 256, TABLE_BYTES = 122880, MAX_SINGLE_ROW = 16384, MAX_ID = 256, MAX_ROW = 1048576;
  function safeCount(v){ return typeof v === "number" && isFinite(v) && Math.floor(v) === v && v >= 0 && v <= 9007199254740991 ? v : null; }
  function idOk(v){ if (typeof v !== "string" || v.length === 0) return false; var n = 0; for (var i = 0; i < v.length; i += 1) { var c = v.charCodeAt(i); if (c >= 0xD800 && c <= 0xDBFF && i + 1 < v.length) { var d = v.charCodeAt(i + 1); if (d >= 0xDC00 && d <= 0xDFFF) i += 1; } n += 1; if (n > MAX_ID) return false; } return n > 0; }
  function obj(v){ return v && typeof v === "object" && !Array.isArray(v) ? v : null; }
  function num(v){ return typeof v === "number" && isFinite(v) && v >= 0 ? v : null; }
  var warnings = { malformed: 0, overlength: 0, unsupported: 0, budget: 0 };
  var rows = [];
  function project(projection, meta, data){
    var tokens = null, modelId = null, providerId = null, agent = null, cost = null, created = null;
    if (projection === "v2") {
      var model = obj(data.model); tokens = obj(data.tokens); var time = obj(data.time);
      modelId = model && typeof model.id === "string" ? model.id : null;
      providerId = model && typeof model.providerID === "string" ? model.providerID : null;
      agent = typeof data.agent === "string" ? data.agent : null;
      cost = num(data.cost);
      created = time && typeof time.created === "number" ? time.created : meta.time_created;
    } else {
      if (data.role !== "assistant") return null;
      tokens = obj(data.tokens);
      modelId = typeof data.modelID === "string" ? data.modelID : null;
      providerId = typeof data.providerID === "string" ? data.providerID : null;
      agent = typeof data.agent === "string" ? data.agent : null;
      cost = num(data.cost);
      created = typeof data.created === "number" ? data.created : meta.time_created;
    }
    if (!idOk(meta.id) || !idOk(meta.session_id)) { warnings.overlength += 1; return null; }
    if (!tokens) { warnings.unsupported += 1; return null; }
    var input, output, reasoning, cacheRead, cacheWrite;
    if (projection === "v2") {
      var cache = obj(tokens.cache) || {};
      input = safeCount(tokens.input); output = safeCount(tokens.output);
      reasoning = tokens.reasoning === undefined ? 0 : safeCount(tokens.reasoning);
      cacheRead = cache.read === undefined ? 0 : safeCount(cache.read);
      cacheWrite = cache.write === undefined ? 0 : safeCount(cache.write);
    } else {
      input = safeCount(tokens.input); output = safeCount(tokens.output);
      reasoning = tokens.reasoning === undefined ? 0 : safeCount(tokens.reasoning);
      cacheRead = tokens.read === undefined ? 0 : safeCount(tokens.read);
      cacheWrite = tokens.write === undefined ? 0 : safeCount(tokens.write);
    }
    if (input === null || output === null || reasoning === null || cacheRead === null || cacheWrite === null) { warnings.unsupported += 1; return null; }
    if (typeof created !== "number" || !isFinite(created) || created <= 0) { warnings.unsupported += 1; return null; }
    return {
      projection: projection,
      session_id: meta.session_id,
      message_id: meta.id,
      parent_session_id: meta.parent_id === undefined ? null : meta.parent_id,
      model_id: modelId,
      provider_id: providerId,
      agent: agent,
      created: created,
      updated: typeof meta.time_updated === "number" ? meta.time_updated : created,
      cost: cost,
      input: input,
      output: output,
      reasoning: reasoning,
      cache_read: cacheRead,
      cache_write: cacheWrite
    };
  }
  function pageOnce(projection, tableName, cursor){
    var tableBytes = 0;
    var page = { after: cursor.after, ceiling: cursor.ceiling, done: cursor.done };
    if (page.done) return page;
    if (page.ceiling === null) {
      var maxRow;
      try { maxRow = db.prepare("SELECT CAST(MAX(rowid) AS TEXT) AS c FROM " + tableName).get(); }
      catch (e) { throw new Error("query-failed"); }
      page.ceiling = maxRow && typeof maxRow.c === "string" ? rowId(maxRow.c) : null;
      if (page.ceiling === null) { page.done = true; return page; }
    }
    var metaSql = projection === "v2"
      ? "SELECT CAST(m.rowid AS TEXT) AS rid, m.id AS id, m.session_id AS session_id, m.type AS type, m.time_created AS time_created, m.time_updated AS time_updated, typeof(m.data) AS dtype, length(CAST(m.data AS BLOB)) AS dlen, s.parent_id AS parent_id FROM session_message m LEFT JOIN session s ON s.id = m.session_id WHERE m.rowid > CAST(? AS INTEGER) AND m.rowid <= CAST(? AS INTEGER) ORDER BY m.rowid LIMIT " + MAX_ROWS
      : "SELECT CAST(m.rowid AS TEXT) AS rid, m.id AS id, m.session_id AS session_id, 'assistant' AS type, m.time_created AS time_created, m.time_updated AS time_updated, typeof(m.data) AS dtype, length(CAST(m.data AS BLOB)) AS dlen, s.parent_id AS parent_id FROM message m LEFT JOIN session s ON s.id = m.session_id WHERE m.rowid > CAST(? AS INTEGER) AND m.rowid <= CAST(? AS INTEGER) ORDER BY m.rowid LIMIT " + MAX_ROWS;
    var raw;
    try {
      raw = page.after === null
        ? db.prepare(metaSql.replace("m.rowid > CAST(? AS INTEGER)", "m.rowid >= CAST(? AS INTEGER)")).all(MIN_ROWID, page.ceiling)
        : db.prepare(metaSql).all(page.after, page.ceiling);
    } catch (e) { throw new Error("query-failed"); }
    if (!raw || raw.length === 0) { page.done = true; return page; }
    var prev = page.after;
    var dataSql = "SELECT data FROM " + tableName + " WHERE rowid = CAST(? AS INTEGER)";
    for (var i = 0; i < raw.length; i += 1) {
      var meta = raw[i];
      if (typeof meta.rid !== "string") { throw new Error("query-failed"); }
      if (projection === "v2" && meta.type !== "assistant") { prev = meta.rid; page.after = meta.rid; continue; }
      if (meta.dtype !== "text") { warnings.unsupported += 1; prev = meta.rid; page.after = meta.rid; continue; }
      if (typeof meta.dlen !== "number" || meta.dlen > MAX_ROW) { warnings.unsupported += 1; prev = meta.rid; page.after = meta.rid; continue; }
      if (!idOk(meta.id) || !idOk(meta.session_id)) { warnings.overlength += 1; prev = meta.rid; page.after = meta.rid; continue; }
      var data;
      try { data = JSON.parse(db.prepare(dataSql).get(meta.rid).data); } catch (e) { warnings.malformed += 1; prev = meta.rid; page.after = meta.rid; continue; }
      if (!obj(data)) { warnings.malformed += 1; prev = meta.rid; page.after = meta.rid; continue; }
      if (projection === "v1" && data.role !== "assistant") { prev = meta.rid; page.after = meta.rid; continue; }
      var projected = project(projection, meta, data);
      if (projected) {
        var encoded = JSON.stringify(projected);
        var size = Buffer.byteLength(encoded, "utf8");
        if (size > MAX_SINGLE_ROW) { warnings.unsupported += 1; prev = meta.rid; page.after = meta.rid; continue; }
        if (tableBytes + size > TABLE_BYTES) { warnings.budget += 1; page.after = prev; page.done = false; return page; }
        rows.push(projected);
        tableBytes += size;
      }
      prev = meta.rid;
      page.after = meta.rid;
    }
    if (page.after === null || BigInt(page.after) >= BigInt(page.ceiling)) page.done = true;
    else if (raw.length < MAX_ROWS) page.done = true;
    return page;
  }
  var nextState;
  try {
    nextState = { v1: pageOnce("v1", "message", v1), v2: pageOnce("v2", "session_message", v2) };
  } catch (e) { fail("query-failed"); return; }
  var reply = {
    v: 1,
    ok: true,
    runtime: process.versions && process.versions.bun ? "Bun " + process.versions.bun : "unknown",
    schema: "1.18.34",
    rows: rows,
    next: nextState,
    warnings: warnings
  };
  if (Buffer.byteLength(JSON.stringify(reply), "utf8") > 262144) { fail("output-overflow"); return; }
  emit(reply);
})();`;

export const OPENCODE_RUNTIME = "Bun 1.3.14";
export const OPENCODE_SCHEMA = "1.18.34";

const executableIdentityCache = new Map<string, string>();

async function openCodeExecutableState(
  deps: NativeCollectorDeps,
): Promise<{ ok: true } | { ok: false; state: SourceState; warning: string }> {
  let info: StatLike | null;
  try {
    info = await deps.opencode.stat(deps.opencode.executable);
  } catch {
    info = null;
  }
  if (!info || !info.isFile()) return { ok: false, state: "unavailable", warning: "executable-missing" };
  const identity = `${deps.opencode.executable}:${info.dev}:${info.ino}:${info.size}:${Math.floor(info.mtimeMs)}`;
  const cachedHash = executableIdentityCache.get(identity);
  const hash = cachedHash ?? (await deps.opencode.hash(deps.opencode.executable)) ?? null;
  if (hash !== null && cachedHash === undefined) executableIdentityCache.set(identity, hash);
  if (hash !== OPENCODE_EXECUTABLE_SHA256) {
    return { ok: false, state: "incompatible", warning: "executable-mismatch" };
  }
  return { ok: true };
}

type OpenCodeReply = {
  v: 1;
  ok: true;
  runtime: string;
  schema: string;
  rows: unknown[];
  next: { v1: OpenCodeCursorState["v1"]; v2: OpenCodeCursorState["v2"] };
  warnings: Record<string, number>;
};

export function parseOpenCodeReply(
  stdout: string,
  previous: OpenCodeCursorState,
): OpenCodeReply | null {
  if (Buffer.byteLength(stdout, "utf8") > OPENCODE_BUDGET.maxReplyBytes) return null;
  let raw: unknown;
  try {
    raw = JSON.parse(stdout.trim());
  } catch {
    return null;
  }
  if (!isObj(raw) || raw.v !== 1 || raw.ok !== true || !Array.isArray(raw.rows)) return null;
  if (raw.runtime !== OPENCODE_RUNTIME || raw.schema !== OPENCODE_SCHEMA) return null;
  const next = raw.next;
  if (!isObj(next) || !isObj(next.v1) || !isObj(next.v2)) return null;
  const sanitize = (
    cursor: Record<string, unknown>,
    prior: OpenCodeCursorState["v1"],
  ): OpenCodeCursorState["v1"] | null => {
    const after = cursor.after === null ? null : canonicalDecimal(cursor.after);
    if (after === null && cursor.after !== null) return null;
    const ceiling = cursor.ceiling === null ? null : canonicalDecimal(cursor.ceiling);
    if (ceiling === null && cursor.ceiling !== null) return null;
    if (after !== null && ceiling !== null && BigInt(after) > BigInt(ceiling)) return null;
    if (prior.ceiling !== null && ceiling !== null && BigInt(ceiling) !== BigInt(prior.ceiling)) return null;
    if (prior.after !== null && after !== null && BigInt(after) < BigInt(prior.after)) return null;
    if (prior.after !== null && after === null && !(prior.done && cursor.done === true)) return null;
    if (prior.ceiling !== null && ceiling === null && !(prior.done && cursor.done === true)) return null;
    if (prior.done && !cursor.done && after !== null && prior.after !== null && BigInt(after) <= BigInt(prior.after)) return null;
    return { after, ceiling, done: cursor.done === true };
  };
  const v1 = sanitize(next.v1, previous.v1);
  const v2 = sanitize(next.v2, previous.v2);
  if (!v1 || !v2) return null;
  const warnings: Record<string, number> = {};
  if (isObj(raw.warnings)) {
    for (const [key, value] of Object.entries(raw.warnings)) {
      if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) warnings[key] = value;
    }
  }
  return {
    v: 1,
    ok: true,
    runtime: raw.runtime,
    schema: raw.schema,
    rows: raw.rows,
    next: { v1, v2 },
    warnings,
  };
}

function canonicalDecimal(value: unknown): string | null {
  if (typeof value !== "string" || !/^-?\d{1,20}$/.test(value)) return null;
  try {
    const numeric = BigInt(value);
    if (numeric < BigInt("-9223372036854775808") || numeric > BigInt("9223372036854775807")) return null;
  } catch {
    return null;
  }
  return value;
}

export async function runOpenCodeSourcePass(deps: NativeCollectorDeps): Promise<{
  items: number;
  warnings: string[];
  state: SourceState;
  lastRefresh: number;
  progress: { filesDone: number; filesTotal: number } | null;
}> {
  if (deps.opencode.ensureAdapter) await deps.opencode.ensureAdapter();
  const executable = await openCodeExecutableState(deps);
  if (!executable.ok) {
    return { items: 0, warnings: [executable.warning], state: executable.state, lastRefresh: 0, progress: null };
  }
  const dbPath = deps.roots.opencodeDb;
  if (dbPath.includes("\u0000") || Buffer.byteLength(dbPath, "utf8") > 4096) {
    return { items: 0, warnings: ["db-invalid"], state: "unavailable", lastRefresh: 0, progress: null };
  }
  try {
    const info = await deps.fs.stat(dbPath);
    if (!info.isFile()) throw new Error("not-file");
  } catch {
    return { items: 0, warnings: ["db-missing"], state: "unavailable", lastRefresh: 0, progress: null };
  }
  let real: string | null;
  try {
    const realRoot = await deps.fs.realpath(deps.roots.opencodeDataRoot);
    const realDb = await deps.fs.realpath(dbPath);
    real = insideRealRoot(realRoot, realDb) ? realDb : null;
  } catch {
    real = null;
  }
  if (!real) return { items: 0, warnings: ["db-invalid"], state: "unavailable", lastRefresh: 0, progress: null };

  const cursor = getOpenCodeCursor(deps.db);
  const argv = [
    "--no-env-file",
    `--config=${deps.opencode.configPath}`,
    `--cwd=${deps.opencode.cwd}`,
    "-e",
    OPENCODE_HELPER_SOURCE,
    real,
    JSON.stringify(cursor),
  ];
  let stdout: string;
  try {
    const result = await deps.opencode.exec(deps.opencode.executable, argv, {
      cwd: deps.opencode.cwd,
      env: { BUN_BE_BUN: "1", XDG_CONFIG_HOME: deps.opencode.xdgConfigHome },
      timeout: OPENCODE_BUDGET.timeoutMs,
      maxBuffer: OPENCODE_BUDGET.maxReplyBytes,
    });
    stdout = result.stdout;
  } catch {
    return { items: 0, warnings: ["exec-failed"], state: "unavailable", lastRefresh: 0, progress: null };
  }

  let parsedRaw: unknown;
  try {
    parsedRaw = JSON.parse(stdout.trim());
  } catch {
    parsedRaw = null;
  }
  if (isObj(parsedRaw) && parsedRaw.ok === false) {
    const error = typeof parsedRaw.error === "string" ? parsedRaw.error : "helper-failed";
    const state: SourceState = error === "schema-fail-closed" || error === "runtime-fail" ? "incompatible" : "unavailable";
    return { items: 0, warnings: [error], state, lastRefresh: 0, progress: null };
  }
  const reply = parseOpenCodeReply(stdout, cursor);
  if (!reply) {
    return { items: 0, warnings: ["malformed-reply"], state: "unavailable", lastRefresh: 0, progress: null };
  }

  const items: NativeUsageItem[] = [];
  const warnings = warningBag();
  for (const row of reply.rows) {
    const out = parseOpenCodeRow(row, { now: deps.now() });
    if ("item" in out) items.push(out.item);
    else warnings.add(out.skip);
  }
  for (const [key, count] of Object.entries(reply.warnings)) {
    if (count > 0) warnings.add(key);
  }
  const next: OpenCodeCursorState = { v: 1, v1: reply.next.v1, v2: reply.next.v2 };
  const complete = next.v1.done && next.v2.done;
  const persisted: OpenCodeCursorState = complete
    ? { v: 1, v1: { after: null, ceiling: null, done: false }, v2: { after: null, ceiling: null, done: false } }
    : next;
  persistOpenCodeBatch(deps.db, items, persisted, deps.now());
  return {
    items: items.length,
    warnings: warnings.values,
    state: complete ? "available" : "indexing",
    lastRefresh: deps.now(),
    progress: null,
  };
}

const inFlight = new WeakMap<Database.Database, Map<Harness, Promise<void>>>();

function settingsSources(settings: SettingsResult): Record<Harness, boolean> | null {
  if (!settings.ok) return null;
  const agents = settings.data.integrations.agents;
  return { claude: agents.claude, codex: agents.codex, opencode: agents.opencode };
}

async function runOneSource(deps: NativeCollectorDeps, source: Harness, enabled: boolean): Promise<void> {
  const now = deps.now();
  const statuses = getSourceStatuses(deps.db);
  const previous = statuses.find((s) => s.source === source);
  if (!enabled) {
    setSourceStatus(deps.db, { source, state: "disabled", lastRefresh: previous?.lastRefresh ?? null, progress: null, warnings: [] }, now);
    return;
  }
  if (
    previous &&
    previous.state === "available" &&
    previous.lastRefresh !== null &&
    now - previous.lastRefresh < JSONL_BUDGET.refreshMs / 1000
  ) {
    return;
  }
  if (source === "opencode") {
    try {
      const result = await runOpenCodeSourcePass(deps);
      setSourceStatus(
        deps.db,
        {
          source,
          state: result.state,
          lastRefresh: result.lastRefresh || previous?.lastRefresh || null,
          progress: result.progress,
          warnings: result.warnings,
        },
        deps.now(),
      );
    } catch {
      setSourceStatus(
        deps.db,
        {
          source,
          state: "unavailable",
          lastRefresh: previous?.lastRefresh ?? null,
          progress: previous?.progress ?? null,
          warnings: ["pass-failed"],
        },
        deps.now(),
      );
    }
    return;
  }
  try {
    const result = await runJsonlSourcePass(deps, source);
    setSourceStatus(
      deps.db,
      {
        source,
        state: result.state,
        lastRefresh:
          result.state === "unavailable" || result.stalled ? previous?.lastRefresh ?? null : now,
        progress: { filesDone: result.filesDone, filesTotal: result.filesTotal },
        warnings: result.warnings,
      },
      deps.now(),
    );
  } catch {
    setSourceStatus(
      deps.db,
      {
        source,
        state: "unavailable",
        lastRefresh: previous?.lastRefresh ?? null,
        progress: previous?.progress ?? null,
        warnings: ["pass-failed"],
      },
      deps.now(),
    );
  }
}

export async function runNativeUsagePass(deps: NativeCollectorDeps): Promise<void> {
  try {
    pruneNativeUsage(deps.db, deps.now());
  } catch {
    /* pruning is best-effort; collection still proceeds */
  }
  let map = inFlight.get(deps.db);
  if (!map) {
    map = new Map();
    inFlight.set(deps.db, map);
  }
  const settings = deps.settings();
  const enabled = settingsSources(settings);
  if (enabled === null) {
    for (const source of HARNESSES) {
      setSourceStatus(deps.db, { source, state: "unavailable", lastRefresh: null, progress: null, warnings: ["settings-error"] }, deps.now());
    }
    return;
  }
  await Promise.all(
    HARNESSES.map(async (source) => {
      const running = map!.get(source);
      if (running) return running;
      const pass = runOneSource(deps, source, enabled[source]).finally(() => map!.delete(source));
      map!.set(source, pass);
      return pass;
    }),
  );
}

async function nativeRootsOverride(): Promise<{ claude: string; codex: string; opencode: string } | null> {
  const root = jaxosHome();
  let raw: unknown;
  try {
    raw = JSON.parse(await readFile(join(root, "native-roots.json"), "utf8"));
  } catch {
    return null;
  }
  const invalid = {
    claude: join(root, "native-invalid", "claude"),
    codex: join(root, "native-invalid", "codex"),
    opencode: join(root, "native-invalid", "opencode"),
  };
  if (!isObj(raw)) return invalid;
  const keys = ["claude", "codex", "opencode"] as const;
  for (const key of keys) {
    if (typeof raw[key] !== "string" || !isAbsolute(raw[key])) return invalid;
  }
  try {
    const boundary = await nodeRealpathAsync(root);
    for (const key of keys) {
      const resolved = await nodeRealpathAsync(raw[key] as string);
      if (resolved !== boundary && !resolved.startsWith(base(boundary))) return invalid;
    }
  } catch {
    return invalid;
  }
  return { claude: raw.claude as string, codex: raw.codex as string, opencode: raw.opencode as string };
}

export async function defaultNativeCollectorDeps(db: Database.Database): Promise<NativeCollectorDeps> {
  const home = homedir();
  const dataRoot = process.env.XDG_DATA_HOME || join(home, ".local", "share");
  const adapterRoot = join(jaxosHome(), "native-opencode");
  const cwd = join(adapterRoot, "cwd");
  const configPath = join(adapterRoot, "empty.toml");
  const xdgConfigHome = join(adapterRoot, "config");
  const override = await nativeRootsOverride();
  const pExecFile = promisify(execFile);
  const fs: NativeFs = {
    stat: (path) => nodeStatAsync(path),
    realpath: (path) => nodeRealpathAsync(path),
    opendir: async (path) => (await nodeOpendirAsync(path)) as DirHandleLike,
    open: async (path) => {
      const handle = await nodeOpenAsync(path, "r");
      return {
        read: (buffer, offset, length, position) => handle.read(buffer, offset, length, position),
        close: () => handle.close(),
        stat: () => handle.stat(),
      };
    },
  };
  return {
    db,
    now: () => Math.floor(Date.now() / 1000),
    nowMs: () => Date.now(),
    settings: readGeneralSettings,
    fs,
    roots: override
      ? { claude: override.claude, codex: override.codex, opencodeDataRoot: override.opencode, opencodeDb: join(override.opencode, "opencode.db") }
      : {
          claude: join(home, ".claude", "projects"),
          codex: join(home, ".codex", "sessions"),
          opencodeDataRoot: join(dataRoot, "opencode"),
          opencodeDb: join(dataRoot, "opencode", "opencode.db"),
        },
    opencode: {
      executable: OPENCODE_EXECUTABLE,
      cwd,
      configPath,
      xdgConfigHome,
      ensureAdapter: async () => {
        await mkdir(cwd, { recursive: true });
        await mkdir(xdgConfigHome, { recursive: true });
        try {
          await access(configPath);
        } catch {
          await writeFile(configPath, "", "utf8");
        }
      },
      exec: async (file, args, opts) => {
        const { stdout } = await pExecFile(file, args, {
          cwd: opts.cwd,
          env: opts.env as NodeJS.ProcessEnv,
          timeout: opts.timeout,
          maxBuffer: opts.maxBuffer,
          encoding: "utf8",
        });
        return { stdout: String(stdout) };
      },
      stat: async (path) => {
        try {
          return await nodeStatAsync(path);
        } catch {
          return null;
        }
      },
      hash: async (path) => {
        try {
          return createHash("sha256").update(await readFile(path)).digest("hex");
        } catch {
          return null;
        }
      },
    },
  };
}
