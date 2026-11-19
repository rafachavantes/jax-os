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
  getPersistentWarnings,
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
  commitReserveMs: 100,
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
  // Supplied by the real file handle; an unavailable fstat must never be permission to read.
  stat(): Promise<StatLike>;
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
  configWarnings?: string[];
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
  constructor(readonly started = false) {
    super("io-timeout");
  }
}

function isTimeout(error: unknown): error is IoTimeout {
  return error instanceof IoTimeout;
}

async function raceTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new IoTimeout(true)), Math.max(0, timeoutMs));
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

// Per-source single-flight runtime: at most one directory iterator and one JSONL
// handle, plus the latch that retains ownership of a timed-out unabortable op.
type OwnedOp = { done: boolean };
type PendingOwner = { pending: OwnedOp | null };
type SourceRuntime = PendingOwner & {
  iterator: { dir: DirHandleLike; path: string } | null;
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

// One owned scope per active pass. Every root, discovery and file operation starts
// through the same monotonic wall deadline and is registered exactly once. A timed-out
// operation keeps the pending latch until it settles AND its handle cleanup completes;
// later polls see the latch and start nothing new.
class IoWindow {
  readonly start: number;
  readonly scheduleDeadline: number;
  readonly wallDeadline: number;
  constructor(
    private readonly owner: PendingOwner,
    private readonly clock: () => number,
    totalMs: number = JSONL_BUDGET.wallDeadlineMs,
    scheduleMs: number = JSONL_BUDGET.softDeadlineMs,
  ) {
    this.start = clock();
    this.scheduleDeadline = this.start + scheduleMs;
    this.wallDeadline = this.start + totalMs;
  }
  get now(): number {
    return this.clock();
  }
  get expired(): boolean {
    return this.clock() >= this.wallDeadline;
  }
  get remainingWall(): number {
    return Math.max(0, this.wallDeadline - this.clock());
  }
  get scheduleOpen(): boolean {
    return this.clock() < this.scheduleDeadline;
  }
  // File/line starts stop early enough to leave discovery its 50 ms slice.
  get lineStartOpen(): boolean {
    return this.clock() < this.start + JSONL_BUDGET.startLineDeadlineMs;
  }

  // Invokes the thunk exactly once, only while the absolute deadline has budget.
  async io<T>(
    thunk: () => Promise<T>,
    deadlineAt: number,
    onLate?: (value: T | undefined) => Promise<void> | void,
  ): Promise<T> {
    const budget = Math.min(deadlineAt, this.wallDeadline) - this.clock();
    if (!(budget > 0)) throw new IoTimeout(false);
    const promise = thunk();
    try {
      return await raceTimeout(promise, budget);
    } catch (error) {
      if (error instanceof IoTimeout) this.latch(promise, onLate);
      throw error;
    }
  }

  // Waits on an already-started cleanup operation using only the remaining budget.
  async settle<T>(promise: Promise<T>): Promise<T> {
    return raceTimeout(promise, Math.max(1, this.remainingWall));
  }

  // Retains the original promise and its cleanup through actual settlement.
  latch<T>(promise: Promise<T>, cleanup?: (value: T | undefined) => Promise<void> | void): void {
    const record: OwnedOp = { done: false };
    this.owner.pending = record;
    void (async () => {
      let value: T | undefined;
      try {
        value = await promise;
      } catch {
        value = undefined;
      }
      try {
        if (cleanup) await cleanup(value);
      } catch {
        /* cleanup settled unsuccessfully; the failure stays visible on the source */
      }
    })().finally(() => {
      record.done = true;
      if (this.owner.pending === record) this.owner.pending = null;
    });
  }

  // Closes/cleans up an owned handle without starting more collection work.
  async closeCleanup<T>(start: () => Promise<T>): Promise<boolean> {
    let promise: Promise<T>;
    try {
      promise = start();
    } catch {
      return true;
    }
    try {
      await this.settle(promise);
      return true;
    } catch (error) {
      if (error instanceof IoTimeout) {
        this.latch(promise);
        return false;
      }
      return true;
    }
  }
}

const joinRel = (a: string, b: string) => (a === "." ? b : `${a}/${b}`);
const relDepth = (rel: string) => (rel === "." ? 0 : rel.split("/").length);
const absOf = (root: string, rel: string) => (rel === "." ? root : join(root, ...rel.split("/")));

async function containReal(
  scope: IoWindow,
  fs: NativeFs,
  realRoot: string,
  absPath: string,
): Promise<string | null> {
  let real: string;
  try {
    real = await scope.io(() => fs.realpath(absPath), scope.wallDeadline, () => undefined);
  } catch (error) {
    if (isTimeout(error)) throw error;
    return null;
  }
  return insideRealRoot(realRoot, real) ? real : null;
}

type RootState =
  | { ok: true; realRoot: string }
  | { ok: false; missing: boolean; pending: boolean };

async function resolveRoot(
  scope: IoWindow,
  deps: NativeCollectorDeps,
  root: string,
): Promise<RootState> {
  let info: StatLike;
  try {
    info = await scope.io(() => deps.fs.stat(root), scope.wallDeadline, () => undefined);
  } catch (error) {
    if (isTimeout(error)) return { ok: false, missing: false, pending: error.started };
    if (errorCode(error) === "ENOENT") return { ok: false, missing: true, pending: false };
    return { ok: false, missing: false, pending: false };
  }
  if (!info.isDirectory()) return { ok: false, missing: false, pending: false };
  try {
    const realRoot = await scope.io(() => deps.fs.realpath(root), scope.wallDeadline, () => undefined);
    return { ok: true, realRoot };
  } catch (error) {
    if (isTimeout(error)) return { ok: false, missing: false, pending: error.started };
    if (errorCode(error) === "ENOENT") return { ok: false, missing: true, pending: false };
    return { ok: false, missing: false, pending: false };
  }
}

type DiscoveryOutcome = {
  rootUnavailable: boolean;
  pending: boolean;
  progressed: boolean;
};

type DiscoveryContext = {
  scope: IoWindow;
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
): Promise<void> {
  const { scope, deps, source } = ctx;
  const rel = joinRel(dirPath, entry.name);
  const abs = absOf(deps.roots[source], rel);
  const depth = relDepth(rel);
  const safeDir = async (): Promise<boolean> => {
    if (!entry.isDirectory()) return false;
    const real = await containReal(scope, deps.fs, ctx.realRoot, abs);
    return real !== null;
  };
  const safeFile = async (): Promise<boolean> => {
    const real = await containReal(scope, deps.fs, ctx.realRoot, abs);
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
  scope: IoWindow,
  runtime: SourceRuntime,
  source: "claude" | "codex",
  realRoot: string,
  budgetMs: number,
  warnings: ReturnType<typeof warningBag>,
): Promise<DiscoveryOutcome> {
  const outcome: DiscoveryOutcome = { rootUnavailable: false, pending: false, progressed: false };
  const state = ensureDiscoveryCycle(deps.db, source, deps.now(), JSONL_BUDGET.refreshMs / 1000);
  if (state.state === "complete") return outcome;
  const ctx: DiscoveryContext = {
    scope, deps, runtime, source, realRoot, cycle: state.cycle, warnings,
  };
  const sliceEnd = Math.min(scope.now + budgetMs, scope.scheduleDeadline);
  let entries = 0;
  let activeDir: string | null = state.currentDir;

  while (entries < JSONL_BUDGET.discoveryEntriesPerPass && scope.now < sliceEnd) {
    const current = activeDir ?? nextPendingDir(deps.db, source);
    if (current === null) {
      finishDiscoveryCycle(deps.db, source, deps.now());
      outcome.progressed = true;
      break;
    }
    activeDir = current;
    if (!runtime.iterator || runtime.iterator.path !== current) {
      // Canonical containment is revalidated immediately before every queued directory
      // open: a symlinked directory (including a constructed subagents path) is rejected
      // before it is enumerated.
      let realDir: string;
      try {
        realDir = await scope.io(
          () => deps.fs.realpath(absOf(deps.roots[source], current)),
          scope.wallDeadline,
          () => undefined,
        );
      } catch (error) {
        if (isTimeout(error)) {
          if (error.started) {
            warnings.add("io-timeout");
            outcome.pending = true;
          }
          return outcome;
        }
        warnings.add("symlink-escape");
        markDirDone(deps.db, source, current, deps.now(), "symlink-escape");
        activeDir = null;
        continue;
      }
      if (!insideRealRoot(realRoot, realDir)) {
        warnings.add("symlink-escape");
        markDirDone(deps.db, source, current, deps.now(), "symlink-escape");
        activeDir = null;
        continue;
      }
      const open = scope.io(
        () => deps.fs.opendir(realDir),
        scope.wallDeadline,
        (dir) => dir?.close(),
      );
      let dir: DirHandleLike;
      try {
        dir = await open;
      } catch (error) {
        if (isTimeout(error)) {
          if (error.started) {
            warnings.add("io-timeout");
            outcome.pending = true;
          }
          return outcome;
        }
        if (current === ".") {
          if (errorCode(error) === "ENOENT") {
            finishDiscoveryCycle(deps.db, source, deps.now());
            break;
          }
          warnings.add("root-unavailable");
          outcome.rootUnavailable = true;
          break;
        }
        if (errorCode(error) === "ENOENT" && current.endsWith("/subagents")) {
          markDirDone(deps.db, source, current, deps.now());
          activeDir = null;
          outcome.progressed = true;
          continue;
        }
        warnings.add("subdir-failed");
        markDirDone(deps.db, source, current, deps.now(), "subdir-failed");
        activeDir = null;
        outcome.progressed = true;
        continue;
      }
      runtime.iterator = { dir, path: current };
      setDiscoveryCurrentDir(deps.db, source, current, deps.now());
      outcome.progressed = true;
    }
    const iterator = runtime.iterator;
    if (!iterator) break;
    const dirChild = current !== "." && current.endsWith("/subagents");
    let exhausted = false;
    while (entries < JSONL_BUDGET.discoveryEntriesPerPass && scope.now < sliceEnd) {
      let entry: DirEntryLike | null;
      try {
        entry = await scope.io(
          () => iterator.dir.read(),
          scope.wallDeadline,
          async () => {
            try {
              await iterator.dir.close();
            } finally {
              runtime.iterator = null;
            }
          },
        );
      } catch (error) {
        if (isTimeout(error) && error.started) {
          warnings.add("io-timeout");
          setDiscoveryCurrentDir(deps.db, source, current, deps.now());
          outcome.pending = true;
          return outcome;
        }
        if (isTimeout(error)) {
          setDiscoveryCurrentDir(deps.db, source, current, deps.now());
          return outcome;
        }
        warnings.add("discovery-failed");
        runtime.iterator = null;
        await scope.closeCleanup(() => iterator.dir.close());
        markDirDone(deps.db, source, current, deps.now(), "discovery-failed");
        activeDir = null;
        return outcome;
      }
      if (entry === null) {
        exhausted = true;
        break;
      }
      entries += 1;
      outcome.progressed = true;
      try {
        await registerEntry(ctx, current, entry, dirChild);
      } catch (error) {
        if (isTimeout(error) && error.started) {
          warnings.add("io-timeout");
          setDiscoveryCurrentDir(deps.db, source, current, deps.now());
          outcome.pending = true;
          return outcome;
        }
        if (isTimeout(error)) {
          setDiscoveryCurrentDir(deps.db, source, current, deps.now());
          return outcome;
        }
        warnings.add("unsupported-layout");
      }
    }
    if (exhausted) {
      const closed = await scope.closeCleanup(() => iterator.dir.close());
      runtime.iterator = null;
      if (!closed) {
        warnings.add("io-timeout");
        outcome.pending = true;
        break;
      }
      markDirDone(deps.db, source, current, deps.now());
      activeDir = null;
      continue;
    }
    setDiscoveryCurrentDir(deps.db, source, current, deps.now());
    return outcome;
  }
  if (!hasPendingDirs(deps.db, source) && runtime.iterator === null) {
    finishDiscoveryCycle(deps.db, source, deps.now());
    outcome.progressed = true;
  }
  return outcome;
}

type IngestionOutcome = {
  filesProcessed: number;
  confirmed: number;
  items: number;
  bytes: number;
  overranReserve: boolean;
  rootUnavailable: boolean;
  pending: boolean;
};

type IngestResult = {
  processed: boolean;
  confirmed: boolean;
  items: number;
  bytes: number;
  pending: boolean;
};

// Same-inode truncation advances a scalar generation suffix; dev/inode stay separate
// physical identity checks.
function generationFor(prev: FileCheckpoint | undefined, physical: string, truncated: boolean): string {
  let epoch = 0;
  if (prev) {
    if (prev.generation === physical) epoch = 0;
    else if (prev.generation.startsWith(`${physical}:`)) {
      const parsed = Number(prev.generation.slice(physical.length + 1));
      epoch = Number.isSafeInteger(parsed) && parsed > 0 ? parsed : 0;
    }
  }
  const next = truncated ? epoch + 1 : epoch;
  return next > 0 ? `${physical}:${next}` : physical;
}

async function ingestFile(
  scope: IoWindow,
  deps: NativeCollectorDeps,
  runtime: SourceRuntime,
  candidate: DiscoveryCandidate,
  source: "claude" | "codex",
  realRoot: string,
  maxSourceBytes: number,
  warnings: ReturnType<typeof warningBag>,
): Promise<IngestResult> {
  const id = candidate.identity;
  const retryAt = deps.now() + JSONL_BUDGET.failureRetryMs / 1000;
  const failed = (warning: string): IngestResult => {
    warnings.add(warning);
    markFileFailure(deps.db, source, id, retryAt, deps.now(), warning);
    return { processed: false, confirmed: false, items: 0, bytes: 0, pending: false };
  };
  const outstanding = (bytes = 0, warning = "io-timeout"): IngestResult => {
    warnings.add(warning);
    return { processed: false, confirmed: false, items: 0, bytes, pending: true };
  };
  const stop = (bytes = 0): IngestResult => ({
    processed: false, confirmed: false, items: 0, bytes, pending: false,
  });

  const cp = getFileCheckpoint(deps.db, source, id);
  const abs = absOf(deps.roots[source], id);
  let real: string | null;
  try {
    real = await containReal(scope, deps.fs, realRoot, abs);
  } catch (error) {
    if (isTimeout(error)) return error.started ? outstanding() : stop();
    real = null;
  }
  if (real === null) return failed("symlink-escape");
  let info: StatLike;
  try {
    info = await scope.io(() => deps.fs.stat(real!), scope.wallDeadline, () => undefined);
  } catch (error) {
    if (isTimeout(error)) return error.started ? outstanding() : stop();
    return failed(errorCode(error) === "ENOENT" ? "missing-file" : "read-failed");
  }
  if (!info.isFile()) return failed("missing-file");
  const physical = `${info.dev}:${info.ino}`;
  // An observed shrink (with or without a same inode) advances the scalar generation.
  const truncated = cp !== undefined && info.size < cp.offset;
  const generation = generationFor(cp, physical, truncated);
  const sameGeneration = cp !== undefined && cp.generation === generation;
  const mtime = Math.floor(info.mtimeMs / 1000);
  const current = sameGeneration && cp!.done && !cp!.discarding && cp!.size === info.size && cp!.mtime === mtime;
  if (current) {
    markFileCurrent(deps.db, source, id, info.size, mtime, deps.now());
    return { processed: false, confirmed: true, items: 0, bytes: 0, pending: false };
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
  let handleOwnedByPending = false;
  let closeTimedOut = false;
  let lineDeadline: number | null = null;
  const items: NativeUsageItem[] = [];
  const contexts: NativeContextRow[] = [];
  let handle: FileHandleLike | null = null;
  try {
    handle = await scope.io(
      () => deps.fs.open(real!),
      scope.wallDeadline,
      (opened) => opened?.close(),
    );
  } catch (error) {
    if (isTimeout(error)) return error.started ? outstanding() : stop();
    return failed("read-failed");
  }

  // The opened-handle identity is mandatory before any read; an unavailable fstat fails
  // the file closed instead of hoping a later path stat succeeds.
  try {
    const opened = await scope.io(
      () => handle!.stat(),
      scope.wallDeadline,
      () => handle!.close(),
    );
    if (`${opened.dev}:${opened.ino}` !== physical) {
      warnings.add("file-replaced");
      if (!(await scope.closeCleanup(() => handle!.close()))) warnings.add("io-timeout");
      handle = null;
      return { processed: false, confirmed: false, items: 0, bytes: 0, pending: false };
    }
  } catch (error) {
    if (isTimeout(error) && error.started) {
      handleOwnedByPending = true;
      return outstanding();
    }
    await scope.closeCleanup(() => handle!.close());
    handle = null;
    if (isTimeout(error)) return stop();
    return failed("read-failed");
  }

  try {
    while (fileBytes < JSONL_BUDGET.stepBytes && sourceBytes < maxSourceBytes) {
      if (scope.expired) break;
      const startingLine = carry.length === 0 && !discarding;
      if (startingLine) {
        if (!scope.lineStartOpen) {
          deadlineHit = true;
          break;
        }
        // The line deadline starts with the first read of a new line and includes it.
        lineDeadline = Math.min(
          scope.now + JSONL_BUDGET.lineDeadlineMs,
          scope.wallDeadline - JSONL_BUDGET.commitReserveMs,
        );
        if (lineDeadline <= scope.now) {
          deadlineHit = true;
          break;
        }
      }
      if (lineDeadline === null) {
        lineDeadline = Math.min(
          scope.now + JSONL_BUDGET.lineDeadlineMs,
          scope.wallDeadline - JSONL_BUDGET.commitReserveMs,
        );
      }
      const bounded = Math.max(
        1,
        Math.min(JSONL_BUDGET.chunkBytes, JSONL_BUDGET.stepBytes - fileBytes, maxSourceBytes - sourceBytes),
      );
      const chunk = Buffer.alloc(bounded);
      const activeHandle = handle;
      // Discarded oversized lines are bounded by the step/wall budget, not a line deadline.
      const readDeadline = discarding
        ? scope.wallDeadline - JSONL_BUDGET.commitReserveMs
        : (lineDeadline as number);
      let read: { bytesRead: number };
      try {
        read = await scope.io(
          () => activeHandle!.read(chunk, 0, bounded, offset),
          readDeadline,
          () => activeHandle!.close(),
        );
      } catch (error) {
        if (isTimeout(error) && error.started) {
          handleOwnedByPending = true;
          return outstanding(sourceBytes);
        }
        if (isTimeout(error)) break;
        return failed("read-failed");
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
        // A line whose IO had already started may finish; a new line never starts after
        // the scheduling deadline, so the checkpoint stays before its first byte.
        if (lineDeadline === null && scope.now >= scope.start + JSONL_BUDGET.startLineDeadlineMs) {
          deadlineHit = true;
          break;
        }
        const lineBytes = carry.subarray(0, newline);
        carry = carry.subarray(newline + 1);
        lineStart = nextLineStart;
        lineDeadline = null;
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
      }
      if (deadlineHit) break;
    }
  } finally {
    if (handle && !handleOwnedByPending) {
      const closed = await scope.closeCleanup(() => handle!.close());
      if (!closed) {
        handle = null;
        closeTimedOut = true;
      }
    }
  }

  if (closeTimedOut) return outstanding(sourceBytes);
  if (scope.expired) return { processed: false, confirmed: false, items: 0, bytes: sourceBytes, pending: false };

  if (done) {
    if (discarding) {
      // EOF inside an oversized line: keep the discard marker so a later append
      // continues discarding instead of parsing a fragment as a new line.
      lineStart = offset;
    } else if (carry.length > 0) {
      done = false;
    }
  }
  const consumed = done ? offset : lineStart;
  const completeAtEof = done && carry.length === 0;

  const checkpoint: FileCheckpoint = {
    source,
    identity: id,
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
    const verify = await scope.io(() => deps.fs.stat(real!), scope.wallDeadline, () => undefined);
    if (`${verify.dev}:${verify.ino}` !== physical) {
      warnings.add("file-replaced");
      return { processed: false, confirmed: false, items: 0, bytes: sourceBytes, pending: false };
    }
  } catch (error) {
    if (isTimeout(error)) return error.started ? outstanding(sourceBytes) : stop(sourceBytes);
    return failed("read-failed");
  }
  if (scope.expired) return { processed: false, confirmed: false, items: 0, bytes: sourceBytes, pending: false };
  try {
    applyFileBatch(deps.db, {
      checkpoint,
      expectedPrior: cp ?? null,
      items,
      contexts,
      now: deps.now(),
    });
  } catch {
    warnings.add("checkpoint-conflict");
    return { processed: false, confirmed: false, items: 0, bytes: sourceBytes, pending: false };
  }
  return { processed: true, confirmed: false, items: items.length, bytes: sourceBytes, pending: false };
}

async function ingestionPass(
  deps: NativeCollectorDeps,
  scope: IoWindow,
  source: "claude" | "codex",
  realRoot: string,
  warnings: ReturnType<typeof warningBag>,
): Promise<IngestionOutcome> {
  const outcome: IngestionOutcome = {
    filesProcessed: 0,
    confirmed: 0,
    items: 0,
    bytes: 0,
    overranReserve: false,
    rootUnavailable: false,
    pending: false,
  };
  const runtime = runtimeFor(deps.db, source);
  const candidates = selectIngestionCandidates(deps.db, source, deps.now(), JSONL_BUDGET.maxFilesPerPass);
  for (const candidate of candidates) {
    if (outcome.filesProcessed >= JSONL_BUDGET.maxFilesPerPass) break;
    if (outcome.bytes >= JSONL_BUDGET.maxBytesPerSource) break;
    if (scope.now >= scope.start + JSONL_BUDGET.startLineDeadlineMs || scope.expired) {
      outcome.overranReserve = true;
      break;
    }
    const result = await ingestFile(
      scope,
      deps,
      runtime,
      candidate,
      source,
      realRoot,
      JSONL_BUDGET.maxBytesPerSource - outcome.bytes,
      warnings,
    );
    outcome.filesProcessed += result.processed ? 1 : 0;
    outcome.confirmed += result.confirmed ? 1 : 0;
    outcome.items += result.items;
    outcome.bytes += result.bytes;
    if (result.pending) {
      outcome.pending = true;
      break;
    }
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
  progressed: boolean;
}> {
  const warnings = warningBag();
  const runtime = runtimeFor(deps.db, source);
  for (const code of getPersistentWarnings(deps.db, source)) warnings.add(code);

  if (deps.configWarnings && deps.configWarnings.length > 0) {
    for (const code of deps.configWarnings) warnings.add(code);
    return {
      items: 0,
      filesProcessed: 0,
      filesTotal: countRegisteredFiles(deps.db, source),
      filesDone: getRegistryCounts(deps.db, source, deps.now()).done,
      warnings: warnings.values,
      state: "unavailable",
      stalled: false,
      progressed: false,
    };
  }

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
      progressed: false,
    };
  }

  const scope = new IoWindow(runtime, deps.nowMs);
  const rootState = await resolveRoot(scope, deps, deps.roots[source]);
  if (!rootState.ok) {
    if (rootState.pending) {
      const counts = getRegistryCounts(deps.db, source, deps.now());
      return {
        items: 0,
        filesProcessed: 0,
        filesTotal: counts.total,
        filesDone: counts.done,
        warnings: warnings.values,
        state: "indexing",
        stalled: true,
        progressed: false,
      };
    }
    if (rootState.missing) {
      const total = countRegisteredFiles(deps.db, source);
      if (total === 0) {
        return {
          items: 0,
          filesProcessed: 0,
          filesTotal: 0,
          filesDone: 0,
          warnings: warnings.values,
          state: "empty",
          stalled: false,
          progressed: true,
        };
      }
      warnings.add("root-unavailable");
      return {
        items: 0,
        filesProcessed: 0,
        filesTotal: total,
        filesDone: getRegistryCounts(deps.db, source, deps.now()).done,
        warnings: warnings.values,
        state: "unavailable",
        stalled: false,
        progressed: false,
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
      progressed: false,
    };
  }

  const knownCount = countRegisteredFiles(deps.db, source);
  const discoveryFirst = runtime.discoveryFirstNext || knownCount === 0;
  runtime.discoveryFirstNext = false;
  let rootUnavailable = false;
  let pending = false;
  let progressed = false;
  let ingestion: IngestionOutcome | null = null;

  const runDiscovery = async (budget: number) => {
    if (!scope.scheduleOpen || scope.expired) return;
    const outcome = await discoveryPass(deps, scope, runtime, source, rootState.realRoot, Math.max(0, budget), warnings);
    if (outcome.rootUnavailable) rootUnavailable = true;
    if (outcome.pending) pending = true;
    if (outcome.progressed) progressed = true;
  };

  if (!scope.scheduleOpen) {
    // Slow root validation consumed the scheduling window: start nothing else this pass.
  } else if (discoveryFirst) {
    await runDiscovery(JSONL_BUDGET.discoveryMsPerPass);
    if (!pending && scope.scheduleOpen) {
      ingestion = await ingestionPass(deps, scope, source, rootState.realRoot, warnings);
    }
  } else {
    ingestion = await ingestionPass(deps, scope, source, rootState.realRoot, warnings);
    if (!pending && ingestion.overranReserve) runtime.discoveryFirstNext = true;
    // The final slice of the scheduling window is reserved for discovery.
    const remaining = Math.min(JSONL_BUDGET.discoveryMsPerPass, scope.scheduleDeadline - scope.now);
    if (!pending && remaining > 0) await runDiscovery(remaining);
  }
  if (ingestion) {
    if (ingestion.overranReserve) runtime.discoveryFirstNext = true;
    if (ingestion.pending) pending = true;
    if (ingestion.filesProcessed > 0 || ingestion.confirmed > 0) progressed = true;
  }

  const counts = getRegistryCounts(deps.db, source, deps.now());
  const coverageFailure = warnings.values.some((code) => coverageFailureCodes.has(code));
  const state = rootUnavailable
    ? "unavailable"
    : pending
      ? "indexing"
      : stateFor(source, deps.db, deps.now(), coverageFailure);
  return {
    items: ingestion?.items ?? 0,
    filesProcessed: ingestion?.filesProcessed ?? 0,
    filesTotal: counts.total,
    filesDone: counts.done,
    warnings: warnings.values,
    state,
    stalled: pending,
    progressed,
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

// OpenCode preparation shares the same bounded single-flight boundary as JSONL sources:
// a timed-out reader keeps its latch until it settles; later polls start nothing new.
const openCodeRuntimes = new WeakMap<Database.Database, PendingOwner>();

function openCodeRuntime(db: Database.Database): PendingOwner {
  let runtime = openCodeRuntimes.get(db);
  if (!runtime) {
    runtime = { pending: null };
    openCodeRuntimes.set(db, runtime);
  }
  return runtime;
}

type PrepFailure = { ok: false; state: SourceState; warning: string; pending: boolean };

async function openCodeExecutableState(
  scope: IoWindow,
  deps: NativeCollectorDeps,
): Promise<{ ok: true } | PrepFailure> {
  let info: StatLike | null;
  try {
    info = await scope.io(
      () => deps.opencode.stat(deps.opencode.executable),
      scope.wallDeadline,
      () => undefined,
    );
  } catch (error) {
    if (isTimeout(error)) {
      return { ok: false, state: "unavailable", warning: "io-pending", pending: error.started };
    }
    info = null;
  }
  if (!info || !info.isFile()) {
    return { ok: false, state: "unavailable", warning: "executable-missing", pending: false };
  }
  const identity = `${deps.opencode.executable}:${info.dev}:${info.ino}:${info.size}:${Math.floor(info.mtimeMs)}`;
  const cachedHash = executableIdentityCache.get(identity);
  let hash: string | null;
  if (cachedHash !== undefined) {
    hash = cachedHash;
  } else {
    try {
      hash = await scope.io(
        () => deps.opencode.hash(deps.opencode.executable),
        scope.wallDeadline,
        () => undefined,
      );
    } catch (error) {
      if (isTimeout(error)) {
        return { ok: false, state: "unavailable", warning: "io-pending", pending: error.started };
      }
      hash = null;
    }
  }
  if (hash !== null && cachedHash === undefined) executableIdentityCache.set(identity, hash);
  if (hash !== OPENCODE_EXECUTABLE_SHA256) {
    return { ok: false, state: "incompatible", warning: "executable-mismatch", pending: false };
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

const OPENCODE_WARNING_KEYS = new Set(["malformed", "overlength", "unsupported", "budget"]);
const OPENCODE_MAX_ROWS = 512;
const OPENCODE_MAX_ROW_BYTES = 16 * 1024;
const OPENCODE_MAX_ROWS_BYTES = 240 * 1024;
const OPENCODE_MAX_HEADER_BYTES = 16 * 1024;

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
  if (raw.rows.length > OPENCODE_MAX_ROWS) return null;
  let rowsBytes = 0;
  for (const row of raw.rows) {
    if (!isObj(row)) return null;
    const size = Buffer.byteLength(JSON.stringify(row), "utf8");
    if (size > OPENCODE_MAX_ROW_BYTES) return null;
    rowsBytes += size;
  }
  if (rowsBytes > OPENCODE_MAX_ROWS_BYTES) return null;
  const header: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (key !== "rows") header[key] = value;
  }
  if (Buffer.byteLength(JSON.stringify(header), "utf8") > OPENCODE_MAX_HEADER_BYTES) return null;
  const next = raw.next;
  if (!isObj(next) || !isObj(next.v1) || !isObj(next.v2)) return null;
  const sanitize = (
    cursor: Record<string, unknown>,
    prior: OpenCodeCursorState["v1"],
  ): OpenCodeCursorState["v1"] | null => {
    if (typeof cursor.done !== "boolean") return null;
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
    return { after, ceiling, done: cursor.done };
  };
  const v1 = sanitize(next.v1, previous.v1);
  const v2 = sanitize(next.v2, previous.v2);
  if (!v1 || !v2) return null;
  const warnings: Record<string, number> = {};
  if (!isObj(raw.warnings)) return null;
  for (const [key, value] of Object.entries(raw.warnings)) {
    if (!OPENCODE_WARNING_KEYS.has(key)) return null;
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) return null;
    warnings[key] = value;
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
    // Canonical spelling only: no "-0", no leading zeroes.
    if (numeric.toString() !== value) return null;
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
  const failed = (warning: string, state: SourceState = "unavailable") => ({
    items: 0,
    warnings: [warning],
    state,
    lastRefresh: 0,
    progress: null,
  });
  const runtime = openCodeRuntime(deps.db);
  if (runtime.pending) return failed("io-pending");
  if (deps.configWarnings && deps.configWarnings.length > 0) {
    return { items: 0, warnings: [...deps.configWarnings], state: "unavailable", lastRefresh: 0, progress: null };
  }
  const scope = new IoWindow(runtime, deps.nowMs);
  if (deps.opencode.ensureAdapter) {
    try {
      await scope.io(() => deps.opencode.ensureAdapter!(), scope.wallDeadline, () => undefined);
    } catch (error) {
      if (isTimeout(error)) return failed(error.started ? "io-pending" : "io-timeout");
      return failed("pass-failed");
    }
  }
  const executable = await openCodeExecutableState(scope, deps);
  if (!executable.ok) return failed(executable.warning, executable.state);
  const dbPath = deps.roots.opencodeDb;
  if (dbPath.includes("\u0000") || Buffer.byteLength(dbPath, "utf8") > 4096) {
    return failed("db-invalid");
  }
  try {
    const info = await scope.io(() => deps.fs.stat(dbPath), scope.wallDeadline, () => undefined);
    if (!info.isFile()) throw new Error("not-file");
  } catch (error) {
    if (isTimeout(error)) return failed(error.started ? "io-pending" : "io-timeout");
    return failed("db-missing");
  }
  let real: string | null;
  try {
    const realRoot = await scope.io(
      () => deps.fs.realpath(deps.roots.opencodeDataRoot),
      scope.wallDeadline,
      () => undefined,
    );
    const realDb = await scope.io(() => deps.fs.realpath(dbPath), scope.wallDeadline, () => undefined);
    real = insideRealRoot(realRoot, realDb) ? realDb : null;
  } catch (error) {
    if (isTimeout(error)) return failed(error.started ? "io-pending" : "io-timeout");
    real = null;
  }
  if (!real) return failed("db-invalid");

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
    return failed("exec-failed");
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
    return failed(error, state);
  }
  const reply = parseOpenCodeReply(stdout, cursor);
  if (!reply) return failed("malformed-reply");

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
          result.state === "unavailable" || result.stalled || !result.progressed
            ? previous?.lastRefresh ?? null
            : now,
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

type StartupRoots = NativeCollectorDeps["roots"];
type StartupConfig = { roots: StartupRoots; warnings: string[] };

// One immutable startup/first-use configuration promise per process boundary. Absence
// alone selects the normal native roots; anything present but unusable selects owned
// invalid fixture roots with an explicit diagnostic, never the production roots.
let startupConfigPromise: Promise<StartupConfig> | null = null;

function normalNativeRoots(): StartupRoots {
  const home = homedir();
  const dataRoot = process.env.XDG_DATA_HOME || join(home, ".local", "share");
  return {
    claude: join(home, ".claude", "projects"),
    codex: join(home, ".codex", "sessions"),
    opencodeDataRoot: join(dataRoot, "opencode"),
    opencodeDb: join(dataRoot, "opencode", "opencode.db"),
  };
}

function invalidNativeRoots(root: string): StartupRoots {
  const opencodeDataRoot = join(root, "native-invalid", "opencode");
  return {
    claude: join(root, "native-invalid", "claude"),
    codex: join(root, "native-invalid", "codex"),
    opencodeDataRoot,
    opencodeDb: join(opencodeDataRoot, "opencode.db"),
  };
}

async function readStartupConfig(): Promise<StartupConfig> {
  const root = jaxosHome();
  let text: string;
  try {
    text = await readFile(join(root, "native-roots.json"), "utf8");
  } catch (error) {
    if (errorCode(error) === "ENOENT") return { roots: normalNativeRoots(), warnings: [] };
    return { roots: invalidNativeRoots(root), warnings: ["config-invalid"] };
  }
  const invalid = (): StartupConfig => ({ roots: invalidNativeRoots(root), warnings: ["config-invalid"] });
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    return invalid();
  }
  if (!isObj(raw)) return invalid();
  const keys = ["claude", "codex", "opencode"] as const;
  for (const key of keys) {
    if (typeof raw[key] !== "string" || !isAbsolute(raw[key] as string)) return invalid();
  }
  try {
    const boundary = await nodeRealpathAsync(root);
    const resolved: Record<string, string> = {};
    for (const key of keys) {
      const real = await nodeRealpathAsync(raw[key] as string);
      if (real !== boundary && !real.startsWith(base(boundary))) return invalid();
      resolved[key] = real;
    }
    return {
      roots: {
        claude: resolved.claude,
        codex: resolved.codex,
        opencodeDataRoot: resolved.opencode,
        opencodeDb: join(resolved.opencode, "opencode.db"),
      },
      warnings: [],
    };
  } catch {
    return invalid();
  }
}

function startupConfig(): Promise<StartupConfig> {
  if (!startupConfigPromise) startupConfigPromise = readStartupConfig();
  return startupConfigPromise;
}

export async function defaultNativeCollectorDeps(db: Database.Database): Promise<NativeCollectorDeps> {
  const adapterRoot = join(jaxosHome(), "native-opencode");
  const cwd = join(adapterRoot, "cwd");
  const configPath = join(adapterRoot, "empty.toml");
  const xdgConfigHome = join(adapterRoot, "config");
  const startup = await startupConfig();
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
    roots: startup.roots,
    configWarnings: startup.warnings,
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
      // Streams fixed bounded chunks; the whole executable is never materialized.
      hash: async (path) => {
        let handle: Awaited<ReturnType<typeof nodeOpenAsync>> | null = null;
        try {
          handle = await nodeOpenAsync(path, "r");
          const digest = createHash("sha256");
          const buffer = Buffer.alloc(64 * 1024);
          let position = 0;
          for (;;) {
            const { bytesRead } = await handle.read(buffer, 0, buffer.length, position);
            if (bytesRead <= 0) break;
            digest.update(buffer.subarray(0, bytesRead));
            position += bytesRead;
          }
          return digest.digest("hex");
        } catch {
          return null;
        } finally {
          await handle?.close().catch(() => undefined);
        }
      },
    },
  };
}
