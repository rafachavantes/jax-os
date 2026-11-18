import { createHash } from "node:crypto";
import { execFile } from "node:child_process";
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readdirSync, readSync, realpathSync, statSync as nodeStat, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join, sep } from "node:path";
import { promisify } from "node:util";
import type Database from "better-sqlite3";
import {
  applyFileBatch,
  getFileCheckpoint,
  getOpenCodeCursor,
  getSourceStatuses,
  persistOpenCodeBatch,
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
  }
  return { allInput, allOutput, headline: allInput + allOutput };
}

// ---------------------------------------------------------------------------
// Bounded native discovery and incremental collection.
// System access is injected: tests never read real transcript roots or run the
// installed OpenCode binary.

export const OPENCODE_EXECUTABLE = "/home/rafa/.opencode/bin/opencode";
export const OPENCODE_EXECUTABLE_SHA256 =
  "9ca0b9953d49997601655e54f846a3efa464f237e47c6f1b04716d0f2e64c4c2";
export const OPENCODE_SCHEMA_VERSION = "1.18.34";

const JSONL_BUDGET = {
  maxFilesPerPass: 6,
  maxBytesPerSource: 16 * 1024 * 1024,
  stepBytes: 2 * 1024 * 1024,
  chunkBytes: 32 * 1024,
  softDeadlineMs: 200,
  wallDeadlineMs: 1200,
  maxLineBytes: 1024 * 1024,
  maxDiscoveryEntries: 64,
};
const OPENCODE_BUDGET = { timeoutMs: 2000, maxReplyBytes: 256 * 1024, refreshMs: 60_000 };

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
export type NativeFs = {
  statSync(path: string): StatLike;
  realpathSync(path: string): string;
  readdirSync(path: string): DirEntryLike[];
  openSync(path: string): number;
  readSync(fd: number, buffer: Buffer, offset: number, length: number, position: number): number;
  closeSync(fd: number): void;
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
    exec: NativeExec;
    stat: (path: string) => StatLike | null;
    hash: (path: string) => string | null;
  };
};

export type DiscoveryCandidate = {
  identity: string;
  path: string;
  child: boolean;
  mtimeMs: number;
};

const base = (p: string) => (p.endsWith(sep) ? p : p + sep);

function insideRealRoot(fs: NativeFs, root: string, path: string): string | null {
  try {
    const realRoot = fs.realpathSync(root);
    const real = fs.realpathSync(path);
    return real === realRoot || real.startsWith(base(realRoot)) ? real : null;
  } catch {
    return null;
  }
}

function dirEntries(fs: NativeFs, path: string): DirEntryLike[] {
  try {
    return fs.readdirSync(path).sort((a, b) => a.name.localeCompare(b.name));
  } catch {
    return [];
  }
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

export function discoverJsonlFiles(
  fs: NativeFs,
  source: "claude" | "codex",
  root: string,
): { files: DiscoveryCandidate[]; warnings: string[]; truncated: boolean } {
  const warnings = warningBag();
  const files: DiscoveryCandidate[] = [];
  let visited = 0;
  const stat = (path: string) => {
    try {
      return fs.statSync(path);
    } catch {
      return null;
    }
  };
  const push = (path: string, identity: string, child: boolean) => {
    const info = stat(path);
    if (!info || !info.isFile()) {
      warnings.add("unsupported-layout");
      return;
    }
    files.push({ path, identity, child, mtimeMs: info.mtimeMs });
  };

  if (source === "claude") {
    for (const project of dirEntries(fs, root)) {
      if (visited >= JSONL_BUDGET.maxDiscoveryEntries) break;
      visited += 1;
      const projectPath = join(root, project.name);
      if (!project.isDirectory() || !insideRealRoot(fs, root, projectPath)) {
        warnings.add("unsupported-layout");
        continue;
      }
      for (const entry of dirEntries(fs, projectPath)) {
        if (visited >= JSONL_BUDGET.maxDiscoveryEntries) break;
        visited += 1;
        const entryPath = join(projectPath, entry.name);
        if ((entry.isFile() || entry.isSymbolicLink()) && entry.name.endsWith(".jsonl")) {
          const real = insideRealRoot(fs, root, entryPath);
          if (!real) warnings.add("symlink-escape");
          else push(real, `${project.name}/${entry.name}`, false);
        } else if (entry.isDirectory()) {
          const subagents = join(entryPath, "subagents");
          if (!insideRealRoot(fs, root, subagents)) {
            warnings.add("unsupported-layout");
            continue;
          }
          for (const child of dirEntries(fs, subagents)) {
            if (visited >= JSONL_BUDGET.maxDiscoveryEntries) break;
            visited += 1;
            if ((!child.isFile() && !child.isSymbolicLink()) || !child.name.endsWith(".jsonl")) {
              warnings.add("unsupported-layout");
              continue;
            }
            const childPath = join(subagents, child.name);
            const real = insideRealRoot(fs, root, childPath);
            if (!real) warnings.add("symlink-escape");
            else push(real, `${project.name}/${entry.name}/subagents/${child.name}`, true);
          }
        } else {
          warnings.add("unsupported-layout");
        }
      }
    }
  } else {
    const queue: { path: string; depth: number }[] = [{ path: root, depth: 0 }];
    while (queue.length > 0 && visited < JSONL_BUDGET.maxDiscoveryEntries) {
      const current = queue.shift()!;
      for (const entry of dirEntries(fs, current.path)) {
        if (visited >= JSONL_BUDGET.maxDiscoveryEntries) break;
        visited += 1;
        const entryPath = join(current.path, entry.name);
        if (current.depth < 3 && entry.isDirectory()) {
          queue.push({ path: entryPath, depth: current.depth + 1 });
          continue;
        }
        if ((entry.isFile() || entry.isSymbolicLink()) && entry.name.startsWith("rollout-") && entry.name.endsWith(".jsonl")) {
          const real = insideRealRoot(fs, root, entryPath);
          if (!real) warnings.add("symlink-escape");
          else push(real, real.slice(base(fs.realpathSync(root)).length), false);
        } else {
          warnings.add("unsupported-layout");
        }
      }
    }
  }
  return {
    files,
    warnings: warnings.values,
    truncated: visited >= JSONL_BUDGET.maxDiscoveryEntries,
  };
}

type ParsedJsonl = {
  items: NativeUsageItem[];
  contexts: NativeContextRow[];
  warnings: string[];
  lineCount: number;
};

function parseJsonlContent(
  content: string,
  source: "claude" | "codex",
  opts: { child: boolean; fileIdentity: string; now: number },
): ParsedJsonl {
  const warnings = warningBag();
  const items: NativeUsageItem[] = [];
  const contexts: NativeContextRow[] = [];
  let lineCount = 0;
  for (const line of content.split("\n")) {
    if (line.trim() === "") continue;
    lineCount += 1;
    if (source === "claude") {
      const out = parseClaudeLine(line, { child: opts.child, now: opts.now });
      if ("item" in out) items.push(out.item);
      else if (out.skip !== "not-assistant") warnings.add(out.skip);
    } else {
      const out = parseCodexLine(line, { fileIdentity: opts.fileIdentity, now: opts.now });
      if ("item" in out) items.push(out.item);
      else if ("context" in out) contexts.push(out.context);
      else warnings.add(out.skip);
    }
  }
  return { items, contexts, warnings: warnings.values, lineCount };
}

export type JsonlPassResult = {
  items: number;
  filesProcessed: number;
  filesTotal: number;
  filesDone: number;
  warnings: string[];
  indexing: boolean;
};

export function runJsonlSourcePass(
  deps: NativeCollectorDeps,
  source: "claude" | "codex",
): JsonlPassResult {
  const root = source === "claude" ? deps.roots.claude : deps.roots.codex;
  const warnings = warningBag();
  const discovery = discoverJsonlFiles(deps.fs, source, root);
  for (const warning of discovery.warnings) warnings.add(warning);
  if (discovery.truncated) warnings.add("discovery-truncated");

  const known = new Map<string, DiscoveryCandidate>();
  for (const candidate of discovery.files) known.set(candidate.identity, candidate);
  for (const checkpoint of (deps.db.prepare("SELECT identity FROM native_usage_files WHERE source = ?").all(source) as { identity: string }[])) {
    if (known.has(checkpoint.identity)) continue;
    const path = join(root, checkpoint.identity);
    const real = insideRealRoot(deps.fs, root, path);
    if (!real) {
      warnings.add("missing-file");
      continue;
    }
    try {
      const info = deps.fs.statSync(real);
      if (info.isFile()) known.set(checkpoint.identity, { identity: checkpoint.identity, path: real, child: false, mtimeMs: info.mtimeMs });
    } catch {
      warnings.add("missing-file");
    }
  }

  const today = utcDay(deps.now());
  const todayStart = Date.parse(`${today}T00:00:00Z`);
  const evaluated = [...known.values()].map((candidate) => {
    const cp = getFileCheckpoint(deps.db, source, candidate.identity);
    const info = deps.fs.statSync(candidate.path);
    const current =
      cp !== undefined && cp.generation === `${info.dev}:${info.ino}` && cp.done &&
      cp.size === info.size && cp.mtime === Math.floor(info.mtimeMs / 1000);
    return { candidate, cp, info, current, processedAt: cp?.processedAt ?? 0 };
  });
  const pending = evaluated
    .filter((entry) => !entry.current)
    .sort((a, b) => {
      const at = a.candidate.mtimeMs >= todayStart ? 1 : 0;
      const bt = b.candidate.mtimeMs >= todayStart ? 1 : 0;
      if (at !== bt) return bt - at;
      return a.processedAt - b.processedAt;
    });

  const wallStart = deps.nowMs();
  let bytes = 0;
  let filesProcessed = 0;
  let itemCount = 0;
  const maxFiles = JSONL_BUDGET.maxFilesPerPass;
  for (const { candidate, cp, info } of pending) {
    if (filesProcessed >= maxFiles) break;
    if (bytes >= JSONL_BUDGET.maxBytesPerSource) break;
    if (deps.nowMs() - wallStart >= JSONL_BUDGET.wallDeadlineMs) break;
    const generation = `${info.dev}:${info.ino}`;
    const sameGeneration = cp !== undefined && cp.generation === generation;
    const startOffset = sameGeneration && cp.offset <= info.size ? cp.offset : 0;
    let offset = startOffset;
    let lineStart = startOffset;
    let carry = Buffer.alloc(0);
    let carryStart = startOffset;
    let done = false;
    let discarding = false;
    let deadlineHit = false;
    let readFailed = false;
    let fileBytes = 0;
    let fd: number | null = null;
    const items: NativeUsageItem[] = [];
    const contexts: NativeContextRow[] = [];
    try {
      fd = deps.fs.openSync(candidate.path);
      while (fileBytes < JSONL_BUDGET.stepBytes && bytes < JSONL_BUDGET.maxBytesPerSource) {
        const size = Math.min(
          JSONL_BUDGET.chunkBytes,
          JSONL_BUDGET.stepBytes - fileBytes,
          JSONL_BUDGET.maxBytesPerSource - bytes,
        );
        if (size <= 0) break;
        const chunk = Buffer.alloc(size);
        const read = deps.fs.readSync(fd, chunk, 0, size, offset);
        if (read <= 0) {
          done = true;
          break;
        }
        const chunkStart = offset;
        offset += read;
        fileBytes += read;
        bytes += read;
        if (carry.length === 0) carryStart = chunkStart;
        carry = Buffer.concat([carry, chunk.subarray(0, read)]);
        for (;;) {
          const newline = carry.indexOf(10);
          if (newline < 0) {
            if (carry.length > JSONL_BUDGET.maxLineBytes) {
              warnings.add("line-too-long");
              carry = Buffer.alloc(0);
              carryStart = offset;
              discarding = true;
            }
            break;
          }
          const nextLineStart = carryStart + newline + 1;
          if (discarding) {
            discarding = false;
            carry = carry.subarray(newline + 1);
            carryStart = nextLineStart;
            lineStart = nextLineStart;
            continue;
          }
          const lineBytes = carry.subarray(0, newline);
          carry = carry.subarray(newline + 1);
          lineStart = nextLineStart;
          carryStart = nextLineStart;
          if (lineBytes.length > JSONL_BUDGET.maxLineBytes) {
            warnings.add("line-too-long");
          } else {
            const parsed = parseJsonlContent(lineBytes.toString("utf8"), source, {
              child: candidate.child,
              fileIdentity: candidate.identity,
              now: deps.now(),
            });
            items.push(...parsed.items);
            contexts.push(...parsed.contexts);
            for (const warning of parsed.warnings) warnings.add(warning);
          }
          if (deps.nowMs() - wallStart > JSONL_BUDGET.softDeadlineMs) {
            deadlineHit = true;
            break;
          }
        }
        if (deadlineHit) break;
        if (deps.nowMs() - wallStart >= JSONL_BUDGET.wallDeadlineMs) break;
      }
    } catch {
      warnings.add("read-failed");
      readFailed = true;
    } finally {
      if (fd !== null) {
        try {
          deps.fs.closeSync(fd);
        } catch {
          /* released by the process exit path */
        }
      }
    }
    if (readFailed) continue;
    const consumed = done ? offset : lineStart;
    const checkpoint: FileCheckpoint = {
      source,
      identity: candidate.identity,
      generation,
      dev: String(info.dev),
      ino: String(info.ino),
      size: info.size,
      mtime: Math.floor(info.mtimeMs / 1000),
      offset: consumed,
      done,
      processedAt: deps.now(),
    };
    try {
      applyFileBatch(deps.db, {
        checkpoint,
        expectedOffset: sameGeneration ? cp!.offset : null,
        items,
        contexts,
        now: deps.now(),
      });
    } catch {
      warnings.add("checkpoint-conflict");
      continue;
    }
    filesProcessed += 1;
    itemCount += items.length;
  }

  const indexing = pending.length - filesProcessed > 0 || discovery.truncated;
  return {
    items: itemCount,
    filesProcessed,
    filesTotal: evaluated.length,
    filesDone: evaluated.length - pending.length + filesProcessed,
    warnings: warnings.values,
    indexing,
  };
}

export const OPENCODE_HELPER_SOURCE = String.raw`(function(){
  function emit(value){ process.stdout.write(JSON.stringify(value)); }
  function fail(code){ emit({v:1,ok:false,error:code}); return; }
  var argv = process.argv || [];
  if (argv.length < 2) { fail("bad-argv"); return; }
  var dbPath = argv[argv.length - 2];
  var stateRaw = argv[argv.length - 1];
  if (typeof dbPath !== "string" || dbPath.indexOf("\u0000") !== -1 || Buffer.byteLength(dbPath, "utf8") > 4096) { fail("bad-path"); return; }
  var state;
  try { state = JSON.parse(stateRaw); } catch (e) { fail("bad-state"); return; }
  if (!state || state.v !== 1 || typeof state.v1 !== "object" || typeof state.v2 !== "object") { fail("bad-state"); return; }
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
  var MAX_ROWS = 256, PAGE_BYTES = 245760, TOTAL_CAP = 262144, MAX_ID = 256, MAX_ROW = 8192;
  function safeCount(v){ return typeof v === "number" && isFinite(v) && Math.floor(v) === v && v >= 0 && v <= 9007199254740991 ? v : null; }
  function idOk(v){ if (typeof v !== "string" || v.length === 0) return false; var n = 0; for (var i = 0; i < v.length; i += 1) { var c = v.charCodeAt(i); if (c >= 0xD800 && c <= 0xDBFF && i + 1 < v.length) { var d = v.charCodeAt(i + 1); if (d >= 0xDC00 && d <= 0xDFFF) i += 1; } n += 1; if (n > MAX_ID) return false; } return n > 0; }
  function obj(v){ return v && typeof v === "object" && !Array.isArray(v) ? v : null; }
  function num(v){ return typeof v === "number" && isFinite(v) && v >= 0 ? v : null; }
  var warnings = { malformed: 0, overlength: 0, unsupported: 0, budget: 0 };
  var rows = [];
  var rids = [];
  var bytesOut = 0;
  function project(projection, row){
    var data;
    try { data = JSON.parse(row.data); } catch (e) { warnings.malformed += 1; return null; }
    if (!obj(data)) { warnings.malformed += 1; return null; }
    var modelId = null, providerId = null, agent = null, cost = null, tokens = null, created = null;
    if (projection === "v2") {
      var model = obj(data.model); tokens = obj(data.tokens); var time = obj(data.time);
      modelId = model && typeof model.id === "string" ? model.id : null;
      providerId = model && typeof model.providerID === "string" ? model.providerID : null;
      agent = typeof data.agent === "string" ? data.agent : null;
      cost = num(data.cost);
      created = time && typeof time.created === "number" ? time.created : row.time_created;
    } else {
      if (data.role !== "assistant") return null;
      tokens = obj(data.tokens);
      modelId = typeof data.modelID === "string" ? data.modelID : null;
      providerId = typeof data.providerID === "string" ? data.providerID : null;
      agent = typeof data.agent === "string" ? data.agent : null;
      cost = num(data.cost);
      created = typeof data.created === "number" ? data.created : row.time_created;
    }
    if (!idOk(row.id) || !idOk(row.session_id)) { warnings.overlength += 1; return null; }
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
      session_id: row.session_id,
      message_id: row.id,
      parent_session_id: row.parent_id === undefined ? null : row.parent_id,
      model_id: modelId,
      provider_id: providerId,
      agent: agent,
      created: created,
      updated: typeof row.time_updated === "number" ? row.time_updated : created,
      cost: cost,
      input: input,
      output: output,
      reasoning: reasoning,
      cache_read: cacheRead,
      cache_write: cacheWrite
    };
  }
  function ceilingOf(tableName){
    try { var row = db.prepare("SELECT MAX(rowid) AS c FROM " + tableName).get(); return row && typeof row.c === "number" ? row.c : 0; }
    catch (e) { return 0; }
  }
  function pageOnce(projection, tableName, cursor){
    var next = { after: cursor.after, ceiling: cursor.ceiling, done: cursor.done };
    if (!next.ceiling) next.ceiling = ceilingOf(tableName);
    if (next.done) { return next; }
    var sql = projection === "v2"
      ? "SELECT m.rowid AS rid, m.id AS id, m.session_id AS session_id, m.time_created AS time_created, m.time_updated AS time_updated, m.data AS data, s.parent_id AS parent_id FROM session_message m LEFT JOIN session s ON s.id = m.session_id WHERE m.type = 'assistant' AND m.rowid > ? AND m.rowid <= ? ORDER BY m.rowid LIMIT 256"
      : "SELECT m.rowid AS rid, m.id AS id, m.session_id AS session_id, m.time_created AS time_created, m.time_updated AS time_updated, m.data AS data, s.parent_id AS parent_id FROM message m LEFT JOIN session s ON s.id = m.session_id WHERE m.rowid > ? AND m.rowid <= ? ORDER BY m.rowid LIMIT 256";
    var page;
    try { page = db.prepare(sql).all(cursor.after, next.ceiling); }
    catch (e) { throw new Error("query-failed"); }
    if (!page || page.length === 0) { next.done = true; return next; }
    for (var i = 0; i < page.length; i += 1) {
      var row = page[i];
      var projected = project(projection, row);
      if (projected) {
        var encoded = JSON.stringify(projected);
        var size = Buffer.byteLength(encoded, "utf8");
        if (size > MAX_ROW) { warnings.unsupported += 1; }
        else if (bytesOut + size > PAGE_BYTES) { warnings.budget += 1; return next; }
        else { rows.push(projected); rids.push({ projection: projection, rid: row.rid }); bytesOut += size; }
      }
      next.after = row.rid;
    }
    if (next.after >= next.ceiling) next.done = true;
    return next;
  }
  var nextState;
  try {
    nextState = { v1: pageOnce("v1", "message", state.v1), v2: pageOnce("v2", "session_message", state.v2) };
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
  var encodedReply = JSON.stringify(reply);
  while (Buffer.byteLength(encodedReply, "utf8") > TOTAL_CAP && reply.rows.length > 0) {
    var dropped = rids.pop();
    reply.rows.pop();
    if (dropped) {
      var last = 0;
      for (var ri = 0; ri < rids.length; ri += 1) { if (rids[ri].projection === dropped.projection) last = rids[ri].rid; }
      nextState[dropped.projection].after = last;
      nextState[dropped.projection].done = false;
    }
    encodedReply = JSON.stringify(reply);
  }
  if (Buffer.byteLength(encodedReply, "utf8") > TOTAL_CAP) { fail("output-overflow"); return; }
  emit(reply);
})();`;

const executableIdentityCache = new Map<string, string>();

function openCodeExecutableState(
  deps: NativeCollectorDeps,
): { ok: true } | { ok: false; state: SourceState; warning: string } {
  let info: StatLike | null;
  try {
    info = deps.opencode.stat(deps.opencode.executable);
  } catch {
    info = null;
  }
  if (!info || !info.isFile()) return { ok: false, state: "unavailable", warning: "executable-missing" };
  const identity = `${deps.opencode.executable}:${info.dev}:${info.ino}:${info.size}:${Math.floor(info.mtimeMs)}`;
  let hash = executableIdentityCache.get(identity);
  if (!hash) {
    hash = deps.opencode.hash(deps.opencode.executable) ?? "";
    executableIdentityCache.set(identity, hash);
  }
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

function parseOpenCodeReply(stdout: string): OpenCodeReply | null {
  if (Buffer.byteLength(stdout, "utf8") > OPENCODE_BUDGET.maxReplyBytes) return null;
  let raw: unknown;
  try {
    raw = JSON.parse(stdout.trim());
  } catch {
    return null;
  }
  if (!isObj(raw) || raw.v !== 1 || raw.ok !== true || !Array.isArray(raw.rows)) return null;
  const next = raw.next;
  if (!isObj(next) || !isObj(next.v1) || !isObj(next.v2)) return null;
  const sanitize = (cursor: Record<string, unknown>): OpenCodeCursorState["v1"] | null => {
    const { after, ceiling, done } = cursor;
    if (typeof after !== "number" || !Number.isSafeInteger(after) || after < 0) return null;
    if (typeof ceiling !== "number" || !Number.isSafeInteger(ceiling) || ceiling < 0) return null;
    return { after, ceiling, done: done === true };
  };
  const v1 = sanitize(next.v1);
  const v2 = sanitize(next.v2);
  if (!v1 || !v2) return null;
  const warnings: Record<string, number> = {};
  if (isObj(raw.warnings)) {
    for (const [key, value] of Object.entries(raw.warnings)) {
      if (typeof value === "number" && Number.isSafeInteger(value) && value >= 0) warnings[key] = value;
    }
  }
  return { v: 1, ok: true, runtime: typeof raw.runtime === "string" ? raw.runtime : "", schema: typeof raw.schema === "string" ? raw.schema : "", rows: raw.rows, next: { v1, v2 }, warnings };
}

export async function runOpenCodeSourcePass(deps: NativeCollectorDeps): Promise<{
  items: number;
  warnings: string[];
  state: SourceState;
  lastRefresh: number;
  progress: { filesDone: number; filesTotal: number } | null;
}> {
  const executable = openCodeExecutableState(deps);
  if (!executable.ok) {
    return { items: 0, warnings: [executable.warning], state: executable.state, lastRefresh: 0, progress: null };
  }
  const dbPath = deps.roots.opencodeDb;
  if (dbPath.includes("\u0000") || Buffer.byteLength(dbPath, "utf8") > 4096) {
    return { items: 0, warnings: ["db-invalid"], state: "unavailable", lastRefresh: 0, progress: null };
  }
  const real = insideRealRoot(deps.fs, deps.roots.opencodeDataRoot, dbPath);
  if (!real) return { items: 0, warnings: ["db-invalid"], state: "unavailable", lastRefresh: 0, progress: null };
  try {
    const info = deps.fs.statSync(real);
    if (!info.isFile()) throw new Error("not-file");
  } catch {
    return { items: 0, warnings: ["db-missing"], state: "unavailable", lastRefresh: 0, progress: null };
  }

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
  const reply = parseOpenCodeReply(stdout);
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
  persistOpenCodeBatch(deps.db, items, complete ? { v: 1, v1: { after: 0, ceiling: 0, done: false }, v2: { after: 0, ceiling: 0, done: false } } : next, deps.now());
  const doneRows = next.v1.after + next.v2.after;
  const totalRows = next.v1.ceiling + next.v2.ceiling;
  return {
    items: items.length,
    warnings: warnings.values,
    state: complete ? "available" : "indexing",
    lastRefresh: deps.now(),
    progress: totalRows > 0 ? { filesDone: doneRows, filesTotal: totalRows } : null,
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
    now - previous.lastRefresh < OPENCODE_BUDGET.refreshMs / 1000
  ) {
    return;
  }
  if (source === "opencode") {
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
    return;
  }
  try {
    const result = runJsonlSourcePass(deps, source);
    const state: SourceState =
      result.indexing
        ? "indexing"
        : result.filesTotal === 0
          ? "empty"
          : "available";
    setSourceStatus(
      deps.db,
      {
        source,
        state,
        lastRefresh: now,
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

export function defaultNativeCollectorDeps(db: Database.Database): NativeCollectorDeps {
  const home = homedir();
  const dataRoot = process.env.XDG_DATA_HOME || join(home, ".local", "share");
  const adapterRoot = join(jaxosHome(), "native-opencode");
  const cwd = join(adapterRoot, "cwd");
  const configPath = join(adapterRoot, "empty.toml");
  const xdgConfigHome = join(adapterRoot, "config");
  mkdirSync(cwd, { recursive: true });
  mkdirSync(xdgConfigHome, { recursive: true });
  if (!existsSync(configPath)) writeFileSync(configPath, "", "utf8");
  const pExecFile = promisify(execFile);
  const fs: NativeFs = {
    statSync: (path) => nodeStat(path),
    realpathSync: (path) => realpathSync(path),
    readdirSync: (path) => readdirSync(path, { withFileTypes: true }) as DirEntryLike[],
    openSync: (path) => openSync(path, "r"),
    readSync: (fd, buffer, offset, length, position) => readSync(fd, buffer, offset, length, position),
    closeSync: (fd) => closeSync(fd),
  };
  return {
    db,
    now: () => Math.floor(Date.now() / 1000),
    nowMs: () => Date.now(),
    settings: readGeneralSettings,
    fs,
    roots: {
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
      stat: (path) => {
        try {
          return nodeStat(path);
        } catch {
          return null;
        }
      },
      hash: (path) => {
        try {
          return createHash("sha256").update(readFileSync(path)).digest("hex");
        } catch {
          return null;
        }
      },
    },
  };
}
