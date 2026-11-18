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
