import {
  CONTRACT_STATUSES,
  LIMITS,
  PROJECT_LABEL,
  RESULTS,
  RUN_ID_RE,
  RUN_KINDS,
  RUNTIMES,
  TOKEN,
  VERDICTS,
  type ContractStatus,
} from "./workflow";

// M3a execution history — client-safe contract. Nothing here carries a raw payload, command,
// manifest value object or diagnostic: every public field is bounded and derived on the server.

export const EXECUTION_KINDS = ["all", "build", "review", "spec", "plan", "diff"] as const;
export type ExecutionKind = (typeof EXECUTION_KINDS)[number];

export const EXECUTION_STATES = [
  "all", "open", "finished", "success", "failure", "blocked", "cancelled", "interrupted",
  "approve", "approve-with-changes", "reject", "unknown",
] as const;
export type ExecutionState = (typeof EXECUTION_STATES)[number];

export const EXECUTION_PERIODS = ["7d", "30d", "all"] as const;
export type ExecutionPeriod = (typeof EXECUTION_PERIODS)[number];

export const EXECUTION_PAGE_SIZE = 20;
export const EXECUTION_CURSOR_MAX = 1024;

export const EXECUTION_WARNINGS = [
  "orphan",
  "role-conflict",
  "kind-role-conflict",
  "contract-missing",
  "contract-invalid",
  "result-unknown",
  "verdict-unknown",
  "period-unknown",
  "attempt-timestamp-unknown",
  "duration-unknown",
] as const;
export type ExecutionWarning = (typeof EXECUTION_WARNINGS)[number];

export type ExecutionOutcome =
  | "open" | "success" | "failure" | "blocked"
  | "cancelled" | "interrupted"
  | "approve" | "approve-with-changes" | "reject" | "unknown";

export type ExecutionFilters = {
  project: string | null;
  kind: ExecutionKind;
  state: ExecutionState;
  period: ExecutionPeriod;
  // Exact run context: both destination and run ID, always period All.
  run: { project: string; runId: string } | null;
  limit: number;
};

const CONTROLLED_KEYS = ["project", "kind", "state", "period", "run", "cursor", "limit"] as const;
const CONTROLLED = new Set<string>(CONTROLLED_KEYS);

function invalid(error: string): { ok: false; error: string } {
  return { ok: false, error };
}

function boundedLabelled(v: string): boolean {
  return v.length <= LIMITS.project && PROJECT_LABEL.test(v);
}

// One parser for the URL contract (page) and the API (route): a malformed value refuses the
// whole scope instead of silently widening it. Duplicate or unknown control parameters are
// malformed, never "last wins".
export function parseExecutionSearch(
  params: URLSearchParams,
): { ok: true; filters: ExecutionFilters } | { ok: false; error: string } {
  for (const key of params.keys()) {
    if (!CONTROLLED.has(key)) return invalid(`unknown parameter: ${key}`);
  }
  for (const key of CONTROLLED_KEYS) {
    if (params.getAll(key).length > 1) return invalid(`duplicate parameter: ${key}`);
  }

  const rawProject = params.get("project");
  const project = rawProject === null ? null : rawProject;
  if (project !== null && !boundedLabelled(project)) return invalid("invalid project");

  const kind = params.get("kind") ?? "all";
  if (!(EXECUTION_KINDS as readonly string[]).includes(kind)) return invalid("invalid kind");

  const stateRaw = params.get("state") ?? "all";
  if (!(EXECUTION_STATES as readonly string[]).includes(stateRaw)) return invalid("invalid state");

  const rawRun = params.get("run");
  if (rawRun !== null && !RUN_ID_RE.test(rawRun)) return invalid("invalid run");

  const rawPeriod = params.get("period");
  let period: ExecutionPeriod = (rawPeriod ?? "30d") as ExecutionPeriod;
  if (!(EXECUTION_PERIODS as readonly string[]).includes(period)) return invalid("invalid period");

  let run: ExecutionFilters["run"] = null;
  if (rawRun !== null) {
    if (project === null) return invalid("run requires a destination");
    run = { project, runId: rawRun };
    if (rawPeriod !== null && rawPeriod !== "all") return invalid("run context uses period all");
    period = "all";
  }

  const rawLimit = params.get("limit");
  const limit = rawLimit === null ? EXECUTION_PAGE_SIZE : Number(rawLimit);
  if (!Number.isInteger(limit) || limit < 1 || limit > EXECUTION_PAGE_SIZE) return invalid("invalid limit");

  return { ok: true, filters: { project, kind: kind as ExecutionKind, state: stateRaw as ExecutionState, period, run, limit } };
}

export function serializeExecutionSearch(
  params: URLSearchParams,
  filters: ExecutionFilters,
): URLSearchParams {
  const next = new URLSearchParams(params);
  for (const key of ["project", "kind", "state", "period", "run", "limit"] as const) next.delete(key);
  if (filters.project !== null) next.set("project", filters.project);
  if (filters.kind !== "all") next.set("kind", filters.kind);
  if (filters.state !== "all") next.set("state", filters.state);
  if (filters.period !== "30d") next.set("period", filters.period);
  if (filters.run !== null) next.set("run", filters.run.runId);
  if (filters.limit !== EXECUTION_PAGE_SIZE) next.set("limit", String(filters.limit));
  return next;
}

// The canonical filter identity a cursor is bound to. Changing any filter (or the run context)
// invalidates every cursor of the previous scope.
export function executionFilterIdentity(filters: ExecutionFilters): string {
  return JSON.stringify({
    project: filters.project,
    kind: filters.kind,
    state: filters.state,
    period: filters.period,
    run: filters.run ? `${filters.run.project}/${filters.run.runId}` : null,
  });
}

export type ExecutionRow = {
  destination: string;
  runId: string;
  latestEventId: number;
  startEventId: number | null;
  finishEventId: number | null;
  role: "builder" | "reviewer" | null;
  kind: (typeof RUN_KINDS)[number] | null;
  runtime: string | null;
  model: string | null;
  caller: string | null;
  profile: string | null;
  phase: string | null;
  target: string | null;
  firstDispatchedAt: string | null;
  attemptStartedAt: string | null;
  finishedAt: string | null;
  outcome: ExecutionOutcome;
  terminal: boolean;
  contractStatus: ContractStatus | null;
  durationMs: number | null;
  warnings: ExecutionWarning[];
};

export type ExecutionEventFact = {
  id: number;
  role: string | null;
  ts: string;
  payload: Record<string, unknown> | null;
};

export type ExecutionFacts = {
  project: string;
  runId: string;
  latestEventId: number;
  start: ExecutionEventFact | null;
  finish: ExecutionEventFact | null;
  firstStartTs: string | null;
  lastFinishTs: string | null;
};

function isRole(v: unknown): v is "builder" | "reviewer" {
  return v === "builder" || v === "reviewer";
}

function finiteMs(ts: string | null): number | null {
  if (ts === null) return null;
  const ms = Date.parse(ts);
  return Number.isFinite(ms) ? ms : null;
}

function boundedString(v: unknown, cap: number, pattern?: RegExp): string | null {
  return typeof v === "string" && v.length > 0 && v.length <= cap && !/[\u0000-\u001f\u007f]/.test(v)
    && (!pattern || pattern.test(v)) ? v : null;
}

function contractOf(payload: Record<string, unknown> | null): ContractStatus {
  if (payload === null) return "ok";
  const raw = payload.contract_status;
  if (raw === undefined || raw === null) return "ok";
  return typeof raw === "string" && (CONTRACT_STATUSES as readonly string[]).includes(raw)
    ? (raw as ContractStatus) : "invalid";
}

// Terminal lifecycle: the latest admissible finish applies only when it is newer than the
// latest admissible start; a later start reopens the same destination/run.
export function terminalOf(facts: Pick<ExecutionFacts, "start" | "finish">): boolean {
  return facts.finish !== null && (facts.start === null || facts.finish.id > facts.start.id);
}

// Duration uses only the latest start preceding the selected finish with finite, ordered
// timestamps. Never zero, never negative, never borrowed from another attempt.
export function executionDurationMs(startTs: string | null, finishTs: string | null): number | null {
  const start = finiteMs(startTs);
  const finish = finiteMs(finishTs);
  if (start === null || finish === null || finish < start) return null;
  return finish - start;
}

// The period anchor: first finite start timestamp by event ID, else the latest finite finish
// timestamp. Returns the raw ISO string or null, so the DB filter and the mapper share it.
export function executionPeriodAnchor(firstStartTs: string | null, lastFinishTs: string | null): string | null {
  if (finiteMs(firstStartTs) !== null) return firstStartTs;
  if (finiteMs(lastFinishTs) !== null) return lastFinishTs;
  return null;
}

export function executionStateMatches(state: Exclude<ExecutionState, "all">, row: Pick<ExecutionRow, "terminal" | "outcome">): boolean {
  if (state === "open") return !row.terminal;
  if (state === "finished") return row.terminal;
  if (!row.terminal) return false;
  return row.outcome === state;
}

// The single pure projection from joined event facts to a public row. `null` means the row is
// not an execution: illegal destination/run identity, or a known merge-only run.
export function buildExecutionRow(facts: ExecutionFacts): ExecutionRow | null {
  if (!PROJECT_LABEL.test(facts.project) || !RUN_ID_RE.test(facts.runId)) return null;
  const start = facts.start && isRole(facts.start.role) ? facts.start : null;
  const finish = facts.finish && isRole(facts.finish.role) ? facts.finish : null;
  const role = (start !== null && isRole(start.role) ? start.role : null)
    ?? (finish !== null && isRole(finish.role) ? finish.role : null);
  const startPayload = start?.payload ?? null;
  const finishPayload = finish?.payload ?? null;
  const rawKind = startPayload?.kind;
  const kind = typeof rawKind === "string" && (RUN_KINDS as readonly string[]).includes(rawKind)
    ? (rawKind as (typeof RUN_KINDS)[number]) : null;
  if (kind === "merge") return null;

  const terminal = finish !== null && (start === null || finish.id > start.id);
  const contractStatus = terminal ? contractOf(finishPayload) : null;
  const warnings: ExecutionWarning[] = [];
  if (start === null && finish !== null) warnings.push("orphan");
  if (start !== null && finish !== null && start.role !== finish.role) warnings.push("role-conflict");
  if (kind !== null && role !== null) {
    const expected = kind === "build" ? "builder" : "reviewer";
    if (role !== expected) warnings.push("kind-role-conflict");
  }

  let outcome: ExecutionOutcome;
  if (!terminal) {
    outcome = "open";
  } else if (contractStatus === "cancelled" || contractStatus === "interrupted") {
    outcome = contractStatus;
  } else if (role === "builder") {
    const result = finishPayload?.result;
    if (typeof result === "string" && (RESULTS as readonly string[]).includes(result)) {
      outcome = result as ExecutionOutcome;
    } else {
      outcome = "unknown";
      warnings.push("result-unknown");
    }
  } else if (role === "reviewer") {
    const verdict = finishPayload?.verdict;
    if (contractStatus === "ok" && typeof verdict === "string" && (VERDICTS as readonly string[]).includes(verdict)) {
      outcome = verdict as ExecutionOutcome;
    } else {
      outcome = "unknown";
      warnings.push("verdict-unknown");
    }
  } else {
    outcome = "unknown";
  }
  if (contractStatus === "missing") warnings.push("contract-missing");
  if (contractStatus === "invalid") warnings.push("contract-invalid");

  const firstDispatchedAt = finiteMs(facts.firstStartTs) !== null ? facts.firstStartTs : null;
  if (executionPeriodAnchor(facts.firstStartTs, facts.lastFinishTs) === null) warnings.push("period-unknown");
  const attemptStartedAt = start && finiteMs(start.ts) !== null ? start.ts : null;
  if (start !== null && attemptStartedAt === null) warnings.push("attempt-timestamp-unknown");
  const finishedAt = terminal && finish && finiteMs(finish.ts) !== null ? finish.ts : null;
  const durationMs = terminal ? executionDurationMs(start?.ts ?? null, finish?.ts ?? null) : null;
  if (terminal && durationMs === null) warnings.push("duration-unknown");

  return {
    destination: facts.project,
    runId: facts.runId,
    latestEventId: facts.latestEventId,
    startEventId: start?.id ?? null,
    finishEventId: finish?.id ?? null,
    role,
    kind,
    runtime: typeof startPayload?.runtime === "string" && (RUNTIMES as readonly string[]).includes(startPayload.runtime)
      ? startPayload.runtime : null,
    model: boundedString(startPayload?.model, LIMITS.model),
    caller: boundedString(startPayload?.caller, 32),
    profile: boundedString(startPayload?.requested_profile, LIMITS.phase),
    phase: boundedString(startPayload?.phase, LIMITS.phase, TOKEN),
    target: boundedString(startPayload?.target, LIMITS.target),
    firstDispatchedAt,
    attemptStartedAt,
    finishedAt,
    outcome,
    terminal,
    contractStatus,
    durationMs,
    warnings,
  };
}

export type ExecutionCursor = {
  v: 1;
  fence: number;
  lastId: number;
  project: string;
  runId: string;
  cutoff: string | null;
  identity: string;
};

function safeInt(v: unknown): v is number {
  return typeof v === "number" && Number.isSafeInteger(v) && v >= 0;
}

export function encodeExecutionCursor(cursor: ExecutionCursor): string {
  return Buffer.from(JSON.stringify(cursor), "utf8").toString("base64url");
}

export function decodeExecutionCursor(raw: string): ExecutionCursor | null {
  if (raw.length === 0 || raw.length > EXECUTION_CURSOR_MAX || !/^[A-Za-z0-9_-]+$/.test(raw)) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(Buffer.from(raw, "base64url").toString("utf8"));
  } catch {
    return null;
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
  const c = parsed as Record<string, unknown>;
  if (c.v !== 1 || !safeInt(c.fence) || !safeInt(c.lastId)) return null;
  if (typeof c.project !== "string" || !boundedLabelled(c.project)) return null;
  if (typeof c.runId !== "string" || !RUN_ID_RE.test(c.runId)) return null;
  if (c.cutoff !== null && (typeof c.cutoff !== "string" || !Number.isFinite(Date.parse(c.cutoff)))) return null;
  if (typeof c.identity !== "string" || c.identity.length > 512) return null;
  return {
    v: 1,
    fence: c.fence,
    lastId: c.lastId,
    project: c.project,
    runId: c.runId,
    cutoff: c.cutoff as string | null,
    identity: c.identity,
  };
}

export function executionPeriodCutoff(period: ExecutionPeriod, nowMs: number): string | null {
  if (period === "all") return null;
  const days = period === "7d" ? 7 : 30;
  return new Date(nowMs - days * 86_400_000).toISOString();
}
