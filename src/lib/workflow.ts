// Workflow event hub — shared vocabulary (spec §3.2 columns + §4.2 event
// matrix). Pure data + types: imported by the ingress validator (collectors),
// the DB module and, later, the /workflows UI. Never put node-only code here.

export const EVENT_TYPES = [
  "run-started", "run-finished", "question", "question-resolved",
  "attention-needed", "turn-stopped", "turn-started", "question-answered",
  "merge-approved", "gc-removed",
  "tool-used", "subagent-started", "subagent-stopped",
  "mission-status-updated", "mission-finished", "pr-opened",
] as const;
export type EventType = (typeof EVENT_TYPES)[number];

export const EMITTERS = [
  "wrapper", "claude-pretool", "claude-posttool", "claude-notification",
  "claude-stop", "codex-stop", "codex-permission",
  "claude-userprompt", "codex-userprompt", "jaxos",
  "claude-toolused", "codex-toolused", "claude-subagentstart", "codex-subagentstart", "claude-subagentstop", "codex-subagentstop",
] as const;
export type Emitter = (typeof EMITTERS)[number];

export const SOURCES = ["deterministic", "behavioral"] as const;
export type Source = (typeof SOURCES)[number];

export const ROLES = ["builder", "reviewer", "lead", "adhoc"] as const;
export type Role = (typeof ROLES)[number];

export const RUNTIMES = ["opencode-grok", "opencode-deepseek", "opencode-builder", "codex", "claude"] as const;
export const CONTRACT_STATUSES = ["ok", "missing", "invalid", "cancelled", "interrupted"] as const;
export type ContractStatus = (typeof CONTRACT_STATUSES)[number];

// MOA-474 §6: where the deciding evidence came from on a run-finished row. Additive/
// optional on every contract_status except `interrupted` (mandatory there) and
// `cancelled` (forbidden there — that row shape is untouched by this spec).
export const STAGES = ["runtime", "verify", "build", "worker", "report"] as const;
export type Stage = (typeof STAGES)[number];

// Native Codex emitters (MOA-469 §3): their hooks may fire with no TMUX at all, and a pane-less
// event is legal — the canonical session UUID is the identity. All Codex identity queries MUST
// constrain to these emitters, never match a UUID alone across runtimes.
export const CODEX_EMITTERS = [
  "codex-stop", "codex-permission", "codex-userprompt",
  "codex-toolused", "codex-subagentstart", "codex-subagentstop", // Phase 3: same pane-less identity rule
] as const;
export type CodexEmitter = (typeof CODEX_EMITTERS)[number];
export const isCodexEmitter = (emitter: Emitter): emitter is CodexEmitter =>
  (CODEX_EMITTERS as readonly string[]).includes(emitter);

// Canonical session UUID (lowercase, hyphenated — the shape Codex emits). Shared by the hook,
// the ingress validator, and the Codex collector.
export const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

// jaxflow run ledger (spec §2.6): kind/caller enums for run-started, verdict/result
// for run-finished's role-conditional outcome field.
export const RUN_KINDS = ["spec", "plan", "diff", "build", "merge"] as const;
export const CALLERS = ["claude", "codex", "jaxos"] as const;
export const VERDICTS = ["approve", "approve-with-changes", "reject"] as const;
export const RESULTS = ["success", "failure", "blocked"] as const;

// jaxflow child.log diagnosis (MOA-470 §4.3): emitted on run-finished iff
// contract_status is missing/invalid, always as a pair. MOA-495 2.1 added the last
// three (Jev Choice refinement of the regex-only fallback -- offline eval: 2/11 vs 10/11).
export const CHILD_LOG_REASONS = [
  "provider-limit", "crash", "no-report", "unknown", "hung", "test-failure", "contract-violation",
] as const;

export const DELIVERIES = ["pending", "delivered", "local", "suppressed"] as const;
export type Delivery = (typeof DELIVERIES)[number];

// Signal quality is a property of the emitter (spec §2 deterministic-emitter
// invariant): only the pane-derived Notification hook is behavioral.
export const SOURCE_BY_EMITTER: Record<Emitter, Source> = {
  wrapper: "deterministic",
  "claude-pretool": "deterministic",
  "claude-posttool": "deterministic",
  "claude-notification": "behavioral",
  "claude-stop": "deterministic",
  "codex-stop": "deterministic",
  "codex-permission": "deterministic",
  "claude-userprompt": "deterministic",
  "codex-userprompt": "deterministic",
  jaxos: "deterministic",
  "claude-toolused": "deterministic",
  "codex-toolused": "deterministic",
  "claude-subagentstart": "deterministic",
  "codex-subagentstart": "deterministic",
  "claude-subagentstop": "deterministic",
  "codex-subagentstop": "deterministic",
};

export type MatrixRow = {
  emitters: readonly Emitter[];
  roles: readonly Role[];
  runScoped: boolean; // run_id required (true) or forbidden (false)
  policy: "local" | "forward";
  suppressWhileQuestionPending?: true;
};

// The exhaustive allowlist (spec §4.2 matrix). Payload key sets live in the
// validator next to their checks (collectors/workflow-events.ts).
export const EVENT_MATRIX: Record<EventType, MatrixRow> = {
  "run-started": { emitters: ["wrapper"], roles: ["builder", "reviewer"], runScoped: true, policy: "local" },
  "run-finished": { emitters: ["wrapper"], roles: ["builder", "reviewer"], runScoped: true, policy: "forward" },
  question: { emitters: ["claude-pretool"], roles: ["lead", "adhoc"], runScoped: false, policy: "forward" },
  "question-resolved": { emitters: ["claude-posttool"], roles: ["lead", "adhoc"], runScoped: false, policy: "local" },
  "attention-needed": {
    emitters: ["claude-notification", "codex-permission", "jaxos"], roles: ["lead", "adhoc"], runScoped: false, policy: "forward",
    suppressWhileQuestionPending: true,
  },
  // MOA-469 live correction (approved 2026-09-10): stops are LOCAL telemetry for both emitters —
  // never forwarded to Telegram. Ingestion, classification, timelines and session correlation are
  // unchanged; listPending additionally excludes any legacy pending stop row (workflows.ts).
  "turn-stopped": { emitters: ["claude-stop", "codex-stop"], roles: ["lead", "adhoc"], runScoped: false, policy: "local" },
  "turn-started": { emitters: ["claude-userprompt", "codex-userprompt"], roles: ["lead", "adhoc"], runScoped: false, policy: "local" },
  "question-answered": { emitters: ["jaxos"], roles: ["lead", "adhoc"], runScoped: false, policy: "local" },
  "merge-approved": { emitters: ["wrapper"], roles: ["lead"], runScoped: false, policy: "forward" },
  // MOA-487 §5 Decision 4: reserved for MOA-488's "mission" entity — placeholder emitters/roles
  // mirror merge-approved; MOA-488 owns emission, payload shape and the real call site.
  "mission-status-updated": { emitters: ["wrapper"], roles: ["lead"], runScoped: false, policy: "forward" },
  "mission-finished": { emitters: ["wrapper"], roles: ["lead"], runScoped: false, policy: "forward" },
  // MOA-465: PR-preset delivery — same shape class as merge-approved (forward,
  // non-run-scoped, lead-only). Payload validated in workflow-events.ts; dedup key
  // (project, branch, sha) in workflows.ts insertEvent.
  "pr-opened": { emitters: ["wrapper"], roles: ["lead"], runScoped: false, policy: "forward" },
  // gc's own audit row (round 3 spec G7/G8) — local telemetry, same choice as turn-stopped.
  "gc-removed": { emitters: ["wrapper"], roles: ["lead"], runScoped: false, policy: "local" },
  // Phase 3 (spec §6, Decision 1): dashboard-only telemetry — local, never forwarded, same
  // roles/scoping as turn-started. HOOK_TELEMETRY_TYPES below is what the 30-day prune and the
  // timeline exclusion key on, so the three stay ONE list.
  "tool-used": { emitters: ["claude-toolused", "codex-toolused"], roles: ["lead", "adhoc"], runScoped: false, policy: "local" },
  "subagent-started": { emitters: ["claude-subagentstart", "codex-subagentstart"], roles: ["lead", "adhoc"], runScoped: false, policy: "local" },
  "subagent-stopped": { emitters: ["claude-subagentstop", "codex-subagentstop"], roles: ["lead", "adhoc"], runScoped: false, policy: "local" },
};

// MOA-487 §5 Decision 2: every forward-policy type, derived from the matrix — never a second
// hardcoded list, so the option list and the matrix can never drift.
export const FORWARDABLE_EVENT_TYPES: readonly EventType[] = EVENT_TYPES.filter(
  (t) => EVENT_MATRIX[t].policy === "forward",
);

// MOA-487 §5 Decision 5: the three "needs Rafa now" signals default on. Also the migration's
// column DEFAULT literal (Task 2) — index.test.ts asserts the two never drift apart.
export const DEFAULT_FORWARD_TYPES: readonly EventType[] = ["question", "attention-needed", "mission-finished"];

// Phase 3 (spec §4.3, Decision 6): the three high-volume hook kinds share one retention window —
// pruned lazily on insert, excluded from the project timeline, never forwarded.
export const HOOK_TELEMETRY_TYPES = ["tool-used", "subagent-started", "subagent-stopped"] as const;
export const HOOK_TELEMETRY_RETENTION_MS = 30 * 24 * 60 * 60 * 1000; // = the 30-day auto-hide cap (F7/F14)
export const HOOK_TELEMETRY_TYPES_SQL = HOOK_TELEMETRY_TYPES.map((t) => `'${t}'`).join(", ");

// Every cap in one place. Emitters truncate free text BEFORE posting; the
// validator rejects over-cap values, it never trims.
export const LIMITS = {
  requestBytes: 64 * 1024,
  payloadBytes: 16 * 1024,
  project: 64,
  runId: 64,
  phase: 64,
  reportPath: 512,
  summary: 200, // spec §3.1: report summary is ONE line, ≤200 chars
  toolUseId: 128,
  questions: 10,
  questionText: 2000,
  header: 64,
  options: 10,
  optionLabel: 200,
  optionDescription: 1000,
  reason: 300,
  attentionExcerpt: 2000,
  capsuleExcerpt: 200,
  messageTail: 2000,
  harnessSession: 128,
  error: 500,
  tmuxTarget: 128,
  pane: 12,
  pollLimitMax: 100,
  ackIdsMax: 100,
  answerValue: 2000,
  answerReply: 20500,
  freeformReply: 2000, // spec §4.6.2 guard 6 — matches answerValue, not the multi-question bundle cap
  target: 512,
  callerSession: 128,
  callerPane: 128,
  model: 384,
  effort: 64,
  verify: 500,
  build: 500,
  session: 80,
  repo: 200,
  childTail: 4096,
  // MOA-495: log_context (2.1) and tally (2.2) run-finished fields. logContextChunk
  // matches scripts/jaxflow_run.py's LOG_CONTEXT_STORE_CHARS -- kept small (not the
  // full ~2KiB Jev scored) so up to 6 chunks plus childTail stay well under
  // LIMITS.payloadBytes.
  logContextChunk: 500,
  logContextMax: 6,
  tally: 300,
  tool: 64, // spec §4.2: a tool NAME only
  agentId: 128,
  agentType: 64,
  // Phase 2 (Mission Control B): argv bounds for the dashboard's own routes. `verify`/`build`
  // above are the ingress caps a build's run-started payload already enforces; these three
  // fields never reach a run-started payload, so they get their own explicit bounds here.
  checks: 2000,
  focus: 2000,
  whitelist: 4096,
  // Mission tracking (spec §6a — shared by the /api/mission routes, the jaxflow CLI's own
  // advisory checks, and the mission-status-updated/mission-finished event payload contract).
  missionName: 80,
  missionGoal: 280,
  missionStatusLine: 280,
  milestoneTitle: 60,
  missionMilestonesMax: 12,
  // MOA-465: a real GitHub PR URL is well under 300 chars.
  prUrl: 300,
} as const;

export type WorkflowQuestionAnswer =
  | { question_number: number; kind: "text"; value: string }
  | { question_number: number; kind: "options"; values: number[] };
export type WorkflowQuestionShape = { multiSelect: boolean; option_count: number };
export type AnswerRequest = { question_event_id: number; reply: string };

export type IndexedReplyResult =
  | { ok: true; answers: WorkflowQuestionAnswer[] }
  | { ok: false; error: string };

const LINE_RE = /^[Qq]?([1-9]|10):[ \t]*(.+)$/;
const OPTIONS_RE = /^\d+(?:[ \t]*,[ \t]*\d+)*$/;

export function parseIndexedReply(reply: string, shapes: WorkflowQuestionShape[]): IndexedReplyResult {
  const lines = reply.split(/\r\n?|\n/).filter((line) => line.length > 0);
  if (lines.length !== shapes.length) return { ok: false, error: "reply incomplete" };
  const byNumber = new Map<number, WorkflowQuestionAnswer>();
  for (const line of lines) {
    const m = LINE_RE.exec(line);
    if (!m) return { ok: false, error: "reply malformed" };
    const question_number = Number(m[1]);
    const value = m[2].trim();
    if (!value) return { ok: false, error: "reply malformed" };
    if (byNumber.has(question_number)) return { ok: false, error: "reply malformed" };
    const shape = shapes[question_number - 1];
    if (!shape) return { ok: false, error: "reply malformed" };
    if (OPTIONS_RE.test(value)) {
      const values = value.split(/[ \t]*,[ \t]*/).map(Number);
      if (new Set(values).size !== values.length) return { ok: false, error: "reply malformed" };
      if (values.some((n) => n < 1 || n > shape.option_count)) return { ok: false, error: "reply malformed" };
      if (shape.multiSelect ? values.length < 1 : values.length !== 1) return { ok: false, error: "reply malformed" };
      byNumber.set(question_number, { question_number, kind: "options", values: [...values].sort((a, b) => a - b) });
    } else {
      if (value.length > LIMITS.answerValue) return { ok: false, error: "reply malformed" };
      byNumber.set(question_number, { question_number, kind: "text", value });
    }
  }
  for (let i = 1; i <= shapes.length; i += 1) {
    if (!byNumber.has(i)) return { ok: false, error: "reply incomplete" };
  }
  return { ok: true, answers: [...byNumber.values()].sort((a, b) => a.question_number - b.question_number) };
}

export const PROJECT_SLUG = /^[a-z0-9][a-z0-9-]*$/; // names tmux sessions (jax-<slug>-lead)
export const TOKEN = /^[A-Za-z0-9._-]+$/; // run_id, phase
export const TOOL_USE_ID = /^[A-Za-z0-9_-]+$/;
export const TMUX_TARGET = /^[\x21-\x7e]+$/; // printable ASCII, no whitespace — exact pane target string
export const PANE_RE = /^%[0-9]{1,10}$/;
export const SHA40 = /^[0-9a-f]{40}$/;
export const GITHUB_PR_URL_RE = /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/pull\/[1-9][0-9]*$/;
// One trimmed line: first and last characters non-whitespace, no tab or newline between,
// and no unpaired surrogate. The exact contract `jaxflow merge` applies to `--phase`
// before it commits (spec §2.9 `phase-invalid`) — the two layers must not disagree about
// what a phase title is. The `u` flag makes the pattern match code POINTS, so a valid
// surrogate pair (an emoji) is one astral character and never matches `\p{Surrogate}`,
// while a lone half stays a surrogate code unit and does: the producer rejects that value
// too, because argv decodes with `surrogateescape` and it cannot be encoded to UTF-16
// (part-2 cold review round 3). `[\s\S]`, not `.`, in the lookahead: without the `s` flag
// JS `.` excludes U+2028 and U+2029, which the body itself permits, so `A\u2028\uD800`
// slipped past a `.*` guard while the producer refused it — the dangerous direction
// (round 4... found in round 5).
export const PHASE_TITLE_RE = /^(?![\s\S]*\p{Surrogate})\S(?:[^\t\r\n]*\S)?$/u;
export const PROJECT_LABEL = /^[\x20-\x7e]{1,64}$/; // printable ASCII incl. space, excludes control chars — Finding 34
export const TMUX_INCARNATION_RE = /^[0-9]{1,10}:[0-9]{1,20}$/; // "<pid>:<start_time>" — Finding 3

// Canonical per-pane identity (Spec B §4.1): a pane id is reused after a tmux server restart, so
// every per-pane read must key on the pair, never on `pane` alone. `PaneRef` stops a caller from
// swapping two positional strings at a function boundary — a caller holding only a pane id cannot
// construct this type, so an omission is a type error rather than a convention. `paneKey` is the
// SQL-side half of the same invariant (migration 5, db/workflows.ts): it computes EXACTLY the
// expression the generated `pane_key` column computes, so the SQL and this function can never
// drift. Two plain strings, not a `PaneRef` object — callers hold the pair in both shapes
// (sometimes a `PaneRef`, sometimes two separate row fields straight off a SELECT).
export type PaneRef = { pane: string; tmuxIncarnation: string };
export const paneKey = (pane: string, tmuxIncarnation: string): string => `${pane}:${tmuxIncarnation}`;

// Shared reply guard (cold review Finding 5): full C0 range + DEL, EXCLUDING \n/\r — those are
// structurally meaningful (line separators for a structured multi-answer reply) and are checked
// separately, with their own error, by each caller (parseAnswerBody, claimFreeformAnswer).
export const CONTROL_CHARS_RE = /[\x00-\x09\x0b\x0c\x0e-\x1f\x7f]/;
export const CAPSULE_STATUSES = ["done", "needs_input", "waiting", "blocked", "unknown"] as const;

// The rung that decided a turn-stopped's status. `tag` is the only value a producer may send;
// every other value is assigned by the derivation (spec §3.2, §3.4).
export type CapsuleStatus = (typeof CAPSULE_STATUSES)[number];

export const CAPSULE_RULES = [
  "tag", "question_open", "attention", "run_in_flight", "run_finished",
  "trailing_question", "deferred", "classified", "abandoned",
  "classifier-off", // MOA-502 Decision 1: settled on the spot when integrations.classifier is off
] as const;
export type CapsuleRule = (typeof CAPSULE_RULES)[number];

export type WorkflowEventInput = {
  run_id: string | null;
  project: string;
  role: Role;
  type: EventType;
  source: Source;
  emitter: Emitter;
  payload: Record<string, unknown>;
  pane?: string | null;
  tmux_incarnation?: string | null;
  harness_session?: string | null;
};

export type WorkflowEventRow = WorkflowEventInput & {
  id: number;
  ts: string;
  delivery: Delivery;
  forwarded_at: string | null;
};

// ---- Phase 2 (Mission Control B, spec §7.4 / Decision 18) ----------------------------------
// The SAME bounds run client-side (the card's forms disable submit) and server-side (the route
// handlers refuse before any spawn). Pure, no node imports — importable by both.

export const RUN_ID_RE = /^[0-9a-f]{12}$/; // `uuid.uuid4().hex[:12]` — every jaxflow run id
// \n, \r, \t and NUL: the four characters that would break jaxflow's own final-output-line
// parsing or argv handling (spec Decision 18, F7). Deliberately narrower than CONTROL_CHARS_RE,
// which is the multi-line `reply` guard and allows \n/\r.
export const SINGLE_LINE_BAN_RE = /[\n\r\t\0]/;

// merge --branch: non-empty, ≤512 UTF-16 units, no whitespace, no NUL — mirrors `branch-invalid`
// (:4493-4498). Round-2 F1: `\s` alone does not catch NUL, so it is banned explicitly.
export function validRefName(v: unknown): v is string {
  return typeof v === "string" && v.length >= 1 && v.length <= LIMITS.target && !/[\s\0]/.test(v);
}

// --checks / --verify / --build / --whitelist: non-blank, single line, capped. The single-line
// rule has NO jaxflow-side equivalent (F7) — this is the only defense.
export function validSingleLineCommand(v: unknown, cap: number): v is string {
  return typeof v === "string" && v.trim().length > 0 && v.length <= cap && !SINGLE_LINE_BAN_RE.test(v);
}

// review --focus: optional; when present, a non-empty single line.
export function validOptionalFocus(v: unknown): v is string | undefined {
  return v === undefined || validSingleLineCommand(v, LIMITS.focus);
}

// review/build --phase: the run-id/phase token shape (`malformed phase`, SKILL.md:26-28).
export function validPhaseToken(v: unknown): v is string {
  return typeof v === "string" && v.length >= 1 && v.length <= LIMITS.phase && TOKEN.test(v);
}

// Mission text fields (spec §6a): trimmed, non-empty, one line (no control character or
// newline), capped at `max`. The caller supplies an
// already-trimmed value; an untrimmed one is refused outright, never silently trimmed. Used for
// name/goal/milestone-title (always required) and for the `status` command's own text (which
// must NOT be blank — spec §7 `malformed status`). The stored status_line CAN be '' (spec §6
// Decision 1); that case is validated separately, only in the event payload (Task 4), which is
// the one place '' is legal.
export function validMissionText(v: unknown, max: number): v is string {
  return typeof v === "string" && v.length >= 1 && v.length <= max && v === v.trim() && !/[\x00-\x1f\x7f]/.test(v);
}

// Whole-string positive integer (spec §6 Decision 5): a milestone TITLE matching this shape is
// refused at `start` (a milestone can never be literally named "3"); `mark`'s own <milestone>
// argument matching this shape is read as a 1-based index instead of a title
// (src/server/db/missions.ts's resolveMilestoneIndex, and the /api/mission start route).
export const MISSION_MILESTONE_NUMERIC_RE = /^[1-9][0-9]*$/;
