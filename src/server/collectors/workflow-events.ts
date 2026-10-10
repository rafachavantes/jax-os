import {
    CALLERS, CAPSULE_STATUSES, CHILD_LOG_REASONS, CONTRACT_STATUSES, CONTROL_CHARS_RE, EMITTERS, EVENT_MATRIX, EVENT_TYPES, GITHUB_PR_URL_RE, isCodexEmitter, LIMITS, PANE_RE, PHASE_TITLE_RE, PROJECT_LABEL, STAGES,
    ROLES, RUN_KINDS, RUNTIMES, RESULTS, SHA40, SOURCES, SOURCE_BY_EMITTER, TMUX_INCARNATION_RE, TMUX_TARGET, TOKEN,
  TOOL_USE_ID, UUID_RE, VERDICTS, type AnswerRequest, type Emitter, type EventType, type Role, type WorkflowEventInput,
  type Source, type WorkflowEventRow, validMissionText,
} from "../../lib/workflow";
import type { WorkflowSessionInput } from "../db/workflows";

// Ingress validation for POST /api/workflow/events — the event matrix (spec
// §4.2) as code. Pure: body in, normalized event or a reason out. Every check
// is a hard reject (no trimming, no defaults beyond run_id=null / payload={}).

export type ParseResult = { ok: true; event: WorkflowEventInput } | { ok: false; error: string };
type Obj = Record<string, unknown>;

// Phase 3: a tool name / agent id must be one printable line — every control char is refused.
const NO_CONTROL_CHARS = /^[^\x00-\x1f\x7f]+$/;

// Round-3 F1: shared bound for findings.{high,medium,low} (spec §9); duplicated in
// scripts/jaxflow_run.py's FINDINGS_COUNT_MAX (Python and TS share no module).
export const FINDINGS_COUNT_MAX = 999;

class Invalid extends Error {}
function fail(msg: string): never {
  throw new Invalid(msg);
}
function isObj(v: unknown): v is Obj {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function oneOf<T extends string>(v: unknown, list: readonly T[]): v is T {
  return typeof v === "string" && (list as readonly string[]).includes(v);
}
function onlyKeys(o: Obj, allowed: readonly string[], where: string): void {
  for (const k of Object.keys(o)) if (!allowed.includes(k)) fail(`${where}: unknown key ${k}`);
}
function optStr(o: Obj, key: string, max: number, pattern?: RegExp): string | undefined {
  const v = o[key];
  if (v === undefined) return undefined;
  if (v === null) fail(`${key} must not be null`); // only head_sha may be explicitly null (its own check)
  if (typeof v !== "string" || v.length === 0) fail(`${key} must be a non-empty string`);
  if (v.length > max) fail(`${key} exceeds ${max} chars`);
  if (pattern && !pattern.test(v)) fail(`${key} malformed`);
  return v;
}
function reqStr(o: Obj, key: string, max: number, pattern?: RegExp): string {
  const v = optStr(o, key, max, pattern);
  if (v === undefined) fail(`${key} required`);
  return v;
}
function posInt(o: Obj, key: string): number {
  const v = o[key];
  if (!Number.isInteger(v) || (v as number) <= 0) fail(`${key} must be a positive integer`);
  return v as number;
}
function nonNegInt(o: Obj, key: string): number {
  const v = o[key];
  if (!Number.isInteger(v) || (v as number) < 0) fail(`${key} must be a non-negative integer`);
  return v as number;
}

// Deeply nested input can make JSON.stringify throw RangeError; that must be a
// validation failure, never an unhandled 500 (cold review 2026-08-19).
function safeStringify(v: unknown): string | null {
  try {
    return JSON.stringify(v);
  } catch {
    return null;
  }
}

function validateQuestions(v: unknown): void {
  if (!Array.isArray(v) || v.length === 0 || v.length > LIMITS.questions) {
    fail(`questions must be a non-empty array (max ${LIMITS.questions})`);
  }
  for (const q of v) {
    if (!isObj(q)) fail("question must be an object");
    onlyKeys(q, ["question", "header", "options", "multiSelect"], "question");
    reqStr(q, "question", LIMITS.questionText);
    optStr(q, "header", LIMITS.header);
    if (q.multiSelect !== undefined && typeof q.multiSelect !== "boolean") fail("multiSelect must be boolean");
    const opts = q.options;
    if (!Array.isArray(opts) || opts.length === 0 || opts.length > LIMITS.options) {
      fail(`options must be a non-empty array (max ${LIMITS.options})`);
    }
    for (const o of opts) {
      if (!isObj(o)) fail("option must be an object");
      onlyKeys(o, ["label", "description"], "option");
      reqStr(o, "label", LIMITS.optionLabel);
      optStr(o, "description", LIMITS.optionDescription);
    }
  }
}

// Per-type payload schema (spec §4.2 matrix, "ingress payload schema" column).
function validatePayload(type: EventType, role: Role, source: Source, p: Obj): void {
  switch (type) {
    case "run-started": {
      onlyKeys(
        p,
        ["phase", "runtime", "kind", "target", "caller", "caller_session", "caller_pane", "model", "effort", "verify", "build", "session", "repo", "requested_profile", "root_build_run_id", "resumes_run_id", "builder_run_id"],
        "payload",
      );
      reqStr(p, "phase", LIMITS.phase, TOKEN);
      if (!oneOf(p.runtime, RUNTIMES)) fail("runtime not allowed");
      if (!oneOf(p.kind, RUN_KINDS)) fail("kind not allowed");
      reqStr(p, "target", LIMITS.target);
      if (!oneOf(p.caller, CALLERS)) fail("caller not allowed");
      reqStr(p, "caller_session", LIMITS.callerSession);
      optStr(p, "caller_pane", LIMITS.callerPane);
      reqStr(p, "model", LIMITS.model);
      reqStr(p, "effort", LIMITS.effort);
      reqStr(p, "session", LIMITS.session);
      reqStr(p, "repo", LIMITS.repo);
      if (p.kind === "build") {
        reqStr(p, "verify", LIMITS.verify);
        optStr(p, "build", LIMITS.build);
      } else if ("verify" in p || "build" in p) {
        fail("verify/build only for kind build");
      }
      const managed = role === "builder" && p.kind === "build" && p.runtime === "opencode-builder";
      if (managed) {
        if (!oneOf(p.requested_profile, ["default", "fallback"])) {
          fail("requested_profile" in p ? "requested_profile not allowed" : "requested_profile required");
        }
        reqStr(p, "root_build_run_id", 12, /^[0-9a-f]{12}$/);
        optStr(p, "resumes_run_id", 12, /^[0-9a-f]{12}$/);
      } else if ("requested_profile" in p || "root_build_run_id" in p || "resumes_run_id" in p) {
        fail("lineage only for managed builder runs");
      }
      if (p.kind === "diff") {
        // F7 (round 4): required at INGRESS for every NEW diff run-started event -- the
        // real producer (`dispatch_diff_review`, `scripts/jaxflow.py:2074`) always knows
        // its own reviewed build's run id, and an unlinked diff distorts loop grouping
        // (F4/L2). A row already PERSISTED without it (predating this requirement) is
        // still tolerated on the READ side -- `getLoopSummary`'s branch fallback (L2) --
        // never re-validated here; ingress only ever sees a row once, on the way in.
        reqStr(p, "builder_run_id", 12, /^[0-9a-f]{12}$/);
      } else if ("builder_run_id" in p) {
        fail("builder_run_id only for kind diff");
      }
      return;
    }
    case "run-finished": {
      onlyKeys(
        p,
        ["phase", "exit_code", "contract_status", "report_path", "summary", "head_sha", "cleanup_error", "verdict", "result", "tail", "reason", "stage", "diagnostic", "findings", "log_context", "tally"],
        "payload",
      );
      reqStr(p, "phase", LIMITS.phase, TOKEN);
      if (!oneOf(p.contract_status, CONTRACT_STATUSES)) fail("contract_status not allowed");
      // MOA-474 D6: two families. `interrupted` joins `cancelled` in the not-finalized
      // family — the run never reached its own finalize step, so exit_code/report_path
      // stay null exactly like a cancelled row (spec §11.1 null-rules-by-family table).
      const notFinalized = p.contract_status === "cancelled" || p.contract_status === "interrupted";
      const cancelled = p.contract_status === "cancelled";
      // exit_code/report_path are null ONLY alongside contract_status: cancelled
      // (spec §2.4 cancelled-payload shape) — every other status keeps today's checks.
      const ec = p.exit_code;
      if (notFinalized) {
        if (ec !== null) fail("exit_code must be null when contract_status is cancelled or interrupted");
      } else if (!Number.isInteger(ec) || (ec as number) < 0 || (ec as number) > 255) {
        fail("exit_code must be an integer 0-255");
      }
      if (notFinalized) {
        if (p.report_path !== null) fail("report_path must be null when contract_status is cancelled or interrupted");
      } else {
        reqStr(p, "report_path", LIMITS.reportPath);
      }
      // MOA-474 §11.1: `[...v].length` counts Unicode code points, matching the
      // producer's `_bound` (scripts/jaxflow_run.py:617-619) -- plain `.length` counts
      // UTF-16 units and over-rejects a producer-valid summary built from non-BMP chars
      // (the same fix `tail`'s check below already has, workflow-events.ts:190).
      const v = p.summary;
      if (v === undefined) fail("summary required");
      if (typeof v !== "string" || v.length === 0) fail("summary must be a non-empty string");
      if ([...v].length > LIMITS.summary) fail(`summary exceeds ${LIMITS.summary} chars`);
      const summary = v;
      if (/[\u0000-\u001f]/.test(summary)) fail("summary must be a single line"); // control chars incl. CR/LF
      optStr(p, "cleanup_error", LIMITS.error);
      const sha = p.head_sha;
      if (role === "builder") {
        if (!("head_sha" in p)) fail("head_sha required for builder runs");
        if (sha !== null && !(typeof sha === "string" && SHA40.test(sha))) fail("head_sha must be a full 40-hex sha or null");
      } else if (sha !== undefined && sha !== null) {
        fail("head_sha only for builder runs");
      }
      // Reviewer verdicts require an ok report. Builder results may also come from
      // jaxflow verification when the report is missing/invalid (MOA-472).
      if (role === "reviewer" && p.contract_status === "ok") {
        if (!oneOf(p.verdict, VERDICTS)) fail("verdict required for reviewer runs with contract_status ok");
      } else if ("verdict" in p) {
        fail("verdict only for reviewer runs with contract_status ok");
      }
      // Round-3 F1: findings is presence-gated like verdict (hard reject if misplaced),
      // but a bound/type violation silently drops just `findings` (informational,
      // unlike verdict) rather than failing the whole otherwise-valid row.
      if ("findings" in p) {
        if (role !== "reviewer" || p.contract_status !== "ok") {
          fail("findings only for reviewer runs with contract_status ok");
        }
        const f = p.findings;
        const boundedCount = (v: unknown): boolean =>
          Number.isInteger(v) && (v as number) >= 0 && (v as number) <= FINDINGS_COUNT_MAX;
        const valid = isObj(f) && boundedCount(f.high) && boundedCount(f.medium) && boundedCount(f.low);
        if (!valid) delete p.findings;
      }
      // MOA-495 2.2: tally is presence-gated like verdict/findings -- a review-round
      // comparison has nothing to say for a builder run or a report that isn't ok.
      if ("tally" in p) {
        if (role !== "reviewer" || p.contract_status !== "ok") {
          fail("tally only for reviewer runs with contract_status ok");
        }
        const t = p.tally;
        if (typeof t !== "string" || t.length === 0) fail("tally must be a non-empty string");
        if ([...t].length > LIMITS.tally) fail(`tally exceeds ${LIMITS.tally} chars`);
        if (/[\u0000-\u001f]/.test(t)) fail("tally must be a single line");
      }
      if (role === "builder" && !cancelled) {
        if (p.contract_status === "ok" && !("result" in p)) fail("result required for builder runs with contract_status ok");
        if (p.contract_status === "interrupted" && p.result !== "failure") fail("result must be failure for an interrupted builder row");
        if ("result" in p && !oneOf(p.result, RESULTS)) fail("result not allowed");
      } else if ("result" in p) {
        fail("result only for builder runs with contract_status ok/missing/invalid/interrupted");
      }
      // MOA-474 §11.1: stage/diagnostic are additive/optional on ok/missing/invalid
      // (backward compatible with a pre-spec row that has neither), MANDATORY on
      // interrupted, and FORBIDDEN on cancelled — that row's shape is untouched by this
      // spec (§11.2).
      if (cancelled) {
        if ("stage" in p) fail("stage only for contract_status other than cancelled");
        if ("diagnostic" in p) fail("diagnostic only for contract_status other than cancelled");
      } else {
        const interrupted = p.contract_status === "interrupted";
        if (interrupted && !("stage" in p)) fail("stage required for contract_status interrupted");
        if ("stage" in p) {
          if (!oneOf(p.stage, STAGES)) fail("stage not allowed");
          if (interrupted && p.stage !== "runtime" && p.stage !== "worker") {
            fail("stage must be runtime or worker for contract_status interrupted");
          }
        }
        if (interrupted && !("diagnostic" in p)) fail("diagnostic required for contract_status interrupted");
        if ("diagnostic" in p) {
          const d = p.diagnostic;
          if (typeof d !== "string" || d.length === 0) fail("diagnostic must be a non-empty string");
          if ([...d].length > 300) fail("diagnostic exceeds 300 chars");
          if (/[\u0000-\u001f]/.test(d)) fail("diagnostic must be a single line");
        }
      }
      // MOA-470 §4.3: tail/reason travel together, and only on missing/invalid -- a
      // stricter rule than verdict/result above (which are role-conditional; these two
      // are not). Backward compatibility: an older row may carry NEITHER key on a
      // missing/invalid row, which this passes (neither key present is never rejected
      // by the "only one present" check below). Check order is deterministic and
      // FIXED: `tail` before `reason` in both branches below -- a row invalid on both
      // counts always reports the `tail` message first (pinned by the vitest case
      // above, "rejects tail/reason on an ok or cancelled row").
      const diagnosable = p.contract_status === "missing" || p.contract_status === "invalid";
      if (!diagnosable) {
        if ("tail" in p) fail("tail only for contract_status missing/invalid");
        if ("reason" in p) fail("reason only for contract_status missing/invalid");
        if ("log_context" in p) fail("log_context only for contract_status missing/invalid");
      } else {
        if ("tail" in p && !("reason" in p)) fail("reason required when tail is present");
        if ("reason" in p && !("tail" in p)) fail("tail required when reason is present");
        if ("tail" in p) {
          // NOT `optStr` (below): a genuinely silent child stream produces an EMPTY
          // `child_log_tail()` result (spec §4.3 has no minimum length), and `optStr`
          // rejects an empty string outright -- that would drop the exact row this
          // whole feature exists to surface.
          const t = p.tail;
          if (typeof t !== "string") fail("tail must be a string");
          // Final review 1f119d5d1420 F2: `[...t].length` counts Unicode CODE POINTS,
          // matching Python's `_bound` (scripts/jaxflow_run.py:473) -- plain `.length`
          // counts UTF-16 units, over-counting a non-BMP character 2x.
          if ([...t].length > LIMITS.childTail) fail(`tail exceeds ${LIMITS.childTail} chars`);
        }
        if ("reason" in p && !oneOf(p.reason, CHILD_LOG_REASONS)) fail("reason not allowed");
        // MOA-495 2.1: the Jev Score log slices, surfaced where `reason`/`tail` are --
        // optional (Jev failure keeps today's behaviour: nothing extra).
        if ("log_context" in p) {
          const lc = p.log_context;
          if (!Array.isArray(lc) || lc.length === 0 || lc.length > LIMITS.logContextMax) {
            fail(`log_context must be an array of 1-${LIMITS.logContextMax} strings`);
          }
          for (const chunk of lc) {
            if (typeof chunk !== "string" || chunk.length === 0) fail("log_context entries must be non-empty strings");
            if ([...chunk].length > LIMITS.logContextChunk) fail(`log_context entry exceeds ${LIMITS.logContextChunk} chars`);
            if (/[\u0000-\u001f]/.test(chunk)) fail("log_context entries must be a single line");
          }
        }
      }
      return;
    }
    case "question":
      // tmux_target is server-enriched at insert time — never accepted from the caller
      onlyKeys(p, ["tool_use_id", "questions"], "payload");
      reqStr(p, "tool_use_id", LIMITS.toolUseId, TOOL_USE_ID);
      validateQuestions(p.questions);
      return;
    case "question-resolved":
      onlyKeys(p, ["tool_use_id"], "payload");
      reqStr(p, "tool_use_id", LIMITS.toolUseId, TOOL_USE_ID);
      return;
    case "attention-needed":
      onlyKeys(p, ["reason", "excerpt"], "payload");
      reqStr(p, "reason", LIMITS.reason);
      optStr(p, "excerpt", LIMITS.attentionExcerpt);
      return;
    case "turn-stopped": {
      // The payload arrives in ONE of two shapes (spec §5). Tagged: the hook matched the
      // capsule tag and decided. Untagged: the hook decided nothing and shipped the text for
      // the ladder. `capsule_status` is optional at ingress for the first time — B6 closed
      // that window against the v0 hook, and this reopens it deliberately, because the route
      // now FILLS the field rather than trusting a producer to have done so.
      onlyKeys(p, ["capsule_status", "capsule_minutes", "excerpt", "message_tail", "capsule_rule",
        "capsule_attempts", "merge_ask", "merge_branch", "merge_target", "merge_head_sha"], "payload");
      const MERGE_REF = /^[A-Za-z0-9._/-]+$/;
      if (p.capsule_rule === "merge-question") {
        // The hook's merge-question shape (merge question contract section 3.8) and NOTHING else:
        // the question wins over any tag, Jev is never consulted, so no tail/minutes/excerpt ride along.
        if (source !== "deterministic") fail("capsule_rule merge-question requires source deterministic");
        if (p.capsule_status !== "needs_input") fail("capsule_rule merge-question requires capsule_status needs_input");
        if (p.capsule_attempts !== 0) fail("capsule_rule merge-question requires capsule_attempts 0");
        if (p.merge_ask !== 1) fail("capsule_rule merge-question requires merge_ask 1");
        for (const k of ["message_tail", "capsule_minutes", "excerpt"]) {
          if (k in p) fail(`${k} cannot accompany a merge-question payload`);
        }
        reqStr(p, "merge_branch", 200, MERGE_REF);
        reqStr(p, "merge_target", 200, MERGE_REF);
        if (!("merge_head_sha" in p) || p.merge_head_sha === undefined) fail("merge_head_sha required");
        if (p.merge_head_sha !== null) reqStr(p, "merge_head_sha", 40, /^[0-9a-f]{7,40}$/);
        return;
      }
      if ("capsule_attempts" in p) fail("payload: unknown key capsule_attempts");
      for (const k of ["merge_branch", "merge_target", "merge_head_sha"]) {
        if (k in p) fail(`${k} requires merge_ask 1 on a merge-question payload`);
      }
      if ("merge_ask" in p && p.merge_ask !== 0) fail("merge_ask must be 0 outside a merge-question payload");
      // ONE discriminator, not a pile of partial checks. The payload is either the tagged
      // shape or the untagged one; anything that mixes them is a producer bug, and a
      // half-checked mix is how a hook's verdict gets silently discarded by the ladder.
      const TAGGED_KEYS = ["capsule_rule", "capsule_status", "capsule_minutes", "excerpt"] as const;
      const tagged = TAGGED_KEYS.some((k) => k in p);
      if (tagged && "message_tail" in p) fail("message_tail cannot accompany a tagged payload");
      if (tagged) {
        // `tag` is the ONLY rule a producer may claim — rungs 2-7 are the route's to assign,
        // and a hook claiming one is either an old build or a forgery. Requiring it here is
        // what stops a tagged payload without the marker from falling through to the ladder
        // and having its verdict thrown away.
        if (!("capsule_rule" in p)) fail("a tagged payload requires capsule_rule tag");
        if (p.capsule_rule !== "tag") fail("capsule_rule not allowed from a producer");
        if (source !== "deterministic") fail("capsule_rule tag requires source deterministic");
        // A tag without a usable status is not a tag: the derivation would otherwise store
        // `capsule_status: undefined` on rung 1 and every reader would see a broken row.
        if (!oneOf(p.capsule_status, CAPSULE_STATUSES)) fail("capsule_rule tag requires a capsule_status");
        if (p.capsule_status === "unknown") fail("capsule_rule tag cannot be unknown");
        if (p.capsule_status === "waiting") {
          const m = p.capsule_minutes;
          if (!Number.isInteger(m) || (m as number) < 1 || (m as number) > 480) fail("capsule_minutes must be 1-480");
        } else if ("capsule_minutes" in p) {
          fail("capsule_minutes only with capsule_status waiting");
        }
        optStr(p, "excerpt", LIMITS.capsuleExcerpt);
      } else {
        optStr(p, "message_tail", LIMITS.messageTail);
      }
      return;
    }
    case "turn-started":
      onlyKeys(p, [], "payload");
      return;
    case "tool-used":
      // Spec §4.2 / Decision 2: the tool NAME only — never arguments, never a path.
      onlyKeys(p, ["tool"], "payload");
      reqStr(p, "tool", LIMITS.tool, NO_CONTROL_CHARS);
      return;
    case "subagent-started":
      onlyKeys(p, ["agent_id", "agent_type"], "payload");
      reqStr(p, "agent_id", LIMITS.agentId, NO_CONTROL_CHARS);
      reqStr(p, "agent_type", LIMITS.agentType, NO_CONTROL_CHARS);
      return;
    case "subagent-stopped":
      onlyKeys(p, ["agent_id"], "payload");
      reqStr(p, "agent_id", LIMITS.agentId, NO_CONTROL_CHARS);
      return;
    case "question-answered": {
      onlyKeys(p, ["kind", "question_event_id", "tool_use_id", "mutation_id", "ok", "error"], "payload");
      if (!oneOf(p.kind, ["structured", "freeform"])) fail("kind not allowed");
      posInt(p, "question_event_id");
      if (!("tool_use_id" in p)) fail("tool_use_id required");
      if (p.kind === "structured") {
        if (typeof p.tool_use_id !== "string" || !TOOL_USE_ID.test(p.tool_use_id) || p.tool_use_id.length > LIMITS.toolUseId) {
          fail("tool_use_id must be a valid string for kind structured");
        }
      } else if (p.tool_use_id !== null) {
        fail("tool_use_id must be null for kind freeform");
      }
      posInt(p, "mutation_id");
      if (typeof p.ok !== "boolean") fail("ok must be boolean");
      optStr(p, "error", LIMITS.error);
      return;
    }
    case "merge-approved": {
      // MOA-465: two OPTIONAL keys added for a PR-preset merge's PR metadata — a local-preset merge
      // omits them, unchanged from today.
      // MOA-510: the OPTIONAL checks object records how --checks was satisfied (reused|run|resumed).
      onlyKeys(p, ["phase", "branch", "sha", "target", "approved_by", "merge_sha", "pr_number", "pr_url", "checks"], "payload");
      // `phase` here is a human phase TITLE (free text, spec §5.1), not the TOKEN-shaped
      // `phase` of a run event. It carries a PATTERN, not just a bound, so this validator
      // enforces the same contract `cmd_merge` enforces before it commits (part-2 cold
      // review rounds 2-4): one trimmed line, no tab, newline or unpaired surrogate.
      // `LIMITS.summary` is 200 and `String.length` counts UTF-16 code units — the same
      // unit the producer measures.
      //
      // The two whitespace definitions are not identical and do not need to be: what
      // matters is that the PRODUCER is never the looser side, since only that direction
      // commits and then fails audit. Measured over every BMP code point (round 4): JS
      // `\s` and Python `str.strip()` differ on exactly six characters. Five (U+001C-001F,
      // U+0085) are Python-only, so `cmd_merge` refuses them and this validator never sees
      // them — harmless. One, U+FEFF, is JS-only, and that IS the dangerous direction, so
      // the producer rejects it explicitly rather than this side being relaxed.
      reqStr(p, "phase", LIMITS.summary, PHASE_TITLE_RE);
      reqStr(p, "branch", LIMITS.target);
      reqStr(p, "target", LIMITS.target);
      for (const key of ["sha", "merge_sha"]) {
        const v = p[key];
        if (!(typeof v === "string" && SHA40.test(v))) fail(`${key} must be a full 40-hex sha`);
      }
      if (p.approved_by !== "rafa") fail("approved_by must be rafa");
      if ("pr_number" in p) posInt(p, "pr_number");
      if ("pr_url" in p) reqStr(p, "pr_url", LIMITS.prUrl, GITHUB_PR_URL_RE);
      if ("checks" in p) {
        const c = p.checks;
        if (!isObj(c)) fail("checks must be an object");
        if (!oneOf(c.mode, ["reused", "run", "resumed"])) fail("checks.mode not allowed");
        if (c.mode === "reused") {
          onlyKeys(c, ["mode", "source_review_run_id", "head_sha"], "checks");
          reqStr(c, "source_review_run_id", 12, /^[0-9a-f]{12}$/);
          reqStr(c, "head_sha", 40, SHA40);
        } else {
          onlyKeys(c, ["mode"], "checks");
        }
      }
      return;
    }
    case "pr-opened": {
      // MOA-465 Ledger & card: repo/branch/base reuse existing bounds (LIMITS.repo/target) —
      // this is a new FIELD, not a new concept, so it gets no new LIMITS key beyond prUrl.
      onlyKeys(p, ["repo", "branch", "sha", "base", "pr_number", "pr_url", "kind", "snapshot_sha"], "payload");
      reqStr(p, "repo", LIMITS.repo);
      reqStr(p, "branch", LIMITS.target);
      reqStr(p, "sha", 40, SHA40);
      reqStr(p, "base", LIMITS.target);
      posInt(p, "pr_number");
      reqStr(p, "pr_url", LIMITS.prUrl, GITHUB_PR_URL_RE);
      if (!oneOf(p.kind, ["feature", "release"])) fail("kind not allowed");
      // Optional regardless of kind (this plan's own resolution, Global Constraints): a direct
      // `jaxflow pr open` call on a release/* branch has no snapshot to report; only `release`
      // (Part 2) ever supplies one.
      optStr(p, "snapshot_sha", 40, SHA40);
      return;
    }
    case "gc-removed": {
      onlyKeys(p, ["run_id", "branch", "worktree", "reason", "age_days"], "payload");
      reqStr(p, "run_id", 12, /^[0-9a-f]{12}$/);
      reqStr(p, "branch", LIMITS.target);
      reqStr(p, "worktree", LIMITS.reportPath);
      reqStr(p, "reason", LIMITS.reason);
      const age = p.age_days;
      if (typeof age !== "number" || !Number.isFinite(age) || age < 0) fail("age_days must be a non-negative number");
      return;
    }
    case "mission-status-updated": {
      // Spec §9 payload contract: the ONE schema, reusing §6a's name/status_line bounds exactly
      // (LIMITS.missionName/missionStatusLine) so this validator, the route body checks, and the
      // jaxflow CLI's own advisory checks can never disagree about a number.
      onlyKeys(p, ["missionId", "missionName", "missionStatusLine", "milestonesDone", "milestonesTotal"], "payload");
      posInt(p, "missionId");
      // Cold review F1: reuse the shared validator (same one the route and the CLI use) instead
      // of the bare length/pattern check, so a leading/trailing space or control character can
      // never sneak into a mission name through this path.
      if (!validMissionText(p.missionName, LIMITS.missionName)) fail(`missionName must be a valid mission text up to ${LIMITS.missionName} chars`);
      // missionStatusLine may be '' (spec §6 Decision 1 — a fresh mission's status_line before
      // the first `status` call) — the ONE mission field validated here that allows blank. Every
      // other value goes through the SAME validMissionText the route and the CLI use (cold review
      // F4), so a control character or an untrimmed value can never sneak through this path.
      const statusLine = p.missionStatusLine;
      if (statusLine !== "" && !validMissionText(statusLine, LIMITS.missionStatusLine)) {
        fail(`missionStatusLine must be '' or a valid mission text up to ${LIMITS.missionStatusLine} chars`);
      }
      const done = nonNegInt(p, "milestonesDone");
      const total = nonNegInt(p, "milestonesTotal");
      if (done > total) fail("milestonesDone must not exceed milestonesTotal");
      // Cold review F3: Task 4's own doc comment already claims this validator "consumes"
      // LIMITS.missionMilestonesMax (spec coverage table, top of this plan) — this is that check.
      if (total > LIMITS.missionMilestonesMax) fail(`milestonesTotal must not exceed ${LIMITS.missionMilestonesMax}`);
      return;
    }
    case "mission-finished": {
      onlyKeys(p, ["missionId", "missionName", "outcome"], "payload");
      posInt(p, "missionId");
      // Cold review F1: same shared validator as mission-status-updated above.
      if (!validMissionText(p.missionName, LIMITS.missionName)) fail(`missionName must be a valid mission text up to ${LIMITS.missionName} chars`);
      if (!oneOf(p.outcome, ["done", "cancelled"])) fail("outcome not allowed");
      return;
    }
  }
}

export function parseIngress(body: unknown): ParseResult {
  try {
    if (!isObj(body)) fail("body must be a JSON object");
    onlyKeys(body, ["project", "run_id", "role", "type", "source", "emitter", "payload", "pane", "tmux_incarnation", "harness_session"], "event");
    const type = body.type;
    if (!oneOf(type, EVENT_TYPES)) fail("type not allowed");
    const rule = EVENT_MATRIX[type];
    const emitter = body.emitter;
    if (!oneOf(emitter, EMITTERS)) fail("emitter not allowed");
    if (!rule.emitters.includes(emitter)) fail(`emitter ${emitter} cannot emit ${type}`);
    const source = body.source;
    if (!oneOf(source, SOURCES)) fail("source not allowed");
    if (type !== "turn-stopped" && source !== SOURCE_BY_EMITTER[emitter]) {
      fail(`source must be ${SOURCE_BY_EMITTER[emitter]} for emitter ${emitter}`);
    }
    const role = body.role;
    if (!oneOf(role, ROLES)) fail("role not allowed");
    if (!rule.roles.includes(role)) fail(`role ${role} not allowed for ${type}`);
    const project = reqStr(body, "project", LIMITS.project, PROJECT_LABEL);
    const run_id = optStr(body, "run_id", LIMITS.runId, TOKEN) ?? null;
    if (rule.runScoped && !run_id) fail("run_id required");
    if (!rule.runScoped && run_id) fail("run_id only for run events");
    const harness_session = optStr(body, "harness_session", LIMITS.harnessSession) ?? null;
    // pane/tmux_incarnation stay `undefined` (not `null`) when absent — matches
    // WorkflowEventInput's optional field shape; insertEvent (Task A1) normalizes
    // either to SQL NULL via `ev.pane ?? null`, so this only matters for callers
    // (and tests) inspecting the parsed event directly.
    const paneRaw = body.pane;
    let pane: string | undefined;
    if (role === "lead" || role === "adhoc") {
      // B6: pane is REQUIRED for hook emitters. `jaxflow_hook.py` bails out when $TMUX_PANE is
      // absent for Claude, so no Claude hook can legitimately reach here without one. adhoc is
      // included, not just lead (Finding 19 / spec Finding 8a) — adhoc is the DEFAULT role (any
      // tmux session not named `*-lead`), so exempting it would leave the common case still
      // accepting the v0 shape.
      //
      // Three carve-outs. `jaxos` — Jax OS's own rows (question-answered, and the staleness
      // alert when its pane is unknown) are written by insertEvent directly and never pass
      // through here; the events route rejects `emitter: "jaxos"` outright besides. `wrapper` is
      // the second: `merge-approved` is the only lead-role event the deterministic wrapper emits,
      // and `jaxflow merge` is a foreground command the tech lead may well run outside tmux.
      // `codex-*` is the third (MOA-469 §3): native Codex hooks fire without TMUX, so the pane is
      // optional — but the canonical session UUID is REQUIRED as the identity. A missing or
      // malformed identity is refused rather than falling back to cwd or a pane.
      if (emitter === "jaxos" || emitter === "wrapper") {
        if (paneRaw !== undefined) {
          if (typeof paneRaw !== "string" || !PANE_RE.test(paneRaw)) fail("pane malformed");
          pane = paneRaw;
        }
      } else if (isCodexEmitter(emitter)) {
        if (paneRaw !== undefined) {
          if (typeof paneRaw !== "string" || !PANE_RE.test(paneRaw)) fail("pane malformed");
          pane = paneRaw;
        }
        if (!harness_session || !UUID_RE.test(harness_session)) fail("codex event requires a canonical session uuid");
      } else {
        if (typeof paneRaw !== "string" || !PANE_RE.test(paneRaw)) fail("pane required");
        pane = paneRaw;
      }
    } else if (paneRaw !== undefined) {
      fail("pane only for role lead or adhoc");
    }
    const incarnationRaw = body.tmux_incarnation;
    let tmux_incarnation: string | undefined;
    if (incarnationRaw !== undefined) {
      if (!pane) fail("tmux_incarnation only permitted alongside pane");
      if (typeof incarnationRaw !== "string" || !TMUX_INCARNATION_RE.test(incarnationRaw)) fail("tmux_incarnation malformed");
      tmux_incarnation = incarnationRaw;
    }
    const payload = body.payload ?? {};
    if (!isObj(payload)) fail("payload must be an object");
    const encoded = safeStringify(payload);
    if (encoded === null) fail("payload not serializable");
    if (Buffer.byteLength(encoded) > LIMITS.payloadBytes) fail(`payload exceeds ${LIMITS.payloadBytes} bytes`);
    validatePayload(type, role, source, payload);
    return { ok: true, event: { run_id, project, role, type, source, emitter, payload, pane, tmux_incarnation, harness_session } };
  } catch (e) {
    if (e instanceof Invalid) return { ok: false, error: e.message };
    return { ok: false, error: "input rejected" }; // never surface an unexpected throw as a 5xx
  }
}

export type SessionParse =
  | { ok: true; kind: "current"; session: WorkflowSessionInput }
  | { ok: false; error: string };

export function parseSessionBody(body: unknown): SessionParse {
  try {
    if (!isObj(body)) fail("body must be a JSON object");
    onlyKeys(body, ["project", "session", "pane", "role", "tmux_incarnation"], "session");
    const project = reqStr(body, "project", LIMITS.project, PROJECT_LABEL);
    const session = reqStr(body, "session", LIMITS.tmuxTarget, TMUX_TARGET);
    const pane = reqStr(body, "pane", LIMITS.pane, PANE_RE);
    const role = body.role;
    if (!oneOf(role, ["lead", "adhoc"] as const)) fail("role not allowed");
    const tmux_incarnation = body.tmux_incarnation;
    if (typeof tmux_incarnation !== "string" || !TMUX_INCARNATION_RE.test(tmux_incarnation)) fail("tmux_incarnation required");
    return { ok: true, kind: "current", session: { project, session, pane, role, tmux_incarnation } };
  } catch (e) {
    if (e instanceof Invalid) return { ok: false, error: e.message };
    return { ok: false, error: "input rejected" }; // never surface an unexpected throw as a 5xx
  }
}

// ---- Outbound shape (spec §4.2 "Hermes/Telegram action" column) ----
// The one-liner Hermes relays verbatim; deterministic + tested so Hermes never
// has to compose (and never reads more than this). Option numbering is
// 1-based in array order — the Phase C answer route maps digits to it.

export type ForwardBody = {
  event_id: number;
  ts: string;
  type: EventType;
  project: string;
  run_id: string | null;
  role: Role;
  source: Source;
  emitter: Emitter;
  message: string;
  payload: Record<string, unknown>;
};

export function renderMessage(
  ev: Pick<WorkflowEventRow, "type" | "project" | "role" | "payload">,
  viewerUrl?: string,
): string {
  const p = ev.payload;
  const s = (k: string): string => (typeof p[k] === "string" ? (p[k] as string) : "");
  const viewer = viewerUrl ? `Viewer: ${viewerUrl}` : "Open the tmux viewer in Jax OS.";
  switch (ev.type) {
    case "run-started":
      return `Project ${ev.project} (phase ${s("phase")}): ${ev.role} run started on ${s("runtime")}.`;
    case "run-finished": {
      const contractStatus = s("contract_status");
      if (contractStatus === "cancelled") {
        return `Project ${ev.project} (phase ${s("phase")}): ${ev.role} run cancelled: ${s("summary")}.`;
      }
      // §12.1's shared <outcome> table, reused verbatim by §12.3: builder reads `result`,
      // reviewer reads `verdict` only when contract_status is ok — but the LITERAL fallback
      // ("no result"/"no verdict") applies whenever the field is simply absent, which covers
      // every non-ok reviewer row (missing/invalid/interrupted) and a legacy builder row with
      // no result key at all (§Backward compatibility) in one read, with no role-conditional
      // branching beyond which field to read.
      const outcome = ev.role === "reviewer"
        ? (s("contract_status") === "ok" && s("verdict") ? s("verdict") : "no verdict")
        : (s("result") || "no result");
      const stage = s("stage");
      const diagnostic = s("diagnostic");
      // stage/diagnostic are additive and always travel together (Part A's ingress) — absent
      // together on every legacy row and on the ok/missing/invalid happy path (§Builder
      // decision table Part 2 row 8 / §Reviewer decision table row 3).
      const stageDiag = stage || diagnostic ? ` · ${stage} — ${diagnostic}` : "";
      if (contractStatus === "interrupted") {
        return `Project ${ev.project} (phase ${s("phase")}): ${ev.role} run interrupted — ${outcome}${stageDiag}.`;
      }
      const reportFlag = contractStatus === "missing" || contractStatus === "invalid" ? ` [report ${contractStatus}]` : "";
      return `Project ${ev.project} (phase ${s("phase")}): ${ev.role} run finished — ${outcome}${stageDiag}, exit ${p.exit_code}${reportFlag}. Report at ${s("report_path")}.`;
    }
    case "question": {
      const qs = Array.isArray(p.questions) ? (p.questions as Obj[]) : [];
      const lines = qs.map((q, i) => {
        const opts = Array.isArray(q.options) ? (q.options as Obj[]) : [];
        const head = `Q${i + 1}: ${q.question}${q.multiSelect === true ? " (choose one or more)" : ""}`;
        const body = opts.map((o, j) => `  ${j + 1}) ${o.label}${o.description ? ` — ${o.description}` : ""}`);
        return [head, ...body].join("\n");
      });
      return `Project ${ev.project}: the tech lead asks —\n${lines.join("\n")}\nReply to this message once, one line per question: "1: 2" or "2: 1,3"; text is also allowed after the colon.`;
    }
    case "question-resolved":
      return `Project ${ev.project}: question ${s("tool_use_id")} resolved.`;
    case "attention-needed":
      return `Project ${ev.project}: the tech lead needs attention (${s("reason")}). ${viewer}`;
    case "turn-stopped":
      return `Project ${ev.project}: tech lead turn ended.${s("excerpt") ? ` Proposal: ${s("excerpt")}` : ""} ${viewer}`;
    case "turn-started":
      return `Project ${ev.project}: tech lead turn started.`;
    case "tool-used":
      return `Project ${ev.project}: tool ${s("tool")} used.`;
    case "subagent-started":
      return `Project ${ev.project}: subagent ${s("agent_type")} started.`;
    case "subagent-stopped":
      return `Project ${ev.project}: subagent stopped.`;
    case "question-answered":
      return `Project ${ev.project}: answer injected for question #${p.question_event_id} (${p.ok === true ? "ok" : `failed: ${s("error")}`}).`;
    case "merge-approved":
      return `Project ${ev.project} (phase ${s("phase")}): ${s("branch")} merged into ${s("target")} as ${s("merge_sha")}, approved by ${s("approved_by")}.`;
    case "pr-opened":
      return `Project ${ev.project}: PR #${p.pr_number} opened (${s("branch")} → ${s("base")}). ${s("pr_url")}`;
    case "gc-removed":
      return `Project ${ev.project}: worktree ${s("worktree")} removed (branch ${s("branch")}, ${s("reason")}).`;
    // temporary until MOA-488 defines the payload (jaxflow-mission spec §9)
    case "mission-status-updated":
      return `Project ${ev.project}: mission status updated.`;
    case "mission-finished":
      return `Project ${ev.project}: mission finished.`;
  }
}

const LOCAL_ONLY: Partial<Record<EventType, readonly string[]>> = {
  // Stored, never forwarded, never returned through a projection. (`message_tail` IS read by
  // the deferred-queue route, which is the single sanctioned reader — see projectPayload.)
  // `merge_head_sha` is audit evidence, never rendered or forwarded.
  "attention-needed": ["excerpt"],
  question: ["tool_use_id", "tmux_target"],
  "turn-stopped": ["message_tail", "merge_head_sha"],
};

/**
 * The single filter between a stored payload and any consumer outside the DB layer.
 * Every projection MUST call this — a new consumer that skips it is the one way
 * `message_tail` can escape, and the tests in both suites exist to catch that.
 *
 * The ONE sanctioned exception is the deferred-queue route, which selects `message_tail`
 * explicitly because the classifier needs the text. It does not go through this helper, and
 * that is why it is a separate route rather than a filter on an existing one.
 */
export function projectPayload(type: EventType, payload: Record<string, unknown>): Record<string, unknown> {
  const drop = LOCAL_ONLY[type];
  if (!drop) return payload;
  return Object.fromEntries(Object.entries(payload).filter(([k]) => !drop.includes(k)));
}

export function toForwardBody(row: WorkflowEventRow, viewerUrl?: string): ForwardBody {
  const payload = projectPayload(row.type, row.payload);
  return {
    event_id: row.id,
    ts: row.ts,
    type: row.type,
    project: row.project,
    run_id: row.run_id,
    role: row.role,
    source: row.source,
    emitter: row.emitter,
    message: row.type === "question" || row.type === "turn-stopped" || row.type === "attention-needed"
      ? `Workflow event #${row.id}\n${renderMessage(row, viewerUrl)}`
      : renderMessage(row, viewerUrl),
    payload,
  };
}

export type AnswerParse = { ok: true; answer: AnswerRequest } | { ok: false; error: string };

export function parseAnswerBody(body: unknown): AnswerParse {
  try {
    if (!isObj(body)) fail("body must be a JSON object");
    onlyKeys(body, ["question_event_id", "event_id", "reply"], "answer");
    const hasQ = "question_event_id" in body;
    const hasE = "event_id" in body;
    if (!hasQ && !hasE) fail("event id must be a positive integer");
    if (hasQ && hasE && body.event_id !== body.question_event_id) fail("event_id and question_event_id disagree");
    const rawId = hasE ? body.event_id : body.question_event_id;
    if (typeof rawId !== "number" || !Number.isInteger(rawId) || rawId <= 0) fail("event id must be a positive integer");
    const question_event_id = rawId;
    const reply = body.reply;
    if (typeof reply !== "string" || reply.length === 0) fail("reply must be a non-empty string");
    if (reply.length > LIMITS.answerReply) fail(`reply exceeds ${LIMITS.answerReply} chars`);
    if (CONTROL_CHARS_RE.test(reply)) fail("reply contains control characters"); // shared guard, Finding 5 (now also rejects tab + DEL)
    const lines = reply.split(/\r\n?|\n/).filter((line) => line.length > 0);
    if (lines.length < 1 || lines.length > LIMITS.questions) fail("reply must have 1-10 non-empty lines");
    return { ok: true, answer: { question_event_id, reply } };
  } catch (e) {
    if (e instanceof Invalid) return { ok: false, error: e.message };
    return { ok: false, error: "input rejected" };
  }
}
