import { describe, expect, it, vi } from "vitest";
const FIXTURE = vi.hoisted(() => {
  const { mkdtempSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const home = mkdtempSync(join(tmpdir(), "jaxos-workflows-home-"));
  const reposRoot = mkdtempSync(join(tmpdir(), "jaxos-workflows-repos-"));
  // Pin REPOS_ROOT (fixed when ../collectors/projects is first imported) to a test-owned temp
  // dir: left ambient it is `$HOME/repos` — nonexistent on a fresh runner (the targetRel
  // test) and the owner's real repos dir (the lastStep test's mkdirSync) otherwise.
  writeFileSync(join(home, "settings.json"), JSON.stringify({ reposRoot }), "utf8");
  process.env.JAXOS_HOME = home;
  return { reposRoot };
});

// MOA-498: insertEvent's new delivery step reads integrations.webhook. Every existing
// forward-policy test in this file predates that gate and expects `pending` with no settings
// arrangement — default the mock to "on" (delegating to the real reader for every other
// field, so import-time consumers like reposRoot()/REPOS_ROOT keep their real values) so this
// build's own regression bar (every existing test keeps passing unmodified) holds; the few
// tests THIS build adds for the off-path override the mock per-test via
// vi.mocked(readGeneralSettings).mockReturnValueOnce(...).
// MOA-502 Decision 1: same pattern for integrations.classifier — every existing test that
// inserts a turn-stopped with a text tail predates derive.ts's new rung-7 gate and expects
// the row queued `deferred`; the gate's own off/malformed paths are covered in derive.test.ts.
vi.mock("../settings", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../settings")>();
  return {
    ...actual,
    readGeneralSettings: vi.fn((...args: Parameters<typeof actual.readGeneralSettings>) => {
      const r = actual.readGeneralSettings(...args);
      return r.ok ? { ...r, data: { ...r.data, integrations: { ...r.data.integrations, webhook: true, classifier: true } } } : r;
    }),
  };
});

import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { openDb } from "./index";
import { REPOS_ROOT } from "../collectors/projects";
import { readGeneralSettings } from "../settings";
import { deriveCapsule } from "./derive";
import { insertMutation } from "./mutations";
import {
  abandonInjectingAnswers, claimAnswer, claimFreeformAnswer, classifyDeferred, failAnswerFinalization, FINALIZATION_ERROR, finishAnswer,
  getAfkEnabled, getForwardTypes, hasPendingQuestion, insertEvent, listPending, markDelivered, RECOVERY_ERROR, RUN_ALREADY_FINISHED, setAfk, setForwardTypes, upsertSession,
  evaluateStaleness, maybeInsertStalenessAlert,
  getProjectPrefs, setProjectPrefs, getArchiveAfterDays, setArchiveAfterDays,
  pruneState, heartbeatFromRows,
  getHubMembership, getProjectPanes, getProjectTimeline, getHubData, lastRunFor,
  getLoopSummary, getWorktreeLedgerRows,
  hasOpenTurn, isWorking, countFreshEvents, getCodexSessions,
  runtimeFromEmitter,
  type StalenessCandidate,
  type LiveSnapshot,
} from "./workflows";
import { DEFAULT_FORWARD_TYPES, FORWARDABLE_EVENT_TYPES, LIMITS, type WorkflowEventInput, type WorkflowEventRow } from "../../lib/workflow";
import { buildMission } from "../../lib/mission";
import type { Project } from "../collectors/projects";
import { parseIngress } from "../collectors/workflow-events";
import type { CodexSnapshot, CodexThread } from "../collectors/workflow-codex";

const NOW = new Date("2026-08-18T12:00:00.000Z");
const INCARNATION = "234790:1787586213";
const started: WorkflowEventInput = {
  run_id: "r1", project: "p1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
  payload: { phase: "B", runtime: "codex" },
};
const finished: WorkflowEventInput = { ...started, type: "run-finished", payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/r.md", summary: "ok", head_sha: "b".repeat(40) } };
// pane + tmux_incarnation on every claimable question — a paneless question is forced terminal
// 'local' plus a workflow-question-expired audit (§3.3); claimAnswer now reads these columns
// off the row (D1), not payload.tmux_target.
const question = (tool: string, pane = "%1"): WorkflowEventInput => ({
  run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic", emitter: "claude-pretool",
  payload: { tool_use_id: tool, questions: [{ question: "q?", options: [{ label: "a" }] }] },
  pane, tmux_incarnation: INCARNATION,
});
const resolved = (tool: string): WorkflowEventInput => ({
  run_id: null, project: "p1", role: "lead", type: "question-resolved", source: "deterministic", emitter: "claude-posttool",
  payload: { tool_use_id: tool },
});
const BATCH_REPLY = "3: ship without deploy\n1: 2\n2: 3,1";
const batchQuestion = (tool: string, pane = "%1"): WorkflowEventInput => ({
  run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic", emitter: "claude-pretool",
  payload: {
    tool_use_id: tool,
    questions: [
      { question: "Which environment?", options: [{ label: "Development" }, { label: "Staging" }], multiSelect: false },
      { question: "Which checks?", options: [{ label: "Tests" }, { label: "Build" }, { label: "Lint" }], multiSelect: true },
      { question: "Any final note?", options: [{ label: "Continue" }, { label: "Stop" }], multiSelect: false },
    ],
  },
  pane, tmux_incarnation: INCARNATION,
});

// Fallout helper (MOA-487 §5 Decision 3, step 4): every "AFK on" fixture in this file predates
// the per-type gate and expects a forward-policy event to reach `pending` on AFK alone. Enabling
// every forwardable type keeps each such site exercising what it always exercised; the gate's
// OWN partial/default behavior is covered by the dedicated describe block added in Step 3 below,
// via raw setAfk/setForwardTypes.
function setAfkOn(db: ReturnType<typeof openDb>): void {
  setAfk(db, true);
  setForwardTypes(db, [...FORWARDABLE_EVENT_TYPES]);
}

describe("insertEvent", () => {
  it("settles a deferred row and refuses to touch it again", () => {
    const db = openDb(":memory:");
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "ambiguous prose" } }, NOW);
    expect(classifyDeferred(db, row.id, { status: "done" })).toBe(true);
    const after = getProjectTimeline(db, "p1")[0];
    expect(after.payload).toMatchObject({ capsule_status: "done", capsule_rule: "classified" });
    expect(classifyDeferred(db, row.id, { status: "blocked" })).toBe(false);
    db.close();
  });

  it("classifyDeferred writes the status and never a merge_ask key", () => {
    const db = openDb(":memory:");
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "posso mergear feat/x em main?" } }, NOW);
    expect(classifyDeferred(db, row.id, { status: "needs_input" })).toBe(true);
    const payload = getProjectTimeline(db, "p1")[0].payload;
    expect(payload).toMatchObject({ capsule_status: "needs_input", capsule_rule: "classified" });
    expect(payload).not.toHaveProperty("merge_ask");
    db.close();
  });

  it("A13: classifyDeferred leaves a merge-question row untouched (not deferred, hook merge_ask 1 survives)", () => {
    const db = openDb(":memory:");
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "deterministic", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { capsule_status: "needs_input", capsule_rule: "merge-question", capsule_attempts: 0, merge_ask: 1,
        merge_branch: "feat/x", merge_target: "main", merge_head_sha: null } }, NOW);
    expect(classifyDeferred(db, row.id, { status: "done" })).toBe(false);
    expect(classifyDeferred(db, row.id, { failed: true })).toBe(false);
    expect(getProjectTimeline(db, "p1")[0].payload).toMatchObject({
      capsule_status: "needs_input", capsule_rule: "merge-question", merge_ask: 1, merge_branch: "feat/x" });
    db.close();
  });

  it("counts failures and abandons at the fifth, in one statement", () => {
    const db = openDb(":memory:");
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "ambiguous prose" } }, NOW);
    for (let i = 1; i <= 4; i++) {
      expect(classifyDeferred(db, row.id, { failed: true })).toBe(true);
      expect(getProjectTimeline(db, "p1")[0].payload).toMatchObject({ capsule_rule: "deferred", capsule_attempts: i });
    }
    expect(classifyDeferred(db, row.id, { failed: true })).toBe(true);
    expect(getProjectTimeline(db, "p1")[0].payload).toMatchObject({ capsule_status: "unknown", capsule_rule: "abandoned", capsule_attempts: 5 });
    expect(classifyDeferred(db, row.id, { failed: true })).toBe(false);
    db.close();
  });

  it("settles a deferred row as indeterminate in one call, leaves capsule_attempts untouched, and is a no-op after", () => {
    const db = openDb(":memory:");
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "ambiguous prose" } }, NOW);
    expect(classifyDeferred(db, row.id, { indeterminate: true })).toBe(true);
    const after = getProjectTimeline(db, "p1")[0];
    expect(after.payload).toMatchObject({ capsule_status: "unknown", capsule_rule: "abandoned", capsule_attempts: 0 });
    expect(classifyDeferred(db, row.id, { indeterminate: true })).toBe(false);
    db.close();
  });

  it("refuses indeterminate on a row that is not deferred", () => {
    const db = openDb(":memory:");
    const other = insertEvent(db, started, NOW);
    expect(classifyDeferred(db, other.id, { indeterminate: true })).toBe(false);
    db.close();
  });

  it("refuses a row that is not deferred and an id that is not a turn-stopped", () => {
    const db = openDb(":memory:");
    const other = insertEvent(db, started, NOW);
    expect(classifyDeferred(db, other.id, { status: "done" })).toBe(false);
    expect(classifyDeferred(db, 9999, { status: "done" })).toBe(false);
    db.close();
  });

  it("two concurrent settles: exactly one wins", () => {
    const db = openDb(":memory:");
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "ambiguous prose" } }, NOW);
    const results = [classifyDeferred(db, row.id, { status: "done" }), classifyDeferred(db, row.id, { status: "blocked" })];
    expect(results.filter(Boolean)).toHaveLength(1);
    db.close();
  });

  it("never returns message_tail through the project timeline", () => {
    const db = openDb(":memory:");
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION, payload: { message_tail: "secret prose" } }, NOW);
    const timeline = getProjectTimeline(db, "p1");
    expect(JSON.stringify(timeline)).not.toContain("secret prose");
    expect(timeline[0].payload).toMatchObject({ capsule_status: "unknown", capsule_rule: "deferred" });
    db.close();
  });
  it("derives the capsule fields on insert and leaves other types untouched", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "prose" } }, NOW);
    expect(stop.payload).toMatchObject({ capsule_status: "unknown", capsule_rule: "deferred", capsule_attempts: 0 });
    expect(stop.source).toBe("behavioral");
    expect(insertEvent(db, started, NOW).payload).toEqual({ phase: "B", runtime: "codex" });
    db.close();
  });
  it("stores and returns harness_session, defaulting to null", () => {
    const db = openDb(":memory:");
    const a = insertEvent(db, { ...started, harness_session: "sess-abc" }, NOW);
    expect(a.harness_session).toBe("sess-abc");
    expect(insertEvent(db, started, NOW).harness_session).toBeNull();
    db.close();
  });
  it("stores server ts, decides local vs pending from the matrix, returns the parsed row", () => {
    const db = openDb(":memory:");
    setAfkOn(db); // AFK gate (Task B4, §4.5 step 3) sits between matrix-local and pending —
    // this test is about matrix routing, not AFK, so AFK is on throughout.
    const a = insertEvent(db, started, NOW);
    expect(a).toMatchObject({ id: 1, ts: "2026-08-18T12:00:00.000Z", delivery: "local", forwarded_at: null, payload: { phase: "B", runtime: "codex" } });
    const b = insertEvent(db, finished, NOW);
    expect(b.delivery).toBe("pending");
    expect(b.id).toBe(2);
    db.close();
  });
  // NOTE (plan-vs-reality, Task B4): the three tests previously here ("suppresses
  // attention-needed while a question is pending for the same project only", "treats a claimed
  // question as no longer pending", "a later turn-stopped for the same project clears pending")
  // exercised the OLD `hasPendingQuestion(db, project)` signature and PROJECT-scoped suppression.
  // Task B4 re-scopes both to (pane, tmux_incarnation) (Findings 47/48/52) — the old signature no
  // longer exists (a 1-arg call is a TS error) and the old "same project" framing is now actively
  // wrong (a different pane in the same project must NOT cross-talk). Superseded, not just
  // updated, by the pane-scoped `describe("hasPendingQuestion ...")` and
  // `describe("insertEvent's AFK gate (§4.5 step 3)")` blocks below, which cover the same
  // scenarios (suppression, claimed-clears-pending, turn-stopped-clears-pending) with correct
  // identity semantics — deleted here rather than kept as stale duplicate coverage.
});

describe("insertEvent — §3.3 pane-less question is terminal, not pending", () => {
  it("forces delivery to local and audits workflow-question-expired when pane is null", () => {
    const db = openDb(":memory:");
    const q: WorkflowEventInput = {
      run_id: null, project: "Acme.AI", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool", payload: { tool_use_id: "tu_1", questions: [{ question: "q?", options: [{ label: "a" }] }] },
      // no pane
    };
    const row = insertEvent(db, q);
    expect(row.delivery).toBe("local");
    expect(listPending(db, 10).find((r) => r.id === row.id)).toBeUndefined();
    const audit = db.prepare(
      "SELECT payload FROM mutations WHERE kind = 'workflow-question-expired'",
    ).get() as { payload: string };
    expect(JSON.parse(audit.payload).question_event_id).toBe(row.id);
  });

  it("a question WITH a pane is unaffected — ordinary pending delivery", () => {
    const db = openDb(":memory:");
    setAfkOn(db); // NOTE (plan-vs-reality, Task B4): this A1-committed test asserted
    // "pending" before the AFK gate (step 3) existed; AFK now sits between the pane-less-question
    // check and the matrix-local check, so a forward-policy question needs AFK on to reach pending.
    const q: WorkflowEventInput = {
      run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool", payload: { tool_use_id: "tu_1", questions: [{ question: "q?", options: [{ label: "a" }] }] },
      pane: "%7", tmux_incarnation: "111:222",
    };
    const row = insertEvent(db, q);
    expect(row.delivery).toBe("pending");
    expect(row.pane).toBe("%7");
    expect(row.tmux_incarnation).toBe("111:222");
    expect(row.payload).not.toHaveProperty("tmux_target"); // enrichment deleted (§4.6.1 forced by A′'s schema)
    // Finding 21 — the SHARED COLS projection must fetch pane/tmux_incarnation too, not just
    // insertEvent's own return value (which builds its object via a spread, independent of COLS).
    const pending = listPending(db, 10).find((r) => r.id === row.id);
    expect(pending?.pane).toBe("%7");
    expect(pending?.tmux_incarnation).toBe("111:222");
  });
});

describe("insertEvent — integrations.webhook gate (MOA-498 D3)", () => {
  it("marks a forward-policy event local, never pending, when the integration is off", () => {
    vi.mocked(readGeneralSettings).mockReturnValueOnce({ ok: true, data: { integrations: { webhook: false } } } as ReturnType<typeof readGeneralSettings>);
    const db = openDb(":memory:");
    setAfkOn(db);
    const row = insertEvent(db, finished, NOW);
    expect(row.delivery).toBe("local");
    db.close();
  });
  it("marks it pending when the integration is on (regression: today's default mock)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const row = insertEvent(db, finished, NOW);
    expect(row.delivery).toBe("pending");
    db.close();
  });
  it("treats an unreadable settings file the same as off (fail closed)", () => {
    vi.mocked(readGeneralSettings).mockReturnValueOnce({ ok: false, error: "settings-unreadable" } as ReturnType<typeof readGeneralSettings>);
    const db = openDb(":memory:");
    setAfkOn(db);
    const row = insertEvent(db, finished, NOW);
    expect(row.delivery).toBe("local");
    db.close();
  });
});

describe("listPending + markDelivered", () => {
  it("lists only pending rows ordered by id and honours limit", () => {
    const db = openDb(":memory:");
    setAfkOn(db); // NOTE (plan-vs-reality, Task B4): run-finished is forward-policy — needs
    // AFK on to reach 'pending' now that step 3's AFK gate exists; unrelated to what this test covers.
    insertEvent(db, started, NOW); // local
    const f1 = insertEvent(db, finished, NOW);
    const f2 = insertEvent(db, { ...finished, run_id: "r2" }, NOW);
    expect(listPending(db, 10).map((r) => r.id)).toEqual([f1.id, f2.id]);
    expect(listPending(db, 1).map((r) => r.id)).toEqual([f1.id]);
    db.close();
  });
  it("acks only pending ids, sets forwarded_at, ignores local/unknown ids, returns count", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const s = insertEvent(db, started, NOW);
    const f1 = insertEvent(db, finished, NOW);
    const f2 = insertEvent(db, { ...finished, run_id: "r2" }, NOW);
    expect(markDelivered(db, [s.id, f1.id, 999], NOW)).toBe(1);
    expect(markDelivered(db, [f1.id], NOW)).toBe(0); // already delivered
    expect(markDelivered(db, [], NOW)).toBe(0);
    const row = db.prepare("SELECT delivery, forwarded_at FROM workflow_events WHERE id = ?").get(f1.id);
    expect(row).toEqual({ delivery: "delivered", forwarded_at: "2026-08-18T12:00:00.000Z" });
    expect(db.prepare("SELECT delivery, forwarded_at FROM workflow_events WHERE id = ?").get(s.id)).toEqual({ delivery: "local", forwarded_at: null });
    expect(listPending(db, 10).map((r) => r.id)).toEqual([f2.id]);
    db.close();
  });
});

describe("upsertSession", () => {
  it("upserts on (pane, tmux_incarnation), not on project", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%7", role: "lead", tmux_incarnation: "111:222" });
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%7", role: "lead", tmux_incarnation: "111:222" }); // re-fire, same identity
    const rows = db.prepare("SELECT COUNT(*) AS n FROM workflow_sessions").get();
    expect(rows).toEqual({ n: 1 });
    // a DIFFERENT incarnation on the same pane is a NEW row, not an overwrite (Finding 3)
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%7", role: "lead", tmux_incarnation: "333:444" });
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_sessions").get()).toEqual({ n: 2 });
    db.close();
  });
});

describe("claimAnswer", () => {
  it("derives correlation, pane+incarnation, shapes, and ordered answers from the stored event", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, batchQuestion("t1"), NOW);
    const first = claimAnswer(db, { question_event_id: q.id, reply: BATCH_REPLY }, NOW);
    expect(first).toMatchObject({ ok: true, claim: {
      project: "p1", pane: "%1", tmux_incarnation: INCARNATION, tool_use_id: "t1",
      question_shapes: [
        { multiSelect: false, option_count: 2 },
        { multiSelect: true, option_count: 3 },
        { multiSelect: false, option_count: 2 },
      ],
      answers: [
        { question_number: 1, kind: "options", values: [2] },
        { question_number: 2, kind: "options", values: [1, 3] },
        { question_number: 3, kind: "text", value: "ship without deploy" },
      ],
    }});
    if (first.ok) {
      expect(first.claim).not.toHaveProperty("target");
      expect(first.claim).not.toHaveProperty("question_ts");
    }
    expect(claimAnswer(db, { question_event_id: q.id, reply: BATCH_REPLY }, NOW)).toEqual({
      ok: false, reason: "not-pending",
    });
    expect(db.prepare("SELECT COUNT(*) AS c FROM mutations").get()).toEqual({ c: 1 });
    const mutation = db.prepare("SELECT kind, ok, error, payload FROM mutations").get() as {
      kind: string; ok: number | null; error: string | null; payload: string;
    };
    const payload = JSON.parse(mutation.payload);
    expect(payload).toMatchObject({ project: "p1", target: "%1", question_event_id: q.id,
      tool_use_id: "t1", answer_count: 3, outcome: "pending" });
    expect(JSON.stringify(payload)).not.toContain("ship without deploy");
    db.close();
  });

  it("rejects malformed replies and missing events before any mutation", () => {
    const db = openDb(":memory:");
    expect(claimAnswer(db, { question_event_id: 999, reply: "1: 1" }, NOW)).toEqual({
      ok: false, reason: "not-pending",
    });
    const malformed = insertEvent(db, question("t3"), NOW);
    expect(claimAnswer(db, { question_event_id: malformed.id, reply: "1: 0" }, NOW)).toEqual({
      ok: false, reason: "invalid-answer",
    });
    expect(db.prepare("SELECT COUNT(*) AS c FROM mutations").get()).toEqual({ c: 0 });
    db.close();
  });

  it("rejects a question after a later same-pane turn-stopped with no mutation", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, question("t-stop-claim"), NOW);
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "deterministic", emitter: "claude-stop", pane: "%9", tmux_incarnation: INCARNATION, payload: {},
    }, NOW);
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "deterministic", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION, payload: {},
    }, NOW);
    expect(claimAnswer(db, { question_event_id: q.id, reply: "1: 1" }, NOW)).toEqual({
      ok: false, reason: "not-pending",
    });
    expect(db.prepare("SELECT COUNT(*) AS c FROM mutations").get()).toEqual({ c: 0 });
    db.close();
  });

  it("allows only one injecting claim per pane", () => {
    const db = openDb(":memory:");
    const a = insertEvent(db, question("t-inj-a"), NOW);
    const b = insertEvent(db, question("t-inj-b"), NOW);
    const other = insertEvent(db, question("t-inj-p2", "%2"), NOW);
    const first = claimAnswer(db, { question_event_id: a.id, reply: "1: 1" }, NOW);
    expect(first.ok).toBe(true);
    expect(claimAnswer(db, { question_event_id: b.id, reply: "1: 1" }, NOW)).toEqual({
      ok: false, reason: "not-pending",
    });
    expect(db.prepare("SELECT COUNT(*) AS c FROM mutations").get()).toEqual({ c: 1 });
    const cross = claimAnswer(db, { question_event_id: other.id, reply: "1: 1" }, NOW);
    expect(cross.ok).toBe(true);
    if (!first.ok) return;
    finishAnswer(db, first.claim, true, null, NOW);
    const after = claimAnswer(db, { question_event_id: b.id, reply: "1: 1" }, NOW);
    expect(after.ok).toBe(true);
    db.close();
  });

  it("a pre-existing claim returns not-pending with no orphan mutation", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, question("t-race"), NOW);
    const priorMutation = insertMutation(db, {
      ts: NOW.toISOString(), kind: "workflow-answer", project: "p1", target: "%1",
      question_event_id: q.id, tool_use_id: "t-race", answer_count: 1, outcome: "failed",
    });
    db.prepare(`INSERT INTO workflow_answer_claims
      (tool_use_id, source_event_id, kind, mutation_id, status, claimed_at)
      VALUES ('t-race', ?, 'structured', ?, 'failed', ?)`).run(q.id, priorMutation, NOW.toISOString());
    expect(claimAnswer(db, { question_event_id: q.id, reply: "1: 1" }, NOW)).toEqual({
      ok: false, reason: "not-pending",
    });
    expect(db.prepare("SELECT COUNT(*) AS c FROM mutations").get()).toEqual({ c: 1 });
    db.close();
  });
});

describe("claimAnswer — reads pane/tmux_incarnation off the question row, leadTarget deleted (D1)", () => {
  it("refuses a pane-less question even when payload still carries a legacy tmux_target", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool",
      payload: {
        tool_use_id: "tu_1",
        questions: [{ question: "ship?", options: [{ label: "yes" }] }],
        tmux_target: "jax-p1-lead:1.1",
      },
    });
    expect(claimAnswer(db, { question_event_id: q.id, reply: "1: 1" })).toEqual({ ok: false, reason: "invalid-target" });
    db.close();
  });

  it("refuses a NULL tmux_incarnation even when pane is set and payload carries a legacy tmux_target", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool", pane: "%1",
      payload: {
        tool_use_id: "tu_1",
        questions: [{ question: "ship?", options: [{ label: "yes" }] }],
        tmux_target: "jax-p1-lead:1.1",
      },
    });
    expect(claimAnswer(db, { question_event_id: q.id, reply: "1: 1" })).toEqual({ ok: false, reason: "invalid-target" });
    db.close();
  });

  it("two panes in the SAME project never block each other's claim (Finding 43 — laterStop/injecting/resolved are pane-scoped)", () => {
    const db = openDb(":memory:");
    const qA = insertEvent(db, question("tu_a"), NOW);
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane: "%2", tmux_incarnation: INCARNATION, payload: { capsule_status: "done" },
    }, NOW);
    const claimedA = claimAnswer(db, { question_event_id: qA.id, reply: "1: 1" }, NOW);
    expect(claimedA.ok).toBe(true);
    if (!claimedA.ok) return;
    finishAnswer(db, claimedA.claim, true, null, NOW);

    const qB = insertEvent(db, question("tu_b", "%2"), NOW);
    const claimedB = claimAnswer(db, { question_event_id: qB.id, reply: "1: 1" }, NOW);
    expect(claimedB.ok).toBe(true);
    const qC = insertEvent(db, question("tu_c"), NOW);
    expect(claimAnswer(db, { question_event_id: qC.id, reply: "1: 1" }, NOW).ok).toBe(true);
    db.close();
  });

  it("a question-resolved on a different pane with the same tool_use_id does not block this pane's claim", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, question("tu_shared"), NOW);
    insertEvent(db, { ...resolved("tu_shared"), pane: "%9", tmux_incarnation: INCARNATION }, NOW);
    expect(claimAnswer(db, { question_event_id: q.id, reply: "1: 1" }, NOW).ok).toBe(true);
    db.close();
  });
});

function idleTurnStopped(db: ReturnType<typeof openDb>, overrides: Partial<WorkflowEventInput> = {}) {
  return insertEvent(db, {
    run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
    emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION, harness_session: "sess-idle",
    payload: { capsule_status: "done" }, ...overrides,
  });
}

describe("claimFreeformAnswer (D2, 7 guards)", () => {
  it("claims a plain reply on an idle pane, tool_use_id null, kind freeform", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    const result = claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.claim).toMatchObject({
        kind: "freeform", tool_use_id: null, text: "ship it", pane: "%1", tmux_incarnation: INCARNATION, role: "lead",
      });
    }
    db.close();
  });

  it("guard 1: a non-turn-stopped source is not-answerable", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, question("tu_1"));
    expect(claimFreeformAnswer(db, { question_event_id: q.id, reply: "ship it" })).toEqual({
      ok: false, reason: "not-answerable",
    });
    db.close();
  });

  it("a pane-less (native) stopped event with a harness_session now succeeds — pane is no longer required (D3)", () => {
    const db = openDb(":memory:");
    const ts = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral",
      emitter: "codex-stop", harness_session: "0191f0aa-dddd-7000-8000-00000000000d",
      pane: null, tmux_incarnation: null, payload: { message_tail: "prose" },
    });
    const result = claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" });
    expect(result.ok).toBe(true);
    if (result.ok) {
      expect(result.claim).toMatchObject({
        kind: "freeform", pane: null, tmux_incarnation: null, emitter: "codex-stop",
        harnessSession: "0191f0aa-dddd-7000-8000-00000000000d", text: "ship it",
      });
    }
    db.close();
  });

  it("a row with no harness_session at all is not-answerable (D3 error handling §5)", () => {
    const db = openDb(":memory:");
    const ts = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral",
      emitter: "codex-stop", pane: null, tmux_incarnation: null, payload: { message_tail: "prose" },
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" })).toEqual({
      ok: false, reason: "not-answerable",
    });
    db.close();
  });

  it("guard 2: a second freeform claim against the same source event is not-pending (UNIQUE source_event_id)", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    claimFreeformAnswer(db, { question_event_id: ts.id, reply: "first" });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "second" })).toEqual({
      ok: false, reason: "not-pending",
    });
    db.close();
  });

  it("guard 3: refused when a later turn-started exists for this pane (spec §4.6.3 idle gate)", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
      emitter: "claude-userprompt", pane: "%1", tmux_incarnation: INCARNATION, harness_session: "sess-idle", payload: {},
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" })).toEqual({
      ok: false, reason: "not-idle",
    });
    db.close();
  });

  it("guard 3 pins turn-started only: a later attention-needed on the same pane does not block", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "attention-needed", source: "behavioral",
      emitter: "claude-notification", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { reason: "agent_needs_input" },
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" }).ok).toBe(true);
    db.close();
  });

  it("guard 3 is harness_session-scoped: a later turn-started for a DIFFERENT session does not block (D3 — pane_key scoping is gone for freeform)", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db); // harness_session: "sess-idle"
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
      emitter: "claude-userprompt", pane: "%9", tmux_incarnation: INCARNATION, harness_session: "sess-other", payload: {},
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" }).ok).toBe(true);
    db.close();
  });

  it("guard 3 refuses on a later turn-started for the SAME harness_session, pane-less or not", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db); // harness_session: "sess-idle"
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
      emitter: "claude-userprompt", pane: null, tmux_incarnation: null, harness_session: "sess-idle", payload: {},
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" })).toEqual({
      ok: false, reason: "not-idle",
    });
    db.close();
  });

  it("guard 4: an injecting claim on the same harness_session blocks; a different session does not", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic", emitter: "claude-pretool",
      payload: { tool_use_id: "tu_inj", questions: [{ question: "q?", options: [{ label: "a" }] }] },
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: "sess-idle",
    });
    expect(claimAnswer(db, { question_event_id: q.id, reply: "1: 1" }).ok).toBe(true);
    const tsSame = idleTurnStopped(db);
    expect(claimFreeformAnswer(db, { question_event_id: tsSame.id, reply: "ship it" })).toEqual({
      ok: false, reason: "not-pending",
    });
    const tsOther = idleTurnStopped(db, { pane: "%2", harness_session: "sess-other" });
    expect(claimFreeformAnswer(db, { question_event_id: tsOther.id, reply: "ship it" }).ok).toBe(true);
    db.close();
  });

  it("guard 5: rejects when the first non-whitespace character is /", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "/help" })).toEqual({
      ok: false, reason: "reply must not start with /",
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "  /help" })).toEqual({
      ok: false, reason: "reply must not start with /",
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship /it" }).ok).toBe(true);
    db.close();
  });

  it("guard 6: rejects text over LIMITS.freeformReply (2000)", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    expect(LIMITS.freeformReply).toBe(2000);
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "x".repeat(2001) })).toEqual({
      ok: false, reason: "length",
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "x".repeat(2000) }).ok).toBe(true);
    db.close();
  });

  it("guard 7: rejects a CR or LF", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "line one\nline two" })).toEqual({
      ok: false, reason: "newline",
    });
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "line one\rline two" })).toEqual({
      ok: false, reason: "newline",
    });
    db.close();
  });

  it("guard 8: rejects tab, DEL, and other C0 control characters (Finding 5)", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    for (const bad of ["hello\tworld", "hello\x7fworld", "hello\x01world"]) {
      expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: bad })).toEqual({
        ok: false, reason: "control characters",
      });
    }
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" }).ok).toBe(true);
    db.close();
  });
});

describe("finishAnswer's question-answered payload carries kind (Finding 36) and mirrors source role", () => {
  it("structured: kind='structured', tool_use_id is the claimed tool_use_id, role from the source event", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, { ...question("tu_1"), role: "adhoc" });
    const claimed = claimAnswer(db, { question_event_id: q.id, reply: "1: 1" });
    expect(claimed.ok).toBe(true);
    if (!claimed.ok) return;
    finishAnswer(db, claimed.claim, true, null);
    const row = db.prepare(
      "SELECT role, payload FROM workflow_events WHERE type = 'question-answered' ORDER BY id DESC LIMIT 1",
    ).get() as { role: string; payload: string };
    expect(row.role).toBe("adhoc");
    expect(JSON.parse(row.payload)).toMatchObject({ kind: "structured", tool_use_id: "tu_1", ok: true });
    db.close();
  });

  it("freeform: kind='freeform', tool_use_id is null", () => {
    const db = openDb(":memory:");
    const ts = idleTurnStopped(db);
    const claimed = claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" });
    expect(claimed.ok).toBe(true);
    if (!claimed.ok) return;
    finishAnswer(db, claimed.claim, true, null);
    const row = db.prepare("SELECT payload FROM workflow_events WHERE type = 'question-answered' ORDER BY id DESC LIMIT 1").get() as { payload: string };
    expect(JSON.parse(row.payload)).toMatchObject({ kind: "freeform", tool_use_id: null, ok: true });
    db.close();
  });
});

describe("finishAnswer + failAnswerFinalization", () => {
  it("finalizes a successful claim and inserts one local outcome event", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, question("t-ok"), NOW);
    const claimed = claimAnswer(db, { question_event_id: q.id, reply: "1: yes" }, NOW);
    expect(claimed.ok).toBe(true);
    if (!claimed.ok) return;
    finishAnswer(db, claimed.claim, true, null, NOW);
    const mutation = db.prepare("SELECT ok, error, payload FROM mutations WHERE id = ?").get(claimed.claim.mutation_id) as {
      ok: number | null; error: string | null; payload: string;
    };
    expect(mutation.ok).toBe(1);
    expect(JSON.parse(mutation.payload).outcome).toBe("done");
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-ok'").get()).toEqual({ status: "done" });
    const events = db.prepare("SELECT type, delivery, payload FROM workflow_events WHERE type = 'question-answered'").all() as {
      type: string; delivery: string; payload: string;
    }[];
    expect(events).toHaveLength(1);
    expect(events[0].delivery).toBe("local");
    expect(JSON.parse(events[0].payload)).toMatchObject({
      question_event_id: q.id, tool_use_id: "t-ok", mutation_id: claimed.claim.mutation_id, ok: true,
    });
    db.close();
  });

  it("finalizes a failed claim with the bounded error and one outcome event", () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, question("t-fail"), NOW);
    const claimed = claimAnswer(db, { question_event_id: q.id, reply: "1: yes" }, NOW);
    expect(claimed.ok).toBe(true);
    if (!claimed.ok) return;
    finishAnswer(db, claimed.claim, false, "pane target is not live", NOW);
    const mutation = db.prepare("SELECT ok, error, payload FROM mutations WHERE id = ?").get(claimed.claim.mutation_id) as {
      ok: number | null; error: string | null; payload: string;
    };
    expect(mutation).toMatchObject({ ok: 0, error: "pane target is not live" });
    expect(JSON.parse(mutation.payload).outcome).toBe("failed");
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-fail'").get()).toEqual({ status: "failed" });
    const events = db.prepare("SELECT payload FROM workflow_events WHERE type = 'question-answered'").all() as { payload: string }[];
    expect(events).toHaveLength(1);
    expect(JSON.parse(events[0].payload)).toMatchObject({
      question_event_id: q.id, tool_use_id: "t-fail", mutation_id: claimed.claim.mutation_id,
      ok: false, error: "pane target is not live",
    });
    db.close();
  });

  it("failAnswerFinalization terminalizes only the named injecting claim", () => {
    const db = openDb(":memory:");
    const a = insertEvent(db, question("t-a"), NOW);
    const b = insertEvent(db, { ...question("t-b", "%2"), project: "p2" }, NOW);
    const first = claimAnswer(db, { question_event_id: a.id, reply: "1: yes" }, NOW);
    const second = claimAnswer(db, { question_event_id: b.id, reply: "1: yes" }, NOW);
    expect(first.ok && second.ok).toBe(true);
    if (!first.ok || !second.ok) return;
    expect(failAnswerFinalization(db, first.claim)).toBe(true);
    const firstMut = db.prepare("SELECT ok, error, payload FROM mutations WHERE id = ?").get(first.claim.mutation_id) as {
      ok: number | null; error: string | null; payload: string;
    };
    expect(firstMut).toMatchObject({ ok: 0, error: FINALIZATION_ERROR });
    expect(JSON.parse(firstMut.payload).outcome).toBe("failed");
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-a'").get()).toEqual({ status: "failed" });
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-b'").get()).toEqual({ status: "injecting" });
    expect(db.prepare("SELECT COUNT(*) AS c FROM workflow_events WHERE type = 'question-answered'").get()).toEqual({ c: 0 });
    expect(failAnswerFinalization(db, first.claim)).toBe(false);
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-a'").get()).toEqual({ status: "failed" });
    db.close();
  });
});

describe("abandonInjectingAnswers", () => {
  it("abandons only injecting claims and their mutations, with no outcome event", () => {
    const db = openDb(":memory:");
    const a = insertEvent(db, question("t-inj-a"), NOW);
    const b = insertEvent(db, { ...question("t-inj-b", "%2"), project: "p2" }, NOW);
    const c = insertEvent(db, { ...question("t-done", "%3"), project: "p3" }, NOW);
    const first = claimAnswer(db, { question_event_id: a.id, reply: "1: yes" }, NOW);
    const second = claimAnswer(db, { question_event_id: b.id, reply: "1: yes" }, NOW);
    const third = claimAnswer(db, { question_event_id: c.id, reply: "1: yes" }, NOW);
    expect(first.ok && second.ok && third.ok).toBe(true);
    if (!first.ok || !second.ok || !third.ok) return;
    finishAnswer(db, third.claim, true, null, NOW);
    expect(abandonInjectingAnswers(db)).toBe(2);
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-inj-a'").get()).toEqual({ status: "abandoned" });
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-inj-b'").get()).toEqual({ status: "abandoned" });
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-done'").get()).toEqual({ status: "done" });
    for (const id of [first.claim.mutation_id, second.claim.mutation_id]) {
      const row = db.prepare("SELECT ok, error, payload FROM mutations WHERE id = ?").get(id) as {
        ok: number | null; error: string | null; payload: string;
      };
      expect(row).toMatchObject({ ok: 0, error: RECOVERY_ERROR });
      expect(JSON.parse(row.payload).outcome).toBe("abandoned");
    }
    expect(JSON.parse(
      (db.prepare("SELECT payload FROM mutations WHERE id = ?").get(third.claim.mutation_id) as { payload: string }).payload,
    ).outcome).toBe("done");
    expect(db.prepare("SELECT COUNT(*) AS c FROM workflow_events WHERE type = 'question-answered'").get()).toEqual({ c: 1 });
    db.close();
  });
});

describe("getAfkEnabled / setAfk", () => {
  it("defaults to OFF and toggles atomically", () => {
    const db = openDb(":memory:");
    expect(getAfkEnabled(db)).toBe(false);
    setAfkOn(db);
    expect(getAfkEnabled(db)).toBe(true);
    setAfk(db, false);
    expect(getAfkEnabled(db)).toBe(false);
  });
});

describe("getForwardTypes / setForwardTypes (§5 Decision 2/6)", () => {
  it("an absent row falls back to DEFAULT_FORWARD_TYPES, then persists and reads back a change", () => {
    const db = openDb(":memory:");
    expect(getForwardTypes(db)).toEqual(DEFAULT_FORWARD_TYPES);
    setForwardTypes(db, ["run-finished", "merge-approved"]);
    expect(getForwardTypes(db)).toEqual(["run-finished", "merge-approved"]);
  });

  it("audits the previous value on each call", () => {
    const db = openDb(":memory:");
    setForwardTypes(db, ["question"]);
    setForwardTypes(db, ["question", "merge-approved"]);
    const rows = db.prepare("SELECT payload FROM mutations WHERE kind = 'workflow-forward-types' ORDER BY id").all() as { payload: string }[];
    expect(rows.map((r) => JSON.parse(r.payload))).toEqual([
      expect.objectContaining({ types: ["question"], previous: DEFAULT_FORWARD_TYPES }),
      expect.objectContaining({ types: ["question", "merge-approved"], previous: ["question"] }),
    ]);
  });
});

describe("hasPendingQuestion (pane+incarnation identity, Findings 47/48/52)", () => {
  // plan fixture gap: the plan's `q()` helper never set `payload.tmux_target`, so the two tests
  // below that call `claimAnswer` on its output always got `invalid-target` (claimAnswer's
  // `leadTarget` check requires it) — no claim row was ever inserted, silently defeating the
  // "resolves via a workflow_answer_claims row" scenario. Added here, matching the project ("p1").
  function q(db: ReturnType<typeof openDb>, overrides: Record<string, unknown> = {}) {
    return insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool", pane: "%1", tmux_incarnation: "111:222",
      payload: { tool_use_id: "tu_1", questions: [{ question: "q?", options: [{ label: "a" }] }] },
      ...overrides,
    });
  }

  it("is true only for the exact pane+incarnation, delivered/pending, unresolved question", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    q(db);
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:222" })).toBe(true);
    expect(hasPendingQuestion(db, { pane: "%2", tmuxIncarnation: "111:222" })).toBe(false); // different pane, same project — no cross-talk (Finding 52)
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:333" })).toBe(false); // same pane, new incarnation — stale session (Finding 47)
  });

  it("a question that terminated local (asked while AFK was off) never counts as pending (Finding 48)", () => {
    const db = openDb(":memory:");
    setAfk(db, false); // question below lands with delivery 'local'
    q(db);
    setAfkOn(db);
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:222" })).toBe(false);
  });

  it("resolves via question-resolved on the SAME pane+incarnation only (Finding 52)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const asked = q(db);
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "question-resolved", source: "deterministic",
      emitter: "claude-posttool", pane: "%9", tmux_incarnation: "999:999", payload: { tool_use_id: "tu_1" } });
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:222" })).toBe(true); // resolution on a DIFFERENT pane doesn't clear it
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "question-resolved", source: "deterministic",
      emitter: "claude-posttool", pane: "%1", tmux_incarnation: "111:222", payload: { tool_use_id: "tu_1" } });
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:222" })).toBe(false);
    void asked;
  });

  it("resolves via a workflow_answer_claims row keyed to the question's own event id", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const asked = q(db);
    claimAnswer(db, { question_event_id: asked.id, reply: "1: 1" });
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:222" })).toBe(false);
  });

  it("a claim on a DIFFERENT question event with a coincidentally-reused tool_use_id never clears this pane's question (Finding 52)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const asked = q(db); // pane %1, incarnation 111:222, tool_use_id "tu_1"
    // an UNRELATED question on a different pane happens to reuse the same tool_use_id string
    const otherQuestion = q(db, { pane: "%9", tmux_incarnation: "999:999" });
    claimAnswer(db, { question_event_id: otherQuestion.id, reply: "1: 1" }); // claims %9's question, NOT %1's
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:222" })).toBe(true); // %1's own question is still unclaimed
    void asked;
  });

  it("resolves via a later turn-stopped on the SAME pane+incarnation only", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    q(db);
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane: "%9", tmux_incarnation: "999:999", payload: { capsule_status: "done" } });
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:222" })).toBe(true); // different pane's turn-stopped doesn't clear it
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: "111:222", payload: { capsule_status: "done" } });
    expect(hasPendingQuestion(db, { pane: "%1", tmuxIncarnation: "111:222" })).toBe(false);
  });
});

describe("insertEvent's AFK gate (§4.5 step 3)", () => {
  it("AFK off suppresses a forward-policy event to local, question and attention-needed included (no interrupt carve-out)", () => {
    const db = openDb(":memory:");
    setAfk(db, false);
    expect(insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool", pane: "%1", tmux_incarnation: "111:222",
      payload: { tool_use_id: "tu_1", questions: [{ question: "q?", options: [{ label: "a" }] }] } }).delivery).toBe("local");
    expect(insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "attention-needed", source: "behavioral",
      emitter: "claude-notification", pane: "%1", tmux_incarnation: "111:222", payload: { reason: "agent_needs_input" } }).delivery).toBe("local");
  });

  it("AFK on delivers pending, unless a same-pane+incarnation question is still owed (attention-needed only)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool", pane: "%1", tmux_incarnation: "111:222",
      payload: { tool_use_id: "tu_1", questions: [{ question: "q?", options: [{ label: "a" }] }] } });
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "attention-needed", source: "behavioral",
      emitter: "claude-notification", pane: "%1", tmux_incarnation: "111:222", payload: { reason: "agent_needs_input" } });
    expect(row.delivery).toBe("suppressed");
  });
});

describe("insertEvent's forward-types gate (§4.5 step 4)", () => {
  it("AFK on + type excluded from forward_types → local", () => {
    const db = openDb(":memory:");
    setAfk(db, true);
    setForwardTypes(db, ["question"]); // run-finished excluded
    expect(insertEvent(db, finished).delivery).toBe("local");
  });

  it("AFK on + type included → falls through to the existing pending logic unchanged", () => {
    const db = openDb(":memory:");
    setAfk(db, true);
    setForwardTypes(db, ["run-finished"]);
    expect(insertEvent(db, finished).delivery).toBe("pending");
  });

  it("attention-needed still suppresses (not local) when included and a same-pane question is pending", () => {
    const db = openDb(":memory:");
    setAfk(db, true);
    setForwardTypes(db, ["question", "attention-needed"]); // both enabled (round 1 F1) — the
    // question itself must reach `pending` to count as pending for attention-needed's check
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool", pane: "%1", tmux_incarnation: "111:222",
      payload: { tool_use_id: "tu_1", questions: [{ question: "q?", options: [{ label: "a" }] }] } });
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "attention-needed", source: "behavioral",
      emitter: "claude-notification", pane: "%1", tmux_incarnation: "111:222", payload: { reason: "agent_needs_input" } });
    expect(row.delivery).toBe("suppressed");
  });

  it("a local-policy type (question-resolved) is unaffected by forward_types regardless of its content", () => {
    const db = openDb(":memory:");
    setAfk(db, true);
    setForwardTypes(db, []); // nothing forwardable — local-policy types don't consult this at all
    const row = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "question-resolved", source: "deterministic",
      emitter: "claude-posttool", payload: { tool_use_id: "tu_1" } });
    expect(row.delivery).toBe("local");
  });
});

// Existing file-level `NOW` is a Date used by claim/insert fixtures (2026-08-18).
// Staleness tests need a millisecond epoch (spec §4.4.2 `now: number`).
const STALE_NOW = Date.parse("2026-08-25T12:00:00.000Z");
const INC = "234790:1787586213";

function seedTurnStopped(
  db: ReturnType<typeof openDb>,
  overrides: Partial<WorkflowEventInput> = {},
  when: Date = new Date(STALE_NOW - 5 * 60 * 1000),
) {
  const { payload, ...rest } = overrides;
  return insertEvent(db, {
    run_id: null, type: "turn-stopped", emitter: "claude-stop", source: "deterministic", role: "lead",
    project: "Acme.AI", pane: "%1", tmux_incarnation: INC,
    payload: { capsule_status: "waiting", capsule_minutes: 1, capsule_rule: "tag", ...(payload as object ?? {}) },
    ...rest,
  }, when);
}

describe("evaluateStaleness (§4.4.1-4.4.2, TS/TSt algorithm)", () => {
  it("alerts possibly-stuck once the waiting deadline has passed", () => {
    const db = openDb(":memory:");
    seedTurnStopped(db); // declared waiting ~1m, 5 min ago (default `when`) — well past deadline
    const { candidates, nullIncarnation } = evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC);
    expect(candidates).toEqual([expect.objectContaining({
      pane: "%1", tmuxIncarnation: INC, project: "Acme.AI", role: "lead", reason: "possibly-stuck",
    })]);
    expect(nullIncarnation).toBe(0);
  });

  it("does not alert before the deadline", () => {
    const db = openDb(":memory:");
    seedTurnStopped(db, {}, new Date(STALE_NOW - 5000)); // 5 seconds ago, deadline is minutes out
    expect(evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC).candidates).toEqual([]);
  });

  it("done/needs_input/blocked never alert regardless of age", () => {
    for (const status of ["done", "needs_input", "blocked"]) {
      const db = openDb(":memory:");
      seedTurnStopped(db, { payload: { capsule_status: status } }, new Date(STALE_NOW - 999 * 60 * 1000));
      expect(evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC).candidates).toEqual([]);
    }
  });

  it("a later turn-started supersedes a stale waiting declaration (Finding 5) — falls to no-signal fallback instead", () => {
    const db = openDb(":memory:");
    seedTurnStopped(db, {}, new Date(STALE_NOW - 999 * 60 * 1000)); // ancient waiting, would be possibly-stuck
    insertEvent(db, { run_id: null, type: "turn-started", emitter: "claude-userprompt", source: "deterministic", role: "lead",
      project: "Acme.AI", pane: "%1", tmux_incarnation: INC, payload: {} }, new Date(STALE_NOW - 1000));
    expect(evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC).candidates).toEqual([]); // fresh turn-started, well under 120min
  });

  it("no-signal fallback fires past 120 minutes of total silence", () => {
    const db = openDb(":memory:");
    insertEvent(db, { run_id: null, type: "turn-stopped", emitter: "claude-stop", source: "deterministic", role: "lead",
      project: "Acme.AI", pane: "%1", tmux_incarnation: INC, payload: { capsule_status: "unknown" } }, new Date(STALE_NOW - 125 * 60 * 1000));
    expect(evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC).candidates).toEqual([expect.objectContaining({ reason: "no-signal" })]);
  });

  it("a dead pane, a mismatched incarnation, or a NULL incarnation never evaluates (Findings 3/19/28)", () => {
    const db = openDb(":memory:");
    seedTurnStopped(db, {}, new Date(STALE_NOW - 999 * 60 * 1000));
    expect(evaluateStaleness(db, STALE_NOW, new Set(), INC).candidates).toEqual([]); // dead — not live at all
    expect(evaluateStaleness(db, STALE_NOW, new Set(["%1"]), "999999:1").candidates).toEqual([]); // wrong incarnation

    const db2 = openDb(":memory:");
    seedTurnStopped(db2, { tmux_incarnation: null }, new Date(STALE_NOW - 999 * 60 * 1000));
    const result = evaluateStaleness(db2, STALE_NOW, new Set(["%1"]), INC);
    expect(result.candidates).toEqual([]);
    expect(result.nullIncarnation).toBe(1);
  });

  it("a pane's own prior synthetic attention-needed row never re-enters the eligibility scan (Rule A, Finding 38)", () => {
    const db = openDb(":memory:");
    seedTurnStopped(db, {}, new Date(STALE_NOW - 999 * 60 * 1000));
    insertEvent(db, { run_id: null, type: "attention-needed", emitter: "jaxos", source: "deterministic", role: "lead",
      project: "Acme.AI", pane: "%1", tmux_incarnation: null, payload: { reason: "possibly-stuck" } }, new Date(STALE_NOW - 1000));
    // the synthetic row is recent but jaxos-emitted — must not count as the "last activity" reference
    const { candidates, nullIncarnation } = evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC);
    expect(candidates).toEqual([expect.objectContaining({ reason: "possibly-stuck" })]); // still overdue, still checked (not nullIncarnation)
    expect(nullIncarnation).toBe(0);
  });

  it("a healthy, non-alerting pane still counts toward `checked` (Findings 9/30)", () => {
    const db = openDb(":memory:");
    seedTurnStopped(db, { payload: { capsule_status: "done" } }, new Date(STALE_NOW - 999 * 60 * 1000)); // done — never alerts, any age
    const result = evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC);
    expect(result.candidates).toEqual([]);
    expect(result.checked).toBe(1); // live + incarnation-confirmed, even though it produced zero candidates
  });

  it("a synthetic jaxos-emitted turn-started can never supersede the agent's own waiting reference (Rule A, Finding 1)", () => {
    const db = openDb(":memory:");
    const ref = seedTurnStopped(db, {}, new Date(STALE_NOW - 999 * 60 * 1000)); // agent-sourced, ancient waiting — overdue
    // A synthetic jaxos-emitted turn-started, with a LATER id than the agent's own turn-stopped,
    // must never be read by the ts/tst reference-state lookup as a real turn-started superseding
    // the waiting declaration — only an agent-sourced turn-started may do that (§4.4.2 step 2).
    insertEvent(db, {
      run_id: null, type: "turn-started", emitter: "jaxos", source: "deterministic", role: "lead",
      project: "Acme.AI", pane: "%1", tmux_incarnation: INC, payload: {},
    });
    const { candidates } = evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC);
    expect(candidates).toEqual([expect.objectContaining({ referenceEventId: ref.id, reason: "possibly-stuck" })]);
  });
});

type TmuxCandidate = Extract<StalenessCandidate, { transport: "tmux" }>;
function stuckCandidate(overrides: Partial<TmuxCandidate> = {}): TmuxCandidate {
  return {
    transport: "tmux", pane: "%1", tmuxIncarnation: INC, project: "Acme.AI", role: "lead",
    reason: "possibly-stuck", referenceEventId: 1,
    payload: { reason: "possibly-stuck", capsule_minutes: 1, declared_at: "2026-08-25T11:55:00.000Z", overdue_by_s: 60, reference_event_id: 1 },
    ...overrides,
  };
}

describe("maybeInsertStalenessAlert (§4.4.3, four-condition atomic pre-gate)", () => {
  it("inserts when AFK is on, no pending question, reference still current, no prior alert for this reason", () => {
    const db = openDb(":memory:");
    const ref = seedTurnStopped(db);
    setAfkOn(db);
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(true);
    const row = db.prepare("SELECT * FROM workflow_events WHERE type = 'attention-needed'").get() as WorkflowEventRow;
    expect(row.project).toBe("Acme.AI"); // Finding 32 — never hardcoded empty
    expect(row.role).toBe("lead");
    expect(row.tmux_incarnation).toBe(INC);
  });

  it("AFK off: inserts nothing, across 10 consecutive overdue ticks (Finding 30)", () => {
    const db = openDb(":memory:");
    const ref = seedTurnStopped(db);
    setAfk(db, false);
    for (let i = 0; i < 10; i++) expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(false);
    expect(db.prepare("SELECT COUNT(*) c FROM workflow_events WHERE type = 'attention-needed'").get()).toEqual({ c: 0 });
  });

  it("AFK later turned on: the very next tick inserts, since nothing was ever written to dedup against (Finding 6)", () => {
    const db = openDb(":memory:");
    const ref = seedTurnStopped(db);
    setAfk(db, false);
    maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW);
    setAfkOn(db);
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(true);
  });

  it("a live, current-incarnation, delivered pending question suppresses; resolving it unblocks the very next tick", () => {
    const db = openDb(":memory:");
    const ref = seedTurnStopped(db);
    setAfkOn(db);
    insertEvent(db, { ...question("t-c3"), pane: "%1", tmux_incarnation: INC });
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(false);
    insertEvent(db, { ...resolved("t-c3"), pane: "%1", tmux_incarnation: INC }); // question-resolved, same pane+incarnation, same tool_use_id
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(true);
  });

  it("a stale-incarnation question never suppresses (Finding 47)", () => {
    const db = openDb(":memory:");
    // AFK goes ON FIRST, deliberately: inserted while AFK is off, the question would terminate
    // `local` and the delivery filter alone would exclude it — the test would pass even with the
    // incarnation check broken. With AFK on it is a real pending question, so ONLY the
    // incarnation scoping can keep it from suppressing.
    setAfkOn(db);
    insertEvent(db, { ...question("t-stale"), pane: "%1", tmux_incarnation: "111:1" }); // OLD incarnation, unanswered
    const stale = db.prepare("SELECT delivery FROM workflow_events ORDER BY id DESC LIMIT 1").get() as { delivery: string };
    expect(stale.delivery).not.toBe("local"); // guard the guard: the fixture must isolate incarnation, not delivery
    const ref = seedTurnStopped(db, { tmux_incarnation: INC }); // NEW incarnation
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(true);
  });

  it("dedup is scoped to pane AND reason — an alert for the other reason never masks this one (spec 4.4.3 condition 4)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const ref = seedTurnStopped(db);
    // First possibly-stuck alert for this reference lands.
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(true);
    // A no-signal alert for the same pane becomes the newest jaxos row. Querying the latest row
    // for the pane WITHOUT filtering by reason would now miss the possibly-stuck alert above and
    // let it be inserted a second time.
    insertEvent(db, {
      run_id: null, project: "jax-os", role: "lead", type: "attention-needed",
      source: "deterministic", emitter: "jaxos", pane: "%1", tmux_incarnation: INC,
      payload: { reason: "no-signal", reference_event_id: ref.id },
    });
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(false);
  });

  it("revalidation ignores a synthetic jaxos-emitted turn-started/turn-stopped (Rule A, Finding 1)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const ref = seedTurnStopped(db); // agent-sourced waiting, overdue
    // A synthetic jaxos-emitted turn-started with a LATER id than `ref` must not be read by the
    // revalidation guard as "a later turn-started superseded the waiting state" — only an
    // agent-sourced turn-started counts (§4.4.3 condition 3, mirroring §4.4.2 step 2).
    insertEvent(db, {
      run_id: null, type: "turn-started", emitter: "jaxos", source: "deterministic", role: "lead",
      project: "Acme.AI", pane: "%1", tmux_incarnation: INC, payload: {},
    });
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(true);
  });

  it("an undelivered (local) question never suppresses (Finding 48)", () => {
    const db = openDb(":memory:");
    setAfk(db, false);
    insertEvent(db, { ...question("t-local"), pane: "%1", tmux_incarnation: INC }); // asked while AFK off -> delivery local
    setAfkOn(db);
    const ref = seedTurnStopped(db);
    expect(maybeInsertStalenessAlert(db, stuckCandidate({ referenceEventId: ref.id }), STALE_NOW)).toBe(true);
  });

  // Cold review Finding 8: this test's original name claimed "real concurrent-writer safety."
  // better-sqlite3's API is fully synchronous, so nothing in this process can interleave
  // dbA's transaction with dbB's — `resultA` always runs to completion before `resultB` even
  // starts. Proven by mutation: changing `.immediate()` to the default deferred mode at
  // maybeInsertStalenessAlert's own transaction still leaves this test green. What it DOES prove
  // is real: two INDEPENDENT connections to the same on-disk file, called one after the other,
  // correctly dedup via condition 4's fresh re-read (dbB sees dbA's already-committed row).
  // What it does NOT prove: transaction-lock behavior under genuine overlapping writers (that
  // would need a second OS thread/process holding a transaction open while this one attempts to
  // write) — not built here; ponytail, revisit only if a real interleaving bug ever surfaces.
  it("two SEPARATE db connections to the same on-disk file dedup correctly — sequential only, not a lock-contention proof (Finding 4/34/8)", () => {
    const dir = mkdtempSync(join(tmpdir(), "jaxos-staleness-"));
    const dbPath = join(dir, "test.db");
    try {
      const dbA = openDb(dbPath);
      const dbB = openDb(dbPath);
      const ref = seedTurnStopped(dbA);
      setAfk(dbA, true);
      const c = stuckCandidate({ referenceEventId: ref.id });
      const resultA = maybeInsertStalenessAlert(dbA, c, STALE_NOW);
      const resultB = maybeInsertStalenessAlert(dbB, c, STALE_NOW); // dbB's transaction re-reads condition 4 fresh, sees dbA's committed row
      expect([resultA, resultB].filter(Boolean).length).toBe(1);
      dbA.close();
      dbB.close();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("reference revalidation: a genuine evaluation-to-insert race — evaluateStaleness picks a current reference, a superseding event lands in the gap, THEN the insert call revalidates and aborts (Finding 46/34)", () => {
    const db = openDb(":memory:");
    const ref = seedTurnStopped(db); // ancient waiting, would be possibly-stuck — the only turn-stopped so far
    setAfkOn(db);
    const evaluated = evaluateStaleness(db, STALE_NOW, new Set(["%1"]), INC);
    expect(evaluated.candidates).toEqual([expect.objectContaining({ referenceEventId: ref.id, reason: "possibly-stuck" })]);
    insertEvent(db, { run_id: null, type: "turn-stopped", emitter: "claude-stop", source: "deterministic", role: "lead",
      project: "Acme.AI", pane: "%1", tmux_incarnation: INC, payload: { capsule_status: "unknown" },
    }, new Date(STALE_NOW - 1000));
    expect(maybeInsertStalenessAlert(db, evaluated.candidates[0], STALE_NOW)).toBe(false);
  });
});

describe("pane identity across tmux incarnations (Spec B §4.1, test 21)", () => {
  const OLD_INC = "1000:1000000000";
  const NEW_INC = INCARNATION;

  function questionAt(db: ReturnType<typeof openDb>, tool: string, incarnation: string, pane = "%1", when: Date = NOW) {
    return insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
      emitter: "claude-pretool", pane, tmux_incarnation: incarnation,
      payload: { tool_use_id: tool, questions: [{ question: "q?", options: [{ label: "a" }] }] },
    }, when);
  }

  function turnStoppedAt(db: ReturnType<typeof openDb>, incarnation: string, pane = "%1", when: Date = NOW) {
    return insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane, tmux_incarnation: incarnation, payload: { capsule_status: "done" },
    }, when);
  }

  function turnStartedAt(db: ReturnType<typeof openDb>, incarnation: string, pane = "%1", when: Date = NOW) {
    return insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
      emitter: "claude-userprompt", pane, tmux_incarnation: incarnation, payload: {},
    }, when);
  }

  function resolvedAt(db: ReturnType<typeof openDb>, tool: string, incarnation: string, pane = "%1", when: Date = NOW) {
    return insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "question-resolved", source: "deterministic",
      emitter: "claude-posttool", pane, tmux_incarnation: incarnation, payload: { tool_use_id: tool },
    }, when);
  }

  it("claimAnswer: a same-pane, other-incarnation turn-stopped does not block the current claim", () => {
    const db = openDb(":memory:");
    const qNew = questionAt(db, "t-new", NEW_INC);
    turnStoppedAt(db, OLD_INC); // higher id than qNew, but a DIFFERENT incarnation
    const result = claimAnswer(db, { question_event_id: qNew.id, reply: "1: 1" }, NOW);
    expect(result.ok).toBe(true);
    db.close();
  });

  it("claimAnswer: a same-pane, other-incarnation question-resolved (same tool_use_id) does not resolve the current claim", () => {
    const db = openDb(":memory:");
    const qNew = questionAt(db, "t-shared", NEW_INC);
    resolvedAt(db, "t-shared", OLD_INC); // same tool_use_id, wrong incarnation
    const result = claimAnswer(db, { question_event_id: qNew.id, reply: "1: 1" }, NOW);
    expect(result.ok).toBe(true);
    db.close();
  });

  it("claimAnswer: a same-pane, other-incarnation injecting claim does not block the current claim", () => {
    const db = openDb(":memory:");
    const qOld = questionAt(db, "t-old", OLD_INC);
    const first = claimAnswer(db, { question_event_id: qOld.id, reply: "1: 1" }, NOW);
    expect(first.ok).toBe(true); // leaves an injecting claim under OLD_INC on pane %1
    const qNew = questionAt(db, "t-new", NEW_INC);
    const second = claimAnswer(db, { question_event_id: qNew.id, reply: "1: 1" }, NOW);
    expect(second.ok).toBe(true);
    db.close();
  });

  it("evaluateStaleness: a current-incarnation event is not paired with an expired incarnation's turn-stopped", () => {
    const db = openDb(":memory:");
    // OLD_INC left an overdue "waiting" turn-stopped — this is the row a pane-only lookup would
    // wrongly select as "the" reference for the pane.
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: OLD_INC,
      payload: { capsule_status: "waiting", capsule_minutes: 1 },
    }, new Date(STALE_NOW - 999 * 60 * 1000));
    // NEW_INC is live and mid-turn — its only event so far is NOT a turn-stopped/turn-started,
    // so a pane-only `ts`/`tst` lookup falls through to the stale OLD_INC row.
    questionAt(db, "t-live", NEW_INC, "%1", new Date(STALE_NOW - 1000));
    const { candidates } = evaluateStaleness(db, STALE_NOW, new Set(["%1"]), NEW_INC);
    expect(candidates).toEqual([]);
    db.close();
  });

  it("maybeInsertStalenessAlert: a candidate whose reference belongs to a different incarnation is never inserted", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const staleRef = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: OLD_INC,
      payload: { capsule_status: "waiting", capsule_minutes: 1 },
    }, new Date(STALE_NOW - 999 * 60 * 1000));
    const candidate: StalenessCandidate = {
      transport: "tmux", pane: "%1", tmuxIncarnation: NEW_INC, project: "p1", role: "lead",
      reason: "possibly-stuck", referenceEventId: staleRef.id,
      payload: {
        reason: "possibly-stuck", capsule_minutes: 1, declared_at: staleRef.ts,
        overdue_by_s: 60, reference_event_id: staleRef.id,
      },
    };
    expect(maybeInsertStalenessAlert(db, candidate, STALE_NOW)).toBe(false);
    db.close();
  });
});

describe("hasOpenTurn / isWorking (§4.1, tests 5/14/17)", () => {
  it("turn-started newer than turn-stopped is an open turn", () => {
    const db = openDb(":memory:");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    expect(hasOpenTurn(db, ref)).toBe(true);
    db.close();
  });

  it("turn-stopped newer than turn-started is not an open turn", () => {
    const db = openDb(":memory:");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
        emitter: "claude-stop", payload: { capsule_status: "done" }, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:05:00Z"),
    );
    expect(hasOpenTurn(db, ref)).toBe(false);
    db.close();
  });

  it("a pane with only turn-stopped has no open turn", () => {
    const db = openDb(":memory:");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
        emitter: "claude-stop", payload: { capsule_status: "done" }, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    expect(hasOpenTurn(db, ref)).toBe(false);
    db.close();
  });

  it("turn-started and turn-stopped under DIFFERENT incarnations do not pair (test 14)", () => {
    const db = openDb(":memory:");
    const started = { pane: "%1", tmuxIncarnation: "100:1" };
    const stopped = { pane: "%1", tmuxIncarnation: "200:1" }; // reissued pane id, new tmux server
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: started.pane, tmux_incarnation: started.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
        emitter: "claude-stop", payload: { capsule_status: "done" }, pane: stopped.pane, tmux_incarnation: stopped.tmuxIncarnation },
      new Date("2026-08-27T10:05:00Z"),
    );
    expect(hasOpenTurn(db, started)).toBe(true); // still open under the OLD incarnation
    expect(hasOpenTurn(db, stopped)).toBe(false); // the new incarnation never opened a turn
    db.close();
  });

  it("isWorking requires liveness — a dead pane with an open turn is idle, not working (test 17)", () => {
    const db = openDb(":memory:");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    const dead: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(), incarnation: null };
    expect(isWorking(db, ref, dead)).toBe(false);
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: "100:1" };
    expect(isWorking(db, ref, live)).toBe(true);
    db.close();
  });

  it("a pane present in the snapshot under a DIFFERENT incarnation does not count as live (test 7)", () => {
    const db = openDb(":memory:");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: "200:1" }; // same pane id, new server
    expect(isWorking(db, ref, live)).toBe(false);
    db.close();
  });
});

describe("countFreshEvents (§4.4, test 6)", () => {
  it("returns 0 when no events are newer than the status.md mtime", () => {
    const db = openDb(":memory:");
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
      new Date("2026-08-01T00:00:00Z"),
    );
    expect(countFreshEvents(db, "p1", "2026-08-27T00:00:00.000Z")).toBe(0);
    db.close();
  });

  it("counts 19 fresh events", () => {
    const db = openDb(":memory:");
    const base = Date.parse("2026-08-27T00:00:00.000Z");
    for (let i = 0; i < 19; i++) {
      insertEvent(
        db,
        { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
          emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
        new Date(base + (i + 1) * 1000),
      );
    }
    expect(countFreshEvents(db, "p1", "2026-08-27T00:00:00.000Z")).toBe(19);
    db.close();
  });

  it("counts 20 fresh events and the badge threshold is reached", () => {
    const db = openDb(":memory:");
    const base = Date.parse("2026-08-27T00:00:00.000Z");
    for (let i = 0; i < 20; i++) {
      insertEvent(
        db,
        { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
          emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
        new Date(base + (i + 1) * 1000),
      );
    }
    expect(countFreshEvents(db, "p1", "2026-08-27T00:00:00.000Z")).toBe(20);
    db.close();
  });

  it("excludes emitter='jaxos' rows from the count (test 9)", () => {
    const db = openDb(":memory:");
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "question-answered", source: "deterministic",
        emitter: "jaxos", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
      new Date("2026-08-27T01:00:00Z"),
    );
    expect(countFreshEvents(db, "p1", "2026-08-27T00:00:00.000Z")).toBe(0);
    db.close();
  });
});

describe("getHubMembership (§7.1b, tests 16/23)", () => {
  it("60 live projects all appear uncapped; the historical set truncates at 50", () => {
    const db = openDb(":memory:");
    const now = Date.parse("2026-08-27T12:00:00Z");
    const paneIds = new Set<string>();
    for (let i = 0; i < 60; i++) {
      const pane = `%${i + 1}`;
      paneIds.add(pane);
      insertEvent(
        db,
        { run_id: null, project: `live-${i}`, role: "lead", type: "turn-started", source: "deterministic",
          emitter: "claude-userprompt", payload: {}, pane, tmux_incarnation: "100:1" },
        new Date(now),
      );
    }
    for (let i = 0; i < 60; i++) {
      insertEvent(
        db,
        { run_id: null, project: `hist-${i}`, role: "lead", type: "turn-stopped", source: "deterministic",
          emitter: "claude-stop", payload: { capsule_status: "done" }, pane: null, tmux_incarnation: null },
        new Date(now - (i + 1) * 60 * 1000), // no pane -> never live, always historical
      );
    }
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds, incarnation: "100:1" };
    const { projects, historicalTruncated } = getHubMembership(db, live, now);
    for (let i = 0; i < 60; i++) expect(projects).toContain(`live-${i}`);
    const historicalCount = projects.filter((p) => p.startsWith("hist-")).length;
    expect(historicalCount).toBe(50);
    expect(historicalTruncated).toBe(true);
    db.close();
  });

  it("a project whose only rows are emitter='jaxos' is not a hub member (test 23)", () => {
    const db = openDb(":memory:");
    const now = Date.parse("2026-08-27T12:00:00Z");
    insertEvent(
      db,
      { run_id: null, project: "ghost", role: "lead", type: "question-answered", source: "deterministic",
        emitter: "jaxos", payload: {}, pane: "%9", tmux_incarnation: "100:1" },
      new Date(now),
    );
    // The pane itself IS live — only its jaxos-sourced row names the project, which must not confer membership.
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%9"]), incarnation: "100:1" };
    const { projects } = getHubMembership(db, live, now);
    expect(projects).not.toContain("ghost");
    db.close();
  });

  it("a project whose only registration is a session row (event POST failed) is still a live member (Finding 4)", () => {
    const db = openDb(":memory:");
    const now = Date.parse("2026-08-27T12:00:00Z");
    // The hook posts the session BEFORE the event (scripts/jaxflow_hook.py:317-327) — this
    // fixture reproduces the event POST having failed, leaving a session row with NO event row
    // at all for this pane/incarnation.
    upsertSession(db, { project: "session-only", session: "main", pane: "%5", role: "lead", tmux_incarnation: "100:1" });
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%5"]), incarnation: "100:1" };
    const { projects } = getHubMembership(db, live, now);
    expect(projects).toContain("session-only");
    db.close();
  });

  it("a pane that switched from project A to project B does not keep A live via A's old events (cold review Finding 2)", () => {
    const db = openDb(":memory:");
    const now = Date.parse("2026-08-27T12:00:00Z");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    // Old activity for project A, well outside the 30-day historical window — the ONLY thing that
    // could still mark it as a member is the live-pane path, which must now ignore it.
    insertEvent(
      db,
      { run_id: null, project: "project-a", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date(now - 60 * 24 * 60 * 60 * 1000),
    );
    // The same pane/incarnation later switches to project B without a tmux restart — the session
    // upsert now points this pane_key at B.
    upsertSession(
      db,
      { project: "project-b", session: "main", pane: ref.pane, role: "lead", tmux_incarnation: ref.tmuxIncarnation },
      new Date(now - 1000),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set([ref.pane]), incarnation: ref.tmuxIncarnation };
    const { projects } = getHubMembership(db, live, now);
    expect(projects).toContain("project-b");
    expect(projects).not.toContain("project-a");
    db.close();
  });

  it("a stale project-A session row does not defeat a later project-B event (branch review Finding 2)", () => {
    const db = openDb(":memory:");
    const now = Date.parse("2026-08-27T12:00:00Z");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    // Session registration is best-effort — this reproduces a pane whose session row still says A
    // (the registration to B never landed) while an agent-sourced event for B DID land, later.
    upsertSession(
      db,
      { project: "project-a", session: "main", pane: ref.pane, role: "lead", tmux_incarnation: ref.tmuxIncarnation },
      new Date(now - 60 * 60 * 1000),
    );
    insertEvent(
      db,
      { run_id: null, project: "project-b", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date(now - 1000),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set([ref.pane]), incarnation: ref.tmuxIncarnation };
    const { projects } = getHubMembership(db, live, now);
    expect(projects).toContain("project-b");
    expect(projects).not.toContain("project-a");
    db.close();
  });
});

describe("getProjectPanes (§4.1, Findings 4/5)", () => {
  it("enumerates a pane whose only registration is a session row, using registered_at as its fallback lastEventTs (Finding 4)", () => {
    const db = openDb(":memory:");
    upsertSession(
      db,
      { project: "p1", session: "main", pane: "%7", role: "adhoc", tmux_incarnation: "100:1" },
      new Date("2026-08-27T09:00:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%7"]), incarnation: "100:1" };
    const panes = getProjectPanes(db, "p1", live);
    expect(panes).toHaveLength(1);
    expect(panes[0]).toMatchObject({ pane: "%7", tmuxIncarnation: "100:1", session: "main", role: "adhoc", live: true });
    expect(panes[0].lastEventTs).toBe("2026-08-27T09:00:00.000Z");
    db.close();
  });

  it("a session-only pane that is no longer live is not enumerated (Finding 6, round-2 cold review)", () => {
    const db = openDb(":memory:");
    upsertSession(
      db,
      { project: "p1", session: "main", pane: "%7", role: "adhoc", tmux_incarnation: "100:1" },
      new Date("2026-08-27T09:00:00Z"),
    );
    // the pane died (or the tmux server restarted) — a later live snapshot shows a completely
    // different pane set and incarnation; nothing ever deletes the stale workflow_sessions row
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%3"]), incarnation: "200:1" };
    const panes = getProjectPanes(db, "p1", live);
    expect(panes).toHaveLength(0);
    db.close();
  });

  it("pendingQuestion is the PENDING row itself, not the newest question overall, when an older question is still pending and a newer one has already resolved (Finding 5)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const older = insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
        emitter: "claude-pretool", pane: "%1", tmux_incarnation: "100:1",
        payload: { tool_use_id: "tu_older", questions: [{ question: "older?", options: [{ label: "a" }] }] } },
      new Date("2026-08-27T10:00:00Z"),
    );
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "question", source: "deterministic",
        emitter: "claude-pretool", pane: "%1", tmux_incarnation: "100:1",
        payload: { tool_use_id: "tu_newer", questions: [{ question: "newer?", options: [{ label: "a" }] }] } },
      new Date("2026-08-27T10:05:00Z"),
    );
    // The newer question resolves; the older one never does — a pane-only lookup that just took
    // the newest question row would wrongly return this resolved one as "the" pending question.
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "question-resolved", source: "deterministic",
        emitter: "claude-posttool", pane: "%1", tmux_incarnation: "100:1", payload: { tool_use_id: "tu_newer" } },
      new Date("2026-08-27T10:06:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: "100:1" };
    const panes = getProjectPanes(db, "p1", live);
    expect(panes[0].pendingQuestion).toMatchObject({ eventId: older.id });
    db.close();
  });

  it("a pane's row for project A reads live:false/working:false after the pane switches to project B (cold review Finding 2)", () => {
    const db = openDb(":memory:");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    // project A has an OPEN turn — never stopped. This is the dangerous case: without the
    // current-project gate, isWorking sees a live pane_key with an open turn and would report
    // working:true for a project the pane already left.
    insertEvent(
      db,
      { run_id: null, project: "project-a", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    // The pane switches to project B — the session upsert now points this pane_key at B.
    upsertSession(
      db,
      { project: "project-b", session: "main", pane: ref.pane, role: "lead", tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:10:00Z"),
    );
    insertEvent(
      db,
      { run_id: null, project: "project-b", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:11:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set([ref.pane]), incarnation: ref.tmuxIncarnation };

    const panesA = getProjectPanes(db, "project-a", live);
    expect(panesA).toHaveLength(1); // the historical row still enumerates — it genuinely happened
    expect(panesA[0]).toMatchObject({ live: false, working: false });

    const panesB = getProjectPanes(db, "project-b", live);
    expect(panesB).toHaveLength(1);
    expect(panesB[0]).toMatchObject({ live: true, working: true });

    db.close();
  });

  it("project A's historical row never shows project B's capsule or pending question after the pane switches (branch review Finding 1)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    insertEvent(
      db,
      { run_id: null, project: "project-a", role: "lead", type: "turn-stopped", source: "deterministic",
        emitter: "claude-stop", payload: { capsule_status: "waiting", capsule_minutes: 5, capsule_rule: "tag" },
        pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    // The pane switches to project B — a later capsule AND a later pending question, both B's.
    upsertSession(
      db,
      { project: "project-b", session: "main", pane: ref.pane, role: "lead", tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:10:00Z"),
    );
    insertEvent(
      db,
      { run_id: null, project: "project-b", role: "lead", type: "turn-stopped", source: "deterministic",
        emitter: "claude-stop", payload: { capsule_status: "needs_input", capsule_rule: "tag" },
        pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:20:00Z"),
    );
    insertEvent(
      db,
      { run_id: null, project: "project-b", role: "lead", type: "question", source: "deterministic",
        emitter: "claude-pretool", pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation,
        payload: { tool_use_id: "tu_b", questions: [{ question: "b?", options: [{ label: "a" }] }] } },
      new Date("2026-08-27T10:21:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set([ref.pane]), incarnation: ref.tmuxIncarnation };

    const panesA = getProjectPanes(db, "project-a", live);
    expect(panesA).toHaveLength(1);
    expect(panesA[0].capsule).toMatchObject({ status: "waiting", minutes: 5 }); // A's own capsule, not B's "needs_input"
    expect(panesA[0].pendingQuestion).toBeNull(); // A never asked a question — B's must not leak here

    const panesB = getProjectPanes(db, "project-b", live);
    expect(panesB[0].capsule).toMatchObject({ status: "needs_input" });
    expect(panesB[0].pendingQuestion).toMatchObject({ payload: { tool_use_id: "tu_b" } });

    db.close();
  });
});

describe("HubPane.runtime — session-runtime-label fix", () => {
  it("runtimeFromEmitter derives claude/codex by prefix, and falls back to unknown", () => {
    expect(runtimeFromEmitter("claude-stop")).toBe("claude");
    expect(runtimeFromEmitter("codex-userprompt")).toBe("codex");
    expect(runtimeFromEmitter("wrapper")).toBe("unknown");
    expect(runtimeFromEmitter(undefined)).toBe("unknown");
  });

  it("getProjectPanes derives the pane's runtime from its latest event's emitter, not the project's builder", () => {
    const db = openDb(":memory:");
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
      new Date("2026-08-27T10:00:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: "100:1" };
    const [pane] = getProjectPanes(db, "p1", live);
    expect(pane.runtime).toBe("claude");
    db.close();
  });

  it("a session-registered pane with no event row yet reads runtime unknown (Finding 4 fallback)", () => {
    const db = openDb(":memory:");
    upsertSession(
      db,
      { project: "p1", session: "main", pane: "%7", role: "adhoc", tmux_incarnation: "100:1" },
      new Date("2026-08-27T09:00:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%7"]), incarnation: "100:1" };
    const [pane] = getProjectPanes(db, "p1", live);
    expect(pane.runtime).toBe("unknown");
    db.close();
  });
});

describe("getHubData assembles the full envelope (integration)", () => {
  it("returns panes, session/role, freshness, and a timeline for a live project", () => {
    const db = openDb(":memory:");
    const ref = { pane: "%1", tmuxIncarnation: "100:1" };
    upsertSession(db, { project: "p1", session: "main", pane: ref.pane, role: "lead", tmux_incarnation: ref.tmuxIncarnation });
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: ref.pane, tmux_incarnation: ref.tmuxIncarnation },
      new Date("2026-08-27T10:00:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: "100:1" };
    const data = getHubData(db, live, { p1: "2026-08-01T00:00:00.000Z" }, null, Date.parse("2026-08-27T10:05:00Z"));
    expect(data.byProject.p1.panes).toHaveLength(1);
    expect(data.byProject.p1.panes[0]).toMatchObject({
      pane: "%1", tmuxIncarnation: "100:1", session: "main", role: "lead", live: true, working: true,
    });
    expect(data.byProject.p1.freshnessCount).toBe(1);
    expect(data.byProject.p1.timeline).toHaveLength(1);
    expect(data.byProject.p1.timeline[0]).toMatchObject({ type: "turn-started", pane: "%1" });
    expect(data.historicalTruncated).toBe(false);
    db.close();
  });

  it("a project with no status.md mtime supplied reports freshnessCount 0", () => {
    const db = openDb(":memory:");
    insertEvent(
      db,
      { run_id: null, project: "hist-only", role: "lead", type: "turn-stopped", source: "deterministic",
        emitter: "claude-stop", payload: { capsule_status: "done" }, pane: null, tmux_incarnation: null },
      new Date("2026-08-27T10:00:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(), incarnation: null };
    const data = getHubData(db, live, {}, null, Date.parse("2026-08-27T10:05:00Z"));
    expect(data.byProject["hist-only"].freshnessCount).toBe(0);
    db.close();
  });

  it("newestEventTs excludes emitter='jaxos' rows (test 9 — the 'N ago' value must never come from a synthetic row)", () => {
    const db = openDb(":memory:");
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
      new Date("2026-08-27T10:00:00Z"),
    );
    // Newer than the agent-sourced row above, but synthetic — must not become newestEventTs.
    insertEvent(
      db,
      { run_id: null, project: "p1", role: "lead", type: "question-answered", source: "deterministic",
        emitter: "jaxos", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
      new Date("2026-08-27T10:30:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: "100:1" };
    const data = getHubData(db, live, {}, null, Date.parse("2026-08-27T10:35:00Z"));
    expect(data.byProject.p1.newestEventTs).toBe("2026-08-27T10:00:00.000Z");
    db.close();
  });

  it("a project literally named __proto__ appears in byProject, never swallowed into the prototype (branch review Finding 6)", () => {
    const db = openDb(":memory:");
    insertEvent(
      db,
      { run_id: null, project: "__proto__", role: "lead", type: "turn-started", source: "deterministic",
        emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" },
      new Date("2026-08-27T10:00:00Z"),
    );
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: "100:1" };
    const data = getHubData(db, live, {}, null, Date.parse("2026-08-27T10:05:00Z"));
    expect(Object.prototype.hasOwnProperty.call(data.byProject, "__proto__")).toBe(true);
    expect(data.byProject["__proto__"].panes).toHaveLength(1);
    db.close();
  });
});

describe("insertEvent — terminal claim on a second run-finished (spec §2.6 cold review G1)", () => {
  it("throws before inserting a second run-finished for the same run_id; the first row is unchanged", () => {
    const db = openDb(":memory:");
    insertEvent(db, started, NOW);
    const first = insertEvent(db, finished, NOW);
    expect(() => insertEvent(db, { ...finished, payload: { ...finished.payload, summary: "second" } }, NOW)).toThrow(
      RUN_ALREADY_FINISHED,
    );
    const rows = db.prepare("SELECT id, payload FROM workflow_events WHERE type = 'run-finished'").all() as { id: number; payload: string }[];
    expect(rows).toHaveLength(1);
    expect(rows[0].id).toBe(first.id);
    expect(JSON.parse(rows[0].payload).summary).toBe("ok");
    db.close();
  });
});

// spec §7.2 #39 (MOA-453). `jaxflow merge` POSTs this audit BEFORE it pushes, so §5.3's
// documented recovery re-runs the identical command and re-POSTs. The audit row must not
// double, and — unlike #33's terminal claim — the caller must NOT be refused: it still has a
// push to reach.
const mergeApproved = (over: Record<string, unknown> = {}): WorkflowEventInput => ({
  run_id: null, project: "p1", role: "lead", type: "merge-approved",
  source: "deterministic", emitter: "wrapper",
  payload: {
    phase: "Phase C", branch: "feat/x", sha: "a".repeat(40), target: "main",
    approved_by: "rafa", merge_sha: "c".repeat(40),
    ...over,
  },
});

describe("insertEvent — merge-approved is idempotent on (project, target, merge_sha) (spec §2.6, MOA-453)", () => {
  it("suppresses the duplicate, returns the stored row, and never throws", () => {
    const db = openDb(":memory:");
    const first = insertEvent(db, mergeApproved(), NOW);
    const second = insertEvent(db, mergeApproved(), new Date("2026-08-18T13:00:00.000Z"));
    expect(second.id).toBe(first.id);
    expect(second.ts).toBe(first.ts);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE type = 'merge-approved'").get())
      .toEqual({ n: 1 });
    db.close();
  });

  it("keeps the FIRST payload when a same-key retry differs in a non-key field (first wins)", () => {
    const db = openDb(":memory:");
    const first = insertEvent(db, mergeApproved(), NOW);
    // Input the producer cannot generate — a resume re-reads the merge commit's subject and
    // requires equality, so `phase` cannot differ between two real POSTs. The point is that
    // insertEvent owes a contract that does not DEPEND on that producer invariant.
    const second = insertEvent(db, mergeApproved({ phase: "Phase D" }), NOW);
    expect(second.id).toBe(first.id);
    const rows = db.prepare("SELECT id, payload FROM workflow_events WHERE type = 'merge-approved'")
      .all() as { id: number; payload: string }[];
    expect(rows).toHaveLength(1);
    expect(JSON.parse(rows[0].payload).phase).toBe("Phase C");
    db.close();
  });

  it.each<[string, Partial<WorkflowEventInput>, Record<string, unknown>]>([
    ["a different project", { project: "p2" }, {}],
    ["a different target", {}, { target: "release" }],
    ["a different merge_sha", {}, { merge_sha: "d".repeat(40) }],
  ])("inserts a second row for %s", (_name, evOver, payloadOver) => {
    const db = openDb(":memory:");
    insertEvent(db, mergeApproved(), NOW);
    insertEvent(db, { ...mergeApproved(payloadOver), ...evOver }, NOW);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE type = 'merge-approved'").get())
      .toEqual({ n: 2 });
    db.close();
  });

  it("does not suppress an unrelated event type that shares the project", () => {
    const db = openDb(":memory:");
    insertEvent(db, mergeApproved(), NOW);
    insertEvent(db, started, NOW);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 2 });
    db.close();
  });
});

const prOpened = (over: Record<string, unknown> = {}): WorkflowEventInput => ({
  run_id: null, project: "p1", role: "lead", type: "pr-opened",
  source: "deterministic", emitter: "wrapper",
  payload: {
    repo: "acme/route-converter-se", branch: "feat/x", sha: "a".repeat(40),
    base: "staging", pr_number: 7, pr_url: "https://github.com/acme/route-converter-se/pull/7",
    kind: "feature",
    ...over,
  },
});

describe("insertEvent — pr-opened is idempotent on (project, branch, sha) (spec MOA-465 Ledger & card)", () => {
  it("suppresses the duplicate, returns the stored row, and never throws", () => {
    const db = openDb(":memory:");
    const first = insertEvent(db, prOpened(), NOW);
    const second = insertEvent(db, prOpened(), new Date("2026-09-27T13:00:00.000Z"));
    expect(second.id).toBe(first.id);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE type = 'pr-opened'").get())
      .toEqual({ n: 1 });
    db.close();
  });

  it("keeps the FIRST payload when a same-key retry differs in a non-key field (first wins)", () => {
    const db = openDb(":memory:");
    const first = insertEvent(db, prOpened(), NOW);
    const second = insertEvent(db, prOpened({ pr_url: "https://github.com/acme/route-converter-se/pull/8" }), NOW);
    expect(second.id).toBe(first.id);
    const rows = db.prepare("SELECT id, payload FROM workflow_events WHERE type = 'pr-opened'")
      .all() as { id: number; payload: string }[];
    expect(rows).toHaveLength(1);
    expect(JSON.parse(rows[0].payload).pr_url).toBe("https://github.com/acme/route-converter-se/pull/7");
    db.close();
  });

  it.each<[string, Partial<WorkflowEventInput>, Record<string, unknown>]>([
    ["a different project", { project: "p2" }, {}],
    ["a different branch", {}, { branch: "feat/y" }],
    ["a different sha", {}, { sha: "b".repeat(40) }],
  ])("inserts a second row for %s (fast-forward refresh, decision 8)", (_name, evOver, payloadOver) => {
    const db = openDb(":memory:");
    insertEvent(db, prOpened(), NOW);
    insertEvent(db, { ...prOpened(payloadOver), ...evOver }, NOW);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE type = 'pr-opened'").get())
      .toEqual({ n: 2 });
    db.close();
  });

  it("does not suppress an unrelated event type that shares the project", () => {
    const db = openDb(":memory:");
    insertEvent(db, prOpened(), NOW);
    insertEvent(db, started, NOW);
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_events").get()).toEqual({ n: 2 });
    db.close();
  });
});

describe("MOA-462 — supersede inferred attention after session progress", () => {
  const SESSION = "session-462";
  const LIVE: LiveSnapshot = { incarnation: INCARNATION, paneCommands: new Map(), paneIds: new Set(["%1"]) };
  const RUN_AT = new Date(NOW.getTime() + 1000);

  function missionProject(over: Partial<Project> = {}): Project {
    return {
      dir: "p1", name: "p1", stage: "build", gate: null, legacyGateField: false,
      builder: "opencode-grok", branch: "main", updated: NOW.toISOString(),
      now: "Last completed milestone", residuals: [], statusMtime: NOW.toISOString(),
      archived: false,
      ...over,
    };
  }

  function readMission(
    db: ReturnType<typeof openDb>,
    live: LiveSnapshot,
    now: Date,
    project: Project = missionProject(),
  ) {
    const data = getHubData(db, live, {}, null, now.getTime());
    const { model } = buildMission(
      { ok: true, data: { projects: [project], skipped: 0, reposRoot: "" } },
      { ok: true, data }, undefined, now.getTime(),
    );
    return { data, model };
  }

  function originalPane(data: ReturnType<typeof getHubData>, project = "p1") {
    return data.byProject[project].panes.find((p) => p.pane === "%1" && p.tmuxIncarnation === INCARNATION)!;
  }

  function classifiedStop(
    db: ReturnType<typeof openDb>,
    status: "needs_input" | "blocked" = "needs_input",
    over: Partial<WorkflowEventInput> = {},
  ) {
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { message_tail: "Ambiguous final response." },
      ...over,
    }, NOW);
    classifyDeferred(db, stop.id, { status });
    return stop;
  }

  function matchingRun(
    db: ReturnType<typeof openDb>,
    over: Partial<WorkflowEventInput> = {},
    when = RUN_AT,
  ) {
    return insertEvent(db, {
      run_id: "moa462-run", project: "p1", role: "builder", type: "run-started",
      source: "deterministic", emitter: "wrapper", harness_session: SESSION,
      pane: null, tmux_incarnation: null,
      payload: { phase: "moa462", runtime: "opencode-grok", caller: "claude" },
      ...over,
    }, when);
  }

  function expectPreserved(
    db: ReturnType<typeof openDb>,
    stop: { id: number },
    original: unknown,
    status: string,
    rule: string,
  ) {
    expect(db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id)).toEqual(original);
    expect(getProjectTimeline(db, "p1").find((e) => e.id === stop.id)?.payload)
      .toMatchObject({ capsule_status: status, capsule_rule: rule });
  }

  it("projects waiting for a later claude same-session run and drops the inbox row", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: "session-462",
      payload: { message_tail: "Ambiguous final response." },
    }, NOW);
    classifyDeferred(db, stop.id, { status: "needs_input" });
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    const runAt = new Date(NOW.getTime() + 1000);
    insertEvent(db, {
      run_id: "moa462-run", project: "p1", role: "builder", type: "run-started",
      source: "deterministic", emitter: "wrapper", harness_session: "session-462",
      pane: null, tmux_incarnation: null,
      payload: { phase: "moa462", runtime: "opencode-grok", caller: "claude" },
    }, runAt);
    const live = { incarnation: INCARNATION, paneCommands: new Map(), paneIds: new Set(["%1"]) };
    const data = getHubData(db, live, {}, null, runAt.getTime());
    expect(data.byProject.p1.panes[0].capsule).toEqual({
      status: "waiting", minutes: null, declaredAt: runAt.toISOString(), eventId: stop.id, mergeAsk: null, question: null, answerable: true,
    });
    const project: Project = {
      dir: "p1", name: "p1", stage: "build", gate: null, legacyGateField: false,
      builder: "opencode-grok", branch: "main", updated: NOW.toISOString(),
      now: "Last completed milestone", residuals: [], statusMtime: NOW.toISOString(), archived: false,
    };
    const { model } = buildMission(
      { ok: true, data: { projects: [project], skipped: 0, reposRoot: "" } },
      { ok: true, data }, undefined, runAt.getTime(),
    );
    expect(model.cards[0].headline).toBe("waiting");
    expect(model.inbox).toHaveLength(0);
    expect(db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id)).toEqual(original);
    expect(getProjectTimeline(db, "p1").find(e => e.type === "turn-stopped")?.payload)
      .toMatchObject({ capsule_status: "needs_input", capsule_rule: "classified" });
    db.close();
  });

  it("a pane-carrying codex stop is not pane authority: no pane capsule, the native session carries run-in-flight (MOA-469 F1)", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "codex-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: "session-462",
      payload: { message_tail: "Ambiguous final response." },
    }, NOW);
    classifyDeferred(db, stop.id, { status: "needs_input" });
    const runAt = new Date(NOW.getTime() + 1000);
    insertEvent(db, {
      run_id: "moa462-run", project: "p1", role: "builder", type: "run-started",
      source: "deterministic", emitter: "wrapper", harness_session: "session-462",
      pane: null, tmux_incarnation: null,
      payload: { phase: "moa462", runtime: "opencode-grok", caller: "codex" },
    }, runAt);
    const live = { incarnation: INCARNATION, paneCommands: new Map(), paneIds: new Set(["%1"]) };
    const data = getHubData(db, live, {}, null, runAt.getTime());
    expect(data.byProject.p1.panes).toEqual([]); // the codex pane is never a HubPane
    const session = data.byProject.p1.codexSessions.find((s) => s.threadId === "session-462")!;
    expect(session.runInFlight).toBe(true); // exact-session correlation survives
    db.close();
  });

  it.each(["needs_input", "blocked"] as const)(
    "a later same-pane turn after classified %s yields working",
    (status) => {
      const db = openDb(":memory:");
      const stop = classifiedStop(db, status);
      const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
      insertEvent(db, {
        run_id: null, project: "p1", role: "lead", type: "turn-started",
        source: "deterministic", emitter: "claude-userprompt",
        pane: "%1", tmux_incarnation: INCARNATION, payload: {},
      }, RUN_AT);
      const { data, model } = readMission(db, LIVE, RUN_AT);
      expect(originalPane(data).capsule).toBeNull();
      expect(model.cards[0].headline).toBe("working");
      expect(model.inbox).toHaveLength(0);
      expectPreserved(db, stop, original, status, "classified");
      db.close();
    },
  );

  it("a classified needs_input stop plus a later matching run yields waiting", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { message_tail: "shall I proceed?" },
    }, NOW);
    // rung 6 is gone (D1) — the poller/Jev equivalent of a settled needs_input row is explicit.
    expect(classifyDeferred(db, stop.id, { status: "needs_input" })).toBe(true);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toEqual({
      status: "waiting", minutes: null, declaredAt: RUN_AT.toISOString(), eventId: stop.id, mergeAsk: null, question: null, answerable: true,
    });
    expect(model.cards[0].headline).toBe("waiting");
    expect(model.inbox).toHaveLength(0);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it.each(["ok", "missing", "cancelled", "invalid", undefined] as const)(
    "a finished matching run with contract_status %s does not resurrect attention",
    (contract_status) => {
      const db = openDb(":memory:");
      const stop = classifiedStop(db);
      const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
      matchingRun(db);
      const payload = contract_status === "cancelled"
        ? { phase: "moa462", exit_code: null, contract_status: "cancelled", report_path: null, summary: "cancelled", head_sha: null }
        : contract_status === undefined
          ? { phase: "moa462", exit_code: 0, report_path: "/r.md", summary: "ok", head_sha: "b".repeat(40) }
          : { phase: "moa462", exit_code: 0, contract_status, report_path: "/r.md", summary: "ok", head_sha: "b".repeat(40) };
      insertEvent(db, {
        run_id: "moa462-run", project: "p1", role: "builder", type: "run-finished",
        source: "deterministic", emitter: "wrapper", payload,
      }, RUN_AT);
      const { data, model } = readMission(db, LIVE, RUN_AT);
      expect(originalPane(data).capsule).toBeNull();
      expect(model.cards[0].headline).toBe("idle");
      expect(model.inbox).toHaveLength(0);
      expectPreserved(db, stop, original, "needs_input", "classified");
      db.close();
    },
  );

  it("two later matching runs keep waiting on the older active one when the newest finished", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    const newerAt = new Date(NOW.getTime() + 2000);
    matchingRun(db, { run_id: "moa462-run-old" }, RUN_AT);
    matchingRun(db, { run_id: "moa462-run-new" }, newerAt);
    insertEvent(db, {
      run_id: "moa462-run-new", project: "p1", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: { phase: "moa462", exit_code: 1, contract_status: "invalid", report_path: "/r.md", summary: "no", head_sha: "b".repeat(40) },
    }, newerAt);
    const { data, model } = readMission(db, LIVE, newerAt);
    expect(originalPane(data).capsule).toEqual({
      status: "waiting", minutes: null, declaredAt: RUN_AT.toISOString(), eventId: stop.id, mergeAsk: null, question: null, answerable: true,
    });
    expect(model.cards[0].headline).toBe("waiting");
    expect(model.inbox).toHaveLength(0);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("read-time supersession still works when the classifier settles after the run", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { message_tail: "Ambiguous final response." },
    }, NOW);
    matchingRun(db);
    expect(classifyDeferred(db, stop.id, { status: "needs_input" })).toBe(true);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toEqual({
      status: "waiting", minutes: null, declaredAt: RUN_AT.toISOString(), eventId: stop.id, mergeAsk: null, question: null, answerable: true,
    });
    expect(model.cards[0].headline).toBe("waiting");
    expect(model.inbox).toHaveLength(0);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("a newer inferred stop after the run keeps needs-you", () => {
    const db = openDb(":memory:");
    const first = classifiedStop(db);
    const firstOriginal = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(first.id);
    matchingRun(db, {
      payload: { phase: "moa462", runtime: "opencode-grok", caller: "codex" },
    });
    const laterAt = new Date(NOW.getTime() + 2000);
    // claude-stop (not codex-stop): the run is caller=codex, so a codex-stop would pair with it
    // and legitimately read waiting after Task 0 — this test is about SUPERSESSION of the earlier
    // needs-you, so the later stop must stay unpaired.
    const later = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { message_tail: "Ambiguous final response." },
    }, laterAt);
    expect(classifyDeferred(db, later.id, { status: "needs_input" })).toBe(true);
    const laterOriginal = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(later.id);
    const { data, model } = readMission(db, LIVE, laterAt);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toEqual([
      expect.objectContaining({ kind: "attention", status: "needs_input", pane: "%1" }),
    ]);
    expectPreserved(db, first, firstOriginal, "needs_input", "classified");
    expectPreserved(db, later, laterOriginal, "needs_input", "classified");
    db.close();
  });

  it.each([
    ["other project", { project: "p2" } satisfies Partial<WorkflowEventInput>],
    ["other session", { harness_session: "session-other" } satisfies Partial<WorkflowEventInput>],
    ["other caller", { payload: { phase: "moa462", runtime: "opencode-grok", caller: "codex" } } satisfies Partial<WorkflowEventInput>],
  ])("retains inferred attention when the later run has %s", (_label, over) => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db, over);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toEqual([
      expect.objectContaining({ kind: "attention", status: "needs_input" }),
    ]);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("retains inferred attention when the stop has no harness_session", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db, "needs_input", { harness_session: null });
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toHaveLength(1);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("retains inferred attention when the run omits harness_session and caller_session", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    insertEvent(db, {
      run_id: "moa462-run", project: "p1", role: "builder", type: "run-started",
      source: "deterministic", emitter: "wrapper",
      pane: null, tmux_incarnation: null,
      payload: { phase: "moa462", runtime: "opencode-grok", caller: "claude" },
    }, RUN_AT);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toHaveLength(1);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("retains inferred attention when the run has no payload.caller", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db, { payload: { phase: "moa462", runtime: "opencode-grok" } });
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toHaveLength(1);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("a wrapper-style run that only carries payload.caller_session still yields waiting", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    insertEvent(db, {
      run_id: "moa462-run", project: "p1", role: "builder", type: "run-started",
      source: "deterministic", emitter: "wrapper",
      pane: null, tmux_incarnation: null,
      payload: { phase: "moa462", runtime: "opencode-grok", caller: "claude", caller_session: SESSION },
    }, RUN_AT);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toEqual({
      status: "waiting", minutes: null, declaredAt: RUN_AT.toISOString(), eventId: stop.id, mergeAsk: null, question: null, answerable: true,
    });
    expect(model.cards[0].headline).toBe("waiting");
    expect(model.inbox).toHaveLength(0);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("does not borrow waiting when the live snapshot has a new incarnation", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, { incarnation: "999:1", paneCommands: new Map(), paneIds: new Set(["%1"]) }, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" }); // stored capsule retained
    // MOA-469 live correction: an expired-incarnation pane is dead, so its stored needs_input is
    // retained but no longer surfaces as current card attention or an inbox row.
    expect(originalPane(data).live).toBe(false);
    expect(model.cards[0].headline).toBe("idle");
    expect(model.inbox).toHaveLength(0);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("does not borrow waiting after the current pane switches project", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    upsertSession(db, {
      project: "p2", session: "main", pane: "%1", role: "lead", tmux_incarnation: INCARNATION,
    }, new Date(NOW.getTime() + 2000));
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" }); // stored capsule retained
    expect(originalPane(data).live).toBe(false);
    // MOA-469 live correction: the pane left for p2 — its stale p1 capsule is no longer current.
    expect(model.cards[0].headline).toBe("idle");
    expect(model.inbox).toHaveLength(0);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it.each([
    ["another pane", { pane: "%2", tmux_incarnation: INCARNATION, project: "p1" }],
    ["another incarnation", { pane: "%1", tmux_incarnation: "other-inc", project: "p1" }],
  ])("retains inferred attention when a later turn-started is on %s", (_label, over) => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    insertEvent(db, {
      run_id: null, role: "lead", type: "turn-started",
      source: "deterministic", emitter: "claude-userprompt", payload: {},
      ...over,
    }, RUN_AT);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox.some((r) => r.kind === "attention")).toBe(true);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("a later turn-started that moved the pane to another project keeps the stored capsule but the pane reads dead for p1 (MOA-469 live correction)", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    insertEvent(db, {
      run_id: null, role: "lead", type: "turn-started",
      source: "deterministic", emitter: "claude-userprompt", payload: {},
      pane: "%1", tmux_incarnation: INCARNATION, project: "p2",
    }, RUN_AT);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" }); // stored capsule retained
    expect(originalPane(data).live).toBe(false); // the pane left for p2 — dead for p1
    expect(model.cards[0].headline).toBe("idle"); // a dead pane carries no current attention
    expect(model.inbox.filter((r) => r.kind === "attention" || r.kind === "question")).toHaveLength(0);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("retains tagged attention despite a matching newer run", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "deterministic", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { capsule_status: "needs_input", capsule_rule: "tag" },
    }, NOW);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toHaveLength(1);
    expectPreserved(db, stop, original, "needs_input", "tag");
    db.close();
  });

  it("retains question_open attention despite a matching newer run", () => {
    const db = openDb(":memory:");
    insertEvent(db, question("t-462"), NOW);
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { message_tail: "Ambiguous final response." },
    }, NOW);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox.some((r) => r.kind === "attention")).toBe(true);
    expectPreserved(db, stop, original, "needs_input", "question_open");
    db.close();
  });

  it("retains attention-rule blocked despite a matching newer run", () => {
    const db = openDb(":memory:");
    insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "attention-needed",
      source: "behavioral", emitter: "claude-notification",
      pane: "%1", tmux_incarnation: INCARNATION, payload: { reason: "agent_needs_input" },
    }, NOW);
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { message_tail: "Ambiguous final response." },
    }, NOW);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "blocked" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toEqual([
      expect.objectContaining({ kind: "attention", status: "blocked" }),
    ]);
    expectPreserved(db, stop, original, "blocked", "attention");
    db.close();
  });

  it("retains a missing-rule stop despite a matching newer run", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { message_tail: "Ambiguous final response." },
    }, NOW);
    db.prepare(
      "UPDATE workflow_events SET payload = json_set(json_remove(payload, '$.capsule_rule'), '$.capsule_status', 'needs_input') WHERE id = ?",
    ).run(stop.id);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toHaveLength(1);
    expect(db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id)).toEqual(original);
    expect(getProjectTimeline(db, "p1").find((e) => e.id === stop.id)?.payload)
      .toMatchObject({ capsule_status: "needs_input" });
    db.close();
  });

  it("retains an unknown legacy rule despite a matching newer run", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop",
      pane: "%1", tmux_incarnation: INCARNATION, harness_session: SESSION,
      payload: { message_tail: "Ambiguous final response." },
    }, NOW);
    db.prepare(
      "UPDATE workflow_events SET payload = json_set(payload, '$.capsule_status', 'needs_input', '$.capsule_rule', 'legacy_unknown') WHERE id = ?",
    ).run(stop.id);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toMatchObject({ status: "needs_input" });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox).toHaveLength(1);
    expectPreserved(db, stop, original, "needs_input", "legacy_unknown");
    db.close();
  });

  it("a matching active run plus a later pending question keeps needs-you and the question", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    insertEvent(db, question("t-462-late"), new Date(NOW.getTime() + 2000));
    const { data, model } = readMission(db, LIVE, RUN_AT);
    expect(originalPane(data).capsule).toEqual({
      status: "waiting", minutes: null, declaredAt: RUN_AT.toISOString(), eventId: stop.id, mergeAsk: null, question: null, answerable: true,
    });
    expect(model.cards[0].headline).toBe("needs-you");
    expect(model.inbox.some((r) => r.kind === "question")).toBe(true);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });

  it("a matching active run plus a project gate reads waiting: the lead owes nothing while its own run is in flight (2026-09-24, arc)", () => {
    const db = openDb(":memory:");
    const stop = classifiedStop(db);
    const original = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(stop.id);
    matchingRun(db);
    const { data, model } = readMission(db, LIVE, RUN_AT, missionProject({ gate: "awaiting-approval" }));
    expect(originalPane(data).capsule).toEqual({
      status: "waiting", minutes: null, declaredAt: RUN_AT.toISOString(), eventId: stop.id, mergeAsk: null, question: null, answerable: true,
    });
    expect(model.cards[0].headline).toBe("waiting");
    expect(model.inbox.some((r) => r.kind === "gate")).toBe(false);
    expect(model.inbox.some((r) => r.kind === "attention")).toBe(false);
    expectPreserved(db, stop, original, "needs_input", "classified");
    db.close();
  });
});

describe("MOA-464 — unfinished run hub projection", () => {
  const T_BUILD = new Date("2026-09-09T10:00:00.000Z");
  const T_DIFF = new Date("2026-09-09T10:02:00.000Z");
  const HUB_NOW = Date.parse("2026-09-09T12:00:00.000Z");
  const LIVE: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(), incarnation: null };

  function start(
    db: ReturnType<typeof openDb>,
    over: Partial<WorkflowEventInput>,
    payload: Record<string, unknown>,
    when: Date,
  ) {
    return insertEvent(db, {
      run_id: "r1", project: "p1", role: "builder", type: "run-started",
      source: "deterministic", emitter: "wrapper", pane: null, tmux_incarnation: null,
      ...over,
      payload: { phase: "B", runtime: "codex", kind: "build", target: "feat/old", ...payload },
    }, when);
  }

  function finish(
    db: ReturnType<typeof openDb>,
    over: Partial<WorkflowEventInput>,
    payload: Record<string, unknown>,
    when: Date,
  ) {
    return insertEvent(db, {
      run_id: "r1", project: "p1", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      ...over,
      payload: {
        phase: "B", exit_code: 0, contract_status: "ok", report_path: "/r.md",
        summary: "ok", head_sha: "b".repeat(40), result: "success",
        ...payload,
      },
    }, when);
  }

  function hub(db: ReturnType<typeof openDb>) {
    return getHubData(db, LIVE, {}, null, HUB_NOW);
  }

  it("selects a newer pane-less diff over a completed build", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "build1" }, { kind: "build", runtime: "codex", target: "feat/old" }, T_BUILD);
    finish(db, { run_id: "build1" }, {}, T_BUILD);
    start(db, { run_id: "diff1", role: "reviewer" }, { kind: "diff", runtime: "claude", target: "fix/example" }, T_DIFF);
    const data = hub(db);
    expect(data.byProject.p1.activeRun).toEqual({
      runId: "diff1", startedAt: "2026-09-09T10:02:00.000Z", kind: "diff",
      runtime: "claude", target: "fix/example", count: 1, lastStep: null,
    });
    expect(data.byProject.p1.activeRun?.target).not.toBe("feat/old");
    expect(data.byProject.p1.activeRun?.runtime).not.toBe("codex");
    const project: Project = {
      dir: "p1", name: "p1", stage: "build", gate: null, legacyGateField: false,
      builder: "codex", branch: "feat/old-build", updated: "2026-09-09T09:00:00.000Z",
      now: "Completed build 88015f165499", residuals: [], statusMtime: "2026-09-09T09:00:00.000Z", archived: false,
    };
    const { model } = buildMission(
      { ok: true, data: { projects: [project], skipped: 0, reposRoot: "" } },
      { ok: true, data },
      { ok: true, data: { p1: { ok: true, prs: [], truncated: false } } },
      HUB_NOW,
    );
    expect(model.cards[0].activeRun).toMatchObject({
      runId: "diff1", kind: "diff", stage: "review", descriptionKey: "reviewDiff",
      targetKind: "branch", target: "fix/example", runtime: "claude", count: 1,
    });
    expect(model.cards[0].stage).toBe("build");
    expect(model.cards[0].branch).toBe("feat/old-build");
    expect(model.cards[0].builder).toBe("codex");
    expect(model.cards[0].now).toBe("Completed build 88015f165499");
    expect(model.cards[0].headline).toBe("idle");
    db.close();
  });

  it.each([
    ["build", "builder", "fix/example", "fix/example"],
    ["diff", "reviewer", "fix/example", "fix/example"],
    ["spec", "reviewer", "/home/rafa/repos/jax-os/.local/docs/specs/foo.md", "foo.md"],
    ["plan", "reviewer", "/home/rafa/repos/jax-os/.local/docs/plans/bar.md", "bar.md"],
  ] as const)("projects kind %s target as %s", (kind, role, target, displayed) => {
    const db = openDb(":memory:");
    start(db, { run_id: kind, role }, { kind, runtime: "claude", target }, T_DIFF);
    expect(hub(db).byProject.p1.activeRun).toMatchObject({ kind, runtime: "claude", target: displayed, count: 1 });
    db.close();
  });

  it("counts distinct unfinished runs and selects the newest start id", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "old" }, { kind: "build", runtime: "codex", target: "old-branch" }, T_BUILD);
    start(db, { run_id: "new", role: "reviewer" }, { kind: "diff", runtime: "claude", target: "new-branch" }, T_DIFF);
    expect(hub(db).byProject.p1.activeRun).toMatchObject({
      runId: "new", kind: "diff", target: "new-branch", count: 2,
    });
    db.close();
  });

  it("activeRuns lists every unfinished run (newest start id first), excluding a finished one", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "old" }, { kind: "build", runtime: "codex", target: "old-branch" }, T_BUILD);
    start(db, { run_id: "new", role: "reviewer" }, { kind: "diff", runtime: "claude", target: "new-branch" }, T_DIFF);
    start(db, { run_id: "done", role: "reviewer" }, { kind: "diff", runtime: "claude", target: "done-branch" }, T_BUILD);
    finish(db, { run_id: "done", role: "reviewer" }, {}, T_DIFF);
    expect(hub(db).byProject.p1.activeRuns).toEqual([
      { runId: "new", role: "reviewer", kind: "diff", runtime: "claude", target: "new-branch", startedAt: "2026-09-09T10:02:00.000Z" },
      { runId: "old", role: "builder", kind: "build", runtime: "codex", target: "old-branch", startedAt: "2026-09-09T10:00:00.000Z" },
    ]);
    db.close();
  });

  it("duplicate starts keep only the newest start row for that run id", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "dup" }, { kind: "build", runtime: "codex", target: "first" }, T_BUILD);
    start(db, { run_id: "dup" }, { kind: "build", runtime: "claude", target: "second" }, T_DIFF);
    expect(hub(db).byProject.p1.activeRun).toEqual({
      runId: "dup", startedAt: "2026-09-09T10:02:00.000Z", kind: "build",
      runtime: "claude", target: "second", count: 1, lastStep: null,
    });
    db.close();
  });

  it("orders by start row id, not timestamp", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "later-ts" }, { kind: "build", runtime: "codex", target: "later" }, new Date("2026-09-09T10:10:00.000Z"));
    start(db, { run_id: "earlier-ts" }, { kind: "diff", runtime: "claude", target: "earlier" }, new Date("2026-09-09T10:00:00.000Z"));
    expect(hub(db).byProject.p1.activeRun).toMatchObject({ runId: "earlier-ts", target: "earlier" });
    db.close();
  });

  it("finishing the newest run leaves the older active one selected", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "old" }, { kind: "build", runtime: "codex", target: "old-branch" }, T_BUILD);
    start(db, { run_id: "new", role: "reviewer" }, { kind: "diff", runtime: "claude", target: "new-branch" }, T_DIFF);
    finish(db, { run_id: "new" }, {}, T_DIFF);
    expect(hub(db).byProject.p1.activeRun).toMatchObject({ runId: "old", kind: "build", target: "old-branch", count: 1 });
    db.close();
  });

  it.each([
    ["success", { contract_status: "ok", result: "success" }],
    ["failed contract", { contract_status: "invalid", result: undefined, exit_code: 1 }],
    ["cancelled", { contract_status: "cancelled", result: undefined, exit_code: null, report_path: null }],
  ])("any run-finished (%s) closes that project/run pair", (_name, payload) => {
    const db = openDb(":memory:");
    start(db, { run_id: "done" }, { kind: "build", runtime: "codex", target: "feat/x" }, T_BUILD);
    finish(db, { run_id: "done" }, payload, T_DIFF);
    expect(hub(db).byProject.p1.activeRun).toBeNull();
    db.close();
  });

  it.each(["missing", "invalid"])("ingests a failure with a %s report and closes the active run", (contract_status) => {
    const db = openDb(":memory:");
    try {
      start(db, {}, {}, T_BUILD);
      const parsed = parseIngress({ ...finished, payload: { ...finished.payload, contract_status, result: "failure", exit_code: 1, tail: "Connection reset by server", reason: "crash" } });
      expect(parsed.ok).toBe(true);
      if (!parsed.ok) throw new Error(parsed.error);
      expect(insertEvent(db, parsed.event, T_DIFF).payload.result).toBe("failure");
      expect(hub(db).byProject.p1.activeRun).toBeNull();
    } finally { db.close(); }
  });

  it("a terminal in another project never closes p1, and a null run_id start is ignored", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "shared" }, { kind: "diff", runtime: "claude", target: "p1-branch" }, T_BUILD);
    start(db, { run_id: "shared", project: "p2" }, { kind: "diff", runtime: "codex", target: "p2-branch" }, T_BUILD);
    finish(db, { run_id: "shared", project: "p2" }, {}, T_DIFF);
    start(db, { run_id: null }, { kind: "build", runtime: "claude", target: "null-id" }, T_DIFF);
    expect(hub(db).byProject.p1.activeRun).toMatchObject({ runId: "shared", target: "p1-branch", count: 1 });
    db.close();
  });

  it("selection survives more than 20 later timeline events and needs no pane", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "diff1", role: "reviewer" }, { kind: "diff", runtime: "claude", target: "fix/example" }, T_DIFF);
    for (let i = 0; i < 21; i++) {
      insertEvent(db, {
        run_id: null, project: "p1", role: "lead", type: "turn-started",
        source: "deterministic", emitter: "claude-userprompt", payload: {},
        pane: "%1", tmux_incarnation: "1:1",
      }, new Date(T_DIFF.getTime() + (i + 1) * 1000));
    }
    const data = hub(db);
    expect(data.byProject.p1.activeRun).toMatchObject({ runId: "diff1", target: "fix/example" });
    expect(data.byProject.p1.timeline).toHaveLength(20);
    expect(data.byProject.p1.timeline.every((e) => e.type === "turn-started")).toBe(true);
    db.close();
  });

  it("no unfinished starts yields explicit null", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "done" }, { kind: "build", runtime: "codex", target: "feat/x" }, T_BUILD);
    finish(db, { run_id: "done" }, {}, T_DIFF);
    expect(hub(db).byProject.p1.activeRun).toBeNull();
    db.close();
  });

  it("unknown kind/runtime and bad targets degrade fields without throwing", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "bad-kind" }, { kind: "nonsense", runtime: "gemini", target: "keep-me" }, T_BUILD);
    expect(hub(db).byProject.p1.activeRun).toMatchObject({
      runId: "bad-kind", kind: null, runtime: null, target: null, count: 1,
    });
    db.close();
  });

  it("oversized, control-character, missing, and trailing-slash spec targets omit target", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "oversize" }, { kind: "build", runtime: "claude", target: "x".repeat(LIMITS.target + 1) }, T_BUILD);
    expect(hub(db).byProject.p1.activeRun?.target).toBeNull();
    db.close();

    const db2 = openDb(":memory:");
    start(db2, { run_id: "ctrl" }, { kind: "build", runtime: "claude", target: "fix/\u0001branch" }, T_BUILD);
    expect(hub(db2).byProject.p1.activeRun?.target).toBeNull();
    db2.close();

    const db3 = openDb(":memory:");
    start(db3, { run_id: "missing" }, { kind: "diff", runtime: "claude", target: "" }, T_BUILD);
    expect(hub(db3).byProject.p1.activeRun?.target).toBeNull();
    db3.close();

    const db4 = openDb(":memory:");
    start(db4, { run_id: "slash", role: "reviewer" }, { kind: "spec", runtime: "claude", target: "docs/specs/foo.md/" }, T_BUILD);
    expect(hub(db4).byProject.p1.activeRun).toMatchObject({ kind: "spec", target: null });
    db4.close();
  });

  it("invalid timestamp omits startedAt", () => {
    const db = openDb(":memory:");
    const row = start(db, { run_id: "ts" }, { kind: "diff", runtime: "claude", target: "fix/x" }, T_DIFF);
    db.prepare("UPDATE workflow_events SET ts = ? WHERE id = ?").run("not-a-date", row.id);
    expect(hub(db).byProject.p1.activeRun).toMatchObject({ runId: "ts", startedAt: null, target: "fix/x" });
    db.close();
  });

  it("malformed and nonobject JSON degrade to null fields and keep the count", () => {
    const db = openDb(":memory:");
    const bad = start(db, { run_id: "malformed" }, { kind: "diff", runtime: "claude", target: "fix/x" }, T_DIFF);
    db.prepare("UPDATE workflow_events SET payload = ? WHERE id = ?").run("{not json", bad.id);
    const badBefore = db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(bad.id);
    const data = hub(db);
    expect(data.byProject.p1.activeRun).toMatchObject({
      runId: "malformed", kind: null, runtime: null, target: null, count: 1,
      startedAt: "2026-09-09T10:02:00.000Z",
    });
    expect(data.byProject.p1.timeline).toEqual([
      expect.objectContaining({ id: bad.id, type: "run-started", payload: {} }),
    ]);
    expect(db.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(bad.id)).toEqual(badBefore);
    db.close();

    const db2 = openDb(":memory:");
    const arr = start(db2, { run_id: "array" }, { kind: "diff", runtime: "claude", target: "fix/x" }, T_DIFF);
    db2.prepare("UPDATE workflow_events SET payload = ? WHERE id = ?").run("[1,2]", arr.id);
    const arrBefore = db2.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(arr.id);
    const arrData = hub(db2);
    expect(arrData.byProject.p1.activeRun).toMatchObject({
      runId: "array", kind: null, runtime: null, target: null, count: 1,
    });
    expect(arrData.byProject.p1.timeline[0]).toMatchObject({ id: arr.id, payload: {} });
    expect(db2.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(arr.id)).toEqual(arrBefore);
    db2.close();

    const db3 = openDb(":memory:");
    const str = start(db3, { run_id: "string" }, { kind: "diff", runtime: "claude", target: "fix/x" }, T_DIFF);
    db3.prepare("UPDATE workflow_events SET payload = ? WHERE id = ?").run('"scalar"', str.id);
    const strBefore = db3.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(str.id);
    const strData = hub(db3);
    expect(strData.byProject.p1.activeRun).toMatchObject({
      runId: "string", kind: null, runtime: null, target: null, count: 1,
    });
    expect(strData.byProject.p1.timeline[0]).toMatchObject({ id: str.id, payload: {} });
    expect(db3.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(str.id)).toEqual(strBefore);
    db3.close();

    const db4 = openDb(":memory:");
    const jsonNull = start(db4, { run_id: "json-null" }, { kind: "diff", runtime: "claude", target: "fix/x" }, T_DIFF);
    db4.prepare("UPDATE workflow_events SET payload = ? WHERE id = ?").run("null", jsonNull.id);
    const nullBefore = db4.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(jsonNull.id);
    const nullData = hub(db4);
    expect(nullData.byProject.p1.activeRun).toMatchObject({
      runId: "json-null", kind: null, runtime: null, target: null, count: 1,
    });
    expect(nullData.byProject.p1.timeline[0]).toMatchObject({ id: jsonNull.id, payload: {} });
    expect(db4.prepare("SELECT payload FROM workflow_events WHERE id = ?").get(jsonNull.id)).toEqual(nullBefore);
    db4.close();
  });

  it("timeline keeps public fields, drops local-only keys, and does not repair storage", () => {
    const db = openDb(":memory:");
    insertEvent(db, {
      type: "turn-stopped", role: "lead", emitter: "claude-stop", source: "deterministic",
      run_id: null, pane: null, tmux_incarnation: null, project: "p1",
      payload: { capsule_status: "done", message_tail: "private-sentinel" },
    }, T_DIFF);
    const before = db.prepare("SELECT id, ts, type, payload FROM workflow_events ORDER BY id").all();
    const data = hub(db);
    expect(JSON.stringify(data.byProject.p1.timeline)).not.toContain("private-sentinel");
    expect(data.byProject.p1.timeline[0].payload).toMatchObject({
      capsule_status: "unknown", capsule_rule: "deferred",
    });
    expect(data.byProject.p1.timeline[0].payload).not.toHaveProperty("message_tail");
    expect(db.prepare("SELECT id, ts, type, payload FROM workflow_events ORDER BY id").all()).toEqual(before);
    db.close();
  });

  it("listPending treats scalar and array payloads as empty objects", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const row = insertEvent(db, { ...finished, run_id: "rowof" }, T_DIFF);
    db.prepare("UPDATE workflow_events SET payload = ? WHERE id = ?").run("[1,2]", row.id);
    expect(listPending(db, 10)[0].payload).toEqual({});
    db.prepare("UPDATE workflow_events SET payload = ? WHERE id = ?").run('"scalar"', row.id);
    expect(listPending(db, 10)[0].payload).toEqual({});
    db.prepare("UPDATE workflow_events SET payload = ? WHERE id = ?").run("null", row.id);
    expect(listPending(db, 10)[0].payload).toEqual({});
    db.close();
  });

  it("exports only HubActiveRun keys and never sentinel payload fields", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "priv" }, {
      kind: "diff", runtime: "claude", target: "fix/example",
      caller_session: "sess-secret", commands: ["rm -rf /"], prompts: "do not leak",
    }, T_DIFF);
    const active = hub(db).byProject.p1.activeRun;
    expect(active && Object.keys(active).sort()).toEqual(
      ["count", "kind", "lastStep", "runId", "runtime", "startedAt", "target"].sort(),
    );
    expect(active).not.toHaveProperty("caller_session");
    expect(active).not.toHaveProperty("commands");
    expect(active).not.toHaveProperty("prompts");
    db.close();
  });

  it("the read does not rewrite stored events or the timeline", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "diff1", role: "reviewer" }, { kind: "diff", runtime: "claude", target: "fix/example" }, T_DIFF);
    const rowsBefore = db.prepare("SELECT id, ts, type, payload FROM workflow_events ORDER BY id").all();
    const timelineBefore = getProjectTimeline(db, "p1");
    hub(db);
    expect(db.prepare("SELECT id, ts, type, payload FROM workflow_events ORDER BY id").all()).toEqual(rowsBefore);
    expect(getProjectTimeline(db, "p1")).toEqual(timelineBefore);
    db.close();
  });

  it("lastRunFor returns null with no run-finished row yet", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, {}, T_BUILD);
    expect(hub(db).byProject.p1.lastRun).toBeNull();
    db.close();
  });

  it("lastRunFor joins kind/runtime/target from the matching run-started row", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, { kind: "build", runtime: "codex", target: "feat/old" }, T_BUILD);
    finish(db, { run_id: "r1" }, { result: "success" }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun).toEqual({
      runId: "r1", role: "builder", kind: "build", runtime: "codex", target: "feat/old",
      finishedAt: "2026-09-09T10:02:00.000Z", outcome: "success", contractStatus: "ok",
      stage: null, diagnostic: null, headSha: "b".repeat(40), findings: null,
      runtimeModel: null, profileName: null, reportRel: null, targetRel: null,
      verifyCommand: null, buildCommand: null,
    });
    db.close();
  });

  it("carries findings through only when the stored payload has a valid one", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1", role: "reviewer" }, { kind: "diff", runtime: "codex", target: "feat/old" }, T_BUILD);
    finish(db, { run_id: "r1", role: "reviewer" }, { verdict: "approve-with-changes", findings: { high: 0, medium: 2, low: 1 } }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun?.findings).toEqual({ high: 0, medium: 2, low: 1 });
    db.close();
  });

  it("reports findings: null when the field is absent", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, {}, T_BUILD);
    finish(db, { run_id: "r1" }, { result: "success" }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun?.findings).toBeNull();
    db.close();
  });

  it("reports findings: null for a stored row with a malformed shape (bypassing ingress, e.g. a legacy row)", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1", role: "reviewer" }, { kind: "diff", runtime: "codex", target: "feat/old" }, T_BUILD);
    db.prepare(
      "INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery) VALUES (?, ?, 'p1', 'reviewer', 'run-finished', 'deterministic', 'wrapper', ?, 'delivered')",
    ).run(
      "2026-09-09T10:02:00.000Z", "r1",
      JSON.stringify({ phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x/r.md", summary: "s", verdict: "approve", findings: { high: 1000, medium: 0, low: 0 } }),
    );
    expect(hub(db).byProject.p1.lastRun?.findings).toBeNull();
    db.close();
  });

  it("reports findings: null for a builder row that carries one (round-1 plan review F4 — findings is gated on role, not just shape)", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, {}, T_BUILD);
    finish(db, { run_id: "r1" }, { findings: { high: 1, medium: 0, low: 0 } }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun?.findings).toBeNull();
    db.close();
  });

  it("reports findings: null for a reviewer row whose contract_status is not ok, even with a well-shaped findings (round-1 plan review F4)", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1", role: "reviewer" }, { kind: "diff", runtime: "codex", target: "feat/old" }, T_BUILD);
    finish(db, { run_id: "r1", role: "reviewer" }, {
      contract_status: "invalid", findings: { high: 1, medium: 0, low: 0 },
    }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun?.findings).toBeNull();
    db.close();
  });

  it("a reviewer verdict is surfaced only on an ok report (diff review 754d9e56254d F1)", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1", role: "reviewer" }, { kind: "diff", runtime: "codex", target: "feat/old" }, T_BUILD);
    // Bypass ingress on purpose: a stored row (legacy or otherwise) that carries a verdict on
    // an invalid report must still read back as "no verdict".
    db.prepare("INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery) VALUES (?, ?, 'p1', 'reviewer', 'run-finished', 'deterministic', 'wrapper', ?, 'delivered')")
      .run("2026-09-09T10:02:00.000Z", "r1", JSON.stringify({ phase: "B", exit_code: 1, contract_status: "invalid", report_path: "/x/r.md", summary: "s", verdict: "approve" }));
    expect(hub(db).byProject.p1.lastRun?.outcome).toBeNull();
    db.close();
  });

  it("hidden once a newer run-started exists, even before that run finishes", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, {}, T_BUILD);
    finish(db, { run_id: "r1" }, { result: "success" }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun).not.toBeNull();
    start(db, { run_id: "r2" }, { kind: "diff", runtime: "claude", target: "fix/x" }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun).toBeNull();
    db.close();
  });

  it("carries stage/diagnostic/contractStatus through for a failing builder row", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, {}, T_BUILD);
    finish(db, { run_id: "r1" }, {
      contract_status: "ok", result: "failure", stage: "runtime",
      diagnostic: "APIError 403 API key budget limit exceeded",
    }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun).toMatchObject({
      outcome: "failure", contractStatus: "ok", stage: "runtime",
      diagnostic: "APIError 403 API key budget limit exceeded",
    });
    db.close();
  });

  it("a reviewer row reads verdict, never result, and omits headSha", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1", role: "reviewer" }, { kind: "diff" }, T_BUILD);
    finish(db, { run_id: "r1", role: "reviewer" }, {
      result: undefined, verdict: "approve-with-changes", head_sha: undefined,
    }, T_DIFF);
    const lastRun = hub(db).byProject.p1.lastRun;
    expect(lastRun).toMatchObject({ role: "reviewer", outcome: "approve-with-changes", headSha: null });
    db.close();
  });

  it("a legacy row with no result key at all reads outcome null (§Backward compatibility)", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, {}, T_BUILD);
    finish(db, { run_id: "r1" }, { result: undefined, contract_status: "missing" }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun).toMatchObject({ outcome: null, contractStatus: "missing", stage: null, diagnostic: null });
    db.close();
  });

  it("an interrupted row is never overwritten by an older run-started target", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, { kind: "build", runtime: "opencode-builder", target: "feat/x" }, T_BUILD);
    finish(db, { run_id: "r1" }, {
      contract_status: "interrupted", result: "failure", exit_code: undefined, report_path: undefined,
      stage: "worker", diagnostic: "worker interrupted by SIGHUP", head_sha: null,
    }, T_DIFF);
    expect(hub(db).byProject.p1.lastRun).toMatchObject({
      contractStatus: "interrupted", outcome: "failure", stage: "worker",
      diagnostic: "worker interrupted by SIGHUP", kind: "build", runtime: "opencode-builder", target: "feat/x",
    });
    db.close();
  });

  it("exports only HubLastRun keys and never a sentinel run-started payload field", () => {
    const db = openDb(":memory:");
    start(db, { run_id: "r1" }, { caller_session: "sess-secret", commands: ["rm -rf /"] }, T_BUILD);
    finish(db, { run_id: "r1" }, {}, T_DIFF);
    const lastRun = hub(db).byProject.p1.lastRun;
    expect(lastRun && Object.keys(lastRun).sort()).toEqual(
      ["contractStatus", "diagnostic", "findings", "finishedAt", "headSha", "kind", "outcome", "role", "runId", "runtime", "stage", "target", "runtimeModel", "profileName", "reportRel", "targetRel", "verifyCommand", "buildCommand"].sort(),
    );
    expect(lastRun).not.toHaveProperty("caller_session");
    db.close();
  });
});

describe("MOA-469 — native Codex session derivation", () => {
  const UUID_A = "0191f0aa-bbbb-7000-8000-00000000000a";
  const UUID_B = "0191f0aa-bbcc-7000-8000-00000000000b";
  const codexEvent = (over: Partial<WorkflowEventInput> = {}): WorkflowEventInput => ({
    run_id: null, project: "p1", role: "lead", type: "turn-stopped",
    source: "behavioral", emitter: "codex-stop", harness_session: UUID_A,
    payload: { message_tail: "prose" }, pane: null, tmux_incarnation: null, ...over,
  });
  type RawThread = { threadId?: string; cwd?: string; owner?: string | null; source?: string; status?: string; activeFlags?: string[]; canAcceptDirectInput?: boolean };
  const thread = (over: RawThread = {}): CodexThread => ({
    threadId: UUID_A, cwd: "/home/rafa/repos/p1", owner: "p1", source: "cli",
    status: "idle", activeFlags: [], canAcceptDirectInput: true, ...over,
  });
  const SNAP = (over: { threads?: CodexThread[]; truncated?: boolean } = {}): CodexSnapshot => ({
    ok: true as const, ts: "2026-09-10T12:00:00.000Z",
    threads: over.threads ?? [thread()], truncated: over.truncated ?? false,
  });

  it("derives one session per distinct codex harness UUID, never fabricating a pane", () => {
    const db = openDb(":memory:");
    insertEvent(db, codexEvent(), NOW);
    insertEvent(db, codexEvent({ harness_session: UUID_B, project: "p2" }), NOW);
    const a = getCodexSessions(db, "p1", SNAP({ threads: [thread(), thread({ threadId: UUID_B })] }));
    expect(a.sourceOk).toBe(true);
    expect(a.sessions.map((s) => s.threadId)).toEqual([UUID_A]);
    expect(a.sessions[0]).toMatchObject({ threadId: UUID_A, status: "idle", working: false, attention: false });
    expect(JSON.stringify(a.sessions[0])).not.toContain("pane");
    expect(JSON.stringify(a.sessions[0])).not.toContain("tmuxIncarnation");
    db.close();
  });

  it("HubCodexSession.capsule carries mergeAsk: null, question: null for a self-tagged row with no merge_ask/message_tail of its own", () => {
    const db = openDb(":memory:");
    insertEvent(db, codexEvent({ payload: { capsule_status: "needs_input", capsule_rule: "tag" } }), NOW);
    const { sessions } = getCodexSessions(db, "p1", SNAP());
    expect(sessions[0].capsule).toMatchObject({ status: "needs_input", mergeAsk: null, question: null });
    db.close();
  });

  it("HubCodexSession.capsule reads merge_ask/message_tail from a classified row, the same way lastCapsule does (D2)", () => {
    const db = openDb(":memory:");
    const ev = insertEvent(db, codexEvent({ payload: { message_tail: "posso mergear feat/x em main?" } }), NOW);
    expect(classifyDeferred(db, ev.id, { status: "needs_input" })).toBe(true);
    const { sessions } = getCodexSessions(db, "p1", SNAP());
    expect(sessions[0].capsule).toMatchObject({
      status: "needs_input", mergeAsk: 0.97, question: "posso mergear feat/x em main?",
    });
    db.close();
  });

  it("HubCodexSession.capsule always carries answerable: true — no proactive Codex liveness check exists (D4)", () => {
    const db = openDb(":memory:");
    insertEvent(db, codexEvent({ payload: { capsule_status: "blocked", capsule_rule: "tag" } }), NOW);
    const { sessions } = getCodexSessions(db, "p1", SNAP());
    expect(sessions[0].capsule).toMatchObject({ status: "blocked", answerable: true });
    db.close();
  });

  it("a UUID-shaped harness_session on a Claude emitter is NOT a codex session", () => {
    const db = openDb(":memory:");
    insertEvent(db, {
      ...codexEvent(), emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
    }, NOW);
    const r = getCodexSessions(db, "p1", SNAP());
    expect(r.sessions).toEqual([]);
    db.close();
  });

  it("maps active/idle/notLoaded and query-failed snapshots truthfully", () => {
    const db = openDb(":memory:");
    insertEvent(db, codexEvent(), NOW);
    const active = getCodexSessions(db, "p1", SNAP({ threads: [thread({ status: "active", activeFlags: [] })] }));
    expect(active.sessions[0]).toMatchObject({ working: true, attention: false, status: "active" });
    const attention = getCodexSessions(db, "p1", SNAP({ threads: [thread({ status: "active", activeFlags: ["waitingOnUserInput"] })] }));
    expect(attention.sessions[0]).toMatchObject({ working: false, attention: true });
    const notLoaded = getCodexSessions(db, "p1", SNAP({ threads: [thread({ status: "notLoaded" })] }));
    expect(notLoaded.sessions[0]).toMatchObject({ working: false, attention: false, status: "notLoaded" });
    const failed = getCodexSessions(db, "p1", { ok: false, ts: "2026-09-10T12:00:00.000Z", error: "no socket" });
    expect(failed.sourceOk).toBe(false);
    expect(failed.sessions[0]).toMatchObject({ working: false, attention: false, status: "unknown" });
    db.close();
  });

  it("a loaded thread whose canonical owner is a project with no telemetry is an untracked warning, never a session", () => {
    const db = openDb(":memory:");
    const r = getCodexSessions(db, "p1", SNAP({ threads: [thread()] }));
    expect(r.sessions).toEqual([]);
    expect(r.untracked).toBe(true);
    db.close();
  });

  it("warns for an untracked loaded root even when another root in the SAME project has telemetry (C3)", () => {
    const db = openDb(":memory:");
    insertEvent(db, codexEvent(), NOW); // UUID_A has telemetry in p1
    const r = getCodexSessions(db, "p1", SNAP({ threads: [thread(), thread({ threadId: UUID_B, owner: "p1" })] }));
    expect(r.sessions.map((s) => s.threadId)).toEqual([UUID_A]);
    expect(r.untracked).toBe(true); // UUID_B is a loaded root owned by p1 with no telemetry
    db.close();
  });

  it("an unresolved owner is never guessed into a project warning (C3)", () => {
    const db = openDb(":memory:");
    const r = getCodexSessions(db, "p1", SNAP({ threads: [thread({ owner: null })] }));
    expect(r.untracked).toBe(false);
    db.close();
  });

  it("two live Codex sessions in the SAME project stay distinct and both render", () => {
    const db = openDb(":memory:");
    insertEvent(db, codexEvent(), NOW);
    insertEvent(db, codexEvent({ harness_session: UUID_B, payload: { message_tail: "other" } }), NOW);
    const r = getCodexSessions(db, "p1", SNAP({ threads: [
      thread(),
      thread({ threadId: UUID_B, status: "active", activeFlags: ["waitingOnApproval"] }),
    ] }));
    expect(r.sessions).toHaveLength(2);
    const a = r.sessions.find((s) => s.threadId === UUID_A)!;
    const b = r.sessions.find((s) => s.threadId === UUID_B)!;
    expect(a.status).toBe("idle");
    expect(b).toMatchObject({ status: "active", attention: true });
    db.close();
  });

  it("an old Claude open turn on a pane now positively running Codex is not working", () => {
    const db = openDb(":memory:");
    insertEvent(db, { ...started, type: "turn-started", run_id: null, role: "lead", source: "deterministic",
      emitter: "claude-userprompt", pane: "%1", tmux_incarnation: INCARNATION, payload: {} }, NOW);
    const liveCodex: LiveSnapshot = { paneIds: new Set(["%1"]), incarnation: INCARNATION, paneCommands: new Map([["%1", "codex"]]) };
    expect(isWorking(db, { pane: "%1", tmuxIncarnation: INCARNATION }, liveCodex)).toBe(false);
    // A generic shell is NOT proof of a runtime switch — the open turn keeps its claim.
    const liveShell: LiveSnapshot = { paneIds: new Set(["%1"]), incarnation: INCARNATION, paneCommands: new Map([["%1", "bash"]]) };
    expect(isWorking(db, { pane: "%1", tmuxIncarnation: INCARNATION }, liveShell)).toBe(true);
    db.close();
  });

  it("a live Claude pane and a Codex session in the same project both survive the hub assembly", () => {
    const db = openDb(":memory:");
    insertEvent(db, { ...started, type: "turn-started", run_id: null, role: "lead", source: "deterministic",
      emitter: "claude-userprompt", pane: "%1", tmux_incarnation: INCARNATION, payload: {} }, NOW);
    insertEvent(db, codexEvent(), NOW);
    const live: LiveSnapshot = { paneIds: new Set(["%1"]), incarnation: INCARNATION, paneCommands: new Map([["%1", "claude"]]) };
    const data = getHubData(db, live, {}, SNAP());
    const project = data.byProject["p1"];
    expect(project.panes[0]).toMatchObject({ pane: "%1", working: true });
    expect(project.codexSessions).toHaveLength(1);
    expect(data.codexSource).toBe("ok");
    db.close();
  });

  it("a failed codex snapshot is reported as codexSource failed with tmux data intact", () => {
    const db = openDb(":memory:");
    insertEvent(db, { ...started, type: "turn-started", run_id: null, role: "lead", source: "deterministic",
      emitter: "claude-userprompt", pane: "%1", tmux_incarnation: INCARNATION, payload: {} }, NOW);
    const live: LiveSnapshot = { paneIds: new Set(["%1"]), incarnation: INCARNATION, paneCommands: new Map([["%1", "claude"]]) };
    const data = getHubData(db, live, {}, { ok: false, ts: "2026-09-10T12:00:00.000Z", error: "no socket" });
    expect(data.codexSource).toBe("failed");
    expect(data.byProject["p1"].panes[0].working).toBe(true);
    db.close();
  });

  // MOA-502 Decision 3: the CLI-present PATH check feeds getHubData's 9th positional param.
  it("codexSource is absent when codexAbsent is true, regardless of the codex snapshot", () => {
    const db = openDb(":memory:");
    const live: LiveSnapshot = { paneIds: new Set(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, null, NOW.getTime(), true, null, null, true);
    expect(data.codexSource).toBe("absent");
    db.close();
  });

  it("codexSource stays failed (not absent) for a present-but-broken CLI (codexAbsent false, snapshot null)", () => {
    const db = openDb(":memory:");
    const live: LiveSnapshot = { paneIds: new Set(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, null, NOW.getTime(), true, null, null, false);
    expect(data.codexSource).toBe("failed");
    db.close();
  });

  it("codexSource stays ok/truncated unaffected when codexAbsent is false and a real snapshot is passed", () => {
    const db = openDb(":memory:");
    const live: LiveSnapshot = { paneIds: new Set(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, SNAP(), NOW.getTime(), true, null, null, false);
    expect(data.codexSource).toBe("ok");
    db.close();
  });
});

describe("MOA-469 — native Codex staleness", () => {
  const UUID = "0191f0aa-abcd-7000-8000-00000000000f";
  const INC = "234790:1787586213";
  const idleThread = () => ({
    threadId: UUID, cwd: "/home/rafa/repos/p1", owner: "p1", source: "cli",
    status: "idle" as const, activeFlags: [], canAcceptDirectInput: true,
  });
  const SNAP = (threads: CodexThread[], ok = true) => (ok
    ? { ok: true as const, ts: "2026-09-10T12:00:00.000Z", threads, truncated: false }
    : { ok: false as const, ts: "2026-09-10T12:00:00.000Z", error: "no socket" });
  const codexStop = (over: Partial<WorkflowEventInput> = {}) => ({
    run_id: null, project: "p1", role: "lead" as const, type: "turn-stopped" as const,
    source: "behavioral" as const, emitter: "codex-stop" as const, harness_session: UUID,
    pane: null, tmux_incarnation: null, payload: { message_tail: "prose" }, ...over,
  });
  const LIVE = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map<string, string>() };

  it("a waiting capsule past deadline with an idle loaded thread is possibly-stuck", () => {
    const db = openDb(":memory:");
    const declared = new Date("2026-09-10T00:00:00.000Z");
    insertEvent(db, codexStop({ payload: { capsule_status: "waiting", capsule_minutes: 1, capsule_rule: "tag" } }), declared);
    const now = Date.parse("2026-09-10T00:03:01.000Z"); // past max(120s, 20%) grace
    const r = evaluateStaleness(db, now, LIVE.paneIds, null, SNAP([idleThread()]));
    expect(r.candidates).toContainEqual(expect.objectContaining({ transport: "codex", threadId: UUID, reason: "possibly-stuck" }));
    expect(r.codexSourceOk).toBe(true);
    db.close();
  });

  it("an unloaded or active thread can never be possibly-stuck", () => {
    for (const status of ["notLoaded", "active", "systemError"]) {
      const db = openDb(":memory:");
      const declared = new Date("2026-09-10T00:00:00.000Z");
      insertEvent(db, codexStop({ payload: { capsule_status: "waiting", capsule_minutes: 1, capsule_rule: "tag" } }), declared);
      const now = Date.parse("2026-09-10T00:03:00.000Z");
      const r = evaluateStaleness(db, now, LIVE.paneIds, null, SNAP([{ ...idleThread(), status }]));
      expect(r.candidates.filter((c) => c.transport === "codex")).toEqual([]);
      db.close();
    }
  });

  it("a failed codex source skips only the native source and reports it", () => {
    const db = openDb(":memory:");
    const declared = new Date("2026-09-10T00:00:00.000Z");
    insertEvent(db, codexStop({ payload: { capsule_status: "waiting", capsule_minutes: 1, capsule_rule: "tag" } }), declared);
    const now = Date.parse("2026-09-10T00:03:01.000Z");
    const r = evaluateStaleness(db, now, LIVE.paneIds, null, SNAP([], false));
    expect(r.candidates).toEqual([]);
    expect(r.codexSourceOk).toBe(false);
    db.close();
  });

  it("a same-project OTHER session never contributes to this session's alert", () => {
    const db = openDb(":memory:");
    const other = "0191f0aa-abce-7000-8000-000000000010";
    const declared = new Date("2026-09-10T00:00:00.000Z");
    insertEvent(db, codexStop(), declared); // this session idle thread, no waiting capsule
    insertEvent(db, codexStop({ harness_session: other, project: "p1", payload: { capsule_status: "waiting", capsule_minutes: 1, capsule_rule: "tag" } }), declared);
    const now = Date.parse("2026-09-10T00:03:01.000Z");
    const r = evaluateStaleness(db, now, LIVE.paneIds, null, SNAP([{ ...idleThread(), threadId: other }]));
    expect(r.candidates.filter((c) => c.transport === "codex" && c.threadId === UUID)).toEqual([]);
    db.close();
  });

  it("a newer Claude event sharing the UUID neither replaces nor invalidates the native reference (C3)", () => {
    const db = openDb(":memory:");
    const codexStopEv = insertEvent(db, codexStop({ payload: { message_tail: "prose" } }), new Date("2026-09-10T00:00:00.000Z"));
    // A NEWER Claude event with the SAME harness_session — before the fix it became the native
    // no-signal reference and later invalidated the revalidation. It must be invisible to both.
    insertEvent(db, { ...codexStop(), emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION, payload: { message_tail: "claude" } }, new Date("2026-09-10T00:01:00.000Z"));
    const now = Date.parse("2026-09-10T02:05:00.000Z"); // >120min since the CODEX stop
    const r = evaluateStaleness(db, now, new Set(), null, SNAP([idleThread()]));
    const cand = r.candidates.find((c) => c.transport === "codex");
    expect(cand).toBeDefined();
    expect(cand!.referenceEventId).toBe(codexStopEv.id);
    setAfkOn(db);
    // Transactional revalidation must accept the native reference despite the newer Claude row.
    expect(maybeInsertStalenessAlert(db, cand!, now)).toBe(true);
    db.close();
  });

  it("maybeInsertStalenessAlert — native: AFK-off inserts nothing; a new same-session turn blocks", () => {
    const db = openDb(":memory:");
    const declared = new Date("2026-09-10T00:00:00.000Z");
    const stop = insertEvent(db, codexStop({ payload: { capsule_status: "waiting", capsule_minutes: 1, capsule_rule: "tag" } }), declared);
    const now = Date.parse("2026-09-10T00:03:01.000Z");
    const r = evaluateStaleness(db, now, LIVE.paneIds, null, SNAP([idleThread()]));
    const cand = r.candidates.find((c) => c.transport === "codex")!;
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(false); // AFK off
    setAfkOn(db);
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(true);
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(false); // dedup by reference
    // A new turn-started in the SAME session invalidates the stale waiting deadline.
    insertEvent(db, codexStop({ type: "turn-started", source: "deterministic", emitter: "codex-userprompt", payload: {} }), new Date("2026-09-10T00:02:00.000Z"));
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(false);
    // The synthetic alert carries the exact thread identity.
    const alert = db.prepare("SELECT harness_session, payload FROM workflow_events WHERE emitter = 'jaxos' AND type = 'attention-needed' ORDER BY id DESC LIMIT 1").get() as { harness_session: string | null; payload: string };
    expect(alert.harness_session).toBe(UUID);
    expect(JSON.parse(alert.payload).reference_event_id).toBe(stop.id);
    db.close();
  });

  it("maybeInsertStalenessAlert — native no-signal: AFK-off forbids, AFK-on permits one deduplicated alert", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, codexStop(), new Date("2026-09-10T00:00:00.000Z"));
    const now = Date.parse("2026-09-10T02:05:00.000Z"); // >120min quiet fallback
    const r = evaluateStaleness(db, now, LIVE.paneIds, null, SNAP([idleThread()]));
    const cand = r.candidates.find((c) => c.transport === "codex" && c.reason === "no-signal")!;
    expect(cand.referenceEventId).toBe(stop.id);
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(false); // AFK off — suppressed before insertion
    setAfkOn(db);
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(true);
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(false); // dedup by reference
    db.close();
  });
});

describe("MOA-469 F1 — a stored Codex row carrying a pane is never pane authority (review 8dc843212345)", () => {
  const UUID = "0191f0aa-f1f1-7000-8000-0000000000f1";
  const thread = (over: Partial<CodexThread> = {}): CodexThread => ({
    threadId: UUID, cwd: "/home/rafa/repos/p1", owner: "p1", source: "cli",
    status: "idle", activeFlags: [], canAcceptDirectInput: true, ...over,
  });
  const SNAP = (threads: CodexThread[] = [thread()]): CodexSnapshot => ({
    ok: true as const, ts: "2026-09-10T12:00:00.000Z", threads, truncated: false,
  });
  const paneCodex = (over: Partial<WorkflowEventInput> = {}): WorkflowEventInput => ({
    run_id: null, project: "p1", role: "lead", type: "turn-stopped",
    source: "behavioral", emitter: "codex-stop", harness_session: UUID,
    payload: { message_tail: "prose" }, pane: "%1", tmux_incarnation: INCARNATION, ...over,
  });
  const claudeEvent = (over: Partial<WorkflowEventInput> = {}): WorkflowEventInput => ({
    run_id: null, project: "p1", role: "lead", type: "turn-started",
    source: "deterministic", emitter: "claude-userprompt",
    payload: {}, pane: "%1", tmux_incarnation: INCARNATION, ...over,
  });
  const live = (): LiveSnapshot => ({ paneIds: new Set(["%1"]), incarnation: INCARNATION, paneCommands: new Map([["%1", "claude"]]) });

  it("(a) a pane-carrying Codex row never produces a HubPane or a tmux staleness candidate — and stays visible as a native session", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, paneCodex(), new Date("2026-09-10T00:00:00.000Z"));
    // Native identity stays visible...
    expect(getCodexSessions(db, "p1", SNAP()).sessions.map((s) => s.threadId)).toEqual([UUID]);
    // ...but it fabricates no pane.
    expect(getProjectPanes(db, "p1", live())).toEqual([]);
    const now = Date.parse("2026-09-10T02:05:00.000Z"); // > 120min quiet
    const r = evaluateStaleness(db, now, new Set(["%1"]), INCARNATION, SNAP());
    expect(r.candidates.filter((c) => c.transport === "tmux")).toEqual([]);
    // The same stored row still feeds the NATIVE alert path (exact session), unchanged.
    expect(r.candidates.filter((c) => c.transport === "codex")).toEqual([
      expect.objectContaining({ threadId: UUID, reason: "no-signal", referenceEventId: stop.id }),
    ]);
    db.close();
  });

  it("(b) later Codex starts/stops/permissions on a Claude pane never close its turn/question or change its capsule", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    insertEvent(db, claudeEvent(), new Date("2026-09-10T00:00:00.000Z"));
    insertEvent(db, claudeEvent({ type: "question", source: "deterministic", emitter: "claude-pretool",
      payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } }), new Date("2026-09-10T00:01:00.000Z"));
    // Newer Codex starts/stops/permissions on the SAME pane, in the SAME project — all must be
    // invisible to Claude pane derivation.
    insertEvent(db, paneCodex({ type: "turn-started", source: "deterministic", emitter: "codex-userprompt", payload: {} }), new Date("2026-09-10T00:02:00.000Z"));
    insertEvent(db, paneCodex({ type: "attention-needed", emitter: "codex-permission", payload: { reason: "permission" } }), new Date("2026-09-10T00:03:00.000Z"));
    insertEvent(db, paneCodex({ payload: { capsule_status: "done", capsule_rule: "tag" } }), new Date("2026-09-10T00:04:00.000Z"));

    const ref = { pane: "%1", tmuxIncarnation: INCARNATION };
    expect(hasOpenTurn(db, ref)).toBe(true);          // the Claude turn stays open
    expect(hasPendingQuestion(db, ref)).toBe(true);   // the Claude question stays open
    const panes = getProjectPanes(db, "p1", live());
    expect(panes).toHaveLength(1);
    expect(panes[0].pendingQuestion).toMatchObject({ pane: "%1" });
    expect(panes[0].capsule).toBeNull();              // no Claude stop yet — the codex capsule is ignored
    expect(panes[0].working).toBe(true);
    db.close();
  });

  it("(b2) a Codex stop does not overwrite a Claude capsule, and a p2 codex pane row never owns the pane", () => {
    const db = openDb(":memory:");
    insertEvent(db, claudeEvent({ type: "turn-stopped", emitter: "claude-stop",
      payload: { capsule_status: "waiting", capsule_minutes: 5, capsule_rule: "tag" } }), new Date("2026-09-10T00:00:00.000Z"));
    insertEvent(db, paneCodex({ payload: { capsule_status: "done", capsule_rule: "tag" } }), new Date("2026-09-10T00:01:00.000Z"));
    insertEvent(db, paneCodex({ project: "p2", payload: { capsule_status: "done", capsule_rule: "tag" } }), new Date("2026-09-10T00:02:00.000Z"));
    const panes = getProjectPanes(db, "p1", live());
    expect(panes).toHaveLength(1);
    expect(panes[0].capsule).toMatchObject({ status: "waiting", minutes: 5 });
    // The pane still belongs to p1: p2 owns no pane despite the pane-carrying codex row.
    expect(getProjectPanes(db, "p2", live())).toEqual([]);
    const hub = getHubData(db, live(), {}, null, Date.parse("2026-09-10T01:00:00.000Z"));
    expect(hub.byProject["p1"].panes).toHaveLength(1);
    expect(hub.byProject["p2"].panes).toEqual([]);
    db.close();
  });

  it("(c) pane metadata on Codex input does not make deriveCapsule consume Claude questions/attention/turn starts", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    insertEvent(db, claudeEvent(), new Date("2026-09-10T00:00:00.000Z"));
    insertEvent(db, claudeEvent({ type: "question", source: "deterministic", emitter: "claude-pretool",
      payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } }), new Date("2026-09-10T00:01:00.000Z"));
    insertEvent(db, claudeEvent({ type: "attention-needed", emitter: "claude-notification", payload: { reason: "permission" } }), new Date("2026-09-10T00:02:00.000Z"));
    // A codex-stop on the same pane: rungs 2/3 must NOT fire off the Claude question/attention.
    expect(deriveCapsule(db, paneCodex()).capsule_rule).toBe("deferred");
    db.close();
  });

  it("(d) a pane-carrying Codex silence alert stays exact-session and AFK-gated, with no duplicate tmux alert", () => {
    const db = openDb(":memory:");
    const stop = insertEvent(db, paneCodex(), new Date("2026-09-10T00:00:00.000Z"));
    const now = Date.parse("2026-09-10T02:05:00.000Z");
    const r = evaluateStaleness(db, now, new Set(["%1"]), INCARNATION, SNAP());
    const cand = r.candidates.find((c) => c.transport === "codex")!;
    expect(r.candidates.filter((c) => c.transport === "tmux")).toEqual([]); // no tmux candidate at all
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(false);           // AFK off
    setAfkOn(db);
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(true);
    expect(maybeInsertStalenessAlert(db, cand, now)).toBe(false);           // dedup by reference
    const alerts = db.prepare(
      "SELECT harness_session, pane, payload FROM workflow_events WHERE emitter = 'jaxos' AND type = 'attention-needed'",
    ).all() as { harness_session: string | null; pane: string | null; payload: string }[];
    expect(alerts).toHaveLength(1);                                          // never a tmux-shaped duplicate
    expect(alerts[0].harness_session).toBe(UUID);
    expect(alerts[0].pane).toBeNull();
    expect(JSON.parse(alerts[0].payload).reference_event_id).toBe(stop.id);
    db.close();
  });

  it("(e) a pending Claude question followed by a pane-carrying codex-stop still accepts a structured claim", () => {
    const db = openDb(":memory:");
    insertEvent(db, claudeEvent(), new Date("2026-09-10T00:00:00.000Z"));
    const q = insertEvent(db, claudeEvent({ type: "question", source: "deterministic", emitter: "claude-pretool",
      payload: { tool_use_id: "t1", questions: [{ question: "q?", options: [{ label: "a" }] }] } }),
      new Date("2026-09-10T00:01:00.000Z"));
    // A later codex-stop on the SAME pane must not invalidate the Claude question's claim.
    insertEvent(db, paneCodex(), new Date("2026-09-10T00:02:00.000Z"));
    expect(claimAnswer(db, { question_event_id: q.id, reply: "1: 1" }, new Date("2026-09-10T00:03:00.000Z")).ok).toBe(true);
    db.close();
  });

  it("(f) a Claude stopped turn followed by a pane-carrying codex-userprompt still accepts a freeform claim", () => {
    const db = openDb(":memory:");
    const ts = insertEvent(db, claudeEvent({ type: "turn-stopped", emitter: "claude-stop",
      harness_session: "0191f0aa-f1f1-7000-8000-0000000000c1",
      payload: { capsule_status: "done", capsule_rule: "tag" } }), new Date("2026-09-10T00:00:00.000Z"));
    // A later codex-userprompt on the SAME pane (a DIFFERENT harness_session) must not make the
    // Claude session non-idle — the freeform claim is session-keyed now (D3), pane metadata on a
    // stored Codex row is never identity.
    insertEvent(db, paneCodex({ type: "turn-started", source: "deterministic", emitter: "codex-userprompt", payload: {} }),
      new Date("2026-09-10T00:01:00.000Z"));
    expect(claimFreeformAnswer(db, { question_event_id: ts.id, reply: "ship it" }, new Date("2026-09-10T00:02:00.000Z")).ok).toBe(true);
    db.close();
  });
});

describe("MOA-469 live corrections — turn-stopped is local telemetry, not a notification", () => {
  const UUID = "0191f0aa-c0de-7000-8000-0000000000c0";
  const stopFor = (over: Partial<WorkflowEventInput> = {}): WorkflowEventInput => ({
    run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
    emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
    payload: { capsule_status: "done", capsule_rule: "tag" }, ...over,
  });

  it("AFK-on stops of every capsule status and both emitters are local and retain their status/rule", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    for (const status of ["done", "needs_input", "waiting", "blocked"] as const) {
      for (const emitter of ["claude-stop", "codex-stop"] as const) {
        const payload = status === "waiting"
          ? { capsule_status: status, capsule_minutes: 5, capsule_rule: "tag" }
          : { capsule_status: status, capsule_rule: "tag" };
        const row = insertEvent(db, stopFor({
          emitter,
          harness_session: emitter === "codex-stop" ? UUID : undefined,
          pane: emitter === "codex-stop" ? undefined : "%1",
          tmux_incarnation: emitter === "codex-stop" ? undefined : INCARNATION,
          payload,
        }));
        expect(row.delivery).toBe("local"); // AFK on, forward policy removed — still no forward
        expect(row.payload).toMatchObject({ capsule_status: status, capsule_rule: "tag" });
        expect(listPending(db, 100).find((r) => r.id === row.id)).toBeUndefined();
      }
    }
    db.close();
  });

  it("a deferred stop stays local yet can still undergo deferred classification", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    const stop = insertEvent(db, {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "ambiguous prose" },
    });
    expect(stop.delivery).toBe("local");
    expect(stop.payload).toMatchObject({ capsule_status: "unknown", capsule_rule: "deferred", capsule_attempts: 0 });
    expect(classifyDeferred(db, stop.id, { status: "done" })).toBe(true);
    expect(getProjectTimeline(db, "p1")[0].payload).toMatchObject({ capsule_status: "done", capsule_rule: "classified" });
    db.close();
  });

  it("listPending excludes an older pending stop while a later eligible run-finished row remains selectable (limit 1)", () => {
    const db = openDb(":memory:");
    setAfkOn(db);
    // A legacy pending stop seeded via SQL — predates this policy, delivery stays 'pending'.
    db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery, pane, tmux_incarnation, harness_session)
       VALUES (?, NULL, 'p1', 'lead', 'turn-stopped', 'deterministic', 'claude-stop', ?, 'pending', '%1', ?, NULL)`,
    ).run(NOW.toISOString(), JSON.stringify({ capsule_status: "needs_input", capsule_rule: "tag" }), INCARNATION);
    const f = insertEvent(db, finished, new Date(NOW.getTime() + 1000));
    expect(f.delivery).toBe("pending"); // run-finished stays forward-policy
    expect(listPending(db, 1).map((r) => r.id)).toEqual([f.id]);
    expect(listPending(db, 100).some((r) => r.type === "turn-stopped")).toBe(false);
    // The seeded stop is neither marked delivered nor deleted.
    const stored = db.prepare("SELECT delivery FROM workflow_events WHERE type = 'turn-stopped'").get() as { delivery: string };
    expect(stored.delivery).toBe("pending");
    db.close();
  });
});

describe("getWorktreeLedgerRows", () => {
  it("returns one row per run, all-time, no 30-day window", () => {
    const db = openDb(":memory:");
    insertEvent(db, { project: "demo", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: { phase: "p", runtime: "opencode-builder", kind: "build", target: "feat/x", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" } });
    insertEvent(db, { project: "demo", run_id: "b1", role: "builder", type: "run-finished", source: "deterministic", emitter: "wrapper", payload: { phase: "p", contract_status: "ok", result: "success", exit_code: 0, report_path: "/r", summary: "s" } });
    const rows = getWorktreeLedgerRows(db, "demo");
    expect(rows).toEqual([{ id: expect.any(Number), runId: "b1", kind: "build", branch: "feat/x", startedAt: expect.any(String), status: "finished", finishedAt: expect.any(String) }]);
  });
});

describe("getLoopSummary", () => {
  it("resolves a diff review's own group through its build, not the diff's own longer phase (round 3 F4); null with no rows", () => {
    const db = openDb(":memory:");
    expect(getLoopSummary(db, "nobody")).toBeNull();
    insertEvent(db, { project: "demo", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: { phase: "moa474a2", runtime: "opencode-builder", kind: "build", target: "feat/x", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" } });
    insertEvent(db, { project: "demo", run_id: "d1", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: { phase: "moa474a2diffr1", runtime: "codex", kind: "diff", target: "feat/x", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", builder_run_id: "b1" } });
    const summary = getLoopSummary(db, "demo");
    expect(summary).not.toBeNull();
    expect(summary?.buildCount).toBe(1);
    expect(summary?.diffCount).toBe(1);
  });

  it("a legacy diff row persisted without builder_run_id still projects via the branch fallback (F7, round 4)", () => {
    // F7 (round 4): ingress now REQUIRES builder_run_id on a new diff run-started event
    // (workflow-events.ts), but a row already in the DB from before that requirement
    // must still read correctly -- `insertEvent` here bypasses ingress entirely, the
    // same as a row persisted years ago would be read today.
    const db = openDb(":memory:");
    insertEvent(db, { project: "demo", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: { phase: "moa467", runtime: "opencode-builder", kind: "build", target: "feat/moa-467", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" } });
    insertEvent(db, { project: "demo", run_id: "d1", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: { phase: "moa467claudediff", runtime: "codex", kind: "diff", target: "feat/moa-467", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r" } }); // no builder_run_id
    const summary = getLoopSummary(db, "demo");
    expect(summary?.buildCount).toBe(1);
    expect(summary?.diffCount).toBe(1); // matched by branch, not builder_run_id
  });

  it("wallDays is windowed to the anchored build's own dispatch, in two scenarios (plan review round 1, F4)", () => {
    const db = openDb(":memory:");
    const buildPayload = (phase: string, target: string) => ({ phase, runtime: "opencode-builder", kind: "build", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" });
    // Plan deviation: distinct merge_sha seeds per merge -- insertEvent's merge-approved
    // idempotency suppression keys on (project, target, merge_sha), so the plan's shared
    // "b"*40 sha silently collapsed m0 and m1 into one row.
    const mergePayload = (phase: string, branch: string, shaSeed = "b") => ({ phase, branch, sha: "a".repeat(40), target: branch, approved_by: "rafa", merge_sha: shaSeed.repeat(40) });
    // reused: feat/reused was merged once for OLD, unrelated work (b0/m0), then reused
    // (b1/m1) -- the second build's own wall time must use the second merge.
    insertEvent(db, { project: "reused", run_id: "b0", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("old-work", "feat/reused") }, new Date("2026-01-01T00:00:00.000Z"));
    insertEvent(db, { project: "reused", run_id: "m0", role: "lead", type: "merge-approved", source: "deterministic", emitter: "wrapper", payload: mergePayload("old-work", "feat/reused", "b") }, new Date("2026-01-03T00:00:00.000Z"));
    insertEvent(db, { project: "reused", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa500", "feat/reused") }, new Date("2026-09-10T00:00:00.000Z"));
    insertEvent(db, { project: "reused", run_id: "m1", role: "lead", type: "merge-approved", source: "deterministic", emitter: "wrapper", payload: mergePayload("moa500", "feat/reused", "c") }, new Date("2026-09-15T00:00:00.000Z"));
    expect(getLoopSummary(db, "reused")?.wallDays).toBe(5); // never the old 2026-01 merge
    // predates: feat/x was merged once (m2) before ITS OWN build (b2) even existed -- no
    // merge inside b2's window means wall time is null/absent, never negative.
    insertEvent(db, { project: "predates", run_id: "m2", role: "lead", type: "merge-approved", source: "deterministic", emitter: "wrapper", payload: mergePayload("old-work", "feat/x") }, new Date("2026-01-01T00:00:00.000Z"));
    insertEvent(db, { project: "predates", run_id: "b2", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa501", "feat/x") }, new Date("2026-09-10T00:00:00.000Z"));
    expect(getLoopSummary(db, "predates")?.wallDays).toBeNull();
  });

  it("wallDays windows by event TIMESTAMPS, not insertion order (plan review round 2, F4)", () => {
    const db = openDb(":memory:");
    const buildPayload = (phase: string, target: string) => ({ phase, runtime: "opencode-builder", kind: "build", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" });
    const mergePayload = (phase: string, branch: string) => ({ phase, branch, sha: "a".repeat(40), target: branch, approved_by: "rafa", merge_sha: "b".repeat(40) });
    // Merge row INSERTED first (lower id) but its real ts is AFTER the build's ts --
    // id-based windowing would wrongly exclude this in-window merge; ts windowing must not.
    insertEvent(db, { project: "outoforder", run_id: "m1", role: "lead", type: "merge-approved", source: "deterministic", emitter: "wrapper", payload: mergePayload("moa900", "feat/y") }, new Date("2026-09-15T00:00:00.000Z"));
    insertEvent(db, { project: "outoforder", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa900", "feat/y") }, new Date("2026-09-10T00:00:00.000Z"));
    expect(getLoopSummary(db, "outoforder")?.wallDays).toBe(5);
  });

  it("counts the whole anchored family, not just the seed's one build (F4, round 4)", () => {
    const db = openDb(":memory:");
    const buildPayload = (phase: string, target: string) => ({ phase, runtime: "opencode-builder", kind: "build", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" });
    const specPayload = (phase: string, target: string) => ({ phase, runtime: "claude", kind: "spec", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r" });
    // b1 then a resumed b2 share the SAME phase (a --resume attempt is its own build, L7);
    // the CLI's own `_anchor_builds_for_prefix` counts both -- the old code hardcoded 1.
    insertEvent(db, { project: "family", run_id: "s1", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: specPayload("moa510", "docs/spec.md") }, new Date("2026-09-01T00:00:00.000Z"));
    insertEvent(db, { project: "family", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa510", "feat/z") }, new Date("2026-09-02T00:00:00.000Z"));
    insertEvent(db, { project: "family", run_id: "b2", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa510", "feat/z") }, new Date("2026-09-03T00:00:00.000Z"));
    const summary = getLoopSummary(db, "family");
    expect(summary?.buildCount).toBe(2);
    expect(summary?.specCount).toBe(1);
  });

  it("a trailing plan review after a build already exists still anchors to that build's family (F4, round 5)", () => {
    const db = openDb(":memory:");
    const buildPayload = (phase: string, target: string) => ({ phase, runtime: "opencode-builder", kind: "build", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" });
    const diffPayload = (phase: string, target: string, builderRunId: string) => ({ phase, runtime: "codex", kind: "diff", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", builder_run_id: builderRunId });
    const planPayload = (phase: string) => ({ phase, runtime: "claude", kind: "plan", target: "docs/plan.md", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r" });
    const mergePayload = (phase: string, branch: string) => ({ phase, branch, sha: "a".repeat(40), target: branch, approved_by: "rafa", merge_sha: "b".repeat(40) });
    // Build, diff, and merge happened FIRST; a plan re-review is dispatched LAST (the
    // seed). Before the fix, seedAnchor stayed null for a spec/plan seed and the
    // no-build fallback below dropped this whole existing family.
    insertEvent(db, { project: "trailing", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa610", "feat/moa-610") }, new Date("2026-09-01T00:00:00.000Z"));
    insertEvent(db, { project: "trailing", run_id: "d1", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: diffPayload("moa610-diff-r1", "feat/moa-610", "b1") }, new Date("2026-09-02T00:00:00.000Z"));
    insertEvent(db, { project: "trailing", run_id: "d1", role: "reviewer", type: "run-finished", source: "deterministic", emitter: "wrapper", payload: { phase: "moa610-diff-r1", contract_status: "ok", verdict: "approve" } }, new Date("2026-09-02T00:00:00.000Z"));
    insertEvent(db, { project: "trailing", run_id: "m1", role: "lead", type: "merge-approved", source: "deterministic", emitter: "wrapper", payload: mergePayload("moa610", "feat/moa-610") }, new Date("2026-09-03T00:00:00.000Z"));
    insertEvent(db, { project: "trailing", run_id: "p1", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: planPayload("moa610") }, new Date("2026-09-04T00:00:00.000Z"));
    const summary = getLoopSummary(db, "trailing");
    expect(summary?.buildCount).toBe(1);
    expect(summary?.diffCount).toBe(1);
    expect(summary?.planCount).toBe(1);
    expect(summary?.approved).toBe(true); // the merge/diff family is counted, not dropped
    // F1 (diff review 79cbcce4, MEDIUM): the plan review trails the merge, so the
    // matched start ts is AFTER mergeTs -- must be null, never a negative wall time.
    expect(summary?.wallDays).toBeNull();
  });

  it("with no build dispatched yet, counts every spec/plan row in the seed's own family, not just the seed (F4, round 4)", () => {
    const db = openDb(":memory:");
    const specPayload = (phase: string) => ({ phase, runtime: "claude", kind: "spec", target: "docs/spec.md", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r" });
    // Two spec-review rounds on the SAME ticket, before any build was ever dispatched.
    insertEvent(db, { project: "prebuild", run_id: "s1", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: specPayload("moa520") }, new Date("2026-09-01T00:00:00.000Z"));
    insertEvent(db, { project: "prebuild", run_id: "s2", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: specPayload("moa520") }, new Date("2026-09-02T00:00:00.000Z"));
    const summary = getLoopSummary(db, "prebuild");
    expect(summary).toMatchObject({ specCount: 2, planCount: 0, buildCount: 0, diffCount: 0, wallDays: null });
  });

  it("picks the window's later build by ts, not insertion id (F5, round 4)", () => {
    const db = openDb(":memory:");
    const buildPayload = (phase: string, target: string) => ({ phase, runtime: "opencode-builder", kind: "build", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" });
    const mergePayload = (phase: string, branch: string) => ({ phase, branch, sha: "a".repeat(40), target: branch, approved_by: "rafa", merge_sha: "b".repeat(40) });
    // Three builds share one phase/branch; the window's end must be the EARLIEST-ts
    // later build (b2, ts 09-10), not merely the first one an id-ordered scan meets.
    // b3 (the temporally-latest build) is inserted FIRST (lowest id) specifically so an
    // id-ordered `.find()` would wrongly land on it instead of the earlier b2.
    insertEvent(db, { project: "windoworder", run_id: "b3", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa540", "feat/w") }, new Date("2026-09-20T00:00:00.000Z"));
    insertEvent(db, { project: "windoworder", run_id: "m1", role: "lead", type: "merge-approved", source: "deterministic", emitter: "wrapper", payload: mergePayload("moa540", "feat/w") }, new Date("2026-09-15T00:00:00.000Z"));
    insertEvent(db, { project: "windoworder", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa540", "feat/w") }, new Date("2026-09-01T00:00:00.000Z"));
    insertEvent(db, { project: "windoworder", run_id: "b2", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa540", "feat/w") }, new Date("2026-09-10T00:00:00.000Z"));
    const summary = getLoopSummary(db, "windoworder");
    expect(summary?.buildCount).toBe(3); // b1, b2, b3 all share the phase (F4)
    // Correct window is [b1's ts, b2's ts) -- m1 (09-15) falls OUTSIDE it, so it must
    // NOT count as this build's merge. An id-ordered window (bounded by b3 instead)
    // would wrongly include m1 and report wallDays: 14.
    expect(summary?.wallDays).toBeNull();
  });

  it("orders same-timestamp reviewer rows by id, matching the CLI's ts,id tie-break (F3)", () => {
    const db = openDb(":memory:");
    const buildPayload = (phase: string, target: string) => ({ phase, runtime: "opencode-builder", kind: "build", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true" });
    const diffPayload = (phase: string, target: string, builderRunId: string) => ({ phase, runtime: "codex", kind: "diff", target, caller: "claude", caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", builder_run_id: builderRunId });
    const sameTs = new Date("2026-09-10T00:00:00.000Z");
    insertEvent(db, { project: "tie", run_id: "b1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper", payload: buildPayload("moa600", "feat/tie") }, new Date("2026-09-01T00:00:00.000Z"));
    // d1 (round 1, reject) inserted before d2 (round 2, approve) -- SAME ts, lower id.
    // A reversed tie-break (id desc, or unordered) would process d2 first and report
    // rounds: 1 instead of the true rounds: 2.
    insertEvent(db, { project: "tie", run_id: "d1", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: diffPayload("moa600-diff-r1", "feat/tie", "b1") }, sameTs);
    insertEvent(db, { project: "tie", run_id: "d1", role: "reviewer", type: "run-finished", source: "deterministic", emitter: "wrapper", payload: { phase: "moa600-diff-r1", contract_status: "ok", verdict: "reject" } }, sameTs);
    insertEvent(db, { project: "tie", run_id: "d2", role: "reviewer", type: "run-started", source: "deterministic", emitter: "wrapper", payload: diffPayload("moa600-diff-r2", "feat/tie", "b1") }, sameTs);
    insertEvent(db, { project: "tie", run_id: "d2", role: "reviewer", type: "run-finished", source: "deterministic", emitter: "wrapper", payload: { phase: "moa600-diff-r2", contract_status: "ok", verdict: "approve" } }, sameTs);
    const summary = getLoopSummary(db, "tie");
    expect(summary?.diffCount).toBe(2);
    expect(summary?.rounds).toBe(2);
    expect(summary?.approved).toBe(true);
  });
});

describe("Phase 2 projections (spec Decisions 2, 9)", () => {
  it("lastCapsule carries the selected turn-stopped row's id as eventId", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%7", role: "lead", tmux_incarnation: "100:1" }, NOW);
    const stop: WorkflowEventInput = {
      run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral", emitter: "claude-stop",
      payload: { message_tail: "Ambiguous final response." }, pane: "%7", tmux_incarnation: "100:1",
    };
    const ev = insertEvent(db, stop, NOW);
    classifyDeferred(db, ev.id, { status: "needs_input" });
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%7"]), incarnation: "100:1" };
    const [pane] = getProjectPanes(db, "p1", live);
    expect(pane.capsule).toEqual({ status: "needs_input", minutes: null, declaredAt: ev.ts, eventId: ev.id, mergeAsk: null, question: "Ambiguous final response.", answerable: false });
    db.close();
  });

  it("derives mergeAsk and question from a rung-7-classified row's message_tail (last line containing '?')", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%1", role: "lead", tmux_incarnation: INCARNATION }, NOW);
    const stop = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "Two things done.\nShould I retry the build?\nOr just report?\nAguardo." } }, NOW);
    classifyDeferred(db, stop.id, { status: "needs_input" });
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: INCARNATION };
    const [pane] = getProjectPanes(db, "p1", live);
    expect(pane.capsule?.mergeAsk).toBe(0.6);
    expect(pane.capsule?.question).toBe("Or just report?");
    db.close();
  });

  it("falls back to the last non-empty trimmed line verbatim when no line contains '?'", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%1", role: "lead", tmux_incarnation: INCARNATION }, NOW);
    const stop = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "First line.\nposso mergear feat/x em main" } }, NOW);
    classifyDeferred(db, stop.id, { status: "needs_input" });
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: INCARNATION };
    const [pane] = getProjectPanes(db, "p1", live);
    expect(pane.capsule?.mergeAsk).toBeNull();
    expect(pane.capsule?.question).toBe("posso mergear feat/x em main");
    db.close();
  });

  it("derives question: null when message_tail has no non-empty line at all (Review Focus 3)", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%1", role: "lead", tmux_incarnation: INCARNATION }, NOW);
    const stop = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "   \n  " } }, NOW);
    classifyDeferred(db, stop.id, { status: "done" });
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: INCARNATION };
    const [pane] = getProjectPanes(db, "p1", live);
    expect(pane.capsule?.question).toBeNull();
    db.close();
  });

  it("a capsule_rule:'tag' row (no message_tail at all) carries mergeAsk: null, question: null (Review Focus 2)", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%1", role: "lead", tmux_incarnation: INCARNATION }, NOW);
    const tagged = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { capsule_rule: "tag", capsule_status: "needs_input" } }, NOW);
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: INCARNATION };
    const [pane] = getProjectPanes(db, "p1", live);
    expect(pane.capsule).toEqual({ status: "needs_input", minutes: null, declaredAt: tagged.ts, eventId: tagged.id, mergeAsk: null, question: null, answerable: false });
    db.close();
  });

  it("a deferred row (rung 7, no Jev call yet) carries mergeAsk: null but a non-null derived question (Review Focus 1)", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%1", role: "lead", tmux_incarnation: INCARNATION }, NOW);
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      payload: { message_tail: "May I merge `feat/x` into `main`?" } }, NOW);
    // rung 6 is gone (D1): insertEvent now settles this as unknown/deferred, queued for the
    // poller — no classifyDeferred call is made here.
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: INCARNATION };
    const [pane] = getProjectPanes(db, "p1", live);
    expect(pane.capsule?.status).toBe("unknown");
    expect(pane.capsule?.mergeAsk).toBeNull();
    expect(pane.capsule?.question).toBe("May I merge `feat/x` into `main`?");
    db.close();
  });

  it("answerable reflects the injected liveness check, keyed on harness_session and eventId, for a self-tagged needs_input capsule too (D4 — not only the rung-7/Jev path)", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%1", role: "lead", tmux_incarnation: INCARNATION }, NOW);
    const HS = "0191f0aa-eeee-7000-8000-00000000000a";
    const tagged = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION, harness_session: HS,
      payload: { capsule_rule: "tag", capsule_status: "needs_input" } }, NOW);
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: INCARNATION };
    const calls: [string, number][] = [];
    const alive = (sessionId: string, eventId: number) => { calls.push([sessionId, eventId]); return true; };
    const [alivePane] = getProjectPanes(db, "p1", live, NOW.getTime(), alive);
    expect(alivePane.capsule?.answerable).toBe(true);
    expect(calls).toEqual([[HS, tagged.id]]);
    const [deadPane] = getProjectPanes(db, "p1", live, NOW.getTime(), () => false);
    expect(deadPane.capsule?.answerable).toBe(false);
    db.close();
  });

  it("answerable is true for a waiting/done capsule without ever consulting the liveness check", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "jax-p1-lead", pane: "%1", role: "lead", tmux_incarnation: INCARNATION }, NOW);
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION, harness_session: "0191f0aa-eeee-7000-8000-00000000000b",
      payload: { capsule_rule: "tag", capsule_status: "done" } }, NOW);
    const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: INCARNATION };
    let called = false;
    const [pane] = getProjectPanes(db, "p1", live, NOW.getTime(), () => { called = true; return false; });
    expect(pane.capsule?.answerable).toBe(true);
    expect(called).toBe(false);
    db.close();
  });

  it("lastRunFor projects verifyCommand/buildCommand from a builder run's run-started payload, null otherwise", () => {
    const db = openDb(":memory:");
    const startedBuild: WorkflowEventInput = {
      run_id: "b1", project: "p1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
      payload: { phase: "B", runtime: "codex", kind: "build", target: "feat/x", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "jax-p1-build-b1", repo: "/r", verify: "pnpm test", build: "pnpm build" },
    };
    const finishedOk: WorkflowEventInput = {
      run_id: "b1", project: "p1", role: "builder", type: "run-finished", source: "deterministic", emitter: "wrapper",
      payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/r.md", summary: "ok", head_sha: "b".repeat(40), result: "success" },
    };
    insertEvent(db, startedBuild, NOW);
    insertEvent(db, finishedOk, NOW);
    expect(lastRunFor(db, "p1")).toMatchObject({ runId: "b1", target: "feat/x", outcome: "success", verifyCommand: "pnpm test", buildCommand: "pnpm build" });

    insertEvent(db, { ...startedBuild, run_id: "b2", payload: { ...startedBuild.payload, verify: "pnpm test", build: undefined } }, NOW);
    insertEvent(db, { ...finishedOk, run_id: "b2" }, NOW);
    expect(lastRunFor(db, "p1")).toMatchObject({ runId: "b2", verifyCommand: "pnpm test", buildCommand: null });

    insertEvent(db, { ...startedBuild, run_id: "r1", role: "reviewer", payload: { phase: "B", runtime: "codex", kind: "diff", target: "feat/x", caller: "claude", caller_session: "s", model: "m", effort: "e", session: "jax-p1-diff-r1", repo: "/r" } }, NOW);
    insertEvent(db, { ...finishedOk, run_id: "r1", role: "reviewer", payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/r.md", summary: "ok", verdict: "approve" } }, NOW);
    expect(lastRunFor(db, "p1")).toMatchObject({ runId: "r1", verifyCommand: null, buildCommand: null });
    db.close();
  });
});

describe("Phase 3 — project_prefs + archive_after_days (spec §10, Decisions 15/16/20)", () => {
  const NOW3 = new Date("2026-09-19T12:00:00.000Z");

  it("getProjectPrefs is empty with no rows and never depends on hub membership (round 3 F1)", () => {
    const db = openDb(":memory:");
    expect(getProjectPrefs(db)).toEqual({});
    // A project with ZERO workflow_events rows — never a hub member — still comes back.
    setProjectPrefs(db, "dormant", { pinned: true }, NOW3);
    expect(getProjectPrefs(db)).toEqual({ dormant: { pinned: true, hiddenAt: null } });
    db.close();
  });

  it("hidden: true stamps hidden_at with the write's own time; hidden: false clears it", () => {
    const db = openDb(":memory:");
    expect(setProjectPrefs(db, "p1", { hidden: true }, NOW3)).toEqual({ pinned: false, hiddenAt: "2026-09-19T12:00:00.000Z" });
    expect(setProjectPrefs(db, "p1", { hidden: false }, new Date(NOW3.getTime() + 1000))).toEqual({ pinned: false, hiddenAt: null });
    db.close();
  });

  it("pinning never resets hidden_at (F6) and unpinning leaves it alone too", () => {
    const db = openDb(":memory:");
    setProjectPrefs(db, "p1", { hidden: true }, NOW3);
    expect(setProjectPrefs(db, "p1", { pinned: true }, new Date(NOW3.getTime() + 60_000)))
      .toEqual({ pinned: true, hiddenAt: "2026-09-19T12:00:00.000Z" });
    expect(setProjectPrefs(db, "p1", { pinned: false }, new Date(NOW3.getTime() + 120_000)))
      .toEqual({ pinned: false, hiddenAt: "2026-09-19T12:00:00.000Z" });
    db.close();
  });

  it("every prefs write is one audited mission-prefs mutation carrying the previous row (rule 8)", () => {
    const db = openDb(":memory:");
    setProjectPrefs(db, "p1", { hidden: true }, NOW3);
    setProjectPrefs(db, "p1", { pinned: true }, NOW3);
    const rows = db.prepare("SELECT payload FROM mutations WHERE kind = 'mission-prefs' ORDER BY id").all() as { payload: string }[];
    expect(rows.map((r) => JSON.parse(r.payload))).toEqual([
      expect.objectContaining({ project: "p1", hidden: true, previous: { pinned: false, hiddenAt: null } }),
      expect.objectContaining({ project: "p1", pinned: true, previous: { pinned: false, hiddenAt: "2026-09-19T12:00:00.000Z" } }),
    ]);
    db.close();
  });

  it("archive_after_days falls back to 14 with no row, round-trips a write, keeps `enabled`, and audits", () => {
    const db = openDb(":memory:");
    expect(getArchiveAfterDays(db)).toBe(14);
    setAfk(db, true, NOW3);
    setArchiveAfterDays(db, 7, NOW3);
    expect(getArchiveAfterDays(db)).toBe(7);
    expect(getAfkEnabled(db)).toBe(true);
    const audit = db.prepare("SELECT payload FROM mutations WHERE kind = 'mission-archive-after-days'").get() as { payload: string };
    expect(JSON.parse(audit.payload)).toMatchObject({ days: 7, previous: 14 });
    db.close();
  });

  it("setArchiveAfterDays refuses anything outside the 1-30 integer range (F7) without writing", () => {
    const db = openDb(":memory:");
    for (const bad of [0, 31, 90, 1.5, -1, Number.NaN]) {
      expect(() => setArchiveAfterDays(db, bad, NOW3)).toThrow(/1-30/);
    }
    expect(getArchiveAfterDays(db)).toBe(14);
    expect(db.prepare("SELECT COUNT(*) AS n FROM mutations WHERE kind = 'mission-archive-after-days'").get()).toEqual({ n: 0 });
    db.close();
  });
});

describe("Phase 3 — lazy 30-day prune of hook telemetry on insert (spec §4.3, Decision 6, F7/F14)", () => {
  const T0 = new Date("2026-09-19T12:00:00.000Z");
  const DAY = 24 * 60 * 60 * 1000;
  const OLD = new Date(T0.getTime() - 31 * DAY);
  const telemetry = (type: "tool-used" | "subagent-started" | "subagent-stopped", payload: Record<string, unknown>): WorkflowEventInput => ({
    run_id: null, project: "p1", role: "lead", type, source: "deterministic",
    emitter: type === "tool-used" ? "claude-toolused" : type === "subagent-started" ? "claude-subagentstart" : "claude-subagentstop",
    payload, pane: "%1", tmux_incarnation: "100:1",
  });
  const count = (db: ReturnType<typeof openDb>, type: string) =>
    (db.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE type = ?").get(type) as { n: number }).n;

  it("deletes tool-used and subagent rows past the cutoff and leaves every other kind untouched", () => {
    const db = openDb(":memory:");
    pruneState.lastMs = 0;
    insertEvent(db, telemetry("tool-used", { tool: "Bash" }), OLD);
    insertEvent(db, telemetry("subagent-started", { agent_id: "a", agent_type: "x" }), OLD);
    insertEvent(db, telemetry("subagent-stopped", { agent_id: "a" }), OLD);
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", payload: { capsule_status: "done" }, pane: "%1", tmux_incarnation: "100:1" }, OLD);
    pruneState.lastMs = 0; // the inserts above ran the guard at OLD; force the next insert to prune
    insertEvent(db, telemetry("tool-used", { tool: "Read" }), T0);
    expect(count(db, "tool-used")).toBe(1); // only the fresh row survives
    expect(count(db, "subagent-started")).toBe(0);
    expect(count(db, "subagent-stopped")).toBe(0);
    expect(count(db, "turn-stopped")).toBe(1); // never pruned, whatever its age
    db.close();
  });

  it("a row exactly 30 days old is kept — the cutoff deletes strictly older rows only", () => {
    const db = openDb(":memory:");
    pruneState.lastMs = 0;
    insertEvent(db, telemetry("tool-used", { tool: "Bash" }), new Date(T0.getTime() - 30 * DAY));
    pruneState.lastMs = 0;
    insertEvent(db, telemetry("tool-used", { tool: "Read" }), T0);
    expect(count(db, "tool-used")).toBe(2);
    db.close();
  });

  it("the once-per-minute guard skips a second prune inside the same window and reopens after 60s", () => {
    const db = openDb(":memory:");
    pruneState.lastMs = 0;
    insertEvent(db, telemetry("tool-used", { tool: "Bash" }), T0); // prunes (nothing), lastMs = T0
    insertEvent(db, telemetry("tool-used", { tool: "Grep" }), OLD); // a stale row arriving late; guard: OLD - T0 < 60s
    insertEvent(db, telemetry("tool-used", { tool: "Edit" }), new Date(T0.getTime() + 1000)); // 1s later: guard holds
    expect(count(db, "tool-used")).toBe(3);
    insertEvent(db, telemetry("tool-used", { tool: "Write" }), new Date(T0.getTime() + 60_000)); // guard reopens: Grep goes
    expect(count(db, "tool-used")).toBe(3); // Bash, Edit, Write
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE json_extract(payload, '$.tool') = 'Grep'").get()).toEqual({ n: 0 });
    db.close();
  });

  it("getProjectTimeline excludes the three telemetry types before its LIMIT (cold review F5)", () => {
    const db = openDb(":memory:");
    for (let i = 0; i < 25; i++) insertEvent(db, telemetry("tool-used", { tool: "Bash" }), new Date(T0.getTime() + i * 1000));
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "deterministic",
      emitter: "claude-stop", payload: { capsule_status: "done" }, pane: "%1", tmux_incarnation: "100:1" }, new Date(T0.getTime() + 26_000));
    const timeline = getProjectTimeline(db, "p1", 20);
    expect(timeline.some((e) => e.type === "tool-used")).toBe(false);
    expect(timeline.some((e) => e.type === "turn-stopped")).toBe(true); // never crowded out by the 25 tool-used rows
    db.close();
  });
});

describe("Phase 3 — heartbeat (spec §6, Decision 5)", () => {
  it("heartbeatFromRows zero-fills 60 per-minute buckets, newest last, and buckets same-minute rows", () => {
    const now = Date.parse("2026-09-19T12:00:30.000Z");
    const rows = [
      { minute: "2026-09-19T12:00", n: 3 }, // the current (partial) minute
      { minute: "2026-09-19T11:30", n: 2 },
    ];
    const out = heartbeatFromRows(rows, now);
    expect(out).toHaveLength(60);
    expect(out[59]).toBe(3); // last element = current minute
    expect(out[29]).toBe(2); // 30 minutes back (now=12:00:30 → the 11:30:30 bucket is index 29)
    expect(out.filter((n) => n === 0).length).toBe(58);
  });

  it("getHubData wires a real per-project heartbeat from workflow_events (excludes emitter=jaxos)", () => {
    const db = openDb(":memory:");
    const now = Date.parse("2026-09-19T12:00:00.000Z");
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "tool-used", source: "deterministic",
      emitter: "claude-toolused", payload: { tool: "Bash" }, pane: "%1", tmux_incarnation: "100:1" }, new Date(now));
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "question-answered", source: "deterministic",
      emitter: "jaxos", payload: {} }, new Date(now));
    const live = { paneIds: new Set(["%1"]), incarnation: "100:1", paneCommands: new Map() };
    const data = getHubData(db, live, { p1: "2020-01-01T00:00:00.000Z" }, null, now, true);
    expect(data.byProject.p1.heartbeat).toHaveLength(60);
    expect(data.byProject.p1.heartbeat[59]).toBe(1); // the tool-used row counts; the jaxos row does not
    db.close();
  });

  it("cold review F1: the SQL cutoff matches the oldest emitted bucket — an event at now-60m+15s is excluded by both", () => {
    const db = openDb(":memory:");
    const now = Date.parse("2026-09-19T12:00:00.000Z");
    // now-60m+15s = 11:00:15 — inside the OLD (buggy) SQL window (>= 11:00:00) but its "11:00"
    // minute is never emitted (the oldest emitted bucket is "11:01"). The fixed cutoff excludes it
    // from the SQL query too, so it is consistently absent instead of queried then discarded.
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "tool-used", source: "deterministic",
      emitter: "claude-toolused", payload: { tool: "Bash" }, pane: "%1", tmux_incarnation: "100:1" },
      new Date(now - 60 * 60_000 + 15_000));
    const live = { paneIds: new Set(["%1"]), incarnation: "100:1", paneCommands: new Map() };
    const data = getHubData(db, live, { p1: "2020-01-01T00:00:00.000Z" }, null, now, true);
    expect(data.byProject.p1.heartbeat.every((n) => n === 0)).toBe(true); // excluded from every bucket
    db.close();
  });
});

describe("Phase 3 — última ação (spec §6, round 2 F6 — 24h inclusive cutoff)", () => {
  const NOW4 = Date.parse("2026-09-19T12:00:00.000Z");
  const DAY = 24 * 60 * 60 * 1000;
  function withTool(db: ReturnType<typeof openDb>, ts: Date) {
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "tool-used", source: "deterministic",
      emitter: "claude-toolused", payload: { tool: "Read" }, pane: "%1", tmux_incarnation: "100:1" }, ts);
  }
  it("a row exactly 24h old still returns non-null; one second older returns null", () => {
    const db1 = openDb(":memory:");
    withTool(db1, new Date(NOW4 - DAY));
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    expect(getHubData(db1, live, {}, null, NOW4, true).byProject.p1.lastAction).toEqual({ tool: "Read", ts: new Date(NOW4 - DAY).toISOString() });
    db1.close();
    const db2 = openDb(":memory:");
    withTool(db2, new Date(NOW4 - DAY - 1000));
    expect(getHubData(db2, live, {}, null, NOW4, true).byProject.p1.lastAction).toBeNull();
    db2.close();
  });
  it("no tool-used row at all is null, not a placeholder", () => {
    const db = openDb(":memory:");
    insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-started", source: "deterministic",
      emitter: "claude-userprompt", payload: {}, pane: "%1", tmux_incarnation: "100:1" }, new Date(NOW4));
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    expect(getHubData(db, live, {}, null, NOW4, true).byProject.p1.lastAction).toBeNull();
    db.close();
  });

  it("cold review F2: a late-inserted older row (higher id, stale ts) does not hide a newer qualifying row (lower id)", () => {
    const db = openDb(":memory:");
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    withTool(db, new Date(NOW4 - 1000)); // id=1, 1s ago, well within the 24h cutoff
    withTool(db, new Date(NOW4 - DAY - 1000)); // id=2, inserted after id=1 but 25h old (outside cutoff)
    expect(getHubData(db, live, {}, null, NOW4, true).byProject.p1.lastAction).toEqual({
      tool: "Read", ts: new Date(NOW4 - 1000).toISOString(),
    });
    db.close();
  });
});

describe("Phase 3 — subagentCount (spec §7, Decision 7, cold review F11)", () => {
  const NOW5 = Date.parse("2026-09-19T12:00:00.000Z");
  const MIN = 60_000;
  function agentEvent(type: "subagent-started" | "subagent-stopped", payload: Record<string, unknown>, extra: Partial<WorkflowEventInput> = {}, ts = new Date(NOW5)) {
    return insertEvent(this_db, { run_id: null, project: "p1", role: "lead", type, source: "deterministic",
      emitter: type === "subagent-started" ? "claude-subagentstart" : "claude-subagentstop",
      payload, pane: "%1", tmux_incarnation: "100:1", ...extra }, ts);
  }
  let this_db: ReturnType<typeof openDb>;

  it("a matched stop counts zero; an unmatched start under/at 30 min counts one; strictly over 30 min counts zero", () => {
    this_db = openDb(":memory:");
    agentEvent("subagent-started", { agent_id: "a1", agent_type: "x" }, {}, new Date(NOW5));
    agentEvent("subagent-stopped", { agent_id: "a1" }, {}, new Date(NOW5 + MIN));
    agentEvent("subagent-started", { agent_id: "a2", agent_type: "x" }, {}, new Date(NOW5 - 30 * MIN)); // exactly 30 min old
    agentEvent("subagent-started", { agent_id: "a3", agent_type: "x" }, {}, new Date(NOW5 - 31 * MIN)); // strictly over 30 min
    const live = { paneIds: new Set(["%1"]), incarnation: "100:1", paneCommands: new Map() };
    const data = getHubData(this_db, live, {}, null, NOW5, true);
    expect(data.byProject.p1.panes[0].subagentCount).toBe(1); // only a2 (a1 stopped, a3 orphaned-out)
    this_db.close();
  });

  it("a matching pane_key under a DIFFERENT project counts zero (F11 project-scoping)", () => {
    this_db = openDb(":memory:");
    insertEvent(this_db, { run_id: null, project: "other", role: "lead", type: "subagent-started",
      source: "deterministic", emitter: "claude-subagentstart", payload: { agent_id: "a1", agent_type: "x" },
      pane: "%1", tmux_incarnation: "100:1" }, new Date(NOW5));
    const live = { paneIds: new Set(["%1"]), incarnation: "100:1", paneCommands: new Map() };
    const data = getHubData(this_db, live, {}, null, NOW5, true);
    expect(data.byProject.p1).toBeUndefined(); // p1 never even mentioned a1 — no membership either
    this_db.close();
  });
});

describe("Phase 3 — subagentCount via getCodexSessions (spec §7, cold review F4 — codex harness_session identity)", () => {
  const NOW6 = Date.parse("2026-09-19T12:00:00.000Z");
  const MIN = 60_000;
  const SESSION = "0191f0aa-8888-7000-8000-000000000008";
  // codex-stop is what makes getCodexSessions list this harness_session at all (its own query is
  // scoped to codex-stop/permission/userprompt, not the new telemetry emitters).
  const anchor = (db: ReturnType<typeof openDb>, ts: Date) => insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped", source: "behavioral", emitter: "codex-stop", harness_session: SESSION, payload: { message_tail: "x" }, pane: null, tmux_incarnation: null }, ts);
  const agent = (db: ReturnType<typeof openDb>, type: "subagent-started" | "subagent-stopped", payload: Record<string, unknown>, ts: Date) => insertEvent(db, { run_id: null, project: "p1", role: "lead", type, source: "deterministic", emitter: type === "subagent-started" ? "codex-subagentstart" : "codex-subagentstop", harness_session: SESSION, payload, pane: null, tmux_incarnation: null }, ts);

  it("counts an unmatched start, zeroes on its matching stop, and never counts a start expired past 30 min", () => {
    const db = openDb(":memory:");
    anchor(db, new Date(NOW6 - 10 * MIN));
    agent(db, "subagent-started", { agent_id: "a1", agent_type: "worker" }, new Date(NOW6 - 5 * MIN));
    expect(getCodexSessions(db, "p1", null, NOW6).sessions[0].subagentCount).toBe(1); // a1: open, unmatched
    agent(db, "subagent-stopped", { agent_id: "a1" }, new Date(NOW6 - 4 * MIN));
    expect(getCodexSessions(db, "p1", null, NOW6).sessions[0].subagentCount).toBe(0); // a1: matched stop
    agent(db, "subagent-started", { agent_id: "a2", agent_type: "worker" }, new Date(NOW6 - 31 * MIN));
    expect(getCodexSessions(db, "p1", null, NOW6).sessions[0].subagentCount).toBe(0); // a2: expired past 30 min
    db.close();
  });
});

describe("Phase 3 — lastRunFor enrichment (spec §11, Decisions 17/18, cold review F2/F3)", () => {
  function startedAndFinished(db: ReturnType<typeof openDb>, sp: Record<string, unknown>, p: Record<string, unknown>, role: "builder" | "reviewer" = "builder") {
    insertEvent(db, { run_id: "a1b2c3d4e5f6", project: "p1", role, type: "run-started", source: "deterministic",
      emitter: "wrapper", payload: { kind: "build", runtime: "codex", caller: "claude", caller_session: "s", repo: "/x", verify: "pnpm test", model: "sonnet", ...sp } }, new Date("2026-09-19T10:00:00.000Z"));
    insertEvent(db, { run_id: "a1b2c3d4e5f6", project: "p1", role, type: "run-finished", source: "deterministic",
      emitter: "wrapper", payload: { exit_code: 0, contract_status: "ok", result: "success", ...p } }, new Date("2026-09-19T10:05:00.000Z"));
  }
  // F2: PROJECT-root relative, mirroring relativeWithin(join(REPOS_ROOT, project), abs) — never a
  // plain ROOTS.repos reduction. F3: this fake only ever sees an ABSOLUTE path — proof lastRunFor
  // normalized the fixture's relative ".local/docs/specs/x-spec.md" target first.
  const relToRepos = (project: string, abs: string): string | null => {
    const root = `${REPOS_ROOT}/${project}/`;
    return abs.startsWith(root) ? abs.slice(root.length) : null;
  };
  it("runtimeModel/profileName/reportRel/targetRel are each sourced from the payload they're actually validated on", () => {
    const db = openDb(":memory:");
    startedAndFinished(db, { kind: "spec", target: ".local/docs/specs/x-spec.md", requested_profile: "fallback" }, { report_path: join(REPOS_ROOT, "p1", ".local/docs/reports/x.md") });
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, null, Date.now(), true, null, relToRepos);
    expect(data.byProject.p1.lastRun?.runtimeModel).toBe("sonnet");
    expect(data.byProject.p1.lastRun?.profileName).toBe("fallback");
    expect(data.byProject.p1.lastRun?.reportRel).toBe(".local/docs/reports/x.md");
    expect(data.byProject.p1.lastRun?.targetRel).toBe(".local/docs/specs/x-spec.md"); // F3: the relative fixture target, normalized against p1's own root before relToRepos ever ran
    db.close();
  });
  it("profileName is null for a reviewer row even with a requested_profile present; runtimeModel is never role-gated", () => {
    const db = openDb(":memory:");
    startedAndFinished(db, { kind: "diff", requested_profile: "fallback" }, {}, "reviewer");
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, null, Date.now(), true);
    expect(data.byProject.p1.lastRun?.profileName).toBeNull();
    expect(data.byProject.p1.lastRun?.runtimeModel).toBe("sonnet");
    db.close();
  });
  it("targetRel is attempted only for spec/plan kinds, never build/diff/merge, and both degrade to null with no relToRepos injected", () => {
    const db = openDb(":memory:");
    startedAndFinished(db, { kind: "build" }, {});
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, null, Date.now(), true); // no relToRepos injected
    expect(data.byProject.p1.lastRun?.targetRel).toBeNull();
    expect(data.byProject.p1.lastRun?.reportRel).toBeNull();
    db.close();
  });
});

describe("Phase 3 — lastStep (spec §8)", () => {
  it("a build activeRun with an injected readChildLogTail gets a redacted, parsed lastStep; every other kind is always null", () => {
    const db = openDb(":memory:");
    const repo = join(REPOS_ROOT, "jax-os");
    mkdirSync(join(FIXTURE.reposRoot, "jax-os"), { recursive: true }); // repo must be real+allowlisted (round 2 F1)
    insertEvent(db, { run_id: "a1b2c3d4e5f6", project: "p1", role: "builder", type: "run-started", source: "deterministic",
      emitter: "wrapper", payload: { kind: "build", runtime: "codex", caller: "claude", caller_session: "s", repo, verify: "pnpm test", model: "sonnet" } }, new Date());
    const tail = JSON.stringify({ type: "text", part: { text: "API_KEY=abcd1234secret\nprogress line" } });
    const readChildLogTail = (path: string) => (path === join(repo, ".local", "runs", "a1b2c3d4e5f6", "child.log") ? tail : null);
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, null, Date.now(), true, readChildLogTail);
    expect(data.byProject.p1.activeRun?.lastStep).toBe("API_KEY=[REDACTED] line");
    expect(data.byProject.p1.activeRun?.lastStep).not.toContain("abcd1234secret");
  });
  it("a non-build run kind never calls the reader and always has lastStep: null", () => {
    const db = openDb(":memory:");
    insertEvent(db, { run_id: "a1b2c3d4e5f6", project: "p1", role: "reviewer", type: "run-started", source: "deterministic",
      emitter: "wrapper", payload: { kind: "diff", runtime: "codex", caller: "claude", caller_session: "s", repo: "/x/repo", model: "sonnet" } }, new Date());
    const readChildLogTail = () => { throw new Error("must not be called for a non-build run"); };
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, null, Date.now(), true, readChildLogTail);
    expect(data.byProject.p1.activeRun?.lastStep).toBeNull();
  });
});

describe("Phase 3 — envelope prefs + settings (spec §10, round 3 F1)", () => {
  it("getHubData's prefs map holds a pinned project's row with zero events and no hub membership", () => {
    const db = openDb(":memory:");
    setProjectPrefs(db, "dormant", { pinned: true });
    const live = { paneIds: new Set<string>(), incarnation: null, paneCommands: new Map() };
    const data = getHubData(db, live, {}, null, Date.now(), true);
    expect(data.prefs).toEqual({ dormant: { pinned: true, hiddenAt: null } });
    expect(data.byProject.dormant).toBeUndefined(); // never a hub member — prefs is independent
    expect(data.settings).toEqual({ archiveAfterDays: 14 });
    db.close();
  });
});

describe("an answered ask clears its capsule (stale card buttons)", () => {
  const live: LiveSnapshot = { paneCommands: new Map(), paneIds: new Set(["%1"]), incarnation: INCARNATION };
  function seed(answer: { ok: boolean } | null) {
    const db = openDb(":memory:");
    upsertSession(db, { project: "p1", session: "main", pane: "%1", role: "lead", tmux_incarnation: INCARNATION }, NOW);
    const stop = insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "turn-stopped",
      source: "behavioral", emitter: "claude-stop", pane: "%1", tmux_incarnation: INCARNATION,
      harness_session: "s1", payload: { message_tail: "should I go on?" } }, NOW);
    classifyDeferred(db, stop.id, { status: "needs_input" });
    if (answer) {
      insertEvent(db, { run_id: null, project: "p1", role: "lead", type: "question-answered", source: "deterministic",
        emitter: "jaxos", payload: { kind: "freeform", question_event_id: stop.id, tool_use_id: null, mutation_id: 1, ok: answer.ok } }, NOW);
    }
    return { db, stopId: stop.id };
  }

  it("keeps the capsule while the ask is unanswered", () => {
    const { db, stopId } = seed(null);
    expect(getProjectPanes(db, "p1", live)[0].capsule).toMatchObject({ status: "needs_input", eventId: stopId });
    db.close();
  });

  it("drops the capsule once a successful answer references it", () => {
    const { db } = seed({ ok: true });
    expect(getProjectPanes(db, "p1", live)[0].capsule).toBeNull();
    db.close();
  });

  it("keeps the capsule when the answer failed (the ask is still open)", () => {
    const { db } = seed({ ok: false });
    expect(getProjectPanes(db, "p1", live)[0].capsule).not.toBeNull();
    db.close();
  });
});
