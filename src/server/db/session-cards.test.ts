// Session cards: DB-level tests over SIMULATED hook/jaxflow rows (in-memory SQLite, injected
// tmux/Codex snapshots). These assert what the hub exposes per pane, per session scope and per
// seat — the model/component half lives in part 2.
import { beforeEach, describe, expect, it, vi } from "vitest";

const FIXTURE = vi.hoisted(() => {
  const { mkdtempSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const home = mkdtempSync(join(tmpdir(), "jaxos-session-cards-home-"));
  const reposRoot = mkdtempSync(join(tmpdir(), "jaxos-session-cards-repos-"));
  writeFileSync(join(home, "settings.json"), JSON.stringify({ reposRoot }), "utf8");
  process.env.JAXOS_HOME = home;
  return { home };
});
// A question is "pending" (answerable from the card) only once it was queued for delivery, which
// needs the webhook integration on — same arrangement as workflows.test.ts.
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

import { openDb } from "./index";
import {
  activeRunsFor, getBuildRunSessions, getHubData, getProjectTimeline, insertEvent, lastRunFor, questionOwnsPane, setAfk, setForwardTypes, upsertSession,
  type HubEnvelopeData, type HubPane, type LiveSnapshot, type RunScope,
} from "./workflows";
import { FORWARDABLE_EVENT_TYPES, HOOK_TELEMETRY_TYPES, type WorkflowEventInput } from "../../lib/workflow";
import type { CodexSnapshot } from "../collectors/workflow-codex";

const P = "acme";
const INC = "4242:1700000000";
const A = "aaaaaaaa-0000-4000-8000-00000000000a";
const B = "bbbbbbbb-0000-4000-8000-00000000000b";
const C = "cccccccc-0000-4000-8000-00000000000c";
const D = "dddddddd-0000-4000-8000-00000000000d";
const Z = "ffffffff-0000-4000-8000-00000000000f";
const S = "55555555-0000-4000-8000-000000000005";
const T0 = Date.parse("2026-09-20T12:00:00.000Z");
const at = (min: number) => new Date(T0 + min * 60_000);
const NOW = T0 + 20 * 60_000;

type Db = ReturnType<typeof openDb>;

function claude(session: string | null, pane: string, type: WorkflowEventInput["type"], emitter: WorkflowEventInput["emitter"], payload: Record<string, unknown> = {}, project = P): WorkflowEventInput {
  return { run_id: null, project, role: "lead", type, source: "deterministic", emitter, payload, pane, tmux_incarnation: INC, harness_session: session };
}
function codex(type: WorkflowEventInput["type"], emitter: WorkflowEventInput["emitter"], payload: Record<string, unknown> = {}): WorkflowEventInput {
  return { run_id: null, project: P, role: "lead", type, source: "deterministic", emitter, payload, pane: null, tmux_incarnation: null, harness_session: C };
}
// jaxflow's own run rows: the caller's session rides in the payload as caller_session.
function runStarted(runId: string, role: "builder" | "reviewer", kind: string, caller: string, callerSession: string, target: string, project = P): WorkflowEventInput {
  return {
    run_id: runId, project, role, type: "run-started", source: "deterministic", emitter: "wrapper",
    payload: { phase: "P1", runtime: role === "builder" ? "opencode-builder" : "codex", kind, target, caller, caller_session: callerSession },
  };
}
function runFinished(runId: string, role: "builder" | "reviewer", payload: Record<string, unknown>, project = P): WorkflowEventInput {
  return { run_id: runId, project, role, type: "run-finished", source: "deterministic", emitter: "wrapper", payload: { phase: "P1", contract_status: "ok", ...payload } };
}
function setAfkOn(db: Db): void {
  setAfk(db, true);
  setForwardTypes(db, [...FORWARDABLE_EVENT_TYPES]);
}
const ask = (session: string | null, pane: string, tool = "tu_a", project = P) =>
  claude(session, pane, "question", "claude-pretool", {
    tool_use_id: tool, questions: [{ question: "Ship it?", options: [{ label: "yes" }, { label: "no" }] }],
  }, project);

const LIVE: LiveSnapshot = { paneIds: new Set(["%1", "%2", "%5"]), incarnation: INC, paneCommands: new Map() };
const CODEX: CodexSnapshot = {
  ok: true, ts: new Date(NOW).toISOString(), truncated: false,
  threads: [{ threadId: C, cwd: "/repos/acme", owner: P, source: "cli", status: "idle", activeFlags: [], canAcceptDirectInput: true }],
};

function hubOf(db: Db, now = NOW): HubEnvelopeData {
  return getHubData(db, LIVE, {}, CODEX, now);
}
const paneOf = (hub: HubEnvelopeData, project: string, pane: string): HubPane =>
  hub.byProject[project].panes.find((p) => p.pane === pane)!;

let db: Db;
beforeEach(() => {
  db = openDb(":memory:");
});

describe("pane session fields", () => {
  it("harnessSession is the newest recorded session on the exact pane key within the project", () => {
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, claude(B, "%1", "tool-used", "claude-toolused", { tool: "Read" }), at(2));
    const pane = paneOf(hubOf(db), P, "%1");
    expect(pane.harnessSession).toBe(B);
    db.close();
  });

  it("keeps each project's own session when one pane moved between projects", () => {
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt", {}, "proj-a"), at(1));
    insertEvent(db, claude(B, "%1", "turn-started", "claude-userprompt", {}, "proj-b"), at(2));
    const hub = hubOf(db);
    const a = paneOf(hub, "proj-a", "%1");
    expect(a.live).toBe(false);
    expect(a.harnessSession).toBe(A);
    expect(paneOf(hub, "proj-b", "%1").harnessSession).toBe(B);
    db.close();
  });

  it("a pane with only a session registration records no harnessSession", () => {
    upsertSession(db, { project: P, session: "acme-lead", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    expect(paneOf(hubOf(db), P, "%1").harnessSession).toBeNull();
    db.close();
  });

  it("capsuleSession is the session behind the capsule, not the one now holding the pane", () => {
    setAfkOn(db);
    upsertSession(db, { project: P, session: "acme-lead", pane: "%2", role: "lead", tmux_incarnation: INC }, at(0));
    insertEvent(db, claude(Z, "%2", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, claude(Z, "%2", "turn-stopped", "claude-stop", { capsule_status: "needs_input", capsule_rule: "tag" }), at(2));
    insertEvent(db, claude(B, "%2", "turn-started", "claude-userprompt"), at(3));
    const pane = paneOf(hubOf(db), P, "%2");
    expect(pane.capsuleSession).toBe(Z);
    expect(pane.harnessSession).toBe(B);
    db.close();
  });

  it("a legacy capsule or question row with no session records a null session", () => {
    setAfkOn(db);
    insertEvent(db, claude(null, "%1", "turn-stopped", "claude-stop", { capsule_status: "needs_input", capsule_rule: "tag" }), at(1));
    insertEvent(db, ask(null, "%1", "tu_legacy"), at(2));
    const pane = paneOf(hubOf(db), P, "%1");
    expect(pane.capsuleSession).toBeNull();
    expect(pane.questionSession).toBeNull();
    db.close();
  });

  it("questionSession is the asking session, distinct from the pane's current one", () => {
    setAfkOn(db);
    upsertSession(db, { project: P, session: "acme-lead", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, ask(A, "%1"), at(2));
    insertEvent(db, claude(B, "%1", "tool-used", "claude-toolused", { tool: "Read" }), at(3));
    const pane = paneOf(hubOf(db), P, "%1");
    expect(pane.questionSession).toBe(A);
    expect(pane.harnessSession).toBe(B);
    db.close();
  });

  it("questionAnswerable is true while the asker owns the pane and absent without a pending question", () => {
    setAfkOn(db);
    upsertSession(db, { project: P, session: "acme-lead", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    const q = insertEvent(db, ask(A, "%1"), at(2));
    const pane = paneOf(hubOf(db), P, "%1");
    expect(pane.pendingQuestion?.eventId).toBe(q.id);
    expect(pane.questionAnswerable).toBe(true);
    insertEvent(db, claude(A, "%2", "tool-used", "claude-toolused", { tool: "Read" }), at(3));
    expect("questionAnswerable" in paneOf(hubOf(db), P, "%2")).toBe(false);
    db.close();
  });

  it("questionAnswerable turns false after a new conversation starts in the pane", () => {
    setAfkOn(db);
    upsertSession(db, { project: P, session: "acme-lead", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, ask(A, "%1"), at(2));
    insertEvent(db, claude(B, "%1", "turn-started", "claude-userprompt"), at(3));
    expect(paneOf(hubOf(db), P, "%1").questionAnswerable).toBe(false);
    db.close();
  });

  it("questionAnswerable turns false after the same session moves the pane to another project", () => {
    setAfkOn(db);
    upsertSession(db, { project: "proj-a", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt", {}, "proj-a"), at(1));
    insertEvent(db, ask(A, "%1", "tu_a", "proj-a"), at(2));
    upsertSession(db, { project: "proj-b", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(3));
    insertEvent(db, claude(A, "%1", "tool-used", "claude-toolused", { tool: "Read" }, "proj-b"), at(3.1));
    expect(paneOf(hubOf(db), "proj-a", "%1").questionAnswerable).toBe(false);
    db.close();
  });

  it("a legacy question (no recorded session) stays answerable", () => {
    setAfkOn(db);
    insertEvent(db, ask(null, "%1", "tu_legacy"), at(1));
    insertEvent(db, claude(B, "%1", "tool-used", "claude-toolused", { tool: "Read" }), at(2));
    expect(paneOf(hubOf(db), P, "%1").questionAnswerable).toBe(true);
    db.close();
  });

  it("the hub flag is exactly the answer route's rule for every pane with a question", () => {
    setAfkOn(db);
    upsertSession(db, { project: P, session: "acme-lead", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    upsertSession(db, { project: P, session: "acme-lead", pane: "%2", role: "lead", tmux_incarnation: INC }, at(0));
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    const q1 = insertEvent(db, ask(A, "%1"), at(2));
    insertEvent(db, claude(B, "%2", "turn-started", "claude-userprompt"), at(3));
    const q2 = insertEvent(db, ask(B, "%2"), at(4));
    insertEvent(db, claude(Z, "%2", "turn-started", "claude-userprompt"), at(5)); // replaces B on %2
    const hub = hubOf(db);
    for (const [id, project, pane] of [[q1.id, P, "%1"], [q2.id, P, "%2"]] as const) {
      const p = paneOf(hub, project, pane);
      expect(p.pendingQuestion?.eventId).toBe(id);
      expect(p.questionAnswerable).toBe(questionOwnsPane(db, {
        id, project, ref: { pane, tmuxIncarnation: INC }, harnessSession: p.questionSession ?? null,
      }));
    }
    expect(paneOf(hub, P, "%1").questionAnswerable).toBe(true);
    expect(paneOf(hub, P, "%2").questionAnswerable).toBe(false);
    db.close();
  });

  it("counts only subagents of the pane's current session", () => {
    const start = (session: string | null, agent: string) =>
      claude(session, "%1", "subagent-started", "claude-toolused", { agent_id: agent });
    insertEvent(db, claude(Z, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, start(Z, "z1"), at(2));
    insertEvent(db, start(Z, "z2"), at(3));
    insertEvent(db, claude(B, "%1", "turn-started", "claude-userprompt"), at(4));
    insertEvent(db, start(B, "b1"), at(5));
    expect(paneOf(hubOf(db), P, "%1").subagentCount).toBe(1);
    db.close();
  });

  it("a new conversation with no subagents reads zero", () => {
    insertEvent(db, claude(Z, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, claude(Z, "%1", "subagent-started", "claude-toolused", { agent_id: "z1" }), at(2));
    insertEvent(db, claude(B, "%1", "turn-started", "claude-userprompt"), at(3));
    expect(paneOf(hubOf(db), P, "%1").subagentCount).toBe(0);
    db.close();
  });

  it("a pane whose rows recorded no session keeps the pane-wide count", () => {
    insertEvent(db, claude(null, "%1", "subagent-started", "claude-toolused", { agent_id: "x1" }), at(1));
    insertEvent(db, claude(null, "%1", "subagent-started", "claude-toolused", { agent_id: "x2" }), at(2));
    expect(paneOf(hubOf(db), P, "%1").subagentCount).toBe(2);
    db.close();
  });

  it("still counts Codex subagents by thread", () => {
    insertEvent(db, codex("turn-started", "codex-userprompt"), at(1));
    insertEvent(db, codex("subagent-started", "codex-userprompt", { agent_id: "c1" }), at(2));
    insertEvent(db, codex("subagent-started", "codex-userprompt", { agent_id: "c2" }), at(3));
    const session = hubOf(db).byProject[P].codexSessions.find((s) => s.threadId === C)!;
    expect(session.subagentCount).toBe(2);
    db.close();
  });
});

describe("scoped reads", () => {
  const session = (id: string): RunScope => ({ kind: "session", session: id });
  const unlinked = (...openSessions: string[]): RunScope => ({ kind: "unlinked", openSessions });

  it("applies the 20-row limit per session and keeps telemetry out of every scope", () => {
    for (let i = 0; i < 25; i++) insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(i));
    for (let i = 0; i < 3; i++) insertEvent(db, claude(B, "%2", "turn-started", "claude-userprompt"), at(30 + i));
    for (const [s, pane] of [[A, "%1"], [B, "%2"]] as const) {
      insertEvent(db, claude(s, pane, "tool-used", "claude-toolused", { tool: "Bash" }), at(40));
      insertEvent(db, claude(s, pane, "subagent-started", "claude-toolused", { agent_id: `x-${s}` }), at(41));
      insertEvent(db, claude(s, pane, "subagent-stopped", "claude-toolused", { agent_id: `x-${s}` }), at(42));
    }
    expect(getProjectTimeline(db, P, session(A))).toHaveLength(20);
    expect(getProjectTimeline(db, P, session(B))).toHaveLength(3);
    for (const rows of [getProjectTimeline(db, P, session(A)), getProjectTimeline(db, P, session(B)), getProjectTimeline(db, P)]) {
      expect(rows.some((e) => (HOOK_TELEMETRY_TYPES as readonly string[]).includes(e.type))).toBe(false);
    }
    expect(getProjectTimeline(db, P)).toHaveLength(20);
    db.close();
  });

  it("puts each row on exactly its owner's scope (runs, jaxos references, unlinked audits)", () => {
    setAfkOn(db);
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    const qa = insertEvent(db, ask(A, "%1"), at(2));
    insertEvent(db, runStarted("run-a", "builder", "build", "claude", A, "feat/a"), at(3));
    insertEvent(db, runFinished("run-a", "builder", { result: "success", exit_code: 0, report_path: "/r/a.md", summary: "ok" }), at(4));
    const bEvent = insertEvent(db, claude(B, "%2", "turn-started", "claude-userprompt"), at(5));
    insertEvent(db, {
      run_id: null, project: P, role: "lead", type: "question-answered", source: "deterministic", emitter: "jaxos",
      payload: { question_event_id: qa.id, ok: true },
    }, at(6));
    insertEvent(db, {
      run_id: null, project: P, role: "lead", type: "attention-needed", source: "deterministic", emitter: "jaxos",
      payload: { reason: "no-signal", reference_event_id: bEvent.id }, pane: "%2", tmux_incarnation: INC,
    }, at(7));
    insertEvent(db, {
      run_id: null, project: P, role: "builder", type: "merge-approved", source: "deterministic", emitter: "wrapper",
      payload: { branch: "feat/a", target: "main", merge_sha: "a".repeat(40), approved_by: "owner" }, pane: "%1", tmux_incarnation: INC,
    }, at(8));

    const aTypes = getProjectTimeline(db, P, session(A)).map((e) => e.type);
    expect(aTypes).toEqual(expect.arrayContaining(["turn-started", "question", "run-started", "run-finished", "question-answered"]));
    expect(aTypes).not.toContain("attention-needed");
    const bTypes = getProjectTimeline(db, P, session(B)).map((e) => e.type);
    expect(bTypes).toContain("attention-needed");
    expect(bTypes).not.toContain("run-finished");
    expect(bTypes).not.toContain("question-answered");
    expect(getProjectTimeline(db, P, unlinked(A, B))).toEqual([]);
    expect(getProjectTimeline(db, P).some((e) => e.type === "merge-approved")).toBe(true);
    db.close();
  });

  it("moves every row of a duplicate run to the latest start's owner", () => {
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, claude(B, "%2", "turn-started", "claude-userprompt"), at(2));
    insertEvent(db, runStarted("run-r", "builder", "build", "claude", A, "feat/r"), at(3));
    insertEvent(db, runStarted("run-r", "builder", "build", "claude", B, "feat/r"), at(4));
    expect(activeRunsFor(db, P, session(B)).map((r) => r.runId)).toEqual(["run-r"]);
    expect(activeRunsFor(db, P, session(A))).toEqual([]);
    expect(getProjectTimeline(db, P, session(A)).map((e) => e.type)).toEqual(["turn-started"]);
    expect(getProjectTimeline(db, P, session(B)).map((e) => e.type).sort()).toEqual(["run-started", "run-started", "turn-started"]);

    insertEvent(db, runFinished("run-r", "builder", { result: "success", exit_code: 0, report_path: "/r/r.md", summary: "ok" }), at(5));
    expect(getProjectTimeline(db, P, session(B)).map((e) => e.type).sort()).toEqual(["run-finished", "run-started", "run-started", "turn-started"]);
    expect(getProjectTimeline(db, P, session(A)).map((e) => e.type)).toEqual(["turn-started"]);
    expect(activeRunsFor(db, P, session(B))).toEqual([]);
    db.close();
  });

  it("an owner-less run-started belongs to no session and to both unlinked forms", () => {
    insertEvent(db, {
      run_id: "run-u", project: P, role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
      payload: { phase: "P1", runtime: "opencode-builder", kind: "build", target: "feat/u", caller: "claude" },
    }, at(1));
    expect(getProjectTimeline(db, P, session(A))).toEqual([]);
    expect(getProjectTimeline(db, P, session(B))).toEqual([]);
    expect(activeRunsFor(db, P, unlinked(A, B)).map((r) => r.runId)).toEqual(["run-u"]);
    expect(activeRunsFor(db, P, unlinked()).map((r) => r.runId)).toEqual(["run-u"]);
    expect(getProjectTimeline(db, P, unlinked(A, B)).map((e) => e.type)).toEqual(["run-started"]);
    db.close();
  });

  it("a dashboard-dispatched run is unlinked while 'jaxos' is not an open seat", () => {
    insertEvent(db, runStarted("run-j", "builder", "build", "jaxos", "jaxos", "feat/j"), at(1));
    expect(activeRunsFor(db, P, unlinked(A, B)).map((r) => r.runId)).toEqual(["run-j"]);
    expect(activeRunsFor(db, P, unlinked(A, B, "jaxos"))).toEqual([]);
    expect(activeRunsFor(db, P, session("jaxos")).map((r) => r.runId)).toEqual(["run-j"]);
    db.close();
  });

  it("a closed owner's run moves between unlinked and its own scope as seats change", () => {
    insertEvent(db, runStarted("run-d", "builder", "build", "claude", D, "feat/d"), at(1));
    expect(activeRunsFor(db, P, unlinked(A)).map((r) => r.runId)).toEqual(["run-d"]);
    expect(activeRunsFor(db, P, unlinked(A, D))).toEqual([]);
    expect(activeRunsFor(db, P, session(D)).map((r) => r.runId)).toEqual(["run-d"]);
    db.close();
  });

  it("scopes lastRunFor so another session's start never hides this session's result", () => {
    insertEvent(db, runStarted("r1", "builder", "build", "claude", A, "feat/1"), at(1));
    insertEvent(db, runFinished("r1", "builder", { result: "success", exit_code: 0, report_path: "/r/1.md", summary: "ok" }), at(2));
    insertEvent(db, runStarted("r2", "builder", "build", "claude", B, "feat/2"), at(3));
    expect(lastRunFor(db, P)).toBeNull();
    expect(lastRunFor(db, P, null, session(A))?.runId).toBe("r1");
    insertEvent(db, runStarted("r3", "builder", "build", "claude", A, "feat/3"), at(4));
    expect(lastRunFor(db, P, null, session(A))).toBeNull();
    db.close();
  });

  it("a reassigned start belongs to its latest owner and never hides another session's result", () => {
    insertEvent(db, runStarted("r1", "builder", "build", "claude", A, "feat/1"), at(1));
    insertEvent(db, runFinished("r1", "builder", { result: "success", exit_code: 0, report_path: "/r/1.md", summary: "ok" }), at(2));
    insertEvent(db, runStarted("r2", "builder", "build", "claude", A, "feat/2"), at(3));
    insertEvent(db, runStarted("r2", "builder", "build", "claude", B, "feat/2"), at(4));
    expect(lastRunFor(db, P, null, session(A))?.runId).toBe("r1");
    expect(lastRunFor(db, P, null, session(B))).toBeNull();
    expect(lastRunFor(db, P)).toBeNull();
    db.close();
  });
});

function seedTwoSessions(target: Db): void {
  upsertSession(target, { project: P, session: "acme-lead", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
  upsertSession(target, { project: P, session: "acme-lead", pane: "%2", role: "lead", tmux_incarnation: INC }, at(0));
  insertEvent(target, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
  insertEvent(target, claude(B, "%2", "turn-started", "claude-userprompt"), at(2));
}
const NO_CODEX: CodexSnapshot = { ok: false, ts: "", error: "codex runtime unavailable" };
const codexThread = (threadId: string): CodexSnapshot => ({
  ok: true, ts: new Date(NOW).toISOString(), truncated: false,
  threads: [{ threadId, cwd: "/repos/acme", owner: P, source: "cli", status: "idle", activeFlags: [], canAcceptDirectInput: true }],
});

describe("session seats", () => {
  it("gives two agents in different panes of one tmux session one seat each", () => {
    seedTwoSessions(db);
    const split = hubOf(db).byProject[P].split!;
    expect(split.seats.map((s) => s.sessionId)).toEqual([A, B]);
    expect(split.seats.map((s) => s.panes)).toEqual([[{ pane: "%1", tmuxIncarnation: INC }], [{ pane: "%2", tmuxIncarnation: INC }]]);
    db.close();
  });

  it("gives one session visible in two panes a single seat", () => {
    seedTwoSessions(db);
    insertEvent(db, claude(A, "%3", "tool-used", "claude-toolused", { tool: "Read" }), at(3));
    const live = { ...LIVE, paneIds: new Set(["%1", "%2", "%3"]) };
    const split = getHubData(db, live, {}, CODEX, NOW).byProject[P].split!;
    expect(split.seats.map((s) => s.sessionId)).toEqual([A, B]);
    expect(split.seats[0].panes.map((p) => p.pane).sort()).toEqual(["%1", "%3"]);
    db.close();
  });

  it("keeps one seat when a Codex thread id equals a tmux seat's session; tmux supplies the location", () => {
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, { run_id: null, project: P, role: "lead", type: "turn-started", source: "deterministic",
      emitter: "codex-userprompt", payload: {}, pane: null, tmux_incarnation: null, harness_session: A }, at(2));
    const split = getHubData(db, LIVE, {}, codexThread(A), NOW).byProject[P].split!;
    expect(split.seats).toHaveLength(1);
    expect(split.seats[0]).toMatchObject({ sessionId: A, transport: "tmux" });
    expect(split.seats[0].panes.length).toBeGreaterThan(0);
    db.close();
  });

  it("mixes Claude tmux seats with Codex-only seats", () => {
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, codex("turn-started", "codex-userprompt"), at(2));
    const split = hubOf(db).byProject[P].split!;
    expect(split.seats.map((s) => [s.sessionId, s.transport])).toEqual([[A, "tmux"], [C, "codex"]]);
    expect(split.seats[1].panes).toEqual([]);
    db.close();
  });

  it("never makes a seat from a pane that now belongs to another project", () => {
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt", {}, "proj-a"), at(1));
    insertEvent(db, claude(B, "%1", "turn-started", "claude-userprompt", {}, "proj-b"), at(2));
    const hub = hubOf(db);
    expect(hub.byProject["proj-a"].split).toBeUndefined();
    expect(hub.byProject["proj-b"].split?.seats.map((s) => s.sessionId)).toEqual([B]);
    db.close();
  });

  it("makes a live pane with no recorded session its own empty seat", () => {
    upsertSession(db, { project: P, session: "acme-lead", pane: "%5", role: "lead", tmux_incarnation: INC }, at(0));
    const split = hubOf(db).byProject[P].split!;
    expect(split.seats).toHaveLength(1);
    expect(split.seats[0]).toMatchObject({ sessionId: null, transport: "tmux", panes: [{ pane: "%5" }] });
    expect(split.seats[0].activity.timeline).toEqual([]);
    expect(split.seats[0].activity.activeRuns).toEqual([]);
    db.close();
  });

  it("with the Codex source unavailable, a closed session's in-flight run lands on the unlinked card", () => {
    insertEvent(db, claude(D, "%3", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, runStarted("run-d", "builder", "build", "claude", D, "feat/d"), at(2));
    const liveEmpty: LiveSnapshot = { paneIds: new Set(), incarnation: INC, paneCommands: new Map() };
    const split = getHubData(db, liveEmpty, {}, NO_CODEX, NOW).byProject[P].split!;
    expect(split.seats).toEqual([]);
    expect(split.unlinked.activeRuns.map((r) => r.runId)).toEqual(["run-d"]);
    db.close();
  });

  it("with the Codex source unavailable, a session that is also a live tmux seat keeps its owned runs", () => {
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, runStarted("run-a", "builder", "build", "claude", A, "feat/a"), at(2));
    const split = getHubData(db, LIVE, {}, NO_CODEX, NOW).byProject[P].split!;
    expect(split.seats.map((s) => s.sessionId)).toEqual([A]);
    expect(split.seats[0].activity.activeRuns.map((r) => r.runId)).toEqual(["run-a"]);
    db.close();
  });

  it("never reorders or re-attributes seats when activity order changes", () => {
    seedTwoSessions(db);
    insertEvent(db, runStarted("run-a", "builder", "build", "claude", A, "feat/a"), at(3));
    insertEvent(db, runStarted("run-b", "builder", "build", "claude", B, "feat/b"), at(4));
    const before = hubOf(db).byProject[P].split!;
    insertEvent(db, claude(B, "%2", "tool-used", "claude-toolused", { tool: "Edit" }), at(5));
    insertEvent(db, claude(A, "%1", "tool-used", "claude-toolused", { tool: "Read" }), at(6));
    const after = hubOf(db, NOW + 60_000).byProject[P].split!;
    const shape = (split: NonNullable<typeof before>) => split.seats.map((s) => [s.sessionId, s.panes.map((p) => p.pane), s.activity.activeRuns.map((r) => r.runId)]);
    expect(shape(after)).toEqual(shape(before));
    expect(after.seats.map((s) => s.sessionId)).toEqual([A, B]);
    db.close();
  });

  it("keeps a closed session's run unlinked and returns it to its seat when the session reopens", () => {
    seedTwoSessions(db);
    insertEvent(db, runStarted("run-b-build", "builder", "build", "claude", B, "feat/b"), at(3));
    const openSplit = hubOf(db).byProject[P].split!;
    expect(openSplit.seats.find((s) => s.sessionId === B)!.activity.activeRuns.map((r) => r.runId)).toEqual(["run-b-build"]);
    expect(openSplit.unlinked.activeRuns).toEqual([]);

    const withoutB: LiveSnapshot = { ...LIVE, paneIds: new Set(["%1"]) };
    const closedSplit = getHubData(db, withoutB, {}, CODEX, NOW).byProject[P].split!;
    expect(closedSplit.seats.map((s) => s.sessionId)).toEqual([A]);
    expect(closedSplit.unlinked.activeRuns.map((r) => r.runId)).toEqual(["run-b-build"]);

    const none: LiveSnapshot = { paneIds: new Set(), incarnation: INC, paneCommands: new Map() };
    const noSeatSplit = getHubData(db, none, {}, CODEX, NOW).byProject[P].split!;
    expect(noSeatSplit.seats).toEqual([]);
    expect(noSeatSplit.unlinked.activeRuns.map((r) => r.runId)).toEqual(["run-b-build"]);

    const reopened = hubOf(db).byProject[P].split!;
    expect(reopened.seats.find((s) => s.sessionId === B)!.activity.activeRuns.map((r) => r.runId)).toEqual(["run-b-build"]);
    expect(reopened.unlinked.activeRuns).toEqual([]);
    db.close();
  });

  it("is present only when there are seats or in-flight unlinked runs", () => {
    insertEvent(db, claude(D, "%3", "turn-stopped", "claude-stop", { capsule_status: "done" }), at(1));
    expect(hubOf(db).byProject[P].split).toBeUndefined();
    upsertSession(db, { project: P, session: "acme-lead", pane: "%5", role: "lead", tmux_incarnation: INC }, at(0));
    expect(hubOf(db).byProject[P].split?.seats).toHaveLength(1);
    db.close();
  });

  it("scopes each seat's runs, heartbeat, last action and newest timestamp, and unlinked keeps the rest", () => {
    seedTwoSessions(db);
    insertEvent(db, runStarted("run-a", "builder", "build", "claude", A, "feat/a"), at(3));
    insertEvent(db, runStarted("run-b", "builder", "build", "claude", B, "feat/b"), at(4));
    insertEvent(db, runStarted("run-d", "builder", "build", "claude", D, "feat/d"), at(5)); // D closed
    insertEvent(db, runStarted("run-j", "builder", "build", "jaxos", "jaxos", "feat/j"), at(6));
    insertEvent(db, claude(A, "%1", "tool-used", "claude-toolused", { tool: "Read" }), at(T0 ? 18 : 18));
    insertEvent(db, claude(B, "%2", "tool-used", "claude-toolused", { tool: "Bash" }), at(15));
    const split = getHubData(db, LIVE, {}, CODEX, NOW + 120_000).byProject[P].split!;
    const seat = (id: string) => split.seats.find((s) => s.sessionId === id)!;
    expect(seat(A).activity.activeRuns.map((r) => r.runId)).toEqual(["run-a"]);
    expect(seat(B).activity.activeRuns.map((r) => r.runId)).toEqual(["run-b"]);
    expect(split.unlinked.activeRuns.map((r) => r.runId).sort()).toEqual(["run-d", "run-j"]);
    expect(seat(A).activity.lastAction?.tool).toBe("Read");
    expect(seat(B).activity.lastAction?.tool).toBe("Bash");
    expect(seat(A).activity.heartbeat.reduce((a, b) => a + b, 0)).toBe(3); // turn-started + run-started + tool-used
    expect(seat(B).activity.heartbeat.reduce((a, b) => a + b, 0)).toBe(3);
    expect(seat(A).activity.newestEventTs! > seat(B).activity.newestEventTs!).toBe(true);
    db.close();
  });
});

describe("session status inputs", () => {
  const seatOf = (hub: HubEnvelopeData, id: string) => hub.byProject[P].split!.seats.find((s) => s.sessionId === id)!;

  it("drives stage/branch from the newest owned run and Now from the newest finished one", () => {
    seedTwoSessions(db);
    insertEvent(db, runStarted("r-spec", "builder", "spec", "claude", A, "docs/s.md"), at(1));
    insertEvent(db, runFinished("r-spec", "builder", { result: "success", exit_code: 0, report_path: "/r/s.md", summary: "spec ok" }), at(2));
    insertEvent(db, runStarted("r-build", "builder", "build", "claude", A, "feat/a"), at(3));
    expect(seatOf(hubOf(db), A).activity.status).toEqual({ latestKind: "build", latestTarget: "feat/a", lastSummary: "spec ok" });
    db.close();
  });

  it("shows a spec target as its basename", () => {
    upsertSession(db, { project: P, session: "acme-lead", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, runStarted("r-spec", "builder", "spec", "claude", A, "docs/s.md"), at(2));
    expect(seatOf(hubOf(db), A).activity.status).toEqual({ latestKind: "spec", latestTarget: "s.md", lastSummary: null });
    db.close();
  });

  it("keeps the summary of a cancelled or interrupted run as Now", () => {
    seedTwoSessions(db);
    insertEvent(db, runStarted("r1", "builder", "build", "claude", A, "feat/1"), at(1));
    insertEvent(db, runFinished("r1", "builder", { contract_status: "cancelled", summary: "cancelled by owner" }), at(2));
    expect(seatOf(hubOf(db), A).activity.status?.lastSummary).toBe("cancelled by owner");
    insertEvent(db, runStarted("r2", "builder", "build", "claude", A, "feat/2"), at(3));
    insertEvent(db, runFinished("r2", "builder", { contract_status: "interrupted", summary: "worker interrupted by SIGHUP" }), at(4));
    expect(seatOf(hubOf(db), A).activity.status?.lastSummary).toBe("worker interrupted by SIGHUP");
    db.close();
  });

  it("a session with no owned runs has no status; an unidentified seat and unlinked are null", () => {
    seedTwoSessions(db);
    upsertSession(db, { project: P, session: "acme-lead", pane: "%5", role: "lead", tmux_incarnation: INC }, at(0));
    const hub = hubOf(db);
    expect(seatOf(hub, A).activity.status).toBeNull();
    expect(hub.byProject[P].split!.seats.find((s) => s.sessionId === null)!.activity.status).toBeNull();
    expect(hub.byProject[P].split!.unlinked.status).toBeNull();
    db.close();
  });

  it("ignores runs owned by another session, including duplicate starts", () => {
    seedTwoSessions(db);
    insertEvent(db, runStarted("r-r", "builder", "build", "claude", A, "feat/r"), at(1));
    insertEvent(db, runStarted("r-r", "builder", "build", "claude", B, "feat/r"), at(2));
    insertEvent(db, runFinished("r-r", "builder", { result: "success", exit_code: 0, report_path: "/r/r.md", summary: "from B" }), at(3));
    expect(seatOf(hubOf(db), A).activity.status).toBeNull();
    expect(seatOf(hubOf(db), B).activity.status).toEqual({ latestKind: "build", latestTarget: "feat/r", lastSummary: "from B" });
    db.close();
  });

  it("never shows an owner-less run in any seat's status", () => {
    seedTwoSessions(db);
    insertEvent(db, {
      run_id: "run-u", project: P, role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
      payload: { phase: "P1", runtime: "opencode-builder", kind: "build", target: "feat/u", caller: "claude" },
    }, at(1));
    insertEvent(db, runFinished("run-u", "builder", { result: "success", summary: "unowned" }), at(2));
    const hub = hubOf(db);
    expect(seatOf(hub, A).activity.status).toBeNull();
    expect(seatOf(hub, B).activity.status).toBeNull();
    db.close();
  });

  it("degrades corrupt inputs silently and caps a long summary", () => {
    seedTwoSessions(db);
    insertEvent(db, {
      run_id: "r-bad", project: P, role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
      payload: { phase: "P1", runtime: "opencode-builder", kind: "zzz", target: "feat/x", caller: "claude", caller_session: A },
    }, at(1));
    expect(seatOf(hubOf(db), A).activity.status).toEqual({ latestKind: null, latestTarget: null, lastSummary: null });
    insertEvent(db, runFinished("r-bad", "builder", { summary: 42 }), at(2));
    expect(seatOf(hubOf(db), A).activity.status?.lastSummary).toBeNull();
    insertEvent(db, runStarted("r-long", "builder", "build", "claude", A, "feat/long"), at(3));
    insertEvent(db, runFinished("r-long", "builder", { summary: "x".repeat(500) }), at(4));
    expect(seatOf(hubOf(db), A).activity.status?.lastSummary).toHaveLength(200);
    db.close();
  });

  it("returns the last finished run in unownedLastRun only while no open session owns it", () => {
    seedTwoSessions(db);
    insertEvent(db, runStarted("run-d", "builder", "build", "claude", D, "feat/d"), at(1));
    insertEvent(db, runFinished("run-d", "builder", { result: "success", exit_code: 0, report_path: "/r/d.md", summary: "done" }), at(2));
    expect(hubOf(db).byProject[P].split!.unownedLastRun?.runId).toBe("run-d");

    insertEvent(db, runStarted("run-a", "builder", "build", "claude", A, "feat/a"), at(3));
    insertEvent(db, runFinished("run-a", "builder", { result: "success", exit_code: 0, report_path: "/r/a.md", summary: "ok" }), at(4));
    expect(hubOf(db).byProject[P].split!.unownedLastRun).toBeNull();

    insertEvent(db, runStarted("run-next", "builder", "build", "claude", D, "feat/next"), at(5));
    expect(hubOf(db).byProject[P].split!.unownedLastRun).toBeNull();
    db.close();
  });

  it("returns an owner-less finished run in unownedLastRun", () => {
    upsertSession(db, { project: P, session: "acme-lead", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    insertEvent(db, claude(A, "%1", "turn-started", "claude-userprompt"), at(1));
    insertEvent(db, {
      run_id: "run-u", project: P, role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
      payload: { phase: "P1", runtime: "opencode-builder", kind: "build", target: "feat/u", caller: "claude" },
    }, at(2));
    insertEvent(db, runFinished("run-u", "builder", { result: "success", exit_code: 0, report_path: "/r/u.md", summary: "ok" }), at(3));
    expect(hubOf(db).byProject[P].split!.unownedLastRun?.runId).toBe("run-u");
    db.close();
  });

  it("puts registry names on identified seats only", () => {
    seedTwoSessions(db);
    upsertSession(db, { project: P, session: "acme-lead", pane: "%5", role: "lead", tmux_incarnation: INC }, at(0));
    const named = getHubData(db, LIVE, {}, CODEX, NOW, true, null, null, false, new Map([[A, "multi-sessao"]]));
    expect(named.byProject[P].split!.seats.map((s) => [s.sessionId, s.name])).toEqual([
      [A, "multi-sessao"], [B, null], [null, null],
    ]);
    expect(hubOf(db).byProject[P].split!.seats.every((s) => s.name === null)).toBe(true);
    db.close();
  });

  it("omits the split key entirely for a project with no seats and nothing in flight", () => {
    insertEvent(db, claude(A, "%3", "turn-stopped", "claude-stop", { capsule_status: "done" }), at(1));
    expect("split" in hubOf(db).byProject[P]).toBe(false);
    db.close();
  });
});

describe("getBuildRunSessions", () => {
  it("keeps one row per run, owned by the latest builder start", () => {
    insertEvent(db, runStarted("r-shared", "builder", "build", "claude", A, "feat/x"), at(1));
    insertEvent(db, runStarted("r-shared", "builder", "build", "claude", B, "feat/x"), at(2));
    expect(getBuildRunSessions(db, P)).toEqual([{ runId: "r-shared", session: B }]);
    db.close();
  });

  it("reports null for a builder run with no recorded session", () => {
    insertEvent(db, {
      run_id: "r-none", project: P, role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
      payload: { phase: "P1", runtime: "opencode-builder", kind: "build", target: "feat/n", caller: "claude" },
    }, at(1));
    expect(getBuildRunSessions(db, P)).toEqual([{ runId: "r-none", session: null }]);
    db.close();
  });

  it("excludes reviewer runs and jaxos-emitted rows, and keeps a dashboard dispatch's session", () => {
    insertEvent(db, runStarted("r-review", "reviewer", "diff", "claude", A, "feat/r"), at(1));
    insertEvent(db, { run_id: "r-jaxos-row", project: P, role: "builder", type: "run-started", source: "deterministic", emitter: "jaxos", payload: { kind: "build", target: "feat/j" } }, at(2));
    insertEvent(db, runStarted("r-dash", "builder", "build", "jaxos", "jaxos", "feat/j"), at(3));
    expect(getBuildRunSessions(db, P)).toEqual([{ runId: "r-dash", session: "jaxos" }]);
    db.close();
  });
});
