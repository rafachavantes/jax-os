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
  activeRunsFor, getHubData, getProjectTimeline, insertEvent, lastRunFor, questionOwnsPane, setAfk, setForwardTypes, upsertSession,
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
