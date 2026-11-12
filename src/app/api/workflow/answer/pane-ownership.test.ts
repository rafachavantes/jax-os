// A structured answer is TYPED into the question's tmux pane, so it may only be accepted while the
// conversation that asked still owns that exact pane (pane + tmux incarnation) — in ANY project.
// These go through the real route with the real claim and the real injector; only tmux itself is
// replaced by a spy, so "no input was sent" is asserted on the actual send-keys calls.
import { describe, expect, it, vi } from "vitest";
import { openDb } from "../../../../server/db";
import { claimAnswer, claimFreeformAnswer, failAnswerFinalization, finishAnswer, insertEvent, questionOwnsPane, upsertSession } from "../../../../server/db/workflows";
import { injectAnswers } from "../../../../server/collectors/workflow-tmux";
import type { WorkflowEventInput } from "../../../../lib/workflow";
import { handleAnswerPost, type AnswerRouteDeps } from "./handler";

const INC = "234790:1787586213";
const NEW_INC = "555001:1799999999"; // the same tmux server after a restart
const SA = "aaaaaaaa-0000-4000-8000-00000000000a"; // conversation in project A that asked
const SB = "bbbbbbbb-0000-4000-8000-00000000000b"; // new conversation the pane started in project B
const T0 = Date.parse("2026-09-20T12:00:00.000Z");
const at = (min: number) => new Date(T0 + min * 60_000);

type Db = ReturnType<typeof openDb>;

function row(project: string, session: string | null, type: WorkflowEventInput["type"], emitter: WorkflowEventInput["emitter"], payload: Record<string, unknown> = {}, inc = INC): WorkflowEventInput {
  return { run_id: null, project, role: "lead", type, source: "deterministic", emitter, payload, pane: "%1", tmux_incarnation: inc, harness_session: session };
}
const askRow = (project: string, session: string | null) => row(project, session, "question", "claude-pretool", {
  tool_use_id: `tu-${project}`, questions: [{ question: "Deploy to staging?", options: [{ label: "yes" }, { label: "no" }] }],
});
const ask = (project: string, session: string) => askRow(project, session);

// tmux stand-in: pane %1 is live under `liveInc`; every send-keys is recorded.
function tmux(liveInc = INC) {
  const [pid, start] = liveInc.split(":");
  const calls: string[][] = [];
  const run = vi.fn(async (_file: string, args: string[]) => {
    calls.push(args);
    if (args[0] === "list-panes") return `%1 ${pid} ${start} claude\n`;
    return "";
  });
  return { run, sendKeys: () => calls.filter((a) => a[0] === "send-keys") };
}

function deps(db: Db, run: ReturnType<typeof tmux>["run"]) {
  const inject = vi.fn((pane: string, shapes: Parameters<typeof injectAnswers>[1], answers: Parameters<typeof injectAnswers>[2], inc: string) =>
    injectAnswers(pane, shapes, answers, inc, run, async () => {}));
  const d: AnswerRouteDeps = {
    getDb: () => db,
    readGeneralSettings: (() => ({ ok: true, data: { ownerName: "" } })) as unknown as AnswerRouteDeps["readGeneralSettings"],
    loadEventType: (x, id) => (x.prepare("SELECT type FROM workflow_events WHERE id = ?").get(id) as { type: string } | undefined)?.type ?? null,
    claimAnswer, claimFreeformAnswer, injectAnswers: inject,
    codexQueueMessage: vi.fn(async () => {}), isAnswerWatcherAlive: vi.fn(() => false), writeAnswerLine: vi.fn(),
    finishAnswer, failAnswerFinalization,
  };
  return { d, inject };
}

async function answer(db: Db, eventId: number, t = tmux()) {
  const { d, inject } = deps(db, t.run);
  const res = await handleAnswerPost(new Request("http://localhost/api/workflow/answer", {
    method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ event_id: eventId, reply: "1: 1" }),
  }), d);
  const claims = (db.prepare("SELECT COUNT(*) AS n FROM workflow_answer_claims").get() as { n: number }).n;
  const mutations = (db.prepare("SELECT COUNT(*) AS n FROM mutations").get() as { n: number }).n;
  return { body: await res.json(), inject, sendKeys: t.sendKeys(), claims, mutations };
}

// Project A's conversation SA, on pane %1, asks a question and then the pane moves on.
function seedQuestionInA(db: Db) {
  upsertSession(db, { project: "proj-a", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
  insertEvent(db, row("proj-a", SA, "turn-started", "claude-userprompt"), at(1));
  return insertEvent(db, ask("proj-a", SA), at(2));
}

describe("answer route — the asking conversation must still own the exact pane, across projects", () => {
  it("rejects project A's old question once the same pane started a new session in project B, before any input", async () => {
    const db = openDb(":memory:");
    const q = seedQuestionInA(db);
    // Same pane, same tmux incarnation: `cd` to project B and a NEW conversation starts there.
    upsertSession(db, { project: "proj-b", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(3));
    insertEvent(db, row("proj-b", SB, "turn-started", "claude-userprompt"), at(3));
    const r = await answer(db, q.id);
    expect(r.body).toEqual({ ok: false, error: "invalid-target" });
    expect(r.inject).not.toHaveBeenCalled();
    expect(r.sendKeys).toEqual([]);
    expect(r.claims).toBe(0); // refused before anything was claimed or audited
    expect(r.mutations).toBe(0);
    // Project A's own events alone say nothing happened after the question — the gap this closes.
    expect(db.prepare("SELECT COUNT(*) AS n FROM workflow_events WHERE project = 'proj-a' AND id > ?").get(q.id)).toEqual({ n: 0 });
    db.close();
  });

  it("rejects a new conversation in the SAME project and pane (/clear)", async () => {
    const db = openDb(":memory:");
    const q = seedQuestionInA(db);
    insertEvent(db, row("proj-a", SB, "turn-started", "claude-userprompt"), at(3));
    const r = await answer(db, q.id);
    expect(r.body).toEqual({ ok: false, error: "invalid-target" });
    expect(r.sendKeys).toEqual([]);
    expect(r.claims).toBe(0);
    expect(r.mutations).toBe(0);
    db.close();
  });

  it("rejects it too when only project B's registration landed (its event never did)", async () => {
    const db = openDb(":memory:");
    const q = seedQuestionInA(db);
    upsertSession(db, { project: "proj-b", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(3));
    const r = await answer(db, q.id);
    expect(r.body).toEqual({ ok: false, error: "invalid-target" });
    expect(r.sendKeys).toEqual([]);
    db.close();
  });

  it("a question from the conversation that still owns the pane stays answerable, and its input is sent", async () => {
    const db = openDb(":memory:");
    const q = seedQuestionInA(db);
    insertEvent(db, row("proj-a", SA, "tool-used", "claude-toolused", { tool: "Read" }), at(3)); // same conversation, still there
    const r = await answer(db, q.id);
    expect(r.body).toMatchObject({ ok: true, data: { question_event_id: q.id, kind: "structured" } });
    expect(r.inject).toHaveBeenCalledOnce();
    expect(r.sendKeys.length).toBeGreaterThan(0);
    expect(r.sendKeys.every((a) => a[2] === "%1")).toBe(true);
    db.close();
  });

  it("rejects it once the SAME conversation is now emitting in another project (section 8.1 condition 3)", async () => {
    const db = openDb(":memory:");
    const q = seedQuestionInA(db);
    // The conversation id survives, but the pane now belongs to project B.
    upsertSession(db, { project: "proj-b", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(3));
    insertEvent(db, row("proj-b", SA, "tool-used", "claude-toolused", { tool: "Read" }), at(3.1));
    const r = await answer(db, q.id);
    expect(r.body).toEqual({ ok: false, error: "invalid-target" });
    expect(r.sendKeys).toEqual([]);
    expect(r.claims).toBe(0);
    // A question asked in project B by that same conversation, with the pane now current there, IS answerable.
    const qb = insertEvent(db, ask("proj-b", SA), at(3.2));
    const rb = await answer(db, qb.id);
    expect(rb.body).toMatchObject({ ok: true, data: { question_event_id: qb.id } });
    expect(rb.sendKeys.length).toBeGreaterThan(0);
    db.close();
  });

  it("keeps legacy questions (no recorded session) answerable even when another session posted later on the pane", async () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "proj-a", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(0));
    const q = insertEvent(db, askRow("proj-a", null), at(2));
    insertEvent(db, row("proj-a", SB, "turn-started", "claude-userprompt"), at(3));
    const r = await answer(db, q.id);
    expect(r.body).toMatchObject({ ok: true, data: { question_event_id: q.id } });
    expect(r.sendKeys.length).toBeGreaterThan(0);
    db.close();
  });

  it("keeps the tmux incarnation check: after a tmux restart the old pane id is refused, with no input", async () => {
    const db = openDb(":memory:");
    const q = seedQuestionInA(db);
    // The tmux server restarted: %1 now exists under a new incarnation, hosting project B's session.
    insertEvent(db, row("proj-b", SB, "turn-started", "claude-userprompt", {}, NEW_INC), at(3));
    const r = await answer(db, q.id, tmux(NEW_INC));
    expect(r.body).toEqual({ ok: false, error: "pane target is not live" });
    expect(r.sendKeys).toEqual([]);
    db.close();
  });
});

describe("questionOwnsPane — the shared rule (spec §8.1 conditions 1 and 3)", () => {
  const ref = { pane: "%1", tmuxIncarnation: INC };
  const q = (project: string, harnessSession: string | null) => ({ id: 1, project, ref, harnessSession });

  it("fails closed when no agent row exists for the pane key", () => {
    const db = openDb(":memory:");
    expect(questionOwnsPane(db, q("proj-a", SA))).toBe(false);
    db.close();
  });

  it("decides by the current project when a different-project registration predates the newest row", () => {
    const db = openDb(":memory:");
    upsertSession(db, { project: "proj-b", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(1));
    insertEvent(db, row("proj-a", SA, "turn-started", "claude-userprompt"), at(2)); // newer than the registration
    expect(questionOwnsPane(db, q("proj-a", SA))).toBe(true);
    db.close();
  });

  it("fails when a different-project registration landed after the newest row but its event never did", () => {
    const db = openDb(":memory:");
    insertEvent(db, row("proj-a", SA, "turn-started", "claude-userprompt"), at(1));
    upsertSession(db, { project: "proj-b", session: "work", pane: "%1", role: "lead", tmux_incarnation: INC }, at(2));
    expect(questionOwnsPane(db, q("proj-a", SA))).toBe(false);
    db.close();
  });

  it("never checks rows from another pane key or incarnation", () => {
    const db = openDb(":memory:");
    insertEvent(db, row("proj-a", SA, "turn-started", "claude-userprompt"), at(1));
    expect(questionOwnsPane(db, { ...q("proj-a", SA), ref: { pane: "%2", tmuxIncarnation: INC } })).toBe(false);
    expect(questionOwnsPane(db, { ...q("proj-a", SA), ref: { pane: "%1", tmuxIncarnation: NEW_INC } })).toBe(false);
    db.close();
  });
});
