import { describe, expect, it, vi } from "vitest";
import { openDb } from "../../../../server/db";
import { claimAnswer, failAnswerFinalization, finishAnswer, insertEvent } from "../../../../server/db/workflows";
import { injectAnswers } from "../../../../server/collectors/workflow-tmux";
import type { WorkflowEventInput } from "../../../../lib/workflow";
import { handleAnswerPost, type AnswerRouteDeps } from "./handler";

const ANSWERS = [
  { question_number: 1, kind: "options" as const, values: [2] },
  { question_number: 2, kind: "options" as const, values: [1, 3] },
  { question_number: 3, kind: "text" as const, value: "ship without deploy" },
];
const CLAIM = {
  question_event_id: 7,
  tool_use_id: "toolu_abc",
  kind: "structured" as const,
  project: "p1",
  pane: "%1",
  tmux_incarnation: "234790:1787586213",
  emitter: "claude-pretool",
  harnessSession: null,
  role: "lead" as const,
  mutation_id: 9,
  question_shapes: [
    { multiSelect: false, option_count: 2 },
    { multiSelect: true, option_count: 3 },
    { multiSelect: false, option_count: 2 },
  ],
  answers: ANSWERS,
  text: null,
};
const FREEFORM_CLAIM = {
  question_event_id: 1,
  tool_use_id: null,
  kind: "freeform" as const,
  project: "p1",
  pane: "%1",
  tmux_incarnation: "234790:1787586213",
  emitter: "claude-stop",
  harnessSession: "0191f0aa-1111-7000-8000-000000000001",
  role: "lead" as const,
  mutation_id: 1,
  question_shapes: null,
  answers: null,
  text: "ship it",
};
const CODEX_FREEFORM_CLAIM = {
  ...FREEFORM_CLAIM, pane: null, tmux_incarnation: null, emitter: "codex-stop",
  harnessSession: "0191f0aa-2222-7000-8000-000000000002",
};

function answerRequest(reply = "1: 2\n2: 1,3\n3: ship without deploy"): Request {
  return new Request("http://localhost/api/workflow/answer", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      question_event_id: CLAIM.question_event_id, reply,
    }),
  });
}

// Only `ownerName` matters to this route — 497A's real reader returns the full GeneralSettings,
// so a loose cast keeps every pre-existing "[Jax OS · Rafa] ..." assertion byte-identical.
const rafaSettings = (ownerName: string) =>
  (() => ({ ok: true, data: { ownerName } })) as unknown as AnswerRouteDeps["readGeneralSettings"];

function fakeDeps(overrides: Partial<AnswerRouteDeps> = {}): AnswerRouteDeps {
  return {
    getDb: vi.fn(() => ({} as ReturnType<AnswerRouteDeps["getDb"]>)),
    readGeneralSettings: rafaSettings("Rafa"),
    loadEventType: vi.fn(() => "question"),
    claimAnswer: vi.fn(() => ({ ok: true, claim: CLAIM })),
    claimFreeformAnswer: vi.fn(() => ({ ok: true, claim: FREEFORM_CLAIM })),
    injectAnswers: vi.fn(async () => {}),
    codexQueueMessage: vi.fn(async () => {}),
    isAnswerWatcherAlive: vi.fn(() => true),
    writeAnswerLine: vi.fn(() => {}),
    finishAnswer: vi.fn(),
    failAnswerFinalization: vi.fn(() => true),
    ...overrides,
  } as AnswerRouteDeps;
}

const NOW = new Date("2026-08-22T12:00:00.000Z");
const SENTINEL = "-ship; $(nope)";
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
  pane, tmux_incarnation: "234790:1787586213",
});

describe("handleAnswerPost", () => {
  it("blocks same-site and cross-site browser requests before opening the DB", async () => {
    const deps = fakeDeps();
    for (const fetchSite of ["same-site", "cross-site"]) {
      const request = new Request("http://localhost/api/workflow/answer", {
        method: "POST",
        headers: { "sec-fetch-site": fetchSite, "content-type": "text/plain" },
        body: JSON.stringify({}),
      });
      expect(await (await handleAnswerPost(request, deps)).json()).toEqual({
        ok: false, error: "cross-site request blocked",
      });
    }
    expect(deps.getDb).not.toHaveBeenCalled();
  });

  it("lets absent, same-origin, and none fetch-site values reach parsing", async () => {
    for (const fetchSite of [undefined, "same-origin", "none"]) {
      const deps = fakeDeps();
      const headers: Record<string, string> = { "content-type": "application/json" };
      if (fetchSite) headers["sec-fetch-site"] = fetchSite;
      const request = new Request("http://localhost/api/workflow/answer", {
        method: "POST",
        headers,
        body: JSON.stringify({ question_event_id: 7, reply: "1: 1" }),
      });
      await handleAnswerPost(request, deps);
      expect(deps.getDb).toHaveBeenCalled();
    }
  });

  it("orders claim, injection, and finish exactly once", async () => {
    const order: string[] = [];
    const deps = fakeDeps({
      claimAnswer: vi.fn(() => {
        order.push("claim");
        return { ok: true as const, claim: CLAIM };
      }),
      injectAnswers: vi.fn(async () => { order.push("inject"); }),
      finishAnswer: vi.fn(() => { order.push("finish"); }),
    });
    const response = await handleAnswerPost(answerRequest(), deps);
    expect(await response.json()).toMatchObject({ ok: true, data: { mutation_id: CLAIM.mutation_id } });
    expect(order).toEqual(["claim", "inject", "finish"]);
    expect(deps.injectAnswers).toHaveBeenCalledWith(CLAIM.pane, CLAIM.question_shapes, CLAIM.answers, CLAIM.tmux_incarnation);
  });

  it("finalizes a stale-pane failure and never retries injection", async () => {
    const deps = fakeDeps({ injectAnswers: vi.fn(async () => { throw new Error("pane target is not live"); }) });
    const response = await handleAnswerPost(answerRequest(), deps);
    expect(await response.json()).toEqual({ ok: false, error: "pane target is not live" });
    expect(deps.injectAnswers).toHaveBeenCalledTimes(1);
    expect(deps.finishAnswer).toHaveBeenCalledWith(expect.anything(), CLAIM, false, "pane target is not live");
  });

  it("reports finalization failure after one successful injection without reinjecting", async () => {
    const deps = fakeDeps({ finishAnswer: vi.fn(() => { throw new Error("db unavailable"); }) });
    const response = await handleAnswerPost(answerRequest(), deps);
    expect(await response.json()).toEqual({ ok: false, error: "answer finalization failed" });
    expect(deps.injectAnswers).toHaveBeenCalledTimes(1);
    expect(deps.failAnswerFinalization).toHaveBeenCalledWith(expect.anything(), CLAIM);
  });

  it("returns the house envelope for malformed JSON", async () => {
    const deps = fakeDeps();
    const request = new Request("http://localhost/api/workflow/answer", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "not-json",
    });
    expect(await (await handleAnswerPost(request, deps)).json()).toEqual({
      ok: false, error: "invalid JSON",
    });
    expect(deps.getDb).not.toHaveBeenCalled();
  });

  it("returns the house envelope for not-pending and invalid-target without injection", async () => {
    for (const reason of ["not-pending", "invalid-target"] as const) {
      const deps = fakeDeps({ claimAnswer: vi.fn(() => ({ ok: false as const, reason })) });
      expect(await (await handleAnswerPost(answerRequest(), deps)).json()).toEqual({
        ok: false, error: reason,
      });
      expect(deps.injectAnswers).not.toHaveBeenCalled();
      expect(deps.finishAnswer).not.toHaveBeenCalled();
    }
  });

  it("maps a sentinel-bearing getDb throw to answer claim failed", async () => {
    const deps = fakeDeps({
      getDb: vi.fn(() => { throw new Error(`boom ${SENTINEL}`); }),
    });
    const response = await handleAnswerPost(answerRequest(), deps);
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ok: false, error: "answer claim failed" });
    expect(deps.injectAnswers).not.toHaveBeenCalled();
    expect(deps.finishAnswer).not.toHaveBeenCalled();
    expect(deps.failAnswerFinalization).not.toHaveBeenCalled();
  });
});

describe("handleAnswerPost — generalized dispatch (D4)", () => {
  function req(body: unknown) {
    return new Request("http://x", { method: "POST", headers: { "sec-fetch-site": "none" }, body: JSON.stringify(body) });
  }

  it("routes a question-typed event through claimAnswer + injectAnswers and returns kind", async () => {
    const deps = fakeDeps({ loadEventType: vi.fn(() => "question") });
    const res = await handleAnswerPost(req({ event_id: 1, reply: "1: 1" }), deps);
    expect(await res.json()).toEqual({
      ok: true,
      data: { question_event_id: CLAIM.question_event_id, tool_use_id: CLAIM.tool_use_id, mutation_id: CLAIM.mutation_id, kind: "structured" },
    });
    expect(deps.claimAnswer).toHaveBeenCalled();
    expect(deps.claimFreeformAnswer).not.toHaveBeenCalled();
    expect(deps.injectAnswers).toHaveBeenCalledWith("%1", CLAIM.question_shapes, CLAIM.answers, "234790:1787586213");
  });

  it("routes a claude-stop freeform claim through writeAnswerLine, never codexQueueMessage or tmux", async () => {
    const deps = fakeDeps({ loadEventType: vi.fn(() => "turn-stopped") });
    const res = await handleAnswerPost(req({ event_id: 1, reply: "ship it" }), deps);
    expect((await res.json()).ok).toBe(true);
    expect(deps.claimFreeformAnswer).toHaveBeenCalled();
    expect(deps.claimAnswer).not.toHaveBeenCalled();
    expect(deps.writeAnswerLine).toHaveBeenCalledWith(FREEFORM_CLAIM.harnessSession, FREEFORM_CLAIM.question_event_id, "[Jax OS · Rafa] ship it");
    expect(deps.codexQueueMessage).not.toHaveBeenCalled();
    expect(deps.injectAnswers).not.toHaveBeenCalled();
  });

  it("routes a codex-stop freeform claim through codexQueueMessage, never writeAnswerLine or tmux", async () => {
    const deps = fakeDeps({
      loadEventType: vi.fn(() => "turn-stopped"),
      claimFreeformAnswer: vi.fn(() => ({ ok: true as const, claim: CODEX_FREEFORM_CLAIM })),
    });
    const res = await handleAnswerPost(req({ event_id: 1, reply: "ship it" }), deps);
    expect((await res.json()).ok).toBe(true);
    expect(deps.codexQueueMessage).toHaveBeenCalledWith(CODEX_FREEFORM_CLAIM.harnessSession, "[Jax OS · Rafa] ship it");
    expect(deps.writeAnswerLine).not.toHaveBeenCalled();
    expect(deps.injectAnswers).not.toHaveBeenCalled();
  });

  it("never prefixes the structured AskUserQuestion path — only the two native freeform deliveries get [Jax OS · Rafa] (D5)", async () => {
    const deps = fakeDeps({ loadEventType: vi.fn(() => "question") });
    await handleAnswerPost(answerRequest(), deps);
    expect(deps.injectAnswers).toHaveBeenCalledWith(CLAIM.pane, CLAIM.question_shapes, CLAIM.answers, CLAIM.tmux_incarnation);
  });

  it("a claude-stop freeform claim whose watcher marker is absent fails the click instead of writing an unread file (F1)", async () => {
    const deps = fakeDeps({ loadEventType: vi.fn(() => "turn-stopped"), isAnswerWatcherAlive: vi.fn(() => false) });
    const res = await handleAnswerPost(req({ event_id: 1, reply: "ship it" }), deps);
    expect(await res.json()).toEqual({ ok: false, error: "answer watcher not live" });
    expect(deps.writeAnswerLine).not.toHaveBeenCalled();
  });

  it("a claude-stop freeform claim whose watcher marker is alive delivers through writeAnswerLine (F1)", async () => {
    const deps = fakeDeps({ loadEventType: vi.fn(() => "turn-stopped"), isAnswerWatcherAlive: vi.fn(() => true) });
    const res = await handleAnswerPost(req({ event_id: 1, reply: "ship it" }), deps);
    expect((await res.json()).ok).toBe(true);
    expect(deps.isAnswerWatcherAlive).toHaveBeenCalledWith(FREEFORM_CLAIM.harnessSession, FREEFORM_CLAIM.question_event_id);
    expect(deps.writeAnswerLine).toHaveBeenCalledWith(FREEFORM_CLAIM.harnessSession, FREEFORM_CLAIM.question_event_id, "[Jax OS · Rafa] ship it");
  });

  it("rejects an attention-needed source as not-answerable before any claim or tmux call (Finding 26)", async () => {
    const deps = fakeDeps({ loadEventType: vi.fn(() => "attention-needed") });
    const res = await handleAnswerPost(req({ event_id: 1, reply: "approve it" }), deps);
    expect(await res.json()).toEqual({ ok: false, error: "not-answerable" });
    expect(deps.claimAnswer).not.toHaveBeenCalled();
    expect(deps.claimFreeformAnswer).not.toHaveBeenCalled();
    expect(deps.injectAnswers).not.toHaveBeenCalled();
    expect(deps.codexQueueMessage).not.toHaveBeenCalled();
    expect(deps.writeAnswerLine).not.toHaveBeenCalled();
  });

  it("rejects a missing event id as not-answerable", async () => {
    const deps = fakeDeps({ loadEventType: vi.fn(() => null) });
    const res = await handleAnswerPost(req({ event_id: 999, reply: "x" }), deps);
    expect(await res.json()).toEqual({ ok: false, error: "not-answerable" });
  });

  it("surfaces a D2 guard rejection reason as the error field", async () => {
    const deps = fakeDeps({
      loadEventType: vi.fn(() => "turn-stopped"),
      claimFreeformAnswer: vi.fn(() => ({ ok: false as const, reason: "reply must not start with /" as const })),
    });
    const res = await handleAnswerPost(req({ event_id: 1, reply: "/nope" }), deps);
    expect(await res.json()).toEqual({ ok: false, error: "reply must not start with /" });
  });

  it("empty ownerName drops the name from the prefix (decision 11)", async () => {
    const deps = fakeDeps({
      loadEventType: vi.fn(() => "turn-stopped"),
      readGeneralSettings: rafaSettings(""),
      writeAnswerLine: vi.fn(),
    });
    await handleAnswerPost(req({ event_id: 1, reply: "ship it" }), deps);
    expect(deps.writeAnswerLine).toHaveBeenCalledWith(expect.anything(), expect.anything(), "[Jax OS] ship it");
  });

  it("a settings read failure also falls back to no name (never throws)", async () => {
    const deps = fakeDeps({
      loadEventType: vi.fn(() => "turn-stopped"),
      readGeneralSettings: (() => ({ ok: false, error: "settings-unreadable" })) as unknown as AnswerRouteDeps["readGeneralSettings"],
      writeAnswerLine: vi.fn(),
    });
    await handleAnswerPost(req({ event_id: 1, reply: "ship it" }), deps);
    expect(deps.writeAnswerLine).toHaveBeenCalledWith(expect.anything(), expect.anything(), "[Jax OS] ship it");
  });
});

describe("handleAnswerPost integration", () => {
  it("sanitizes a collector execFile failure into durable rows", async () => {
    const db = openDb(":memory:");
    const q = insertEvent(db, batchQuestion("t-int"), NOW);
    const run = vi.fn(async (_file: string, args: string[]) => {
      if (args[0] === "list-panes") return "%1 234790 1787586213 bash\n";
      throw Object.assign(new Error(`Command failed: tmux ${args.join(" ")} ${SENTINEL}`), {
        cmd: `tmux send-keys ${SENTINEL}`,
        stdout: SENTINEL,
        stderr: SENTINEL,
      });
    });
    const deps: AnswerRouteDeps = {
      getDb: () => db,
      readGeneralSettings: rafaSettings("Rafa"),
      loadEventType: (d, id) => {
        const row = d.prepare("SELECT type FROM workflow_events WHERE id = ?").get(id) as { type: string } | undefined;
        return row?.type ?? null;
      },
      claimAnswer,
      claimFreeformAnswer: vi.fn(),
      injectAnswers: (pane, shapes, answers, tmuxIncarnation) => injectAnswers(pane, shapes, answers, tmuxIncarnation, run, async () => {}),
      codexQueueMessage: vi.fn(async () => {}),
      isAnswerWatcherAlive: vi.fn(() => false),
      writeAnswerLine: vi.fn(),
      finishAnswer,
      failAnswerFinalization,
    };
    const response = await handleAnswerPost(
      new Request("http://localhost/api/workflow/answer", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question_event_id: q.id, reply: `1: 2\n2: 1,3\n3: ${SENTINEL}` }),
      }),
      deps,
    );
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ ok: false, error: "tmux answer injection failed" });
    const mutation = db.prepare("SELECT error, payload FROM mutations").get() as { error: string; payload: string };
    expect(mutation.error).toBe("tmux answer injection failed");
    const answered = db.prepare("SELECT payload FROM workflow_events WHERE type = 'question-answered'").get() as { payload: string };
    expect(JSON.parse(answered.payload).error).toBe("tmux answer injection failed");
    const durable = JSON.stringify({
      mutations: db.prepare("SELECT * FROM mutations").all(),
      events: db.prepare("SELECT * FROM workflow_events").all(),
      claims: db.prepare("SELECT * FROM workflow_answer_claims").all(),
    });
    expect(durable).not.toContain(SENTINEL);
    db.close();
  });

  it("terminalizes only the named claim when finishAnswer throws after claim", async () => {
    const db = openDb(":memory:");
    const a = insertEvent(db, batchQuestion("t-fin-a"), NOW);
    const b = insertEvent(db, { ...batchQuestion("t-fin-b", "%2"), project: "p2" }, NOW);
    const second = claimAnswer(db, { question_event_id: b.id, reply: "1: 1\n2: 1\n3: keep" }, NOW);
    expect(second.ok).toBe(true);
    const deps: AnswerRouteDeps = {
      getDb: () => db,
      readGeneralSettings: rafaSettings("Rafa"),
      loadEventType: (d, id) => {
        const row = d.prepare("SELECT type FROM workflow_events WHERE id = ?").get(id) as { type: string } | undefined;
        return row?.type ?? null;
      },
      claimAnswer,
      claimFreeformAnswer: vi.fn(),
      injectAnswers: vi.fn(async () => {}),
      codexQueueMessage: vi.fn(async () => {}),
      isAnswerWatcherAlive: vi.fn(() => false),
      writeAnswerLine: vi.fn(),
      finishAnswer: () => { throw new Error(`finish ${SENTINEL}`); },
      failAnswerFinalization,
    };
    const response = await handleAnswerPost(
      new Request("http://localhost/api/workflow/answer", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question_event_id: a.id, reply: "1: 2\n2: 1,3\n3: note" }),
      }),
      deps,
    );
    expect(await response.json()).toEqual({ ok: false, error: "answer finalization failed" });
    expect(deps.injectAnswers).toHaveBeenCalledTimes(1);
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-fin-a'").get()).toEqual({
      status: "failed",
    });
    expect(db.prepare("SELECT status FROM workflow_answer_claims WHERE tool_use_id = 't-fin-b'").get()).toEqual({
      status: "injecting",
    });
    expect(db.prepare("SELECT COUNT(*) AS c FROM workflow_events WHERE type = 'question-answered'").get()).toEqual({ c: 0 });
    db.close();
  });
});
