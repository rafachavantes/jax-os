import { describe, expect, it } from "vitest";
import { openDb } from "./index";
import { insertEvent, setAfk } from "./workflows";
import { deriveCapsule, hasOpenQuestion } from "./derive";
import { readGeneralSettings } from "../settings";
import type { WorkflowEventInput } from "../../lib/workflow";

const INC = "234790:1787586213";
const STOP: WorkflowEventInput = {
  run_id: null, project: "p1", role: "lead", type: "turn-stopped",
  source: "behavioral", emitter: "claude-stop", payload: { message_tail: "prose" },
  pane: "%1", tmux_incarnation: INC, harness_session: "sess-a",
};

it("rung 1: an event that arrived tagged is believed and no further rung runs", () => {
  const db = openDb(":memory:");
  const d = deriveCapsule(db, { ...STOP, source: "deterministic", payload: { capsule_status: "done", capsule_rule: "tag" } });
  expect(d).toEqual({ capsule_status: "done", capsule_rule: "tag", source: "deterministic" });
  db.close();
});

it("rung 1 does not fire on a tag with an unusable status — it falls through", () => {
  const db = openDb(":memory:");
  const d = deriveCapsule(db, { ...STOP, payload: { capsule_rule: "tag" } });
  expect(d.capsule_rule).not.toBe("tag");
  expect(d.capsule_status).toBe("unknown");
  db.close();
});

it("rung 1 also believes the hook's merge-question shape: needs_input, rule merge-question, deterministic", () => {
  const db = openDb(":memory:");
  const d = deriveCapsule(db, { ...STOP, source: "deterministic", payload: {
    capsule_status: "needs_input", capsule_rule: "merge-question", capsule_attempts: 0,
    merge_ask: 1, merge_branch: "feat/x", merge_target: "main", merge_head_sha: null } });
  expect(d).toEqual({ capsule_status: "needs_input", capsule_rule: "merge-question", source: "deterministic" });
  db.close();
});

it("rung 2: an unresolved question on the same pane yields needs_input", () => {
  const db = openDb(":memory:");
  setAfk(db, true);
  insertEvent(db, { ...STOP, type: "question", source: "deterministic", emitter: "claude-pretool",
                    payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } });
  expect(deriveCapsule(db, STOP)).toMatchObject({ capsule_status: "needs_input", capsule_rule: "question_open" });
  db.close();
});

it("rung 2 runs PRE-insert: deriving after the row exists could never fire it", () => {
  const db = openDb(":memory:");
  setAfk(db, true);
  insertEvent(db, { ...STOP, type: "question", source: "deterministic", emitter: "claude-pretool",
                    payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } });
  const before = deriveCapsule(db, STOP);
  insertEvent(db, STOP);
  const after = deriveCapsule(db, STOP);
  expect(before.capsule_rule).toBe("question_open");
  expect(after.capsule_rule).not.toBe("question_open");
  db.close();
});

it("rung 3: an attention-needed newer than the last turn-started yields blocked", () => {
  const db = openDb(":memory:");
  insertEvent(db, { ...STOP, type: "turn-started", source: "deterministic", emitter: "claude-userprompt", payload: {} });
  insertEvent(db, { ...STOP, type: "attention-needed", emitter: "claude-notification", payload: { reason: "permission" } });
  expect(deriveCapsule(db, STOP)).toMatchObject({ capsule_status: "blocked", capsule_rule: "attention" });
  db.close();
});

it("rung 3 with no turn-started at all still fires — the bound is optional", () => {
  const db = openDb(":memory:");
  insertEvent(db, { ...STOP, type: "attention-needed", emitter: "claude-notification", payload: { reason: "permission" } });
  expect(deriveCapsule(db, STOP)).toMatchObject({ capsule_rule: "attention" });
  db.close();
});

it("an attention-needed OLDER than the last turn-started does not fire", () => {
  const db = openDb(":memory:");
  insertEvent(db, { ...STOP, type: "attention-needed", emitter: "claude-notification", payload: { reason: "permission" } });
  insertEvent(db, { ...STOP, type: "turn-started", source: "deterministic", emitter: "claude-userprompt", payload: {} });
  expect(deriveCapsule(db, STOP).capsule_rule).not.toBe("attention");
  db.close();
});

it("2 beats 3: an open question outranks an older attention", () => {
  const db = openDb(":memory:");
  setAfk(db, true);
  insertEvent(db, { ...STOP, type: "attention-needed", emitter: "claude-notification", payload: { reason: "permission" } });
  insertEvent(db, { ...STOP, type: "question", source: "deterministic", emitter: "claude-pretool",
                    payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } });
  expect(deriveCapsule(db, STOP).capsule_rule).toBe("question_open");
  db.close();
});

const RUN_STARTED = (over: Record<string, unknown> = {}) => ({
  run_id: "r1", project: "p1", role: "builder" as const, type: "run-started" as const,
  source: "deterministic" as const, emitter: "wrapper" as const,
  payload: { phase: "B", runtime: "codex", caller_session: "sess-a" }, ...over,
});

it("rung 4: a run started from this session with no finish yields waiting", () => {
  const db = openDb(":memory:");
  insertEvent(db, RUN_STARTED());
  expect(deriveCapsule(db, STOP)).toMatchObject({ capsule_status: "waiting", capsule_rule: "run_in_flight" });
  db.close();
});

it("rung 5: the matching finish yields done", () => {
  const db = openDb(":memory:");
  insertEvent(db, RUN_STARTED());
  insertEvent(db, { ...RUN_STARTED(), type: "run-finished", payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x", summary: "s", head_sha: null } });
  expect(deriveCapsule(db, STOP)).toMatchObject({ capsule_status: "done", capsule_rule: "run_finished" });
  db.close();
});

it("a finish for a DIFFERENT run_id leaves the first run waiting", () => {
  const db = openDb(":memory:");
  insertEvent(db, RUN_STARTED());
  insertEvent(db, { ...RUN_STARTED({ run_id: "r2" }), type: "run-finished", payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x", summary: "s", head_sha: null } });
  expect(deriveCapsule(db, STOP).capsule_rule).toBe("run_in_flight");
  db.close();
});

it("a run dispatched from ANOTHER session is invisible to this pane", () => {
  const db = openDb(":memory:");
  insertEvent(db, RUN_STARTED({ payload: { phase: "B", runtime: "codex", caller_session: "sess-other" } }));
  expect(deriveCapsule(db, STOP, classifierOn).capsule_rule).toBe("deferred");
  db.close();
});

it("an event with no harness_session skips 4 and 5 entirely", () => {
  const db = openDb(":memory:");
  insertEvent(db, RUN_STARTED());
  expect(deriveCapsule(db, { ...STOP, harness_session: null }, classifierOn).capsule_rule).toBe("deferred");
  db.close();
});

it("the Codex gate: runs without a verified caller never pair for codex-stop", () => {
  const db = openDb(":memory:");
  insertEvent(db, RUN_STARTED()); // no `caller` key at all — legacy shape
  expect(deriveCapsule(db, STOP).capsule_rule).toBe("run_in_flight"); // Claude keeps legacy pairing
  expect(deriveCapsule(db, { ...STOP, emitter: "codex-stop" }, classifierOn).capsule_rule).toBe("deferred"); // codex requires caller=codex
  db.close();
});

it("codex-stop pairs only with a caller=codex run in the SAME project", () => {
  const UUID = "0191f0aa-9999-7000-8000-000000000009";
  const codexStop = { ...STOP, emitter: "codex-stop" as const, harness_session: UUID };
  const run = (over: Record<string, unknown> = {}) => ({
    run_id: "r-codex", project: "p1", role: "builder" as const, type: "run-started" as const,
    source: "deterministic" as const, emitter: "wrapper" as const,
    payload: { phase: "B", runtime: "codex", caller: "codex", caller_session: UUID }, ...over,
  });
  const db = openDb(":memory:");
  insertEvent(db, run());
  expect(deriveCapsule(db, codexStop).capsule_rule).toBe("run_in_flight");
  db.close();

  // A run dispatched by Claude with the SAME UUID must never pair with the codex-stop.
  const db2 = openDb(":memory:");
  insertEvent(db2, run({ payload: { phase: "B", runtime: "codex", caller: "claude", caller_session: UUID } }));
  expect(deriveCapsule(db2, codexStop, classifierOn).capsule_rule).toBe("deferred");
  db2.close();

  // A codex run in ANOTHER project is invisible to this session.
  const db3 = openDb(":memory:");
  insertEvent(db3, run({ project: "p2" }));
  expect(deriveCapsule(db3, codexStop, classifierOn).capsule_rule).toBe("deferred");
  db3.close();
});

it("a UUID-shaped harness_session on a Claude emitter never pairs with a codex run", () => {
  const UUID = "0191f0aa-aaaa-7000-8000-00000000000a";
  const db = openDb(":memory:");
  insertEvent(db, {
    run_id: "r1", project: "p1", role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
    payload: { phase: "B", runtime: "codex", caller: "codex", caller_session: UUID },
  });
  expect(deriveCapsule(db, { ...STOP, harness_session: UUID }, classifierOn).capsule_rule).toBe("deferred");
  db.close();
});

it("2 and 3 still beat 4: an open question outranks a run in flight", () => {
  const db = openDb(":memory:");
  setAfk(db, true);
  insertEvent(db, RUN_STARTED());
  insertEvent(db, { ...STOP, type: "question", source: "deterministic", emitter: "claude-pretool",
                    payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } });
  expect(deriveCapsule(db, STOP).capsule_rule).toBe("question_open");
  db.close();
});

const tail = (message_tail: string) => ({ ...STOP, payload: { message_tail } });

it("rung 6 is gone: a trailing question with no tag now falls straight through to rung 7 (deferred)", () => {
  const db = openDb(":memory:");
  expect(deriveCapsule(db, tail("some prose\nshall I proceed?"), classifierOn).capsule_rule).toBe("deferred");
  expect(deriveCapsule(db, tail("shall I proceed?\n\n  \n"), classifierOn).capsule_rule).toBe("deferred");
  db.close();
});

it("a mid-message question with a statement last still defers (unchanged)", () => {
  const db = openDb(":memory:");
  expect(deriveCapsule(db, tail("should I? yes.\nDone, all tests pass."), classifierOn).capsule_rule).toBe("deferred");
  db.close();
});

const FENCE = "\u0060\u0060\u0060";

it("a trailing question inside a code fence also defers now — the no-markdown-awareness distinction rung 6 pinned is moot once rung 6 is gone", () => {
  const db = openDb(":memory:");
  const closed = `here:\n${FENCE}sh\ngrep 'x?'\nwhat?\n${FENCE}`;
  const open = `here:\n${FENCE}sh\nls\nreally?`;
  expect(deriveCapsule(db, tail(closed), classifierOn).capsule_rule).toBe("deferred");
  expect(deriveCapsule(db, tail(open), classifierOn).capsule_rule).toBe("deferred");
  db.close();
});

it("a whitespace-only or absent tail is settled at ingress, never queued", () => {
  const db = openDb(":memory:");
  expect(deriveCapsule(db, tail("   \n\n  "))).toMatchObject({ capsule_status: "unknown", capsule_rule: "abandoned" });
  expect(deriveCapsule(db, { ...STOP, payload: {} })).toMatchObject({ capsule_status: "unknown", capsule_rule: "abandoned" });
  db.close();
});

it("rung 7 defers with no outbound call", () => {
  const db = openDb(":memory:");
  expect(deriveCapsule(db, tail("some ambiguous prose"), classifierOn)).toEqual({
    capsule_status: "unknown", capsule_rule: "deferred", source: "behavioral",
  });
  db.close();
});

const seedQuestion = (db: ReturnType<typeof openDb>) => insertEvent(db, { ...STOP, type: "question", source: "deterministic",
  emitter: "claude-pretool", payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } });
const seedAttention = (db: ReturnType<typeof openDb>) => insertEvent(db, { ...STOP, type: "attention-needed",
  emitter: "claude-notification", payload: { reason: "permission" } });
const seedRun = (db: ReturnType<typeof openDb>) => insertEvent(db, RUN_STARTED());
const seedFinish = (db: ReturnType<typeof openDb>) => { seedRun(db); insertEvent(db, { ...RUN_STARTED(), type: "run-finished",
  payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x", summary: "s", head_sha: null } }); };
const TAGGED = { ...STOP, source: "deterministic" as const, payload: { capsule_status: "done", capsule_rule: "tag" } };
const ASKING = { ...STOP, payload: { message_tail: "shall I?" } };

const PAIRS: [string, (db: ReturnType<typeof openDb>) => void, WorkflowEventInput, string][] = [
  ["1v2", seedQuestion,  TAGGED, "tag"],
  ["1v3", seedAttention, TAGGED, "tag"],
  ["1v4", seedRun,       TAGGED, "tag"],
  ["1v5", seedFinish,    TAGGED, "tag"],
  ["1v6", () => {},     { ...TAGGED, payload: { ...TAGGED.payload, message_tail: "really?" } }, "tag"],
  ["2v3", (db) => { seedAttention(db); seedQuestion(db); }, STOP,   "question_open"],
  ["2v4", (db) => { seedRun(db);       seedQuestion(db); }, STOP,   "question_open"],
  ["2v5", (db) => { seedFinish(db);    seedQuestion(db); }, STOP,   "question_open"],
  ["2v6", seedQuestion,  ASKING, "question_open"],
  ["3v4", (db) => { seedRun(db);    seedAttention(db); }, STOP,   "attention"],
  ["3v5", (db) => { seedFinish(db); seedAttention(db); }, STOP,   "attention"],
  ["3v6", seedAttention, ASKING, "attention"],
  ["4v5", (db) => { seedFinish(db); insertEvent(db, RUN_STARTED({ run_id: "r2" })); }, STOP, "run_in_flight"],
  ["4v6", seedRun,       ASKING, "run_in_flight"],
  ["5v6", seedFinish,    ASKING, "run_finished"],
];

it.each(PAIRS)("precedence %s: the higher rung wins", (_name, seed, ev, expected) => {
  const db = openDb(":memory:");
  setAfk(db, true);
  seed(db);
  expect(deriveCapsule(db, ev).capsule_rule).toBe(expected);
  db.close();
});

// ---------------------------------------------------------------------------
// Diff-review fixes (run 61ee7646fd4c). Each test fails if its fix is reverted.
// ---------------------------------------------------------------------------

const RUN = (over: Record<string, unknown> = {}) => ({
  run_id: "r1", project: "p1", role: "builder" as const, type: "run-started" as const,
  source: "deterministic" as const, emitter: "wrapper" as const,
  payload: { phase: "B", runtime: "opencode-grok", kind: "plan", target: "/x/p.md",
             caller: "claude", caller_session: "sess-a", caller_pane: "%3",
             model: "grok-4.6", effort: "high", verify: "true",
             session: "jax-p1-builder-r1", repo: "/home/rafa/repos/jax-os" },
  ...over,
});
const FINISH = (run_id: string) => ({
  run_id, project: "p1", role: "builder" as const, type: "run-finished" as const,
  source: "deterministic" as const, emitter: "wrapper" as const,
  payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x", summary: "s", head_sha: null },
});

it("F2: rung 2 fires with AFK OFF, where a question is stored delivery:local", () => {
  const db = openDb(":memory:");
  // No setAfk — AFK defaults off, which is Rafa's normal state. 16 of 20 live questions are
  // `local`; scoping rung 2 to delivered questions made the most important rung nearly dead.
  const q = insertEvent(db, { ...STOP, type: "question", source: "deterministic", emitter: "claude-pretool",
                              payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } });
  expect(q.delivery).toBe("local");
  expect(deriveCapsule(db, STOP)).toMatchObject({ capsule_status: "needs_input", capsule_rule: "question_open" });
  db.close();
});

it("F5: jaxflow's nested caller_session is normalized into the indexed column", () => {
  const db = openDb(":memory:");
  expect(insertEvent(db, RUN()).harness_session).toBe("sess-a");
  // A top-level value the producer sent WINS — normalization fills the column, never overwrites
  // it. (The value stays bounded because caller_session has its own 128-char ingress limit.)
  expect(insertEvent(db, { ...RUN({ run_id: "r9" }), harness_session: "explicit" }).harness_session).toBe("explicit");
  db.close();
});

it("F3a: an older run still in flight is not masked by a newer finished one", () => {
  const db = openDb(":memory:");
  insertEvent(db, RUN());                          // r1 starts, never finishes
  insertEvent(db, RUN({ run_id: "r2" }));          // r2 starts
  insertEvent(db, FINISH("r2"));                   // r2 finishes — the NEWEST run is done
  expect(deriveCapsule(db, STOP)).toMatchObject({ capsule_status: "waiting", capsule_rule: "run_in_flight" });
  db.close();
});

it("F3b: rung 5 requires the finish to be newer than the last turn-started", () => {
  const db = openDb(":memory:");
  insertEvent(db, RUN());
  insertEvent(db, FINISH("r1"));
  expect(deriveCapsule(db, STOP)).toMatchObject({ capsule_rule: "run_finished" });
  // A new turn began AFTER that run finished: the run is no longer "just finished".
  insertEvent(db, { ...STOP, type: "turn-started", source: "deterministic", emitter: "claude-userprompt", payload: {} });
  expect(deriveCapsule(db, STOP).capsule_rule).not.toBe("run_finished");
  db.close();
});

it("F4: an ingress-settled abandoned row carries capsule_attempts 0", () => {
  const db = openDb(":memory:");
  const row = insertEvent(db, { ...STOP, payload: {} });   // no message_tail at all
  expect(row.payload).toMatchObject({ capsule_status: "unknown", capsule_rule: "abandoned", capsule_attempts: 0 });
  db.close();
});

// ---------------------------------------------------------------------------
// MOA-469 F1 (review 8dc843212345): a Codex event's optional pane is NOT
// authority for Claude pane derivation. Each test fails if codex rows are
// allowed back into a pane-keyed predicate.
// ---------------------------------------------------------------------------

const CODEX_UUID = "0191f0aa-f1f1-7000-8000-0000000000f1";
const CODEX_STOP_PANE: WorkflowEventInput = {
  ...STOP, emitter: "codex-stop", harness_session: CODEX_UUID, payload: { message_tail: "prose" },
};

it("F1: a pane-carrying codex-stop does not consume a Claude question on that pane", () => {
  const db = openDb(":memory:");
  setAfk(db, true);
  insertEvent(db, { ...STOP, type: "question", source: "deterministic", emitter: "claude-pretool",
                    payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } });
  expect(deriveCapsule(db, CODEX_STOP_PANE, classifierOn).capsule_rule).toBe("deferred");
  db.close();
});

it("F1: a pane-carrying codex-stop does not consume Claude attention on that pane", () => {
  const db = openDb(":memory:");
  insertEvent(db, { ...STOP, type: "turn-started", source: "deterministic", emitter: "claude-userprompt", payload: {} });
  insertEvent(db, { ...STOP, type: "attention-needed", emitter: "claude-notification", payload: { reason: "permission" } });
  expect(deriveCapsule(db, CODEX_STOP_PANE, classifierOn).capsule_rule).toBe("deferred");
  db.close();
});

it("F1: a Claude turn-started on the pane does not bound a native codex run_finished", () => {
  const db = openDb(":memory:");
  insertEvent(db, { run_id: "rc", project: "p1", role: "builder", type: "run-started", source: "deterministic",
    emitter: "wrapper", payload: { phase: "B", runtime: "codex", caller: "codex", caller_session: CODEX_UUID } });
  insertEvent(db, { run_id: "rc", project: "p1", role: "builder", type: "run-finished", source: "deterministic",
    emitter: "wrapper", payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x", summary: "s", head_sha: null } });
  // A NEWER Claude turn on the same pane — before the fix it suppressed the native finish.
  insertEvent(db, { ...STOP, type: "turn-started", source: "deterministic", emitter: "claude-userprompt", payload: {} });
  expect(deriveCapsule(db, CODEX_STOP_PANE).capsule_rule).toBe("run_finished");
  db.close();
});

it("F1: hasOpenQuestion is not closed by a later codex-stop on the same pane", () => {
  const db = openDb(":memory:");
  setAfk(db, true);
  insertEvent(db, { ...STOP, type: "question", source: "deterministic", emitter: "claude-pretool",
                    payload: { tool_use_id: "t1", questions: [{ question: "q?" }] } });
  insertEvent(db, CODEX_STOP_PANE);
  expect(hasOpenQuestion(db, { pane: "%1", tmuxIncarnation: INC }, false)).toBe(true);
  db.close();
});

// ---------------------------------------------------------------------------
// MOA-502 Decision 1: rung 7 reads integrations.classifier live through the
// injected reader (defaults to the real readGeneralSettings).
// ---------------------------------------------------------------------------

type ReadSettings = typeof readGeneralSettings;
const classifierOn = () => ({ ok: true, data: { integrations: { classifier: true } } }) as ReturnType<ReadSettings>;
const classifierOff = () => ({ ok: true, data: { integrations: { classifier: false } } }) as ReturnType<ReadSettings>;

describe("deriveCapsule rung 7 — integrations.classifier gate (spec MOA-502 Decision 1)", () => {
  it("settles unknown/classifier-off with no queueing when the setting is off", () => {
    const db = openDb(":memory:");
    const result = deriveCapsule(db, tail("still thinking about the approach"), classifierOff);
    expect(result).toEqual({ capsule_status: "unknown", capsule_rule: "classifier-off", source: "behavioral" });
    db.close();
  });

  it("takes the same classifier-off path on a malformed settings read (safe no-egress default)", () => {
    const db = openDb(":memory:");
    const result = deriveCapsule(db, tail("still thinking"),
      () => ({ ok: false, error: "settings-malformed" }) as ReturnType<ReadSettings>);
    expect(result.capsule_rule).toBe("classifier-off");
    db.close();
  });

  it("keeps today's deferred/queued behavior unchanged when the setting is on", () => {
    const db = openDb(":memory:");
    const result = deriveCapsule(db, tail("still thinking"), classifierOn);
    expect(result).toEqual({ capsule_status: "unknown", capsule_rule: "deferred", source: "behavioral" });
    db.close();
  });
});
