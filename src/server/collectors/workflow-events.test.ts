import { describe, expect, it } from "vitest";
import { parseAnswerBody, parseIngress, parseSessionBody, renderMessage, toForwardBody } from "./workflow-events";
import { EVENT_MATRIX, EVENT_TYPES, HOOK_TELEMETRY_TYPES, LIMITS, parseIndexedReply, type WorkflowEventRow } from "../../lib/workflow";

const RUN_STARTED = {
  project: "jax-os", run_id: "r-001", role: "builder", type: "run-started",
  source: "deterministic", emitter: "wrapper",
  payload: {
    phase: "B", runtime: "opencode-deepseek", kind: "plan", target: "/home/rafa/repos/jax-os/.local/docs/plans/example.md",
    caller: "claude", caller_session: "sess-abc123", caller_pane: "%3", model: "grok-4.6", effort: "high",
    session: "jax-jax-os-builder-r-001", repo: "/home/rafa/repos/jax-os",
  },
};
const RUN_FINISHED = {
  project: "jax-os", run_id: "r-001", role: "builder", type: "run-finished",
  source: "deterministic", emitter: "wrapper",
  payload: {
    phase: "B", exit_code: 0, contract_status: "ok", report_path: "/home/rafa/repos/jax-os/.local/reports/r-001.md",
    summary: "done", head_sha: "a".repeat(40), result: "success",
  },
};
const QUESTION = {
  project: "jax-os", role: "lead", pane: "%1", type: "question", source: "deterministic", emitter: "claude-pretool",
  payload: {
    tool_use_id: "toolu_01ABC",
    questions: [{ question: "Merge now?", header: "Merge", options: [{ label: "Yes" }, { label: "No", description: "wait" }], multiSelect: false }],
  },
};

function expectInvalid(body: unknown, re: RegExp) {
  const r = parseIngress(body);
  expect(r.ok).toBe(false);
  if (!r.ok) expect(r.error).toMatch(re);
}

describe("parseIngress — happy paths", () => {
  it("accepts a builder run-started and normalizes run_id/payload", () => {
    const r = parseIngress(RUN_STARTED);
    expect(r).toMatchObject({ ok: true, event: { ...RUN_STARTED, run_id: "r-001", harness_session: null } });
  });
  it("accepts run-finished with a 40-hex head_sha (builder) and with null head_sha", () => {
    expect(parseIngress(RUN_FINISHED).ok).toBe(true);
    expect(parseIngress({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, head_sha: null } }).ok).toBe(true);
  });
  it("accepts a reviewer run-finished without head_sha", () => {
    const { head_sha: _drop, result: _dropResult, ...p } = RUN_FINISHED.payload;
    expect(parseIngress({ ...RUN_FINISHED, role: "reviewer", payload: { ...p, verdict: "approve" } }).ok).toBe(true);
  });
  it("accepts a lead question (run_id absent) and keeps questions verbatim", () => {
    const r = parseIngress(QUESTION);
    expect(r.ok).toBe(true);
    if (r.ok) {
      expect(r.event.run_id).toBeNull();
      expect(r.event.payload.questions).toEqual(QUESTION.payload.questions);
    }
  });
  it("accepts a capsule-shaped turn-stopped", () => {
    expect(parseIngress({ project: "p1", role: "lead", pane: "%1", type: "turn-stopped", source: "deterministic", emitter: "claude-stop", payload: { capsule_status: "done", capsule_rule: "tag" } }).ok).toBe(true);
  });
  it("accepts attention-needed only as behavioral from claude-notification", () => {
    expect(parseIngress({ project: "p1", role: "lead", pane: "%1", type: "attention-needed", source: "behavioral", emitter: "claude-notification", payload: { reason: "agent_needs_input", excerpt: "pane" } }).ok).toBe(true);
    expectInvalid({ project: "p1", role: "lead", pane: "%1", type: "attention-needed", source: "deterministic", emitter: "claude-notification", payload: { reason: "x" } }, /source must be behavioral/);
  });
  it("accepts question-answered from jaxos", () => {
    expect(parseIngress({ project: "p1", role: "lead", type: "question-answered", source: "deterministic", emitter: "jaxos", payload: { kind: "structured", question_event_id: 12, tool_use_id: "toolu_1", mutation_id: 99, ok: false, error: "pane gone" } }).ok).toBe(true);
  });
});

describe("parseIngress — allowlists", () => {
  it("rejects non-object bodies and unknown top-level keys", () => {
    expectInvalid(null, /object/);
    expectInvalid([], /object/);
    expectInvalid({ ...RUN_STARTED, ts: "2026-01-01T00:00:00Z" }, /unknown key ts/);
  });
  it("rejects unknown type / emitter / source / role", () => {
    expectInvalid({ ...RUN_STARTED, type: "run-exploded" }, /type not allowed/);
    expectInvalid({ ...RUN_STARTED, emitter: "hermes" }, /emitter not allowed/);
    expectInvalid({ ...RUN_STARTED, source: "guess" }, /source not allowed/);
    expectInvalid({ ...RUN_STARTED, role: "pm" }, /role not allowed/);
  });
  it("rejects emitter/type and role/type mismatches", () => {
    expectInvalid({ ...QUESTION, emitter: "wrapper" }, /emitter wrapper cannot emit question/);
    expectInvalid({ ...RUN_STARTED, role: "lead" }, /role lead not allowed for run-started/);
  });
  it("enforces run_id presence per type", () => {
    const { run_id: _drop, ...noRun } = RUN_STARTED;
    expectInvalid(noRun, /run_id required/);
    expectInvalid({ ...QUESTION, run_id: "r-1" }, /run_id only for run events/);
  });
  it("validates project label and run_id/phase tokens", () => {
    // Task A2 widens project validation from PROJECT_SLUG (lowercase-slug-only) to
    // PROJECT_LABEL (printable ASCII incl. space) — "Jax OS" is now valid; a control
    // character is the thing that's still rejected.
    expectInvalid({ ...RUN_STARTED, project: "p1\n" }, /project malformed/);
    expectInvalid({ ...RUN_STARTED, project: "a".repeat(65) }, /project exceeds 64/);
    expectInvalid({ ...RUN_STARTED, run_id: "r 1" }, /run_id malformed/);
    expectInvalid({ ...RUN_STARTED, payload: { phase: "B/2", runtime: "codex" } }, /phase malformed/);
  });
});

describe("parseIngress — payload schemas", () => {
  it("rejects unknown payload keys and bad enum values", () => {
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, extra: 1 } }, /unknown key extra/);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, runtime: "gemini" } }, /runtime not allowed/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, contract_status: "great" } }, /contract_status not allowed/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, exit_code: 1.5 } }, /exit_code/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, exit_code: 300 } }, /exit_code/);
  });
  it("enforces head_sha rules per role", () => {
    const { head_sha: _drop, ...noSha } = RUN_FINISHED.payload;
    expectInvalid({ ...RUN_FINISHED, payload: noSha }, /head_sha required for builder runs/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, head_sha: "abc123" } }, /head_sha must be a full 40-hex sha or null/);
    expectInvalid({ ...RUN_FINISHED, role: "reviewer" }, /head_sha only for builder runs/);
  });
  it("rejects a caller-supplied tmux_target and malformed questions", () => {
    expectInvalid({ ...QUESTION, payload: { ...QUESTION.payload, tmux_target: "x:0.0" } }, /unknown key tmux_target/);
    expectInvalid({ ...QUESTION, payload: { tool_use_id: "toolu_1", questions: [] } }, /questions must be a non-empty array/);
    expectInvalid({ ...QUESTION, payload: { tool_use_id: "toolu_1", questions: [{ question: "q", options: [] }] } }, /options must be a non-empty array/);
    expectInvalid({ ...QUESTION, payload: { tool_use_id: "toolu_1", questions: [{ question: "q", options: [{ label: "a", weird: 1 }] }] } }, /option: unknown key weird/);
    expectInvalid({ ...QUESTION, payload: { tool_use_id: "bad id", questions: QUESTION.payload.questions } }, /tool_use_id malformed/);
  });
  it("enforces string caps and the payload byte cap", () => {
    expectInvalid({ project: "p1", role: "lead", pane: "%1", type: "turn-stopped", source: "deterministic", emitter: "claude-stop", payload: { capsule_status: "done", capsule_rule: "tag", excerpt: "x".repeat(201) } }, /excerpt exceeds 200/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, summary: "s".repeat(201) } }, /summary exceeds 200/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, summary: "two\nlines" } }, /summary must be a single line/);
    expectInvalid({ project: "p1", role: "lead", pane: "%1", type: "attention-needed", source: "behavioral", emitter: "claude-notification", payload: { reason: "r", excerpt: "x".repeat(20_000) } }, /payload exceeds/);
  });
  it("validates question-answered ids and ok flag", () => {
    expectInvalid({ project: "p1", role: "lead", type: "question-answered", source: "deterministic", emitter: "jaxos", payload: { kind: "structured", question_event_id: 0, tool_use_id: "t", mutation_id: 1, ok: true } }, /question_event_id/);
    expectInvalid({ project: "p1", role: "lead", type: "question-answered", source: "deterministic", emitter: "jaxos", payload: { kind: "structured", question_event_id: 1, tool_use_id: "t", mutation_id: 1, ok: "yes" } }, /ok must be boolean/);
  });
  it("rejects an unserializable payload instead of throwing", () => {
    const cyclic: Record<string, unknown> = { reason: "r" };
    cyclic.self = cyclic;
    expectInvalid({ project: "p1", role: "lead", pane: "%1", type: "attention-needed", source: "behavioral", emitter: "claude-notification", payload: cyclic }, /payload not serializable/);
  });
  it("rejects a deeply nested payload without throwing", () => {
    let deep: Record<string, unknown> = {};
    for (let i = 0; i < 60_000; i++) deep = { n: deep };
    const r = parseIngress({ project: "p1", role: "lead", pane: "%1", type: "attention-needed", source: "behavioral", emitter: "claude-notification", payload: { reason: "r", excerpt: "x", extra: deep } });
    expect(r.ok).toBe(false);
  });
});

describe("parseIngress — pane and tmux_incarnation (D1, Finding 3)", () => {
  // B6: `payload: {}` is the retired legacy shape — every turn-stopped now carries a capsule verdict.
  const base = { project: "p1", role: "lead", type: "turn-stopped", source: "deterministic", emitter: "claude-stop", payload: { capsule_status: "done", capsule_rule: "tag" } };

  it("accepts a well-formed pane for role lead, with or without tmux_incarnation", () => {
    expect(parseIngress({ ...base, pane: "%27" })).toMatchObject({ ok: true });
    expect(parseIngress({ ...base, pane: "%27", tmux_incarnation: "234790:1787586213" })).toMatchObject({ ok: true });
  });

  it("rejects a malformed pane, and rejects pane for a non-lead role", () => {
    expect(parseIngress({ ...base, pane: "27" })).toMatchObject({ ok: false });
    expect(parseIngress({ ...base, pane: "" })).toMatchObject({ ok: false });
    expect(parseIngress({
      project: "p1", run_id: "r1", role: "builder", type: "run-started", source: "deterministic",
      emitter: "wrapper", payload: { phase: "B", runtime: "codex" }, pane: "%1",
    })).toMatchObject({ ok: false });
  });

  it("rejects tmux_incarnation without pane, and a malformed incarnation string", () => {
    expect(parseIngress({ ...base, tmux_incarnation: "234790:1787586213" })).toMatchObject({ ok: false });
    expect(parseIngress({ ...base, pane: "%27", tmux_incarnation: "not-an-incarnation" })).toMatchObject({ ok: false });
  });

  it("B6: pane is REQUIRED for role lead and adhoc — the A′ window is closed", () => {
    expect(parseIngress({ ...base })).toMatchObject({ ok: false, error: expect.stringMatching(/pane/) });
    expect(parseIngress({ ...base, role: "adhoc" })).toMatchObject({ ok: false, error: expect.stringMatching(/pane/) });
  });

  it("B6: an event emitted by jaxos itself is exempt — it is not a hook", () => {
    // The staleness alert and question-answered are written by insertEvent directly and can carry
    // no pane; the events route rejects `emitter: "jaxos"` outright, so this exemption is never
    // reachable over HTTP. It exists so the validator agrees with Rule A (§4.4) and with the B6
    // barrier query, both of which already carve out `emitter = 'jaxos'` for the same reason.
    expect(parseIngress({
      project: "p1", role: "lead", type: "question-answered", source: "deterministic", emitter: "jaxos",
      payload: { kind: "freeform", question_event_id: 1, tool_use_id: null, mutation_id: 2, ok: true },
    })).toMatchObject({ ok: true });
  });
});

describe("parseIngress — native Codex identity (MOA-469 §3)", () => {
  const UUID = "0191f0aa-7777-7000-8000-000000000007";
  const codexStop = (over: Record<string, unknown> = {}) => ({
    project: "p1", role: "lead", type: "turn-stopped", source: "behavioral", emitter: "codex-stop",
    harness_session: UUID, payload: { message_tail: "prose" }, ...over,
  });

  it("accepts a pane-less Codex stop/permission/userprompt with a canonical UUID", () => {
    expect(parseIngress(codexStop()).ok).toBe(true);
    expect(parseIngress({ project: "p1", role: "lead", type: "attention-needed", source: "deterministic", emitter: "codex-permission", harness_session: UUID, payload: { reason: "permission_request" } }).ok).toBe(true);
    expect(parseIngress({ project: "p1", role: "lead", type: "turn-started", source: "deterministic", emitter: "codex-userprompt", harness_session: UUID, payload: {} }).ok).toBe(true);
  });

  it("refuses a Codex event with a missing or non-canonical session UUID", () => {
    const { harness_session: _drop, ...noSession } = codexStop();
    expect(parseIngress(noSession)).toMatchObject({ ok: false, error: "codex event requires a canonical session uuid" });
    expect(parseIngress(codexStop({ harness_session: UUID.toUpperCase() }))).toMatchObject({ ok: false });
    expect(parseIngress(codexStop({ harness_session: "not-a-uuid" }))).toMatchObject({ ok: false });
    expect(parseIngress(codexStop({ harness_session: "0191f0aa-7777-7000-8000-00000000000" }))).toMatchObject({ ok: false });
  });

  it("still accepts optional pane metadata on a Codex event, and refuses a malformed one", () => {
    expect(parseIngress(codexStop({ pane: "%9", tmux_incarnation: "1:2" })).ok).toBe(true);
    expect(parseIngress(codexStop({ pane: "not-a-pane" })).ok).toBe(false);
  });

  it("does not accept a pane-less event from a non-Codex hook emitter", () => {
    expect(parseIngress({ project: "p1", role: "lead", type: "turn-stopped", source: "behavioral", emitter: "claude-stop", payload: { message_tail: "x" } }))
      .toMatchObject({ ok: false, error: "pane required" });
  });
});

describe("parseIngress — turn-stopped two-shape schema (§4.3, §5.1 Finding 29)", () => {
  const envelope = { project: "p1", role: "lead", pane: "%1", type: "turn-stopped" as const };

  it("accepts every new capsule-shaped payload variant", () => {
    expect(parseIngress({
      ...envelope, source: "deterministic", emitter: "claude-stop",
      payload: { capsule_status: "waiting", capsule_minutes: 20, capsule_rule: "tag", excerpt: "back in 20" },
    })).toMatchObject({ ok: true });
    for (const status of ["done", "needs_input", "blocked"]) {
      expect(parseIngress({ ...envelope, source: "deterministic", emitter: "claude-stop", payload: { capsule_status: status, capsule_rule: "tag" } }))
        .toMatchObject({ ok: true });
    }
    expect(parseIngress({ ...envelope, source: "behavioral", emitter: "codex-stop", harness_session: "0191f0aa-6666-7000-8000-000000000006", payload: { message_tail: "prose" } }))
      .toMatchObject({ ok: true });
  });

  it("rejects capsule_minutes without waiting, and excerpt over 200 chars or off the deterministic source", () => {
    expect(parseIngress({ ...envelope, source: "deterministic", emitter: "claude-stop", payload: { capsule_status: "done", capsule_rule: "tag", capsule_minutes: 5 } }))
      .toMatchObject({ ok: false });
    expect(parseIngress({
      ...envelope, source: "deterministic", emitter: "claude-stop",
      payload: { capsule_status: "waiting", capsule_minutes: 5, capsule_rule: "tag", excerpt: "x".repeat(201) },
    })).toMatchObject({ ok: false });
    expect(parseIngress({
      ...envelope, source: "behavioral", emitter: "codex-stop",
      payload: { capsule_status: "unknown", excerpt: "leaked text" },
    })).toMatchObject({ ok: false }); // Finding 1: excerpt only ever on the deterministic/tag-matched branch
  });

  it("B6: the A′-window legacy shapes are now REJECTED (Finding 29 window closed)", () => {
    // Empty payload is the empty-message case and is accepted; an excerpt without the tag
    // marker is a half-tagged payload and is refused so its verdict cannot be discarded.
    expect(parseIngress({ ...envelope, source: "deterministic", emitter: "claude-stop", payload: { excerpt: "y".repeat(300) } }))
      .toMatchObject({ ok: false, error: expect.stringMatching(/capsule_rule/) });
    // The emitter itself no longer exists.
    expect(parseIngress({ ...envelope, source: "deterministic", emitter: "codex-notify", payload: { capsule_status: "done" } }))
      .toMatchObject({ ok: false, error: expect.stringMatching(/emitter/) });
  });

  it("excerpt rides any matched capsule status, not only waiting (spec 4.3 excerpt policy)", () => {
    // The tag MATCHED — that is what `source: "deterministic"` means — so a lead who wrote
    // text after `[JAXFLOW: done]` has an excerpt. An earlier implementation restricted this
    // to `waiting` and silently rejected every other valid capsule carrying one.
    for (const status of ["done", "needs_input", "blocked"]) {
      expect(parseIngress({
        ...envelope, source: "deterministic", emitter: "claude-stop",
        payload: { capsule_status: status, capsule_rule: "tag", excerpt: "shipped the migration" },
      })).toMatchObject({ ok: true });
    }
    // The behavioral (classifier) branch never produces an excerpt — that guard stays.
    expect(parseIngress({
      ...envelope, source: "behavioral", emitter: "claude-stop",
      payload: { capsule_status: "done", excerpt: "nope" },
    })).toMatchObject({ ok: false });
  });

  it("source/emitter 1:1 is exempted for turn-stopped only — every other type still enforces it", () => {
    expect(parseIngress({ ...envelope, source: "behavioral", emitter: "claude-stop", payload: { message_tail: "prose" } }))
      .toMatchObject({ ok: true }); // claude-stop's declared source is deterministic; turn-stopped is exempt
    expect(parseIngress({
      project: "p1", role: "lead", pane: "%1", type: "question-resolved",
      source: "behavioral", emitter: "claude-posttool", payload: { tool_use_id: "tu_1" },
    })).toMatchObject({ ok: false }); // question-resolved is NOT exempt — still must match SOURCE_BY_EMITTER
  });
});

const STOP = {
  project: "jax-os", role: "lead", type: "turn-stopped", pane: "%1",
  source: "behavioral", emitter: "claude-stop", payload: {},
};

describe("parseIngress — untagged/tagged turn-stopped shapes and harness_session", () => {
  it("accepts an untagged turn-stopped carrying only a message_tail", () => {
    const r = parseIngress({ ...STOP, payload: { message_tail: "some prose" } });
    expect(r.ok).toBe(true);
  });

  it("accepts a complete tagged turn-stopped and refuses a behavioral one", () => {
    expect(parseIngress({ ...STOP, source: "deterministic",
      payload: { capsule_status: "done", capsule_rule: "tag", excerpt: "x" } }).ok).toBe(true);
    expectInvalid({ ...STOP, payload: { capsule_status: "done", capsule_rule: "tag", excerpt: "x" } },
                  /capsule_rule tag requires source deterministic/);
  });

  it("refuses any payload that mixes the two shapes, and an over-long tail", () => {
    for (const mix of [{ excerpt: "b" }, { capsule_status: "done" }, { capsule_rule: "tag" }, { capsule_minutes: 5 }]) {
      expectInvalid({ ...STOP, source: "deterministic", payload: { message_tail: "a", ...mix } },
                    /message_tail cannot accompany a tagged payload/);
    }
    expectInvalid({ ...STOP, payload: { message_tail: "x".repeat(2001) } }, /message_tail exceeds 2000/);
  });

  it("refuses an EMPTY message_tail — the hook emits an absent field, never an empty one", () => {
  // Spec §3.2 as amended 2026-09-07: a whitespace-only or fully-redacted message returns a
  // payload with no message_tail key at all, so "" is unreachable from any real producer and
  // the validator stays strict rather than being loosened for a shape nobody can send.
  expectInvalid({ ...STOP, payload: { message_tail: "" } }, /message_tail must be a non-empty string/);
  expect(parseIngress({ ...STOP, payload: {} }).ok).toBe(true);
});

it("refuses a tagged payload that forgot the rule marker — its verdict must not be discarded", () => {
    expectInvalid({ ...STOP, source: "deterministic", payload: { capsule_status: "done", excerpt: "x" } },
                  /a tagged payload requires capsule_rule tag/);
  });

  it("accepts capsule_rule tag from the hook and refuses every other value", () => {
    expect(parseIngress({ ...STOP, source: "deterministic", payload: { capsule_status: "done", capsule_rule: "tag" } }).ok).toBe(true);
    expectInvalid({ ...STOP, payload: { message_tail: "x", capsule_rule: "deferred" } }, /message_tail cannot accompany a tagged payload/);
    expectInvalid({ ...STOP, source: "deterministic", payload: { capsule_status: "done", capsule_rule: "deferred" } }, /capsule_rule not allowed from a producer/);
    expectInvalid({ ...STOP, payload: { message_tail: "x", capsule_attempts: 0 } }, /payload: unknown key capsule_attempts/);
  });

  it("refuses a tag that carries no usable verdict", () => {
    expectInvalid({ ...STOP, source: "deterministic", payload: { capsule_rule: "tag" } }, /requires a capsule_status/);
    expectInvalid({ ...STOP, source: "deterministic", payload: { capsule_rule: "tag", capsule_status: "unknown" } }, /cannot be unknown/);
    expectInvalid({ ...STOP, payload: { capsule_rule: "tag", capsule_status: "done" } }, /requires source deterministic/);
  });

  it("accepts a bounded harness_session on any event and refuses an over-long one", () => {
    expect(parseIngress({ ...STOP, harness_session: "sess-abc", payload: { message_tail: "x" } }).ok).toBe(true);
    expectInvalid({ ...STOP, harness_session: "x".repeat(129), payload: { message_tail: "x" } }, /harness_session exceeds 128/);
  });

  it("accepts a payload with neither text field nor status", () => {
    expect(parseIngress({ ...STOP, payload: {} }).ok).toBe(true);
  });
});

describe("parseIngress — role adhoc accepted for every hook-produced type (Finding 8a)", () => {
  it.each([
    ["question", "claude-pretool", "deterministic", { tool_use_id: "tu_1", questions: [{ question: "q?", options: [{ label: "a" }] }] }],
    ["question-resolved", "claude-posttool", "deterministic", { tool_use_id: "tu_1" }],
    ["attention-needed", "claude-notification", "behavioral", { reason: "agent_needs_input" }],
    ["turn-stopped", "claude-stop", "deterministic", { capsule_status: "done", capsule_rule: "tag" }],
    ["turn-started", "claude-userprompt", "deterministic", {}],
  ] as const)("%s accepts role adhoc", (type, emitter, source, payload) => {
    expect(parseIngress({ project: "p1", role: "adhoc", pane: "%1", type, source, emitter, payload }))
      .toMatchObject({ ok: true, event: { role: "adhoc" } });
  });
});

describe("parseIngress — question-answered payload schema (§6 route 5)", () => {
  const envelope = { project: "p1", role: "lead", type: "question-answered" as const, source: "deterministic", emitter: "jaxos" };

  it("requires kind and a matching tool_use_id nullability", () => {
    expect(parseIngress({ ...envelope, payload: { kind: "structured", question_event_id: 1, tool_use_id: "tu_1", mutation_id: 2, ok: true } }))
      .toMatchObject({ ok: true });
    expect(parseIngress({ ...envelope, payload: { kind: "freeform", question_event_id: 1, tool_use_id: null, mutation_id: 2, ok: true } }))
      .toMatchObject({ ok: true });
    expect(parseIngress({ ...envelope, payload: { kind: "freeform", question_event_id: 1, tool_use_id: "tu_1", mutation_id: 2, ok: true } }))
      .toMatchObject({ ok: false }); // freeform must be null
    expect(parseIngress({ ...envelope, payload: { kind: "structured", question_event_id: 1, tool_use_id: null, mutation_id: 2, ok: true } }))
      .toMatchObject({ ok: false }); // structured must be a real string
  });

  it("accepts a codex finalization carrying its exact thread identity and no fake pane", () => {
    expect(parseIngress({ ...envelope, harness_session: "0191f0aa-aaaa-7000-8000-00000000000a",
      payload: { kind: "freeform", question_event_id: 1, tool_use_id: null, mutation_id: 2, ok: true } }))
      .toMatchObject({ ok: true, event: { harness_session: "0191f0aa-aaaa-7000-8000-00000000000a" } });
  });
});

describe("parseIngress — PROJECT_LABEL replaces PROJECT_SLUG at ingress (Finding 34)", () => {
  it.each(["PIJ", "Acme.AI", "CS_standalone", "comenta.ia.br", "EBPMN 2"])("accepts real repo basename %s", (project) => {
    expect(parseIngress({
      project, role: "lead", pane: "%1", type: "turn-stopped",
      source: "deterministic", emitter: "claude-stop", payload: { capsule_status: "done", capsule_rule: "tag" },
    })).toMatchObject({ ok: true });
  });

  it("rejects a control character or non-ASCII project value", () => {
    expect(parseIngress({
      project: "p1\n", role: "lead", pane: "%1", type: "turn-stopped",
      source: "deterministic", emitter: "claude-stop", payload: { capsule_status: "done" },
    })).toMatchObject({ ok: false });
    expect(parseIngress({
      project: "café", role: "lead", pane: "%1", type: "turn-stopped",
      source: "deterministic", emitter: "claude-stop", payload: { capsule_status: "done" },
    })).toMatchObject({ ok: false });
  });
});

describe("parseIngress — turn-started (NEW, §4.6.3)", () => {
  it("accepts an empty payload, rejects any key", () => {
    expect(parseIngress({
      project: "p1", role: "lead", pane: "%1", type: "turn-started",
      source: "deterministic", emitter: "claude-userprompt", payload: {},
    })).toMatchObject({ ok: true });
    expect(parseIngress({
      project: "p1", role: "lead", pane: "%1", type: "turn-started",
      source: "deterministic", emitter: "codex-userprompt", payload: { prompt: "leaked" },
    })).toMatchObject({ ok: false }); // §4.7: prompt is never forwarded into the payload
  });
});

describe("parseSessionBody — re-keyed shape + backward-compat (§3.1b, Finding 40)", () => {
  it("accepts the new shape and stores it as kind current", () => {
    const parsed = parseSessionBody({ project: "p1", session: "jax-p1-lead", pane: "%7", role: "lead", tmux_incarnation: "111:222" });
    expect(parsed).toEqual({
      ok: true, kind: "current",
      session: { project: "p1", session: "jax-p1-lead", pane: "%7", role: "lead", tmux_incarnation: "111:222" },
    });
  });

  it("accepts role adhoc, rejects role outside lead|adhoc or a missing role", () => {
    expect(parseSessionBody({ project: "p1", session: "s", pane: "%7", role: "adhoc", tmux_incarnation: "1:2" })).toMatchObject({ ok: true });
    expect(parseSessionBody({ project: "p1", session: "s", pane: "%7", role: "builder", tmux_incarnation: "1:2" })).toMatchObject({ ok: false });
    expect(parseSessionBody({ project: "p1", session: "s", pane: "%7", tmux_incarnation: "1:2" })).toMatchObject({ ok: false });
  });

  it("requires tmux_incarnation — unlike the events route, this is never optional (Finding 40)", () => {
    expect(parseSessionBody({ project: "p1", session: "s", pane: "%7", role: "lead" })).toMatchObject({ ok: false });
  });

  it("reuses the PROJECT_LABEL fixtures", () => {
    for (const project of ["PIJ", "Acme.AI", "CS_standalone", "comenta.ia.br", "EBPMN 2"]) {
      expect(parseSessionBody({ project, session: "s", pane: "%7", role: "lead", tmux_incarnation: "1:2" })).toMatchObject({ ok: true });
    }
  });

  it("B6: a legacy {project, tmux_target} body is now REJECTED", () => {
    expect(parseSessionBody({ project: "p1", tmux_target: "jax-p1-lead:1.1" }))
      .toMatchObject({ ok: false, error: expect.stringMatching(/tmux_target/) });
    expect(parseSessionBody({ project: "jax-os", tmux_target: "%12" })).toMatchObject({ ok: false });
  });

  it("rejects a non-object body", () => {
    expect(parseSessionBody(null)).toMatchObject({ ok: false });
  });
});

const ROW = (over: Partial<WorkflowEventRow>): WorkflowEventRow => ({
  id: 7, ts: "2026-08-18T12:00:00.000Z", run_id: null, project: "jax-os", role: "lead", type: "turn-stopped",
  source: "deterministic", emitter: "claude-stop", payload: {}, delivery: "pending", forwarded_at: null, ...over,
});

describe("renderMessage", () => {
  it("renders run-finished exactly as the matrix action column, flagging non-ok reports", () => {
    const base = ROW({ run_id: "r-001", role: "builder", type: "run-finished", emitter: "wrapper", payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x/r-001.md", summary: "all green", head_sha: null } });
    expect(renderMessage(base)).toBe("Project jax-os (phase B): builder run finished — no result, exit 0. Report at /x/r-001.md.");
    expect(renderMessage({ ...base, payload: { ...base.payload, exit_code: 1, contract_status: "missing" } })).toBe(
      "Project jax-os (phase B): builder run finished — no result, exit 1 [report missing]. Report at /x/r-001.md.",
    );
  });

  it("renders a builder run-finished with result/stage/diagnostic (spec §12.3)", () => {
    const row = ROW({ run_id: "r-001", role: "builder", type: "run-finished", emitter: "wrapper", payload: { phase: "B", exit_code: 1, contract_status: "ok", report_path: "/x/r-001.md", summary: "s", head_sha: null, result: "failure", stage: "runtime", diagnostic: "APIError 403 API key budget limit exceeded" } });
    expect(renderMessage(row)).toBe(
      "Project jax-os (phase B): builder run finished — failure · runtime — APIError 403 API key budget limit exceeded, exit 1. Report at /x/r-001.md.",
    );
  });

  it("renders a reviewer run-finished with a real verdict, and the literal no verdict when absent", () => {
    const approved = ROW({ run_id: "r-002", role: "reviewer", type: "run-finished", emitter: "wrapper", payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x/r-002.md", summary: "s", verdict: "approve" } });
    expect(renderMessage(approved)).toBe("Project jax-os (phase B): reviewer run finished — approve, exit 0. Report at /x/r-002.md.");
    const noVerdict = ROW({ run_id: "r-003", role: "reviewer", type: "run-finished", emitter: "wrapper", payload: { phase: "B", exit_code: 1, contract_status: "invalid", report_path: "/x/r-003.md", summary: "s", stage: "report", diagnostic: "report invalid" } });
    expect(renderMessage(noVerdict)).toBe(
      "Project jax-os (phase B): reviewer run finished — no verdict · report — report invalid, exit 1 [report invalid]. Report at /x/r-003.md.",
    );
  });

  it("never renders a verdict carried by a non-ok reviewer row (diff review 754d9e56254d F1)", () => {
    const row = ROW({ run_id: "r-006", role: "reviewer", type: "run-finished", emitter: "wrapper", payload: { phase: "B", exit_code: 1, contract_status: "invalid", report_path: "/x/r-006.md", summary: "s", verdict: "approve" } });
    expect(renderMessage(row)).toBe("Project jax-os (phase B): reviewer run finished — no verdict, exit 1 [report invalid]. Report at /x/r-006.md.");
  });

  it("renders the interrupted branch with no report_path/report-flag segment (spec §12.3)", () => {
    const row = ROW({ run_id: "r-004", role: "builder", type: "run-finished", emitter: "wrapper", payload: { phase: "B", exit_code: null, contract_status: "interrupted", report_path: null, summary: "worker interrupted by SIGHUP", result: "failure", stage: "worker", diagnostic: "worker interrupted by SIGHUP", head_sha: null } });
    expect(renderMessage(row)).toBe("Project jax-os (phase B): builder run interrupted — failure · worker — worker interrupted by SIGHUP.");
  });

  it("interrupted row 1a renders stage runtime, not the literal worker (spec §Test matrix)", () => {
    const row = ROW({ run_id: "r-005", role: "reviewer", type: "run-finished", emitter: "wrapper", payload: { phase: "B", exit_code: null, contract_status: "interrupted", report_path: null, summary: "APIError 403", stage: "runtime", diagnostic: "APIError 403 API key budget limit exceeded" } });
    expect(renderMessage(row)).toBe("Project jax-os (phase B): reviewer run interrupted — no verdict · runtime — APIError 403 API key budget limit exceeded.");
  });
  it("renders questions with 1-based numbered options and the reply hint", () => {
    const q = ROW({ type: "question", emitter: "claude-pretool", payload: { tool_use_id: "t", tmux_target: null, questions: [{ question: "Merge?", options: [{ label: "Yes" }, { label: "No", description: "wait for CI" }] }, { question: "Which?", multiSelect: true, options: [{ label: "A" }] }] } });
    expect(renderMessage(q)).toBe(
      "Project jax-os: the tech lead asks —\nQ1: Merge?\n  1) Yes\n  2) No — wait for CI\nQ2: Which? (choose one or more)\n  1) A\nReply to this message once, one line per question: \"1: 2\" or \"2: 1,3\"; text is also allowed after the colon.",
    );
  });
  it("renders attention/turn-stopped with the viewer link when given, words otherwise", () => {
    const a = ROW({ type: "attention-needed", source: "behavioral", emitter: "claude-notification", payload: { reason: "agent_needs_input", excerpt: "secret pane" } });
    expect(renderMessage(a, "https://viewer.example")).toBe("Project jax-os: the tech lead needs attention (agent_needs_input). Viewer: https://viewer.example");
    expect(renderMessage(a)).toBe("Project jax-os: the tech lead needs attention (agent_needs_input). Open the tmux viewer in Jax OS.");
    expect(renderMessage(ROW({ payload: { excerpt: "Merge B now." } }))).toBe("Project jax-os: tech lead turn ended. Proposal: Merge B now. Open the tmux viewer in Jax OS.");
    expect(renderMessage(ROW({}))).toBe("Project jax-os: tech lead turn ended. Open the tmux viewer in Jax OS.");
  });
  it("renders the local types too (dashboard use)", () => {
    expect(renderMessage(ROW({ run_id: "r1", role: "reviewer", type: "run-started", emitter: "wrapper", payload: { phase: "B", runtime: "codex" } }))).toBe("Project jax-os (phase B): reviewer run started on codex.");
    expect(renderMessage(ROW({ type: "question-resolved", emitter: "claude-posttool", payload: { tool_use_id: "t1" } }))).toBe("Project jax-os: question t1 resolved.");
    expect(renderMessage(ROW({ type: "question-answered", emitter: "jaxos", payload: { question_event_id: 3, tool_use_id: "t1", mutation_id: 9, ok: false, error: "pane gone" } }))).toBe("Project jax-os: answer injected for question #3 (failed: pane gone).");
  });
});

describe("toForwardBody", () => {
  it("has the fixed key set, strips the attention excerpt, keeps everything else", () => {
    const a = ROW({ type: "attention-needed", source: "behavioral", emitter: "claude-notification", payload: { reason: "r", excerpt: "local only" } });
    const body = toForwardBody(a);
    expect(Object.keys(body)).toEqual(["event_id", "ts", "type", "project", "run_id", "role", "source", "emitter", "message", "payload"]);
    expect(body.payload).toEqual({ reason: "r" });
    expect(body.event_id).toBe(7);
    const f = ROW({ run_id: "r-001", role: "builder", type: "run-finished", emitter: "wrapper", payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x", summary: "s", head_sha: null } });
    expect(toForwardBody(f).payload).toEqual(f.payload);
    expect(toForwardBody(f).message).toMatch(/^Project jax-os \(phase B\)/);
  });
  it("prefixes question messages with the event-id correlation marker", () => {
    const q = ROW({
      id: 42,
      type: "question",
      emitter: "claude-pretool",
      payload: { tool_use_id: "t", questions: [{ question: "Merge?", options: [{ label: "Yes" }] }] },
    });
    expect(toForwardBody(q).message.startsWith("Workflow event #42\n")).toBe(true);
  });
  it("strips tool_use_id and tmux_target from forwarded question payload only", () => {
    const stored = {
      tool_use_id: "toolu_secret",
      tmux_target: "jax-jax-os-lead:0.0",
      questions: [{ question: "Merge?", options: [{ label: "Yes" }] }],
    };
    const q = ROW({ id: 42, type: "question", emitter: "claude-pretool", payload: stored });
    const body = toForwardBody(q);
    expect(body.payload).toEqual({ questions: stored.questions });
    expect(body.message.startsWith("Workflow event #42\n")).toBe(true);
    expect(q.payload).toEqual(stored);
  });
});

describe("toForwardBody — Workflow event #<id> marker widened to question/turn-stopped/attention-needed (§4.6.5)", () => {
  it("strips message_tail from a forwarded turn-stopped and keeps every other key", () => {
    const row = ROW({ payload: { capsule_status: "unknown", capsule_rule: "deferred", capsule_attempts: 0, message_tail: "secret prose" } });
    const body = toForwardBody(row);
    expect(body.payload).toEqual({ capsule_status: "unknown", capsule_rule: "deferred", capsule_attempts: 0 });
    expect(JSON.stringify(body)).not.toContain("secret prose");
  });

  it("prefixes a turn-stopped forward with Workflow event #<id> (was unprefixed in v0)", () => {
    const event = ROW({ id: 7, type: "turn-stopped", payload: { capsule_status: "done" } });
    expect(toForwardBody(event).message.startsWith("Workflow event #7\n")).toBe(true);
  });

  it("prefixes an attention-needed forward with Workflow event #<id> (was unprefixed in v0)", () => {
    const event = ROW({
      type: "attention-needed", source: "behavioral", emitter: "claude-notification",
      payload: { reason: "no-signal" },
    });
    expect(toForwardBody(event).message.startsWith("Workflow event #7\n")).toBe(true);
  });

  it("changes question's marker text from Workflow question #<id> to Workflow event #<id>", () => {
    const event = ROW({
      id: 7, type: "question", emitter: "claude-pretool",
      payload: { tool_use_id: "tu_1", questions: [] },
    });
    expect(toForwardBody(event).message.startsWith("Workflow event #7\n")).toBe(true);
    expect(toForwardBody(event).message.startsWith("Workflow question #7\n")).toBe(false);
  });

  it("never prefixes run-finished (forward policy but never answerable)", () => {
    const event = ROW({
      run_id: "r1", role: "builder", type: "run-finished", emitter: "wrapper",
      payload: { phase: "B", exit_code: 0, contract_status: "ok", report_path: "/x", summary: "s", head_sha: null },
    });
    expect(toForwardBody(event).message.startsWith("Workflow event #")).toBe(false);
    expect(toForwardBody(event).message.startsWith("Workflow question #")).toBe(false);
  });
});

const SHAPES = [
  { multiSelect: false, option_count: 2 },
  { multiSelect: true, option_count: 3 },
  { multiSelect: false, option_count: 2 },
];

describe("parseAnswerBody", () => {
  it("accepts only event id plus the bounded raw Telegram reply", () => {
    expect(parseAnswerBody({ question_event_id: 7, reply: "1: 2\n2: 1,3\n3: ship without deploy" })).toEqual({
      ok: true,
      answer: { question_event_id: 7, reply: "1: 2\n2: 1,3\n3: ship without deploy" },
    });
    expect(parseAnswerBody({ question_event_id: 7, tool_use_id: "client-token", reply: "1: 1" }).ok).toBe(false);
    expect(parseAnswerBody({ question_event_id: 7, reply: "x".repeat(20_501) }).ok).toBe(false);
    expect(parseAnswerBody({ question_event_id: 0, reply: "1: 1" }).ok).toBe(false);
  });

  it("rejects tab, DEL and other C0 control characters, but keeps CR/LF as line separators (Finding 13)", () => {
    // The shared guard was only covered on the freeform claim path (workflows.test.ts "guard 8"),
    // so deleting this caller's check left every checked-in test green — Finding 13.
    for (const bad of ["1: 1\tx", "1: 1\x7f", "1: 1\x01", "1: 1\x1f", "1: 1\x0b", "1: 1\x0c", "1: 1\x00"]) {
      expect(parseAnswerBody({ question_event_id: 7, reply: bad })).toEqual({
        ok: false, error: "reply contains control characters",
      });
    }
    expect(parseAnswerBody({ question_event_id: 7, reply: "1: 1\n2: 2" }).ok).toBe(true);
    expect(parseAnswerBody({ question_event_id: 7, reply: "1: 1\r\n2: 2" }).ok).toBe(true);
    expect(parseAnswerBody({ question_event_id: 7, reply: "1: 1\r2: 2" }).ok).toBe(true);
  });
});

describe("parseAnswerBody — event_id alias (D4)", () => {
  it("accepts event_id as an alias for question_event_id", () => {
    expect(parseAnswerBody({ event_id: 7, reply: "ship it" })).toEqual({
      ok: true, answer: { question_event_id: 7, reply: "ship it" },
    });
  });

  it("still accepts the original question_event_id field name", () => {
    expect(parseAnswerBody({ question_event_id: 7, reply: "1: 1" })).toEqual({
      ok: true, answer: { question_event_id: 7, reply: "1: 1" },
    });
  });

  it("accepts both id fields when they agree", () => {
    expect(parseAnswerBody({ event_id: 7, question_event_id: 7, reply: "ship it" })).toEqual({
      ok: true, answer: { question_event_id: 7, reply: "ship it" },
    });
  });

  it("rejects both id fields when they disagree (spec §6 route 5)", () => {
    const r = parseAnswerBody({ event_id: 7, question_event_id: 8, reply: "ship it" });
    expect(r.ok).toBe(false);
    if (!r.ok) expect(r.error).not.toMatch(/unknown key/);
  });

  it("rejects a body carrying neither field", () => {
    expect(parseAnswerBody({ reply: "ship it" }).ok).toBe(false);
  });
});

describe("parseIndexedReply", () => {
  it("parses one complete indexed reply in stored-question order", () => {
    expect(parseIndexedReply("3: ship without deploy\n1: 2\n2: 3,1", SHAPES)).toEqual({ ok: true, answers: [
      { question_number: 1, kind: "options", values: [2] },
      { question_number: 2, kind: "options", values: [1, 3] },
      { question_number: 3, kind: "text", value: "ship without deploy" },
    ] });
  });

  it("normalizes CRLF replies before classifying numeric answers", () => {
    expect(parseIndexedReply("1: 2\r\n2: 1,3\r\n3: note", SHAPES)).toEqual(
      parseIndexedReply("1: 2\n2: 1,3\n3: note", SHAPES),
    );
  });

  it("accepts the rendered Q prefix and normalizes it away", () => {
    expect(parseIndexedReply("Q1: 2\nq2: 1,3\nQ3: note", SHAPES)).toEqual(
      parseIndexedReply("1: 2\n2: 1,3\n3: note", SHAPES),
    );
  });

  it("rejects incomplete, duplicate, malformed, cardinality-invalid, and out-of-range replies", () => {
    expect(parseIndexedReply("1: 2\n2: 1,3", SHAPES).ok).toBe(false);
    expect(parseIndexedReply("1: 2\n1: 1\n3: note", SHAPES).ok).toBe(false);
    expect(parseIndexedReply("1: 1,2\n2: 1\n3: note", SHAPES).ok).toBe(false);
    expect(parseIndexedReply("1: 2\n2: 4\n3: note", SHAPES).ok).toBe(false);
    expect(parseIndexedReply("1: 2\n2: 1,1\n3: note", SHAPES).ok).toBe(false);
    expect(parseIndexedReply("1: 2\n2: 1,3\n4: note", SHAPES).ok).toBe(false);
    expect(parseIndexedReply("1: 0\n2: 1\n3: note", SHAPES).ok).toBe(false);
    expect(parseIndexedReply("1: \n2: 1\n3: note", SHAPES).ok).toBe(false);
  });
});

describe("cleanup_error payload", () => {
  it("accepts cleanup_error only on run-finished at the error bound", () => {
    expect(parseIngress({
      ...RUN_FINISHED,
      payload: { ...RUN_FINISHED.payload, cleanup_error: "scratch rmtree failed" },
    }).ok).toBe(true);
    expect(parseIngress({
      ...RUN_FINISHED,
      payload: { ...RUN_FINISHED.payload, cleanup_error: "x".repeat(500) },
    }).ok).toBe(true);
    expectInvalid({
      ...RUN_FINISHED,
      payload: { ...RUN_FINISHED.payload, cleanup_error: "x".repeat(501) },
    }, /cleanup_error exceeds 500/);
    expectInvalid({
      ...RUN_STARTED,
      payload: { ...RUN_STARTED.payload, cleanup_error: "nope" },
    }, /unknown key cleanup_error/);
    expectInvalid({
      ...QUESTION,
      payload: { ...QUESTION.payload, cleanup_error: "nope" },
    }, /unknown key cleanup_error/);
  });
});

// jaxflow slice A (spec §2.6): run-started/run-finished gain the run-ledger
// fields, contract_status gains "cancelled", RUNTIMES gains "claude".
describe("parseIngress — jaxflow run-ledger fields (spec §2.6, §7.2 #28-30, #32)", () => {
  it("accepts run-started with all new fields, rejects an out-of-bound or out-of-enum one", () => {
    expect(parseIngress(RUN_STARTED).ok).toBe(true);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, kind: "nonsense" } }, /kind not allowed/);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, caller: "gemini" } }, /caller not allowed/);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, target: "x".repeat(513) } }, /target exceeds 512/);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, model: "x".repeat(385) } }, /model exceeds 384/);
  });
  it("accepts opencode-builder and combined-id/unicode models at the new ceiling", () => {
    expect(parseIngress({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, runtime: "opencode-builder" } }).ok).toBe(true);
    expect(parseIngress({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, model: "x".repeat(384) } }).ok).toBe(true);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, model: "x".repeat(385) } }, /model exceeds 384/);
    expect(parseIngress({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, model: "é".repeat(384) } }).ok).toBe(true);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, model: "é".repeat(385) } }, /model exceeds 384/);
    expect(parseIngress({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, model: "conn/" + "m".repeat(200) } }).ok).toBe(true);
    expect(parseIngress({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, effort: "x".repeat(64) } }).ok).toBe(true);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, effort: "x".repeat(65) } }, /effort exceeds 64/);
  });
  it("requires lineage on managed builder run-started and rejects it elsewhere", () => {
    const managed = {
      ...RUN_STARTED.payload, runtime: "opencode-builder", kind: "build", verify: "true",
      requested_profile: "default", root_build_run_id: "aabbccddeeff",
    };
    expect(parseIngress({ ...RUN_STARTED, payload: managed }).ok).toBe(true);
    expect(parseIngress({
      ...RUN_STARTED,
      payload: { ...managed, requested_profile: "fallback", resumes_run_id: "ffeeddccbbaa" },
    }).ok).toBe(true);
    const { requested_profile: _dropProfile, ...noProfile } = managed;
    expectInvalid({ ...RUN_STARTED, payload: noProfile }, /requested_profile required/);
    const { root_build_run_id: _dropRoot, ...noRoot } = managed;
    expectInvalid({ ...RUN_STARTED, payload: noRoot }, /root_build_run_id required/);
    expectInvalid({ ...RUN_STARTED, payload: { ...managed, requested_profile: "other" } }, /requested_profile not allowed/);
    expectInvalid({ ...RUN_STARTED, payload: { ...managed, root_build_run_id: "AABBCCDDEEFF" } }, /root_build_run_id malformed/);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, requested_profile: "default" } }, /lineage only for managed builder runs/);
    expectInvalid({ ...RUN_STARTED, payload: { ...managed, extra: 1 } }, /unknown key extra/);
  });
  it("accepts run-started with caller_pane absent entirely (no TMUX_PANE at dispatch)", () => {
    const { caller_pane: _drop, ...p } = RUN_STARTED.payload;
    const r = parseIngress({ ...RUN_STARTED, payload: p });
    expect(r.ok).toBe(true);
    if (r.ok) expect("caller_pane" in r.event.payload).toBe(false);
  });
  it("requires verify for kind build, forbids it otherwise", () => {
    const buildPayload = { ...RUN_STARTED.payload, kind: "build", verify: "pnpm exec vitest run src" };
    expect(parseIngress({ ...RUN_STARTED, payload: buildPayload }).ok).toBe(true);
    const { verify: _drop, ...noVerify } = buildPayload;
    expectInvalid({ ...RUN_STARTED, payload: noVerify }, /verify required/);
    expectInvalid({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, verify: "echo hi" } }, /verify\/build only for kind build/);
  });
  it("verdict is reviewer-only, present exactly when contract_status is ok", () => {
    const { head_sha: _drop, result: _dropResult, ...base } = RUN_FINISHED.payload;
    expect(parseIngress({ ...RUN_FINISHED, role: "reviewer", payload: { ...base, verdict: "approve" } }).ok).toBe(true);
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, verdict: "approve" } }, /verdict only for reviewer runs/);
    expectInvalid({ ...RUN_FINISHED, role: "reviewer", payload: base }, /verdict required for reviewer runs/);
  });
  it("result is builder-only and required when contract_status is ok", () => {
    expect(parseIngress(RUN_FINISHED).ok).toBe(true);
    const { head_sha: _drop, result: _dropResult, ...base } = RUN_FINISHED.payload;
    expectInvalid(
      { ...RUN_FINISHED, role: "reviewer", payload: { ...base, verdict: "approve", result: "success" } },
      /result only for builder runs/,
    );
    const { result: _dropResult2, ...noResult } = RUN_FINISHED.payload;
    expectInvalid({ ...RUN_FINISHED, payload: noResult }, /result required for builder runs/);
  });
  it.each(["missing", "invalid"])("accepts verified builder results with a %s report", (contract_status) => {
    const base = { ...RUN_FINISHED.payload, contract_status };
    for (const result of ["success", "failure", "blocked"]) {
      expect(parseIngress({ ...RUN_FINISHED, payload: { ...base, result } })).toMatchObject({ ok: true, event: { payload: { contract_status, result } } });
    }
    for (const result of [null, "unknown", 1, {}]) expectInvalid({ ...RUN_FINISHED, payload: { ...base, result } }, /result/);
    expectInvalid({ ...RUN_FINISHED, role: "reviewer", payload: { ...base, head_sha: null } }, /result only for builder runs/);
  });
  it("accepts contract_status cancelled with null exit_code/report_path, rejects null elsewhere", () => {
    const { result: _drop, ...base } = RUN_FINISHED.payload;
    const cancelled = { ...base, contract_status: "cancelled", exit_code: null, report_path: null, summary: "cancelled by claude at 2026-09-06T12:00:00Z" };
    expect(parseIngress({ ...RUN_FINISHED, payload: cancelled }).ok).toBe(true);
    expectInvalid({ ...RUN_FINISHED, payload: { ...cancelled, result: "failure" } }, /result only for builder runs/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...cancelled, contract_status: "ok" } }, /exit_code must be an integer/);
  });
  it("renders a cancelled run-finished as 'run cancelled: <summary>', not the exit/report format", () => {
    const row = ROW({
      run_id: "r-001", role: "builder", type: "run-finished", emitter: "wrapper",
      payload: { phase: "B", exit_code: null, contract_status: "cancelled", report_path: null, summary: "cancelled by claude at 2026-09-06T12:00:00Z", head_sha: null },
    });
    expect(renderMessage(row)).toBe("Project jax-os (phase B): builder run cancelled: cancelled by claude at 2026-09-06T12:00:00Z.");
  });
  it("accepts runtime claude on run-started (fixes cold review G4)", () => {
    expect(parseIngress({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, runtime: "claude" } }).ok).toBe(true);
  });
  it("accepts tail+reason together on a missing/invalid row, rejects either alone (MOA-470 §4.3)", () => {
    const { result: _drop, ...base } = RUN_FINISHED.payload;
    const missingRow = { ...base, contract_status: "missing", exit_code: 0, tail: "no report written", reason: "no-report" };
    expect(parseIngress({ ...RUN_FINISHED, payload: missingRow }).ok).toBe(true);
    const invalidRow = { ...missingRow, contract_status: "invalid" };
    expect(parseIngress({ ...RUN_FINISHED, payload: invalidRow }).ok).toBe(true);
    // legacy shape: neither key, still accepted (backward compatibility).
    const { tail: _t, reason: _r, ...legacyMissing } = missingRow;
    expect(parseIngress({ ...RUN_FINISHED, payload: legacyMissing }).ok).toBe(true);
    // exactly one of the two present is rejected -- they travel together or not at all.
    const { reason: _dropReason, ...tailOnly } = missingRow;
    expectInvalid({ ...RUN_FINISHED, payload: tailOnly }, /reason required when tail is present/);
    const { tail: _dropTail, ...reasonOnly } = missingRow;
    expectInvalid({ ...RUN_FINISHED, payload: reasonOnly }, /tail required when reason is present/);
  });
  it("rejects tail/reason on an ok or cancelled row", () => {
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, reason: "crash" } }, /reason only for contract_status missing\/invalid/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, tail: "x" } }, /tail only for contract_status missing\/invalid/);
    const { result: _drop, ...cancelledBase } = RUN_FINISHED.payload;
    const cancelled = { ...cancelledBase, contract_status: "cancelled", exit_code: null, report_path: null, tail: "x", reason: "crash" };
    // Cold review 786debd43314 F2: both keys are present here, and the validator below
    // checks `tail` before `reason` (same order as the ok-row `tail`-only case above) --
    // so `tail`'s message wins even though `reason` is ALSO invalid on this row. The
    // check order is deterministic, not arbitrary: it matches the `tail`-before-`reason`
    // order the "required when present" branch below already uses.
    expectInvalid({ ...RUN_FINISHED, payload: cancelled }, /tail only for contract_status missing\/invalid/);
  });
  it("rejects tail over 4096 chars and reason outside the enum", () => {
    const { result: _drop, ...base } = RUN_FINISHED.payload;
    const row = { ...base, contract_status: "missing", exit_code: 0, reason: "no-report", tail: "x".repeat(4097) };
    expectInvalid({ ...RUN_FINISHED, payload: row }, /tail exceeds 4096/);
    expectInvalid({ ...RUN_FINISHED, payload: { ...row, tail: "ok", reason: "not-a-real-reason" } }, /reason not allowed/);
  });
  it("counts tail length by Unicode code points, not UTF-16 units (final review 1f119d5d1420 F2)", () => {
    // Python's `_bound` (scripts/jaxflow_run.py:501) walks CODE POINTS -- a non-BMP
    // emoji is one unit of the 4096 budget. Plain `.length` counts UTF-16 CODE UNITS
    // (a surrogate pair, two units). 2049 emojis = 2049 code points / 4098 UTF-16 units,
    // so `.length` would reject and `[...t].length` accepts. 4096 emojis cannot be the
    // fixture: they exceed LIMITS.payloadBytes (16 KiB) before the char-count check runs.
    const { result: _drop, ...base } = RUN_FINISHED.payload;
    const underCap = { ...base, contract_status: "missing", exit_code: 0, reason: "no-report", tail: "😀".repeat(2049) };
    expect(parseIngress({ ...RUN_FINISHED, payload: underCap }).ok).toBe(true);
    const over4096 = { ...underCap, tail: "😀".repeat(2049) + "x".repeat(2048) };
    expectInvalid({ ...RUN_FINISHED, payload: over4096 }, /tail exceeds 4096/);
  });
  it("accepts the three MOA-495 2.1 reasons added alongside Jev refinement", () => {
    const { result: _drop, ...base } = RUN_FINISHED.payload;
    for (const reason of ["hung", "test-failure", "contract-violation"]) {
      const row = { ...base, contract_status: "missing", exit_code: 0, reason, tail: "x" };
      expect(parseIngress({ ...RUN_FINISHED, payload: row }).ok).toBe(true);
    }
  });
  it("accepts log_context alongside tail/reason, up to 6 single-line entries", () => {
    const { result: _drop, ...base } = RUN_FINISHED.payload;
    const row = {
      ...base, contract_status: "missing", exit_code: 0, reason: "crash", tail: "x",
      log_context: ["first relevant chunk", "second relevant chunk"],
    };
    const r = parseIngress({ ...RUN_FINISHED, payload: row });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.event.payload.log_context).toEqual(["first relevant chunk", "second relevant chunk"]);
  });
  it("rejects log_context on an ok row, an empty array, over 6 entries, or an over-length/multi-line entry", () => {
    const { result: _drop, ...base } = RUN_FINISHED.payload;
    expectInvalid(
      { ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, log_context: ["x"] } },
      /log_context only for contract_status missing\/invalid/,
    );
    const row = { ...base, contract_status: "missing", exit_code: 0, reason: "crash", tail: "x" };
    expectInvalid({ ...RUN_FINISHED, payload: { ...row, log_context: [] } }, /log_context must be an array/);
    expectInvalid(
      { ...RUN_FINISHED, payload: { ...row, log_context: Array(7).fill("x") } },
      /log_context must be an array/,
    );
    expectInvalid(
      { ...RUN_FINISHED, payload: { ...row, log_context: ["x".repeat(501)] } },
      /log_context entry exceeds 500/,
    );
    expectInvalid(
      { ...RUN_FINISHED, payload: { ...row, log_context: ["line one\nline two"] } },
      /log_context entries must be a single line/,
    );
  });
});

// MOA-474 §11.1: contract_status gains "interrupted" (the not-finalized family), and
// run-finished gains stage/diagnostic -- optional/additive except on interrupted, where
// they are mandatory, and cancelled, where they stay forbidden.
describe("parseIngress — MOA-474 interrupted rows and stage/diagnostic", () => {
  it("run-finished accepts contract_status interrupted with stage/diagnostic", () => {
    const result = parseIngress({
      run_id: "aaaabbbbcccc", project: "demo", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: {
        phase: "P1", exit_code: null, contract_status: "interrupted", report_path: null,
        summary: "worker interrupted by SIGTERM", stage: "worker",
        diagnostic: "worker interrupted by SIGTERM", head_sha: null, result: "failure",
      },
    });
    expect(result.ok).toBe(true);
  });

  it("run-finished rejects stage outside the enum", () => {
    const result = parseIngress({
      run_id: "aaaabbbbcccc", project: "demo", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: {
        phase: "P1", exit_code: 1, contract_status: "missing", report_path: "/r.md",
        summary: "s", stage: "bogus", diagnostic: "d", head_sha: null,
      },
    });
    expect(result).toEqual({ ok: false, error: "stage not allowed" });
  });

  it("run-finished requires stage and diagnostic when contract_status is interrupted", () => {
    const base = {
      run_id: "aaaabbbbcccc", project: "demo", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
    } as const;
    const noStage = parseIngress({
      ...base,
      payload: {
        phase: "P1", exit_code: null, contract_status: "interrupted", report_path: null,
        summary: "s", diagnostic: "d", head_sha: null, result: "failure",
      },
    });
    expect(noStage).toEqual({ ok: false, error: "stage required for contract_status interrupted" });
    const noDiagnostic = parseIngress({
      ...base,
      payload: {
        phase: "P1", exit_code: null, contract_status: "interrupted", report_path: null,
        summary: "s", stage: "worker", head_sha: null, result: "failure",
      },
    });
    expect(noDiagnostic).toEqual({ ok: false, error: "diagnostic required for contract_status interrupted" });
  });

  it("run-finished rejects verify/build stage on an interrupted row", () => {
    const result = parseIngress({
      run_id: "aaaabbbbcccc", project: "demo", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: {
        phase: "P1", exit_code: null, contract_status: "interrupted", report_path: null,
        summary: "s", stage: "verify", diagnostic: "d", head_sha: null, result: "failure",
      },
    });
    expect(result).toEqual({ ok: false, error: "stage must be runtime or worker for contract_status interrupted" });
  });

  it("run-finished forbids stage/diagnostic on a cancelled row", () => {
    const result = parseIngress({
      run_id: "aaaabbbbcccc", project: "demo", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: {
        phase: "P1", exit_code: null, contract_status: "cancelled", report_path: null,
        summary: "cancelled by claude at 2026-01-01T00:00:00Z", stage: "worker", head_sha: null,
      },
    });
    expect(result).toEqual({ ok: false, error: "stage only for contract_status other than cancelled" });
  });

  it("run-finished forbids verdict on an interrupted reviewer row", () => {
    const result = parseIngress({
      run_id: "aaaabbbbcccc", project: "demo", role: "reviewer", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: {
        phase: "P1", exit_code: null, contract_status: "interrupted", report_path: null,
        summary: "s", stage: "worker", diagnostic: "d", verdict: "approve",
      },
    });
    expect(result).toEqual({ ok: false, error: "verdict only for reviewer runs with contract_status ok" });
  });

  it("run-finished accepts diagnostic up to 300 code points, rejects 301", () => {
    const ok300 = parseIngress({
      run_id: "aaaabbbbcccc", project: "demo", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: {
        phase: "P1", exit_code: null, contract_status: "interrupted", report_path: null,
        summary: "s".repeat(200), stage: "worker", diagnostic: "d".repeat(300),
        head_sha: null, result: "failure",
      },
    });
    expect(ok300.ok).toBe(true);
    const over301 = parseIngress({
      run_id: "aaaabbbbcccc", project: "demo", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: {
        phase: "P1", exit_code: null, contract_status: "interrupted", report_path: null,
        summary: "s".repeat(200), stage: "worker", diagnostic: "d".repeat(301),
        head_sha: null, result: "failure",
      },
    });
    expect(over301).toEqual({ ok: false, error: "diagnostic exceeds 300 chars" });
  });

  it("run-finished's summary bound counts Unicode code points, not UTF-16 units", () => {
    // 200 non-BMP emoji = 200 code points but 400 UTF-16 units. Pins the D-family fix:
    // `[...v].length`, not `.length`, on the summary check (workflow-events.ts:142-143).
    const summary200Emoji = "\u{1F600}".repeat(200);
    const result = parseIngress({
      run_id: "aaaabbbbcccc", project: "demo", role: "builder", type: "run-finished",
      source: "deterministic", emitter: "wrapper",
      payload: {
        phase: "P1", exit_code: 0, contract_status: "ok", report_path: "/r.md",
        summary: summary200Emoji, head_sha: null, result: "success",
      },
    });
    expect(result.ok).toBe(true);
  });
});

describe("run-started build command (MOA-454)", () => {
  it("accepts a kind:build payload carrying both verify and build", () => {
    const p = { ...RUN_STARTED.payload, kind: "build", verify: "pnpm test", build: "pnpm build" };
    expect(parseIngress({ ...RUN_STARTED, payload: p }).ok).toBe(true);
  });

  it("accepts a kind:build payload with verify only — build is optional", () => {
    const p = { ...RUN_STARTED.payload, kind: "build", verify: "pnpm test" };
    expect("build" in p).toBe(false);
    expect(parseIngress({ ...RUN_STARTED, payload: p }).ok).toBe(true);
  });

  it("rejects build on a payload whose kind is not build", () => {
    const p = { ...RUN_STARTED.payload, build: "pnpm build" };
    expectInvalid({ ...RUN_STARTED, payload: p }, /verify\/build only for kind build/);
  });

  it("rejects a build value over LIMITS.build", () => {
    const p = { ...RUN_STARTED.payload, kind: "build", verify: "pnpm test", build: "x".repeat(LIMITS.build + 1) };
    expectInvalid({ ...RUN_STARTED, payload: p }, /build exceeds/);
  });
});

const mergeEvent = (over: Record<string, unknown> = {}) => ({
  project: "jax-os",
  role: "lead",
  type: "merge-approved",
  source: "deterministic",
  emitter: "wrapper",
  payload: {
    phase: "Phase X",
    branch: "feat/x",
    sha: "a".repeat(40),
    target: "main",
    approved_by: "rafa",
    merge_sha: "c".repeat(40),
  },
  ...over,
});

const prOpenedEvent = (over: Record<string, unknown> = {}) => ({
  project: "jax-os",
  role: "lead",
  type: "pr-opened",
  source: "deterministic",
  emitter: "wrapper",
  payload: {
    repo: "acme/route-converter-se",
    branch: "feat/x",
    sha: "a".repeat(40),
    base: "staging",
    pr_number: 7,
    pr_url: "https://github.com/acme/route-converter-se/pull/7",
    kind: "feature",
  },
  ...over,
});

describe("merge-approved", () => {
  it("validates a well-formed event with a pane", () => {
    const r = parseIngress(mergeEvent({ pane: "%1" }));
    expect(r.ok).toBe(true);
  });

  it("validates without a pane at all (wrapper carve-out to the lead pane rule)", () => {
    const r = parseIngress(mergeEvent());
    expect(r.ok).toBe(true);
  });

  it("rejects a malformed pane", () => {
    const r = parseIngress(mergeEvent({ pane: "not-a-pane" }));
    expect(r).toMatchObject({ ok: false, error: "pane malformed" });
  });

  it("still requires a pane for a lead event from a hook emitter", () => {
    const r = parseIngress({
      project: "jax-os", role: "lead", type: "turn-started",
      source: "deterministic", emitter: "claude-userprompt", payload: {},
    });
    expect(r).toMatchObject({ ok: false, error: "pane required" });
  });

  it.each(["sha", "merge_sha"])("rejects a %s that is not full 40-hex", (key) => {
    const ev = mergeEvent();
    (ev.payload as Record<string, unknown>)[key] = "abc123";
    expect(parseIngress(ev).ok).toBe(false);
  });

  it.each([
    [" Phase X ", "untrimmed"],
    ["Release\nsecond line", "an embedded newline"],
    ["A\tB", "a tab"],
    ["", "empty"],
    ["\u{1F680}".repeat(201), "over 200 UTF-16 code units"],
    ["\uD800", "a lone high surrogate"],
    ["Phase \uDFFF X", "a lone low surrogate inside a title"],
    ["A\u2028\uD800", "a lone surrogate after a LINE SEPARATOR — `.` would not have seen it"],
    ["A\u2029\uD800", "the same after a PARAGRAPH SEPARATOR"],
    ["\uFEFFPhase X", "a leading BOM — the one character JS calls whitespace and Python"
                    + " does not, so the producer rejects it too rather than relying on"
                    + " this side (round 4)"],
  ])("rejects %s (%s) — the same contract cmd_merge enforces", (phase) => {
    const ev = mergeEvent();
    (ev.payload as Record<string, unknown>).phase = phase;
    expect(parseIngress(ev).ok).toBe(false);
  });

  it("accepts a title of exactly 200 UTF-16 code units", () => {
    // 100 emoji are 100 characters to Python and 200 units to `String.length`; the
    // producer bounds by units precisely so this case agrees on both sides.
    const ev = mergeEvent();
    (ev.payload as Record<string, unknown>).phase = "\u{1F680}".repeat(100);
    expect(parseIngress(ev).ok).toBe(true);
  });

  it("rejects an unknown payload key", () => {
    const ev = mergeEvent();
    (ev.payload as Record<string, unknown>).extra = "x";
    expect(parseIngress(ev).ok).toBe(false);
  });

  it("accepts a reused checks object (review run id + head sha)", () => {
    const ev = mergeEvent();
    (ev.payload as Record<string, unknown>).checks = {
      mode: "reused", source_review_run_id: "e".repeat(12), head_sha: "a".repeat(40),
    };
    expect(parseIngress(ev).ok).toBe(true);
  });

  it("accepts the run and resumed checks objects, which carry mode only", () => {
    for (const mode of ["run", "resumed"]) {
      const ev = mergeEvent();
      (ev.payload as Record<string, unknown>).checks = { mode };
      expect(parseIngress(ev).ok).toBe(true);
    }
  });

  it("still validates events that predate the checks key", () => {
    expect(parseIngress(mergeEvent()).ok).toBe(true);
  });

  it("rejects malformed checks objects", () => {
    const bad: unknown[] = [
      "reused",
      null,
      [],
      {},
      { mode: "skipped" },
      { mode: "run", head_sha: "a".repeat(40) },
      { mode: "resumed", source_review_run_id: "e".repeat(12) },
      { mode: "reused" },
      { mode: "reused", source_review_run_id: "nope", head_sha: "a".repeat(40) },
      { mode: "reused", source_review_run_id: "e".repeat(12), head_sha: "abc" },
      { mode: "reused", source_review_run_id: "e".repeat(12), head_sha: "a".repeat(40), extra: 1 },
    ];
    for (const checks of bad) {
      const ev = mergeEvent();
      (ev.payload as Record<string, unknown>).checks = checks;
      expect(parseIngress(ev).ok, JSON.stringify(checks)).toBe(false);
    }
  });

  it("rejects a run_id (the event is not run-scoped)", () => {
    expect(parseIngress(mergeEvent({ run_id: "abc123abc123" })).ok).toBe(false);
  });

  it("renders a deterministic non-empty message", () => {
    const msg = renderMessage({
      type: "merge-approved", project: "jax-os", role: "lead",
      payload: mergeEvent().payload,
    });
    expect(msg).toContain("feat/x");
    expect(msg).toContain("main");
    expect(msg.length).toBeGreaterThan(0);
  });

  it("accepts optional pr_number/pr_url metadata for a PR-preset merge (spec Ledger & card growth)", () => {
    const ev = mergeEvent();
    (ev.payload as Record<string, unknown>).pr_number = 7;
    (ev.payload as Record<string, unknown>).pr_url = "https://github.com/acme/route-converter-se/pull/7";
    expect(parseIngress(ev).ok).toBe(true);
  });

  it("still validates pr_number/pr_url when present (positive int; a real github.com pull URL)", () => {
    const bad1 = mergeEvent();
    (bad1.payload as Record<string, unknown>).pr_number = -1;
    expect(parseIngress(bad1).ok).toBe(false);
    const bad2 = mergeEvent();
    (bad2.payload as Record<string, unknown>).pr_url = "not a url";
    expect(parseIngress(bad2).ok).toBe(false);
  });

  it("a local-preset merge omits pr_number/pr_url and still validates (unchanged from today)", () => {
    expect(parseIngress(mergeEvent()).ok).toBe(true);
  });
});

describe("run-finished — findings (round-3 cold review F1, MOA-480)", () => {
  const { head_sha: _drop, result: _dropResult, ...REVIEWER_BASE } = RUN_FINISHED.payload;

  it("accepts findings at the 0 and 999 boundaries on a reviewer ok run", () => {
    const low = parseIngress({
      ...RUN_FINISHED, role: "reviewer",
      payload: { ...REVIEWER_BASE, verdict: "approve", findings: { high: 0, medium: 0, low: 0 } },
    });
    expect(low.ok).toBe(true);
    if (low.ok) expect(low.event.payload.findings).toEqual({ high: 0, medium: 0, low: 0 });

    const high = parseIngress({
      ...RUN_FINISHED, role: "reviewer",
      payload: { ...REVIEWER_BASE, verdict: "approve", findings: { high: 999, medium: 999, low: 999 } },
    });
    expect(high.ok).toBe(true);
    if (high.ok) expect(high.event.payload.findings).toEqual({ high: 999, medium: 999, low: 999 });
  });

  it.each([
    ["over the max", { high: 1000, medium: 0, low: 0 }],
    ["negative", { high: -1, medium: 0, low: 0 }],
    ["non-integer", { high: 1.5, medium: 0, low: 0 }],
    ["wrong type", { high: "3", medium: 0, low: 0 }],
  ] as const)("drops the whole findings object, never the event, when a count is %s", (_label, findings) => {
    const r = parseIngress({
      ...RUN_FINISHED, role: "reviewer",
      payload: { ...REVIEWER_BASE, verdict: "approve", findings },
    });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.event.payload.findings).toBeUndefined();
  });

  it("rejects findings on a builder run", () => {
    expectInvalid(
      { ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, findings: { high: 0, medium: 0, low: 0 } } },
      /findings only for reviewer runs/,
    );
  });

  it("rejects findings on a non-ok reviewer run", () => {
    expectInvalid(
      {
        ...RUN_FINISHED, role: "reviewer",
        payload: { phase: "B", exit_code: 1, contract_status: "invalid", report_path: "/x/r-006.md", summary: "s", findings: { high: 0, medium: 0, low: 0 } },
      },
      /findings only for reviewer runs/,
    );
  });

  it("absent findings never fails an otherwise-valid reviewer ok payload", () => {
    expect(parseIngress({ ...RUN_FINISHED, role: "reviewer", payload: { ...REVIEWER_BASE, verdict: "approve" } }).ok).toBe(true);
  });

  it("accepts a tally string on a reviewer ok run (MOA-495 2.2)", () => {
    const r = parseIngress({
      ...RUN_FINISHED, role: "reviewer",
      payload: { ...REVIEWER_BASE, verdict: "approve-with-changes", tally: "tally: 3 findings — 1 repeated (F2≈prev F1 0.89), 2 new" },
    });
    expect(r.ok).toBe(true);
    if (r.ok) expect(r.event.payload.tally).toBe("tally: 3 findings — 1 repeated (F2≈prev F1 0.89), 2 new");
  });

  it("rejects tally on a builder run or a non-ok reviewer run, and an over-length/multi-line value", () => {
    expectInvalid(
      { ...RUN_FINISHED, payload: { ...RUN_FINISHED.payload, tally: "tally: 1 findings — 0 repeated, 1 new" } },
      /tally only for reviewer runs/,
    );
    expectInvalid(
      {
        ...RUN_FINISHED, role: "reviewer",
        payload: { phase: "B", exit_code: 1, contract_status: "invalid", report_path: "/x/r-006.md", summary: "s", tally: "tally: 1 findings — 0 repeated, 1 new" },
      },
      /tally only for reviewer runs/,
    );
    expectInvalid(
      { ...RUN_FINISHED, role: "reviewer", payload: { ...REVIEWER_BASE, verdict: "approve", tally: "x".repeat(301) } },
      /tally exceeds 300/,
    );
    expectInvalid(
      { ...RUN_FINISHED, role: "reviewer", payload: { ...REVIEWER_BASE, verdict: "approve", tally: "line one\nline two" } },
      /tally must be a single line/,
    );
  });
});

describe("gc-removed", () => {
  const GC_REMOVED = {
    project: "demo", role: "lead" as const, type: "gc-removed" as const, source: "deterministic" as const, emitter: "wrapper" as const,
    payload: { run_id: "aaaabbbbcccc", branch: "feat/x", worktree: "/home/rafa/repos/demo-feat-x", reason: "success", age_days: 12 },
  };

  it("accepts the exact shape, rejects extra/missing keys and a top-level run_id, renders non-empty", () => {
    expect(parseIngress(GC_REMOVED).ok).toBe(true);
    expectInvalid({ ...GC_REMOVED, payload: { ...GC_REMOVED.payload, extra: 1 } }, /unknown key/);
    const { reason: _drop, ...noReason } = GC_REMOVED.payload;
    expectInvalid({ ...GC_REMOVED, payload: noReason }, /reason required/);
    expectInvalid({ ...GC_REMOVED, run_id: "aaaabbbbcccc" }, /run_id only for run events/);
    expect(renderMessage({ type: "gc-removed", project: "demo", role: "lead", payload: GC_REMOVED.payload }).length).toBeGreaterThan(0);
  });
});

describe("run-started builder_run_id (diff reviews, round 3 F4)", () => {
  const DIFF_STARTED = {
    // F5 (round 2, MEDIUM): run-started is runScoped -- top-level run_id is required.
    project: "demo", run_id: "ddddeeeeffff", role: "reviewer" as const, type: "run-started" as const, source: "deterministic" as const, emitter: "wrapper" as const,
    payload: { phase: "p", runtime: "codex" as const, kind: "diff" as const, target: "feat/x", caller: "claude" as const, caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", builder_run_id: "aaaabbbbcccc" },
  };

  it("accepts a valid builder_run_id, rejects a malformed or missing one, and rejects a non-diff kind (F7, round 4)", () => {
    expect(parseIngress(DIFF_STARTED).ok).toBe(true);
    expectInvalid({ ...DIFF_STARTED, payload: { ...DIFF_STARTED.payload, builder_run_id: "not-hex" } }, /builder_run_id malformed/);
    // F7 (round 4): required at INGRESS for every NEW diff run-started event -- an
    // unlinked diff distorts loop grouping (F4/L2). A row already persisted without it
    // (predating this requirement) is a read-side compatibility concern (getLoopSummary's
    // branch fallback), never an ingress one -- see "legacy read" below.
    const { builder_run_id: _drop, ...noLink } = DIFF_STARTED.payload;
    expectInvalid({ ...DIFF_STARTED, payload: noLink }, /builder_run_id required/);
    const buildPayload = { phase: "p", runtime: "codex" as const, kind: "build" as const, target: "feat/x", caller: "claude" as const, caller_session: "s", model: "m", effort: "e", session: "sess", repo: "/r", verify: "true", builder_run_id: "aaaabbbbcccc" };
    expectInvalid({ ...DIFF_STARTED, role: "builder" as const, payload: buildPayload }, /builder_run_id only for kind diff/);
  });
});

describe("Phase 2: caller jaxos", () => {
  it("a run-started row with caller jaxos passes ingress like claude/codex", () => {
    const body = { ...RUN_STARTED, payload: { ...RUN_STARTED.payload, caller: "jaxos", caller_session: "jaxos" } };
    expect(parseIngress(body).ok).toBe(true);
    expect(parseIngress({ ...RUN_STARTED, payload: { ...RUN_STARTED.payload, caller: "dashboard" } }).ok).toBe(false);
  });
});

describe("Phase 3 — tool-used / subagent-started / subagent-stopped (spec §6, Decisions 1/2)", () => {
  const BASE = { project: "p1", role: "lead", pane: "%1", source: "deterministic" } as const;
  const CODEX = { project: "p1", role: "adhoc", source: "deterministic", harness_session: "0191f0aa-1111-7000-8000-000000000001" } as const;

  it("accepts the exact tool-used shape from both runtimes; rejects an extra key, a missing key and an over-cap name", () => {
    expect(parseIngress({ ...BASE, type: "tool-used", emitter: "claude-toolused", payload: { tool: "Bash" } }))
      .toMatchObject({ ok: true, event: { type: "tool-used", emitter: "claude-toolused", payload: { tool: "Bash" } } });
    expect(parseIngress({ ...CODEX, type: "tool-used", emitter: "codex-toolused", payload: { tool: "shell" } }).ok).toBe(true);
    expectInvalid({ ...BASE, type: "tool-used", emitter: "claude-toolused", payload: { tool: "Bash", tool_input: { command: "rm -rf" } } }, /unknown key tool_input/);
    expectInvalid({ ...BASE, type: "tool-used", emitter: "claude-toolused", payload: {} }, /tool required/);
    expectInvalid({ ...BASE, type: "tool-used", emitter: "claude-toolused", payload: { tool: "x".repeat(65) } }, /tool exceeds 64/);
    expectInvalid({ ...BASE, type: "tool-used", emitter: "claude-toolused", payload: { tool: "Bash\n" } }, /tool malformed/);
  });

  it("accepts subagent-started with agent_id + agent_type only and subagent-stopped with agent_id only", () => {
    expect(parseIngress({ ...BASE, type: "subagent-started", emitter: "claude-subagentstart", payload: { agent_id: "a1", agent_type: "Explore" } }).ok).toBe(true);
    expect(parseIngress({ ...BASE, type: "subagent-stopped", emitter: "claude-subagentstop", payload: { agent_id: "a1" } }).ok).toBe(true);
    expect(parseIngress({ ...CODEX, type: "subagent-started", emitter: "codex-subagentstart", payload: { agent_id: "a1", agent_type: "worker" } }).ok).toBe(true);
    expectInvalid({ ...BASE, type: "subagent-started", emitter: "claude-subagentstart", payload: { agent_id: "a1" } }, /agent_type required/);
    expectInvalid({ ...BASE, type: "subagent-started", emitter: "claude-subagentstart", payload: { agent_id: "a1", agent_type: "Explore", prompt: "task text" } }, /unknown key prompt/);
    expectInvalid({ ...BASE, type: "subagent-stopped", emitter: "claude-subagentstop", payload: { agent_id: "a1", agent_type: "Explore" } }, /unknown key agent_type/);
  });

  it("emitter/type pairs are exact: a stop emitter cannot post a start; wrapper and jaxos cannot post telemetry", () => {
    expectInvalid({ ...BASE, type: "subagent-started", emitter: "claude-subagentstop", payload: { agent_id: "a1", agent_type: "x" } }, /cannot emit subagent-started/);
    expectInvalid({ ...BASE, type: "tool-used", emitter: "jaxos", payload: { tool: "Bash" } }, /cannot emit tool-used/);
    expectInvalid({ ...BASE, type: "tool-used", emitter: "wrapper", payload: { tool: "Bash" } }, /cannot emit tool-used/);
  });

  it("a codex telemetry row is pane-less but needs the canonical session uuid; a claude one needs its pane", () => {
    expectInvalid({ project: "p1", role: "lead", type: "tool-used", emitter: "codex-toolused", source: "deterministic", payload: { tool: "shell" } }, /canonical session uuid/);
    expectInvalid({ project: "p1", role: "lead", type: "tool-used", emitter: "claude-toolused", source: "deterministic", payload: { tool: "Bash" } }, /pane required/);
  });

  it("EVENT_MATRIX stays exhaustive over EVENT_TYPES and the telemetry list is exactly the three kinds", () => {
    for (const type of EVENT_TYPES) expect(EVENT_MATRIX[type]).toBeDefined();
    for (const type of HOOK_TELEMETRY_TYPES) expect(EVENT_MATRIX[type]).toMatchObject({ runScoped: false, policy: "local", roles: ["lead", "adhoc"] });
    expect(HOOK_TELEMETRY_TYPES).toEqual(["tool-used", "subagent-started", "subagent-stopped"]);
  });
});

describe("mission-status-updated / mission-finished payloads (spec §9 — real fields, replacing MOA-487's reserved stub)", () => {
  const STATUS_PAYLOAD = { missionId: 1, missionName: "Ship it", missionStatusLine: "phase 1 merged", milestonesDone: 1, milestonesTotal: 3 };
  const FINISH_PAYLOAD = { missionId: 1, missionName: "Ship it", outcome: "done" };

  it("parseIngress accepts a well-formed payload for each type, wrapper emitter, role lead, no run_id", () => {
    expect(parseIngress({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: STATUS_PAYLOAD }))
      .toMatchObject({ ok: true, event: { type: "mission-status-updated", payload: STATUS_PAYLOAD } });
    expect(parseIngress({ project: "_mission", role: "lead", type: "mission-finished", source: "deterministic", emitter: "wrapper", payload: FINISH_PAYLOAD }))
      .toMatchObject({ ok: true, event: { type: "mission-finished", payload: FINISH_PAYLOAD } });
  });

  it("mission-status-updated accepts an empty missionStatusLine ('' — spec §6 Decision 1) but rejects one over the bound", () => {
    expect(parseIngress({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, missionStatusLine: "" } }).ok).toBe(true);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, missionStatusLine: "x".repeat(281) } }, /missionStatusLine/);
  });

  it("mission-status-updated reuses validMissionText for missionStatusLine — rejects a control character, still allows '' (cold review F4)", () => {
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, missionStatusLine: "a\nb" } }, /missionStatusLine/);
    expect(parseIngress({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, missionStatusLine: "" } }).ok).toBe(true);
  });

  it("mission-status-updated rejects milestonesDone > milestonesTotal, a non-integer, or a negative count", () => {
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, milestonesDone: 4, milestonesTotal: 3 } }, /milestonesDone/);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, milestonesTotal: -1 } }, /milestonesTotal/);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, milestonesDone: 1.5 } }, /milestonesDone/);
  });

  it("mission-status-updated rejects milestonesTotal above the shared cap (12) but allows exactly 12 (cold review F3, spec §6a)", () => {
    expect(parseIngress({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, milestonesDone: 12, milestonesTotal: 12 } }).ok).toBe(true);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, milestonesDone: 13, milestonesTotal: 13 } }, /milestonesTotal/);
  });

  it("mission-finished rejects an outcome outside done|cancelled", () => {
    expectInvalid({ project: "_mission", role: "lead", type: "mission-finished", source: "deterministic", emitter: "wrapper", payload: { ...FINISH_PAYLOAD, outcome: "success" } }, /outcome not allowed/);
  });

  it("mission-status-updated and mission-finished reuse validMissionText for missionName — rejects leading/trailing whitespace and a control character (cold review F1)", () => {
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, missionName: " Ship it" } }, /missionName/);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, missionName: "Ship it " } }, /missionName/);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-status-updated", source: "deterministic", emitter: "wrapper", payload: { ...STATUS_PAYLOAD, missionName: "Ship\nit" } }, /missionName/);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-finished", source: "deterministic", emitter: "wrapper", payload: { ...FINISH_PAYLOAD, missionName: " Ship it" } }, /missionName/);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-finished", source: "deterministic", emitter: "wrapper", payload: { ...FINISH_PAYLOAD, missionName: "Ship it " } }, /missionName/);
    expectInvalid({ project: "_mission", role: "lead", type: "mission-finished", source: "deterministic", emitter: "wrapper", payload: { ...FINISH_PAYLOAD, missionName: "Ship\nit" } }, /missionName/);
  });

  it("rejects an unknown payload key and a run_id on either type (not run-scoped)", () => {
    for (const [type, payload] of [["mission-status-updated", STATUS_PAYLOAD], ["mission-finished", FINISH_PAYLOAD]] as const) {
      expectInvalid({ project: "_mission", role: "lead", type, source: "deterministic", emitter: "wrapper", payload: { ...payload, extra: 1 } }, /unknown key extra/);
      expectInvalid({ project: "_mission", role: "lead", type, source: "deterministic", emitter: "wrapper", run_id: "r1", payload }, /run_id only for run events/);
    }
  });

  it("renderMessage still returns a non-empty one-line message for both (MOA-487's own placeholder copy, untouched by this task)", () => {
    expect(renderMessage({ type: "mission-status-updated", project: "_mission", role: "lead", payload: STATUS_PAYLOAD }).length).toBeGreaterThan(0);
    expect(renderMessage({ type: "mission-finished", project: "_mission", role: "lead", payload: FINISH_PAYLOAD }).length).toBeGreaterThan(0);
  });
});

describe("pr-opened (spec MOA-465 Ledger & card)", () => {
  it("validates a well-formed feature-PR event", () => {
    expect(parseIngress(prOpenedEvent()).ok).toBe(true);
  });

  it("validates a well-formed release-PR event carrying snapshot_sha", () => {
    const ev = prOpenedEvent({
      payload: { ...prOpenedEvent().payload, branch: "release/2026-09-27-staging-promotion",
                 base: "main", kind: "release", snapshot_sha: "b".repeat(40) },
    });
    expect(parseIngress(ev).ok).toBe(true);
  });

  it("snapshot_sha is optional even when kind is release (this plan's own resolution — see Global Constraints)", () => {
    const ev = prOpenedEvent({ payload: { ...prOpenedEvent().payload, kind: "release" } });
    expect(parseIngress(ev).ok).toBe(true);
  });

  it("rejects a malformed snapshot_sha when present", () => {
    const ev = prOpenedEvent({ payload: { ...prOpenedEvent().payload, snapshot_sha: "abc" } });
    expect(parseIngress(ev).ok).toBe(false);
  });

  it("rejects a kind outside feature/release", () => {
    const ev = prOpenedEvent({ payload: { ...prOpenedEvent().payload, kind: "hotfix" } });
    expect(parseIngress(ev).ok).toBe(false);
  });

  it("rejects a sha that is not full 40-hex", () => {
    const ev = prOpenedEvent({ payload: { ...prOpenedEvent().payload, sha: "abc123" } });
    expect(parseIngress(ev).ok).toBe(false);
  });

  it("rejects a pr_url that is not a github.com pull URL", () => {
    const ev = prOpenedEvent({ payload: { ...prOpenedEvent().payload, pr_url: "https://gitlab.com/acme/x/pull/7" } });
    expect(parseIngress(ev).ok).toBe(false);
  });

  it("rejects a non-positive pr_number", () => {
    const ev = prOpenedEvent({ payload: { ...prOpenedEvent().payload, pr_number: 0 } });
    expect(parseIngress(ev).ok).toBe(false);
  });

  it("rejects an unknown payload key and a run_id (not run-scoped)", () => {
    const ev1 = prOpenedEvent({ payload: { ...prOpenedEvent().payload, extra: 1 } });
    expect(parseIngress(ev1).ok).toBe(false);
    expect(parseIngress(prOpenedEvent({ run_id: "abc123abc123" })).ok).toBe(false);
  });

  it("renders a deterministic non-empty message naming the PR", () => {
    const msg = renderMessage({ type: "pr-opened", project: "jax-os", role: "lead", payload: prOpenedEvent().payload });
    expect(msg).toContain("#7");
    expect(msg.length).toBeGreaterThan(0);
  });
});
