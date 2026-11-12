// AFK with TWO sessions in the same project, end to end through the real routes:
//   AFK toggle (/api/workflow/afk) -> hook ingress (/api/workflow/events, real parseIngress +
//   insertEvent delivery ladder) -> the real webhook forwarder (only `fetch` is stubbed, so every
//   notification's signed body is captured) -> the answer route (real claim + real injector, only
//   tmux is stubbed) -> the hub route polled like the dashboard.
//
// Part 1 (this task): session display names in the notification text, on BOTH the push path and
// the fallback poll.
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const ENV = vi.hoisted(() => {
  const { mkdtempSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const home = mkdtempSync(join(tmpdir(), "jaxos-afk-home-"));
  writeFileSync(join(home, "settings.json"), JSON.stringify({ reposRoot: tmpdir() }), "utf8");
  process.env.JAXOS_HOME = home;
  const saved = { url: process.env.NOTIFICATION_WEBHOOK_URL, secret: process.env.NOTIFICATION_WEBHOOK_SECRET };
  process.env.NOTIFICATION_WEBHOOK_URL = "https://notify.example.test/hook";
  process.env.NOTIFICATION_WEBHOOK_SECRET = "afk-test-secret";
  return { saved };
});

// Every route under test calls the real singleton getDb(); swap ONLY that export for an in-memory DB.
let testDb: Database.Database;
vi.mock("../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});
// The webhook integration is on (the owner enabled it); everything else is the real settings reader.
vi.mock("../../../server/settings", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../server/settings")>();
  return {
    ...actual,
    readGeneralSettings: (...args: Parameters<typeof actual.readGeneralSettings>) => {
      const r = actual.readGeneralSettings(...args);
      return r.ok ? { ...r, data: { ...r.data, integrations: { ...r.data.integrations, webhook: true } } } : r;
    },
  };
});
// The names reader is wrapped in a spy so "session-less rows never read the registry" is
// assertable; every other export (and the real reader) stays as-is.
vi.mock("../../../server/collectors/session-names", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../server/collectors/session-names")>();
  return { ...actual, readSessionNames: vi.fn(actual.readSessionNames) };
});

import { openDb } from "../../../server/db";
import {
  claimAnswer, claimFreeformAnswer, failAnswerFinalization, finishAnswer, getAfkEnabled, getForwardTypes, getHubData,
  setAfk as setAfkFlag, setForwardTypes, type HubEnvelopeData,
} from "../../../server/db/workflows";
import { injectAnswers } from "../../../server/collectors/workflow-tmux";
import { signBody } from "../../../server/collectors/workflow-webhook";
import { readSessionNames } from "../../../server/collectors/session-names";
import { POST as postEventRoute, GET as pollEventsRoute } from "./events/route";
import { handleAfkPost } from "./afk/handler";
import { handleAnswerPost } from "./answer/handler";
import { handleHubGet } from "../mission/hub/handler";

const P = "afkproj";
const INC = "4242:1700000000";
const u = (c: string) => `${c.repeat(8)}-0000-4000-8000-${c.repeat(12)}`;
const A = u("a"), B = u("b"), B2 = u("c");

// ---- the notification webhook: real forwarder, fetch captured ----------------------------------
type Sent = { body: Record<string, unknown>; raw: string; signature: string | null };
let sent: Sent[] = [];
let forwardFailures = 0; // the next N forwards answer 500 (the event stays pending)
const fetchSpy = vi.fn(async (_url: string, init: RequestInit) => {
  if (forwardFailures > 0) {
    forwardFailures -= 1;
    return new Response(null, { status: 500 });
  }
  const raw = String(init.body);
  sent.push({ body: JSON.parse(raw) as Record<string, unknown>, raw, signature: new Headers(init.headers).get("X-Hub-Signature-256") });
  return new Response(null, { status: 200 });
});
vi.stubGlobal("fetch", fetchSpy);
afterAll(() => {
  vi.unstubAllGlobals();
  if (ENV.saved.url === undefined) delete process.env.NOTIFICATION_WEBHOOK_URL; else process.env.NOTIFICATION_WEBHOOK_URL = ENV.saved.url;
  if (ENV.saved.secret === undefined) delete process.env.NOTIFICATION_WEBHOOK_SECRET; else process.env.NOTIFICATION_WEBHOOK_SECRET = ENV.saved.secret;
});

// ---- routes ----------------------------------------------------------------------------------
const json = (url: string, body: unknown) => new Request(url, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });

async function setAfk(body: { enabled?: boolean; forwardTypes?: string[] }) {
  const res = await handleAfkPost(json("http://localhost/api/workflow/afk", body), {
    getDb: () => testDb, getAfkEnabled, setAfk: setAfkFlag, getForwardTypes, setForwardTypes,
  });
  const out = await res.json();
  expect(out.ok).toBe(true);
  return out.data as { enabled: boolean; forwardTypes: string[] };
}

// What the hook (scripts/jaxflow_hook.py) POSTs for a Claude session in a tmux pane.
function hook(session: string, pane: string, type: string, emitter: string, payload: Record<string, unknown> = {}) {
  return { project: P, role: "lead", type, source: "deterministic", emitter, payload, pane, tmux_incarnation: INC, harness_session: session };
}
const ask = (session: string, pane: string, tool: string, text: string) =>
  hook(session, pane, "question", "claude-pretool", { tool_use_id: tool, questions: [{ question: text, options: [{ label: "yes" }, { label: "no" }] }] });
// What `jaxflow build` POSTs for its runs (run ids are 12 hex chars, like uuid4().hex[:12]).
const RUN1 = "0000000000a1", RUN2 = "0000000000a2";
const runStarted = (runId: string, session: string) => ({
  project: P, run_id: runId, role: "builder", type: "run-started", source: "deterministic", emitter: "wrapper",
  payload: { phase: "P1", runtime: "opencode-builder", kind: "build", target: `feat/${runId}`, caller: "claude", caller_session: session,
    model: "m", effort: "high", session: `jax-${runId}`, repo: "/r/afkproj", verify: "pnpm test", requested_profile: "default", root_build_run_id: runId },
});
const runFinished = (runId: string, reportPath = `/r/afkproj/.local/reports/${runId}.md`) => ({
  project: P, run_id: runId, role: "builder", type: "run-finished", source: "deterministic", emitter: "wrapper",
  payload: { phase: "P1", exit_code: 0, contract_status: "ok", report_path: reportPath, summary: "done", head_sha: "a".repeat(40), result: "success" },
});

async function post(body: unknown): Promise<{ id: number; delivery: string; forwarded: boolean }> {
  const out = await (await postEventRoute(json("http://localhost/api/workflow/events", body))).json();
  if (!out.ok) throw new Error(`ingress refused the simulated event: ${out.error}`);
  return { id: out.data.id, delivery: out.data.delivery, forwarded: out.forwarded };
}
async function pollPending(): Promise<Record<string, unknown>[]> {
  const out = await (await pollEventsRoute(new Request("http://localhost/api/workflow/events?pending=1"))).json();
  expect(out.ok).toBe(true);
  return out.data;
}

// tmux stand-in for the answer route: the given panes are live; every send-keys is recorded.
let keys: string[][] = [];
function answerDeps(livePanes: string[]) {
  const run = vi.fn(async (_f: string, args: string[]) => {
    if (args[0] === "list-panes") return livePanes.map((p) => `${p} 4242 1700000000 claude`).join("\n") + "\n";
    keys.push(args);
    return "";
  });
  return {
    getDb: () => testDb,
    readGeneralSettings: (() => ({ ok: true, data: { ownerName: "" } })) as never,
    loadEventType: (d: Database.Database, id: number) => (d.prepare("SELECT type FROM workflow_events WHERE id = ?").get(id) as { type: string } | undefined)?.type ?? null,
    claimAnswer, claimFreeformAnswer,
    injectAnswers: (pane: string, shapes: Parameters<typeof injectAnswers>[1], answers: Parameters<typeof injectAnswers>[2], inc: string) => injectAnswers(pane, shapes, answers, inc, run, async () => {}),
    codexQueueMessage: vi.fn(async () => {}), isAnswerWatcherAlive: vi.fn(() => false), writeAnswerLine: vi.fn(),
    finishAnswer, failAnswerFinalization,
  };
}
async function answer(eventId: number, reply: string, livePanes = ["%1", "%2"]) {
  return (await handleAnswerPost(json("http://localhost/api/workflow/answer", { event_id: eventId, reply }), answerDeps(livePanes))).json();
}

// The dashboard's hub poll, through the real route.
async function pollHub(livePanes: string[]): Promise<HubEnvelopeData> {
  const res = await handleHubGet({
    getDb: () => testDb,
    getProjects: () => ({ projects: [{ dir: P, name: "AFK project", stage: "build", gate: null, legacyGateField: false, builder: "opencode", branch: "main",
      updated: "2026-09-20T09:00:00-03:00", now: "", residuals: [], statusMtime: "2026-09-20T12:00:00.000Z", archived: false }], skipped: 0, reposRoot: "/r" }),
    listLivePanes: async () => ({ paneIds: new Set(livePanes), incarnation: INC, paneCommands: new Map() }),
    collectCodexSnapshot: async () => ({ ok: false, ts: "", error: "n/a" }),
    isCodexCliPresent: () => false, codexEnabled: () => false, getHubData,
  });
  const out = await res.json();
  expect(out.ok).toBe(true);
  return out.data as HubEnvelopeData;
}

const rowOf = (id: number) => testDb.prepare("SELECT * FROM workflow_events WHERE id = ?").get(id) as Record<string, unknown>;

// A Claude registry entry: `~/.claude/sessions/<pid>.json`.
function seedRegistry(home: string, sessionId: string, name: string): void {
  const dir = join(home, ".claude", "sessions");
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "1.json"), JSON.stringify({ sessionId, name }), "utf8");
}

let home: string;

beforeEach(async () => {
  testDb = openDb(":memory:");
  sent = [];
  keys = [];
  forwardFailures = 0;
  fetchSpy.mockClear();
  vi.mocked(readSessionNames).mockClear();
  home = mkdtempSync(join(tmpdir(), "jaxos-afk-names-"));
  vi.stubEnv("HOME", home);
  // The owner turns AFK on (default forward types) and two sessions work in the same project.
  expect(await setAfk({ enabled: true })).toMatchObject({ enabled: true, forwardTypes: ["question", "attention-needed", "mission-finished"] });
  await post(hook(A, "%1", "turn-started", "claude-userprompt"));
  await post(hook(B, "%2", "turn-started", "claude-userprompt"));
});

afterEach(() => {
  vi.unstubAllEnvs();
  rmSync(home, { recursive: true, force: true });
});

describe("session names in notification text", () => {
  it("names the asking session in the pushed message and leaves the unnamed one unchanged", async () => {
    seedRegistry(home, A, "multi-sessao");
    const qa = await post(ask(A, "%1", "tu_a", "Ship feat/a behind a flag?"));
    const qb = await post(ask(B, "%2", "tu_b", "Drop the legacy endpoint?"));
    expect([qa.delivery, qb.delivery]).toEqual(["delivered", "delivered"]);
    expect(sent).toHaveLength(2);
    const [na, nb] = sent;
    expect(String(na.body.message)).toContain(`Workflow event #${qa.id}`);
    expect(String(na.body.message)).toContain("Project afkproj · multi-sessao:");
    expect(String(nb.body.message)).toContain(`Workflow event #${qb.id}`);
    expect(String(nb.body.message)).toContain("Project afkproj:");
    expect(String(nb.body.message)).not.toContain(" · ");
    expect(na.body.payload).toEqual({ questions: [{ question: "Ship feat/a behind a flag?", options: [{ label: "yes" }, { label: "no" }] }] });
    expect(nb.body.payload).toEqual({ questions: [{ question: "Drop the legacy endpoint?", options: [{ label: "yes" }, { label: "no" }] }] });
    for (const n of sent) expect(n.signature).toBe(signBody(n.raw, "afk-test-secret"));
  });

  it("does not name a run-finished row: no session, no lookup, identical text", async () => {
    await setAfk({ forwardTypes: ["question", "attention-needed", "mission-finished", "run-finished"] });
    const first = await post(runFinished(RUN1, "/r/afkproj/.local/reports/r.md"));
    seedRegistry(home, B, "multi-sessao");
    vi.mocked(readSessionNames).mockClear();
    const second = await post(runFinished(RUN2, "/r/afkproj/.local/reports/r.md"));
    expect([first.delivery, second.delivery]).toEqual(["delivered", "delivered"]);
    expect(sent).toHaveLength(2);
    expect(sent[0].body.message).toBe(sent[1].body.message);
    expect(String(sent[1].body.message)).not.toContain("multi-sessao");
    expect(String(sent[1].body.message)).not.toContain(" · ");
    expect(vi.mocked(readSessionNames)).not.toHaveBeenCalled();
  });

  it("never reads the registry for session-less rows — on the push path or the fallback poll", async () => {
    await setAfk({ forwardTypes: ["question", "attention-needed", "mission-finished", "run-finished"] });
    vi.mocked(readSessionNames).mockClear();
    forwardFailures = 1;
    const done = await post(runFinished(RUN1));
    expect(done.delivery).toBe("pending");
    expect(vi.mocked(readSessionNames)).not.toHaveBeenCalled();
    const pending = await pollPending();
    expect(pending.map((p) => p.event_id)).toEqual([done.id]);
    expect(vi.mocked(readSessionNames)).not.toHaveBeenCalled();
    // A row that DOES carry a session reads the registry once.
    await post(ask(A, "%1", "tu_a", "Ship?"));
    expect(vi.mocked(readSessionNames)).toHaveBeenCalled();
  });

  it("the fallback poll renders the same name the push would have carried", async () => {
    seedRegistry(home, A, "multi-sessao");
    forwardFailures = 1;
    const qa = await post(ask(A, "%1", "tu_a", "Ship feat/a behind a flag?"));
    expect(qa.delivery).toBe("pending");
    const pending = await pollPending();
    expect(pending.map((p) => p.event_id)).toEqual([qa.id]);
    expect(String(pending[0].message)).toContain(`Workflow event #${qa.id}`);
    expect(String(pending[0].message)).toContain("Project afkproj · multi-sessao:");
  });

  it("a missing or unreadable registry is silent: no name, ok response, no warning", async () => {
    seedRegistry(home, A, "multi-sessao");
    rmSync(join(home, ".claude"), { recursive: true, force: true });
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    try {
      const qa = await post(ask(A, "%1", "tu_a", "Ship?"));
      expect(qa.delivery).toBe("delivered");
      expect(String(sent[0].body.message)).toContain("Project afkproj:");
      expect(String(sent[0].body.message)).not.toContain(" · ");
      expect(warn).not.toHaveBeenCalled();
      expect(error).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
      error.mockRestore();
    }
  });

  it("keeps delivery identity: a named event is delivered exactly like an unnamed twin", async () => {
    seedRegistry(home, A, "multi-sessao");
    const qa = await post(ask(A, "%1", "tu_a", "A?"));
    const qb = await post(ask(B, "%2", "tu_b", "B?"));
    const rowA = rowOf(qa.id), rowB = rowOf(qb.id);
    expect(rowA).toMatchObject({ delivery: "delivered", harness_session: A });
    expect(rowB).toMatchObject({ delivery: "delivered", harness_session: B });
    expect(rowA.forwarded_at).toEqual(expect.any(String));
    expect(rowB.forwarded_at).toEqual(expect.any(String));
    expect([sent[0].body.event_id, sent[1].body.event_id]).toEqual([qa.id, qb.id]);
    expect(await pollPending()).toEqual([]);
  });
});
