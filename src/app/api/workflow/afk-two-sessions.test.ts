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
// The board render path is client code: give it the same stand-ins the component tests use.
vi.mock("next-intl", () => ({ useTranslations: () => (key: string) => key, useLocale: () => "en-US" }));
vi.mock("@/components/RelativeTime", () => ({ RelativeTime: () => null }));
vi.mock("@tanstack/react-query", () => ({ useQueryClient: () => ({}), useQuery: () => ({ data: undefined }) }));
// The names reader is wrapped in a spy so "session-less rows never read the registry" is
// assertable; every other export (and the real reader) stays as-is.
vi.mock("../../../server/collectors/session-names", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../server/collectors/session-names")>();
  return { ...actual, readSessionNames: vi.fn(actual.readSessionNames) };
});

import { openDb } from "../../../server/db";
import {
  claimAnswer, claimFreeformAnswer, failAnswerFinalization, finishAnswer, getAfkEnabled, getForwardTypes, getHubData, listPending,
  setAfk as setAfkFlag, setForwardTypes, type HubEnvelopeData,
} from "../../../server/db/workflows";
import { toForwardBody } from "../../../server/collectors/workflow-events";
import { injectAnswers } from "../../../server/collectors/workflow-tmux";
import { signBody } from "../../../server/collectors/workflow-webhook";
import { readSessionNames } from "../../../server/collectors/session-names";
import { POST as postEventRoute, GET as pollEventsRoute } from "./events/route";
import { POST as ackPost } from "./events/ack/route";
import { handleAfkPost } from "./afk/handler";
import { handleAnswerPost } from "./answer/handler";
import { handleHubGet } from "../mission/hub/handler";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { buildMission } from "@/lib/mission";
import { ProjectsColumn } from "@/components/mission/ProjectsColumn";

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
const eventCount = () => (testDb.prepare("SELECT COUNT(*) AS n FROM workflow_events").get() as { n: number }).n;
async function ack(ids: number[]) {
  const out = await (await ackPost(json("http://localhost/api/workflow/events/ack", { ids }))).json();
  expect(out.ok).toBe(true);
  return out.data as { acked: number };
}

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

describe("AFK on, two sessions in one project", () => {
  it("gives each session's question its own notification and answer destination", async () => {
    const qa = await post(ask(A, "%1", "tu_a", "Ship feat/a behind a flag?"));
    const qb = await post(ask(B, "%2", "tu_b", "Drop the legacy endpoint?"));
    expect([[qa.delivery, qa.forwarded], [qb.delivery, qb.forwarded]]).toEqual([["delivered", true], ["delivered", true]]);
    expect(sent).toHaveLength(2);
    const [na, nb] = sent;
    expect(na.body).toMatchObject({ event_id: qa.id, project: P, type: "question" });
    expect(nb.body).toMatchObject({ event_id: qb.id, project: P, type: "question" });
    expect(String(na.body.message)).toContain(`Workflow event #${qa.id}`);
    expect(String(na.body.message)).toContain("Ship feat/a behind a flag?");
    expect(String(na.body.message)).not.toContain("Drop the legacy endpoint?");
    for (const n of sent) expect(n.signature).toBe(signBody(n.raw, "afk-test-secret"));
    expect(rowOf(qa.id)).toMatchObject({ pane: "%1", tmux_incarnation: INC, harness_session: A, delivery: "delivered" });
    expect(rowOf(qb.id)).toMatchObject({ pane: "%2", tmux_incarnation: INC, harness_session: B, delivery: "delivered" });
    expect(await pollPending()).toEqual([]);
  });

  it("answering one question types only into its own pane; the other stays pending and unchanged", async () => {
    const qa = await post(ask(A, "%1", "tu_a", "Ship feat/a behind a flag?"));
    const qb = await post(ask(B, "%2", "tu_b", "Drop the legacy endpoint?"));
    const before = rowOf(qb.id);
    expect(await answer(qa.id, "1: 1")).toMatchObject({ ok: true, data: { question_event_id: qa.id } });
    expect(keys.length).toBeGreaterThan(0);
    expect(keys.every((k) => k[0] === "send-keys" && k[2] === "%1")).toBe(true);
    expect(rowOf(qb.id)).toEqual(before);
    expect(sent).toHaveLength(2); // the answer itself notifies nobody
    const hub = await pollHub(["%1", "%2"]);
    const panes = Object.fromEntries(hub.byProject[P].panes.map((p) => [p.pane, p]));
    expect(panes["%1"].pendingQuestion).toBeNull();
    expect(panes["%2"].pendingQuestion?.eventId).toBe(qb.id);
    expect(panes["%2"].questionAnswerable).toBe(true);
    keys = [];
    expect(await answer(qb.id, "1: 2")).toMatchObject({ ok: true });
    expect(keys.every((k) => k[2] === "%2")).toBe(true);
  });

  it("a partial ack delivers only the acked event; the other keeps its identity", async () => {
    forwardFailures = 2;
    const qa = await post(ask(A, "%1", "tu_a", "A?"));
    const qb = await post(ask(B, "%2", "tu_b", "B?"));
    expect([qa.delivery, qb.delivery]).toEqual(["pending", "pending"]);
    expect(rowOf(qa.id).forwarded_at).toBeNull();
    const beforeB = rowOf(qb.id);
    expect(await ack([qa.id])).toEqual({ acked: 1 });
    expect(await pollPending()).toEqual([expect.objectContaining({ event_id: qb.id })]);
    expect(rowOf(qa.id)).toMatchObject({ delivery: "delivered" });
    expect(rowOf(qa.id).forwarded_at).toEqual(expect.any(String));
    expect(rowOf(qb.id)).toMatchObject({ delivery: "pending" });
    expect(rowOf(qb.id).forwarded_at).toBe(beforeB.forwarded_at);
    expect((await pollPending()).map((p) => p.event_id)).toEqual([qb.id]);
  });

  it("once B's conversation is replaced, its old question can no longer reach the pane", async () => {
    const qb = await post(ask(B, "%2", "tu_b", "Drop the legacy endpoint?"));
    await post(hook(B2, "%2", "turn-started", "claude-userprompt")); // /clear or a restarted agent in %2
    expect(await answer(qb.id, "1: 1")).toEqual({ ok: false, error: "invalid-target" });
    expect(keys).toEqual([]);
    expect(testDb.prepare("SELECT COUNT(*) AS n FROM workflow_answer_claims").get()).toEqual({ n: 0 });
    const hub = await pollHub(["%1", "%2"]);
    expect(hub.byProject[P].panes.find((p) => p.pane === "%2")?.questionAnswerable).toBe(false);
    expect(sent).toHaveLength(1); // the question's own notification; the new turn-start sent nothing
  });

  it("once B's pane is closed, its old question can't be delivered anywhere", async () => {
    const qb = await post(ask(B, "%2", "tu_b", "Drop the legacy endpoint?"));
    expect(await answer(qb.id, "1: 1", ["%1"])).toEqual({ ok: false, error: "pane target is not live" });
    expect(keys).toEqual([]);
  });

  it("polling the hub and the events list many times creates no notifications or rows", async () => {
    const qa = await post(ask(A, "%1", "tu_a", "Ship feat/a behind a flag?"));
    await post(ask(B, "%2", "tu_b", "Drop the legacy endpoint?"));
    const events = eventCount();
    const before = rowOf(qa.id);
    for (let i = 0; i < 5; i++) {
      expect((await pollHub(["%1", "%2"])).byProject[P].panes.filter((p) => p.pendingQuestion)).toHaveLength(2);
      expect(await pollPending()).toEqual([]);
      await pollEventsRoute(new Request("http://localhost/api/workflow/events?pending=1"));
    }
    expect(sent).toHaveLength(2);
    expect(eventCount()).toBe(events);
    expect(rowOf(qa.id)).toMatchObject({ delivery: before.delivery, forwarded_at: before.forwarded_at });
  });

  it("a failed push stays pending and the fallback poll carries the same message and event", async () => {
    forwardFailures = 1;
    const qa = await post(ask(A, "%1", "tu_a", "Ship feat/a behind a flag?"));
    expect(qa.delivery).toBe("pending");
    const pending = await pollPending();
    expect(pending).toHaveLength(1);
    expect(pending[0].event_id).toBe(qa.id);
    const expected = toForwardBody(listPending(testDb, 50).find((r) => r.id === qa.id)!);
    expect(pending[0].message).toBe(expected.message);
    await ack([qa.id]);
    expect(rowOf(qa.id)).toMatchObject({ delivery: "delivered", forwarded_at: expect.any(String) });
    expect(await pollPending()).toEqual([]);
    expect(eventCount()).toBe(3); // two turn-starts + the one question, no duplicate
  });

  it("AFK off keeps questions local, and turning it back on replays nothing", async () => {
    expect(await setAfk({ enabled: false })).toMatchObject({ enabled: false });
    const qa = await post(ask(A, "%1", "tu_a", "Ship feat/a behind a flag?"));
    const qb = await post(ask(B, "%2", "tu_b", "Drop the legacy endpoint?"));
    expect([qa.delivery, qb.delivery]).toEqual(["local", "local"]);
    expect([qa.forwarded, qb.forwarded]).toEqual([false, false]);
    expect(sent).toEqual([]);
    expect(await pollPending()).toEqual([]);
    await setAfk({ enabled: true });
    expect(await pollPending()).toEqual([]);
    expect(sent).toEqual([]);
  });

  it("AFK on still only sends opted-in types: a turn stop stays local", async () => {
    const stop = await post(hook(A, "%1", "turn-stopped", "claude-stop", { capsule_status: "done", capsule_rule: "tag" }));
    expect(stop.delivery).toBe("local");
    expect(sent).toEqual([]);
  });

  it("rendering group cards and polling generate no additional notifications or event rows", async () => {
    const qa = await post(ask(A, "%1", "tu_a", "A?"));
    const qb = await post(ask(B, "%2", "tu_b", "B?"));
    expect([qa.delivery, qb.delivery]).toEqual(["delivered", "delivered"]);
    const rows = eventCount();
    const transports = fetchSpy.mock.calls.length;

    const hub = await pollHub(["%1", "%2"]);
    const mission = buildMission(
      {
        ok: true,
        data: {
          projects: [{ dir: P, name: "AFK project", stage: "build", gate: null, legacyGateField: false, builder: "opencode", branch: "main",
            updated: "2026-09-20T09:00:00-03:00", now: "", residuals: [], statusMtime: "2026-09-20T12:00:00.000Z", archived: false }],
          skipped: 0, reposRoot: "/r",
        },
      },
      { ok: true, data: hub },
      undefined,
      Date.now(),
    );
    expect(mission.model.cards[0].group).toBeDefined();
    const html = renderToStaticMarkup(createElement(ProjectsColumn, {
      mission, reposRoot: "/r", counts: null, filter: null, expandedDir: null, onToggleExpand: () => {},
    }));
    expect(html).toContain('data-testid="group-header"');

    for (let i = 0; i < 3; i++) await pollHub(["%1", "%2"]);
    await pollPending();
    expect(eventCount()).toBe(rows);
    expect(fetchSpy.mock.calls.length).toBe(transports);
  });

  it("keeps a closed session's runs on the unlinked card until nothing is in flight", async () => {
    await post(runStarted(RUN1, B));
    await post(runStarted(RUN2, B));
    // B's pane closes while both builds keep going; A is still open.
    let hub = await pollHub(["%1"]);
    let split = hub.byProject[P].split!;
    expect(split.seats.map((s) => s.sessionId)).toEqual([A]);
    expect(split.unlinked.activeRuns.map((r) => r.runId).sort()).toEqual([RUN1, RUN2]);
    expect(split.seats[0].activity.activeRuns).toEqual([]);
    // Default forward types: a finished run stays local — no notification.
    expect((await post(runFinished(RUN1))).delivery).toBe("local");
    expect(sent).toEqual([]);
    // The owner opts in to run results: the next one is sent exactly once.
    await setAfk({ forwardTypes: ["question", "attention-needed", "mission-finished", "run-finished"] });
    const done = await post(runFinished(RUN2));
    expect(done).toMatchObject({ delivery: "delivered", forwarded: true });
    expect(sent).toHaveLength(1);
    expect(sent[0].body).toMatchObject({ event_id: done.id, type: "run-finished", run_id: RUN2, project: P });
    // A run result is not a question: an "answer" to it is refused, and no input goes anywhere.
    expect(await answer(done.id, "1: 1")).toEqual({ ok: false, error: "not-answerable" });
    expect(keys).toEqual([]);
    // Nothing in flight any more: the extra card is gone.
    hub = await pollHub(["%1"]);
    split = hub.byProject[P].split!;
    expect(split.unlinked.activeRuns).toEqual([]);
    expect(await pollPending()).toEqual([]);
    expect(sent).toHaveLength(1);
  });
});
