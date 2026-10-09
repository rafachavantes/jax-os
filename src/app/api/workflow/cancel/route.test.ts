import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";

let testDb: Database.Database;
vi.mock("../../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});

import { openDb } from "../../../../server/db";
import { JAXFLOW_PROGRAM, JAXFLOW_SCRIPT, type ChildResult, type ChildRunner } from "../../../../server/collectors/workflow-spawn";
import { handleCancelPost, type CancelRouteDeps } from "./handler";

const RUN_ID = "a1b2c3d4e5f6";
type Call = { file: string; args: string[]; opts: unknown };

function fakeRun(result: ChildResult, calls: Call[] = []): { run: ChildRunner; calls: Call[] } {
  const run: ChildRunner = async (file, args, opts) => { calls.push({ file, args, opts }); return result; };
  return { run, calls };
}
const ok = (stdout: string): ChildResult => ({ ok: true, code: null, exitCode: 0, stdoutTail: stdout, stderrTail: "", durationMs: 3 });
const refusal = (code: string): ChildResult => ({ ok: false, code, exitCode: 2, stdoutTail: "", stderrTail: `${code}\nhint: x\n`, durationMs: 3 });
const unconfirmed: ChildResult = { ok: false, code: null, exitCode: null, stdoutTail: "", stderrTail: "", durationMs: 25_000 };

function post(body: unknown, headers: Record<string, string> = {}) {
  return new Request("http://localhost/api/workflow/cancel", {
    method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body),
  });
}
type Row = { ok: number | null; error: string | null; kind: string; payload: string };
const rows = () => testDb.prepare("SELECT kind, ok, error, payload FROM mutations").all() as Row[];

describe("POST /api/workflow/cancel", () => {
  beforeEach(() => { testDb = openDb(":memory:"); });
  afterEach(() => { testDb.close(); });

  it("spawns exactly `jaxflow cancel <run_id>` — no cwd, no env — and returns the outcome line", async () => {
    const { run, calls } = fakeRun(ok("finalized by worker — success · - — -\n"));
    const res = await handleCancelPost(post({ run_id: RUN_ID }), { run });
    expect(await res.json()).toEqual({ ok: true, data: { run_id: RUN_ID, outcome: "finalized by worker — success · - — -" } });
    expect(calls).toEqual([{ file: JAXFLOW_PROGRAM, args: [JAXFLOW_SCRIPT, "cancel", RUN_ID], opts: { timeoutMs: 25_000 } }]);
    const [row] = rows();
    expect(row.kind).toBe("workflow-cancel");
    expect(row.ok).toBe(1);
    expect(JSON.parse(row.payload)).toMatchObject({ argv: ["cancel", RUN_ID], outcome: "done", exitCode: 0, stdoutTail: "finalized by worker — success · - — -\n" });
  });

  it("maps an exit-2 refusal to mutation-rejected with the code, and records the stderr tail", async () => {
    const { run } = fakeRun(refusal("already-finished"));
    const res = await handleCancelPost(post({ run_id: RUN_ID }), { run });
    expect(await res.json()).toMatchObject({ ok: false, code: "mutation-rejected", error: "already-finished" });
    expect(res.status).toBe(200);
    const [row] = rows();
    expect(row.ok).toBe(0);
    expect(JSON.parse(row.payload)).toMatchObject({ argv: ["cancel", RUN_ID], outcome: "failed", exitCode: 2, stderrTail: "already-finished\nhint: x\n" });
  });

  it("maps a timeout/spawn error to mutation-unconfirmed and still finalizes the row", async () => {
    const { run } = fakeRun(unconfirmed);
    const res = await handleCancelPost(post({ run_id: RUN_ID }), { run });
    expect(await res.json()).toMatchObject({ ok: false, code: "mutation-unconfirmed" });
    expect(JSON.parse(rows()[0].payload)).toMatchObject({ outcome: "abandoned", exitCode: null });
  });

  it("refuses before any spawn when the audit insert fails", async () => {
    testDb.exec("CREATE TRIGGER fail_ins BEFORE INSERT ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const { run, calls } = fakeRun(ok("x"));
    const res = await handleCancelPost(post({ run_id: RUN_ID }), { run });
    expect(await res.json()).toMatchObject({ ok: false, code: "audit-unavailable" });
    expect(calls).toEqual([]);
  });

  it("refuses a malformed body or run_id without spawning; blocks cross-site requests", async () => {
    const { run, calls } = fakeRun(ok("x"));
    for (const body of [{}, { run_id: "../x" }, { run_id: "A1B2C3D4E5F6" }, [], "x"]) {
      expect(await (await handleCancelPost(post(body), { run })).json()).toEqual({ ok: false, error: "invalid payload" });
    }
    expect(await (await handleCancelPost(post({ run_id: RUN_ID }, { "sec-fetch-site": "cross-site" }), { run })).json())
      .toEqual({ ok: false, error: "cross-site request blocked" });
    expect(calls).toEqual([]);
    expect(rows()).toEqual([]);
  });

  it("round-1 F1: route.ts exports a one-argument POST", async () => {
    const { POST } = await import("./route");
    expect(POST.length).toBe(1);
  });
});
