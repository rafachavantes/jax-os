import { beforeEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const harness = vi.hoisted(() => ({ getDb: vi.fn() }));
vi.mock("@/server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/server/db")>();
  return { ...actual, getDb: () => harness.getDb() };
});

import { openDb } from "@/server/db";
import { GET } from "./route";

const RUN = "0123456789ab";

function req(qs: string) {
  return new Request(`http://127.0.0.1/api/executions${qs}`);
}

function seed(db: ReturnType<typeof openDb>) {
  db.prepare(
    `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
     VALUES ('2026-11-18T10:00:00.000Z', ?, 'jax-os', 'builder', 'run-started', 'deterministic', 'wrapper', ?, 'local')`,
  ).run(RUN, JSON.stringify({ kind: "build" }));
  db.prepare(
    `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
     VALUES ('2026-11-18T10:01:00.000Z', ?, 'jax-os', 'builder', 'run-finished', 'deterministic', 'wrapper', ?, 'local')`,
  ).run(RUN, JSON.stringify({ contract_status: "ok", result: "success" }));
}

describe("GET /api/executions", () => {
  beforeEach(() => {
    const db = openDb(":memory:");
    seed(db);
    harness.getDb.mockReset();
    harness.getDb.mockReturnValue(db);
  });

  it("rejects malformed filters and cursors before touching the db", async () => {
    expect(await (await GET(req("?kind=nope"))).json()).toEqual({ ok: false, error: "invalid kind" });
    expect(await (await GET(req("?offset=20"))).json()).toEqual({ ok: false, error: "unknown parameter: offset" });
    expect(await (await GET(req("?cursor=%%%"))).json()).toEqual({ ok: false, error: "invalid cursor" });
    expect(harness.getDb).not.toHaveBeenCalled();
  });

  it("serves durable rows and a next cursor through the envelope", async () => {
    const body = await (await GET(req("?period=all"))).json();
    expect(body.ok).toBe(true);
    expect(body.data.rows[0]).toMatchObject({ destination: "jax-os", runId: RUN, outcome: "success" });
    expect(body.data.nextCursor).toBeNull();
  });

  it("never awaits an inspection source: the route module imports no collector", () => {
    const src = readFileSync(fileURLToPath(new URL("./route.ts", import.meta.url)), "utf8");
    expect(src).not.toMatch(/execution-inspection|execution-worker|execution-artifacts|workflow-tmux/);
    expect(src).not.toMatch(/from "@\/server\/collectors/);
  });
});
