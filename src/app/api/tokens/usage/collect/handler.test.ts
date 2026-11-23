import { describe, expect, it, vi } from "vitest";
import { openDb } from "../../../../../server/db";
import { setSourceStatus } from "../../../../../server/db/native-usage";
import { handleUsageCollect, type CollectDeps } from "./handler";

const NOW = Date.parse("2026-11-18T12:00:00Z") / 1000;
const SOURCES = ["claude", "codex", "opencode"] as const;

function deps(overrides: Partial<CollectDeps> = {}): CollectDeps {
  const db = openDb(":memory:");
  return {
    db: () => db,
    pass: async () => {},
    now: Date.now,
    deadlineMs: 30_000,
    ...overrides,
  };
}

function post(headers: Record<string, string> = {}): Request {
  return new Request("http://127.0.0.1/api/tokens/usage/collect", {
    method: "POST",
    headers: { host: "127.0.0.1", ...headers },
  });
}

function setAll(db: ReturnType<typeof openDb>, state: "indexing" | "available"): void {
  for (const source of SOURCES) {
    setSourceStatus(db, { source, state, lastRefresh: null, progress: null, warnings: [] }, NOW);
  }
}

describe("handleUsageCollect", () => {
  it("runs passes until every source leaves indexing and reports caughtUp", async () => {
    let calls = 0;
    const pass = vi.fn(async (db) => {
      calls += 1;
      setAll(db, calls < 3 ? "indexing" : "available");
    });
    const response = await handleUsageCollect(post(), deps({ pass }));
    const body = await response.json();
    expect(response.status).toBe(200);
    expect(pass).toHaveBeenCalledTimes(3);
    expect(body).toEqual({
      ok: true,
      data: {
        passes: 3,
        caughtUp: true,
        sources: SOURCES.map((source) => ({ source, state: "available", lastRefresh: null })),
      },
    });
  });

  it("stops at the deadline with caughtUp false, and a second call starts again", async () => {
    let clock = 0;
    const now = () => (clock += 10_000);
    const pass = vi.fn(async (db) => setAll(db, "indexing"));
    const routeDeps = deps({ pass, now });
    const first = await (await handleUsageCollect(post(), routeDeps)).json();
    expect(first.data.caughtUp).toBe(false);
    expect(first.data.passes).toBe(3);
    const second = await (await handleUsageCollect(post(), routeDeps)).json();
    expect(second.data.caughtUp).toBe(false);
    expect(second.data.passes).toBe(3);
    expect(pass).toHaveBeenCalledTimes(6);
  });

  it("stops with partial progress when a pass throws", async () => {
    const pass = vi.fn(async () => {
      throw new Error("collector down");
    });
    const body = await (await handleUsageCollect(post(), deps({ pass }))).json();
    expect(pass).toHaveBeenCalledTimes(1);
    expect(body.ok).toBe(true);
    expect(body.data.passes).toBe(0);
    expect(body.data.caughtUp).toBe(false);
  });

  it("refuses a non-loopback Host and forwarded/origin headers without running a pass", async () => {
    const pass = vi.fn(async () => {});
    const routeDeps = deps({ pass });
    const refused: Record<string, string>[] = [
      { host: "jax.tailnet.ts.net" },
      { "x-forwarded-for": "100.64.0.1" },
      { origin: "https://example.com" },
      { "sec-fetch-site": "cross-site" },
      { "x-forwarded-host": "example.com" },
    ];
    for (const headers of refused) {
      const response = await handleUsageCollect(post(headers), routeDeps);
      expect(response.status).toBe(403);
      expect(await response.json()).toEqual({ ok: false, error: "loopback only" });
    }
    expect(pass).not.toHaveBeenCalled();
  });

  it("accepts localhost as loopback and returns the safe envelope when the DB is unavailable", async () => {
    const pass = vi.fn(async () => {});
    const response = await handleUsageCollect(post({ host: "localhost:3100" }), deps({ pass }));
    expect(response.status).toBe(200);
    expect(pass).toHaveBeenCalledTimes(1);

    const broken = await (await handleUsageCollect(post(), deps({ db: () => { throw new Error("boom"); } }))).json();
    expect(broken).toEqual({ ok: false, error: "usage unavailable" });
  });
});
