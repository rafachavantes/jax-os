import { describe, expect, it, vi } from "vitest";
import { openDb } from "../../../../server/db";
import { upsertNativeUsageItem } from "../../../../server/db/native-usage";
import { parseClaudeLine } from "../../../../server/collectors/native-usage";
import { handleTokensUsage, type UsageRouteDeps } from "./handler";

const NOW = Date.parse("2026-11-18T12:00:00Z") / 1000;

function deps(overrides: Partial<UsageRouteDeps> = {}): UsageRouteDeps & { afterCalls: (() => void | Promise<void>)[] } {
  const db = openDb(":memory:");
  const out = parseClaudeLine(JSON.stringify({
    type: "assistant", sessionId: "sess-1", timestamp: "2026-11-17T10:00:00Z",
    message: { id: "m1", model: "claude-sonnet-5-5", usage: { input_tokens: 10, output_tokens: 5 } },
  }), { child: false, now: NOW });
  if (!("item" in out)) throw new Error(out.skip);
  upsertNativeUsageItem(db, out.item);
  const afterCalls: (() => void | Promise<void>)[] = [];
  return {
    db: () => db,
    after: (callback) => afterCalls.push(callback),
    collect: async () => {},
    afterCalls,
    ...overrides,
  };
}

function get(path: string): Request {
  return new Request(`http://localhost/api/tokens/usage${path}`);
}

describe("handleTokensUsage", () => {
  it("returns the persisted snapshot and defers one collection pass until after the response", async () => {
    const collect = vi.fn(async () => {});
    const routeDeps = deps({ collect });
    const response = await handleTokensUsage(get("?range=30d&agent=Claude%20Code&model=claude-sonnet-5-5"), routeDeps);
    const body = await response.json();
    expect(response.status).toBe(200);
    expect(body.ok).toBe(true);
    expect(body.data.totalTokens).toBe(15);
    expect(body.data.period).toBe(30);
    expect(body.data.filters.agents).toEqual(["Claude Code"]);
    expect(collect).not.toHaveBeenCalled();
    expect(routeDeps.afterCalls).toHaveLength(1);
    await routeDeps.afterCalls[0]();
    expect(collect).toHaveBeenCalledTimes(1);
  });

  it("returns a healthy empty snapshot, not an error, for an empty period", async () => {
    const empty = deps({ db: () => openDb(":memory:"), collect: async () => {} });
    const response = await handleTokensUsage(get("?range=7d"), empty);
    const body = await response.json();
    expect(body.ok).toBe(true);
    expect(body.data.totalTokens).toBe(0);
    expect(body.data.sources).toHaveLength(3);
  });

  it("rejects invalid query input as a safe error", async () => {
    const routeDeps = deps();
    expect((await handleTokensUsage(get("?range=14d"), routeDeps)).status).toBe(400);
    expect(await (await handleTokensUsage(get(`?agent=${"x".repeat(300)}`), routeDeps)).json())
      .toEqual({ ok: false, error: "bad agent filter" });
    expect(await (await handleTokensUsage(get("?model=%00"), routeDeps)).json())
      .toEqual({ ok: false, error: "bad model filter" });
  });

  it("turns a snapshot query failure into the safe error envelope", async () => {
    const routeDeps = deps({ db: () => { throw new Error("boom"); } });
    expect(await (await handleTokensUsage(get("?range=7d"), routeDeps)).json())
      .toEqual({ ok: false, error: "usage unavailable" });
  });

  it("never lets a rejected collection pass escape the deferred callback", async () => {
    const routeDeps = deps({ collect: async () => { throw new Error("collector down"); } });
    const response = await handleTokensUsage(get("?range=7d"), routeDeps);
    expect((await response.json()).ok).toBe(true);
    await expect(routeDeps.afterCalls[0]()).resolves.toBeUndefined();
  });
});
