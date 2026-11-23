import { describe, expect, it, vi } from "vitest";
import { openDb } from "../../../../server/db";
import { upsertNativeUsageItem } from "../../../../server/db/native-usage";
import { parseClaudeLine } from "../../../../server/collectors/native-usage";
import { handleTokensUsage, type UsageRouteDeps } from "./handler";

const NOW = Date.parse("2026-11-18T12:00:00Z") / 1000;

function deps(overrides: Partial<UsageRouteDeps> = {}): UsageRouteDeps {
  const db = openDb(":memory:");
  const out = parseClaudeLine(JSON.stringify({
    type: "assistant", sessionId: "sess-1", timestamp: "2026-11-17T10:00:00Z",
    message: { id: "m1", model: "claude-sonnet-5-5", usage: { input_tokens: 10, output_tokens: 5 } },
  }), { child: false, now: NOW });
  if (!("item" in out)) throw new Error(out.skip);
  upsertNativeUsageItem(db, out.item);
  return { db: () => db, ...overrides };
}

function get(path: string): Request {
  return new Request(`http://localhost/api/tokens/usage${path}`);
}

describe("handleTokensUsage", () => {
  it("returns the persisted snapshot and never collects", async () => {
    const routeDeps = deps();
    expect(Object.keys(routeDeps)).toEqual(["db"]);
    const response = await handleTokensUsage(get("?range=30d&agent=Claude%20Code&model=claude-sonnet-5-5"), routeDeps);
    const body = await response.json();
    expect(response.status).toBe(200);
    expect(body.ok).toBe(true);
    expect(body.data.totalTokens).toBe(15);
    expect(body.data.period).toBe(30);
    expect(body.data.filters.agents).toEqual(["Claude Code"]);
  });

  it("returns a healthy empty snapshot, not an error, for an empty period", async () => {
    const empty = deps({ db: () => openDb(":memory:") });
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

  it("fails the safe envelope on an unrepresentable aggregate instead of rendering a wrapped total", async () => {
    const db = openDb(":memory:");
    const huge = 5_000_000_000_000_000;
    for (const itemId of ["m1", "m2"]) {
      db.prepare(
        `INSERT INTO native_usage_items (harness, item_id, session_id, provenance, agent, ts, updated_ts, day, all_input, all_output)
         VALUES ('claude', ?, 'sess-1', 'lead', 'Claude Code', ?, ?, '2026-11-17', ?, 0)`,
      ).run(itemId, NOW, NOW, huge);
    }
    const routeDeps = { db: () => db };
    const body = await (await handleTokensUsage(get("?range=7d"), routeDeps)).json();
    expect(body).toEqual({ ok: false, error: "usage unavailable" });
    db.close();
  });

  it("turns a snapshot query failure into the safe error envelope", async () => {
    const routeDeps = deps({ db: () => { throw new Error("boom"); } });
    expect(await (await handleTokensUsage(get("?range=7d"), routeDeps)).json())
      .toEqual({ ok: false, error: "usage unavailable" });
  });

});
