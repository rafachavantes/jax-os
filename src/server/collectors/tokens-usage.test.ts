import { describe, expect, it } from "vitest";
import { openDb } from "../db";
import { upsertNativeUsageItem } from "../db/native-usage";
import { parseClaudeLine, parseOpenCodeRow } from "./native-usage";
import { getTokensUsage, isRange, RANGES } from "./tokens-usage";

const NOW = Date.parse("2026-11-18T12:00:00Z") / 1000;

function claudeItem(id: string, sessionId = "sess-1") {
  const out = parseClaudeLine(JSON.stringify({
    type: "assistant", sessionId, timestamp: "2026-11-17T10:00:00Z",
    message: { id, model: "claude-sonnet-5-5", usage: { input_tokens: 10, output_tokens: 5 } },
  }), { child: false, now: NOW });
  if (!("item" in out)) throw new Error(out.skip);
  return out.item;
}

function openCodeItem() {
  const out = parseOpenCodeRow({
    projection: "v2", session_id: "ses-oc", message_id: "m1", parent_session_id: null,
    model_id: "model-x", provider_id: "provider-x", agent: "build",
    created: 1_794_909_600_000, updated: 1_794_913_200_000,
    cost: 12.5, input: 100, output: 30, reasoning: 7, cache_read: 11, cache_write: 5,
  }, { now: NOW });
  if (!("item" in out)) throw new Error(out.skip);
  return out.item;
}

describe("tokens usage facade", () => {
  it("maps the range contract and rejects unknown ranges", () => {
    expect(RANGES).toEqual({ "7d": 7, "30d": 30, "90d": 90 });
    expect(isRange("7d")).toBe(true);
    expect(isRange("14d")).toBe(false);
  });

  it("reads the persisted native snapshot with agent and model filters", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, claudeItem("m1"));
    upsertNativeUsageItem(db, openCodeItem());
    const all = getTokensUsage(db, { range: "30d" }, NOW);
    expect(all.totalTokens).toBe(168);
    const claude = getTokensUsage(db, { range: "30d", agent: "Claude Code" }, NOW);
    expect(claude.totalTokens).toBe(15);
    expect(claude.byAgent).toEqual([expect.objectContaining({ name: "Claude Code", tokens: 15, currency: null })]);
    const model = getTokensUsage(db, { range: "30d", model: "model-x" }, NOW);
    expect(model.totalTokens).toBe(153);
    expect(model.costCoverage).toEqual({ measured: 1, total: 1 });
    expect(model.totalCost).toBeNull();
    db.close();
  });
});
