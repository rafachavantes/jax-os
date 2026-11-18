import { describe, expect, it } from "vitest";
import {
  accountingSummary,
  canonicalOpenCodeItems,
  mergeNativeItem,
  parseClaudeLine,
  parseCodexLine,
  parseOpenCodeRow,
  utcDay,
  type NativeUsageItem,
} from "./native-usage";

const NOW = Date.parse("2026-11-18T12:00:00Z") / 1000;

function claude(over: Record<string, unknown> = {}): string {
  return JSON.stringify({
    type: "assistant",
    sessionId: "sess-1",
    timestamp: "2026-11-17T10:00:00.000Z",
    message: {
      id: "msg-1",
      model: "claude-sonnet-5-5",
      usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 2, cache_creation_input_tokens: 1 },
    },
    ...over,
  });
}

function codex(over: Record<string, unknown> = {}, usageOver: Record<string, unknown> = {}, ts = "2026-11-17T10:00:00.000Z"): string {
  return JSON.stringify({
    type: "token_usage_record",
    timestamp: ts,
    payload: {
      session_id: "root-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      root_turn_id: "turn-1",
      response_id: "resp-1",
      usage: {
        input_tokens: 100,
        cached_input_tokens: 40,
        cache_write_input_tokens: 0,
        output_tokens: 20,
        reasoning_output_tokens: 5,
        total_tokens: 120,
        ...usageOver,
      },
      ...over,
    },
  });
}

function opencode(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    projection: "v2",
    session_id: "ses-1",
    message_id: "msg-1",
    parent_session_id: null,
    model_id: "model-x",
    provider_id: "provider-x",
    agent: "build",
    created: 1_760_000_000_000,
    updated: 1_760_000_100_000,
    cost: 12.5,
    input: 100,
    output: 30,
    reasoning: 7,
    cache_read: 11,
    cache_write: 5,
    ...over,
  };
}

function claudeItem(line: string, child = false): NativeUsageItem {
  const out = parseClaudeLine(line, { child, now: NOW });
  if (!("item" in out)) throw new Error(`expected item, got ${out.skip}`);
  return out.item;
}

describe("parseClaudeLine", () => {
  it("normalizes the three non-overlapping input classes and output", () => {
    const item = claudeItem(claude());
    expect(item).toMatchObject({
      harness: "claude",
      itemId: "msg-1",
      sessionId: "sess-1",
      isChild: false,
      model: "claude-sonnet-5-5",
      allInput: 13,
      allOutput: 5,
      day: "2026-11-17",
      cost: null,
    });
    expect(item.details).toEqual({ input: 10, output: 5, cacheRead: 2, cacheWrite: 1, reasoning: null });
  });

  it("excludes malformed, future, negative, nonfinite and unsafe counters", () => {
    const usage = (u: Record<string, unknown>) =>
      JSON.stringify({ type: "assistant", sessionId: "s", timestamp: "2026-11-17T10:00:00.000Z", message: { id: "m", usage: u } });
    expect(parseClaudeLine("{", { child: false, now: NOW })).toMatchObject({ skip: "malformed" });
    expect(parseClaudeLine(usage({ input_tokens: -1, output_tokens: 1 }), { child: false, now: NOW })).toMatchObject({ skip: "bad-counter" });
    expect(parseClaudeLine(usage({ input_tokens: 1.5, output_tokens: 1 }), { child: false, now: NOW })).toMatchObject({ skip: "bad-counter" });
    expect(parseClaudeLine(usage({ input_tokens: 1, output_tokens: Number.MAX_SAFE_INTEGER + 2 }), { child: false, now: NOW })).toMatchObject({ skip: "bad-counter" });
    const future = JSON.stringify({ type: "assistant", sessionId: "s", timestamp: "2027-01-01T00:00:00Z", message: { id: "m", usage: { input_tokens: 1, output_tokens: 1 } } });
    expect(parseClaudeLine(future, { child: false, now: NOW })).toMatchObject({ skip: "bad-timestamp" });
    expect(parseClaudeLine(JSON.stringify({ type: "assistant", sessionId: "s", message: { id: "m", usage: { input_tokens: 1, output_tokens: 1 } } }), { child: false, now: NOW })).toMatchObject({ skip: "bad-timestamp" });
  });

  it("requires explicit native identity and an explicit child agent id for child files", () => {
    expect(parseClaudeLine(JSON.stringify({ type: "assistant", sessionId: "s", timestamp: "2026-11-17T10:00:00Z", message: { usage: { input_tokens: 1, output_tokens: 1 } } }), { child: false, now: NOW })).toMatchObject({ skip: "bad-identity" });
    expect(parseClaudeLine(claude(), { child: true, now: NOW })).toMatchObject({ skip: "bad-child-identity" });
    const child = claudeItem(JSON.stringify({ ...JSON.parse(claude()), agentId: "agent-7" }), true);
    expect(child).toMatchObject({ isChild: true, childId: "agent-7", provenance: "child" });
  });

  it("keeps a child session out of the lead identity", () => {
    const item = claudeItem(JSON.stringify({ ...JSON.parse(claude()), agentId: "agent-7" }), true);
    expect(item.sessionId).toBe("sess-1");
    expect(item.childId).toBe("agent-7");
    expect(item.isChild).toBe(true);
  });

  it("never carries sentinel text into the normalized item", () => {
    const item = claudeItem(JSON.stringify({
      ...JSON.parse(claude()),
      message: { ...JSON.parse(claude()).message, content: "SENTINEL_SECRET_TEXT", usage: { input_tokens: 1, output_tokens: 1 } },
    }));
    expect(JSON.stringify(item)).not.toContain("SENTINEL_SECRET_TEXT");
  });
});

describe("mergeNativeItem (Claude comparator)", () => {
  it("picks the latest valid counters and the earliest occurrence for attribution, independent of order", () => {
    const early = claudeItem(claude({
      sessionId: "sess-b",
      timestamp: "2026-11-17T09:00:00Z",
      message: { id: "msg-1", model: "m", usage: { input_tokens: 10, output_tokens: 1 } },
    }));
    const late = claudeItem(claude({
      sessionId: "sess-a",
      timestamp: "2026-11-17T11:00:00Z",
      message: { id: "msg-1", model: "m", usage: { input_tokens: 10, output_tokens: 9 } },
    }));
    const forward = mergeNativeItem(early, late);
    const reverse = mergeNativeItem(late, early);
    expect(forward).toEqual(reverse);
    expect(forward).toMatchObject({ allOutput: 9, sessionId: "sess-b", ts: early.ts, day: "2026-11-17" });
  });

  it("breaks identical counters between identities by earliest timestamp then smallest tuple", () => {
    const a = claudeItem(claude({ sessionId: "sess-b", timestamp: "2026-11-17T09:00:00Z" }));
    const b = claudeItem(claude({ sessionId: "sess-a", timestamp: "2026-11-17T09:00:00Z" }));
    expect(mergeNativeItem(a, b).sessionId).toBe("sess-a");
    expect(mergeNativeItem(b, a).sessionId).toBe("sess-a");
  });

  it("breaks equal timestamps with greater output, then a stable numeric tuple", () => {
    const low = claudeItem(claude({ timestamp: "2026-11-17T10:00:00Z", message: { id: "msg-1", model: "m", usage: { input_tokens: 10, output_tokens: 1 } } }));
    const high = claudeItem(claude({ timestamp: "2026-11-17T10:00:00Z", message: { id: "msg-1", model: "m", usage: { input_tokens: 10, output_tokens: 4 } } }));
    expect(mergeNativeItem(low, high).allOutput).toBe(4);
    const moreCache = claudeItem(claude({ timestamp: "2026-11-17T10:00:00Z", message: { id: "msg-1", model: "m", usage: { input_tokens: 10, output_tokens: 4, cache_read_input_tokens: 7 } } }));
    const result = mergeNativeItem(high, moreCache);
    expect(result.allInput).toBe(17);
    expect(mergeNativeItem(moreCache, high)).toEqual(result);
  });

  it("resolves conflicting model attribution to unknown and keeps it stable across reversed ingestion", () => {
    const a = claudeItem(claude({ message: { id: "msg-1", model: "model-a", usage: { input_tokens: 10, output_tokens: 5 } } }));
    const b = claudeItem(claude({ message: { id: "msg-1", model: "model-b", usage: { input_tokens: 10, output_tokens: 5 } } }));
    expect(mergeNativeItem(a, b)).toMatchObject({ model: null, modelConflict: true });
    expect(mergeNativeItem(b, a)).toEqual(mergeNativeItem(a, b));
  });

  it("uses the earliest occurrence for the UTC day across a midnight update", () => {
    const before = claudeItem(claude({ timestamp: "2026-11-17T23:59:00Z" }));
    const after = claudeItem(claude({ timestamp: "2026-11-18T00:01:00Z", message: { id: "msg-1", model: "claude-sonnet-5-5", usage: { input_tokens: 10, output_tokens: 9 } } }));
    const merged = mergeNativeItem(before, after);
    expect(merged.day).toBe("2026-11-17");
    expect(utcDay(merged.ts)).toBe("2026-11-17");
    expect(merged.allOutput).toBe(9);
  });
});

describe("parseCodexLine", () => {
  it("uses only per-response usage and key (thread,response) dedupe across mirrored files", () => {
    const a = parseCodexLine(codex(), { fileIdentity: "rollout-a.jsonl", now: NOW });
    const b = parseCodexLine(codex(), { fileIdentity: "rollout-b.jsonl", now: NOW });
    if (!("item" in a) || !("item" in b)) throw new Error("expected codex items");
    expect(a.item).toMatchObject({
      harness: "codex", itemId: "thread-1|resp-1", sessionId: "thread-1", provenance: "root-1",
      allInput: 100, allOutput: 20, model: null, isChild: false,
    });
    expect(a.item.details).toEqual({ input: 100, output: 20, cacheRead: 40, cacheWrite: 0, reasoning: 5 });
    expect(mergeNativeItem(a.item, b.item).itemId).toBe("thread-1|resp-1");
  });

  it("never adds cumulative turn/thread objects", () => {
    const line = JSON.stringify({
      type: "token_usage_record",
      timestamp: "2026-11-17T10:00:00Z",
      payload: {
        session_id: "root-1", thread_id: "thread-1", turn_id: "t", response_id: "r",
        usage: { input_tokens: 10, output_tokens: 5, total_tokens: 15 },
        turn_token_usage: { input_tokens: 900, output_tokens: 500, total_tokens: 1400 },
        thread_token_usage: { input_tokens: 9000, output_tokens: 5000, total_tokens: 14000 },
      },
    });
    const out = parseCodexLine(line, { fileIdentity: "rollout.jsonl", now: NOW });
    if (!("item" in out)) throw new Error("expected item");
    expect(out.item.allInput + out.item.allOutput).toBe(15);
  });

  it("rejects a violated total/cached/reasoning constraint and unsupported variants", () => {
    const badTotal = parseCodexLine(codex({}, { total_tokens: 121 }), { fileIdentity: "f", now: NOW });
    expect(badTotal).toMatchObject({ skip: "bad-constraint" });
    const badCached = parseCodexLine(codex({}, { cached_input_tokens: 101 }), { fileIdentity: "f", now: NOW });
    expect(badCached).toMatchObject({ skip: "bad-constraint" });
    const badReasoning = parseCodexLine(codex({}, { reasoning_output_tokens: 21 }), { fileIdentity: "f", now: NOW });
    expect(badReasoning).toMatchObject({ skip: "bad-constraint" });
    expect(parseCodexLine(JSON.stringify({ type: "event_msg", timestamp: "2026-11-17T10:00:00Z", payload: { type: "token_count" } }), { fileIdentity: "f", now: NOW })).toMatchObject({ skip: "unsupported" });
  });

  it("exposes cache write as a detail without adding it", () => {
    const out = parseCodexLine(codex({}, { cache_write_input_tokens: 9 }), { fileIdentity: "f", now: NOW });
    if (!("item" in out)) throw new Error("expected item");
    expect(out.item.allInput).toBe(100);
    expect(out.item.details.cacheWrite).toBe(9);
  });

  it("parses the turn-context model join key", () => {
    const line = JSON.stringify({ type: "turn_context", timestamp: "2026-11-17T10:00:00Z", payload: { turn_id: "turn-1", root_turn_id: "turn-1", model: "gpt-5.6-luna" } });
    const out = parseCodexLine(line, { fileIdentity: "rollout.jsonl", now: NOW });
    expect(out).toEqual({ context: { fileIdentity: "rollout.jsonl", turnId: "turn-1", rootTurnId: "turn-1", model: "gpt-5.6-luna" } });
  });
});

describe("OpenCode accounting", () => {
  it("maps the two disjoint classes exactly (116 input / 37 output / 153 headline)", () => {
    const out = parseOpenCodeRow(opencode(), { now: NOW });
    if (!("item" in out)) throw new Error(`expected item, got ${out.skip}`);
    expect(out.item).toMatchObject({ allInput: 116, allOutput: 37, projection: "v2", cost: { amount: 12.5, provenance: "source-reported", currency: null } });
    expect(accountingSummary([out.item])).toEqual({ allInput: 116, allOutput: 37, headline: 153 });
  });

  it("keeps V2 canonical per session and excludes V1 instead of adding it", () => {
    const v1 = parseOpenCodeRow(opencode({ projection: "v1", message_id: "msg-1" }), { now: NOW });
    const v2 = parseOpenCodeRow(opencode({ projection: "v2", message_id: "msg-1" }), { now: NOW });
    const v1Other = parseOpenCodeRow(opencode({ projection: "v1", message_id: "msg-2" }), { now: NOW });
    if (!("item" in v1) || !("item" in v2) || !("item" in v1Other)) throw new Error("expected items");
    const canonical = canonicalOpenCodeItems([v1.item, v2.item, v1Other.item]);
    expect(canonical.items).toEqual([v2.item]);
    expect(canonical.excludedV1).toBe(2);
    expect(canonical.mixedSessions).toBe(1);
  });

  it("supports pure V1 when no V2 assistant projection exists", () => {
    const v1 = parseOpenCodeRow(opencode({ projection: "v1" }), { now: NOW });
    if (!("item" in v1)) throw new Error("expected item");
    const canonical = canonicalOpenCodeItems([v1.item]);
    expect(canonical.items).toEqual([v1.item]);
    expect(canonical.excludedV1).toBe(0);
  });

  it("marks a native child session and retains its parent link", () => {
    const child = parseOpenCodeRow(opencode({ parent_session_id: "ses-parent" }), { now: NOW });
    if (!("item" in child)) throw new Error("expected item");
    expect(child.item).toMatchObject({ isChild: true, childId: "ses-1", parentId: "ses-parent" });
  });

  it("excludes overlength Unicode IDs, malformed counters and malformed rows", () => {
    expect(parseOpenCodeRow(opencode({ message_id: "😀".repeat(257) }), { now: NOW })).toMatchObject({ skip: "id-too-long" });
    expect(parseOpenCodeRow(opencode({ input: -1 }), { now: NOW })).toMatchObject({ skip: "bad-counter" });
    expect(parseOpenCodeRow({ projection: "v3" }, { now: NOW })).toMatchObject({ skip: "unsupported" });
    expect(parseOpenCodeRow(opencode({ created: 0 }), { now: NOW })).toMatchObject({ skip: "bad-timestamp" });
  });

  it("keeps unknown cost null rather than zero", () => {
    const out = parseOpenCodeRow(opencode({ cost: null }), { now: NOW });
    if (!("item" in out)) throw new Error("expected item");
    expect(out.item.cost).toBeNull();
  });
});
