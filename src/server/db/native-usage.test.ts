import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { openDb } from "./index";
import {
  applyFileBatch,
  getFileCheckpoint,
  getSourceStatuses,
  insertNativeContext,
  pruneNativeUsage,
  queryNativeUsage,
  setSourceStatus,
  UNKNOWN_MODEL_FILTER,
  upsertNativeUsageItem,
  type FileCheckpoint,
} from "./native-usage";
import {
  parseClaudeLine,
  parseCodexLine,
  parseOpenCodeRow,
  type NativeUsageItem,
} from "../collectors/native-usage";

const NOW = Date.parse("2026-11-18T12:00:00Z") / 1000;
const DAY = 86400;

function claude(raw: Record<string, unknown>): NativeUsageItem {
  const line = JSON.stringify({
    type: "assistant",
    sessionId: "sess-1",
    timestamp: "2026-11-17T10:00:00.000Z",
    message: { id: "msg-1", model: "claude-sonnet-5-5", usage: { input_tokens: 10, output_tokens: 5 } },
    ...raw,
  });
  const out = parseClaudeLine(line, { child: false, now: NOW });
  if (!("item" in out)) throw new Error(out.skip);
  return out.item;
}

function codex(fileIdentity: string, over: Record<string, unknown> = {}, usage: Record<string, unknown> = {}): NativeUsageItem {
  const line = JSON.stringify({
    type: "token_usage_record",
    timestamp: "2026-11-17T10:00:00.000Z",
    payload: {
      session_id: "root-1",
      thread_id: "thread-1",
      turn_id: "turn-1",
      root_turn_id: "turn-1",
      response_id: "resp-1",
      usage: { input_tokens: 100, output_tokens: 20, total_tokens: 120, ...usage },
      ...over,
    },
  });
  const out = parseCodexLine(line, { fileIdentity, now: NOW });
  if (!("item" in out)) throw new Error("skip" in out ? out.skip : "context");
  return out.item;
}

function opencode(over: Record<string, unknown>): NativeUsageItem {
  const out = parseOpenCodeRow({
    projection: "v2",
    session_id: "ses-1",
    message_id: "msg-1",
    parent_session_id: null,
    model_id: "model-x",
    provider_id: "provider-x",
    agent: "build",
    created: 1_794_909_600_000,
    updated: 1_794_913_200_000,
    cost: 12.5,
    input: 100,
    output: 30,
    reasoning: 7,
    cache_read: 11,
    cache_write: 5,
    ...over,
  }, { now: NOW });
  if (!("item" in out)) throw new Error(out.skip);
  return out.item;
}

const checkpoint = (over: Partial<FileCheckpoint> = {}): FileCheckpoint => ({
  source: "claude",
  identity: "p/one.jsonl",
  generation: "1:1",
  dev: "1",
  ino: "1",
  size: 100,
  mtime: NOW,
  offset: 100,
  done: true,
  processedAt: NOW,
  ...over,
});

describe("migration 15 native usage tables", () => {
  it("adds numeric-only item, checkpoint, context, projection and source tables", () => {
    const db = openDb(":memory:");
    const tables = (db.prepare("SELECT name FROM sqlite_master WHERE type = 'table'").all() as { name: string }[]).map((t) => t.name);
    expect(tables).toEqual(expect.arrayContaining([
      "native_usage_items", "native_usage_files", "native_usage_contexts",
      "native_usage_session_facts", "native_usage_sources",
    ]));
    expect(db.pragma("user_version", { simple: true })).toBe(15);
    db.close();
  });
});

describe("native item persistence", () => {
  it("keeps one row per global native identity across mirrored files", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, {
      checkpoint: checkpoint({ identity: "a.jsonl" }),
      expectedOffset: null,
      items: [codex("a.jsonl")],
      now: NOW,
    });
    applyFileBatch(db, {
      checkpoint: checkpoint({ identity: "b.jsonl" }),
      expectedOffset: null,
      items: [codex("b.jsonl")],
      now: NOW,
    });
    expect(db.prepare("SELECT COUNT(*) AS n, SUM(all_input) AS input FROM native_usage_items").get())
      .toEqual({ n: 1, input: 100 });
    db.close();
  });

  it("reversed ingestion produces the same Claude counters, model union and attribution", () => {
    const early = claude({
      sessionId: "sess-b",
      timestamp: "2026-11-17T09:00:00Z",
      message: { id: "msg-1", model: "model-a", usage: { input_tokens: 10, output_tokens: 1 } },
    });
    const late = claude({
      sessionId: "sess-a",
      timestamp: "2026-11-17T11:00:00Z",
      message: { id: "msg-1", model: "model-b", usage: { input_tokens: 10, output_tokens: 9 } },
    });
    const dbA = dbFor([early, late]);
    const dbB = dbFor([late, early]);
    const snapshots = [dbA, dbB].map((db) => {
      const snap = queryNativeUsage(db, { period: 30, now: NOW });
      return { totalTokens: snap.totalTokens, byModel: snap.byModel, sessions: snap.sessions };
    });
    expect(snapshots[0]).toEqual(snapshots[1]);
    expect(snapshots[0].totalTokens).toBe(19);
    expect(snapshots[0].byModel.map((m) => m.name)).toEqual([UNKNOWN_MODEL_FILTER]);
    dbA.close();
    dbB.close();
  });

  it("warns on conflicting native identities while keeping deterministic attribution", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, claude({ sessionId: "sess-b" }));
    upsertNativeUsageItem(db, claude({ sessionId: "sess-a" }));
    const snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.warnings).toContain("identity-conflict");
    expect(snap.sessions.lead).toBe(1);
    expect(db.prepare("SELECT session_id, identity_conflict FROM native_usage_items").get())
      .toEqual({ session_id: "sess-a", identity_conflict: 1 });
    db.close();
  });

  it("an item upsert and the consumed offset share one transaction", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, { checkpoint: checkpoint({ offset: 50 }), expectedOffset: 0, items: [], now: NOW });
    expect(() => applyFileBatch(db, {
      checkpoint: checkpoint({ offset: 100 }),
      expectedOffset: 49,
      items: [claude({})],
      now: NOW,
    })).toThrow("checkpoint-conflict");
    expect(db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 0 });
    expect(getFileCheckpoint(db, "claude", "p/one.jsonl")?.offset).toBe(50);
    db.close();
  });

  it("restarts a truncated/replaced file at zero without losing measured identities", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, { checkpoint: checkpoint({ offset: 100 }), expectedOffset: null, items: [claude({})], now: NOW });
    applyFileBatch(db, {
      checkpoint: checkpoint({ generation: "2:2", dev: "2", ino: "2", offset: 40 }),
      expectedOffset: null,
      items: [claude({})],
      now: NOW,
    });
    expect(getFileCheckpoint(db, "claude", "p/one.jsonl")).toMatchObject({ generation: "2:2", offset: 40 });
    expect(db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    db.close();
  });

  it("survives a concurrent duplicate upsert on a file-backed db", () => {
    const base = process.env.JAXOS_HOME || join(process.cwd(), ".local/verify-home");
    mkdirSync(base, { recursive: true });
    const dir = mkdtempSync(join(base, "native-usage-"));
    const path = join(dir, "jaxos.db");
    const a = openDb(path);
    const b = openDb(path);
    upsertNativeUsageItem(a, claude({}));
    upsertNativeUsageItem(b, claude({ message: { id: "msg-1", model: "claude-sonnet-5-5", usage: { input_tokens: 10, output_tokens: 5 } } }));
    expect(a.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    a.close();
    b.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("stores no raw transcript or sentinel text in any table", () => {
    const db = openDb(":memory:");
    const item = claude({ message: { id: "msg-1", content: "SENTINEL_SECRET_TEXT", usage: { input_tokens: 1, output_tokens: 1 } } });
    upsertNativeUsageItem(db, item);
    const dump = JSON.stringify(db.prepare("SELECT * FROM native_usage_items").all());
    expect(dump).not.toContain("SENTINEL_SECRET_TEXT");
    expect(item.details).toEqual({ input: 1, output: 1, cacheRead: 0, cacheWrite: 0, reasoning: null });
    db.close();
  });
});

describe("Codex context joins", () => {
  it("backfills an unknown model when the turn context arrives on a later pass", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, { checkpoint: checkpoint({ source: "codex", identity: "r.jsonl" }), expectedOffset: null, items: [codex("r.jsonl")], now: NOW });
    expect(queryNativeUsage(db, { period: 30, now: NOW }).byModel.map((m) => m.name)).toEqual([UNKNOWN_MODEL_FILTER]);
    insertNativeContext(db, { fileIdentity: "r.jsonl", turnId: "turn-1", rootTurnId: "turn-1", model: "gpt-5.6-luna" }, NOW);
    const snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.byModel.map((m) => m.name)).toEqual(["gpt-5.6-luna"]);
    db.close();
  });

  it("resolves a conflicting context to unknown and stays unknown", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, { checkpoint: checkpoint({ source: "codex", identity: "r.jsonl" }), expectedOffset: null, items: [codex("r.jsonl")], now: NOW });
    insertNativeContext(db, { fileIdentity: "r.jsonl", turnId: "turn-1", rootTurnId: "turn-1", model: "model-a" }, NOW);
    expect(insertNativeContext(db, { fileIdentity: "r.jsonl", turnId: "turn-1", rootTurnId: "turn-1", model: "model-b" }, NOW)).toBe("conflict");
    insertNativeContext(db, { fileIdentity: "r.jsonl", turnId: "turn-1", rootTurnId: "turn-1", model: "model-a" }, NOW);
    const snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.byModel.map((m) => m.name)).toEqual([UNKNOWN_MODEL_FILTER]);
    db.close();
  });

  it("keeps unknown when context was evicted, never guessing from file order", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, codex("r.jsonl", { turn_id: "gone", root_turn_id: "gone" }));
    pruneNativeUsage(db, NOW);
    const snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.byModel.map((m) => m.name)).toEqual([UNKNOWN_MODEL_FILTER]);
    db.close();
  });
});

describe("OpenCode projection reconciliation", () => {
  it("keeps pure V1, then deterministically recomputes to V2 when it arrives", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, opencode({ projection: "v1", input: 1, output: 1, reasoning: 0, cache_read: 0, cache_write: 0 }));
    let snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalTokens).toBe(2);
    expect(snap.mixedProjectionSessions).toBe(0);

    upsertNativeUsageItem(db, opencode({ projection: "v2" }));
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalTokens).toBe(153);
    expect(snap.mixedProjectionSessions).toBe(1);
    expect(snap.excludedV1Items).toBe(1);
    expect(snap.warnings).toContain("mixed-projection");

    upsertNativeUsageItem(db, opencode({ projection: "v1", input: 1, output: 1, reasoning: 0, cache_read: 0, cache_write: 0 }));
    expect(queryNativeUsage(db, { period: 30, now: NOW }).totalTokens).toBe(153);
    db.close();
  });

  it("records session projection facts without raw JSON", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, opencode({ projection: "v2" }));
    expect(db.prepare("SELECT * FROM native_usage_session_facts").get())
      .toEqual({ harness: "opencode", session_id: "ses-1", projection: "v2", time_updated: 1_794_913_200 });
    db.close();
  });
});

describe("queryNativeUsage", () => {
  it("uses UTC calendar buckets including the current partial day and a long-running session landing today", () => {
    const db = openDb(":memory:");
    const atBoundary = claude({ timestamp: "2026-11-12T00:00:00Z", message: { id: "m1", usage: { input_tokens: 3, output_tokens: 4 } } });
    const old = claude({ timestamp: "2026-11-11T23:59:59Z", message: { id: "m2", usage: { input_tokens: 100, output_tokens: 100 } } });
    const today = claude({ timestamp: "2026-11-18T01:00:00Z", message: { id: "m3", usage: { input_tokens: 1, output_tokens: 1 } } });
    for (const item of [atBoundary, old, today]) upsertNativeUsageItem(db, item);
    const snap = queryNativeUsage(db, { period: 7, now: NOW });
    expect(snap.daily).toHaveLength(7);
    expect(snap.daily[0].date).toBe("2026-11-12");
    expect(snap.daily[0].tokens).toBe(7);
    expect(snap.daily[6].date).toBe("2026-11-18");
    expect(snap.daily[6].tokens).toBe(2);
    expect(snap.totalTokens).toBe(9);
    expect(snap.history.includesCurrentPartialDay).toBe(true);
    db.close();
  });

  it("applies agent and model filters to totals, slices and session counts", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, claude({}));
    upsertNativeUsageItem(db, opencode({}));
    const all = queryNativeUsage(db, { period: 30, now: NOW });
    const claudeOnly = queryNativeUsage(db, { period: 30, agent: "Claude Code", now: NOW });
    const modelOnly = queryNativeUsage(db, { period: 30, model: "model-x", now: NOW });
    expect(all.totalTokens).toBe(15 + 153);
    expect(claudeOnly.totalTokens).toBe(15);
    expect(claudeOnly.byAgent).toHaveLength(1);
    expect(claudeOnly.sessions).toEqual({ lead: 1, child: 0 });
    expect(modelOnly.totalTokens).toBe(153);
    expect(modelOnly.filters.agents).toEqual(expect.arrayContaining(["Claude Code", "build"]));
    expect(modelOnly.totalCacheRead).toBe(11);
    expect(modelOnly.totalCacheWrite).toBe(5);
    expect(modelOnly.totalReasoning).toBe(7);
    db.close();
  });

  it("never turns unknown cost into zero and labels measured coverage", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, opencode({}));
    let snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalCost).toBeNull();
    expect(snap.costCoverage).toEqual({ measured: 1, total: 1 });
    db.prepare("UPDATE native_usage_items SET cost_currency = 'USD' WHERE harness = 'opencode'").run();
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalCost).toBe(12.5);
    upsertNativeUsageItem(db, claude({}));
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalCost).toBe(12.5);
    expect(snap.costCoverage).toEqual({ measured: 1, total: 2 });
    db.close();
  });

  it("counts distinct lead and child sessions separately", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, claude({}));
    const child = parseClaudeLine(JSON.stringify({
      type: "assistant", sessionId: "sess-1", agentId: "agent-7", timestamp: "2026-11-17T10:00:00Z",
      message: { id: "msg-2", usage: { input_tokens: 1, output_tokens: 1 } },
    }), { child: true, now: NOW });
    if (!("item" in child)) throw new Error("expected child item");
    upsertNativeUsageItem(db, child.item);
    expect(queryNativeUsage(db, { period: 30, now: NOW }).sessions).toEqual({ lead: 1, child: 1 });
    db.close();
  });

  it("prunes only history before the 91-day cutoff and keeps data when a source file disappears", () => {
    const db = openDb(":memory:");
    const old = claude({ timestamp: "2026-08-18T10:00:00Z", message: { id: "old", usage: { input_tokens: 5, output_tokens: 5 } } });
    const recent = claude({ timestamp: "2026-08-21T10:00:00Z", message: { id: "recent", usage: { input_tokens: 2, output_tokens: 2 } } });
    upsertNativeUsageItem(db, old);
    upsertNativeUsageItem(db, recent);
    expect(queryNativeUsage(db, { period: 90, now: NOW }).history.cutoffDay).toBe("2026-08-20");
    pruneNativeUsage(db, NOW);
    const snap = queryNativeUsage(db, { period: 90, now: NOW });
    expect(snap.totalTokens).toBe(4);
    expect(db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    db.close();
  });

  it("exposes disabled, indexing and unavailable source states distinctly", () => {
    const db = openDb(":memory:");
    setSourceStatus(db, { source: "claude", state: "disabled", lastRefresh: null, progress: null, warnings: [] }, NOW);
    setSourceStatus(db, { source: "codex", state: "unavailable", lastRefresh: 9, progress: null, warnings: ["read-failed"] }, NOW);
    const snap = queryNativeUsage(db, { period: 7, now: NOW });
    expect(snap.sources.map((s) => s.state)).toEqual(["disabled", "unavailable", "indexing"]);
    expect(snap.history.indexing).toBe(true);
    expect(snap.warnings).toContain("read-failed");
    expect(getSourceStatuses(db)).toHaveLength(2);
    db.close();
  });
});

function dbFor(items: NativeUsageItem[]) {
  const db = openDb(":memory:");
  for (const item of items) upsertNativeUsageItem(db, item);
  return db;
}
