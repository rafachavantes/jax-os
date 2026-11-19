import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { openDb } from "./index";
import {
  addPendingDir,
  applyFileBatch,
  ensureDiscoveryCycle,
  finishDiscoveryCycle,
  getFileCheckpoint,
  getPersistentWarnings,
  getSourceStatuses,
  insertNativeContext,
  markDirDone,
  markFileFailure,
  pruneNativeUsage,
  queryNativeUsage,
  registerNativeFile,
  selectIngestionCandidates,
  setSourceStatus,
  UNKNOWN_MODEL_FILTER,
  upsertNativeUsageItem,
  type FileCheckpoint,
  type NativeUsageSnapshot,
  type UsageSlice,
} from "./native-usage";
import {
  parseClaudeLine,
  parseCodexLine,
  parseOpenCodeRow,
  utcDay,
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
      "native_usage_dirs", "native_usage_discovery",
    ]));
    const indexes = (db.prepare("SELECT name FROM sqlite_master WHERE type = 'index'").all() as { name: string }[])
      .map((i) => i.name);
    expect(indexes).toContain("native_usage_items_context_ref");
    expect(db.pragma("user_version", { simple: true })).toBe(15);
    db.close();
  });
});

describe("native item persistence", () => {
  it("keeps one row per global native identity across mirrored files", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, {
      checkpoint: checkpoint({ identity: "a.jsonl" }),
      expectedPrior: null,
      items: [codex("a.jsonl")],
      now: NOW,
    });
    applyFileBatch(db, {
      checkpoint: checkpoint({ identity: "b.jsonl" }),
      expectedPrior: null,
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
    applyFileBatch(db, { checkpoint: checkpoint({ offset: 50 }), expectedPrior: null, items: [], now: NOW });
    expect(() => applyFileBatch(db, {
      checkpoint: checkpoint({ offset: 100 }),
      expectedPrior: checkpoint({ offset: 49 }),
      items: [claude({})],
      now: NOW,
    })).toThrow("checkpoint-conflict");
    expect(db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 0 });
    expect(getFileCheckpoint(db, "claude", "p/one.jsonl")?.offset).toBe(50);
    db.close();
  });

  it("restarts a truncated/replaced file at zero without losing measured identities", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, { checkpoint: checkpoint({ offset: 100 }), expectedPrior: null, items: [claude({})], now: NOW });
    applyFileBatch(db, {
      checkpoint: checkpoint({ generation: "2:2", dev: "2", ino: "2", offset: 40 }),
      expectedPrior: checkpoint({ offset: 100 }),
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
    applyFileBatch(db, { checkpoint: checkpoint({ source: "codex", identity: "r.jsonl" }), expectedPrior: null, items: [codex("r.jsonl")], now: NOW });
    expect(queryNativeUsage(db, { period: 30, now: NOW }).byModel.map((m) => m.name)).toEqual([UNKNOWN_MODEL_FILTER]);
    insertNativeContext(db, { fileIdentity: "r.jsonl", turnId: "turn-1", rootTurnId: "turn-1", model: "gpt-5.6-luna" }, NOW);
    const snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.byModel.map((m) => m.name)).toEqual(["gpt-5.6-luna"]);
    db.close();
  });

  it("resolves a conflicting context to unknown and stays unknown", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, { checkpoint: checkpoint({ source: "codex", identity: "r.jsonl" }), expectedPrior: null, items: [codex("r.jsonl")], now: NOW });
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
    // The V1 row mirrors the same native message, so it is not a coverage gap.
    expect(snap.mixedProjectionSessions).toBe(0);
    expect(snap.excludedV1Items).toBe(0);
    expect(snap.warnings).not.toContain("mixed-projection");

    upsertNativeUsageItem(db, opencode({
      projection: "v1", message_id: "msg-unmatched", input: 1, output: 1, reasoning: 0, cache_read: 0, cache_write: 0,
    }));
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalTokens).toBe(153);
    expect(snap.mixedProjectionSessions).toBe(1);
    expect(snap.excludedV1Items).toBe(1);
    expect(snap.warnings).toContain("mixed-projection");
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

describe("retention, CAS and cost coverage", () => {
  it("rejects an item older than the 91-day cutoff inside ingestion while advancing the offset", () => {
    const db = openDb(":memory:");
    const old = claude({ timestamp: "2026-07-01T10:00:00Z", message: { id: "old", usage: { input_tokens: 5, output_tokens: 5 } } });
    const recent = claude({ timestamp: "2026-11-17T10:00:00Z", message: { id: "recent", usage: { input_tokens: 1, output_tokens: 1 } } });
    const result = applyFileBatch(db, {
      checkpoint: checkpoint({ offset: 50 }),
      expectedPrior: null,
      items: [old, recent],
      now: NOW,
    });
    expect(result.rejected).toBe(1);
    expect(getFileCheckpoint(db, "claude", "p/one.jsonl")?.offset).toBe(50);
    expect(db.prepare("SELECT item_id FROM native_usage_items").all()).toEqual([{ item_id: "recent" }]);
    db.close();
  });

  it("refuses to overwrite newer progress and to reuse a completed transition", () => {
    const db = openDb(":memory:");
    applyFileBatch(db, { checkpoint: checkpoint({ offset: 50 }), expectedPrior: null, items: [], now: NOW });
    expect(() => applyFileBatch(db, {
      checkpoint: checkpoint({ offset: 60 }),
      expectedPrior: checkpoint({ offset: 40 }),
      items: [],
      now: NOW,
    })).toThrow("checkpoint-conflict");
    applyFileBatch(db, {
      checkpoint: checkpoint({ generation: "2:2", dev: "2", ino: "2", offset: 10 }),
      expectedPrior: checkpoint({ offset: 50 }),
      items: [],
      now: NOW,
    });
    expect(() => applyFileBatch(db, {
      checkpoint: checkpoint({ generation: "2:2", dev: "2", ino: "2", offset: 20 }),
      expectedPrior: checkpoint({ offset: 50 }),
      items: [],
      now: NOW,
    })).toThrow("checkpoint-conflict");
    expect(getFileCheckpoint(db, "claude", "p/one.jsonl")).toMatchObject({ generation: "2:2", offset: 10 });
    db.close();
  });

  it("retains referenced old contexts and prunes unreferenced ones", () => {
    const db = openDb(":memory:");
    db.prepare(
      `INSERT INTO native_usage_items (harness, item_id, session_id, provenance, agent, ts, updated_ts, day, all_input, all_output, file_identity, turn_id, root_turn_id)
       VALUES ('codex', 'thread-1|resp-1', 'thread-1', 'root-1', 'Codex', ?, ?, ?, 1, 1, 'r.jsonl', 'turn-1', 'turn-1')`,
    ).run(NOW, NOW, utcDay(NOW - 200 * DAY));
    insertNativeContext(db, { fileIdentity: "r.jsonl", turnId: "turn-1", rootTurnId: "turn-1", model: "gpt-5.6-luna" }, NOW - 200 * DAY);
    insertNativeContext(db, { fileIdentity: "gone.jsonl", turnId: "turn-x", rootTurnId: "turn-x", model: "gpt-5.6-luna" }, NOW - 200 * DAY);
    db.prepare("UPDATE native_usage_items SET day = ? WHERE item_id = 'thread-1|resp-1'").run(utcDay(NOW));
    pruneNativeUsage(db, NOW);
    const contexts = db.prepare("SELECT file_identity FROM native_usage_contexts ORDER BY file_identity").all() as { file_identity: string }[];
    expect(contexts).toEqual([{ file_identity: "r.jsonl" }]);
    db.close();
  });

  it("counts compatible cost coverage separately from unknown-unit amounts", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, opencode({}));
    upsertNativeUsageItem(db, claude({}));
    let snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalCost).toBeNull();
    expect(snap.costCoverage).toEqual({ measured: 0, total: 2, unknownUnit: 1 });
    db.prepare("UPDATE native_usage_items SET cost_currency = 'USD' WHERE harness = 'opencode'").run();
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalCost).toBe(12.5);
    expect(snap.costCoverage).toEqual({ measured: 1, total: 2, unknownUnit: 0 });
    db.close();
  });

  it("fails the query envelope on an unrepresentable aggregate instead of wrapping", () => {
    const db = openDb(":memory:");
    const huge = 5_000_000_000_000_000;
    for (const id of ["m1", "m2"]) {
      db.prepare(
        `INSERT INTO native_usage_items (harness, item_id, session_id, provenance, agent, ts, updated_ts, day, all_input, all_output)
         VALUES ('claude', ?, 'sess-1', 'lead', 'Claude Code', ?, ?, ?, ?, 0)`,
      ).run(id, NOW, NOW, utcDay(NOW), huge);
    }
    expect(() => queryNativeUsage(db, { period: 7, now: NOW })).toThrow("aggregate-overflow");
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
    expect(snap.costCoverage).toEqual({ measured: 0, total: 1, unknownUnit: 1 });
    db.prepare("UPDATE native_usage_items SET cost_currency = 'USD' WHERE harness = 'opencode'").run();
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalCost).toBe(12.5);
    upsertNativeUsageItem(db, claude({}));
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.totalCost).toBe(12.5);
    expect(snap.costCoverage).toEqual({ measured: 1, total: 2, unknownUnit: 0 });
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
    const boundary = claude({ timestamp: "2026-08-20T00:00:00Z", message: { id: "boundary", usage: { input_tokens: 1, output_tokens: 0 } } });
    const recent = claude({ timestamp: "2026-08-21T10:00:00Z", message: { id: "recent", usage: { input_tokens: 2, output_tokens: 2 } } });
    upsertNativeUsageItem(db, old);
    upsertNativeUsageItem(db, boundary);
    upsertNativeUsageItem(db, recent);
    expect(queryNativeUsage(db, { period: 90, now: NOW }).history.cutoffDay).toBe("2026-08-20");
    pruneNativeUsage(db, NOW);
    const snap = queryNativeUsage(db, { period: 90, now: NOW });
    expect(snap.totalTokens).toBe(4);
    expect(db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 2 });
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

describe("checkpoint compare-and-swap", () => {
  it("rejects a stale writer whose observed predecessor has since transitioned", () => {
    const db = openDb(":memory:");
    const observed = checkpoint({ generation: "1:1", offset: 100 });
    applyFileBatch(db, { checkpoint: observed, expectedPrior: null, items: [], now: NOW });
    const next = checkpoint({ generation: "2:2", dev: "2", ino: "2", offset: 50 });
    applyFileBatch(db, { checkpoint: next, expectedPrior: observed, items: [], now: NOW });
    expect(() => applyFileBatch(db, {
      checkpoint: checkpoint({ generation: "3:3", dev: "3", ino: "3", offset: 20 }),
      expectedPrior: observed,
      items: [claude({})],
      now: NOW,
    })).toThrow("checkpoint-conflict");
    expect(db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 0 });
    expect(getFileCheckpoint(db, "claude", "p/one.jsonl")).toMatchObject({ generation: "2:2", offset: 50 });
    db.close();
  });

  it("requires an exact prior state instead of an offset-zero wildcard", () => {
    const db = openDb(":memory:");
    const placeholder = checkpoint({ generation: "", offset: 0, done: false });
    registerNativeFile(db, "claude", "p/one.jsonl", false, NOW);
    expect(getFileCheckpoint(db, "claude", "p/one.jsonl")).toMatchObject({ generation: "", offset: 0 });
    applyFileBatch(db, {
      checkpoint: checkpoint({ generation: "1:1", offset: 10 }),
      expectedPrior: placeholder,
      items: [],
      now: NOW,
    });
    expect(() => applyFileBatch(db, {
      checkpoint: checkpoint({ generation: "1:1", offset: 20 }),
      expectedPrior: null,
      items: [],
      now: NOW,
    })).toThrow("checkpoint-conflict");
    expect(getFileCheckpoint(db, "claude", "p/one.jsonl")?.offset).toBe(10);
    db.close();
  });
});

describe("indexed ingestion selection", () => {
  function seed(db: ReturnType<typeof openDb>, identities: string[]): void {
    for (const identity of identities) registerNativeFile(db, "claude", identity, false, NOW);
  }
  function setMtime(db: ReturnType<typeof openDb>, identity: string, mtime: number): void {
    db.prepare("UPDATE native_usage_files SET mtime = ? WHERE source = 'claude' AND identity = ?").run(mtime, identity);
  }

  it("picks the two persisted-newest mtimes with identity tie-breaks, not visit order", () => {
    const db = openDb(":memory:");
    const ids = Array.from({ length: 6 }, (_, i) => `p/f${i}.jsonl`);
    seed(db, ids);
    for (const id of ids) setMtime(db, id, NOW - 3600);
    setMtime(db, ids[4], NOW);
    setMtime(db, ids[5], NOW);
    const first = selectIngestionCandidates(db, "claude", NOW, 6);
    expect(first.slice(0, 2).map((c) => c.identity)).toEqual(["p/f4.jsonl", "p/f5.jsonl"]);
    expect(first).toHaveLength(6);
    expect(first.slice(2).map((c) => c.identity)).not.toContain("p/f4.jsonl");
    db.close();
  });

  it("keeps four fair slots rotating over done files and defers failed entries", () => {
    const db = openDb(":memory:");
    const ids = Array.from({ length: 10 }, (_, i) => `p/f${i}.jsonl`);
    seed(db, ids);
    for (const id of ids) {
      db.prepare("UPDATE native_usage_files SET done = 1 WHERE source = 'claude' AND identity = ?").run(id);
    }
    setMtime(db, ids[0], NOW);
    setMtime(db, ids[1], NOW);
    markFileFailure(db, "claude", ids[2], NOW + 60, NOW, "missing-file");
    const first = selectIngestionCandidates(db, "claude", NOW, 6);
    expect(first.map((c) => c.identity)).not.toContain(ids[2]);
    expect(first).toHaveLength(6);
    const fairFirst = first.slice(2).map((c) => c.identity);
    const second = selectIngestionCandidates(db, "claude", NOW, 6);
    const fairSecond = second.slice(2).map((c) => c.identity);
    expect(fairFirst).not.toEqual(fairSecond);
    expect(new Set([...fairFirst, ...fairSecond]).size).toBeGreaterThan(4);
    expect(getPersistentWarnings(db, "claude")).toEqual(["missing-file"]);
    db.close();
  });

  it("selects two genuinely newest known tails among 500 same-day files within two calls", () => {
    const db = openDb(":memory:");
    const ids = Array.from({ length: 500 }, (_, i) => `p/f${String(i).padStart(3, "0")}.jsonl`);
    seed(db, ids);
    for (const id of ids) setMtime(db, id, NOW - 600);
    setMtime(db, ids[498], NOW);
    setMtime(db, ids[499], NOW);
    expect(selectIngestionCandidates(db, "claude", NOW, 2).map((c) => c.identity))
      .toEqual(["p/f498.jsonl", "p/f499.jsonl"]);
    expect(selectIngestionCandidates(db, "claude", NOW, 2).map((c) => c.identity))
      .toEqual(["p/f498.jsonl", "p/f499.jsonl"]);
    db.close();
  });

  it("plans hot, fair and pending-directory queries without a temporary sort", () => {
    const db = openDb(":memory:");
    const plan = (sql: string, ...params: unknown[]) =>
      (db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...params) as { detail: string }[]).map((r) => r.detail);
    for (const details of [
      plan(
        "SELECT identity FROM native_usage_files WHERE source = ? AND failed_until <= ? ORDER BY mtime DESC, identity ASC LIMIT ?",
        "claude", NOW, 2,
      ),
      plan(
        "SELECT identity FROM native_usage_files WHERE source = ? AND failed_until <= ? ORDER BY visit_ord ASC, identity ASC LIMIT ?",
        "claude", NOW, 6,
      ),
      plan("SELECT path FROM native_usage_dirs WHERE source = ? AND state = 'pending' ORDER BY path ASC LIMIT 1", "claude"),
      plan("SELECT path FROM native_usage_dirs WHERE source = ? AND state = 'pending' ORDER BY path DESC LIMIT 1", "codex"),
    ]) {
      expect(details.join("\n")).not.toMatch(/TEMP B-TREE/i);
    }
    db.close();
  });
});

describe("directory failure facts across discovery cycles", () => {
  const CADENCE = 60;

  it("retains a failed directory row across cycle start and intermediate polls", () => {
    const db = openDb(":memory:");
    const first = ensureDiscoveryCycle(db, "claude", NOW, CADENCE);
    addPendingDir(db, "claude", "proj", false, first.cycle, NOW);
    markDirDone(db, "claude", "proj", NOW, "subdir-failed");
    markDirDone(db, "claude", ".", NOW);
    finishDiscoveryCycle(db, "claude", NOW);
    expect(getPersistentWarnings(db, "claude")).toEqual(["subdir-failed"]);

    const second = ensureDiscoveryCycle(db, "claude", NOW + 61, CADENCE);
    expect(second.cycle).toBe(2);
    expect(
      db.prepare("SELECT state, cycle, last_warning FROM native_usage_dirs WHERE source = 'claude' AND path = 'proj'").get(),
    ).toEqual({ state: "done", cycle: 1, last_warning: "subdir-failed" });
    expect(getPersistentWarnings(db, "claude")).toEqual(["subdir-failed"]);
    expect(getPersistentWarnings(db, "claude")).toEqual(["subdir-failed"]);

    addPendingDir(db, "claude", "proj", false, second.cycle, NOW + 61);
    expect(
      db.prepare("SELECT state, cycle, last_warning FROM native_usage_dirs WHERE source = 'claude' AND path = 'proj'").get(),
    ).toEqual({ state: "pending", cycle: 2, last_warning: "subdir-failed" });

    markDirDone(db, "claude", "proj", NOW + 62);
    expect(getPersistentWarnings(db, "claude")).toEqual([]);
    db.close();
  });

  it("clears an unseen failed directory only after a successful complete enumeration", () => {
    const db = openDb(":memory:");
    ensureDiscoveryCycle(db, "claude", NOW, CADENCE);
    addPendingDir(db, "claude", "proj/gone", false, 1, NOW);
    markDirDone(db, "claude", "proj/gone", NOW, "discovery-failed");
    markDirDone(db, "claude", ".", NOW);
    finishDiscoveryCycle(db, "claude", NOW);
    expect(getPersistentWarnings(db, "claude")).toEqual(["discovery-failed"]);

    ensureDiscoveryCycle(db, "claude", NOW + 61, CADENCE);
    markDirDone(db, "claude", ".", NOW + 61);
    finishDiscoveryCycle(db, "claude", NOW + 61);
    expect(db.prepare("SELECT COUNT(*) AS n FROM native_usage_dirs WHERE source = 'claude' AND path = 'proj/gone'").get())
      .toEqual({ n: 0 });
    expect(getPersistentWarnings(db, "claude")).toEqual([]);
    db.close();
  });

  it("retains a descendant's failure when an ancestor fails in the current cycle", () => {
    const db = openDb(":memory:");
    ensureDiscoveryCycle(db, "claude", NOW, CADENCE);
    addPendingDir(db, "claude", "proj", false, 1, NOW);
    addPendingDir(db, "claude", "proj/a", false, 1, NOW);
    markDirDone(db, "claude", "proj/a", NOW, "discovery-failed");
    markDirDone(db, "claude", "proj", NOW, "subdir-failed");
    markDirDone(db, "claude", ".", NOW);
    finishDiscoveryCycle(db, "claude", NOW);

    ensureDiscoveryCycle(db, "claude", NOW + 61, CADENCE);
    addPendingDir(db, "claude", "proj", false, 2, NOW + 61);
    markDirDone(db, "claude", "proj", NOW + 61, "subdir-failed");
    markDirDone(db, "claude", ".", NOW + 61);
    finishDiscoveryCycle(db, "claude", NOW + 61);
    expect(
      db.prepare("SELECT cycle, state, last_warning FROM native_usage_dirs WHERE source = 'claude' AND path = 'proj/a'").get(),
    ).toEqual({ cycle: 1, state: "done", last_warning: "discovery-failed" });
    expect(getPersistentWarnings(db, "claude").sort()).toEqual(["discovery-failed", "subdir-failed"]);
    db.close();
  });

  it("plans the registry maintenance statements without a temporary sort", () => {
    const db = openDb(":memory:");
    const plan = (sql: string, ...params: unknown[]) =>
      (db.prepare(`EXPLAIN QUERY PLAN ${sql}`).all(...params) as { detail: string }[]).map((r) => r.detail);
    for (const details of [
      plan("DELETE FROM native_usage_dirs WHERE source = ? AND last_warning IS NULL", "claude"),
      plan("DELETE FROM native_usage_dirs WHERE source = ? AND cycle < ?", "claude", 1),
      plan(
        "SELECT 1 AS n FROM native_usage_dirs WHERE source = ? AND cycle = ? AND last_warning IS NOT NULL LIMIT 1",
        "claude",
        1,
      ),
    ]) {
      expect(details.join("\n")).not.toMatch(/TEMP B-TREE/i);
    }
    db.close();
  });
});

function dbFor(items: NativeUsageItem[]) {
  const db = openDb(":memory:");
  for (const item of items) upsertNativeUsageItem(db, item);
  return db;
}

// ---------------------------------------------------------------------------
// Frozen parity oracle: the pre-change 342 JS reduction, kept test-only.

type LegacyQuery = { period: 7 | 30 | 90; agent?: string | null; model?: string | null; now: number };

type LegacyRow = {
  harness: string;
  session_id: string;
  child_id: string;
  is_child: number;
  agent: string;
  model: string | null;
  day: string;
  all_input: number;
  all_output: number;
  cache_read: number | null;
  cache_write: number | null;
  reasoning: number | null;
  cost_amount: number | null;
  cost_currency: string | null;
  identity_conflict: number;
};

type LegacySlice = { name: string; allInput: number; allOutput: number; tokens: number; cost: number | null; currency: string | null; items: number };

function legacySliceCost(rows: LegacyRow[]): { amount: number | null; currency: string | null; measured: number } {
  const costed = rows.filter((r) => r.cost_amount !== null);
  const usable = costed.filter((r) => r.cost_currency !== null);
  if (usable.length === 0) return { amount: null, currency: null, measured: 0 };
  const currencies = new Set(usable.map((r) => r.cost_currency));
  if (currencies.size !== 1) return { amount: null, currency: null, measured: 0 };
  let amount = 0;
  for (const row of usable) amount += row.cost_amount as number;
  return { amount, currency: [...currencies][0], measured: usable.length };
}

function legacySlice(name: string, rows: LegacyRow[]): LegacySlice {
  const allInput = rows.reduce((total, r) => total + r.all_input, 0);
  const allOutput = rows.reduce((total, r) => total + r.all_output, 0);
  const cost = legacySliceCost(rows);
  return { name, allInput, allOutput, tokens: allInput + allOutput, cost: cost.amount, currency: cost.currency, items: rows.length };
}

function legacyDetail(rows: LegacyRow[], key: "cache_read" | "cache_write" | "reasoning"): number | null {
  const values = rows.map((r) => r[key]).filter((v): v is number => v !== null);
  return values.length > 0 ? values.reduce((total, v) => total + v, 0) : null;
}

function legacyDaysBetween(startDay: string, endDay: string): string[] {
  const out: string[] = [];
  const start = Date.parse(`${startDay}T00:00:00Z`);
  const end = Date.parse(`${endDay}T00:00:00Z`);
  for (let ms = start; ms <= end; ms += 86400_000) out.push(new Date(ms).toISOString().slice(0, 10));
  return out;
}

// Independently stated F1 expectation: only selected V1 rows whose same session has a
// canonical V2 and whose own native message has no same-session V2 mirror are a gap.
function unmatchedV1(db: ReturnType<typeof openDb>, opts: LegacyQuery): { items: number; sessions: number } {
  const endDay = utcDay(opts.now);
  const startDay = utcDay(opts.now - (opts.period - 1) * 86400);
  const params: unknown[] = [startDay, endDay];
  let filter = "";
  if (opts.agent) {
    filter += " AND i.agent = ?";
    params.push(opts.agent);
  }
  if (opts.model !== undefined && opts.model !== null) {
    if (opts.model === UNKNOWN_MODEL_FILTER) filter += " AND i.model IS NULL";
    else {
      filter += " AND i.model = ?";
      params.push(opts.model);
    }
  }
  return db
    .prepare(
      `SELECT COUNT(*) AS items, COUNT(DISTINCT i.session_id) AS sessions
         FROM native_usage_items i
        WHERE i.harness = 'opencode' AND i.projection = 'v1'
          AND substr(i.item_id, -3) = '|v1'
          AND i.day >= ? AND i.day <= ?${filter}
          AND EXISTS (
            SELECT 1 FROM native_usage_items v2
             WHERE v2.harness = 'opencode' AND v2.projection = 'v2' AND v2.session_id = i.session_id
          )
          AND NOT EXISTS (
            SELECT 1 FROM native_usage_items mirror
             WHERE mirror.harness = 'opencode' AND mirror.projection = 'v2'
               AND mirror.session_id = i.session_id
               AND mirror.item_id = substr(i.item_id, 1, length(i.item_id) - 3) || '|v2'
          )`,
    )
    .get(...params) as { items: number; sessions: number };
}

function legacySnapshot(db: ReturnType<typeof openDb>, opts: LegacyQuery) {
  const days = opts.period;
  const endDay = utcDay(opts.now);
  const startDay = utcDay(opts.now - (days - 1) * 86400);
  const params: unknown[] = [startDay, endDay];
  let where = `i.day >= ? AND i.day <= ? AND (i.harness <> 'opencode' OR i.projection = 'v2' OR NOT EXISTS (
    SELECT 1 FROM native_usage_items v2
     WHERE v2.harness = 'opencode' AND v2.projection = 'v2' AND v2.session_id = i.session_id
  ))`;
  if (opts.agent) {
    where += " AND i.agent = ?";
    params.push(opts.agent);
  }
  if (opts.model !== undefined && opts.model !== null) {
    if (opts.model === UNKNOWN_MODEL_FILTER) where += " AND i.model IS NULL";
    else {
      where += " AND i.model = ?";
      params.push(opts.model);
    }
  }
  const rows = db
    .prepare(
      `SELECT i.harness, i.session_id, i.child_id, i.is_child, i.agent, i.model, i.day,
              i.all_input, i.all_output, i.cache_read, i.cache_write, i.reasoning,
              i.cost_amount, i.cost_currency, i.identity_conflict
         FROM native_usage_items i
        WHERE ${where}`,
    )
    .all(...params) as LegacyRow[];

  const daily = legacyDaysBetween(startDay, endDay).map((date) => ({ date, input: 0, output: 0, tokens: 0 }));
  const byDate = new Map(daily.map((d) => [d.date, d]));
  const agents = new Map<string, LegacyRow[]>();
  const models = new Map<string, LegacyRow[]>();
  const lead = new Set<string>();
  const child = new Set<string>();
  for (const row of rows) {
    const bucket = byDate.get(row.day);
    if (bucket) {
      bucket.input += row.all_input;
      bucket.output += row.all_output;
      bucket.tokens = bucket.input + bucket.output;
    }
    const agentRows = agents.get(row.agent) ?? [];
    agentRows.push(row);
    agents.set(row.agent, agentRows);
    const modelName = row.model ?? UNKNOWN_MODEL_FILTER;
    const modelRows = models.get(modelName) ?? [];
    modelRows.push(row);
    models.set(modelName, modelRows);
    if (row.is_child === 1) child.add(`${row.session_id}|${row.child_id}`);
    else lead.add(row.session_id);
  }
  const totalCost = legacySliceCost(rows);
  const unknownUnit = rows.filter((r) => r.cost_amount !== null && r.cost_currency === null).length;
  const sources = getSourceStatuses(db);
  const warnings = new Set(sources.flatMap((s) => s.warnings));
  const f1 = unmatchedV1(db, opts);
  if (f1.sessions > 0) warnings.add("mixed-projection");
  if (rows.some((r) => r.identity_conflict === 1)) warnings.add("identity-conflict");
  const totalInput = rows.reduce((total, r) => total + r.all_input, 0);
  const totalOutput = rows.reduce((total, r) => total + r.all_output, 0);
  return {
    totalInput,
    totalOutput,
    totalTokens: totalInput + totalOutput,
    totalCacheRead: legacyDetail(rows, "cache_read"),
    totalCacheWrite: legacyDetail(rows, "cache_write"),
    totalReasoning: legacyDetail(rows, "reasoning"),
    totalCost: totalCost.amount,
    costCurrency: totalCost.currency,
    costCoverage: { measured: totalCost.measured, total: rows.length, unknownUnit },
    daily,
    byModel: [...models.entries()]
      .map(([name, grouped]) => legacySlice(name, grouped))
      .sort((a, b) => b.tokens - a.tokens || (a.name === UNKNOWN_MODEL_FILTER ? 1 : b.name === UNKNOWN_MODEL_FILTER ? -1 : 0)),
    byAgent: [...agents.entries()].map(([name, grouped]) => legacySlice(name, grouped)).sort((a, b) => b.tokens - a.tokens),
    sessions: { lead: lead.size, child: child.size },
    mixedProjectionSessions: f1.sessions,
    excludedV1Items: f1.items,
    warnings: [...warnings].sort(),
  };
}

function expectSlices(actual: UsageSlice[], expected: LegacySlice[]): void {
  expect(actual.map((s) => s.name)).toEqual(expected.map((s) => s.name));
  expect(actual.map((s) => ({
    name: s.name, allInput: s.allInput, allOutput: s.allOutput, tokens: s.tokens, items: s.items, currency: s.currency,
  }))).toEqual(expected.map((s) => ({
    name: s.name, allInput: s.allInput, allOutput: s.allOutput, tokens: s.tokens, items: s.items, currency: s.currency,
  })));
  actual.forEach((slice, index) => {
    const expectedCost = expected[index].cost;
    if (expectedCost === null) expect(slice.cost).toBeNull();
    else expect(slice.cost as number).toBeCloseTo(expectedCost, 10);
  });
}

function expectMatchesLegacy(snap: NativeUsageSnapshot, legacy: ReturnType<typeof legacySnapshot>): void {
  expect(snap.totalInput).toBe(legacy.totalInput);
  expect(snap.totalOutput).toBe(legacy.totalOutput);
  expect(snap.totalTokens).toBe(legacy.totalTokens);
  expect(snap.totalCacheRead).toBe(legacy.totalCacheRead);
  expect(snap.totalCacheWrite).toBe(legacy.totalCacheWrite);
  expect(snap.totalReasoning).toBe(legacy.totalReasoning);
  if (legacy.totalCost === null) expect(snap.totalCost).toBeNull();
  else expect(snap.totalCost as number).toBeCloseTo(legacy.totalCost, 10);
  expect(snap.costCurrency).toBe(legacy.costCurrency);
  expect(snap.costCoverage).toEqual(legacy.costCoverage);
  expect(snap.daily).toEqual(legacy.daily);
  expectSlices(snap.byModel, legacy.byModel);
  expectSlices(snap.byAgent, legacy.byAgent);
  expect(snap.sessions).toEqual(legacy.sessions);
  expect(snap.mixedProjectionSessions).toBe(legacy.mixedProjectionSessions);
  expect(snap.excludedV1Items).toBe(legacy.excludedV1Items);
  expect(snap.warnings).toEqual(legacy.warnings);
}

const DAY_MS = 86400_000;
const createdAt = (daysAgo: number) => Date.parse("2026-11-17T10:00:00Z") - daysAgo * DAY_MS;

function seedAggregationFixture(db: ReturnType<typeof openDb>): void {
  const claudeAt = (daysAgo: number) =>
    claude({ timestamp: new Date(createdAt(daysAgo)).toISOString(), message: { id: `c-${daysAgo}`, model: "claude-sonnet-5-5", usage: { input_tokens: 7, output_tokens: 3, cache_read_input_tokens: 2 } } });
  upsertNativeUsageItem(db, claudeAt(0));
  upsertNativeUsageItem(db, claudeAt(3));
  upsertNativeUsageItem(db, claudeAt(40));
  const child = parseClaudeLine(JSON.stringify({
    type: "assistant", sessionId: "sess-child", agentId: "agent-7", timestamp: "2026-11-17T10:00:00Z",
    message: { id: "child-1", usage: { input_tokens: 1, output_tokens: 1 } },
  }), { child: true, now: NOW });
  if (!("item" in child)) throw new Error("expected child item");
  upsertNativeUsageItem(db, child.item);

  upsertNativeUsageItem(db, codex("r.jsonl", { response_id: "resp-a" }));
  upsertNativeUsageItem(db, codex("r.jsonl", { response_id: "resp-b" }, { input_tokens: 5, output_tokens: 2, total_tokens: 7 }));

  const oc = (over: Record<string, unknown>) => opencode({ created: createdAt(1), ...over });
  upsertNativeUsageItem(db, oc({ session_id: "ses-usd", message_id: "m-usd", model_id: "model-x", created: createdAt(2) }));
  upsertNativeUsageItem(db, oc({ session_id: "ses-unit", message_id: "m-unit", model_id: "model-x", cost: 2.5 }));
  upsertNativeUsageItem(db, oc({ session_id: "ses-mirror", message_id: "m-mirror", projection: "v1", cost: null, input: 4, output: 2 }));
  upsertNativeUsageItem(db, oc({ session_id: "ses-mirror", message_id: "m-mirror", projection: "v2", created: createdAt(35) }));
  upsertNativeUsageItem(db, oc({ session_id: "ses-gap", message_id: "m-gap-v1", projection: "v1", cost: null, input: 9, output: 1 }));
  upsertNativeUsageItem(db, oc({ session_id: "ses-gap", message_id: "m-gap-v2", projection: "v2", model_id: "model-y", created: createdAt(30) }));
  db.prepare("UPDATE native_usage_items SET cost_currency = 'USD' WHERE harness = 'opencode' AND session_id = 'ses-usd'").run();
  db.prepare("UPDATE native_usage_items SET cost_currency = 'EUR' WHERE harness = 'opencode' AND session_id = 'ses-gap' AND projection = 'v2'").run();
  upsertNativeUsageItem(db, claude({ sessionId: "c-conflict", timestamp: "2026-11-17T10:00:00Z", message: { id: "c-0", usage: { input_tokens: 1, output_tokens: 1 } } }));
}

describe("scoped projection diagnostics and reduced SQL aggregation", () => {
  it("parity-matches the frozen 342 reduction across periods, agents and models", () => {
    const db = openDb(":memory:");
    seedAggregationFixture(db);
    setSourceStatus(db, { source: "claude", state: "available", lastRefresh: NOW - 5, progress: { filesDone: 4, filesTotal: 4 }, warnings: ["unsupported-layout"] }, NOW);
    setSourceStatus(db, { source: "opencode", state: "available", lastRefresh: NOW - 5, progress: null, warnings: [] }, NOW);
    const queries: LegacyQuery[] = [
      { period: 30, now: NOW },
      { period: 7, now: NOW },
      { period: 90, now: NOW },
      { period: 30, agent: "Claude Code", now: NOW },
      { period: 30, agent: "build", now: NOW },
      { period: 30, model: "model-x", now: NOW },
      { period: 30, model: UNKNOWN_MODEL_FILTER, now: NOW },
      { period: 90, agent: "build", model: "model-x", now: NOW },
    ];
    for (const opts of queries) {
      expectMatchesLegacy(queryNativeUsage(db, opts), legacySnapshot(db, opts));
    }
    db.close();
  });

  it("counts only an unmatched selected V1 as a projection gap", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, opencode({ projection: "v1", message_id: "mirror", input: 1, output: 1 }));
    upsertNativeUsageItem(db, opencode({ projection: "v2", message_id: "mirror" }));
    let snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.excludedV1Items).toBe(0);
    expect(snap.mixedProjectionSessions).toBe(0);

    upsertNativeUsageItem(db, opencode({ projection: "v1", message_id: "unmatched", input: 1, output: 1 }));
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.excludedV1Items).toBe(1);
    expect(snap.mixedProjectionSessions).toBe(1);

    upsertNativeUsageItem(db, opencode({ projection: "v1", message_id: "other-session", session_id: "ses-2", input: 2, output: 2 }));
    upsertNativeUsageItem(db, opencode({ projection: "v2", message_id: "other-v2", session_id: "ses-2", model_id: "model-z", agent: "plan" }));
    snap = queryNativeUsage(db, { period: 30, now: NOW });
    expect(snap.excludedV1Items).toBe(2);
    expect(snap.mixedProjectionSessions).toBe(2);
    expect(queryNativeUsage(db, { period: 30, agent: "build", now: NOW }).excludedV1Items).toBe(2);
    expect(queryNativeUsage(db, { period: 30, agent: "plan", now: NOW }).excludedV1Items).toBe(0);
    expect(queryNativeUsage(db, { period: 30, model: "model-z", now: NOW }).excludedV1Items).toBe(0);
    expect(queryNativeUsage(db, { period: 30, model: "model-x", now: NOW }).excludedV1Items).toBe(2);
    db.close();
  });

  it("keeps a cross-period canonical V2 able to suppress a selected V1", () => {
    const db = openDb(":memory:");
    upsertNativeUsageItem(db, opencode({ projection: "v1", message_id: "old-v1", input: 50, output: 50, created: createdAt(0) }));
    upsertNativeUsageItem(db, opencode({ projection: "v2", message_id: "old-v2", created: createdAt(20) }));
    const snap = queryNativeUsage(db, { period: 7, now: NOW });
    expect(snap.totalTokens).toBe(0);
    expect(snap.excludedV1Items).toBe(1);
    expect(snap.mixedProjectionSessions).toBe(1);
    db.close();
  });

  it("accepts an aggregate exactly at the safe limit and rejects one above it", () => {
    const db = openDb(":memory:");
    db.prepare(
      `INSERT INTO native_usage_items (harness, item_id, session_id, provenance, agent, ts, updated_ts, day, all_input, all_output)
       VALUES ('claude', 'max', 'sess-max', 'lead', 'Claude Code', ?, ?, ?, ?, 0)`,
    ).run(NOW, NOW, utcDay(NOW), Number.MAX_SAFE_INTEGER);
    const snap = queryNativeUsage(db, { period: 7, now: NOW });
    expect(snap.totalTokens).toBe(Number.MAX_SAFE_INTEGER);
    expect(snap.daily.at(-1)?.tokens).toBe(Number.MAX_SAFE_INTEGER);
    db.prepare("UPDATE native_usage_items SET all_input = all_input + 1 WHERE item_id = 'max'").run();
    expect(() => queryNativeUsage(db, { period: 7, now: NOW })).toThrow("aggregate-overflow");
    db.close();
  });

  it("reduces transfer to returned day and facet groups instead of items", () => {
    const db = openDb(":memory:");
    const insert = db.prepare(
      `INSERT INTO native_usage_items
         (harness, item_id, session_id, provenance, agent, model, ts, updated_ts, day, all_input, all_output, cost_amount, cost_currency)
       VALUES ('opencode', ?, ?, 'v2', ?, ?, ?, ?, ?, 100, 20, ?, ?)`,
    );
    db.transaction(() => {
      for (let i = 0; i < 2000; i += 1) {
        insert.run(
          `item-${i}`, `sess-${i}`,
          `agent-${i % 3}`, `model-${i % 4}`,
          NOW, NOW, utcDay(NOW - (i % 5) * DAY),
          1.5, i % 2 === 0 ? "USD" : null,
        );
      }
    })();
    const original = db.prepare.bind(db);
    let transferred = 0;
    (db as unknown as { prepare: typeof db.prepare }).prepare = ((sql: string) => {
      const statement = original(sql);
      if (/GROUP BY/i.test(sql) && /native_usage_items/i.test(sql)) {
        const all = statement.all.bind(statement);
        statement.all = ((...args: unknown[]) => {
          const rows = all(...args) as unknown[];
          transferred += rows.length;
          return rows;
        }) as typeof statement.all;
      }
      return statement;
    }) as typeof db.prepare;
    // Fixture observation: 2000 items collapse to 5 day groups + facet groups; the
    // measured row transfer stays far below the item count (groups, not items).
    const snap = queryNativeUsage(db, { period: 7, now: NOW });
    expect(snap.totalTokens).toBe(2000 * 120);
    expect(transferred).toBeLessThan(100);
    expect(db.prepare).not.toBe(original);
    db.close();
  });
});
