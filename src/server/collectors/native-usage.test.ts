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
    expect(mergeNativeItem(a, b).identityConflict).toBe(true);
    expect(mergeNativeItem(b, a).identityConflict).toBe(true);
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

// ---------------------------------------------------------------------------
// Bounded collection and the OpenCode helper boundary.

import {
  mkdirSync,
  mkdtempSync,
  openSync,
  readSync,
  closeSync,
  realpathSync as fsRealpathSync,
  readdirSync as fsReaddirSync,
  rmSync,
  statSync as fsStatSync,
  symlinkSync,
  truncateSync,
  writeFileSync,
  appendFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import { runInNewContext } from "node:vm";
import BetterSqlite3 from "better-sqlite3";
import { parseSettings, type SettingsResult } from "../settings";
import { getSourceStatuses, queryNativeUsage } from "../db/native-usage";
import { openDb } from "../db";
import {
  defaultNativeCollectorDeps,
  discoverJsonlFiles,
  OPENCODE_EXECUTABLE_SHA256,
  OPENCODE_HELPER_SOURCE,
  runNativeUsagePass,
  type NativeCollectorDeps,
  type NativeFs,
  type StatLike,
} from "./native-usage";

const VERIFY_HOME = process.env.JAXOS_HOME || join(process.cwd(), ".local/verify-home");
mkdirSync(VERIFY_HOME, { recursive: true });

const nativeFs: NativeFs = {
  statSync: (p) => fsStatSync(p),
  realpathSync: (p) => fsRealpathSync(p),
  readdirSync: (p) => fsReaddirSync(p, { withFileTypes: true }) as never,
  openSync: (p) => openSync(p, "r"),
  readSync: (fd, buffer, offset, length, position) => readSync(fd, buffer, offset, length, position),
  closeSync: (fd) => closeSync(fd),
};

type MakeDepsOptions = {
  enabled?: { claude: boolean; codex: boolean; opencode: boolean };
  settings?: () => SettingsResult;
  fs?: NativeFs;
  exec?: NativeCollectorDeps["opencode"]["exec"];
  stat?: (path: string) => StatLike | null;
  hash?: (path: string) => string | null;
  base?: string;
  nowOffset?: number;
};

function makeDeps(opts: MakeDepsOptions = {}): NativeCollectorDeps {
  const base = opts.base ?? mkdtempSync(join(VERIFY_HOME, "native-collector-"));
  const claude = join(base, "claude");
  const codex = join(base, "codex");
  const data = join(base, "opencode-data");
  mkdirSync(claude, { recursive: true });
  mkdirSync(codex, { recursive: true });
  mkdirSync(data, { recursive: true });
  const enabled = opts.enabled ?? { claude: true, codex: true, opencode: true };
  const hash = opts.hash ?? (() => OPENCODE_EXECUTABLE_SHA256);
  const stat = opts.stat ?? (() => ({
    dev: 1, ino: 1, size: 100, mtimeMs: Date.parse("2026-11-18T10:00:00Z"),
    isFile: () => true, isDirectory: () => false, isSymbolicLink: () => false,
  }));
  let clock = Date.parse("2026-11-18T12:00:00Z") + (opts.nowOffset ?? 0);
  const emptyReply = JSON.stringify({
    v: 1, ok: true, runtime: "Bun 1.3.14", schema: "1.18.34", rows: [],
    next: { v1: { after: 0, ceiling: 0, done: true }, v2: { after: 0, ceiling: 0, done: true } },
    warnings: {},
  });
  return {
    db: openDb(":memory:"),
    now: () => Math.floor(clock / 1000),
    nowMs: () => (clock += 1),
    settings: opts.settings ?? (() => parseSettings({ integrations: { agents: enabled } })),
    fs: opts.fs ?? nativeFs,
    roots: {
      claude,
      codex,
      opencodeDataRoot: data,
      opencodeDb: join(data, "opencode.db"),
    },
    opencode: {
      executable: join(base, "fake-opencode"),
      cwd: join(base, "adapter-cwd"),
      configPath: join(base, "adapter-empty.toml"),
      xdgConfigHome: join(base, "adapter-config"),
      exec: opts.exec ?? (async () => ({ stdout: emptyReply })),
      stat,
      hash,
    },
  };
}

function writeClaude(deps: NativeCollectorDeps, rel: string, lines: string[]): string {
  const path = join(deps.roots.claude, rel);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, lines.join("\n") + "\n");
  return path;
}

const claudeLine = (id: string, extra: Record<string, unknown> = {}, usage: Record<string, unknown> = {}) =>
  JSON.stringify({
    type: "assistant",
    sessionId: "sess-1",
    timestamp: "2026-11-17T10:00:00Z",
    message: { id, model: "claude-sonnet-5-5", usage: { input_tokens: 10, output_tokens: 5, ...usage } },
    ...extra,
  });

const codexLine = (responseId: string, threadId = "thread-1") =>
  JSON.stringify({
    type: "token_usage_record",
    timestamp: "2026-11-17T10:00:00Z",
    payload: {
      session_id: "root-1", thread_id: threadId, turn_id: "turn-1", root_turn_id: "turn-1",
      response_id: responseId,
      usage: { input_tokens: 7, output_tokens: 3, total_tokens: 10 },
    },
  });

describe("bounded JSONL collection", () => {
  it("collects lead and explicit child layouts, dedupes mirrors and warns unsupported layouts", async () => {
    const deps = makeDeps();
    writeClaude(deps, "proj-a/s1.jsonl", [claudeLine("msg-1")]);
    writeClaude(deps, "proj-b/s2.jsonl", [claudeLine("msg-1")]);
    writeClaude(deps, "proj-a/child/subagents/agent-1.jsonl", [claudeLine("msg-2", { agentId: "agent-1" })]);
    writeFileSync(join(deps.roots.claude, "proj-a", "notes.txt"), "not jsonl");
    await runNativeUsagePass(deps);
    const snap = queryNativeUsage(deps.db, { period: 30, now: NOW });
    expect(snap.totalTokens).toBe(30);
    expect(snap.sessions).toEqual({ lead: 1, child: 1 });
    expect(snap.sources.find((s) => s.source === "claude")?.state).toBe("available");
    expect(snap.warnings).toContain("unsupported-layout");
    deps.db.close();
  });

  it("rejects a symlink escaping the native root", async () => {
    const deps = makeDeps();
    const outside = join(deps.roots.claude, "..", "outside.jsonl");
    writeFileSync(outside, claudeLine("msg-outside") + "\n");
    mkdirSync(join(deps.roots.claude, "proj"), { recursive: true });
    symlinkSync(outside, join(deps.roots.claude, "proj", "evil.jsonl"));
    await runNativeUsagePass(deps);
    expect(queryNativeUsage(deps.db, { period: 30, now: NOW }).totalTokens).toBe(0);
    expect(getSourceStatuses(deps.db).find((s) => s.source === "claude")?.warnings).toContain("symlink-escape");
    deps.db.close();
  });

  it("caps files per pass, completes across passes, handles append and truncation idempotently", async () => {
    const deps = makeDeps();
    const files = Array.from({ length: 8 }, (_, i) => writeClaude(deps, `p/f${i}.jsonl`, [claudeLine(`msg-${i}`)]));
    await runNativeUsagePass(deps);
    let status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
    expect(status?.state).toBe("indexing");
    expect(status?.progress?.filesDone).toBeLessThanOrEqual(6);
    await runNativeUsagePass(deps);
    status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
    expect(status?.state).toBe("available");
    expect(queryNativeUsage(deps.db, { period: 30, now: NOW }).sessions.lead).toBe(1);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 8 });

    appendFileSync(files[0], claudeLine("msg-appended", { sessionId: "sess-2" }) + "\n");
    await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 9 });

    truncateSync(files[1], 0);
    await runNativeUsagePass({ ...deps, now: () => NOW + 122 });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 9 });
    deps.db.close();
  });

  it("checkpoints before an incomplete trailing line and completes it on the next pass", async () => {
    const deps = makeDeps();
    const complete = claudeLine("msg-complete");
    const partial = JSON.stringify({
      type: "assistant", sessionId: "sess-1", timestamp: "2026-11-17T11:00:00Z",
      message: { id: "msg-partial", usage: { input_tokens: 2, output_tokens: 2 } },
    });
    const path = join(deps.roots.claude, "p/partial.jsonl");
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${complete}\n${partial.slice(0, 20)}`);
    await runNativeUsagePass(deps);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    appendFileSync(path, `${partial.slice(20)}\n`);
    await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 2 });
    deps.db.close();
  });

  it("skips an overlength line and still indexes the following valid line", async () => {
    const deps = makeDeps();
    writeClaude(deps, "p/long.jsonl", ["x".repeat(1024 * 1024 + 10), claudeLine("msg-after-long")]);
    await runNativeUsagePass(deps);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    expect(getSourceStatuses(deps.db).find((s) => s.source === "claude")?.warnings).toContain("line-too-long");
    deps.db.close();
  });

  it("isolates a per-file read failure while the healthy source and file still advance", async () => {
    const deps = makeDeps();
    const good = writeClaude(deps, "p/good.jsonl", [claudeLine("msg-good")]);
    writeClaude(deps, "p/bad.jsonl", [claudeLine("msg-bad")]);
    const failingFs: NativeFs = {
      ...nativeFs,
      openSync: (path) => {
        if (path.endsWith("bad.jsonl")) throw new Error("EACCES");
        return openSync(path, "r");
      },
    };
    deps.fs = failingFs;
    await runNativeUsagePass(deps);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    const status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
    expect(status?.warnings).toContain("read-failed");
    expect(status?.state).toBe("indexing");
    deps.db.close();
  });

  it("does not touch any native root when the source is disabled", async () => {
    const deps = makeDeps({ enabled: { claude: false, codex: false, opencode: false } });
    const spy: NativeFs = {
      ...nativeFs,
      statSync: () => { throw new Error("must not stat"); },
      realpathSync: () => { throw new Error("must not realpath"); },
      readdirSync: () => { throw new Error("must not readdir"); },
    };
    deps.fs = spy;
    await runNativeUsagePass(deps);
    expect(getSourceStatuses(deps.db).map((s) => s.state)).toEqual(["disabled", "disabled", "disabled"]);
    deps.db.close();
  });

  it("reports an unreadable settings source as unavailable, never disabled", async () => {
    const deps = makeDeps({ settings: () => ({ ok: false, error: "settings-unreadable" }) });
    await runNativeUsagePass(deps);
    const statuses = getSourceStatuses(deps.db);
    expect(statuses.every((s) => s.state === "unavailable" && s.warnings.includes("settings-error"))).toBe(true);
    deps.db.close();
  });
});

describe("startup-only synthetic native roots", () => {
  it("accepts roots beneath JAXOS_HOME and fails closed on an escaping root", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "roots-"));
    const claude = join(base, "claude");
    const codex = join(base, "codex");
    const opencode = join(base, "opencode");
    mkdirSync(claude, { recursive: true });
    mkdirSync(codex, { recursive: true });
    mkdirSync(opencode, { recursive: true });
    writeFileSync(join(base, "native-roots.json"), JSON.stringify({ claude, codex, opencode }));
    const previous = process.env.JAXOS_HOME;
    process.env.JAXOS_HOME = base;
    try {
      const db = openDb(":memory:");
      const deps = defaultNativeCollectorDeps(db);
      expect(deps.roots).toMatchObject({
        claude, codex, opencodeDataRoot: opencode, opencodeDb: join(opencode, "opencode.db"),
      });
      db.close();
      writeFileSync(join(base, "native-roots.json"), JSON.stringify({ claude: "/tmp", codex, opencode }));
      const escapedDb = openDb(":memory:");
      const escaped = defaultNativeCollectorDeps(escapedDb);
      expect(escaped.roots.claude).toContain("native-invalid");
      expect(escaped.roots.claude).not.toBe("/tmp");
      escapedDb.close();
    } finally {
      if (previous === undefined) delete process.env.JAXOS_HOME;
      else process.env.JAXOS_HOME = previous;
      rmSync(base, { recursive: true, force: true });
    }
  });
});

describe("Codex discovery", () => {
  it("accepts only the sessions/YYYY/MM/DD rollout layout", () => {
    const deps = makeDeps();
    const dir = join(deps.roots.codex, "2026", "11", "17");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "rollout-2026-11-17.jsonl"), codexLine("r1"));
    writeFileSync(join(dir, "not-a-rollout.jsonl"), codexLine("r2"));
    const found = discoverJsonlFiles(nativeFs, "codex", deps.roots.codex);
    expect(found.files.map((f) => f.identity)).toEqual(["2026/11/17/rollout-2026-11-17.jsonl"]);
    expect(found.warnings).toContain("unsupported-layout");
    deps.db.close();
  });

  it("collects a Codex rollout and resolves its model from a turn context", async () => {
    const deps = makeDeps();
    const dir = join(deps.roots.codex, "2026", "11", "17");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "rollout-x.jsonl"), [
      JSON.stringify({ type: "turn_context", timestamp: "2026-11-17T09:00:00Z", payload: { turn_id: "turn-1", root_turn_id: "turn-1", model: "gpt-5.6-luna" } }),
      codexLine("r1"),
    ].join("\n") + "\n");
    await runNativeUsagePass(deps);
    const snap = queryNativeUsage(deps.db, { period: 30, now: NOW });
    expect(snap.totalTokens).toBe(10);
    expect(snap.byModel.map((m) => m.name)).toEqual(["gpt-5.6-luna"]);
    deps.db.close();
  });
});

function runHelper(dbPath: string, state: unknown): { reply: Record<string, unknown>; opts: Record<string, unknown> } {
  let stdout = "";
  let opts: Record<string, unknown> = {};
  class ShimDatabase {
    db: BetterSqlite3.Database;
    constructor(path: string, o: { readonly?: boolean; strict?: boolean }) {
      opts = o as Record<string, unknown>;
      this.db = new BetterSqlite3(path, { readonly: o.readonly === true, fileMustExist: true });
    }
    prepare(sql: string) {
      return this.db.prepare(sql);
    }
  }
  const sandbox = {
    process: {
      argv: ["bun", "--no-env-file", "x", "-e", OPENCODE_HELPER_SOURCE, dbPath, JSON.stringify(state)],
      stdout: { write: (chunk: string) => { stdout += chunk; } },
      exit: () => { throw new Error("__exit__"); },
      versions: { bun: "1.3.14" },
    },
    require: (name: string) => {
      if (name === "bun:sqlite") return { Database: ShimDatabase };
      throw new Error("unexpected require");
    },
    Buffer,
    JSON,
    isFinite,
    Error,
  };
  try {
    runInNewContext(OPENCODE_HELPER_SOURCE, sandbox);
  } catch (error) {
    if ((error as Error).message !== "__exit__") throw error;
  }
  return { reply: JSON.parse(stdout) as Record<string, unknown>, opts };
}

function nativeOpenCodeDb(base: string, rows = 300): string {
  const path = join(base, "opencode.db");
  const db = new BetterSqlite3(path);
  db.exec(`
    CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT);
    CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, time_created INTEGER, time_updated INTEGER, data TEXT);
    CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT, seq INTEGER, time_created INTEGER, time_updated INTEGER, data TEXT);
    CREATE TABLE migration (id INTEGER PRIMARY KEY);
  `);
  db.prepare("INSERT INTO session (id, parent_id) VALUES ('ses-1', 'ses-parent')").run();
  const insertV1 = db.prepare("INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, 'ses-1', ?, ?, ?)");
  for (let i = 1; i <= rows; i += 1) {
    insertV1.run(`m${i}`, 1_794_909_600_000, 1_794_909_700_000, JSON.stringify({
      role: "assistant", modelID: "model-x", providerID: "provider-x", agent: "build",
      created: 1_794_909_600_000, cost: 1.5,
      tokens: { input: 100, output: 30, reasoning: 7, read: 11, write: 5 },
      content: "SENTINEL_SECRET_TEXT",
    }));
  }
  db.prepare("INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data) VALUES ('v2-1', 'ses-1', 'assistant', 1, ?, ?, ?)").run(
    1_794_909_600_000, 1_794_909_800_000,
    JSON.stringify({
      type: "assistant", agent: "build", model: { id: "model-y", providerID: "provider-y" },
      time: { created: 1_794_909_600_000, completed: 1_794_909_700_000 }, cost: 2.5,
      tokens: { input: 100, output: 30, reasoning: 7, cache: { read: 11, write: 5 } },
      content: "SENTINEL_SECRET_TEXT",
    }),
  );
  db.close();
  return path;
}

const freshCursor = { v: 1, v1: { after: 0, ceiling: 0, done: false }, v2: { after: 0, ceiling: 0, done: false } };

describe("fixed OpenCode helper (NodeVM with a bun:sqlite shim)", () => {
  it("pages both projections by rowid, stays readonly and emits no message text", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-"));
    const path = nativeOpenCodeDb(base);
    const first = runHelper(path, freshCursor);
    expect(first.opts).toEqual({ readonly: true, strict: true });
    const reply = first.reply as {
      ok: boolean; rows: { projection: string }[]; next: { v1: { after: number; done: boolean }; v2: { done: boolean } };
      warnings: Record<string, number>;
    };
    expect(reply.ok).toBe(true);
    expect(reply.rows.filter((r) => r.projection === "v1")).toHaveLength(256);
    expect(reply.rows.filter((r) => r.projection === "v2")).toHaveLength(1);
    expect(reply.next.v1.done).toBe(false);
    expect(reply.next.v2.done).toBe(true);
    expect(JSON.stringify(reply)).not.toContain("SENTINEL_SECRET_TEXT");

    const second = runHelper(path, { v: 1, v1: reply.next.v1, v2: reply.next.v2 });
    const secondReply = second.reply as { rows: { projection: string }[]; next: { v1: { done: boolean } } };
    expect(secondReply.rows.filter((r) => r.projection === "v1")).toHaveLength(44);
    expect(secondReply.next.v1.done).toBe(true);

    const readonly = new BetterSqlite3(path, { readonly: true, fileMustExist: true });
    expect(() => readonly.prepare("CREATE TABLE nope (x)").run()).toThrow();
    readonly.close();
    rmSync(base, { recursive: true, force: true });
  });

  it("stops before the 240KiB row budget and checkpoints the consumed rows", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-cap-"));
    const path = join(base, "cap.db");
    const db = new BetterSqlite3(path);
    db.exec(`
      CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT);
      CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, time_created INTEGER, time_updated INTEGER, data TEXT);
      CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT, seq INTEGER, time_created INTEGER, time_updated INTEGER, data TEXT);
      CREATE TABLE migration (id INTEGER PRIMARY KEY);
      INSERT INTO session (id, parent_id) VALUES ('ses-1', NULL);
    `);
    const insert = db.prepare("INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, 'ses-1', 1, 1, ?)");
    db.transaction(() => {
      for (let i = 1; i <= 300; i += 1) {
        insert.run(`m${i}`, JSON.stringify({
          role: "assistant", modelID: "model-x", agent: "a".repeat(3000),
          created: 1_794_909_600_000, tokens: { input: 1, output: 1 },
        }));
      }
    })();
    db.close();
    const reply = runHelper(path, freshCursor).reply as {
      ok: boolean; rows: unknown[]; next: { v1: { after: number; ceiling: number; done: boolean } };
    };
    expect(reply.ok).toBe(true);
    expect(reply.rows.length).toBeGreaterThan(0);
    expect(reply.rows.length).toBeLessThan(300);
    expect(reply.next.v1.after).toBeGreaterThan(0);
    expect(reply.next.v1.after).toBeLessThan(reply.next.v1.ceiling);
    expect(reply.next.v1.done).toBe(false);
    expect(Buffer.byteLength(JSON.stringify(reply), "utf8")).toBeLessThanOrEqual(262144);
    rmSync(base, { recursive: true, force: true });
  });

  it("fails closed on a missing table, bad state and unsupported schema", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-"));
    const path = join(base, "empty.db");
    const empty = new BetterSqlite3(path);
    empty.exec("CREATE TABLE migration (id INTEGER PRIMARY KEY);");
    empty.close();
    expect(runHelper(path, freshCursor).reply).toEqual({ v: 1, ok: false, error: "schema-fail-closed" });
    expect(runHelper(path, { v: 2 }).reply).toEqual({ v: 1, ok: false, error: "bad-state" });
    rmSync(base, { recursive: true, force: true });
  });

  it("advances a malformed row's cursor while excluding its content", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-"));
    const path = nativeOpenCodeDb(base, 1);
    const db = new BetterSqlite3(path);
    db.prepare("INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES ('bad', 'ses-1', 1, 1, '{not-json')").run();
    db.close();
    const reply = runHelper(path, freshCursor).reply as { warnings: Record<string, number>; next: { v1: { after: number } } };
    expect(reply.warnings.malformed).toBe(1);
    expect(reply.next.v1.after).toBeGreaterThan(0);
    rmSync(base, { recursive: true, force: true });
  });
});

describe("OpenCode source pass through the injected executor", () => {
  const v2Row = {
    projection: "v2", session_id: "ses-1", message_id: "msg-1", parent_session_id: null,
    model_id: "model-x", provider_id: "provider-x", agent: "build",
    created: 1_794_909_600_000, updated: 1_794_909_800_000,
    cost: 12.5, input: 100, output: 30, reasoning: 7, cache_read: 11, cache_write: 5,
  };
  const reply = JSON.stringify({
    v: 1, ok: true, runtime: "Bun 1.3.14", schema: "1.18.34", rows: [v2Row],
    next: { v1: { after: 0, ceiling: 0, done: true }, v2: { after: 1, ceiling: 1, done: true } },
    warnings: {},
  });

  function withDbFile(opts: MakeDepsOptions = {}): NativeCollectorDeps {
    const deps = makeDeps(opts);
    writeFileSync(deps.roots.opencodeDb, "");
    return deps;
  }

  it("passes the exact fixed argv, cwd and narrow environment, and persists real usage", async () => {
    let seen: { file: string; args: string[]; opts: { cwd: string; env: Record<string, string> } } | null = null;
    const deps = withDbFile({
      exec: async (file, args, execOpts) => {
        seen = { file, args, opts: execOpts };
        return { stdout: reply };
      },
    });
    await runNativeUsagePass(deps);
    expect(seen).not.toBeNull();
    const call = seen as unknown as { file: string; args: string[]; opts: { cwd: string; env: Record<string, string> } };
    expect(call.file).toBe(deps.opencode.executable);
    expect(call.args[0]).toBe("--no-env-file");
    expect(call.args[1]).toBe(`--config=${deps.opencode.configPath}`);
    expect(call.args[2]).toBe(`--cwd=${deps.opencode.cwd}`);
    expect(call.args[3]).toBe("-e");
    expect(call.args[4]).toBe(OPENCODE_HELPER_SOURCE);
    expect(call.args[5]).toBe(fsRealpathSync(deps.roots.opencodeDb));
    expect(JSON.parse(call.args[6])).toEqual(freshCursor);
    expect(call.opts.cwd).toBe(deps.opencode.cwd);
    expect(call.opts.env).toEqual({ BUN_BE_BUN: "1", XDG_CONFIG_HOME: deps.opencode.xdgConfigHome });
    const snap = queryNativeUsage(deps.db, { period: 30, now: NOW });
    expect(snap.totalTokens).toBe(153);
    expect(snap.sources.find((s) => s.source === "opencode")?.state).toBe("available");
    deps.db.close();
  });

  it("treats a changed executable identity with a mismatched hash as incompatible, retaining last-good rows", async () => {
    let calls = 0;
    let hashCalls = 0;
    let size = 100;
    const deps = withDbFile({
      exec: async () => { calls += 1; return { stdout: reply }; },
      stat: () => ({
        dev: 1, ino: 1, size, mtimeMs: Date.parse("2026-11-18T10:00:00Z"),
        isFile: () => true, isDirectory: () => false, isSymbolicLink: () => false,
      }),
      hash: () => { hashCalls += 1; return size === 100 ? OPENCODE_EXECUTABLE_SHA256 : "deadbeef"; },
    });
    await runNativeUsagePass(deps);
    expect(calls).toBe(1);
    expect(hashCalls).toBe(1);
    size = 101;
    await runNativeUsagePass({ ...deps, now: () => Math.floor((Date.parse("2026-11-18T13:00:00Z")) / 1000) });
    const status = getSourceStatuses(deps.db).find((s) => s.source === "opencode");
    expect(status?.state).toBe("incompatible");
    expect(status?.warnings).toContain("executable-mismatch");
    expect(queryNativeUsage(deps.db, { period: 30, now: NOW }).totalTokens).toBe(153);
    deps.db.close();
  });

  it("reports a missing native database without touching other sources", async () => {
    const deps = makeDeps({ exec: async () => { throw new Error("must not run"); } });
    await runNativeUsagePass(deps);
    const status = getSourceStatuses(deps.db).find((s) => s.source === "opencode");
    expect(status?.state).toBe("unavailable");
    expect(status?.warnings).toContain("db-missing");
    deps.db.close();
  });

  it("isolates helper schema failure as incompatible and does not throw", async () => {
    const deps = withDbFile({
      exec: async () => ({ stdout: JSON.stringify({ v: 1, ok: false, error: "schema-fail-closed" }) }),
    });
    await runNativeUsagePass(deps);
    const status = getSourceStatuses(deps.db).find((s) => s.source === "opencode");
    expect(status).toMatchObject({ state: "incompatible" });
    expect(status?.warnings).toContain("schema-fail-closed");
    deps.db.close();
  });

  it("skips a completed source until its 60s refresh cadence elapses", async () => {
    let calls = 0;
    const deps = withDbFile({ exec: async () => { calls += 1; return { stdout: reply }; } });
    await runNativeUsagePass(deps);
    const later = { ...deps, now: () => Math.floor(Date.parse("2026-11-18T12:00:30Z") / 1000) };
    await runNativeUsagePass(later);
    expect(calls).toBe(1);
    const afterMinute = { ...deps, now: () => Math.floor(Date.parse("2026-11-18T12:01:01Z") / 1000) };
    await runNativeUsagePass(afterMinute);
    expect(calls).toBe(2);
    deps.db.close();
  });

  it("does not let a malformed reply advance or delete last-good state", async () => {
    const deps = withDbFile({ exec: async () => ({ stdout: "not-json" }) });
    await runNativeUsagePass(deps);
    const status = getSourceStatuses(deps.db).find((s) => s.source === "opencode");
    expect(status?.state).toBe("unavailable");
    expect(status?.warnings).toContain("malformed-reply");
    deps.db.close();
  });
});


describe("Fixture A: 50k rows per projection table", () => {
  it("completes one captured cycle within 200 successful helper passes", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-cycle-"));
    const path = join(base, "cycle.db");
    const db = new BetterSqlite3(path);
    db.exec(`
      CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT);
      CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, time_created INTEGER, time_updated INTEGER, data TEXT);
      CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT, seq INTEGER, time_created INTEGER, time_updated INTEGER, data TEXT);
      CREATE TABLE migration (id INTEGER PRIMARY KEY);
      INSERT INTO session (id, parent_id) VALUES ('ses-1', NULL);
    `);
    const v1 = db.prepare("INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, 'ses-1', ?, ?, ?)");
    const v2 = db.prepare("INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data) VALUES (?, 'ses-1', 'assistant', ?, ?, ?, ?)");
    const v1Data = JSON.stringify({ role: "assistant", modelID: "model-x", agent: "build", created: 1_794_909_600_000, tokens: { input: 10, output: 5, reasoning: 1, read: 2, write: 1 } });
    const v2Data = JSON.stringify({ type: "assistant", agent: "build", model: { id: "model-y", providerID: "p" }, time: { created: 1_794_909_600_000 }, tokens: { input: 10, output: 5, reasoning: 1, cache: { read: 2, write: 1 } } });
    db.transaction(() => {
      for (let i = 1; i <= 50_000; i += 1) {
        v1.run(`m${i}`, 1_794_909_600_000, 1_794_909_600_000, v1Data);
        v2.run(`s${i}`, i, 1_794_909_600_000, 1_794_909_600_000, v2Data);
      }
    })();
    db.close();

    let state: { v: 1; v1: { after: number; ceiling: number; done: boolean }; v2: { after: number; ceiling: number; done: boolean } } = freshCursor as never;
    let passes = 0;
    let replied = 0;
    do {
      const out = runHelper(path, state).reply as {
        ok: boolean;
        rows: unknown[];
        next?: { v1: typeof state.v1; v2: typeof state.v2 };
      };
      expect(out.ok).toBe(true);
      replied += out.rows.length;
      state = { v: 1, v1: out.next!.v1, v2: out.next!.v2 };
      passes += 1;
      expect(passes).toBeLessThanOrEqual(200);
    } while (!(state.v1.done && state.v2.done));
    expect(passes).toBeLessThanOrEqual(200);
    expect(state.v1.after).toBe(50_000);
    expect(state.v2.after).toBe(50_000);
    expect(replied).toBe(100_000);
    rmSync(base, { recursive: true, force: true });
  }, 120_000);
});
