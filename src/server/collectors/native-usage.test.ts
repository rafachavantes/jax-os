import { afterEach, describe, expect, it, vi } from "vitest";
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

  it("labels the harness OpenCode and keeps the provider in the model", () => {
    const out = parseOpenCodeRow(opencode(), { now: NOW });
    if (!("item" in out)) throw new Error("expected item");
    expect(out.item.agent).toBe("OpenCode");
    expect(out.item.model).toBe("provider-x/model-x");
    expect(out.item.rawModel).toBe("provider-x/model-x");
    const bare = parseOpenCodeRow(opencode({ provider_id: undefined }), { now: NOW });
    if (!("item" in bare)) throw new Error("expected item");
    expect(bare.item.model).toBe("model-x");
    expect(bare.item.rawModel).toBe("model-x");
    const noModel = parseOpenCodeRow(opencode({ model_id: null }), { now: NOW });
    if (!("item" in noModel)) throw new Error("expected item");
    expect(noModel.item.model).toBeNull();
    expect(noModel.item.rawModel).toBeNull();
  });

  it("keeps a valid cwd and drops malformed ones", () => {
    const cwdOf = (over: Record<string, unknown>) => {
      const out = parseOpenCodeRow(opencode(over), { now: NOW });
      if (!("item" in out)) throw new Error("expected item");
      return out.item.cwd;
    };
    expect(cwdOf({ cwd: "/home/rafa/repos/demo" })).toBe("/home/rafa/repos/demo");
    expect(cwdOf({ projection: "v1", cwd: "/home/rafa/repos/demo-v1" })).toBe("/home/rafa/repos/demo-v1");
    expect(cwdOf({ cwd: 42 })).toBeNull();
    expect(cwdOf({ cwd: "x".repeat(600) })).toBeNull();
    expect(cwdOf({ cwd: "bad\u0000cwd" })).toBeNull();
    expect(cwdOf({ cwd: undefined })).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Bounded collection and the OpenCode helper boundary.

import {
  mkdirSync,
  mkdtempSync,
  realpathSync as fsRealpathSync,
  rmSync,
  statSync as fsStatSync,
  symlinkSync,
  truncateSync,
  writeFileSync,
  appendFileSync,
  utimesSync,
  chmodSync,
} from "node:fs";
import {
  open as fsOpenAsync,
  opendir as fsOpendirAsync,
  realpath as fsRealpathAsync,
  stat as fsStatAsync,
} from "node:fs/promises";
import { dirname, join } from "node:path";
import { runInNewContext } from "node:vm";
import BetterSqlite3 from "better-sqlite3";
import { parseSettings, type SettingsResult } from "../settings";
import {
  applyFileBatch,
  getFileCheckpoint,
  getOpenCodeCursor,
  getPersistentWarnings,
  getSourceStatuses,
  markFileFailure,
  queryNativeUsage,
} from "../db/native-usage";
import { openDb } from "../db";
import {
  defaultNativeCollectorDeps,
  hashFileStreaming,
  OPENCODE_EXECUTABLE_SHA256,
  OPENCODE_HELPER_SOURCE,
  parseOpenCodeReply,
  runNativeUsagePass,
  type DirEntryLike,
  type DirHandleLike,
  type NativeCollectorDeps,
  type NativeFs,
  type StatLike,
} from "./native-usage";

const VERIFY_HOME = process.env.JAXOS_HOME || join(process.cwd(), ".local/verify-home");
mkdirSync(VERIFY_HOME, { recursive: true });

// Fixture-owned directory iterators: open handles are closed at test teardown so a
// deferred or unfinished read never leaks a GC close warning across tests.
const ownedDirHandles = new Set<DirHandleLike>();
function trackedDir(handle: DirHandleLike): DirHandleLike {
  const tracked: DirHandleLike = {
    read: () => handle.read(),
    close: async () => {
      ownedDirHandles.delete(tracked);
      await handle.close();
    },
  };
  ownedDirHandles.add(tracked);
  return tracked;
}
afterEach(async () => {
  const handles = [...ownedDirHandles];
  ownedDirHandles.clear();
  for (const handle of handles) {
    try {
      await handle.close();
    } catch {
      /* teardown only: the fixture-owned handle is already unusable */
    }
  }
});

const nativeFs: NativeFs = {
  stat: (p) => fsStatAsync(p),
  realpath: (p) => fsRealpathAsync(p),
  opendir: async (p) => trackedDir((await fsOpendirAsync(p)) as unknown as DirHandleLike),
  open: async (p) => {
    const handle = await fsOpenAsync(p, "r");
    return {
      read: (buffer, offset, length, position) => handle.read(buffer, offset, length, position),
      close: () => handle.close(),
      stat: () => handle.stat(),
    };
  },
};

type MakeDepsOptions = {
  enabled?: { claude: boolean; codex: boolean; opencode: boolean };
  settings?: () => SettingsResult;
  fs?: NativeFs;
  exec?: NativeCollectorDeps["opencode"]["exec"];
  stat?: (path: string) => Promise<StatLike | null>;
  hash?: (path: string) => Promise<string | null>;
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
  const hash = opts.hash ?? (async () => OPENCODE_EXECUTABLE_SHA256);
  const stat = opts.stat ?? (async () => ({
    dev: 1, ino: 1, size: 100, mtimeMs: Date.parse("2026-11-18T10:00:00Z"),
    isFile: () => true, isDirectory: () => false, isSymbolicLink: () => false,
  }));
  let clock = Date.parse("2026-11-18T12:00:00Z") + (opts.nowOffset ?? 0);
  const emptyReply = JSON.stringify({
    v: 1, ok: true, runtime: "Bun 1.3.14", schema: "1.18.34", rows: [],
    next: { v1: { after: null, ceiling: null, done: true }, v2: { after: null, ceiling: null, done: true } },
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
    const seenWarnings = new Set<string>();
    for (let pass = 0; pass < 6; pass += 1) {
      await runNativeUsagePass(deps);
      const status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
      for (const warning of status?.warnings ?? []) seenWarnings.add(warning);
      if (status?.state === "available") break;
    }
    const snap = queryNativeUsage(deps.db, { period: 30, now: NOW });
    expect(snap.totalTokens).toBe(30);
    expect(snap.sessions).toEqual({ lead: 1, child: 1 });
    expect(snap.sources.find((s) => s.source === "claude")?.state).toBe("available");
    expect([...seenWarnings]).toContain("unsupported-layout");
    deps.db.close();
  });

  it("skips Claude subagent meta files silently while an unknown file still warns", async () => {
    const run = async (extra: string) => {
      const deps = makeDeps();
      writeClaude(deps, "p/session/subagents/agent-x.jsonl", [claudeLine("m1", { agentId: "agent-x" })]);
      writeFileSync(join(deps.roots.claude, "p", "session", "subagents", extra), "{}");
      const warnings = new Set<string>();
      for (let pass = 0; pass < 6; pass += 1) {
        await runNativeUsagePass(deps);
        const status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
        for (const w of status?.warnings ?? []) warnings.add(w);
        if (status?.state === "available") break;
      }
      const child = deps.db
        .prepare("SELECT COUNT(*) AS n FROM native_usage_files WHERE source = 'claude' AND child = 1")
        .get() as { n: number };
      deps.db.close();
      return { warnings, child: child.n };
    };
    const meta = await run("agent-x.meta.json");
    expect(meta.warnings).not.toContain("unsupported-layout");
    expect(meta.child).toBe(1);
    expect((await run("other.txt")).warnings).toContain("unsupported-layout");
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
    for (let pass = 0; pass < 8; pass += 1) {
      status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
      if (status?.state === "available") break;
      await runNativeUsagePass(deps);
    }
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

  it("advances a scalar generation when the same inode is truncated", async () => {
    const deps = makeDeps();
    const path = writeClaude(deps, "p/one.jsonl", [claudeLine("m-one")]);
    for (let pass = 0; pass < 4; pass += 1) {
      await runNativeUsagePass({ ...deps, now: () => NOW + pass * 61 });
      if (getFileCheckpoint(deps.db, "claude", "p/one.jsonl")?.done) break;
    }
    const before = getFileCheckpoint(deps.db, "claude", "p/one.jsonl");
    expect(before?.generation).toMatch(/^\d+:\d+$/);
    truncateSync(path, 0);
    await runNativeUsagePass({ ...deps, now: () => NOW + 61 * 6 });
    const after = getFileCheckpoint(deps.db, "claude", "p/one.jsonl");
    expect(after?.generation).toBe(`${before!.generation}:1`);
    expect(after?.offset).toBe(0);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
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
      open: async (path) => {
        if (path.endsWith("bad.jsonl")) throw new Error("EACCES");
        return nativeFs.open(path);
      },
    };
    deps.fs = failingFs;
    await runNativeUsagePass(deps);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    const status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
    expect(status?.warnings).toContain("read-failed");
    expect(status?.state).toBe("available");
    deps.db.close();
  });

  it("does not touch any native root when the source is disabled", async () => {
    const deps = makeDeps({ enabled: { claude: false, codex: false, opencode: false } });
    const spy: NativeFs = {
      ...nativeFs,
      stat: async () => { throw new Error("must not stat"); },
      realpath: async () => { throw new Error("must not realpath"); },
      opendir: async () => { throw new Error("must not opendir"); },
    };
    deps.fs = spy;
    deps.opencode.ensureAdapter = async () => { throw new Error("must not prepare"); };
    deps.opencode.exec = async () => { throw new Error("must not execute"); };
    deps.opencode.hash = async () => { throw new Error("must not hash"); };
    deps.opencode.stat = async () => { throw new Error("must not stat the executable"); };
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
  async function withFreshBoundary<T>(home: string, run: (mod: typeof import("./native-usage")) => Promise<T>): Promise<T> {
    const previous = process.env.JAXOS_HOME;
    process.env.JAXOS_HOME = home;
    vi.resetModules();
    try {
      const mod = await import("./native-usage");
      return await run(mod);
    } finally {
      vi.resetModules();
      if (previous === undefined) delete process.env.JAXOS_HOME;
      else process.env.JAXOS_HOME = previous;
    }
  }

  it("captures one immutable configuration for the whole process boundary", async () => {
    const base = mkdtempSync(join(VERIFY_HOME, "roots-"));
    const claude = join(base, "claude");
    const codex = join(base, "codex");
    const opencode = join(base, "opencode");
    mkdirSync(claude, { recursive: true });
    mkdirSync(codex, { recursive: true });
    mkdirSync(opencode, { recursive: true });
    writeFileSync(join(base, "native-roots.json"), JSON.stringify({ claude, codex, opencode }));
    try {
      await withFreshBoundary(base, async (mod) => {
        const db = openDb(":memory:");
        const deps = await mod.defaultNativeCollectorDeps(db);
        expect(deps.roots).toEqual({
          claude, codex, opencodeDataRoot: opencode, opencodeDb: join(opencode, "opencode.db"),
        });
        expect(deps.configWarnings).toEqual([]);
        writeFileSync(join(base, "native-roots.json"), JSON.stringify({ claude: "/tmp", codex, opencode }));
        const again = await mod.defaultNativeCollectorDeps(openDb(":memory:"));
        expect(again.roots).toEqual(deps.roots);
        db.close();
      });
    } finally {
      rmSync(base, { recursive: true, force: true });
    }
  });

  it("fails closed to owned invalid roots with a diagnostic for a present unusable override", async () => {
    const cases: { name: string; write: (dir: string) => void }[] = [
      { name: "malformed", write: (dir) => writeFileSync(join(dir, "native-roots.json"), "{not-json") },
      { name: "non-object", write: (dir) => writeFileSync(join(dir, "native-roots.json"), "[]") },
      { name: "relative", write: (dir) => writeFileSync(join(dir, "native-roots.json"), JSON.stringify({ claude: "rel", codex: "rel", opencode: "rel" })) },
      { name: "escaping", write: (dir) => writeFileSync(join(dir, "native-roots.json"), JSON.stringify({ claude: "/tmp", codex: dir, opencode: dir })) },
      {
        name: "unreadable",
        write: (dir) => {
          const path = join(dir, "native-roots.json");
          writeFileSync(path, JSON.stringify({ claude: dir, codex: dir, opencode: dir }));
          chmodSync(path, 0o000);
        },
      },
    ];
    for (const testCase of cases) {
      const base = mkdtempSync(join(VERIFY_HOME, "roots-invalid-"));
      testCase.write(base);
      try {
        await withFreshBoundary(base, async (mod) => {
          const deps = await mod.defaultNativeCollectorDeps(openDb(":memory:"));
          expect(deps.roots.claude).toContain("native-invalid");
          expect(deps.roots.claude).not.toBe("/tmp");
          expect(deps.configWarnings).toEqual(["config-invalid"]);
        });
      } finally {
        rmSync(base, { recursive: true, force: true });
      }
    }
  });

  it("selects the normal native roots only when the override is absent", async () => {
    const base = mkdtempSync(join(VERIFY_HOME, "roots-absent-"));
    try {
      await withFreshBoundary(base, async (mod) => {
        const deps = await mod.defaultNativeCollectorDeps(openDb(":memory:"));
        expect(deps.roots.claude).not.toContain("native-invalid");
        expect(deps.configWarnings).toEqual([]);
      });
    } finally {
      rmSync(base, { recursive: true, force: true });
    }
  });
});

describe("Codex discovery", () => {
  it("accepts only the sessions/YYYY/MM/DD rollout layout", async () => {
    const deps = makeDeps();
    const dir = join(deps.roots.codex, "2026", "11", "17");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "rollout-2026-11-17.jsonl"), codexLine("r1"));
    writeFileSync(join(dir, "not-a-rollout.jsonl"), codexLine("r2"));
    await runNativeUsagePass(deps);
    const registered = deps.db.prepare("SELECT identity FROM native_usage_files WHERE source = 'codex'").all() as { identity: string }[];
    expect(registered.map((f) => f.identity)).toEqual(["2026/11/17/rollout-2026-11-17.jsonl"]);
    expect(getSourceStatuses(deps.db).find((s) => s.source === "codex")?.warnings).toContain("unsupported-layout");
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

function runHelper(
  dbPath: string,
  state: unknown,
): { reply: Record<string, unknown>; opts: Record<string, unknown>; sql: string[] } {
  let stdout = "";
  let opts: Record<string, unknown> = {};
  const sql: string[] = [];
  class ShimDatabase {
    db: BetterSqlite3.Database;
    constructor(path: string, o: { readonly?: boolean; strict?: boolean }) {
      opts = o as Record<string, unknown>;
      this.db = new BetterSqlite3(path, { readonly: o.readonly === true, fileMustExist: true });
    }
    prepare(statement: string) {
      sql.push(statement);
      return this.db.prepare(statement);
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
  return { reply: JSON.parse(stdout) as Record<string, unknown>, opts, sql };
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

const freshCursor = {
  v: 1 as const,
  v1: { after: null, ceiling: null, done: false },
  v2: { after: null, ceiling: null, done: false },
};

describe("fixed OpenCode helper (NodeVM with a bun:sqlite shim)", () => {
  it("pages both projections by rowid, stays readonly and emits no message text", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-"));
    const path = nativeOpenCodeDb(base);
    const first = runHelper(path, freshCursor);
    expect(first.opts).toEqual({ readonly: true, strict: true });
    const reply = first.reply as {
      ok: boolean; rows: { projection: string }[]; next: { v1: { after: string; done: boolean }; v2: { done: boolean } };
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
  }, 30_000);

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
      ok: boolean; rows: unknown[]; next: { v1: { after: string; ceiling: string; done: boolean } };
    };
    expect(reply.ok).toBe(true);
    expect(reply.rows.length).toBeGreaterThan(0);
    expect(reply.rows.length).toBeLessThan(300);
    expect(BigInt(reply.next.v1.after)).toBeGreaterThan(0n);
    expect(BigInt(reply.next.v1.after)).toBeLessThan(BigInt(reply.next.v1.ceiling));
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
    const reply = runHelper(path, freshCursor).reply as { warnings: Record<string, number>; next: { v1: { after: string } } };
    expect(reply.warnings.malformed).toBe(1);
    expect(BigInt(reply.next.v1.after)).toBeGreaterThan(0n);
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
    next: { v1: { after: null, ceiling: null, done: true }, v2: { after: "1", ceiling: "1", done: true } },
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

  it("attributes an OpenCode pass from a launch record and leaves Claude/Codex labels untouched", async () => {
    const launch = {
      projection: "v2", session_id: "ses-attr", message_id: "msg-attr", parent_session_id: null,
      model_id: "jaxflow-builder-default", provider_id: "opencode-go", agent: "build",
      cwd: "/w-attr",
      created: 1_794_909_600_000, updated: 1_794_909_800_000,
      cost: 12.5, input: 100, output: 30, reasoning: 7, cache_read: 11, cache_write: 5,
    };
    const deps = withDbFile({
      exec: async () => ({
        stdout: JSON.stringify({
          v: 1, ok: true, runtime: "Bun 1.3.14", schema: "1.18.34", rows: [launch],
          next: { v1: { after: null, ceiling: null, done: true }, v2: { after: "1", ceiling: "1", done: true } },
          warnings: {},
        }),
      }),
    });
    deps.db.prepare(
      `INSERT INTO workflow_events (ts, run_id, project, role, type, source, emitter, payload, delivery)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      "2026-11-17T09:00:00Z", "aaaabbbbcccc", "demo", "builder", "builder-launched",
      "deterministic", "wrapper",
      JSON.stringify({
        launched_at: "2026-11-17T09:00:00Z", worktree: "/w-attr",
        runtime_model: "opencode-go/jaxflow-builder-default",
        model: "opencode-go/deepseek-v4.1-flash", profile: "default",
      }),
      "local",
    );
    const insertItem = deps.db.prepare(
      `INSERT INTO native_usage_items (harness, item_id, session_id, child_id, parent_id, is_child,
         provenance, agent, model, model_conflict, identity_conflict, ts, updated_ts, day,
         all_input, all_output, projection)
       VALUES (?, ?, 'ses-old', '', NULL, 0, 'test', ?, ?, 0, 0, 1794909600, 1794909600, '2026-11-17', 1, 1, NULL)`,
    );
    insertItem.run("claude", "m-c", "Claude Code", "claude-sonnet-5-5");
    insertItem.run("codex", "r-x", "Codex", "gpt-5.6-luna");

    await runNativeUsagePass(deps);
    expect(deps.db.prepare(
      "SELECT model, raw_model, cwd, agent FROM native_usage_items WHERE harness = 'opencode'",
    ).get()).toEqual({
      model: "opencode-go/deepseek-v4.1-flash",
      raw_model: "opencode-go/jaxflow-builder-default",
      cwd: "/w-attr",
      agent: "OpenCode",
    });
    expect(deps.db.prepare("SELECT agent, model FROM native_usage_items WHERE harness = 'claude'").get())
      .toEqual({ agent: "Claude Code", model: "claude-sonnet-5-5" });
    expect(deps.db.prepare("SELECT agent, model FROM native_usage_items WHERE harness = 'codex'").get())
      .toEqual({ agent: "Codex", model: "gpt-5.6-luna" });
    deps.db.close();
  });

  it("treats a changed executable identity with a mismatched hash as incompatible, retaining last-good rows", async () => {
    let calls = 0;
    let hashCalls = 0;
    let size = 100;
    const deps = withDbFile({
      exec: async () => { calls += 1; return { stdout: reply }; },
      stat: async () => ({
        dev: 1, ino: 1, size, mtimeMs: Date.parse("2026-11-18T10:00:00Z"),
        isFile: () => true, isDirectory: () => false, isSymbolicLink: () => false,
      }),
      hash: async () => { hashCalls += 1; return size === 100 ? OPENCODE_EXECUTABLE_SHA256 : "deadbeef"; },
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


describe("durable discovery and fair selection", () => {
  it("indexes 100+ Claude files across projects in a finite number of passes", async () => {
    const deps = makeDeps();
    for (let project = 0; project < 4; project += 1) {
      for (let i = 0; i < 30; i += 1) {
        writeClaude(deps, `proj-${project}/s${i}.jsonl`, [
          claudeLine(`m-${project}-${i}`, { sessionId: `sess-${project}-${i}` }),
        ]);
      }
    }
    let passes = 0;
    while (passes < 80) {
      await runNativeUsagePass(deps);
      passes += 1;
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number };
      if (row.n === 120) break;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 120 });
    expect(getSourceStatuses(deps.db).find((s) => s.source === "claude")?.state).toBe("available");
    deps.db.close();
  }, 60_000);

  it("indexes 90+ Codex date directories without enumerating every parent forever", async () => {
    const deps = makeDeps();
    for (let month = 9; month <= 11; month += 1) {
      for (let day = 1; day <= 30; day += 1) {
        const dir = join(deps.roots.codex, "2026", String(month).padStart(2, "0"), String(day).padStart(2, "0"));
        mkdirSync(dir, { recursive: true });
        writeFileSync(join(dir, "rollout-x.jsonl"), `${codexLine(`r-${month}-${day}`)}\n`);
      }
    }
    let passes = 0;
    while (passes < 60) {
      await runNativeUsagePass(deps);
      passes += 1;
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number };
      if (row.n === 90) break;
    }
    expect(passes).toBeLessThan(60);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 90 });
    expect(getSourceStatuses(deps.db).find((s) => s.source === "codex")?.state).toBe("available");
    deps.db.close();
  }, 60_000);

  it("resumes a mid-cycle restart from persisted registry state and keeps child classification", async () => {
    const deps = makeDeps();
    for (let i = 0; i < 40; i += 1) writeClaude(deps, `p/f${i}.jsonl`, [claudeLine(`m${i}`)]);
    writeClaude(deps, "p/session/subagents/agent-1.jsonl", [claudeLine("child-1", { agentId: "agent-1" })]);
    for (let pass = 0; pass < 4; pass += 1) await runNativeUsagePass(deps);
    const registered = deps.db
      .prepare("SELECT identity, child FROM native_usage_files WHERE source = 'claude' AND identity LIKE '%subagents%'")
      .get() as { identity: string; child: number } | undefined;
    expect(registered?.child).toBe(1);

    const restarted = { ...deps };
    let passes = 0;
    while (passes < 20) {
      await runNativeUsagePass(restarted);
      passes += 1;
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number };
      if (row.n === 41) break;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 41 });
    expect(queryNativeUsage(deps.db, { period: 30, now: NOW }).sessions).toEqual({ lead: 1, child: 1 });
    deps.db.close();
  }, 60_000);

  it("reports a root permission failure as unavailable, never as empty", async () => {
    const deps = makeDeps();
    const denied: NativeFs = {
      ...nativeFs,
      opendir: async (path) => {
        if (path === deps.roots.claude) {
          const error = new Error("EACCES") as Error & { code: string };
          error.code = "EACCES";
          throw error;
        }
        return nativeFs.opendir(path);
      },
    };
    deps.fs = denied;
    await runNativeUsagePass(deps);
    const status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
    expect(status?.state).toBe("unavailable");
    expect(status?.warnings).toContain("root-unavailable");
    deps.db.close();
  });

  it("keeps advancing a known file while discovery is still incomplete", async () => {
    const deps = makeDeps();
    for (let i = 0; i < 100; i += 1) writeClaude(deps, `p/f${i}.jsonl`, [claudeLine(`m${i}`)]);
    await runNativeUsagePass(deps);
    const first = deps.db
      .prepare("SELECT identity FROM native_usage_files WHERE source = 'claude' ORDER BY identity LIMIT 1")
      .get() as { identity: string };
    appendFileSync(join(deps.roots.claude, first.identity), `${claudeLine("m-appended", { sessionId: "sess-appended" })}\n`);
    await runNativeUsagePass(deps);
    const appended = deps.db
      .prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-appended'")
      .get() as { n: number };
    expect(appended.n).toBe(1);
    deps.db.close();
  }, 60_000);

  it("reaches a changed non-hot file within 126 passes and a hot tail within two", async () => {
    const deps = makeDeps();
    const files = Array.from({ length: 500 }, (_, i) => {
      const path = writeClaude(deps, `p/f${i}.jsonl`, [claudeLine(`m${i}`)]);
      utimesSync(path, new Date("2026-11-10T00:00:00Z"), new Date("2026-11-10T00:00:00Z"));
      return path;
    });
    let clock = NOW;
    const tick = () => {
      clock += 61;
      return clock;
    };
    let passes = 0;
    while (passes < 140) {
      await runNativeUsagePass({ ...deps, now: () => tick(), nowMs: () => Date.now() });
      passes += 1;
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number };
      if (row.n === 500) break;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 500 });

    const last = deps.db
      .prepare("SELECT identity FROM native_usage_files WHERE source = 'claude' ORDER BY visit_ord DESC LIMIT 1")
      .get() as { identity: string };
    const targetPath = join(deps.roots.claude, last.identity);
    appendFileSync(targetPath, `${claudeLine("m-tail", { sessionId: "sess-tail" })}\n`);
    utimesSync(targetPath, new Date("2026-11-10T00:00:00Z"), new Date("2026-11-10T00:00:00Z"));
    let tailPasses = 0;
    while (tailPasses < 126) {
      await runNativeUsagePass({ ...deps, now: () => tick(), nowMs: () => Date.now() });
      tailPasses += 1;
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-tail'").get() as { n: number };
      if (row.n === 1) break;
    }
    expect(tailPasses).toBeLessThanOrEqual(126);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-tail'").get()).toEqual({ n: 1 });

    const hotPath = files[0];
    appendFileSync(hotPath, `${claudeLine("m-hot", { sessionId: "sess-hot" })}\n`);
    let hotPasses = 0;
    while (hotPasses < 2) {
      await runNativeUsagePass({ ...deps, now: () => tick(), nowMs: () => Date.now() });
      hotPasses += 1;
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-hot'").get() as { n: number };
      if (row.n === 1) break;
    }
    expect(hotPasses).toBeLessThanOrEqual(2);
    deps.db.close();
  }, 120_000);

  it("stops starting new lines at the scheduling deadline and finishes later", async () => {
    const deps = makeDeps();
    const lines = Array.from({ length: 2000 }, (_, i) => claudeLine(`m${i}`, { sessionId: `sess-${i}` }));
    writeClaude(deps, "p/many.jsonl", lines);
    await runNativeUsagePass(deps);
    const first = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number };
    expect(first.n).toBeLessThan(2000);
    expect(first.n).toBeGreaterThan(0);
    let passes = 1;
    while (passes < 20) {
      await runNativeUsagePass(deps);
      passes += 1;
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number };
      if (row.n === 2000) break;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 2000 });
    deps.db.close();
  }, 60_000);

  it("discovers a new file after the completed cycle and refresh cadence", async () => {
    const deps = makeDeps();
    writeClaude(deps, "p/first.jsonl", [claudeLine("m-first")]);
    let passes = 0;
    while (passes < 6) {
      await runNativeUsagePass(deps);
      passes += 1;
      if (getSourceStatuses(deps.db).find((s) => s.source === "claude")?.state === "available") break;
    }
    writeClaude(deps, "p/second.jsonl", [claudeLine("m-second", { sessionId: "sess-2" })]);
    for (let extra = 0; extra < 3; extra += 1) {
      await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-second'").get() as { n: number };
      if (row.n === 1) break;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-second'").get()).toEqual({ n: 1 });
    deps.db.close();
  }, 60_000);

  it("retains ownership of a timed-out filesystem operation instead of accumulating reads", async () => {
    const deps = makeDeps();
    writeClaude(deps, "p/blocked.jsonl", [claudeLine("m-blocked")]);
    let openCalls = 0;
    const blocked: NativeFs = {
      ...nativeFs,
      open: async (path) => {
        openCalls += 1;
        if (openCalls === 1) return new Promise<never>(() => {});
        return nativeFs.open(path);
      },
    };
    deps.fs = blocked;
    await runNativeUsagePass(deps);
    expect(openCalls).toBe(1);
    await runNativeUsagePass(deps);
    expect(openCalls).toBe(1);
    const status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
    expect(status?.warnings).toContain("io-pending");
    deps.db.close();
  });
});

describe("oversized-line discard progress", () => {
  it("discards a 3 MiB line across passes and ingests the valid tail", async () => {
    const deps = makeDeps();
    const path = join(deps.roots.claude, "p/huge.jsonl");
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${"x".repeat(3 * 1024 * 1024)}\n${claudeLine("m-tail-after-discard")}\n`);
    await runNativeUsagePass(deps);
    let cp = getFileCheckpoint(deps.db, "claude", "p/huge.jsonl");
    expect(cp).toMatchObject({ discarding: true, done: false });
    expect(cp!.offset).toBeGreaterThan(0);
    const seenWarnings = new Set<string>();
    for (let pass = 0; pass < 6; pass += 1) {
      for (const warning of getSourceStatuses(deps.db).find((s) => s.source === "claude")?.warnings ?? []) {
        seenWarnings.add(warning);
      }
      cp = getFileCheckpoint(deps.db, "claude", "p/huge.jsonl");
      if (cp?.done && !cp.discarding) break;
      await runNativeUsagePass(deps);
    }
    expect(cp).toMatchObject({ discarding: false, done: true });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    expect([...seenWarnings]).toContain("line-too-long");
    deps.db.close();
  }, 60_000);

  it("resumes discard state after a restart and resets it on truncation", async () => {
    const deps = makeDeps();
    const path = join(deps.roots.claude, "p/huge.jsonl");
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${"y".repeat(3 * 1024 * 1024)}\n${claudeLine("m-after-restart")}\n`);
    await runNativeUsagePass(deps);
    expect(getFileCheckpoint(deps.db, "claude", "p/huge.jsonl")?.discarding).toBe(true);
    const restarted = { ...deps };
    for (let pass = 0; pass < 6; pass += 1) {
      await runNativeUsagePass(restarted);
      const done = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number };
      if (done.n === 1) break;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });

    writeFileSync(path, `${"z".repeat(3 * 1024 * 1024)}\n`);
    await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
    expect(getFileCheckpoint(deps.db, "claude", "p/huge.jsonl")?.discarding).toBe(true);
    writeFileSync(path, `${claudeLine("m-after-truncate")}\n`);
    await runNativeUsagePass({ ...deps, now: () => NOW + 122 });
    const cp = getFileCheckpoint(deps.db, "claude", "p/huge.jsonl");
    expect(cp?.discarding).toBe(false);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-after-truncate'").get()).toEqual({ n: 1 });
    deps.db.close();
  }, 120_000);
});

describe("OpenCode rowid and raw-page regressions", () => {
  function baseDb(path: string): BetterSqlite3.Database {
    const db = new BetterSqlite3(path);
    db.exec(`
      CREATE TABLE session (id TEXT PRIMARY KEY, parent_id TEXT);
      CREATE TABLE message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, time_created INTEGER, time_updated INTEGER, data TEXT);
      CREATE TABLE session_message (id TEXT PRIMARY KEY, session_id TEXT NOT NULL, type TEXT, seq INTEGER, time_created INTEGER, time_updated INTEGER, data TEXT);
      CREATE TABLE migration (id INTEGER PRIMARY KEY);
      INSERT INTO session (id, parent_id) VALUES ('ses-1', NULL);
    `);
    return db;
  }

  it("handles signed MIN, zero and unsafe rowids as canonical decimal strings", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-rowid-"));
    const path = join(base, "rowid.db");
    const db = baseDb(path);
    const data = JSON.stringify({
      role: "assistant", modelID: "model-x", agent: "build",
      created: 1_794_909_600_000, tokens: { input: 1, output: 1 },
    });
    db.prepare("INSERT INTO message (rowid, id, session_id, time_created, time_updated, data) VALUES (-9223372036854775808, 'm-min', 'ses-1', 1, 1, ?)").run(data);
    db.prepare("INSERT INTO message (rowid, id, session_id, time_created, time_updated, data) VALUES (0, 'm-zero', 'ses-1', 1, 1, ?)").run(data);
    db.prepare("INSERT INTO message (rowid, id, session_id, time_created, time_updated, data) VALUES (9223372036854775806, 'm-unsafe', 'ses-1', 1, 1, ?)").run(data);
    db.close();
    const reply = runHelper(path, freshCursor).reply as {
      ok: boolean;
      rows: { message_id: string }[];
      next: { v1: { after: string; ceiling: string; done: boolean } };
    };
    expect(reply.ok).toBe(true);
    expect(reply.rows.map((r) => r.message_id)).toEqual(["m-min", "m-zero", "m-unsafe"]);
    expect(reply.next.v1.after).toBe("9223372036854775806");
    expect(reply.next.v1.ceiling).toBe("9223372036854775806");
    expect(reply.next.v1.done).toBe(true);
    rmSync(base, { recursive: true, force: true });
  });

  it("advances only through one raw page before filtering a 300-row user prefix", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-prefix-"));
    const path = join(base, "prefix.db");
    const db = baseDb(path);
    const insert = db.prepare(
      "INSERT INTO session_message (id, session_id, type, seq, time_created, time_updated, data) VALUES (?, 'ses-1', ?, ?, 1, 1, ?)",
    );
    for (let i = 1; i <= 300; i += 1) insert.run(`u${i}`, "user", i, JSON.stringify({ type: "user" }));
    insert.run("a1", "assistant", 301, JSON.stringify({
      type: "assistant", agent: "build", model: { id: "model-y", providerID: "p" },
      time: { created: 1_794_909_600_000 }, tokens: { input: 2, output: 3 },
    }));
    db.close();
    const first = runHelper(path, freshCursor).reply as {
      rows: unknown[];
      next: { v2: { after: string; ceiling: string; done: boolean } };
    };
    expect(first.rows).toHaveLength(0);
    expect(first.next.v2.after).toBe("256");
    expect(first.next.v2.done).toBe(false);
    const second = runHelper(path, { v: 1, v1: { after: null, ceiling: null, done: true }, v2: first.next.v2 }).reply as {
      rows: { message_id: string }[];
      next: { v2: { done: boolean } };
    };
    expect(second.rows.map((r) => r.message_id)).toEqual(["a1"]);
    expect(second.next.v2.done).toBe(true);
    rmSync(base, { recursive: true, force: true });
  });

  it("skips a >1 MiB BLOB by length before fetching data", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-blob-"));
    const path = join(base, "blob.db");
    const db = baseDb(path);
    const small = JSON.stringify({
      role: "assistant", modelID: "model-x", agent: "build",
      created: 1_794_909_600_000, tokens: { input: 1, output: 1 },
    });
    const huge = JSON.stringify({
      role: "assistant", modelID: "model-x", agent: "build", padding: "p".repeat(1024 * 1024 + 32),
      created: 1_794_909_600_000, tokens: { input: 1, output: 1 },
    });
    const insert = db.prepare("INSERT INTO message (id, session_id, time_created, time_updated, data) VALUES (?, 'ses-1', 1, 1, ?)");
    insert.run("small-1", small);
    insert.run("huge", huge);
    insert.run("small-2", small.replace('"input":1,"output":1', '"input":2,"output":2'));
    db.close();
    const result = runHelper(path, freshCursor);
    const reply = result.reply as { ok: boolean; rows: { message_id: string }[]; warnings: Record<string, number> };
    expect(reply.ok).toBe(true);
    expect(reply.rows.map((r) => r.message_id)).toEqual(["small-1", "small-2"]);
    expect(reply.warnings.unsupported).toBe(1);
    expect(result.sql.filter((statement) => statement.startsWith("SELECT data FROM"))).toHaveLength(2);
    rmSync(base, { recursive: true, force: true });
  });

  it("rejects malformed reply identity, cursor regressions and ceiling changes", () => {
    const previous = {
      v: 1 as const,
      v1: { after: "10", ceiling: "20", done: false },
      v2: { after: null, ceiling: null, done: false },
    };
    const valid = {
      v: 1, ok: true, runtime: "Bun 1.3.14", schema: "1.18.34", rows: [],
      next: { v1: { after: "12", ceiling: "20", done: false }, v2: { after: null, ceiling: null, done: false } },
      warnings: {},
    };
    expect(parseOpenCodeReply(JSON.stringify(valid), previous)).not.toBeNull();
    expect(parseOpenCodeReply(JSON.stringify({ ...valid, runtime: "Bun 9.9.9" }), previous)).toBeNull();
    expect(parseOpenCodeReply(JSON.stringify({ ...valid, schema: "9.9.9" }), previous)).toBeNull();
    const regressed = { ...valid, next: { ...valid.next, v1: { after: "9", ceiling: "20", done: false } } };
    expect(parseOpenCodeReply(JSON.stringify(regressed), previous)).toBeNull();
    const movedCeiling = { ...valid, next: { ...valid.next, v1: { after: "12", ceiling: "21", done: false } } };
    expect(parseOpenCodeReply(JSON.stringify(movedCeiling), previous)).toBeNull();
    const unsafe = { ...valid, next: { ...valid.next, v1: { after: "9223372036854775808", ceiling: "9223372036854775808", done: false } } };
    expect(parseOpenCodeReply(JSON.stringify(unsafe), previous)).toBeNull();
  });

  it("completes a cap-throttled page sequence without duplicates or rewind", () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-cap-seq-"));
    const path = join(base, "cap-seq.db");
    const db = baseDb(path);
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
    let state: { v: 1; v1: { after: string | null; ceiling: string | null; done: boolean }; v2: { after: string | null; ceiling: string | null; done: boolean } } = freshCursor;
    const seen: string[] = [];
    let passes = 0;
    do {
      const reply = runHelper(path, state).reply as {
        ok: boolean;
        rows: { message_id: string }[];
        next: { v1: typeof state.v1; v2: typeof state.v2 };
      };
      expect(reply.ok).toBe(true);
      seen.push(...reply.rows.map((r) => r.message_id));
      state = { v: 1, v1: reply.next.v1, v2: reply.next.v2 };
      passes += 1;
    } while (!(state.v1.done && state.v2.done) && passes < 50);
    expect(passes).toBeLessThan(50);
    expect(seen).toHaveLength(300);
    expect(new Set(seen).size).toBe(300);
    rmSync(base, { recursive: true, force: true });
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

    let state: { v: 1; v1: { after: string | null; ceiling: string | null; done: boolean }; v2: { after: string | null; ceiling: string | null; done: boolean } } = freshCursor;
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
    expect(passes).toBeGreaterThanOrEqual(196);
    expect(passes).toBeLessThanOrEqual(200);
    expect(BigInt(state.v1.after ?? "0")).toBe(50_000n);
    expect(BigInt(state.v2.after ?? "0")).toBe(50_000n);
    expect(replied).toBe(100_000);
    rmSync(base, { recursive: true, force: true });
  }, 120_000);
});

describe("one owned IO scope and honest source lifecycle", () => {
  async function settleDiscovery(deps: NativeCollectorDeps, source: "claude" | "codex", max = 8): Promise<void> {
    for (let pass = 0; pass < max; pass += 1) {
      await runNativeUsagePass({ ...deps, now: () => NOW + pass * 61 });
      const status = getSourceStatuses(deps.db).find((s) => s.source === source);
      if (status?.state !== "indexing") return;
    }
  }

  it("rejects a queued subagents symlink before enumerating its target", async () => {
    const deps = makeDeps();
    const outside = join(deps.roots.claude, "..", "outside-subagents");
    mkdirSync(outside, { recursive: true });
    writeFileSync(join(outside, "leak.jsonl"), `${claudeLine("m-leak")}\n`);
    mkdirSync(join(deps.roots.claude, "proj", "uuid"), { recursive: true });
    symlinkSync(outside, join(deps.roots.claude, "proj", "uuid", "subagents"));
    const opened: string[] = [];
    deps.fs = {
      ...nativeFs,
      opendir: async (path) => {
        opened.push(fsRealpathSync(path));
        return nativeFs.opendir(path);
      },
    };
    await settleDiscovery(deps, "claude");
    expect(opened).not.toContain(fsRealpathSync(outside));
    expect(getSourceStatuses(deps.db).find((s) => s.source === "claude")?.warnings)
      .toContain("symlink-escape");
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 0 });
    deps.db.close();
  });

  it("requires opened-handle identity before reading and closes on an fstat failure", async () => {
    const deps = makeDeps();
    writeClaude(deps, "p/one.jsonl", [claudeLine("m-one")]);
    let reads = 0;
    let closed = 0;
    deps.fs = {
      ...nativeFs,
      open: async (path) => {
        const handle = await nativeFs.open(path);
        return {
          read: (buffer, offset, length, position) => {
            reads += 1;
            return handle.read(buffer, offset, length, position);
          },
          close: async () => {
            closed += 1;
            await handle.close();
          },
          stat: async () => {
            throw new Error("EIO");
          },
        };
      },
    };
    await settleDiscovery(deps, "claude");
    expect(reads).toBe(0);
    expect(closed).toBe(1);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 0 });
    expect(getSourceStatuses(deps.db).find((s) => s.source === "claude")?.warnings)
      .toContain("read-failed");
    deps.db.close();
  });

  it("retains the original root operation across polls instead of starting another", async () => {
    const deps = makeDeps();
    let statCalls = 0;
    deps.fs = {
      ...nativeFs,
      stat: (path) => {
        if (path !== deps.roots.claude) return nativeFs.stat(path);
        statCalls += 1;
        return new Promise<never>(() => {});
      },
    };
    await runNativeUsagePass(deps);
    expect(statCalls).toBe(1);
    await runNativeUsagePass(deps);
    expect(statCalls).toBe(1);
    expect(getSourceStatuses(deps.db).find((s) => s.source === "claude")?.warnings)
      .toContain("io-pending");
    deps.db.close();
  });

  it("closes a late successful open exactly once after a timed-out attempt", async () => {
    const deps = makeDeps();
    writeClaude(deps, "p/late.jsonl", [claudeLine("m-late")]);
    let openCalls = 0;
    let closed = 0;
    let releaseOpen: (() => void) | null = null;
    deps.fs = {
      ...nativeFs,
      open: async (path) => {
        openCalls += 1;
        if (openCalls === 1) {
          const real = await nativeFs.open(path);
          return new Promise((resolve) => {
            releaseOpen = () => resolve({
              read: (buffer, offset, length, position) => real.read(buffer, offset, length, position),
              close: async () => {
                closed += 1;
                await real.close();
              },
              stat: () => real.stat(),
            });
          });
        }
        return nativeFs.open(path);
      },
    };
    await runNativeUsagePass(deps);
    expect(openCalls).toBe(1);
    expect(closed).toBe(0);
    releaseOpen!();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(closed).toBe(1);
    await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
    expect(openCalls).toBe(2);
    deps.db.close();
  });

  it("starts no discovery operation after the scheduling deadline", async () => {
    const deps = makeDeps();
    let ms = 0;
    const events: string[] = [];
    deps.fs = {
      ...nativeFs,
      stat: async (path) => {
        events.push(`stat@${ms}`);
        ms += 500;
        return fsStatAsync(path);
      },
      realpath: async (path) => {
        events.push(`realpath@${ms}`);
        ms += 500;
        return fsRealpathAsync(path);
      },
      opendir: async (path) => {
        events.push(`opendir@${ms}`);
        return nativeFs.opendir(path);
      },
    };
    await runNativeUsagePass({ ...deps, now: () => NOW, nowMs: () => ms });
    expect(events.some((event) => event.startsWith("opendir"))).toBe(false);
    deps.db.close();
  });

  it("finishes a supported 512 KiB line inside its single line allowance", async () => {
    const deps = makeDeps();
    const padding = "x".repeat(470 * 1024);
    writeClaude(deps, "p/big.jsonl", [
      JSON.stringify({
        type: "assistant", sessionId: "sess-big", timestamp: "2026-11-17T10:00:00Z",
        message: { id: "m-big", usage: { input_tokens: 1, output_tokens: 1 }, content: padding },
      }),
    ]);
    let ms = 0;
    let reads = 0;
    deps.fs = {
      ...nativeFs,
      open: async (path) => {
        const handle = await nativeFs.open(path);
        return {
          read: (buffer, offset, length, position) => {
            reads += 1;
            ms += 60;
            return handle.read(buffer, offset, length, position);
          },
          close: () => handle.close(),
          stat: () => handle.stat(),
        };
      },
    };
    const timed = { ...deps, now: () => NOW, nowMs: () => ms };
    for (let pass = 0; pass < 3; pass += 1) {
      await runNativeUsagePass(timed);
      const row = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-big'").get() as { n: number };
      if (row.n === 1) break;
    }
    expect(reads).toBeGreaterThan(8);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-big'").get()).toEqual({ n: 1 });
    expect(getFileCheckpoint(deps.db, "claude", "p/big.jsonl")?.offset).toBeGreaterThan(470 * 1024);
    deps.db.close();
  });

  it("reports a missing previously-known root as unavailable with retained rows", async () => {
    const deps = makeDeps();
    writeClaude(deps, "p/one.jsonl", [claudeLine("m-one")]);
    await settleDiscovery(deps, "claude");
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    rmSync(deps.roots.claude, { recursive: true, force: true });
    await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
    const status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
    expect(status?.state).toBe("unavailable");
    expect(status?.warnings).toContain("root-unavailable");
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    deps.db.close();
  });

  it("keeps a persisted failure warning across a later indexing poll", async () => {
    const deps = makeDeps();
    writeClaude(deps, "p/000bad.jsonl", [claudeLine("m-bad")]);
    for (let i = 0; i < 20; i += 1) writeClaude(deps, `p/f${String(i).padStart(3, "0")}.jsonl`, [claudeLine(`m-${i}`)]);
    deps.fs = {
      ...nativeFs,
      open: async (path) => {
        if (path.endsWith("000bad.jsonl")) throw new Error("EACCES");
        return nativeFs.open(path);
      },
    };
    await runNativeUsagePass(deps);
    expect(getPersistentWarnings(deps.db, "claude")).toContain("read-failed");
    await runNativeUsagePass(deps);
    const status = getSourceStatuses(deps.db).find((s) => s.source === "claude");
    expect(status?.state).toBe("indexing");
    expect(status?.warnings).toContain("read-failed");
    deps.fs = nativeFs;
    for (let pass = 1; pass <= 6; pass += 1) {
      await runNativeUsagePass({ ...deps, now: () => NOW + pass * 61 });
      if (!getPersistentWarnings(deps.db, "claude").includes("read-failed")) break;
    }
    expect(getPersistentWarnings(deps.db, "claude")).not.toContain("read-failed");
    deps.db.close();
  });

  it("keeps one original OpenCode hash reader across repeated preparation timeouts", async () => {
    let hashCalls = 0;
    const deps = makeDeps({
      hash: () => {
        hashCalls += 1;
        return new Promise<never>(() => {});
      },
    });
    writeFileSync(deps.roots.opencodeDb, "");
    await runNativeUsagePass(deps);
    expect(hashCalls).toBe(1);
    await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
    expect(hashCalls).toBe(1);
    const status = getSourceStatuses(deps.db).find((s) => s.source === "opencode");
    expect(status?.warnings).toContain("io-pending");
    deps.db.close();
  });

  it("hashes the default dependency with bounded chunks instead of readFile", async () => {
    const { createHash } = await import("node:crypto");
    const base = mkdtempSync(join(VERIFY_HOME, "hash-"));
    const path = join(base, "executable.bin");
    const content = Buffer.alloc(150 * 1024);
    for (let i = 0; i < content.length; i += 1) content[i] = i % 251;
    writeFileSync(path, content);
    const deps = await defaultNativeCollectorDeps(openDb(":memory:"));
    const hash = await deps.opencode.hash(path);
    expect(hash).toBe(createHash("sha256").update(content).digest("hex"));
    rmSync(base, { recursive: true, force: true });
  });
});

describe("monotonic budgeting and cooperative hash cancellation", () => {
  function openCodeOnlyDeps(opts: MakeDepsOptions = {}): NativeCollectorDeps {
    const deps = makeDeps({ ...opts, enabled: { claude: false, codex: false, opencode: true } });
    writeFileSync(deps.roots.opencodeDb, "");
    return deps;
  }

  const fileStat = async () => ({
    dev: 1, ino: 1, size: 8, mtimeMs: 0,
    isFile: () => true, isDirectory: () => false, isSymbolicLink: () => false,
  });

  it("keeps budget time monotonic across a backwards wall-clock step while UTC stays on wall time", async () => {
    const deps = await defaultNativeCollectorDeps(openDb(":memory:"));
    const wall = vi.spyOn(Date, "now").mockReturnValue(Date.parse("2026-11-18T12:00:00Z"));
    const before = deps.nowMs();
    const utcBefore = deps.now();
    wall.mockReturnValue(Date.parse("2026-11-18T11:58:00Z"));
    const after = deps.nowMs();
    expect(after).toBeGreaterThanOrEqual(before);
    expect(deps.now()).toBe(utcBefore - 120);
    wall.mockRestore();
  });

  it("stops the production-default streaming hash before the next chunk after its preparation timeout", async () => {
    const deps = openCodeOnlyDeps();
    let opens = 0;
    let reads = 0;
    let closes = 0;
    let readStarted = false;
    let releaseRead: (() => void) | null = null;
    const fakeFs: NativeFs = {
      ...nativeFs,
      open: async () => {
        opens += 1;
        return {
          read: async () => {
            reads += 1;
            if (reads === 1) {
              readStarted = true;
              return new Promise<{ bytesRead: number }>((resolve) => {
                releaseRead = () => resolve({ bytesRead: 4 });
              });
            }
            return { bytesRead: 0 };
          },
          close: async () => { closes += 1; },
          stat: fileStat,
        };
      },
    };
    deps.opencode.hash = (path, cancellation) => hashFileStreaming(fakeFs, path, cancellation);

    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      const pass = runNativeUsagePass(deps);
      for (let i = 0; i < 200 && !readStarted; i += 1) await vi.advanceTimersByTimeAsync(1);
      // Reaching the hash before any timer advance is part of the scenario: the first
      // read is the unabortable operation the public timeout must abandon.
      expect(readStarted).toBe(true);
      await vi.advanceTimersByTimeAsync(1300);
      await pass;
    } finally {
      vi.useRealTimers();
    }
    const timedOut = getSourceStatuses(deps.db).find((s) => s.source === "opencode");
    expect(timedOut).toMatchObject({ state: "unavailable" });
    expect(timedOut?.warnings).toContain("io-pending");
    expect({ opens, reads, closes }).toEqual({ opens: 1, reads: 1, closes: 0 });

    await runNativeUsagePass(deps);
    expect({ opens, reads }).toEqual({ opens: 1, reads: 1 });

    releaseRead!();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect({ opens, reads, closes }).toEqual({ opens: 1, reads: 1, closes: 1 });

    await runNativeUsagePass(deps);
    expect({ opens, reads, closes }).toEqual({ opens: 2, reads: 2, closes: 2 });
    deps.db.close();
  });

  it("keeps single flight while a cancelled hash close is still pending", async () => {
    const deps = openCodeOnlyDeps();
    let opens = 0;
    let reads = 0;
    let readStarted = false;
    let releaseRead: (() => void) | null = null;
    let releaseClose: (() => void) | null = null;
    const fakeFs: NativeFs = {
      ...nativeFs,
      open: async () => {
        opens += 1;
        return {
          read: async () => {
            reads += 1;
            if (reads === 1) {
              readStarted = true;
              return new Promise<{ bytesRead: number }>((resolve) => {
                releaseRead = () => resolve({ bytesRead: 2 });
              });
            }
            return { bytesRead: 0 };
          },
          close: () => new Promise<void>((resolve) => { releaseClose = resolve; }),
          stat: fileStat,
        };
      },
    };
    deps.opencode.hash = (path, cancellation) => hashFileStreaming(fakeFs, path, cancellation);

    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      const pass = runNativeUsagePass(deps);
      for (let i = 0; i < 200 && !readStarted; i += 1) await vi.advanceTimersByTimeAsync(1);
      expect(readStarted).toBe(true);
      await vi.advanceTimersByTimeAsync(1300);
      await pass;
    } finally {
      vi.useRealTimers();
    }
    releaseRead!();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(releaseClose).not.toBeNull();

    await runNativeUsagePass(deps);
    expect({ opens, reads }).toEqual({ opens: 1, reads: 1 });
    expect(getSourceStatuses(deps.db).find((s) => s.source === "opencode")?.warnings).toContain("io-pending");

    releaseClose!();
    await new Promise((resolve) => setTimeout(resolve, 20));
    await runNativeUsagePass(deps);
    expect(opens).toBe(2);
    deps.db.close();
  });
});

describe("interrupted discovery registration and directory failure mapping", () => {
  function countFiles(deps: NativeCollectorDeps): number {
    return (deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_files WHERE source = 'claude'").get() as { n: number }).n;
  }

  function countItems(deps: NativeCollectorDeps): { n: number; distinct_count: number } {
    return deps.db.prepare("SELECT COUNT(*) AS n, COUNT(DISTINCT item_id) AS distinct_count FROM native_usage_items").get() as {
      n: number;
      distinct_count: number;
    };
  }

  function statusOf(deps: NativeCollectorDeps) {
    return getSourceStatuses(deps.db).find((s) => s.source === "claude");
  }

  it("replays an interrupted registration beyond the first 64 entries and defers a terminal directory to the next cadence", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    for (let i = 0; i < 70; i += 1) {
      writeClaude(deps, `proj/f${String(i).padStart(2, "0")}.jsonl`, [claudeLine(`m-${i}`)]);
    }
    writeClaude(deps, "zz-other/good.jsonl", [claudeLine("m-good", { sessionId: "sess-good" })]);
    const stuckAbs = join(deps.roots.claude, "proj", "f65.jsonl");
    const projAbs = join(deps.roots.claude, "proj");
    const sortedDir = (real: Awaited<ReturnType<NativeFs["opendir"]>>): typeof real => {
      let buffered: DirEntryLike[] | null = null;
      return {
        read: async () => {
          if (buffered === null) {
            const entries: DirEntryLike[] = [];
            for (;;) {
              const entry = await real.read();
              if (entry === null) break;
              entries.push(entry);
            }
            buffered = entries.sort((a, b) => a.name.localeCompare(b.name));
          }
          return buffered.shift() ?? null;
        },
        close: () => real.close(),
      };
    };
    let stuckCalls = 0;
    let fault = true;
    let releaseStuck: (() => void) | null = null;
    deps.fs = {
      ...nativeFs,
      opendir: async (path) => {
        const real = await nativeFs.opendir(path);
        return path === projAbs ? sortedDir(real) : real;
      },
      realpath: async (path) => {
        if (fault && path === stuckAbs) {
          stuckCalls += 1;
          const real = await fsRealpathAsync(path);
          return new Promise<string>((resolve) => {
            releaseStuck = () => resolve(real);
          });
        }
        return nativeFs.realpath(path);
      },
    };
    let now = NOW;
    const tick = () => (now += 61);
    const pass = () => runNativeUsagePass({ ...deps, now: tick, nowMs: () => 0 });

    while (stuckCalls === 0) await pass();
    expect(stuckCalls).toBe(1);
    expect(statusOf(deps)?.state).toBe("indexing");
    expect(statusOf(deps)?.warnings).toContain("io-timeout");
    expect(statusOf(deps)?.state).not.toBe("empty");
    expect(countFiles(deps)).toBe(65);

    await pass();
    expect(stuckCalls).toBe(1);
    releaseStuck!();
    await new Promise((resolve) => setTimeout(resolve, 20));

    while (stuckCalls < 2) await pass();
    expect(stuckCalls).toBe(2);
    expect(countFiles(deps)).toBe(65);
    expect(statusOf(deps)?.warnings).toContain("io-timeout");
    releaseStuck!();
    await new Promise((resolve) => setTimeout(resolve, 20));

    for (let round = 0; round < 8; round += 1) {
      await pass();
      if (getPersistentWarnings(deps.db, "claude").includes("discovery-failed")) break;
    }
    expect(getPersistentWarnings(deps.db, "claude")).toContain("discovery-failed");
    expect(statusOf(deps)?.warnings).toContain("discovery-failed");
    expect(countFiles(deps)).toBe(66);

    // Within the 60 s cadence the terminal directory is not retried and the failure
    // stays visible on later polls.
    const before = stuckCalls;
    now += 1;
    await runNativeUsagePass({ ...deps, now: () => now, nowMs: () => 0 });
    expect(stuckCalls).toBe(before);
    expect(statusOf(deps)?.warnings).toContain("discovery-failed");

    fault = false;
    let settled = 0;
    while (settled < 30 && countItems(deps).n < 71) {
      await pass();
      settled += 1;
    }
    expect(countItems(deps)).toEqual({ n: 71, distinct_count: 71 });
    expect(countFiles(deps)).toBe(71);
    expect(getPersistentWarnings(deps.db, "claude")).toEqual([]);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id IN ('m-65','m-66','m-67','m-68','m-69')").get())
      .toEqual({ n: 5 });
    deps.db.close();
  }, 60_000);

  it("owns a timed-out directory read and its close before reopening the directory", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    writeClaude(deps, "p/one.jsonl", [claudeLine("m-one")]);
    const pAbs = join(deps.roots.claude, "p");
    let reads = 0;
    let closes = 0;
    let readStarted = false;
    let releaseRead: (() => void) | null = null;
    let releaseClose: (() => void) | null = null;
    let resolveReadStarted: (() => void) | null = null;
    const readStartedBarrier = new Promise<void>((resolve) => { resolveReadStarted = resolve; });
    deps.fs = {
      ...nativeFs,
      opendir: async (path) => {
        const real = await nativeFs.opendir(path);
        if (path !== pAbs) return real;
        return {
          read: () => {
            reads += 1;
            if (reads === 1) {
              readStarted = true;
              resolveReadStarted?.();
              return new Promise<DirEntryLike | null>((resolve) => {
                releaseRead = () =>
                  resolve({ name: "one.jsonl", isFile: () => true, isDirectory: () => false, isSymbolicLink: () => false });
              });
            }
            return real.read();
          },
          close: async () => {
            closes += 1;
            if (closes === 1) {
              return new Promise<void>((resolve) => {
                releaseClose = async () => {
                  await real.close();
                  resolve();
                };
              });
            }
            await real.close();
          },
        };
      },
    };

    // The readiness barrier must use a REAL timer, captured before fake timers are enabled: the
    // production pass first performs real fs work, so advancing virtual time cannot move it.
    const realSetTimeout = globalThis.setTimeout;
    const realClearTimeout = globalThis.clearTimeout;
    let readinessTimer: ReturnType<typeof setTimeout> | undefined;
    const realReadinessFailure = new Promise<void>((_, reject) => {
      readinessTimer = realSetTimeout(() => reject(new Error("directory read did not start")), 5_000);
      readinessTimer.unref?.();
    });
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      const pass = runNativeUsagePass({ ...deps, now: () => NOW });
      await Promise.race([readStartedBarrier, realReadinessFailure]);
      expect(readStarted).toBe(true);
      await vi.advanceTimersByTimeAsync(1400);
      await pass;
    } finally {
      if (readinessTimer) realClearTimeout(readinessTimer);
      vi.useRealTimers();
    }
    expect(statusOf(deps)?.state).toBe("indexing");
    expect(statusOf(deps)?.warnings).toContain("io-timeout");

    releaseRead!();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect({ reads, closes }).toEqual({ reads: 1, closes: 1 });
    expect(releaseClose).not.toBeNull();

    await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
    expect(reads).toBe(1);
    expect(statusOf(deps)?.warnings).toContain("io-pending");

    releaseClose!();
    await new Promise((resolve) => setTimeout(resolve, 20));
    let settled = 0;
    while (settled < 8) {
      const counted = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-one'").get() as { n: number };
      if (counted.n === 1) break;
      await runNativeUsagePass({ ...deps, now: () => NOW + 61 * (settled + 1) });
      settled += 1;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-one'").get()).toEqual({ n: 1 });
    deps.db.close();
  }, 60_000);

  it("treats a normally absent optional subagents directory as quiet", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    writeClaude(deps, "proj/lead.jsonl", [claudeLine("m-lead")]);
    mkdirSync(join(deps.roots.claude, "proj", "3f0c2a51-8a4b-4c5d-9e6f-0123456789ab"), { recursive: true });
    for (let pass = 0; pass < 8; pass += 1) {
      await runNativeUsagePass({ ...deps, now: () => NOW + pass * 61 });
      if (statusOf(deps)?.state === "available") break;
    }
    expect(statusOf(deps)?.warnings).not.toContain("symlink-escape");
    expect(statusOf(deps)?.warnings).not.toContain("subdir-failed");
    expect(getPersistentWarnings(deps.db, "claude")).toEqual([]);
    expect(
      deps.db.prepare("SELECT state, last_warning FROM native_usage_dirs WHERE source = 'claude' AND path = ?")
        .get("proj/3f0c2a51-8a4b-4c5d-9e6f-0123456789ab/subagents"),
    ).toEqual({ state: "done", last_warning: null });
    deps.db.close();
  }, 30_000);

  it("classifies an unreadable queued directory as subdir-failed, never as a symlink escape", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    writeClaude(deps, "good/lead.jsonl", [claudeLine("m-good")]);
    writeClaude(deps, "bad/lead-bad.jsonl", [claudeLine("m-bad")]);
    const badAbs = join(deps.roots.claude, "bad");
    let calls = 0;
    deps.fs = {
      ...nativeFs,
      realpath: async (path) => {
        if (path === badAbs) {
          calls += 1;
          if (calls % 2 === 0) {
            const error = new Error("EACCES") as Error & { code: string };
            error.code = "EACCES";
            throw error;
          }
        }
        return nativeFs.realpath(path);
      },
    };
    for (let pass = 0; pass < 6; pass += 1) await runNativeUsagePass({ ...deps, now: () => NOW + pass * 61 });
    const status = statusOf(deps);
    expect(status?.warnings).toContain("subdir-failed");
    expect(status?.warnings).not.toContain("symlink-escape");
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-good'").get()).toEqual({ n: 1 });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-bad'").get()).toEqual({ n: 0 });
    expect(deps.db.prepare("SELECT state, last_warning FROM native_usage_dirs WHERE source = 'claude' AND path = 'bad'").get())
      .toEqual({ state: "done", last_warning: "subdir-failed" });
    deps.db.close();
  }, 30_000);

  it("keeps a new cycle from clearing an unrevisited directory failure", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    for (let i = 0; i < 40; i += 1) {
      writeClaude(deps, `a${String(i).padStart(2, "0")}/f.jsonl`, [claudeLine(`m-a${i}`)]);
    }
    writeClaude(deps, "zbad/f.jsonl", [claudeLine("m-bad")]);
    const badAbs = join(deps.roots.claude, "zbad");
    let calls = 0;
    let fault = true;
    deps.fs = {
      ...nativeFs,
      realpath: async (path) => {
        if (path === badAbs) {
          calls += 1;
          if (fault && calls % 2 === 0) {
            const error = new Error("EACCES") as Error & { code: string };
            error.code = "EACCES";
            throw error;
          }
        }
        return nativeFs.realpath(path);
      },
    };
    let now = NOW;
    const pass = () => runNativeUsagePass({ ...deps, now: () => (now += 61) });
    const cycle = () =>
      (deps.db.prepare("SELECT cycle FROM native_usage_discovery WHERE source = 'claude'").get() as { cycle: number })
        .cycle;
    const badItem = () =>
      deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-bad'").get() as { n: number };

    await pass();
    const firstCycle = cycle();
    let seenNewCycleWithWarning = false;
    for (let passIndex = 0; passIndex < 40; passIndex += 1) {
      await pass();
      if (!getPersistentWarnings(deps.db, "claude").includes("subdir-failed")) continue;
      if (cycle() > firstCycle) {
        seenNewCycleWithWarning = true;
        break;
      }
    }
    expect(seenNewCycleWithWarning).toBe(true);
    expect(
      deps.db.prepare("SELECT last_warning FROM native_usage_dirs WHERE source = 'claude' AND path = 'zbad'").get(),
    ).toEqual({ last_warning: "subdir-failed" });
    expect(statusOf(deps)?.warnings).toContain("subdir-failed");
    expect(badItem()).toEqual({ n: 0 });

    fault = false;
    let settled = 0;
    while (settled < 30 && badItem().n === 0) {
      await pass();
      settled += 1;
    }
    expect(badItem()).toEqual({ n: 1 });
    expect(getPersistentWarnings(deps.db, "claude")).not.toContain("subdir-failed");
    deps.db.close();
  }, 60_000);

  it("reports a root that becomes unreadable after validation as unavailable", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    writeClaude(deps, "p/lead.jsonl", [claudeLine("m-1")]);
    let rootCalls = 0;
    deps.fs = {
      ...nativeFs,
      realpath: async (path) => {
        if (path === deps.roots.claude) {
          rootCalls += 1;
          if (rootCalls > 1) {
            const error = new Error("EACCES") as Error & { code: string };
            error.code = "EACCES";
            throw error;
          }
        }
        return nativeFs.realpath(path);
      },
    };
    await runNativeUsagePass({ ...deps, now: () => NOW });
    const status = statusOf(deps);
    expect(status?.state).toBe("unavailable");
    expect(status?.warnings).toContain("root-unavailable");
    expect(status?.warnings).not.toContain("symlink-escape");
    deps.db.close();
  });
});

describe("strict OpenCode helper reply contract", () => {
  const previous = {
    v: 1 as const,
    v1: { after: "10", ceiling: "20", done: false },
    v2: { after: null, ceiling: null, done: false },
  };
  const base = {
    v: 1, ok: true, runtime: "Bun 1.3.14", schema: "1.18.34", rows: [] as unknown[],
    next: { v1: { after: "12", ceiling: "20", done: false }, v2: { after: null, ceiling: null, done: false } },
    warnings: {} as Record<string, unknown>,
  };
  const parse = (over: Record<string, unknown>) => parseOpenCodeReply(JSON.stringify({ ...base, ...over }), previous);

  it("rejects unknown warning keys, noninteger counts, nonboolean done and frame sizes", () => {
    expect(parse({ warnings: { mystery: 1 } })).toBeNull();
    expect(parse({ warnings: { malformed: 1.5 } })).toBeNull();
    expect(parse({ next: { ...base.next, v1: { after: "12", ceiling: "20", done: "yes" } } })).toBeNull();
    expect(parse({ rows: ["not-an-object"] })).toBeNull();
    expect(parse({ rows: Array.from({ length: 513 }, () => ({})) })).toBeNull();
    expect(parse({ padding: "p".repeat(17 * 1024) })).toBeNull();
    expect(parse({ rows: Array.from({ length: 16 }, () => ({ pad: "p".repeat(16 * 1024) })) })).toBeNull();
  });

  it("rejects alternate rowid spellings while accepting the signed minimum", () => {
    expect(parse({ next: { ...base.next, v1: { after: "-0", ceiling: "20", done: false } } })).toBeNull();
    expect(parse({ next: { ...base.next, v1: { after: "007", ceiling: "20", done: false } } })).toBeNull();
    const min = parseOpenCodeReply(JSON.stringify({
      ...base,
      next: {
        v1: { after: "-9223372036854775808", ceiling: "-9223372036854775808", done: true },
        v2: { after: null, ceiling: null, done: true },
      },
    }), { v: 1, v1: { after: null, ceiling: null, done: false }, v2: { after: null, ceiling: null, done: false } });
    expect(min?.next.v1.after).toBe("-9223372036854775808");
  });
});

// ---------------------------------------------------------------------------
// Refresh lifecycle, stable tails, failure categories and helper diagnostics.

describe("background refresh and genuinely pending work", () => {
  const statusOf = (deps: NativeCollectorDeps, source: "claude" | "codex" | "opencode" = "claude") =>
    getSourceStatuses(deps.db).find((s) => s.source === source);
  const cycleOf = (deps: NativeCollectorDeps) =>
    (deps.db.prepare("SELECT cycle, state, last_complete FROM native_usage_discovery WHERE source = 'claude'").get() as
      | { cycle: number; state: string; last_complete: number | null }
      | undefined);
  const itemTotal = (deps: NativeCollectorDeps) =>
    (deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number }).n;
  const steppingPass = (deps: NativeCollectorDeps) => {
    let now = NOW;
    return () => {
      now += 61;
      const at = now;
      return runNativeUsagePass({ ...deps, now: () => at });
    };
  };

  it("keeps a completed history refresh as background availability and paces new pending files", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    for (let i = 0; i < 70; i += 1) {
      writeClaude(deps, `p${String(i).padStart(2, "0")}/f.jsonl`, [claudeLine(`m-${i}`, { sessionId: `s-${i}` })]);
    }
    let clock = NOW;
    const pass = () => {
      clock += 61;
      const at = clock;
      return runNativeUsagePass({ ...deps, now: () => at });
    };
    let guard = 0;
    const states: string[] = [];
    let discoveryPasses = 0;
    while (guard < 200) {
      await pass();
      guard += 1;
      discoveryPasses += 1;
      const status = statusOf(deps);
      states.push(status?.state ?? "none");
      if (cycleOf(deps)?.state === "complete") break;
    }
    // D: the finite fixture's deterministic discovery depth for one complete cycle.
    // Every later insertion is measured against this derived bound, not a magic number.
    const D = discoveryPasses;
    expect(D).toBeGreaterThan(0);
    expect(states).toContain("indexing");
    expect(cycleOf(deps)?.state).toBe("complete");
    while (itemTotal(deps) < 70 && guard < 300) {
      await pass();
      guard += 1;
    }
    expect(itemTotal(deps)).toBe(70);
    expect(statusOf(deps)?.state).toBe("available");

    // Public cadence: a sub-minute poll after completion starts no background work and
    // does not move the last successful refresh.
    const completedCycle = cycleOf(deps)!.cycle;
    const completedRefresh = statusOf(deps)!.lastRefresh;
    await runNativeUsagePass({ ...deps, now: () => clock + 30 });
    expect(cycleOf(deps)!.cycle).toBe(completedCycle);
    expect(statusOf(deps)!.lastRefresh).toBe(completedRefresh);

    // The next full-cadence pass starts a refresh cycle over already-indexed history: it
    // must stay available, surface the background diagnostic and never fall back to
    // steady indexing.
    const initialCycle = cycleOf(deps)!.cycle;
    while (cycleOf(deps)!.cycle === initialCycle && guard < 400) {
      await pass();
      guard += 1;
    }
    expect(cycleOf(deps)!.cycle).toBeGreaterThan(initialCycle);
    const refreshCycle = cycleOf(deps)!.cycle;
    let refreshPasses = 0;
    const refreshStates: string[] = [];
    while (guard < 600) {
      await pass();
      guard += 1;
      refreshPasses += 1;
      const status = statusOf(deps);
      refreshStates.push(status?.state ?? "none");
      if (cycleOf(deps)!.cycle !== refreshCycle) break;
      if (cycleOf(deps)!.state === "complete") break;
    }
    expect(refreshStates.every((s) => s === "available")).toBe(true);
    expect(refreshPasses).toBeGreaterThan(0);

    // Wait for the next cycle to start, then add files before its final unvisited
    // directory (p69 sorts last). They must be found within that cycle's D-pass depth.
    while (cycleOf(deps)!.cycle === refreshCycle && guard < 800) {
      await pass();
      guard += 1;
    }
    const discoveryCycle = cycleOf(deps)!.cycle;
    expect(discoveryCycle).toBeGreaterThan(refreshCycle);
    for (let i = 0; i < 8; i += 1) {
      writeClaude(deps, `p69/extra${i}.jsonl`, [claudeLine(`m-extra-${i}`, { sessionId: `s-extra-${i}` })]);
    }
    let extraDiscoveryPasses = 0;
    let pendingStateSeen = false;
    while (guard < 1000) {
      await pass();
      guard += 1;
      extraDiscoveryPasses += 1;
      if (statusOf(deps)?.state === "indexing") pendingStateSeen = true;
      const registered = deps.db
        .prepare("SELECT COUNT(*) AS n FROM native_usage_files WHERE source = 'claude' AND identity LIKE 'p69/extra%'")
        .get() as { n: number };
      if (registered.n === 8 && cycleOf(deps)!.state === "complete") break;
      if (cycleOf(deps)!.cycle !== discoveryCycle) break;
    }
    expect(extraDiscoveryPasses).toBeLessThanOrEqual(D);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_files WHERE source = 'claude' AND identity LIKE 'p69/extra%'").get())
      .toEqual({ n: 8 });
    expect(pendingStateSeen).toBe(true);
    let extraGuard = 0;
    while (extraGuard < 30) {
      const row = deps.db.prepare("SELECT 1 AS n FROM native_usage_items WHERE item_id = 'm-extra-0'").get() as { n: number } | undefined;
      if (row) break;
      await pass();
      extraGuard += 1;
    }
    expect(deps.db.prepare("SELECT 1 AS n FROM native_usage_items WHERE item_id = 'm-extra-0'").get()).toEqual({ n: 1 });

    // An insertion behind the cursor (p00 already enumerated) is not found in the rest
    // of that cycle; the following complete cycle re-enumerates it and picks it up.
    while (!(cycleOf(deps)!.state === "complete" && cycleOf(deps)!.cycle === discoveryCycle) && guard < 1200) {
      await pass();
      guard += 1;
    }
    while (cycleOf(deps)!.cycle === discoveryCycle && guard < 1400) {
      await pass();
      guard += 1;
    }
    const cursorCycle = cycleOf(deps)!.cycle;
    expect(cursorCycle).toBeGreaterThan(discoveryCycle);
    let sawDoneDir = false;
    while (!sawDoneDir && cycleOf(deps)!.cycle === cursorCycle && guard < 1600) {
      await pass();
      guard += 1;
      const dir = deps.db.prepare("SELECT state, cycle FROM native_usage_dirs WHERE source = 'claude' AND path = 'p00'").get() as
        | { state: string; cycle: number }
        | undefined;
      sawDoneDir = dir?.cycle === cursorCycle && dir.state === "done";
    }
    expect(sawDoneDir).toBe(true);
    expect(cycleOf(deps)!.cycle).toBe(cursorCycle);
    writeClaude(deps, "p00/behind.jsonl", [claudeLine("m-behind", { sessionId: "s-behind" })]);
    let behindSeenInCycle = false;
    while (cycleOf(deps)!.cycle === cursorCycle && guard < 1800) {
      await pass();
      guard += 1;
      if (cycleOf(deps)!.cycle !== cursorCycle) break;
      behindSeenInCycle = deps.db
        .prepare("SELECT 1 AS n FROM native_usage_files WHERE source = 'claude' AND identity = 'p00/behind.jsonl'")
        .get() !== undefined;
      if (behindSeenInCycle) break;
    }
    expect(behindSeenInCycle).toBe(false);
    expect(cycleOf(deps)!.cycle).toBeGreaterThan(cursorCycle);
    const behindRow = () => deps.db
      .prepare("SELECT 1 AS n FROM native_usage_files WHERE source = 'claude' AND identity = 'p00/behind.jsonl'")
      .get() as { n: number } | undefined;
    let nextPasses = 0;
    while (!behindRow() && nextPasses < D && guard < 2000) {
      await pass();
      guard += 1;
      nextPasses += 1;
    }
    expect(behindRow()).toEqual({ n: 1 });
    expect(nextPasses).toBeLessThanOrEqual(D);
    deps.db.close();
  }, 120_000);

  it("settles a physically unchanged unterminated tail without losing append recovery", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    const complete = claudeLine("m-complete");
    const partial = JSON.stringify({
      type: "assistant", sessionId: "sess-1", timestamp: "2026-11-17T11:00:00Z",
      message: { id: "m-tail", usage: { input_tokens: 2, output_tokens: 2 } },
    });
    const path = join(deps.roots.claude, "p/partial.jsonl");
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, `${complete}\n${partial.slice(0, 20)}`);
    let now = NOW;
    const pass = () => {
      now += 61;
      const at = now;
      return runNativeUsagePass({ ...deps, now: () => at });
    };

    await pass();
    let cp = getFileCheckpoint(deps.db, "claude", "p/partial.jsonl");
    expect(cp?.done).toBe(false);
    const preTail = cp!.offset;
    expect(preTail).toBeLessThan(cp!.size);

    await pass();
    cp = getFileCheckpoint(deps.db, "claude", "p/partial.jsonl");
    expect(cp?.done).toBe(true);
    expect(cp?.offset).toBe(preTail);
    expect(getPersistentWarnings(deps.db, "claude")).toContain("incomplete-tail");
    expect(statusOf(deps)?.state).toBe("available");

    // The unchanged fast path preserves and re-emits the warning.
    await pass();
    expect(getFileCheckpoint(deps.db, "claude", "p/partial.jsonl")).toMatchObject({ done: true, offset: preTail });
    expect(getPersistentWarnings(deps.db, "claude")).toContain("incomplete-tail");
    expect(statusOf(deps)?.warnings).toContain("incomplete-tail");
    expect(statusOf(deps)?.state).toBe("available");

    appendFileSync(path, `${partial.slice(20)}\n`);
    for (let passIndex = 0; passIndex < 6; passIndex += 1) {
      await pass();
      const count = deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get() as { n: number };
      if (count.n === 2) break;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 2 });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-tail'").get()).toEqual({ n: 1 });
    expect(getPersistentWarnings(deps.db, "claude")).not.toContain("incomplete-tail");
    expect(getFileCheckpoint(deps.db, "claude", "p/partial.jsonl")).toMatchObject({ done: true });

    rmSync(path);
    writeFileSync(path, `${complete}\n`);
    for (let passIndex = 0; passIndex < 6; passIndex += 1) {
      await pass();
      const cp = getFileCheckpoint(deps.db, "claude", "p/partial.jsonl");
      if (cp?.done && cp.offset === cp.size) break;
    }
    expect(getFileCheckpoint(deps.db, "claude", "p/partial.jsonl")).toMatchObject({ done: true, offset: fsStatSync(path).size });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-complete'").get()).toEqual({ n: 1 });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 2 });
    expect(getPersistentWarnings(deps.db, "claude")).not.toContain("incomplete-tail");
    deps.db.close();
  }, 60_000);

  it("indexes the first OpenCode cycle then keeps completed re-sweeps available", async () => {
    const row = {
      projection: "v2", session_id: "ses-1", message_id: "msg-1", parent_session_id: null,
      model_id: "model-x", provider_id: "provider-x", agent: "build",
      created: 1_794_909_600_000, updated: 1_794_909_800_000,
      cost: 12.5, input: 100, output: 30, reasoning: 7, cache_read: 11, cache_write: 5,
    };
    const reply = (next: unknown, rows: unknown[] = []) =>
      JSON.stringify({ v: 1, ok: true, runtime: "Bun 1.3.14", schema: "1.18.34", rows, next, warnings: {} });
    const firstPage = reply({ v1: { after: "1", ceiling: "1", done: true }, v2: { after: "1", ceiling: "2", done: false } }, [row]);
    const completePage = reply({ v1: { after: "1", ceiling: "1", done: true }, v2: { after: "2", ceiling: "2", done: true } });
    const resweepPage = reply({ v1: { after: "1", ceiling: "1", done: true }, v2: { after: "1", ceiling: "2", done: false } }, [
      { ...row, updated: 1_794_910_000_000, output: 40 },
    ]);
    const resweepPage2 = reply({ v1: { after: "1", ceiling: "1", done: true }, v2: { after: "2", ceiling: "2", done: false } }, [
      { ...row, message_id: "msg-2", output: 5, reasoning: 0, cost: 1 },
    ]);
    const resweepDone = reply({ v1: { after: "1", ceiling: "1", done: true }, v2: { after: "2", ceiling: "2", done: true } });
    let calls = 0;
    const argvStates: string[] = [];
    const deps = makeDeps({
      enabled: { claude: false, codex: false, opencode: true },
      exec: async (_file, args) => {
        argvStates.push(args[6]);
        calls += 1;
        if (calls === 1) return { stdout: firstPage };
        if (calls === 2) return { stdout: completePage };
        if (calls === 3) return { stdout: resweepPage };
        if (calls === 4) return { stdout: resweepPage2 };
        return { stdout: resweepDone };
      },
    });
    writeFileSync(deps.roots.opencodeDb, "");
    let now = NOW;
    const pass = () => {
      now += 61;
      const at = now;
      return runNativeUsagePass({ ...deps, now: () => at });
    };

    await pass();
    expect(statusOf(deps, "opencode")?.state).toBe("indexing");
    expect(getOpenCodeCursor(deps.db).complete).toBe(false);
    expect(Object.keys(JSON.parse(argvStates[0])).sort()).toEqual(["v", "v1", "v2"]);

    await pass();
    expect(statusOf(deps, "opencode")?.state).toBe("available");
    expect(getOpenCodeCursor(deps.db).complete).toBe(true);
    expect(getOpenCodeCursor(deps.db)).toMatchObject({
      v1: { after: null, ceiling: null, done: false },
      v2: { after: null, ceiling: null, done: false },
    });

    // A re-sweep after first completion is background availability, not indexing.
    const callsBeforeWait = calls;
    await runNativeUsagePass({ ...deps, now: () => now + 30 });
    expect(calls).toBe(callsBeforeWait);
    await pass();
    expect(statusOf(deps, "opencode")?.state).toBe("available");
    expect(statusOf(deps, "opencode")?.warnings).not.toContain("background-refresh");
    expect(getOpenCodeCursor(deps.db).complete).toBe(true);
    expect(Object.keys(JSON.parse(argvStates.at(-1)!)).sort()).toEqual(["v", "v1", "v2"]);
    const updated = queryNativeUsage(deps.db, { period: 30, now: NOW });
    expect(updated.byAgent.find((a) => a.name === "OpenCode")?.allOutput).toBe(47);

    // Several bounded background pages stay available until the re-sweep completes.
    await pass();
    expect(statusOf(deps, "opencode")?.state).toBe("available");
    expect(statusOf(deps, "opencode")?.warnings).not.toContain("background-refresh");
    expect(queryNativeUsage(deps.db, { period: 30, now: NOW }).byAgent.find((a) => a.name === "OpenCode")?.allOutput).toBe(52);

    await pass();
    expect(statusOf(deps, "opencode")?.state).toBe("available");
    expect(statusOf(deps, "opencode")?.warnings).not.toContain("background-refresh");
    expect(getOpenCodeCursor(deps.db).complete).toBe(true);

    // Restart: the marker survives and no helper input widens the strict protocol.
    const restarted = { ...deps, exec: async () => ({ stdout: reply({ v1: { after: null, ceiling: null, done: true }, v2: { after: null, ceiling: null, done: true } }) }) };
    now += 61;
    await runNativeUsagePass({ ...restarted, now: () => now });
    expect(getOpenCodeCursor(deps.db).complete).toBe(true);
    expect(statusOf(deps, "opencode")?.state).toBe("available");
    deps.db.close();
  }, 60_000);
});

describe("real failure categories and quiet non-usage records", () => {
  const statusOf = (deps: NativeCollectorDeps, source: "claude" | "codex" | "opencode" = "claude") =>
    getSourceStatuses(deps.db).find((s) => s.source === source);
  const registryRow = (deps: NativeCollectorDeps, identity: string) =>
    deps.db.prepare("SELECT * FROM native_usage_files WHERE source = 'claude' AND identity = ?").get(identity) as
      | { last_warning: string | null; done: number; offset: number }
      | undefined;

  it("retires only a confirmed missing file after a complete cycle and keeps its usage", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    writeClaude(deps, "proj/keep.jsonl", [claudeLine("m-keep", { sessionId: "s-keep" })]);
    writeClaude(deps, "proj/gone.jsonl", [claudeLine("m-gone", { sessionId: "s-gone" })]);
    let now = NOW;
    const pass = () => {
      now += 61;
      const at = now;
      return runNativeUsagePass({ ...deps, now: () => at });
    };
    for (let i = 0; i < 8; i += 1) {
      await pass();
      if (statusOf(deps)?.state === "available") break;
    }
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 2 });
    rmSync(join(deps.roots.claude, "proj", "gone.jsonl"));
    const seen = new Set<string>();
    for (let i = 0; i < 40; i += 1) {
      await pass();
      for (const warning of statusOf(deps)?.warnings ?? []) seen.add(warning);
      if (registryRow(deps, "proj/gone.jsonl") === undefined) break;
    }
    expect([...seen]).not.toContain("symlink-escape");
    expect(registryRow(deps, "proj/gone.jsonl")).toBeUndefined();
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 2 });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-gone'").get()).toEqual({ n: 1 });
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-keep'").get()).toEqual({ n: 1 });

    // Reappearance is a fresh safe discovery; dedupe keeps the measured item single.
    writeClaude(deps, "proj/gone.jsonl", [claudeLine("m-gone", { sessionId: "s-gone" })]);
    for (let i = 0; i < 30; i += 1) {
      await pass();
      if (registryRow(deps, "proj/gone.jsonl")) break;
    }
    expect(registryRow(deps, "proj/gone.jsonl")).toBeDefined();
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-gone'").get()).toEqual({ n: 1 });
    deps.db.close();
  }, 60_000);

  it("keeps a rediscovered unreadable file with a read-failed fact instead of retiring it", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    writeClaude(deps, "proj/gone.jsonl", [claudeLine("m-gone")]);
    let now = NOW;
    const pass = () => {
      now += 61;
      const at = now;
      return runNativeUsagePass({ ...deps, now: () => at });
    };
    for (let i = 0; i < 8; i += 1) {
      await pass();
      if (statusOf(deps)?.state === "available") break;
    }
    appendFileSync(join(deps.roots.claude, "proj", "gone.jsonl"), `${claudeLine("m-more", { sessionId: "s-more" })}\n`);
    deps.fs = {
      ...nativeFs,
      open: async (target) => {
        if (target.endsWith("gone.jsonl")) throw new Error("EACCES");
        return nativeFs.open(target);
      },
    };
    for (let i = 0; i < 30; i += 1) {
      await pass();
      if (registryRow(deps, "proj/gone.jsonl")?.last_warning === "read-failed") break;
    }
    expect(registryRow(deps, "proj/gone.jsonl")?.last_warning).toBe("read-failed");
    for (let i = 0; i < 40; i += 1) await pass();
    expect(registryRow(deps, "proj/gone.jsonl")?.last_warning).toBe("read-failed");
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-gone'").get()).toEqual({ n: 1 });
    deps.db.close();
  }, 60_000);

  it("classifies realpath failures without mislabelling them as symlink escapes", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    writeClaude(deps, "proj/denied.jsonl", [claudeLine("m-denied")]);
    const deniedAbs = join(deps.roots.claude, "proj", "denied.jsonl");
    deps.fs = {
      ...nativeFs,
      realpath: async (target) => {
        if (target === deniedAbs) {
          const error = new Error("EACCES") as Error & { code: string };
          error.code = "EACCES";
          throw error;
        }
        return nativeFs.realpath(target);
      },
    };
    let now = NOW;
    for (let i = 0; i < 4; i += 1) await runNativeUsagePass({ ...deps, now: () => (now += 61) });
    const denied = statusOf(deps);
    expect(denied?.warnings).toContain("read-failed");
    expect(denied?.warnings).not.toContain("symlink-escape");
    expect(denied?.state).toBe("unavailable");

    const missingDeps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    writeClaude(missingDeps, "proj/vanishing.jsonl", [claudeLine("m-vanish")]);
    const vanishAbs = join(missingDeps.roots.claude, "proj", "vanishing.jsonl");
    missingDeps.fs = {
      ...nativeFs,
      realpath: async (target) => {
        if (target === vanishAbs) {
          const error = new Error("ENOENT") as Error & { code: string };
          error.code = "ENOENT";
          throw error;
        }
        return nativeFs.realpath(target);
      },
    };
    let missingNow = NOW;
    for (let i = 0; i < 4; i += 1) await runNativeUsagePass({ ...missingDeps, now: () => (missingNow += 61) });
    const vanished = statusOf(missingDeps);
    expect(vanished?.warnings).toContain("missing-file");
    expect(vanished?.warnings).not.toContain("symlink-escape");
    deps.db.close();
    missingDeps.db.close();
  }, 60_000);

  const seedMeasuredFile = (deps: NativeCollectorDeps, rel: string, id: string) => {
    const line = claudeLine(id);
    writeClaude(deps, rel, [line]);
    const parsed = parseClaudeLine(line, { child: false, now: NOW });
    if (!("item" in parsed)) throw new Error(parsed.skip);
    applyFileBatch(deps.db, {
      checkpoint: {
        source: "claude", identity: rel, generation: "1:1", dev: "1", ino: "1",
        size: line.length, mtime: NOW, offset: line.length, done: true, processedAt: NOW,
        child: false, discarding: false,
      },
      expectedPrior: null,
      items: [parsed.item],
      now: NOW,
    });
  };

  it("keeps a discovered unreadable file with a read-failed fact instead of retiring it", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    seedMeasuredFile(deps, "proj/denied.jsonl", "m-denied");
    // Backoff makes ingestion skip this row in the pass where discovery completes.
    markFileFailure(deps.db, "claude", "proj/denied.jsonl", NOW + 60, NOW, "missing-file");
    const deniedAbs = join(deps.roots.claude, "proj", "denied.jsonl");
    deps.fs = {
      ...nativeFs,
      realpath: async (target) => {
        if (target === deniedAbs) {
          const error = new Error("EACCES") as Error & { code: string };
          error.code = "EACCES";
          throw error;
        }
        return nativeFs.realpath(target);
      },
    };
    let now = NOW;
    const pass = () => runNativeUsagePass({ ...deps, now: () => now });
    await pass();
    for (let i = 0; i < 10; i += 1) {
      now += 61;
      await pass();
      if (registryRow(deps, "proj/denied.jsonl")?.last_warning === "read-failed") break;
    }
    expect(registryRow(deps, "proj/denied.jsonl")?.last_warning).toBe("read-failed");
    for (let i = 0; i < 6; i += 1) {
      now += 61;
      await pass();
    }
    expect(registryRow(deps, "proj/denied.jsonl")?.last_warning).toBe("read-failed");
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE item_id = 'm-denied'").get()).toEqual({ n: 1 });
    expect(statusOf(deps)?.warnings).not.toContain("symlink-escape");
    deps.db.close();
  }, 60_000);

  it("does not retire missing rows beneath an unreadable ancestor", async () => {
    const deps = makeDeps({ enabled: { claude: true, codex: false, opencode: false } });
    seedMeasuredFile(deps, "good/keep.jsonl", "m-keep");
    seedMeasuredFile(deps, "bad/x.jsonl", "m-x");
    markFileFailure(deps.db, "claude", "bad/x.jsonl", NOW + 60, NOW, "missing-file");
    const badDir = join(deps.roots.claude, "bad");
    const badFile = join(badDir, "x.jsonl");
    deps.fs = {
      ...nativeFs,
      realpath: async (target) => {
        if (target === badDir || target === badFile) {
          const error = new Error("EACCES") as Error & { code: string };
          error.code = "EACCES";
          throw error;
        }
        return nativeFs.realpath(target);
      },
    };
    let now = NOW;
    await runNativeUsagePass({ ...deps, now: () => now });
    for (let i = 0; i < 12; i += 1) {
      now += 61;
      await runNativeUsagePass({ ...deps, now: () => now });
    }
    expect(registryRow(deps, "bad/x.jsonl")).toBeDefined();
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 2 });
    expect(statusOf(deps)?.warnings).toContain("read-failed");
    expect(statusOf(deps)?.warnings).not.toContain("symlink-escape");
    deps.db.close();
  }, 60_000);

  it("keeps ordinary Codex records quiet while real accounting failures still warn", async () => {
    const deps = makeDeps({ enabled: { claude: false, codex: true, opencode: false } });
    const dir = join(deps.roots.codex, "2026", "11", "17");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "rollout-mixed.jsonl"), [
      JSON.stringify({ type: "session_meta", timestamp: "2026-11-17T09:00:00Z", payload: { id: "root-1", instructions: "SENTINEL" } }),
      JSON.stringify({ type: "response_item", timestamp: "2026-11-17T09:00:01Z", payload: { type: "message", role: "assistant", content: [] } }),
      JSON.stringify({ type: "event_msg", timestamp: "2026-11-17T09:00:02Z", payload: { type: "agent_message", message: "hi" } }),
      codexLine("r-clean"),
      JSON.stringify({ type: "event_msg", timestamp: "2026-11-17T09:00:03Z", payload: { type: "token_count" } }),
      JSON.stringify({
        type: "token_usage_record", timestamp: "2026-11-17T09:00:04Z",
        payload: {
          session_id: "root-1", thread_id: "thread-1", turn_id: "turn-1", root_turn_id: "turn-1",
          response_id: "r-missing-usage",
        },
      }),
      "{not-json",
    ].join("\n") + "\n");
    await runNativeUsagePass(deps);
    const status = statusOf(deps, "codex");
    // Ordinary records are internal skips; only the real accounting failures surface,
    // and the internal non-usage code never becomes a public warning.
    expect(status?.warnings).toEqual(["malformed", "unsupported"]);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items WHERE harness = 'codex'").get()).toEqual({ n: 1 });
    expect(JSON.stringify(getSourceStatuses(deps.db))).not.toContain("SENTINEL");
    deps.db.close();
  });
});

describe("settings retention and fixed helper diagnostics", () => {
  it("retains last refresh and progress when settings become unreadable", async () => {
    const deps = makeDeps();
    writeClaude(deps, "p/one.jsonl", [claudeLine("m-one")]);
    await runNativeUsagePass(deps);
    await runNativeUsagePass({ ...deps, now: () => NOW + 61 });
    const before = getSourceStatuses(deps.db).find((s) => s.source === "claude")!;
    expect(before.lastRefresh).not.toBeNull();
    deps.settings = () => ({ ok: false, error: "settings-unreadable" });
    await runNativeUsagePass({ ...deps, now: () => NOW + 122 });
    for (const status of getSourceStatuses(deps.db)) {
      expect(status.state).toBe("unavailable");
      expect(status.warnings).toEqual(["settings-error"]);
    }
    const claude = getSourceStatuses(deps.db).find((s) => s.source === "claude")!;
    expect(claude.lastRefresh).toBe(before.lastRefresh);
    expect(claude.progress).toEqual(before.progress);
    expect(deps.db.prepare("SELECT COUNT(*) AS n FROM native_usage_items").get()).toEqual({ n: 1 });
    deps.db.close();
  }, 60_000);

  it("maps helper validation failures to localized fixed warnings and never leaks raw text", async () => {
    const base = mkdtempSync(join(VERIFY_HOME, "helper-map-"));
    const makeCase = (error: string | undefined) => {
      const deps = makeDeps({
        enabled: { claude: false, codex: false, opencode: true },
        exec: async () => ({ stdout: JSON.stringify({ v: 1, ok: false, ...(error ? { error } : {}) }) }),
      });
      writeFileSync(deps.roots.opencodeDb, "");
      return deps;
    };
    const expected: [string | undefined, string][] = [
      ["bad-argv", "malformed-reply"],
      ["bad-path", "malformed-reply"],
      ["bad-state", "malformed-reply"],
      ["helper-failed", "exec-failed"],
      ["SENTINEL_RAW_HELPER_ERROR", "exec-failed"],
      ["schema-fail-closed", "schema-fail-closed"],
      ["runtime-fail", "runtime-fail"],
      ["open-failed", "open-failed"],
      ["query-failed", "query-failed"],
      ["output-overflow", "output-overflow"],
    ];
    for (const [error, warning] of expected) {
      const deps = makeCase(error);
      await runNativeUsagePass(deps);
      const status = getSourceStatuses(deps.db).find((s) => s.source === "opencode")!;
      expect(status.warnings).toEqual([warning]);
      expect(JSON.stringify(status)).not.toContain("SENTINEL_RAW_HELPER_ERROR");
      deps.db.close();
    }
    rmSync(base, { recursive: true, force: true });
  }, 60_000);
});

describe("Codex compressed rollouts", () => {
  const put = (deps: NativeCollectorDeps, day: string, name: string, data: string) => {
    const dir = join(deps.roots.codex, day);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, name), data);
  };

  it("skips .jsonl.zst silently while an unknown file still warns", async () => {
    const warns = async (name: string) => {
      const deps = makeDeps();
      put(deps, "2026/11/17", name, "x");
      await runNativeUsagePass(deps);
      const w = getSourceStatuses(deps.db).find((s) => s.source === "codex")?.warnings ?? [];
      deps.db.close();
      return w;
    };
    expect(await warns("rollout-a.jsonl.zst")).not.toContain("unsupported-layout");
    expect(await warns("other.txt")).toContain("unsupported-layout");
  });
});
