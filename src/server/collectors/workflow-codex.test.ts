import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { createServer as createHttpServer } from "node:http";
import { createServer, type Server, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  CODEX_LOADED_LIMIT, CODEX_READ_CONCURRENCY, codexQueueMessage, codexSocketPath, codexThreadFromRaw, collectCodexSnapshot,
  createCodexConnector, isCodexCliPresent, parseCodexSource, parseCodexStatus,
  type CodexConnector, type CodexRpc,
} from "./workflow-codex";

// Next's bundled ws also ships a server, used only to drive a REAL handshake in the lifecycle test.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { WebSocketServer } = require("next/dist/compiled/ws") as {
  WebSocketServer: new (opts: { server: Server }) => { close(): void };
};

const UUID = "0191f0aa-0000-7000-8000-000000000001";

function thread(over: Record<string, unknown> = {}) {
  return { id: UUID, cwd: "/home/rafa/repos/jax-os", source: "cli", status: { type: "idle" }, canAcceptDirectInput: true, ...over };
}

function fakeConnector(handler: (method: string, params: Record<string, unknown>) => unknown, calls: string[] = []): CodexConnector {
  return async (): Promise<CodexRpc> => ({
    async call(method, params) {
      calls.push(method);
      const r = handler(method, params);
      if (r instanceof Error) throw r;
      return r;
    },
    notify(method, params) {
      calls.push(method);
      handler(method, params);
    },
    close() {
      calls.push("close");
    },
  });
}

describe("parseCodexSource / parseCodexStatus", () => {
  it("normalizes the SessionSource union and refuses unknown shapes", () => {
    expect(parseCodexSource("cli")).toBe("cli");
    expect(parseCodexSource({ subAgent: {} })).toBe("subAgent");
    expect(parseCodexSource({ custom: "x" })).toBe("custom");
    expect(parseCodexSource(42)).toBe("unknown");
    expect(parseCodexSource(null)).toBe("unknown");
  });

  it("maps active flags and treats unknown status as unavailable, never active", () => {
    expect(parseCodexStatus({ type: "active", activeFlags: ["waitingOnUserInput"] })).toEqual({ status: "active", activeFlags: ["waitingOnUserInput"] });
    expect(parseCodexStatus({ type: "idle" })).toEqual({ status: "idle", activeFlags: [] });
    expect(parseCodexStatus({ type: "notLoaded" })).toEqual({ status: "notLoaded", activeFlags: [] });
    expect(parseCodexStatus({ type: "systemError" })).toEqual({ status: "systemError", activeFlags: [] });
    expect(parseCodexStatus({ type: "somethingNew" })).toEqual({ status: "unknown", activeFlags: [] });
    expect(parseCodexStatus(undefined)).toEqual({ status: "unknown", activeFlags: [] });
  });
});

describe("codexThreadFromRaw", () => {
  it("keeps only allowlisted metadata and admits cli/vscode/appServer", () => {
    for (const source of ["cli", "vscode", "appServer"]) {
      const t = codexThreadFromRaw(thread({ source, secretTranscript: "do not keep" }));
      expect(t).toMatchObject({ threadId: UUID, source, status: "idle", canAcceptDirectInput: true });
      expect(JSON.stringify(t)).not.toContain("do not keep");
    }
  });

  it("excludes exec, subAgent, unknown and non-canonical ids", () => {
    expect(codexThreadFromRaw(thread({ source: "exec" }))).toBeNull();
    expect(codexThreadFromRaw(thread({ source: { subAgent: {} } }))).toBeNull();
    expect(codexThreadFromRaw(thread({ source: "unknown" }))).toBeNull();
    expect(codexThreadFromRaw(thread({ id: "not-a-uuid" }))).toBeNull();
    expect(codexThreadFromRaw(null)).toBeNull();
  });

  it("keeps an active thread's flags and a non-active one's empty flags", () => {
    expect(codexThreadFromRaw(thread({ status: { type: "active", activeFlags: ["waitingOnApproval"] } })))
      .toMatchObject({ status: "active", activeFlags: ["waitingOnApproval"] });
    expect(codexThreadFromRaw(thread({ status: { type: "notLoaded" } })))
      .toMatchObject({ status: "notLoaded", activeFlags: [] });
  });
});

describe("collectCodexSnapshot", () => {
  it("runs initialize → initialized → loaded/list → metadata-only read and returns typed threads", async () => {
    const calls: string[] = [];
    const reads: Record<string, unknown>[] = [];
    const connector = fakeConnector((method, params) => {
      if (method === "thread/loaded/list") return { data: [UUID] };
      if (method === "thread/read") {
        reads.push(params);
        return { thread: thread() };
      }
      return {};
    }, calls);
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x/app-server-control.sock", ownerFor: () => null });
    expect(calls).toEqual(["initialize", "initialized", "thread/loaded/list", "thread/read", "close"]);
    expect(reads[0]).toEqual({ threadId: UUID, includeTurns: false }); // metadata only
    expect(snap).toMatchObject({ ok: true, truncated: false, threads: [{ threadId: UUID, status: "idle" }] });
  });

  it("honors pagination and caps loaded discovery at 100 with an explicit truncation warning", async () => {
    const page1 = Array.from({ length: 60 }, (_, i) => `0191f0aa-0000-7000-8000-${String(i).padStart(12, "0")}`);
    const page2 = Array.from({ length: 60 }, (_, i) => `0191f0ab-0000-7000-8000-${String(i).padStart(12, "0")}`);
    let call = 0;
    const connector = fakeConnector((method, params) => {
      if (method === "thread/loaded/list") {
        call += 1;
        if (call === 1) return { data: page1, nextCursor: "c1" };
        return { data: page2, nextCursor: null };
      }
      if (method === "thread/read") return { thread: thread({ id: params.threadId }) };
      return {};
    });
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", ownerFor: () => null });
    expect(snap.ok).toBe(true);
    if (snap.ok) {
      expect(snap.threads).toHaveLength(CODEX_LOADED_LIMIT);
      expect(snap.truncated).toBe(true);
    }
  });

  it("reads metadata with at most four concurrent requests", async () => {
    const ids = Array.from({ length: 10 }, (_, i) => `0191f0aa-0000-7000-8000-${String(i).padStart(12, "0")}`);
    let inFlight = 0;
    let peak = 0;
    const connector: CodexConnector = async () => ({
      async call(method, params) {
        if (method === "thread/loaded/list") return { data: ids };
        if (method === "thread/read") {
          inFlight += 1;
          peak = Math.max(peak, inFlight);
          await new Promise((r) => setTimeout(r, 1));
          inFlight -= 1;
          return { thread: thread({ id: params.threadId }) };
        }
        return {};
      },
      notify() {},
      close() {},
    });
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", ownerFor: () => null });
    expect(snap.ok).toBe(true);
    expect(peak).toBeLessThanOrEqual(CODEX_READ_CONCURRENCY);
  });

  it("a failed connection is a typed source failure and always closes the transport", async () => {
    const close = vi.fn();
    const connector: CodexConnector = async () => ({ call: async () => { throw new Error("boom"); }, notify() {}, close });
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", ownerFor: () => null });
    expect(snap).toMatchObject({ ok: false, error: "codex runtime unavailable" });
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("a connector rejection is a source failure, never an empty success", async () => {
    const connector: CodexConnector = async () => { throw new Error("no socket"); };
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", ownerFor: () => null });
    expect(snap.ok).toBe(false);
  });
});

describe("collectCodexSnapshot — partial discovery tolerance (C1)", () => {
  it("drops only the failed thread/read and flags the snapshot incomplete", async () => {
    const good = "0191f0aa-0000-7000-8000-000000000001";
    const bad = "0191f0aa-0000-7000-8000-000000000002";
    const connector = fakeConnector((method, params) => {
      if (method === "thread/loaded/list") return { data: [good, bad] };
      if (method === "thread/read") {
        if (params.threadId === bad) return new Error("evicted");
        return { thread: thread({ id: params.threadId }) };
      }
      return {};
    });
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", ownerFor: () => null });
    expect(snap.ok).toBe(true);
    if (snap.ok) {
      expect(snap.threads.map((t) => t.threadId)).toEqual([good]);
      expect(snap.truncated).toBe(true);
    }
  });

  it("a malformed list payload is a source failure, never a successful empty discovery", async () => {
    const connector = fakeConnector((method) => (method === "thread/loaded/list" ? { nope: true } : {}));
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", ownerFor: () => null });
    expect(snap).toMatchObject({ ok: false, error: "codex runtime unavailable" });
  });

  it("reaching the 16-page cap with another cursor is an incomplete snapshot", async () => {
    let page = 0;
    const connector = fakeConnector((method, params) => {
      if (method === "thread/loaded/list") {
        page += 1;
        // One new id per page, always offering another cursor — never hits the 100-id cap.
        return { data: [`0191f0aa-0000-7000-8000-${String(page).padStart(12, "0")}`], nextCursor: "more" };
      }
      if (method === "thread/read") return { thread: thread({ id: params.threadId }) };
      return {};
    });
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", ownerFor: () => null });
    expect(snap.ok).toBe(true);
    if (snap.ok) {
      expect(snap.threads).toHaveLength(16);
      expect(snap.truncated).toBe(true);
    }
  });

  it("resolves each distinct cwd's owner exactly once via the injected resolver", async () => {
    const ids = ["0191f0aa-0000-7000-8000-000000000001", "0191f0aa-0000-7000-8000-000000000002"];
    const connector = fakeConnector((method, params) => {
      if (method === "thread/loaded/list") return { data: ids };
      if (method === "thread/read") return { thread: thread({ id: params.threadId, cwd: "/repo/one" }) };
      return {};
    });
    const ownerFor = vi.fn((cwd: string) => (cwd === "/repo/one" ? "one" : null));
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", ownerFor });
    expect(snap.ok).toBe(true);
    if (snap.ok) expect(snap.threads.every((t) => t.owner === "one")).toBe(true);
    expect(ownerFor).toHaveBeenCalledTimes(1); // deduped by cwd — one probe per directory
  });

  it("a failed or never-settling owner resolver yields owner:null within the shared deadline (F1/F2)", async () => {
    const connector: CodexConnector = async () => ({
      async call(method) {
        if (method === "thread/loaded/list") {
          await new Promise((r) => setTimeout(r, 50)); // RPC consumes part of the budget
          return { data: [UUID] };
        }
        if (method === "thread/read") return { thread: thread({ cwd: "/repo/one" }) };
        return {};
      },
      notify() {},
      close() {},
    });
    const ownerFor = vi.fn(() => new Promise<string | null>(() => {})); // never settles
    const start = Date.now();
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", timeoutMs: 150, ownerFor });
    const elapsed = Date.now() - start;
    expect(snap.ok).toBe(true);
    if (snap.ok) {
      expect(snap.threads).toHaveLength(1);
      expect(snap.threads[0].owner).toBeNull(); // no basename guess, no lost metadata
      expect(snap.threads[0].status).toBe("idle");
    }
    // ONE shared 150ms budget, never a second full timeout: a never-settling resolver cannot hold
    // the collection open past the deadline (generous allowance for CI scheduling).
    expect(elapsed).toBeLessThan(1000);
  });

  it("resolves owners that settle within the shared deadline and closes the transport (F2)", async () => {
    const close = vi.fn();
    const connector: CodexConnector = async () => ({
      async call(method) {
        if (method === "thread/loaded/list") return { data: [UUID] };
        if (method === "thread/read") return { thread: thread({ cwd: "/repo/one" }) };
        return {};
      },
      notify() {},
      close,
    });
    const ownerFor = vi.fn(async () => { await new Promise((r) => setTimeout(r, 30)); return "one"; });
    const snap = await collectCodexSnapshot(connector, { socketPath: "/x", timeoutMs: 200, ownerFor });
    expect(snap.ok).toBe(true);
    if (snap.ok) expect(snap.threads[0].owner).toBe("one");
    expect(close).toHaveBeenCalledTimes(1); // resource cleanup on the normal path
  });
});

describe("createCodexConnector — connection lifecycle bounds (C1)", () => {
  function setup() {
    const dir = mkdtempSync(join(tmpdir(), "jax-codex-"));
    return { dir, sock: join(dir, "s.sock") };
  }
  function cleanup(dir: string, sockets: Set<Socket>, servers: { close(): void }[]) {
    for (const s of sockets) s.destroy();
    for (const s of servers) s.close();
    rmSync(dir, { recursive: true, force: true });
  }

  it("rejects promptly when the peer accepts but never completes the upgrade", async () => {
    const { dir, sock } = setup();
    const sockets = new Set<Socket>();
    const server = createServer((s) => { sockets.add(s); s.on("close", () => sockets.delete(s)); });
    try {
      await new Promise<void>((r) => server.listen(sock, r));
      const start = Date.now();
      await expect(createCodexConnector(sock, 120)).rejects.toThrow(/timeout/);
      expect(Date.now() - start).toBeLessThan(3000);
    } finally {
      cleanup(dir, sockets, [server]);
    }
  });

  it("rejects when the peer closes before the handshake opens", async () => {
    const { dir, sock } = setup();
    const sockets = new Set<Socket>();
    const server = createServer((s) => { sockets.add(s); s.on("close", () => sockets.delete(s)); s.destroy(); });
    try {
      await new Promise<void>((r) => server.listen(sock, r));
      await expect(createCodexConnector(sock, 5000)).rejects.toThrow();
    } finally {
      cleanup(dir, sockets, [server]);
    }
  });

  it("rejects a pending RPC when the shared deadline fires after open", async () => {
    const { dir, sock } = setup();
    const sockets = new Set<Socket>();
    // A real HTTP server is required: only it emits 'upgrade', which ws's WebSocketServer hooks.
    const httpServer = createHttpServer((_req, res) => res.end());
    httpServer.on("connection", (s) => { sockets.add(s); s.on("close", () => sockets.delete(s)); });
    const wss = new WebSocketServer({ server: httpServer });
    try {
      await new Promise<void>((r) => httpServer.listen(sock, r));
      const rpc = await createCodexConnector(sock, 500);
      await expect(rpc.call("initialize", {})).rejects.toThrow(/timeout/);
      rpc.close();
    } finally {
      cleanup(dir, sockets, [httpServer, wss]);
    }
  });
});

describe("codexSocketPath", () => {
  it("derives the fixed suffix from CODEX_HOME, defaulting to ~/.codex", () => {
    expect(codexSocketPath({ CODEX_HOME: "/tmp/ch" } as unknown as NodeJS.ProcessEnv)).toBe("/tmp/ch/app-server-control/app-server-control.sock");
    expect(codexSocketPath({} as unknown as NodeJS.ProcessEnv)).toMatch(/\/\.codex\/app-server-control\/app-server-control\.sock$/);
  });
});

describe("isCodexCliPresent (spec MOA-502 Decision 3)", () => {
  const dirs: string[] = [];
  function tmpBin(files: Record<string, { mode: number; dir?: boolean }>): string {
    const dir = mkdtempSync(join(tmpdir(), "jax-codex-path-"));
    dirs.push(dir);
    mkdirSync(join(dir, "bin"));
    for (const [name, spec] of Object.entries(files)) {
      const p = join(dir, "bin", name);
      if (spec.dir) mkdirSync(p);
      else writeFileSync(p, "#!/bin/sh\n", { mode: spec.mode });
    }
    return join(dir, "bin");
  }
  afterEach(() => {
    for (const d of dirs) rmSync(d, { recursive: true, force: true });
    dirs.length = 0;
  });

  // This project's ProcessEnv requires NODE_ENV; the signature only reads PATH, so build
  // the injected env the same way the codexSocketPath test above casts its own fixture.
  const env = (PATH?: string) => ({ PATH } as unknown as NodeJS.ProcessEnv);

  it("is true when a file named codex exists under a PATH directory", () => {
    expect(isCodexCliPresent(env(tmpBin({ codex: { mode: 0o755 } })))).toBe(true);
  });

  it("is false when no PATH directory has a codex file", () => {
    expect(isCodexCliPresent(env(tmpBin({ other: { mode: 0o755 } })))).toBe(false);
  });

  it("is false when PATH is unset", () => {
    expect(isCodexCliPresent(env())).toBe(false);
  });

  it("is false when a directory (not a file) named codex sits on PATH (F2 — a directory must not masquerade as the CLI)", () => {
    expect(isCodexCliPresent(env(tmpBin({ codex: { mode: 0o755, dir: true } })))).toBe(false);
  });

  it("is false when the codex file exists but is not executable (mode 0o644)", () => {
    expect(isCodexCliPresent(env(tmpBin({ codex: { mode: 0o644 } })))).toBe(false);
  });
});

describe("codexQueueMessage (native-answer-delivery spec D3)", () => {
  const THREAD = "0191f0aa-ffff-7000-8000-000000000009";

  it("calls the exact command jaxflow.py's own callback uses for a codex caller", async () => {
    const calls: string[][] = [];
    const run = async (file: string, args: string[]) => { calls.push([file, ...args]); return ""; };
    await codexQueueMessage(THREAD, "[Jax OS · Rafa] pode", run);
    expect(calls).toEqual([["codex", "queue", "--thread", THREAD, "--message", "[Jax OS · Rafa] pode"]]);
  });

  it("classifies a missing CLI binary as cli-missing", async () => {
    const run = async () => { throw Object.assign(new Error("spawn codex ENOENT"), { code: "ENOENT" }); };
    await expect(codexQueueMessage(THREAD, "pode", run)).rejects.toThrow("codex queue cli missing");
  });

  it("classifies a killed/timed-out call as timeout", async () => {
    const run = async () => { throw Object.assign(new Error("timeout"), { killed: true, signal: "SIGTERM" }); };
    await expect(codexQueueMessage(THREAD, "pode", run)).rejects.toThrow("codex queue timeout");
  });

  it("classifies a plain non-zero-exit failure as queue-failed", async () => {
    const run = async () => { throw new Error("Command failed: codex queue --thread x --message y"); };
    await expect(codexQueueMessage(THREAD, "pode", run)).rejects.toThrow("codex queue failed");
  });
});
