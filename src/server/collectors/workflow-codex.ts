// Read-only metadata collector for the local Codex App Server (MOA-469 §4).
//
// The collector owns system access; the DB derivation consumes an injected typed snapshot.
// Query metadata only — no transcript content, no resuming, no subscribing, no starting or
// steering turns. The transport is a bounded local RPC over the same-user Unix socket, using
// Next's bundled ws client (no new dependency, no handwritten WebSocket framing). Every
// caller gets a typed CodexSnapshot; a failed connection is a source failure, never an empty
// successful result and never an erasure of tmux data.
import { execFile } from "node:child_process";
import { homedir } from "node:os";
import { join } from "node:path";
import { promisify } from "node:util";
import { UUID_RE } from "../../lib/workflow";
import { isCliPresent } from "./cli-presence";
import { ownerProjectName, type GitProbe } from "./projects";

const execFileAsync = promisify(execFile);

export const CODEX_SOCKET_DIR = "app-server-control";
export const CODEX_SOCKET_FILE = "app-server-control.sock";
export const CODEX_TIMEOUT_MS = 5000;
export const CODEX_MAX_MESSAGE_BYTES = 1024 * 1024;
export const CODEX_LOADED_LIMIT = 100;
export const CODEX_READ_CONCURRENCY = 4;
const ADMITTED_SOURCES = new Set(["cli", "vscode", "appServer"]);

// PATH-presence check (MOA-502 D3); the probe itself now lives in cli-presence.ts (MOA-504 D4).
export function isCodexCliPresent(env: NodeJS.ProcessEnv = process.env): boolean {
  return isCliPresent("codex", env);
}

export type CodexThread = {
  threadId: string;
  cwd: string;
  // Canonical control project (basename of the resolved Git owner) or null when unresolved. Filled
  // by collectCodexSnapshot via the injected owner resolver, never a DB-side basename guess (C3).
  owner: string | null;
  source: string;
  // "unknown" means the runtime returned an unparseable status — unavailable, never active.
  status: string;
  activeFlags: string[];
  canAcceptDirectInput: boolean;
};

export type CodexSnapshot =
  | { ok: true; ts: string; threads: CodexThread[]; truncated: boolean }
  | { ok: false; ts: string; error: string };

export type CodexRpc = {
  call(method: string, params: Record<string, unknown>): Promise<unknown>;
  // JSON-RPC notification (no response expected) — the protocol's `initialized` handshake.
  notify(method: string, params: Record<string, unknown>): void;
  close(): void;
};
export type CodexConnector = (socketPath: string, deadlineMs: number) => Promise<CodexRpc>;

// Same-user CODEX_HOME (default ~/.codex), fixed suffix. Never accept a socket path from HTTP.
export function codexSocketPath(env: NodeJS.ProcessEnv = process.env): string {
  const home = (env.CODEX_HOME || join(homedir(), ".codex")) as string;
  return join(home, CODEX_SOCKET_DIR, CODEX_SOCKET_FILE);
}

// ---- Pure parsing helpers (injected raw RPC payloads in tests) ------------------------------

export function parseCodexSource(raw: unknown): string {
  if (typeof raw === "string") return raw;
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    if ("subAgent" in (raw as Record<string, unknown>)) return "subAgent";
    if ("custom" in (raw as Record<string, unknown>)) return "custom";
  }
  return "unknown";
}

export function parseCodexStatus(raw: unknown): { status: string; activeFlags: string[] } {
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    const o = raw as Record<string, unknown>;
    if (typeof o.type === "string") {
      if (o.type === "active") {
        const flags = Array.isArray(o.activeFlags)
          ? (o.activeFlags as unknown[]).filter((f): f is string => typeof f === "string")
          : [];
        return { status: "active", activeFlags: flags };
      }
      if (o.type === "idle" || o.type === "notLoaded" || o.type === "systemError") {
        return { status: o.type, activeFlags: [] };
      }
    }
  }
  return { status: "unknown", activeFlags: [] };
}

// Parse ONE Thread object from thread/read's `{thread}` (or a bare thread). Exclude exec and
// subAgent sources; admit cli/vscode/appServer roots only. A malformed/inadmissible thread is
// dropped — it is not evidence of a human lead.
export function codexThreadFromRaw(raw: unknown): CodexThread | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  const threadId = typeof o.id === "string" && UUID_RE.test(o.id) ? o.id : null;
  if (!threadId) return null;
  const source = parseCodexSource(o.source);
  if (!ADMITTED_SOURCES.has(source)) return null;
  const { status, activeFlags } = parseCodexStatus(o.status);
  return {
    threadId,
    cwd: typeof o.cwd === "string" ? o.cwd : "",
    owner: null, // resolved at the collector boundary, never by the pure parser
    source,
    status,
    activeFlags,
    canAcceptDirectInput: o.canAcceptDirectInput === true,
  };
}

// ---- Bounded local RPC over the bundled ws client -------------------------------------------

// Narrow local type: Next publishes no declaration for its bundled ws client.
type CodexWs = {
  on(event: "open" | "error" | "close", cb: (arg?: unknown) => void): void;
  on(event: "message", cb: (raw: Buffer, isBinary: boolean) => void): void;
  send(data: string): void;
  terminate(): void;
};
type CodexWsCtor = { new (url: string, opts: { perMessageDeflate: boolean; maxPayload: number }): CodexWs };
// eslint-disable-next-line @typescript-eslint/no-require-imports
const WebSocket = require("next/dist/compiled/ws") as CodexWsCtor;

export function createCodexConnector(socketPath: string, deadlineMs: number): Promise<CodexRpc> {
  return new Promise<CodexRpc>((resolve, reject) => {
    const ws = new WebSocket(`ws+unix://${socketPath}:/`, {
      perMessageDeflate: false,
      maxPayload: CODEX_MAX_MESSAGE_BYTES,
    });
    const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
    let nextId = 0;
    let deadlineTimer: NodeJS.Timeout | null = null;
    let settled = false;
    let closed = false;

    const teardown = (message: string) => {
      if (closed) return;
      closed = true;
      if (deadlineTimer) clearTimeout(deadlineTimer);
      try {
        ws.terminate();
      } catch {
        // socket already gone — the pending rejections below still matter
      }
      for (const { reject: r } of pending.values()) r(new Error(message));
      pending.clear();
    };

    // Settles the (possibly still unopened) connector promise AND rejects any pending RPC, then
    // tears down. After open the promise is already resolved, so only the pending rejections and
    // socket teardown remain. Idempotent via `closed`/`settled`.
    const fail = (err: unknown) => {
      const message = err instanceof Error ? err.message : "codex runtime unavailable";
      const wasSettled = settled;
      settled = true;
      teardown(message);
      if (!wasSettled) reject(err instanceof Error ? err : new Error("codex runtime unavailable"));
    };

    // One shared deadline for handshake + init + all RPC work, armed at CONNECTION CREATION (C1):
    // a peer that accepts the socket but never completes the HTTP upgrade fires no open/error/close
    // and must still be bounded. Never re-armed after open.
    deadlineTimer = setTimeout(() => fail(new Error("codex runtime timeout")), deadlineMs);

    ws.on("error", (err) => fail(err));
    ws.on("close", () => fail(new Error("codex runtime unavailable")));
    ws.on("message", (raw: Buffer) => {
      let msg: unknown;
      try {
        msg = JSON.parse(raw.toString("utf8"));
      } catch {
        return; // malformed frame — ignore, never a completeness claim
      }
      if (!msg || typeof msg !== "object") return;
      const o = msg as { id?: unknown; result?: unknown; error?: unknown };
      if (typeof o.id !== "number") return; // unsolicited notification — ignored (§4)
      const p = pending.get(o.id);
      if (!p) return;
      pending.delete(o.id);
      if (o.error !== undefined) {
        p.reject(new Error("codex rpc failed"));
      } else {
        p.resolve(o.result);
      }
    });
    ws.on("open", () => {
      if (settled) return;
      settled = true;
      const rpc: CodexRpc = {
        call(method, params) {
          return new Promise<unknown>((res, rej) => {
            if (closed) {
              rej(new Error("codex runtime unavailable"));
              return;
            }
            const callId = ++nextId;
            pending.set(callId, { resolve: res, reject: rej });
            try {
              ws.send(JSON.stringify({ method, id: callId, params }));
            } catch {
              pending.delete(callId);
              rej(new Error("codex runtime unavailable"));
            }
          });
        },
        notify(method, params) {
          if (closed) return;
          try {
            ws.send(JSON.stringify({ method, params }));
          } catch {
            // notification delivery is best-effort — the deadline teardown surfaces failure
          }
        },
        close() {
          teardown("codex socket closed");
        },
      };
      resolve(rpc);
    });
  });
}

// ---- Collector ------------------------------------------------------------------------------

export type OwnerResolver = (cwd: string, signal?: AbortSignal) => Promise<string | null> | string | null;

// One probe per distinct cwd (deduplicated by the collector), each bounded by the shared collection
// deadline via `signal`. A probe after expiry returns null without spawning; an in-flight probe is
// aborted at expiry. The per-probe timeout is a backstop for direct (unbounded) callers only.
const gitProbe: GitProbe = async (args, cwd, signal) => {
  if (signal?.aborted) return null; // never launch a git probe after the collection deadline (F2)
  try {
    const { stdout } = await execFileAsync("git", args, { cwd, timeout: CODEX_TIMEOUT_MS, signal, maxBuffer: 1024 * 1024, encoding: "utf8" });
    return { ok: true, stdout };
  } catch {
    return null;
  }
};

// Real resolver: canonical Git owner at the collector boundary (C3), mirroring MOA-467.
export const resolveProjectOwner: OwnerResolver = (cwd, signal) => ownerProjectName(cwd, gitProbe, signal);

// Resolve every distinct thread cwd against `deadlineAt`, the SAME absolute deadline armed for the
// RPC connector (F2): ownership never gets a fresh budget after the RPC work. On expiry, production
// probes are aborted, remaining owners stay null, and the readable snapshot is still returned.
// A resolver that settles after the timeout cannot mutate the returned threads because owners are
// applied only from the settled map.
async function resolveOwners(threads: CodexThread[], ownerFor: OwnerResolver, deadlineAt: number): Promise<void> {
  const remaining = deadlineAt - Date.now();
  if (remaining <= 0) return; // budget already spent — no second probe after the deadline
  const cwds = [...new Set(threads.map((t) => t.cwd).filter(Boolean))];
  const abort = new AbortController();
  let timer: NodeJS.Timeout | null = null;
  const owners = await new Promise<Map<string, string | null> | null>((resolve) => {
    timer = setTimeout(() => {
      abort.abort(); // cancel production subprocesses still in flight
      resolve(null);
    }, remaining);
    Promise.all(cwds.map(async (cwd): Promise<[string, string | null]> => {
      // A failed/timed-out probe degrades to an unresolved owner, never a whole-snapshot failure.
      const owner = await Promise.resolve().then(() => ownerFor(cwd, abort.signal)).catch(() => null);
      return [cwd, owner];
    })).then(
      (entries) => {
        if (timer) clearTimeout(timer);
        resolve(new Map(entries));
      },
      () => {
        if (timer) clearTimeout(timer);
        resolve(null);
      },
    );
  });
  if (timer) clearTimeout(timer);
  if (owners) {
    for (const t of threads) t.owner = t.cwd ? (owners.get(t.cwd) ?? null) : null;
  }
}

export async function collectCodexSnapshot(
  connector: CodexConnector,
  opts: { socketPath?: string; timeoutMs?: number; ownerFor?: OwnerResolver } = {},
): Promise<CodexSnapshot> {
  const socketPath = opts.socketPath ?? codexSocketPath();
  const timeoutMs = opts.timeoutMs ?? CODEX_TIMEOUT_MS;
  const ownerFor = opts.ownerFor ?? resolveProjectOwner;
  const ts = new Date().toISOString();
  // One absolute deadline for the WHOLE collection — connection, init, all RPCs AND ownership
  // discovery (F2). Never re-armed, never given a fresh budget after the RPC work.
  const deadlineAt = Date.now() + timeoutMs;
  let rpc: CodexRpc | null = null;
  try {
    rpc = await connector(socketPath, timeoutMs);
    await rpc.call("initialize", {
      clientInfo: { name: "jax-os", version: "1" },
      capabilities: { experimentalApi: true },
    });
    rpc.notify("initialized", {}); // JSON-RPC notification — no response is ever sent back
    const ids: string[] = [];
    let cursor: string | null = null;
    let truncated = false;
    for (let page = 0; page < 16; page += 1) {
      const res = await rpc.call("thread/loaded/list", { limit: CODEX_LOADED_LIMIT, cursor });
      // A malformed list payload is a SOURCE FAILURE, never a successful empty discovery (C1).
      if (!res || typeof res !== "object" || Array.isArray(res) || !Array.isArray((res as { data?: unknown }).data)) {
        throw new Error("malformed thread/loaded/list");
      }
      const data = (res as { data: unknown[] }).data.filter((x): x is string => typeof x === "string");
      ids.push(...data);
      // Cap discovery at CODEX_LOADED_LIMIT root candidates; hitting it is an explicit source
      // warning (conservative: even an exactly-full final page may have more), never a silent
      // completeness claim.
      if (ids.length >= CODEX_LOADED_LIMIT) {
        truncated = true;
        break;
      }
      const next = (res as { nextCursor?: unknown }).nextCursor;
      if (typeof next !== "string" || !next) break;
      // The 16-page cap with another cursor present is an incomplete snapshot, not completeness.
      if (page === 15) {
        truncated = true;
        break;
      }
      cursor = next;
    }
    const unique = [...new Set(ids)].slice(0, CODEX_LOADED_LIMIT);
    const threads: CodexThread[] = [];
    const queue = [...unique];
    const workers = Array.from(
      { length: Math.min(CODEX_READ_CONCURRENCY, queue.length) },
      async () => {
        while (queue.length > 0) {
          const threadId = queue.shift() as string;
          let res: unknown;
          try {
            res = await rpc!.call("thread/read", { threadId, includeTurns: false });
          } catch {
            // One evicted/unreadable thread must not blank every other project's state: it is an
            // incomplete snapshot (the existing truncation warning) and the queue continues.
            truncated = true;
            continue;
          }
          const threadRaw = res && typeof res === "object" && "thread" in (res as Record<string, unknown>)
            ? (res as { thread: unknown }).thread
            : res;
          const t = codexThreadFromRaw(threadRaw);
          if (t) threads.push(t);
        }
      },
    );
    await Promise.all(workers);
    await resolveOwners(threads, ownerFor, deadlineAt);
    return { ok: true, ts, threads, truncated };
  } catch {
    return { ok: false, ts, error: "codex runtime unavailable" };
  } finally {
    rpc?.close();
  }
}

// ---- Native answer delivery (native-answer-delivery spec D3) --------------------------------

// The exact mechanism jaxflow's own completion callback already uses for a codex caller
// (scripts/jaxflow.py:2650-2660: subprocess.run(["codex","queue","--thread",session,"--message",
// line], timeout=10)). This is a plain CLI call, not the read-only RPC connector above
// (collectCodexSnapshot) — that boundary is about the RPC session, not this separate `codex
// queue` CLI surface (spec §2's D3, "no starting or steering turns" refers to the RPC only).
type QueueExec = (file: string, args: string[]) => Promise<string>;
const CODEX_QUEUE_TIMEOUT_MS = 10_000;
const systemQueueExec: QueueExec = async (file, args) => {
  const { stdout } = await execFileAsync(file, args, { timeout: CODEX_QUEUE_TIMEOUT_MS, encoding: "utf8" });
  return stdout;
};

export async function codexQueueMessage(threadId: string, text: string, run: QueueExec = systemQueueExec): Promise<void> {
  try {
    await run("codex", ["queue", "--thread", threadId, "--message", text]);
  } catch (e) {
    const err = e as NodeJS.ErrnoException & { killed?: boolean; signal?: NodeJS.Signals | null };
    if (err.code === "ENOENT") throw new Error("codex queue cli missing");
    if (err.killed || err.signal) throw new Error("codex queue timeout");
    throw new Error("codex queue failed");
  }
}
