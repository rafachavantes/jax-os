// M5 Task 1: the ONLY own-DB read the Environment diagnostics is allowed. Every read runs in a
// dedicated readonly Node worker using the owning better-sqlite3 library — never getDb(), never
// migrations, never a foreign database, never a blocking SQL call on the request thread.
import { Worker } from "node:worker_threads";
import { join } from "node:path";
import { PANE_RE, paneKey } from "../../lib/workflow";
import type { LastKnownPaneContext } from "./workflows";
import { paneAgentFilter } from "./derive";
import { jaxosHome } from "../env";

export const OWN_DB_TIMEOUT_MS = 200;
// Bounded pane identity: pane id plus server incarnation may never exceed this many characters.
export const PANE_IDENTITY_MAX = 1024;

export type OwnDbSnapshot =
  | {
    ok: true;
    schemaVersion: number | null;
    latestMetricTs: string | null;
    paneContext: LastKnownPaneContext | null;
  }
  | { ok: false; error: "database-unavailable" };

export type OwnDbRequest = { pane?: string | null; incarnation?: string | null };

export type OwnDbWorkerData = {
  path: string;
  paneKey: string | null;
  paneSql: string | null;
};

export type OwnDbWorkerRunner = (data: OwnDbWorkerData) => Promise<OwnDbSnapshot>;

const UNAVAILABLE: OwnDbSnapshot = { ok: false, error: "database-unavailable" };

// Fixed SQL only; the pane key is always a bound parameter. Selected in the main thread so the
// worker never rebuilds policy (`paneAgentFilter`) or accepts caller-supplied SQL.
function paneContextSql(): string {
  return `SELECT project, role, harness_session, ts FROM workflow_events
    WHERE pane_key = ? AND pane IS NOT NULL AND ${paneAgentFilter()}
    ORDER BY id DESC LIMIT 1`;
}

export function ownPaneKey(pane: string | null | undefined, incarnation: string | null | undefined): string | null {
  if (typeof pane !== "string" || !PANE_RE.test(pane)) return null;
  if (typeof incarnation !== "string" || incarnation.length === 0) return null;
  const key = paneKey(pane, incarnation);
  return key.length <= PANE_IDENTITY_MAX ? key : null;
}

export const OWN_DB_WORKER_SOURCE = `
const { parentPort, workerData } = require("node:worker_threads");
let db = null;
try {
  const Database = require("better-sqlite3");
  db = new Database(workerData.path, { readonly: true, fileMustExist: true, timeout: 200 });
  db.prepare("SELECT 1").get();
  const version = db.pragma("user_version", { simple: true });
  const schemaVersion = typeof version === "number" ? version : null;
  let latestMetricTs = null;
  if (schemaVersion === null || schemaVersion >= 3) {
    const row = db.prepare("SELECT ts FROM metrics ORDER BY ts DESC, id DESC LIMIT 1").get();
    latestMetricTs = row && typeof row.ts === "string" ? row.ts : null;
  }
  let paneContext = null;
  if (workerData.paneKey && workerData.paneSql) {
    const row = db.prepare(workerData.paneSql).get(workerData.paneKey);
    if (row && typeof row.project === "string" && typeof row.ts === "string") {
      paneContext = {
        project: row.project,
        role: typeof row.role === "string" ? row.role : null,
        session: typeof row.harness_session === "string" ? row.harness_session : null,
        observedAt: row.ts,
      };
    }
  }
  parentPort.postMessage({ ok: true, schemaVersion, latestMetricTs, paneContext });
} catch {
  parentPort.postMessage({ ok: false, error: "database-unavailable" });
} finally {
  try { if (db) db.close(); } catch {}
}
`;

export function validateOwnDbMessage(value: unknown): OwnDbSnapshot {
  if (!value || typeof value !== "object") return UNAVAILABLE;
  const row = value as Record<string, unknown>;
  if (row.ok === false) return UNAVAILABLE;
  if (row.ok !== true) return UNAVAILABLE;
  const schemaVersion = typeof row.schemaVersion === "number" ? row.schemaVersion : null;
  const latestMetricTs = typeof row.latestMetricTs === "string" ? row.latestMetricTs : null;
  const rawContext = row.paneContext;
  let paneContext: LastKnownPaneContext | null = null;
  if (rawContext && typeof rawContext === "object") {
    const ctx = rawContext as Record<string, unknown>;
    if (typeof ctx.project === "string" && typeof ctx.observedAt === "string") {
      paneContext = {
        project: ctx.project,
        role: typeof ctx.role === "string" ? ctx.role : null,
        session: typeof ctx.session === "string" ? ctx.session : null,
        observedAt: ctx.observedAt,
      };
    }
  }
  return { ok: true, schemaVersion, latestMetricTs, paneContext };
}

export type OwnDbWorkerLike = {
  once(event: "message" | "error" | "exit", listener: (value?: unknown) => void): unknown;
  terminate(): Promise<number> | number;
};

export type OwnDbWorkerFactory = (data: OwnDbWorkerData) => OwnDbWorkerLike;

// The deadline covers worker creation. A validated message is published only after the worker
// exits normally (handle closed) while the original allowance still has time; a timeout
// terminates once but keeps the promise pending until the actual exit, so callers retain
// ownership of the original operation. An error event alone does not release that ownership.
export function createOwnDbWorkerRunner(
  factory: OwnDbWorkerFactory,
  timeoutMs: number = OWN_DB_TIMEOUT_MS,
  now: () => number = () => performance.now(),
): OwnDbWorkerRunner {
  return (data) => new Promise<OwnDbSnapshot>((resolve) => {
    const startedAt = now();
    let settled = false;
    let timedOut = false;
    let failed = false;
    let message: OwnDbSnapshot | null = null;
    let terminateStarted = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const settle = (value: OwnDbSnapshot) => {
      if (settled) return;
      settled = true;
      if (timer) clearTimeout(timer);
      resolve(value);
    };
    const terminate = () => {
      if (terminateStarted || worker === null) return;
      terminateStarted = true;
      try {
        Promise.resolve(worker.terminate()).catch(() => {});
      } catch {
        // ignore: ownership still ends only at the actual exit event
      }
    };
    let worker: OwnDbWorkerLike | null = null;
    try {
      worker = factory(data);
    } catch {
      settle(UNAVAILABLE);
      return;
    }
    // Startup time spent before control returned is already part of the original deadline; an
    // expired startup never receives a fresh full allowance.
    const remaining = timeoutMs - (now() - startedAt);
    if (remaining <= 0) {
      timedOut = true;
      terminate();
    } else {
      timer = setTimeout(() => { timedOut = true; terminate(); }, remaining);
    }
    const current = worker;
    current.once("message", (value) => {
      if (timedOut) return; // discard a late result; the exit settles the operation
      // Keep the value aside: success is published only after the worker actually exits with its
      // handle closed, never before cleanup.
      message = validateOwnDbMessage(value);
    });
    current.once("error", () => {
      failed = true;
      terminate();
    });
    current.once("exit", (code) => {
      // A retained message is published only for a normal code-0 exit that still fits inside
      // the original allowance; a late or abnormal exit is unavailability, not success.
      const normalExit = code === 0 && now() - startedAt < timeoutMs;
      settle(normalExit && !timedOut && !failed && message !== null ? message : UNAVAILABLE);
    });
  });
}

export const systemOwnDbWorkerFactory: OwnDbWorkerFactory = (data) =>
  new Worker(OWN_DB_WORKER_SOURCE, { eval: true, workerData: data });

const systemRunner = createOwnDbWorkerRunner(systemOwnDbWorkerFactory);

export function createOwnDbReader(run: OwnDbWorkerRunner = systemRunner) {
  return async (req: OwnDbRequest = {}): Promise<OwnDbSnapshot> => {
    const key = ownPaneKey(req.pane ?? null, req.incarnation ?? null);
    return run({
      path: join(jaxosHome(), "jaxos.db"),
      paneKey: key,
      paneSql: key === null ? null : paneContextSql(),
    });
  };
}

export const readOwnDb = createOwnDbReader();
