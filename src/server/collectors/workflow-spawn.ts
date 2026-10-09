// Phase 2 (Mission Control B, spec §7 / Decisions 11-12-22): ONE spawn helper for the three
// action routes. Mirrors the execFile convention workflow-codex.ts's gitProbe and github.ts's
// systemExec already use (argv array, never a shell string), but NEVER throws — a refusal
// (exit 2), a timeout, a spawn error and success all resolve to the same ChildResult, redacted
// and capped once, here, so every mutations row and JSON response sees the same text.
import { execFile } from "node:child_process";
import { join } from "node:path";
import { LIMITS } from "../../lib/workflow";
import { redactSecrets } from "./redact";

// The same script `~/.local/bin/jaxflow` execs (`exec python3 /home/rafa/repos/jax-os/scripts/jaxflow.py "$@"`),
// addressed absolutely so `cwd` can be the PROJECT repo. jaxos.service runs from the jax-os checkout.
export const JAXFLOW_PROGRAM = "python3";
export const JAXFLOW_SCRIPT = join(process.cwd(), "scripts", "jaxflow.py");
export const CANCEL_TIMEOUT_MS = 25_000; // spec Decision 5: above cmd_cancel's own 15s wait
// dispatch/merge hold the request while jaxflow reserves a worktree, re-runs verify (diff review)
// or runs --checks (merge) — minutes, not seconds. Single-user dashboard, no proxy timeout (spec §13.5).
export const ACTION_TIMEOUT_MS = 15 * 60_000;

export type ChildResult = {
  ok: boolean;              // exit 0
  code: string | null;      // exit 2 only: the first non-empty stderr line (jaxflow's refusal code)
  exitCode: number | null;  // null: never started (spawn error) or killed (timeout)
  stdoutTail: string;       // redacted, last LIMITS.childTail chars
  stderrTail: string;       // redacted, last LIMITS.childTail chars
  durationMs: number;
};
export type SpawnOpts = { cwd?: string; env?: Record<string, string>; timeoutMs: number };
export type ChildRunner = (file: string, args: string[], opts: SpawnOpts) => Promise<ChildResult>;

function tail(text: string): string {
  const s = redactSecrets(text);
  return s.length > LIMITS.childTail ? s.slice(-LIMITS.childTail) : s;
}

export function lastLine(text: string): string {
  const lines = text.split(/\r?\n/).map((l) => l.trim()).filter((l) => l.length > 0);
  return lines.length > 0 ? lines[lines.length - 1] : "";
}

// The audit fields every route stores on the mutations row, on every outcome (Decision 12).
export function childDetails(r: ChildResult): Record<string, unknown> {
  return { exitCode: r.exitCode, stdoutTail: r.stdoutTail, stderrTail: r.stderrTail, durationMs: r.durationMs };
}

export const runChild: ChildRunner = (file, args, opts) =>
  new Promise((resolve) => {
    const started = Date.now();
    const env = opts.env ? { ...process.env, ...opts.env } : process.env;
    // Round-2 F1: Node validates argv synchronously (e.g. a NUL byte throws ERR_INVALID_ARG_VALUE
    // before the child ever starts). A Promise executor's synchronous throw becomes a REJECTED
    // promise, not this function's usual resolved failure shape — callers (`await deps.run(...)`)
    // outside a try/catch would then see an uncaught rejection surface as a 500. Catch it here so
    // every spawn failure, sync or async, resolves the same way.
    try {
      execFile(
        file, args,
        { cwd: opts.cwd, env, timeout: opts.timeoutMs, maxBuffer: 4 * 1024 * 1024, encoding: "utf8" },
        (err, stdout, stderr) => {
          const durationMs = Date.now() - started;
          const stdoutTail = tail(stdout ?? "");
          const stderrTail = tail(stderr ?? "");
          if (!err) {
            resolve({ ok: true, code: null, exitCode: 0, stdoutTail, stderrTail, durationMs });
            return;
          }
          // execFile's error carries the numeric exit code in `code` when the child exited on its
          // own; a string errno (ENOENT) on a spawn failure; and `killed: true` with a null code on
          // a timeout. Only exit 2 is a jaxflow refusal.
          const e = err as Error & { code?: number | string };
          const exitCode = typeof e.code === "number" ? e.code : null;
          const code = exitCode === 2 ? (redactSecrets(stderr ?? "").split(/\r?\n/).find((l) => l.trim().length > 0)?.trim() ?? null) : null;
          resolve({ ok: false, code, exitCode, stdoutTail, stderrTail, durationMs });
        },
      );
    } catch (e) {
      const durationMs = Date.now() - started;
      const message = e instanceof Error ? e.message : String(e);
      resolve({ ok: false, code: null, exitCode: null, stdoutTail: "", stderrTail: tail(message), durationMs });
    }
  });

// `args` is the jaxflow argv WITHOUT the program — exactly what the mutations row records.
export function jaxflow(args: string[], opts: SpawnOpts, run: ChildRunner = runChild): Promise<ChildResult> {
  return run(JAXFLOW_PROGRAM, [JAXFLOW_SCRIPT, ...args], opts);
}
