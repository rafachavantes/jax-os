import { getDb } from "./db";
import { finishMutation, insertMutation } from "./db/mutations";
import { MutationRejected, type MutationFailure } from "../lib/mutationOutcome";

// tmux take/release: the insert IS the effect. Other sanctioned mutations use runMutation.
export function logMutation(entry: Record<string, unknown>) {
  insertMutation(getDb(), entry);
}

export function fileEffect<T>(effect: () => T): T {
  try { return effect(); }
  catch (e) {
    const code = (e as NodeJS.ErrnoException | null)?.code;
    const message = e instanceof Error ? e.message : "";
    if (code === "EEXIST" || message === "destination exists") throw new MutationRejected("already exists");
    if (code === "ENOTEMPTY") throw new MutationRejected("directory not empty");
    if (message === "changed on disk") throw new MutationRejected("changed on disk");
    if (message === "outside allowlist" || message === "bad name" || message === "symlink source")
      throw new MutationRejected(message);
    if (["ENOENT", "ENOTDIR", "EACCES", "EPERM", "EISDIR"].includes(code ?? ""))
      throw new MutationRejected("file operation refused");
    throw e;
  }
}

export async function runMutation<T>(
  entry: Record<string, unknown>,
  effect: () => T | Promise<T>,
  details?: (value: T) => Record<string, unknown>,
  // Phase 2 (spec §7, round-2 F5): merged into the row when the effect throws — a refusal or an
  // unconfirmed outcome still records the child's exit code and redacted tails.
  failureDetails?: () => Record<string, unknown>,
): Promise<{ ok: true; value: T } | (MutationFailure & { status?: number; value?: T })> {
  let db: ReturnType<typeof getDb>;
  let id: number;
  try {
    db = getDb();
    id = insertMutation(db, { ...entry, ok: undefined, error: undefined, outcome: "pending" });
  } catch {
    return { ok: false, code: "audit-unavailable", effect: "not-applied", audit: "unavailable",
      error: "audit unavailable; action not performed" };
  }
  let value: T;
  try { value = await effect(); }
  catch (e) {
    const rejected = e instanceof MutationRejected;
    const error = rejected ? e.message : "action outcome unconfirmed; check current state before retrying";
    const certainty = rejected ? "not-applied" : "unconfirmed";
    try { finishMutation(db, id, false, error, rejected ? "failed" : "abandoned", failureDetails?.()); }
    catch {
      return { ok: false, code: "audit-finalization-failed", effect: certainty, audit: "pending",
        error: rejected ? "action rejected; audit recording failed" : "action outcome unconfirmed; audit recording failed",
        status: rejected ? e.status : undefined };
    }
    return { ok: false, code: rejected ? "mutation-rejected" : "mutation-unconfirmed",
      effect: certainty, audit: "recorded", error, status: rejected ? e.status : undefined };
  }
  try { finishMutation(db, id, true, null, "done", details?.(value)); }
  catch {
    return { ok: false, code: "audit-finalization-failed", effect: "applied", audit: "pending",
      error: "action applied but audit recording failed; check current state before retrying", value };
  }
  return { ok: true, value };
}
