// Native answer delivery for Claude (spec D3/D4): a file-drop mailbox at
// ~/.jax-os/callbacks/<harness_session>/answer-<eventId>.{armed,answer} that scripts/
// jaxflow_hook.py's Stop-hook watcher polls. CALLBACKS_ROOT/WATCH_DEADLINE_S are a SEPARATE copy
// of jaxflow_hook.py's own values (no shared import across languages) — kept in lockstep by hand.
import { mkdirSync, renameSync, statSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { UUID_RE } from "../../lib/workflow";
import { jaxosHome } from "../env";

export const CALLBACKS_ROOT = join(jaxosHome(), "callbacks");
// Mirrors jaxflow_hook.py's WATCH_DEADLINE_S (14280s — the Stop hook's asyncRewake timeout minus
// a 120s margin): a marker older than this belongs to a watcher no longer polling (killed, or
// its own deadline already fired).
export const WATCH_DEADLINE_S = 14280;

type WriteFs = { mkdirSync: typeof mkdirSync; writeFileSync: typeof writeFileSync; renameSync: typeof renameSync };
const systemWriteFs: WriteFs = { mkdirSync, writeFileSync, renameSync };

// Refuses unless sessionId is a canonical UUID — never a path outside CALLBACKS_ROOT (D3), the
// same refusal jaxflow.py's _write_callback_file applies to its own caller_session.
function assertCanonicalSession(sessionId: string): void {
  if (typeof sessionId !== "string" || !UUID_RE.test(sessionId)) throw new Error("invalid harness session");
}

// Atomic tmp+rename write, mirroring jaxflow.py's _write_callback_file. At most one watcher is
// ever live per session (D3), so no claim/rename step is needed on this side.
export function writeAnswerLine(sessionId: string, eventId: number, text: string, fs: WriteFs = systemWriteFs): void {
  assertCanonicalSession(sessionId);
  const dir = join(CALLBACKS_ROOT, sessionId);
  fs.mkdirSync(dir, { recursive: true });
  const dest = join(dir, `answer-${eventId}.answer`);
  const tmp = `${dest}.tmp`;
  fs.writeFileSync(tmp, text, "utf8");
  fs.renameSync(tmp, dest);
}

// D4: does the watcher's own `.armed` marker exist and is it still within its liveness window?
// The watcher removes it on delivery and on its own expiry, so plain existence is the live case;
// the mtime bound backstops a killed watcher's stale marker. Same UUID refusal as above.
export function isAnswerWatcherAlive(
  sessionId: string, eventId: number, stat: typeof statSync = statSync, now: () => number = Date.now,
): boolean {
  if (typeof sessionId !== "string" || !UUID_RE.test(sessionId)) return false;
  try {
    const st = stat(join(CALLBACKS_ROOT, sessionId, `answer-${eventId}.armed`));
    return now() - st.mtimeMs < WATCH_DEADLINE_S * 1000;
  } catch {
    return false;
  }
}
