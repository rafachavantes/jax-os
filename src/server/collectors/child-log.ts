import { closeSync, fstatSync, lstatSync, openSync, readSync } from "node:fs";
import { join } from "node:path";
import { ROOTS, relativeWithin } from "./files";

// Same 12-lowercase-hex shape ingress already enforces for root_build_run_id/resumes_run_id
// (workflow-events.ts:117-118) — mirrored, never redefined looser (no path-traversal segment).
const RUN_ID_RE = /^[0-9a-f]{12}$/;

// Mirrors scripts/jaxflow_run.py's child_log_path(repo, run_id) exactly (spec §8, cold review F1,
// risk 4; round 2 F1 hardens it). Both inputs are unvalidated for path shape at ingress — checked
// HERE, before any join reaches openSync: `run_id` against the shape above, `repo` against the
// Files allowlist via `relativeWithin` (files.ts, same realpathSync containment `resolveWithin`
// uses). Either failure returns null (best-effort, no throw).
export function childLogPath(repo: string, runId: string): string | null {
  if (!RUN_ID_RE.test(runId)) return null;
  if (relativeWithin(ROOTS.repos, repo) === null) return null;
  return join(repo, ".local", "runs", runId, "child.log");
}

// Mirrors jaxflow_run.py's CHILD_LOG_CLASSIFY_WINDOW (64 KiB).
export const CHILD_LOG_TAIL_BYTES = 65536;

// Reads at most the last CHILD_LOG_TAIL_BYTES bytes via a seek-from-end (spec §8) — never a
// full-file read. Returns null on ANY failure (missing file, permission, not-a-regular-file):
// best-effort enrichment, never a thrown poll (AGENTS.md error contract).
export function readChildLogTail(path: string): string | null {
  try { // round 2 F1: lstat (not stat) so a symlinked child.log is REJECTED, not followed
    if (!lstatSync(path).isFile()) return null;
  } catch {
    return null;
  }
  let fd: number;
  try {
    fd = openSync(path, "r");
  } catch {
    return null;
  }
  try {
    const size = fstatSync(fd).size;
    const len = Math.min(size, CHILD_LOG_TAIL_BYTES);
    const start = size - len;
    const buf = Buffer.alloc(len);
    let read = 0;
    while (read < len) {
      const n = readSync(fd, buf, read, len - read, start + read);
      if (n === 0) break;
      read += n;
    }
    return buf.subarray(0, read).toString("utf8");
  } catch {
    return null;
  } finally {
    closeSync(fd);
  }
}

// TS port of scripts/jaxflow_run.py's _last_stream_text_line (spec §8, cold review F12, round 2
// F5). Scans EVERY line independently; a line that fails to JSON-parse, isn't an object, or isn't
// a `text` event with a string `part.text` is SKIPPED — never aborts the scan, so a malformed
// TRAILING record never blanks an earlier valid one. Of the lines that DO parse as a `text` event,
// keeps the LAST one and returns the FIRST physical line of its part.text, bounded to 300 chars.
// `text` is expected already redacted by the caller (§8 — this function never touches secrets).
export function lastStreamTextLine(text: string): string | null {
  let lastText: string | null = null;
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    let event: unknown;
    try {
      event = JSON.parse(line);
    } catch {
      continue;
    }
    if (!event || typeof event !== "object" || Array.isArray(event)) continue;
    const e = event as Record<string, unknown>;
    if (e.type !== "text") continue;
    const part = e.part;
    const t = part && typeof part === "object" && !Array.isArray(part) ? (part as Record<string, unknown>).text : undefined;
    if (typeof t === "string" && t.trim().length > 0) lastText = t;
  }
  if (lastText === null) return null;
  const firstLine = lastText.split(/\r?\n/)[0].trim();
  return firstLine.length > 0 ? firstLine.slice(0, 300) : null;
}
