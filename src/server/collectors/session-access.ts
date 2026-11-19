// The only new native system access for exact session targets (M3b KISS): one fixed
// `tmux list-panes -a` snapshot with an injected parser. Names are DATA (the last field), never
// executable arguments; the ID-only target is built from validated native IDs.

import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { INCARNATION_RE, PANE_RE, SESSION_NAME_CAP, TARGET_RE } from "../../lib/sessionAccess";

export const SESSION_ACCESS_FORMAT =
  "#{pane_id}\t#{session_id}\t#{window_id}\t#{pid}\t#{start_time}\t#{pane_dead}\t#{window_active}\t#{pane_active}\t#{session_name}";

const MAX_ROWS = 256;
const MAX_BYTES = 128 * 1024;
const SESSION_ID_RE = /^\$[0-9]{1,10}$/;
const WINDOW_ID_RE = /^@[0-9]{1,10}$/;

export type SessionAccessRow = {
  pane: string;
  sessionId: string;
  windowId: string;
  serverPid: string;
  serverStart: string;
  dead: boolean;
  windowActive: boolean;
  paneActive: boolean;
  sessionName: string;
  // "" when the row's native IDs are malformed; `eligible` is the only field a resolver may trust.
  target: string;
  incarnation: string;
  eligible: boolean;
};

export type SessionAccessSnapshot = { rows: SessionAccessRow[]; serverIncarnation: string | null };

export type SessionAccessRunner = (file: string, args: string[]) => Promise<string>;

const execFileAsync = promisify(execFile);
const systemExec: SessionAccessRunner = async (file, args) => {
  const { stdout } = await execFileAsync(file, args, { encoding: "utf8", timeout: 2_500, maxBuffer: MAX_BYTES });
  return stdout;
};

function unreadable(): never {
  throw new Error("session snapshot unreadable");
}

function byteLength(value: string): number {
  return Buffer.byteLength(value, "utf8");
}

export function parseSessionAccessSnapshot(stdout: string): SessionAccessSnapshot {
  if (byteLength(stdout) > MAX_BYTES) unreadable();
  const lines = stdout.split("\n").map((line) => line.replace(/\r$/, "")).filter((line) => line.length > 0);
  if (lines.length > MAX_ROWS) unreadable();

  const parsed = lines.map((line) => {
    const fields = line.split("\t");
    if (fields.length !== 9) unreadable();
    const [pane, sessionId, windowId, serverPid, serverStart, dead, windowActive, paneActive, sessionName] = fields;
    if (dead !== "0" && dead !== "1") unreadable();
    if (windowActive !== "0" && windowActive !== "1") unreadable();
    if (paneActive !== "0" && paneActive !== "1") unreadable();
    const validIncarnation = /^[0-9]{1,10}$/.test(serverPid) && /^[0-9]{1,20}$/.test(serverStart);
    const validIds = PANE_RE.test(pane) && SESSION_ID_RE.test(sessionId) && WINDOW_ID_RE.test(windowId);
    const validName = sessionName.length > 0 && byteLength(sessionName) <= SESSION_NAME_CAP && !/[\u0000-\u001f]/.test(sessionName);
    return {
      pane, sessionId, windowId, serverPid, serverStart,
      dead: dead === "1", windowActive: windowActive === "1", paneActive: paneActive === "1", sessionName,
      validIncarnation, validIds, validName,
    };
  });

  const incarnations = new Set(parsed.filter((r) => r.validIncarnation).map((r) => `${r.serverPid}:${r.serverStart}`));
  if (incarnations.size > 1) unreadable();
  const serverIncarnation = incarnations.size === 1 ? [...incarnations][0] : null;
  if (serverIncarnation !== null && !INCARNATION_RE.test(serverIncarnation)) unreadable();

  const paneCounts = new Map<string, number>();
  for (const r of parsed) paneCounts.set(r.pane, (paneCounts.get(r.pane) ?? 0) + 1);

  const rows = parsed.map((r): SessionAccessRow => {
    const target = r.validIds ? `${r.sessionId}:${r.windowId}.${r.pane}` : "";
    return {
      pane: r.pane, sessionId: r.sessionId, windowId: r.windowId, serverPid: r.serverPid, serverStart: r.serverStart,
      dead: r.dead, windowActive: r.windowActive, paneActive: r.paneActive, sessionName: r.sessionName,
      target,
      incarnation: r.validIncarnation ? `${r.serverPid}:${r.serverStart}` : "",
      eligible: r.validIds && r.validIncarnation && r.validName && (paneCounts.get(r.pane) ?? 0) === 1 && TARGET_RE.test(target),
    };
  });
  return { rows, serverIncarnation };
}

const NO_SERVER_SHAPES = [
  /^no server running on [^\s]+$/i,
  /^no server running at [^\s]+$/i,
  /^no server running [^\s]+$/i,
  /^error connecting to [^\s]+ \(No such file or directory\)$/i,
];

function isQuietNoServer(e: unknown): boolean {
  const err = e as { code?: unknown; stderr?: unknown } | null;
  if (!err || err.code !== 1 || typeof err.stderr !== "string") return false;
  const stderr = err.stderr.trim();
  return NO_SERVER_SHAPES.some((shape) => shape.test(stderr));
}

export async function collectSessionAccessSnapshot(run: SessionAccessRunner = systemExec): Promise<SessionAccessSnapshot> {
  let stdout: string;
  try {
    stdout = await run("tmux", ["list-panes", "-a", "-F", SESSION_ACCESS_FORMAT]);
  } catch (e) {
    if (isQuietNoServer(e)) return { rows: [], serverIncarnation: null };
    throw new Error("session snapshot unavailable");
  }
  return parseSessionAccessSnapshot(stdout);
}
