import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { PANE_RE, type WorkflowQuestionAnswer, type WorkflowQuestionShape } from "../../lib/workflow";
import type { CodexTerminalObservation } from "./codex-tmux";

type ExecFile = (file: string, args: string[]) => Promise<string>;
type Pause = (ms: number) => Promise<void>;
export type AnswerAction =
  | { kind: "key"; key: string }
  | { kind: "literal"; value: string };

const execFileAsync = promisify(execFile);
const systemExec: ExecFile = async (file, args) => {
  const { stdout } = await execFileAsync(file, args, {
    encoding: "utf8",
    timeout: 5_000,
    maxBuffer: 1024 * 1024,
  });
  return stdout;
};
const systemPause: Pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
export const KEY_INTERVAL_MS = 200;

const downs = (count: number): AnswerAction[] =>
  Array.from({ length: count }, () => ({ kind: "key" as const, key: "Down" }));

export function buildAnswerActions(
  questions: WorkflowQuestionShape[],
  answers: WorkflowQuestionAnswer[],
): AnswerAction[] {
  const actions: AnswerAction[] = [];
  const byNumber = new Map(answers.map((answer) => [answer.question_number, answer]));
  for (let i = 0; i < questions.length; i += 1) {
    const question = questions[i];
    const answer = byNumber.get(i + 1);
    if (!answer) throw new Error("validated answer missing");
    if (answer.kind === "text") {
      actions.push(...downs(question.option_count));
      actions.push({ kind: "literal", value: answer.value });
      if (question.multiSelect) actions.push({ kind: "key", key: "Down" });
      actions.push({ kind: "key", key: "Enter" });
      continue;
    }
    const values = [...answer.values].sort((a, b) => a - b);
    if (!question.multiSelect) {
      actions.push(...downs(values[0] - 1), { kind: "key", key: "Enter" });
      continue;
    }
    let focused = 1;
    for (const value of values) {
      actions.push(...downs(value - focused), { kind: "key", key: "Space" });
      focused = value;
    }
    actions.push(...downs(question.option_count - focused + 2), { kind: "key", key: "Enter" });
  }
  if (questions.length > 1 || questions[0]?.multiSelect) {
    actions.push({ kind: "key", key: "Enter" });
  }
  return actions;
}

async function assertLivePane(pane: string, tmuxIncarnation: string, run: ExecFile): Promise<void> {
  if (!PANE_RE.test(pane)) throw new Error("pane target is not live");
  const live = await listLivePanes(run);
  if (!live.paneIds.has(pane) || live.incarnation !== tmuxIncarnation) {
    throw new Error("pane target is not live");
  }
}

export async function injectAnswers(
  pane: string,
  questions: WorkflowQuestionShape[],
  answers: WorkflowQuestionAnswer[],
  tmuxIncarnation: string,
  run: ExecFile = systemExec,
  pause: Pause = systemPause,
): Promise<void> {
  await assertLivePane(pane, tmuxIncarnation, run);
  for (const action of buildAnswerActions(questions, answers)) {
    const args = action.kind === "literal"
      ? ["send-keys", "-t", pane, "-l", "--", action.value]
      : ["send-keys", "-t", pane, action.key];
    try {
      await run("tmux", args);
    } catch {
      throw new Error("tmux answer injection failed");
    }
    await pause(KEY_INTERVAL_MS);
  }
}

const LIVE_LINE_RE = /^(%[0-9]{1,10}) ([0-9]{1,10}) ([0-9]{1,20}) (.*)$/;

// Hub snapshot (M3b): one fixed tab-delimited command per poll, title last and never trusted as a
// whole line — a title containing a tab or newline shifts the field count and fails the snapshot.
export const LIVE_PANES_FORMAT = [
  "#{pane_id}", "#{pid}", "#{start_time}", "#{pane_pid}", "#{pane_dead}",
  "#{pane_current_command}", "#{session_name}", "#{pane_title}",
].join("\t");

const TITLE_CAP = 1024;
const SESSION_CAP = 256;
const controlFree = (value: string) => !/[\u0000-\u001f\u007f]/.test(value);

type NativeLiveRow = {
  pane: string; pid: string; start: string; panePid: string; dead: string;
  command: string; session: string; title: string;
};

// Framing failure (null) fails the WHOLE snapshot; invalid title/session metadata only makes the
// row's observation ineligible (handled by the caller).
function parseNativeLiveLine(line: string): NativeLiveRow | null {
  const f = line.split("\t");
  if (f.length !== 8) return null;
  const [pane, pid, start, panePid, dead, command, session, title] = f;
  if (!PANE_RE.test(pane) || !/^[0-9]{1,10}$/.test(pid) || !/^[0-9]{1,20}$/.test(start) || !/^[0-9]{1,10}$/.test(panePid)) return null;
  if (dead !== "0" && dead !== "1") return null;
  return { pane, pid, start, panePid, dead, command, session, title };
}

// Runtime association for a live pane's CURRENT process (MOA-469 §4): positively-Codex commands are
// runtime evidence; a generic shell or unknown command is NOT proof of a runtime switch — callers
// flag uncertainty rather than assigning a different session.
export function isCodexCommand(cmd: string): boolean {
  const base = cmd.split("/").pop()?.toLowerCase() ?? "";
  return base === "codex" || base.startsWith("codex");
}

// tmux's own wording for "no server running at all" or a missing socket — the only genuinely
// quiet-empty cases (§7.3). Permission denied, corrupt output, a timeout/overflow or a missing
// binary is a real source failure and must propagate, not become "zero live panes" (Finding 5).
const NO_SERVER_RE = /no server running|error connecting[^\n]*No such file or directory/i;

// M5 Task 2 (F14): pane PID is its OWN observed fact, inserted before the free command field.
// It is NOT the server `#{pid}`/`#{start_time}` incarnation pair. Optional in the return type so
// older injected fixtures (command in the fourth field) keep parsing as command-only.
export type LivePanesResult = {
  paneIds: Set<string>;
  incarnation: string | null;
  paneCommands: Map<string, string>;
  panePids?: Map<string, number>;
  paneDead?: Map<string, boolean>;
  // Native rows only (real production output). Legacy injected fixtures keep their own branch and
  // produce no binding observations.
  codexTerminals?: CodexTerminalObservation[];
};

export async function listLivePanes(
  run: ExecFile = systemExec,
): Promise<LivePanesResult> {
  let output: string;
  try {
    output = await run("tmux", ["list-panes", "-a", "-F", LIVE_PANES_FORMAT]);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    if (NO_SERVER_RE.test(message)) {
      return { paneIds: new Set(), incarnation: null, paneCommands: new Map(), panePids: new Map(), paneDead: new Map(), codexTerminals: [] }; // no tmux server — zero live panes, not an error
    }
    throw e; // genuine command failure — let the caller's error contract surface it
  }
  const paneIds = new Set<string>();
  const paneCommands = new Map<string, string>();
  const panePids = new Map<string, number>();
  const paneDead = new Map<string, boolean>();
  const codexTerminals: CodexTerminalObservation[] = [];
  let hasNativeRows = false;
  let incarnation: string | null = null;
  let invalid = 0;
  for (const rawLine of output.split("\n")) {
    const line = rawLine.replace(/\r$/, "");
    if (!line.trim()) continue;
    if (line.includes("\t")) {
      hasNativeRows = true;
      const native = parseNativeLiveLine(line);
      if (!native) { invalid += 1; continue; }
      paneIds.add(native.pane);
      panePids.set(native.pane, Number(native.panePid));
      paneDead.set(native.pane, native.dead === "1");
      if (native.command) paneCommands.set(native.pane, native.command);
      if (incarnation === null) incarnation = `${native.pid}:${native.start}`;
      const eligible = native.session.length > 0 && Buffer.byteLength(native.session, "utf8") <= SESSION_CAP
        && controlFree(native.session) && native.title.length > 0 && controlFree(native.title)
        && Buffer.byteLength(native.title, "utf8") <= TITLE_CAP;
      codexTerminals.push({
        pane: native.pane,
        tmuxIncarnation: `${native.pid}:${native.start}`,
        session: native.session,
        command: native.command,
        dead: native.dead === "1",
        eligible,
        title: eligible ? native.title : "",
      });
      continue;
    }
    const trimmed = line.trim();
    const m = LIVE_LINE_RE.exec(trimmed);
    if (!m) { invalid += 1; continue; }
    paneIds.add(m[1]);
    // New format: "<pane_pid> <pane_dead> <command…>"; the legacy injected fixtures carry
    // "<pane_pid> <command…>" or only the command, which keep parsing as before.
    // Native rows may carry a blank pane_current_command, so the command suffix is optional;
    // the pinned ambiguous `<pid> 0|1` legacy shape is native pane_dead with no command.
    const withDeath = /^([0-9]{1,10})\s+([01])(?:\s+(.*))?$/.exec(m[4]);
    const withPid = withDeath ? null : /^([0-9]{1,10})\s(.*)$/.exec(m[4]);
    if (withDeath) {
      panePids.set(m[1], Number(withDeath[1]));
      paneDead.set(m[1], withDeath[2] === "1");
      if (withDeath[3]) paneCommands.set(m[1], withDeath[3]);
    } else if (withPid) {
      panePids.set(m[1], Number(withPid[1]));
      if (withPid[2]) paneCommands.set(m[1], withPid[2]);
    } else if (m[4]) {
      paneCommands.set(m[1], m[4]);
    }
    if (incarnation === null) incarnation = `${m[2]}:${m[3]}`; // server-scope — read once, from the first good line
  }
  // A nonempty snapshot with unparseable rows is corrupt, never a confident pane set.
  if (invalid > 0) throw new Error("tmux snapshot unreadable");
  const base = { paneIds, incarnation, paneCommands, panePids, paneDead };
  return hasNativeRows ? { ...base, codexTerminals } : base;
}

// Sessions mosaic spec §7 Decision 10: "last N output lines live". `-S -{lines-1}` asks tmux for
// exactly `lines` rows counted back from the bottom of the pane. Round-2 F2: NO try/catch here —
// a rejected/timed-out `run` must propagate as a rejection, the function must never resolve with
// an empty or partial string; the caller (the panes-tail route, Task 5) is what turns a
// rejection into `lines: null` for that one pane.
export async function capturePaneTail(pane: string, lines: number, run: ExecFile = systemExec): Promise<string[]> {
  const raw = await run("tmux", ["capture-pane", "-p", "-t", pane, "-S", `-${lines - 1}`]);
  // ponytail: trimEnd() strips tmux's own trailing-space pane padding, never real content —
  // leading whitespace (indentation) is preserved. slice(0,200) matches LIMITS.capsuleExcerpt's
  // order of magnitude (src/lib/workflow.ts).
  return raw.replace(/\n$/, "").split("\n").map((line) => line.trimEnd().slice(0, 200));
}
