import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { PANE_RE, type WorkflowQuestionAnswer, type WorkflowQuestionShape } from "../../lib/workflow";

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

// Runtime association for a live pane's CURRENT process (MOA-469 §4): positively-Codex commands are
// runtime evidence; a generic shell or unknown command is NOT proof of a runtime switch — callers
// flag uncertainty rather than assigning a different session.
export function isCodexCommand(cmd: string): boolean {
  const base = cmd.split("/").pop()?.toLowerCase() ?? "";
  return base === "codex" || base.startsWith("codex");
}

// tmux's own wording for "no server running at all" — the one genuinely quiet-empty case (§7.3).
// Anything else (missing binary/ENOENT, permission, a corrupt socket, ...) is a real source
// failure and must propagate, not be swallowed into "zero live panes" (Finding 5).
const NO_SERVER_RE = /no server running|error connecting/i;

export async function listLivePanes(
  run: ExecFile = systemExec,
): Promise<{ paneIds: Set<string>; incarnation: string | null; paneCommands: Map<string, string> }> {
  let output: string;
  try {
    output = await run("tmux", ["list-panes", "-a", "-F", "#{pane_id} #{pid} #{start_time} #{pane_current_command}"]);
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    if (NO_SERVER_RE.test(message)) {
      return { paneIds: new Set(), incarnation: null, paneCommands: new Map() }; // no tmux server — zero live panes, not an error
    }
    throw e; // genuine command failure — let the caller's error contract surface it
  }
  const paneIds = new Set<string>();
  const paneCommands = new Map<string, string>();
  let incarnation: string | null = null;
  for (const line of output.split("\n")) {
    const m = LIVE_LINE_RE.exec(line.trim());
    if (!m) continue;
    paneIds.add(m[1]);
    if (m[4]) paneCommands.set(m[1], m[4]);
    if (incarnation === null) incarnation = `${m[2]}:${m[3]}`; // server-scope — read once, from the first good line
  }
  return { paneIds, incarnation, paneCommands };
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
