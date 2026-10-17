import { describe, expect, it, vi } from "vitest";
import type { WorkflowQuestionAnswer, WorkflowQuestionShape } from "../../lib/workflow";
import { buildAnswerActions, capturePaneTail, injectAnswers, isCodexCommand, listLivePanes, KEY_INTERVAL_MS } from "./workflow-tmux";

const PANE = "%1";
const INCARNATION = "234790:1787586213";

const SHAPES: WorkflowQuestionShape[] = [
  { multiSelect: false, option_count: 2 },
  { multiSelect: true, option_count: 3 },
  { multiSelect: false, option_count: 2 },
];

const ANSWERS: WorkflowQuestionAnswer[] = [
  { question_number: 1, kind: "options", values: [2] },
  { question_number: 2, kind: "options", values: [1, 3] },
  { question_number: 3, kind: "text", value: "-ship; $(nope)" },
];

describe("buildAnswerActions", () => {
  it("pairs by question_number and emits one action per key", () => {
    expect(buildAnswerActions(SHAPES, [...ANSWERS].reverse())).toEqual([
      { kind: "key", key: "Down" }, { kind: "key", key: "Enter" },
      { kind: "key", key: "Space" }, { kind: "key", key: "Down" },
      { kind: "key", key: "Down" }, { kind: "key", key: "Space" },
      { kind: "key", key: "Down" }, { kind: "key", key: "Down" },
      { kind: "key", key: "Enter" },
      { kind: "key", key: "Down" }, { kind: "key", key: "Down" },
      { kind: "literal", value: "-ship; $(nope)" }, { kind: "key", key: "Enter" },
      { kind: "key", key: "Enter" },
    ]);
  });

  it("submits one single-select question without a review Enter", () => {
    expect(buildAnswerActions(
      [{ multiSelect: false, option_count: 2 }],
      [{ question_number: 1, kind: "options", values: [2] }],
    )).toEqual([
      { kind: "key", key: "Down" }, { kind: "key", key: "Enter" },
    ]);
  });

  it("traverses Submit then review Enter for one multiselect question", () => {
    expect(buildAnswerActions(
      [{ multiSelect: true, option_count: 3 }],
      [{ question_number: 1, kind: "options", values: [1, 3] }],
    )).toEqual([
      { kind: "key", key: "Space" }, { kind: "key", key: "Down" },
      { kind: "key", key: "Down" }, { kind: "key", key: "Space" },
      { kind: "key", key: "Down" }, { kind: "key", key: "Down" },
      { kind: "key", key: "Enter" }, { kind: "key", key: "Enter" },
    ]);
  });

  it("types a multiselect Other answer then submits review", () => {
    expect(buildAnswerActions(
      [{ multiSelect: true, option_count: 3 }],
      [{ question_number: 1, kind: "text", value: "ship without deploy" }],
    )).toEqual([
      { kind: "key", key: "Down" }, { kind: "key", key: "Down" }, { kind: "key", key: "Down" },
      { kind: "literal", value: "ship without deploy" },
      { kind: "key", key: "Down" }, { kind: "key", key: "Enter" }, { kind: "key", key: "Enter" },
    ]);
  });
});

describe("injectAnswers", () => {
  it("revalidates the live pane+incarnation via listLivePanes, then sends one argv call and pause(200) per action", async () => {
    const calls: string[][] = [];
    const pauses: number[] = [];
    const run = vi.fn(async (_file: string, args: string[]) => {
      calls.push(args);
      if (args[0] === "list-panes") return `${PANE} 234790 1787586213 bash\nother 234790 1787586213 bash\n`;
      return "";
    });
    const pause = vi.fn(async (ms: number) => { pauses.push(ms); });
    await injectAnswers(PANE, SHAPES, ANSWERS, INCARNATION, run, pause);
    expect(calls[0]).toEqual(["list-panes", "-a", "-F", "#{pane_id} #{pid} #{start_time} #{pane_current_command}"]);
    const sends = calls.slice(1);
    expect(sends[0]).toEqual(["send-keys", "-t", PANE, "Down"]);
    expect(sends.at(-1)).toEqual(["send-keys", "-t", PANE, "Enter"]);
    expect(pauses).toEqual(Array(sends.length).fill(200));
  });

  it("fails before any send when the pane is not live at all", async () => {
    const run = vi.fn(async (_file: string, args: string[]) => {
      if (args[0] === "list-panes") return "%9 234790 1787586213 bash\n";
      throw new Error("should not send");
    });
    await expect(injectAnswers(PANE, SHAPES, ANSWERS, INCARNATION, run, async () => {})).rejects.toThrow(
      "pane target is not live",
    );
    expect(run).toHaveBeenCalledTimes(1);
    expect(run.mock.calls.some(([, args]) => args[0] === "send-keys")).toBe(false);
  });

  it("fails before any send when the pane is live but the server incarnation changed (Finding 25)", async () => {
    const run = vi.fn(async (_file: string, args: string[]) => {
      if (args[0] === "list-panes") return `${PANE} 999999 1787599999 bash\n`;
      throw new Error("should not send");
    });
    await expect(injectAnswers(PANE, SHAPES, ANSWERS, INCARNATION, run, async () => {})).rejects.toThrow(
      "pane target is not live",
    );
    expect(run.mock.calls.some(([, args]) => args[0] === "send-keys")).toBe(false);
  });

  it("rejects a pane string that fails PANE_RE before ever calling tmux (Finding 25 — validate before send)", async () => {
    const run = vi.fn(async () => { throw new Error("should not be called"); });
    await expect(injectAnswers("not-a-pane", SHAPES, ANSWERS, INCARNATION, run, async () => {})).rejects.toThrow(
      "pane target is not live",
    );
    expect(run).not.toHaveBeenCalled();
  });

  it("sanitizes execFile failures so answer text never leaves the collector", async () => {
    const sentinel = "-ship; $(nope)";
    const run = vi.fn(async (_file: string, args: string[]) => {
      if (args[0] === "list-panes") return `${PANE} 234790 1787586213 bash\n`;
      const err = Object.assign(new Error(`Command failed: tmux ${args.join(" ")} ${sentinel}`), {
        cmd: `tmux ${args.join(" ")} ${sentinel}`, stdout: sentinel, stderr: sentinel,
      });
      throw err;
    });
    await expect(injectAnswers(PANE, SHAPES, ANSWERS, INCARNATION, run, async () => {})).rejects.toSatisfy(
      (e: unknown) => e instanceof Error && e.message === "tmux answer injection failed" && !e.message.includes(sentinel),
    );
  });

  it("a NULL stored tmux_incarnation is treated identically to a mismatch — no pane-membership-only fallback (Finding 28)", async () => {
    const run = vi.fn(async (_file: string, args: string[]) => {
      if (args[0] === "list-panes") return `${PANE} 234790 1787586213 bash\n`;
      throw new Error("should not send");
    });
    await expect(injectAnswers(PANE, SHAPES, ANSWERS, null as unknown as string, run, async () => {})).rejects.toThrow(
      "pane target is not live",
    );
    expect(run).toHaveBeenCalledTimes(1);
    expect(run.mock.calls.some(([, args]) => args[0] === "send-keys")).toBe(false);
  });
});

describe("runtime command association (MOA-469)", () => {
  it("recognizes positive Codex commands and refuses generic shells", () => {
    expect(isCodexCommand("codex")).toBe(true);
    expect(isCodexCommand("/home/rafa/.local/bin/codex")).toBe(true);
    expect(isCodexCommand("codex2")).toBe(true);
    expect(isCodexCommand("bash")).toBe(false);
    expect(isCodexCommand("zsh")).toBe(false);
  });

  it("records the pane_current_command per pane in the live snapshot", async () => {
    const run = vi.fn(async (_file: string, args: string[]) => {
      expect(args).toEqual(["list-panes", "-a", "-F", "#{pane_id} #{pid} #{start_time} #{pane_current_command}"]);
      return "%1 234790 1787586213 codex\n%2 234790 1787586213 bash\n";
    });
    const live = await listLivePanes(run);
    expect(live.paneCommands.get("%1")).toBe("codex");
    expect(live.paneCommands.get("%2")).toBe("bash");
    expect(live.paneCommands.has("%3")).toBe(false);
  });
});

describe("listLivePanes", () => {
  it("returns pane-id membership plus the ONE server-wide incarnation string", async () => {
    const run = vi.fn(async (_file: string, args: string[]) => {
      expect(args).toEqual(["list-panes", "-a", "-F", "#{pane_id} #{pid} #{start_time} #{pane_current_command}"]);
      return "%1 234790 1787586213 bash\n%2 234790 1787586213 bash\n"; // pid/start_time identical on every line
    });
    const live = await listLivePanes(run);
    expect(live).toEqual({ paneIds: new Set(["%1", "%2"]), incarnation: "234790:1787586213", paneCommands: new Map([["%1", "bash"], ["%2", "bash"]]) });
  });

  it("returns an empty set and null incarnation when tmux has no server (never throws)", async () => {
    const run = vi.fn(async () => { throw new Error("no server running on /tmp/tmux-1000/default"); });
    expect(await listLivePanes(run)).toEqual({ paneIds: new Set(), incarnation: null, paneCommands: new Map() });
  });

  it("also reads an 'error connecting' failure as no-server (never throws)", async () => {
    const run = vi.fn(async () => { throw new Error("error connecting to /tmp/tmux-1000/default (No such file or directory)"); });
    expect(await listLivePanes(run)).toEqual({ paneIds: new Set(), incarnation: null, paneCommands: new Map() });
  });

  it("a genuine tmux command failure (missing binary, permission, ...) throws instead of reading as an empty snapshot (branch review Finding 5)", async () => {
    const run = vi.fn(async () => { throw new Error("spawn tmux ENOENT"); });
    await expect(listLivePanes(run)).rejects.toThrow("spawn tmux ENOENT");
  });

  it("skips a malformed line rather than throwing, still resolves incarnation from a good line", async () => {
    const run = vi.fn(async () => "not-a-valid-line\n%1 234790 1787586213 bash\n");
    const live = await listLivePanes(run);
    expect(live).toEqual({ paneIds: new Set(["%1"]), incarnation: "234790:1787586213", paneCommands: new Map([["%1", "bash"]]) });
  });

  it("treats timeout as source failure, not parsed panes", async () => {
    const run = vi.fn(async () => {
      throw Object.assign(new Error("timeout"), {
        code: "ETIMEDOUT",
        stdout: "%1 234790 1787586213 bash\n",
      });
    });
    await expect(listLivePanes(run)).rejects.toMatchObject({ code: "ETIMEDOUT" });
  });

  it("treats output overflow as source failure, not partial parsed success", async () => {
    const run = vi.fn(async () => {
      throw Object.assign(new Error("overflow"), {
        code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
        stdout: "%1 234790 1787586213 bash\n",
      });
    });
    await expect(listLivePanes(run)).rejects.toMatchObject({ code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER" });
  });
});

describe("capturePaneTail (Sessions mosaic spec §7 Decision 10)", () => {
  it("requests exactly `lines` back from the bottom and returns exactly N trimmed lines for whatever run() returns", async () => {
    const run = vi.fn(async (_file: string, args: string[]) => {
      expect(args).toEqual(["capture-pane", "-p", "-t", "%1", "-S", "-7"]);
      return "line1  \nline2\nline3\n";
    });
    const lines = await capturePaneTail("%1", 8, run);
    expect(lines).toEqual(["line1", "line2", "line3"]);
  });

  it("truncates a line over 200 chars", async () => {
    const long = "x".repeat(250);
    const run = vi.fn(async () => `${long}\n`);
    const lines = await capturePaneTail("%1", 1, run);
    expect(lines).toEqual([long.slice(0, 200)]);
    expect(lines[0]).toHaveLength(200);
  });

  it("rejects — never resolves with an empty/partial string — on a non-zero exit (round-2 F2)", async () => {
    const run = vi.fn(async () => { throw new Error("tmux: no such pane"); });
    await expect(capturePaneTail("%9", 4, run)).rejects.toThrow("tmux: no such pane");
  });

  it("rejects on a timed-out run the same way (round-2 F2)", async () => {
    const run = vi.fn(async () => { throw Object.assign(new Error("timeout"), { code: "ETIMEDOUT" }); });
    await expect(capturePaneTail("%1", 4, run)).rejects.toMatchObject({ code: "ETIMEDOUT" });
  });
});
