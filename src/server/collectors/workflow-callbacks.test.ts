import { describe, expect, it, vi } from "vitest";
import { CALLBACKS_ROOT, isAnswerWatcherAlive, WATCH_DEADLINE_S, writeAnswerLine } from "./workflow-callbacks";

const UUID = "0191f0aa-cccc-7000-8000-000000000003";

describe("writeAnswerLine (D3)", () => {
  it("writes atomically: content goes to a .tmp path, then is renamed onto the real one, under CALLBACKS_ROOT/<sessionId>", () => {
    const calls: string[] = [];
    const fs = {
      mkdirSync: vi.fn((dir: string) => { calls.push(`mkdir ${dir}`); }),
      writeFileSync: vi.fn((path: string, text: string) => { calls.push(`write ${path} ${text}`); }),
      renameSync: vi.fn((from: string, to: string) => { calls.push(`rename ${from} -> ${to}`); }),
    };
    writeAnswerLine(UUID, 42, "[Jax OS · Rafa] pode", fs as never);
    const dir = `${CALLBACKS_ROOT}/${UUID}`;
    expect(calls).toEqual([
      `mkdir ${dir}`,
      `write ${dir}/answer-42.answer.tmp [Jax OS · Rafa] pode`,
      `rename ${dir}/answer-42.answer.tmp -> ${dir}/answer-42.answer`,
    ]);
  });

  it("throws and touches nothing when sessionId is not a canonical UUID", () => {
    const fs = { mkdirSync: vi.fn(), writeFileSync: vi.fn(), renameSync: vi.fn() };
    expect(() => writeAnswerLine("../../etc", 1, "x", fs as never)).toThrow("invalid harness session");
    expect(fs.mkdirSync).not.toHaveBeenCalled();
    expect(fs.writeFileSync).not.toHaveBeenCalled();
  });
});

describe("isAnswerWatcherAlive (D4)", () => {
  it("true when the .armed marker exists and its mtime is younger than WATCH_DEADLINE_S", () => {
    const stat = vi.fn(() => ({ mtimeMs: 1_000_000 }) as never);
    const now = () => 1_000_000 + (WATCH_DEADLINE_S - 1) * 1000;
    expect(isAnswerWatcherAlive(UUID, 42, stat, now)).toBe(true);
    expect(stat).toHaveBeenCalledWith(`${CALLBACKS_ROOT}/${UUID}/answer-42.armed`);
  });

  it("false when the marker's mtime is older than WATCH_DEADLINE_S (a killed watcher's stale marker)", () => {
    const stat = vi.fn(() => ({ mtimeMs: 1_000_000 }) as never);
    const now = () => 1_000_000 + (WATCH_DEADLINE_S + 1) * 1000;
    expect(isAnswerWatcherAlive(UUID, 42, stat, now)).toBe(false);
  });

  it("false when the marker file does not exist (stat throws)", () => {
    const stat = vi.fn(() => { throw new Error("ENOENT"); });
    expect(isAnswerWatcherAlive(UUID, 42, stat as never)).toBe(false);
  });

  it("false, and never calls stat, when sessionId is not a canonical UUID", () => {
    const stat = vi.fn();
    expect(isAnswerWatcherAlive("not-a-uuid", 1, stat as never)).toBe(false);
    expect(stat).not.toHaveBeenCalled();
  });
});
