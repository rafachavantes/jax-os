import { describe, expect, it } from "vitest";
import {
  SESSION_ACCESS_FORMAT,
  collectSessionAccessSnapshot,
  parseSessionAccessSnapshot,
  type SessionAccessRunner,
} from "./session-access";

const row = (overrides: Partial<Record<string, string>> = {}) => {
  const fields = {
    pane: "%7", session: "$2", window: "@3", pid: "1234", start: "1700000000",
    dead: "0", windowActive: "1", paneActive: "1", name: "project agent",
    command: "claude", title: "jax-os | 01a1ea73-662c-7c31-9f94-cf156...", ...overrides,
  };
  return [fields.pane, fields.session, fields.window, fields.pid, fields.start,
    fields.dead, fields.windowActive, fields.paneActive, fields.name, fields.command, fields.title].join("\t");
};

function runner(stdout: string | Error): SessionAccessRunner & { calls: [string, string[]][] } {
  const calls: [string, string[]][] = [];
  const run = (async (file: string, args: string[]) => {
    calls.push([file, args]);
    if (stdout instanceof Error) throw stdout;
    return stdout;
  }) as SessionAccessRunner & { calls: [string, string[]][] };
  run.calls = calls;
  return run;
}

function exitError(code: number, stderr: string): Error & { code: number; stderr: string } {
  return Object.assign(new Error("Command failed"), { code, stderr });
}

describe("parseSessionAccessSnapshot", () => {
  it("maps the fixed native row to an ID-only target and server incarnation", () => {
    const snapshot = parseSessionAccessSnapshot(`${row()}\n`);
    expect(snapshot.serverIncarnation).toBe("1234:1700000000");
    expect(snapshot.rows).toHaveLength(1);
    expect(snapshot.rows[0]).toMatchObject({
      pane: "%7", sessionId: "$2", windowId: "@3", sessionName: "project agent",
      target: "$2:@3.%7", incarnation: "1234:1700000000", dead: false, eligible: true,
      windowActive: true, paneActive: true, command: "claude",
      title: "jax-os | 01a1ea73-662c-7c31-9f94-cf156...",
    });
  });

  it("keeps dead panes visible as metadata while marking them dead", () => {
    const snapshot = parseSessionAccessSnapshot(`${row({ dead: "1", paneActive: "0" })}\n`);
    expect(snapshot.rows[0].dead).toBe(true);
    expect(snapshot.rows[0].eligible).toBe(true);
  });

  it("fails the whole source on a structurally unreadable row", () => {
    expect(() => parseSessionAccessSnapshot("unreadable native row\n")).toThrow("session snapshot unreadable");
    expect(() => parseSessionAccessSnapshot(`${row()}\textra\n`)).toThrow("session snapshot unreadable");
    expect(() => parseSessionAccessSnapshot(`${row()}\textra\textra2\n`)).toThrow("session snapshot unreadable");
    expect(() => parseSessionAccessSnapshot(`${row({ dead: "2" })}\n`)).toThrow("session snapshot unreadable");
  });

  it("fails the whole source on mixed server incarnations and over-cap output", () => {
    expect(() => parseSessionAccessSnapshot(`${row()}\n${row({ pane: "%8", pid: "9999" })}\n`))
      .toThrow("session snapshot unreadable");
    expect(() => parseSessionAccessSnapshot(Array.from({ length: 257 }, (_, i) => row({ pane: `%${i + 1}` })).join("\n")))
      .toThrow("session snapshot unreadable");
    expect(() => parseSessionAccessSnapshot(`${row()}\n${"x".repeat(129 * 1024)}\n`))
      .toThrow("session snapshot unreadable");
  });

  it("discards only the raw title metadata when it carries controls or exceeds its cap", () => {
    const snapshot = parseSessionAccessSnapshot([
      row({ pane: "%1", title: "bad\u0000title" }),
      row({ pane: "%2", title: "x".repeat(1025) }),
    ].join("\n"));
    expect(snapshot.rows[0].title).toBeUndefined();
    expect(snapshot.rows[1].title).toBeUndefined();
    expect(snapshot.rows.map((r) => r.eligible)).toEqual([true, true]);
    expect(snapshot.rows[0].command).toBe("claude");
  });

  it("keeps same-title panes independently eligible while duplicate pane rows stay ineligible", () => {
    const sameTitle = parseSessionAccessSnapshot([row({ pane: "%1" }), row({ pane: "%2" })].join("\n"));
    expect(sameTitle.rows.map((r) => r.eligible)).toEqual([true, true]);
    const duplicate = parseSessionAccessSnapshot([row({ pane: "%1" }), row({ pane: "%1" })].join("\n"));
    expect(duplicate.rows.map((r) => r.eligible)).toEqual([false, false]);
  });

  it("marks invalid IDs, over-cap names and multi-session panes ineligible without failing the source", () => {
    const snapshot = parseSessionAccessSnapshot([
      row({ pane: "%bad" }),
      row({ pane: "%8", name: "n".repeat(257) }),
      row({ pane: "%9", name: "other" }),
      row({ pane: "%9", name: "other-2" }),
      row({ pane: "%10" }),
    ].join("\n"));
    const byPane = new Map(snapshot.rows.map((r) => [r.pane, r.eligible]));
    expect(byPane.get("%bad")).toBe(false);
    expect(byPane.get("%8")).toBe(false);
    expect(byPane.get("%9")).toBe(false);
    expect(byPane.get("%10")).toBe(true);
  });
});

describe("collectSessionAccessSnapshot", () => {
  it("runs the one fixed native command", async () => {
    const run = runner(`${row()}\n`);
    const snapshot = await collectSessionAccessSnapshot(run);
    expect(run.calls).toEqual([["tmux", ["list-panes", "-a", "-F", SESSION_ACCESS_FORMAT]]]);
    expect(snapshot.rows[0].target).toBe("$2:@3.%7");
  });

  it("treats the exact no-server stderr shapes as a quiet empty snapshot", async () => {
    for (const stderr of [
      "no server running on /tmp/tmux-1000/default",
      "no server running at /tmp/tmux-1000/default",
      "no server running /tmp/tmux-1000/default",
      "error connecting to /tmp/tmux-1000/default (No such file or directory)",
    ]) {
      const snapshot = await collectSessionAccessSnapshot(runner(exitError(1, `${stderr}\n`)));
      expect(snapshot).toEqual({ rows: [], serverIncarnation: null });
    }
  });

  it("keeps any other failure a bounded source failure, never a quiet empty snapshot", async () => {
    await expect(collectSessionAccessSnapshot(runner(exitError(1, "permission denied"))))
      .rejects.toThrow("session snapshot unavailable");
    await expect(collectSessionAccessSnapshot(runner(Object.assign(new Error("spawn tmux ENOENT"), { code: "ENOENT" }))))
      .rejects.toThrow("session snapshot unavailable");
    await expect(collectSessionAccessSnapshot(runner(exitError(2, "no server running on /tmp/s"))))
      .rejects.toThrow("session snapshot unavailable");
    await expect(collectSessionAccessSnapshot(runner(exitError(1, "no server running on /tmp/a\nno server running on /tmp/b"))))
      .rejects.toThrow("session snapshot unavailable");
  });
});
