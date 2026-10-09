import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, vi } from "vitest";
vi.hoisted(() => {
  const { mkdtempSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const dir = mkdtempSync(join(tmpdir(), "jaxos-childlog-home-"));
  const reposDir = mkdtempSync(join(tmpdir(), "jaxos-childlog-repos-"));
  // Pin ROOTS.repos (fixed when ./files is first imported, transitively by ./child-log) to a
  // test-owned temp dir. Left ambient it is `$HOME/repos` — nonexistent on a fresh runner, and
  // the owner's real repos dir here, which no test may read or create under.
  writeFileSync(join(dir, "settings.json"), JSON.stringify({ reposRoot: reposDir }), "utf8");
  process.env.JAXOS_HOME = dir;
});
import { childLogPath, lastStreamTextLine, readChildLogTail } from "./child-log";
import { ROOTS } from "./files";

describe("childLogPath (spec §8, cold review F1 — mirrors jaxflow_run.py's child_log_path exactly)", () => {
  it("<repo>/.local/runs/<run_id>/child.log", () => {
    const repo = join(ROOTS.repos, "jax-os");
    mkdirSync(repo, { recursive: true });
    expect(childLogPath(repo, "a1b2c3d4e5f6")).toBe(join(repo, ".local", "runs", "a1b2c3d4e5f6", "child.log"));
  });

  it("round 2 F1: a malformed run_id (traversal/length/case) or an out-of-allowlist repo (sibling escape or ..) returns null", () => {
    expect(childLogPath("/home/rafa/repos/jax-os", "../../etc/passwd")).toBeNull();
    expect(childLogPath("/home/rafa/repos/jax-os", "a1b2c3d4e5f")).toBeNull();
    expect(childLogPath("/home/rafa/repos/jax-os", "A1B2C3D4E5F6")).toBeNull();
    expect(childLogPath(mkdtempSync(join(tmpdir(), "jax-childlog-repo-")), "a1b2c3d4e5f6")).toBeNull();
    expect(childLogPath("/home/rafa/repos/jax-os/../../etc", "a1b2c3d4e5f6")).toBeNull();
  });
});

describe("readChildLogTail (spec §8 — last 64 KiB via seek-from-end, never a full-file read)", () => {
  it("returns the whole file when it fits", () => {
    const dir = mkdtempSync(join(tmpdir(), "jax-childlog-"));
    const path = join(dir, "child.log");
    writeFileSync(path, "line one\nline two\n");
    expect(readChildLogTail(path)).toBe("line one\nline two\n");
  });

  it("returns only the last 65536 bytes of a larger file", () => {
    const dir = mkdtempSync(join(tmpdir(), "jax-childlog-"));
    const path = join(dir, "child.log");
    const big = "a".repeat(70_000) + "TAIL_MARKER";
    writeFileSync(path, big);
    const tail = readChildLogTail(path);
    expect(tail).not.toBeNull();
    expect(tail!.length).toBe(65536);
    expect(tail!.endsWith("TAIL_MARKER")).toBe(true);
  });

  it("returns null for a missing file — best-effort, never throws", () => {
    expect(readChildLogTail("/nonexistent/path/child.log")).toBeNull();
  });

  it("round 2 F1: a symlinked child.log is rejected by lstat and never opened", () => {
    const dir = mkdtempSync(join(tmpdir(), "jax-childlog-"));
    writeFileSync(join(dir, "secret.log"), "leaked");
    symlinkSync(join(dir, "secret.log"), join(dir, "child.log"));
    expect(readChildLogTail(join(dir, "child.log"))).toBeNull();
  });
});

describe("lastStreamTextLine (spec §8, cold review F12, round 2 F5 — TS port of _last_stream_text_line)", () => {
  const textEvent = (text: string) => JSON.stringify({ type: "text", part: { text } });

  it("returns the first physical line of the LAST valid text event's part.text, bounded to 300 chars", () => {
    const stream = [textEvent("first event\nsecond line"), textEvent("winning event\nignored second line")].join("\n");
    expect(lastStreamTextLine(stream)).toBe("winning event");
  });

  it("the winning event need not be the last physical line — a trailing non-text event doesn't blank it", () => {
    const stream = [textEvent("the real winner"), JSON.stringify({ type: "tool_use", part: {} })].join("\n");
    expect(lastStreamTextLine(stream)).toBe("the real winner");
  });

  it("a malformed/incomplete LAST record never blanks an EARLIER valid text event (round 2 F5)", () => {
    const stream = [textEvent("earlier valid event"), "{not valid json"].join("\n");
    expect(lastStreamTextLine(stream)).toBe("earlier valid event");
  });

  it("no valid text event anywhere returns null", () => {
    expect(lastStreamTextLine("not json\n{}\n" + JSON.stringify({ type: "text", part: {} }))).toBeNull();
    expect(lastStreamTextLine("")).toBeNull();
  });

  it("bounds the returned line to 300 chars", () => {
    expect(lastStreamTextLine(textEvent("x".repeat(400)))).toBe("x".repeat(300));
  });
});
