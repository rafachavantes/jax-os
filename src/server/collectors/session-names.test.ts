import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdirSync, mkdtempSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseClaudeSessionRecord, parseCodexSessionIndex, readSessionNames, sanitizeSessionName } from "./session-names";

let home: string;
let warn: ReturnType<typeof vi.spyOn>;
let error: ReturnType<typeof vi.spyOn>;

beforeEach(() => {
  home = mkdtempSync(join(tmpdir(), "jaxos-names-"));
  warn = vi.spyOn(console, "warn").mockImplementation(() => {});
  error = vi.spyOn(console, "error").mockImplementation(() => {});
});

afterEach(() => {
  expect(warn).not.toHaveBeenCalled();
  expect(error).not.toHaveBeenCalled();
  warn.mockRestore();
  error.mockRestore();
  rmSync(home, { recursive: true, force: true });
});

function seedClaude(): string {
  const dir = join(home, ".claude", "sessions");
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, "100.json"), JSON.stringify({ sessionId: "S1", name: "jax-os-1f", nameSource: "derived", pid: 9 }), "utf8");
  writeFileSync(join(dir, "100.key"), JSON.stringify({ sessionId: "KEYED", name: "secret" }), "utf8");
  writeFileSync(join(dir, "notes.txt"), JSON.stringify({ sessionId: "TXT", name: "notes" }), "utf8");
  writeFileSync(join(dir, "bad.json"), "{", "utf8");
  mkdirSync(join(dir, "x.json"));
  symlinkSync(join(dir, "100.key"), join(dir, "linked.json"));
  return dir;
}

function seedCodex(): void {
  mkdirSync(join(home, ".codex"), { recursive: true });
  writeFileSync(
    join(home, ".codex", "session_index.jsonl"),
    '{"id":"T1","thread_name":"codex-one","updated_at":"2026-11-02T00:00:00Z"}\n',
    "utf8",
  );
}

describe("sanitizeSessionName", () => {
  it("trims, strips control characters and caps at 80 code points", () => {
    expect(sanitizeSessionName("  multi-sessao \n")).toBe("multi-sessao");
    expect(sanitizeSessionName("a\u0000b\u001b[31mc\u007f")).not.toMatch(/[\x00-\x1f\x7f]/);
    expect(sanitizeSessionName("x".repeat(500))).toHaveLength(80);
  });

  it("rejects non-strings and empty results", () => {
    for (const bad of ["", "   ", 42, null, {}]) expect(sanitizeSessionName(bad)).toBeNull();
  });
});

describe("parseClaudeSessionRecord", () => {
  it("reads exactly sessionId and name, ignoring every other field", () => {
    expect(parseClaudeSessionRecord('{"sessionId":"S1","name":"multi-sessao","nameSource":"user","pid":9,"cwd":"/x"}'))
      .toEqual({ sessionId: "S1", name: "multi-sessao" });
  });

  it("rejects malformed and unusable records", () => {
    for (const bad of [
      "not json",
      "[]",
      "null",
      '{"sessionId":"S1"}',
      '{"sessionId":7,"name":"n"}',
      '{"sessionId":"S1","name":"  "}',
    ]) expect(parseClaudeSessionRecord(bad)).toBeNull();
  });
});

describe("parseCodexSessionIndex", () => {
  it("keeps the last valid line per id and tolerates garbage lines", () => {
    const text = [
      '{"id":"T1","thread_name":"old","updated_at":"2026-11-01T00:00:00Z"}',
      "garbage",
      '{"id":"T1","thread_name":"new","updated_at":"2026-11-02T00:00:00Z"}',
      '{"id":"T2","thread_name":"other","updated_at":"2026-11-02T00:00:00Z"}',
      "",
    ].join("\n");
    expect(parseCodexSessionIndex(text)).toEqual(new Map([["T1", "new"], ["T2", "other"]]));
  });

  it("an unusable last name removes the id", () => {
    expect(parseCodexSessionIndex('{"id":"T3","thread_name":"x"}\n{"id":"T3","thread_name":""}\n').has("T3")).toBe(false);
  });

  it("returns an empty map for empty text and skips valid JSON that is not an object", () => {
    expect(parseCodexSessionIndex("")).toEqual(new Map());
    expect(parseCodexSessionIndex('"just a string"\n[]\nnull\n').size).toBe(0);
  });
});

describe("readSessionNames", () => {
  it("reads .json files and the Codex index, never .key/.txt/symlinks/directories", () => {
    seedClaude();
    seedCodex();
    const names = readSessionNames(home);
    expect(names.get("S1")).toBe("jax-os-1f");
    expect(names.get("T1")).toBe("codex-one");
    expect(names.has("KEYED")).toBe(false);
    expect(names.has("TXT")).toBe(false);
  });

  it("also reads the registries of claude-switch accounts (sessions run with their CLAUDE_CONFIG_DIR)", () => {
    const dir = join(home, ".claude-switch", "accounts", "work", "sessions");
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "42.json"), JSON.stringify({ sessionId: "W1", name: "jev-recipes" }));
    writeFileSync(join(dir, "42.abc.key"), JSON.stringify({ sessionId: "WKEY", name: "secret" }));
    const names = readSessionNames(home);
    expect(names.get("W1")).toBe("jev-recipes");
    expect(names.has("WKEY")).toBe(false);
  });

  it("missing HOME registries produce an empty map, never an error", () => {
    expect(readSessionNames(join(home, "empty"))).toEqual(new Map());
  });

  it("the Codex source works with no Claude registry at all", () => {
    seedCodex();
    expect(readSessionNames(home)).toEqual(new Map([["T1", "codex-one"]]));
  });

  it("the Claude source works when the Codex index is absent", () => {
    seedClaude();
    const names = readSessionNames(home);
    expect(names.get("S1")).toBe("jax-os-1f");
  });

  it("the Claude source works when the Codex index path is a directory", () => {
    seedClaude();
    mkdirSync(join(home, ".codex", "session_index.jsonl"), { recursive: true });
    const names = readSessionNames(home);
    expect(names.get("S1")).toBe("jax-os-1f");
  });
});
