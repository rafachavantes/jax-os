import { describe, expect, it } from "vitest";
import { gitStatusForDir, parseGitStatusPorcelain } from "./gitStatus";
import type { GitProbe } from "./projects";

describe("parseGitStatusPorcelain", () => {
  it("classifies each porcelain code into the 4-bucket status, keyed by the immediate path", () => {
    const map = parseGitStatusPorcelain(
      [" M modified.ts", "A  added.ts", " D deleted.ts", "?? untracked.ts", "R  old.ts -> renamed.ts"].join("\n"),
    );
    expect(map.get("modified.ts")).toBe("modified");
    expect(map.get("added.ts")).toBe("added");
    expect(map.get("deleted.ts")).toBe("deleted");
    expect(map.get("untracked.ts")).toBe("untracked");
    expect(map.get("renamed.ts")).toBe("modified");
    expect(map.has("old.ts")).toBe(false);
  });

  it("reports a wholly untracked directory as a single trailing-slash entry, trimmed", () => {
    expect(parseGitStatusPorcelain("?? newdir/\n").get("newdir")).toBe("untracked");
  });

  it("tints the immediate child that CONTAINS a nested change, by its first path segment", () => {
    const map = parseGitStatusPorcelain(" M sub/inner/deep.ts\n");
    expect(map.get("sub")).toBe("modified");
    expect(map.has("sub/inner/deep.ts")).toBe(false);
  });

  it("keeps the highest-priority status when a directory aggregates mixed changes", () => {
    const map = parseGitStatusPorcelain(["?? sub/new.ts", " M sub/changed.ts"].join("\n"));
    expect(map.get("sub")).toBe("modified"); // modified outranks untracked
  });

  it("ignores blank output", () => {
    expect(parseGitStatusPorcelain("").size).toBe(0);
    expect(parseGitStatusPorcelain("\n\n").size).toBe(0);
  });

  it("keeps a literal ' -> ' in a modified file's own name intact, parsing the arrow only for rename/copy codes (F3)", () => {
    const map = parseGitStatusPorcelain(" M old -> new.ts\n");
    expect(map.get("old -> new.ts")).toBe("modified");
    expect(map.has("new.ts")).toBe(false);
  });
});

describe("gitStatusForDir", () => {
  const probeReturning = (stdout: string): GitProbe => async () => ({ ok: true, stdout });

  it("resolves modified/added/untracked from a successful probe", async () => {
    const map = await gitStatusForDir("/repo", probeReturning([" M a.ts", "A  b.ts", "?? c.ts"].join("\n")));
    expect(map?.get("a.ts")).toBe("modified");
    expect(map?.get("b.ts")).toBe("added");
    expect(map?.get("c.ts")).toBe("untracked");
  });

  it("returns an empty (not null) map for a clean repo with no changes", async () => {
    const map = await gitStatusForDir("/repo", probeReturning(""));
    expect(map).not.toBeNull();
    expect(map?.size).toBe(0);
  });

  it("degrades to null when the probe reports a failed exit (e.g. non-repo)", async () => {
    const map = await gitStatusForDir("/not-a-repo", async () => ({ ok: false, stdout: "fatal: not a git repository" }));
    expect(map).toBeNull();
  });

  it("degrades to null when the probe itself returns null (timeout/exec failure)", async () => {
    expect(await gitStatusForDir("/repo", async () => null)).toBeNull();
  });

  it("calls the probe with the exact porcelain args and the listed directory as cwd (assert the wiring, not just the effect)", async () => {
    let seenArgs: string[] = [];
    let seenCwd = "";
    const probe: GitProbe = async (args, cwd) => {
      seenArgs = args;
      seenCwd = cwd;
      return { ok: true, stdout: "" };
    };
    await gitStatusForDir("/some/dir", probe);
    expect(seenArgs).toEqual(["status", "--porcelain=v1", "--untracked-files=normal", "."]);
    expect(seenCwd).toBe("/some/dir");
  });

  it("degrades to null when the probe itself rejects, not just when it resolves to a failure (cold review round 1 F2)", async () => {
    const map = await gitStatusForDir("/repo", async () => {
      throw new Error("boom");
    });
    expect(map).toBeNull();
  });
});
