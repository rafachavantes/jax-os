import { describe, expect, it, vi } from "vitest";
import { getActivity, mergeActivity, parseGitLog } from "./activity";

const US = "\x1f";

describe("parseGitLog", () => {
  it("parses \\x1f-delimited lines", () => {
    const out = `abc123${US}Rafa${US}feat: add thing${US}1751800000\n`;
    expect(parseGitLog("jax-os", out)).toEqual([
      { repo: "jax-os", hash: "abc123", author: "Rafa", message: "feat: add thing", epoch: 1751800000 },
    ]);
  });

  it("keeps pipes in commit subjects (the reason %x1f is the delimiter)", () => {
    const out = `abc${US}Rafa${US}fix: a | b || c${US}1${'\n'}`;
    expect(parseGitLog("r", out)[0].message).toBe("fix: a | b || c");
  });

  it("drops malformed lines and empty output", () => {
    expect(parseGitLog("r", "")).toEqual([]);
    expect(parseGitLog("r", "garbage line\n")).toEqual([]);
  });
});

describe("mergeActivity", () => {
  it("merges, sorts desc by epoch, caps at 10", () => {
    const a = parseGitLog("a", Array.from({ length: 8 }, (_, i) => `h${i}${US}x${US}m${US}${100 + i}`).join("\n"));
    const b = parseGitLog("b", Array.from({ length: 8 }, (_, i) => `h${i}${US}x${US}m${US}${200 + i}`).join("\n"));
    const merged = mergeActivity([a, b]);
    expect(merged).toHaveLength(10);
    expect(merged[0]).toMatchObject({ repo: "b", epoch: 207 });
    expect(merged.map((c) => c.epoch)).toEqual([...merged.map((c) => c.epoch)].sort((x, y) => y - x));
  });
});

const LOCAL_OPTS = { encoding: "utf8" as const, timeout: 5_000, maxBuffer: 1024 * 1024 };
const GIT_LOG = ["-C", "/tmp/repos/r", "log", "-n", "5", "--format=%h%x1f%an%x1f%s%x1f%ct"];
const PARTIAL_LOG = `abc123\x1fRafa\x1ffeat: leaked\x1f1751800000\n`;

describe("getActivity executor bounds", () => {
  it("treats timeout as source failure, not parsed commits", () => {
    const exec = vi.fn(() => {
      throw Object.assign(new Error("timeout"), { code: "ETIMEDOUT", stdout: PARTIAL_LOG });
    });
    expect(() => getActivity([{ dir: "r" }], "/tmp/repos", exec)).toThrow();
    expect(exec).toHaveBeenCalledWith("git", GIT_LOG, LOCAL_OPTS);
  });

  it("treats output overflow as source failure, not partial parsed success", () => {
    const exec = vi.fn(() => {
      throw Object.assign(new Error("overflow"), {
        code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
        stdout: PARTIAL_LOG,
      });
    });
    expect(() => getActivity([{ dir: "r" }], "/tmp/repos", exec)).toThrow();
  });

  it("still skips a non-limit git failure for that repo", () => {
    const exec = vi.fn(() => {
      throw Object.assign(new Error("not a git repo"), { code: 128 });
    });
    expect(getActivity([{ dir: "r" }], "/tmp/repos", exec)).toEqual([]);
  });
});
