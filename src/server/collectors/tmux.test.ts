import { describe, expect, it, vi } from "vitest";
import { getAgents, isNoServerError, joinAgents, parseTmuxLs } from "./tmux";
import type { Project } from "./projects";

const project = (over: Partial<Project>): Project => ({
  dir: "x",
  name: "X",
  stage: "build",
  gate: null,
  legacyGateField: false,
  builder: "claude-code",
  branch: "main",
  updated: "",
  now: "",
  residuals: [],
  statusMtime: "",
  archived: false,
  ...over,
});

describe("parseTmuxLs", () => {
  it("parses the 5-field -F output (verified on tmux 3.4)", () => {
    const out =
      "Acme-AI|1783254396|0|2|/home/rafa/repos/Acme.AI\nPIJ|1783254396|1|1|/home/rafa/repos/PIJ\n";
    expect(parseTmuxLs(out)).toEqual([
      { name: "Acme-AI", createdEpoch: 1783254396, attached: false, windows: 2, path: "/home/rafa/repos/Acme.AI" },
      { name: "PIJ", createdEpoch: 1783254396, attached: true, windows: 1, path: "/home/rafa/repos/PIJ" },
    ]);
  });

  it("keeps pipes inside session names", () => {
    expect(parseTmuxLs("weird|name|123|2|1|/tmp")).toEqual([
      { name: "weird|name", createdEpoch: 123, attached: true, windows: 1, path: "/tmp" },
    ]);
  });

  it("keeps pipes inside paths", () => {
    expect(parseTmuxLs("s1|123|0|1|/tmp/odd|dir")).toEqual([
      { name: "s1", createdEpoch: 123, attached: false, windows: 1, path: "/tmp/odd|dir" },
    ]);
  });

  it("returns [] on empty output", () => {
    expect(parseTmuxLs("")).toEqual([]);
  });
});

describe("joinAgents", () => {
  it("links sessions to projects via the tmux field, leaves others unmanaged", () => {
    const sessions = parseTmuxLs("apg-build|100|1|2|/home/rafa/repos/apg\nrandom|200|0|1|/tmp");
    const agents = joinAgents(sessions, [
      project({ dir: "apg", name: "APG Platform", tmux: "apg-build" }),
    ]);
    expect(agents).toEqual([
      { session: "apg-build", createdEpoch: 100, attached: true, windows: 2, path: "/home/rafa/repos/apg", project: "APG Platform", builder: "claude-code" },
      { session: "random", createdEpoch: 200, attached: false, windows: 1, path: "/tmp", project: undefined, builder: undefined },
    ]);
  });
});

describe("isNoServerError", () => {
  it("matches both tmux no-server stderr variants", () => {
    expect(isNoServerError("no server running on /tmp/tmux-1000/default")).toBe(true);
    expect(isNoServerError("error connecting to /tmp/tmux-1000/default (No such file or directory)")).toBe(true);
  });

  it("does not match real failures", () => {
    expect(isNoServerError("permission denied")).toBe(false);
    expect(isNoServerError("")).toBe(false);
  });
});

const LOCAL_OPTS = { encoding: "utf8" as const, timeout: 5_000, maxBuffer: 1024 * 1024 };
const TMUX_LS = ["ls", "-F", "#{session_name}|#{session_created}|#{session_attached}|#{session_windows}|#{session_path}"];

describe("getAgents executor bounds", () => {
  it("keeps no-tmux-server as quiet empty", () => {
    const exec = vi.fn(() => {
      throw Object.assign(new Error("tmux"), { stderr: "no server running on /tmp/tmux-1000/default" });
    });
    expect(getAgents(exec)).toEqual([]);
    expect(exec).toHaveBeenCalledWith("tmux", TMUX_LS, LOCAL_OPTS);
  });

  it("treats timeout as source failure, not parsed sessions", () => {
    const exec = vi.fn(() => {
      throw Object.assign(new Error("timeout"), {
        code: "ETIMEDOUT",
        stdout: "s1|123|0|1|/tmp\n",
      });
    });
    expect(() => getAgents(exec)).toThrow();
  });

  it("treats output overflow as source failure, not partial parsed success", () => {
    const exec = vi.fn(() => {
      throw Object.assign(new Error("overflow"), {
        code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
        stdout: "s1|123|0|1|/tmp\n",
      });
    });
    expect(() => getAgents(exec)).toThrow();
  });
});
