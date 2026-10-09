import { execFileSync } from "node:child_process";
import { getProjects, type Project } from "./projects";
import { LOCAL_OPTS } from "../inputLimits";

export type TmuxSession = {
  name: string;
  createdEpoch: number;
  attached: boolean;
  windows: number;
  path: string;
};
export type Agent = {
  session: string;
  createdEpoch: number;
  attached: boolean;
  windows: number;
  path: string;
  project?: string;
  builder?: string;
};

// ponytail: greedy-front regex anchored on the rightmost numeric triple — a
// path containing "|<int>|<int>|<int>|" would mis-split; delimiter collision
// is unavoidable when both ends can contain "|". Pathological for fs paths.
export function parseTmuxLs(out: string): TmuxSession[] {
  return out
    .split(/\r?\n/)
    .map((l) => l.match(/^(.*)\|(\d+)\|(\d+)\|(\d+)\|(.*)$/))
    .filter((m): m is RegExpMatchArray => m !== null)
    .map((m) => ({
      name: m[1],
      createdEpoch: Number(m[2]),
      attached: m[3] !== "0",
      windows: Number(m[4]),
      path: m[5],
    }));
}

export function joinAgents(sessions: TmuxSession[], projects: Project[]): Agent[] {
  return sessions.map((s) => {
    const p = projects.find((proj) => proj.tmux === s.name);
    return {
      session: s.name,
      createdEpoch: s.createdEpoch,
      attached: s.attached,
      windows: s.windows,
      path: s.path,
      project: p?.name,
      builder: p?.builder,
    };
  });
}

// tmux exits 1 both when the server is down and when it has no sessions
// (exit-empty default kills the empty server) — Rafa's call: that reads as
// "zero agents" (quiet empty), not source-unavailable. Real failures
// (missing binary, socket permissions) still throw -> {ok:false} warning.
export function isNoServerError(stderr: string): boolean {
  return /no server running|error connecting/i.test(stderr);
}

// Untested shell glue. No tmux server (down or empty) -> [] (quiet empty);
// real failures throw and the API route maps that to {ok:false} warning.
export type Utf8ExecSync = (
  file: string,
  args: readonly string[],
  options: { encoding: "utf8"; timeout: number; maxBuffer: number },
) => string;

export function getAgents(exec: Utf8ExecSync = execFileSync as Utf8ExecSync): Agent[] {
  let out: string;
  try {
    out = exec(
      "tmux",
      ["ls", "-F", "#{session_name}|#{session_created}|#{session_attached}|#{session_windows}|#{session_path}"],
      LOCAL_OPTS,
    );
  } catch (e) {
    const stderr = String((e as { stderr?: unknown }).stderr ?? "");
    if (isNoServerError(stderr)) return [];
    throw e;
  }
  let projects: Project[] = [];
  try {
    projects = getProjects().projects;
  } catch {
    // projects scan failing must not hide live sessions — list them unmanaged
  }
  return joinAgents(parseTmuxLs(out), projects);
}
