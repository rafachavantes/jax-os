// Shared fixtures for the multi-session card tests (spec 2026-11-12, part 2). Not a test file:
// vitest's `--dir src` collects `*.test.ts(x)` only, so this module ships with the suite.
import type { Envelope } from "./api";
import { buildMission, type Mission } from "./mission";
import type { Project } from "../server/collectors/projects";
import type { PrResult } from "../server/collectors/github";
import type { WorktreeResult } from "../server/collectors/worktrees";
import type {
  HubActiveRunSummary, HubEnvelopeData, HubPane, HubProject, HubScopedActivity, HubSessionSeat, HubSessionSplit,
} from "../server/db/workflows";

export const NOW = Date.parse("2026-11-12T12:00:00-03:00");

// A 36-char session id from a short seed (the real ids are UUID-shaped 36-char strings).
export function U(c: string): string {
  return c.padEnd(36, "0").slice(0, 36);
}

export function mkProject(over: Partial<Project> & { dir: string }): Project {
  return {
    dir: over.dir,
    ...(over.path ? { path: over.path } : {}),
    name: over.name ?? over.dir,
    stage: over.stage ?? "build",
    gate: over.gate ?? null,
    legacyGateField: over.legacyGateField ?? false,
    builder: over.builder ?? "claude-code",
    branch: over.branch ?? "main",
    tmux: over.tmux,
    flag: over.flag,
    updated: over.updated ?? "2026-11-10T10:00:00-03:00",
    now: over.now ?? "",
    residuals: over.residuals ?? [],
    statusMtime: over.statusMtime ?? "2026-11-10T10:00:00-03:00",
    archived: over.archived ?? false,
  };
}

export function mkPane(over: Partial<HubPane> & { pane: string }): HubPane {
  return {
    pane: over.pane,
    tmuxIncarnation: over.tmuxIncarnation ?? "100:1000",
    session: "session" in over ? over.session ?? null : "s1",
    role: over.role ?? "lead",
    live: over.live ?? true,
    working: over.working ?? false,
    capsule: over.capsule ?? null,
    pendingQuestion: over.pendingQuestion ?? null,
    lastEventTs: over.lastEventTs ?? new Date(NOW - 1_000).toISOString(),
    subagentCount: over.subagentCount ?? 0,
    runtime: over.runtime ?? "claude",
    ...("harnessSession" in over ? { harnessSession: over.harnessSession } : {}),
    ...("capsuleSession" in over ? { capsuleSession: over.capsuleSession } : {}),
    ...("questionSession" in over ? { questionSession: over.questionSession } : {}),
    ...("questionAnswerable" in over ? { questionAnswerable: over.questionAnswerable } : {}),
  };
}

export function mkActivity(over: Partial<HubScopedActivity> = {}): HubScopedActivity {
  return {
    timeline: [], activeRun: null, activeRuns: [], lastRun: null,
    heartbeat: new Array(60).fill(0), lastAction: null, newestEventTs: null, status: null,
    ...over,
  };
}

export function mkSeat(over: Partial<HubSessionSeat> & { sessionId: string | null }): HubSessionSeat {
  return {
    sessionId: over.sessionId,
    transport: over.transport ?? "tmux",
    panes: over.panes ?? [],
    name: over.name ?? null,
    activity: over.activity ?? mkActivity(),
  };
}

export function mkSplit(over: Partial<HubSessionSplit> = {}): HubSessionSplit {
  return {
    seats: over.seats ?? [],
    unlinked: over.unlinked ?? mkActivity(),
    unownedLastRun: over.unownedLastRun ?? null,
  };
}

export function mkHubProject(over: Partial<HubProject> = {}): HubProject {
  return {
    panes: over.panes ?? [],
    codexSessions: over.codexSessions ?? [],
    codexUntracked: over.codexUntracked ?? false,
    pendingQuestion: over.pendingQuestion ?? null,
    freshnessCount: over.freshnessCount ?? 0,
    newestEventTs: over.newestEventTs ?? null,
    timeline: over.timeline ?? [],
    activeRun: over.activeRun,
    activeRuns: over.activeRuns ?? [],
    lastRun: over.lastRun ?? null,
    loopSummary: over.loopSummary,
    heartbeat: over.heartbeat ?? new Array(60).fill(0),
    lastAction: over.lastAction ?? null,
    ...(over.split ? { split: over.split } : {}),
  };
}

export function runSummary(runId: string, over: Partial<HubActiveRunSummary> = {}): HubActiveRunSummary {
  return {
    runId,
    role: over.role ?? "builder",
    kind: over.kind ?? "build",
    runtime: over.runtime ?? "claude",
    target: over.target ?? null,
    startedAt: over.startedAt ?? new Date(NOW - 5 * 60_000).toISOString(),
  };
}

export function missionOf(
  hp: Record<string, HubProject>,
  opts: {
    projects?: Project[];
    worktrees?: Record<string, WorktreeResult>;
    prs?: Record<string, PrResult>;
    codexSource?: "ok" | "failed" | "truncated" | "absent";
    hubOk?: boolean;
  } = {},
): Mission {
  const projects = opts.projects ?? Object.keys(hp).map((dir) => mkProject({ dir }));
  const hub: Envelope<HubEnvelopeData> | undefined = opts.hubOk === false
    ? { ok: false, error: "hub failed" }
    : {
        ok: true,
        data: {
          byProject: hp, historicalTruncated: false, codexSource: opts.codexSource ?? "ok", tmuxSource: "ok",
          prefs: {}, settings: { archiveAfterDays: 14 },
        },
      };
  const prs: Envelope<Record<string, PrResult>> | undefined = opts.prs ? { ok: true, data: opts.prs } : undefined;
  const worktrees: Envelope<Record<string, WorktreeResult>> | undefined = opts.worktrees ? { ok: true, data: opts.worktrees } : undefined;
  return buildMission({ ok: true, data: { projects, skipped: 0, reposRoot: "" } }, hub, prs, NOW, worktrees);
}
