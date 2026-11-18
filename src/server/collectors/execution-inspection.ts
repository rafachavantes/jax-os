import type Database from "better-sqlite3";
import { getCurrentRunStates } from "../db/workflows";
import {
  InspectionBudget, artifactKey, inspectArtifactsDetailed,
  type ArtifactDeps, type ArtifactLink, type ExecutionArtifactFacts,
} from "./execution-artifacts";
import { inspectWorkers, workerKey, type ProcReader, type WorkerDeps, type WorkerObservation, type WorkerTarget } from "./execution-worker";
import type { ExecutionOutcome } from "../../lib/executions";
import type { ContractStatus } from "../../lib/workflow";
import type { GitProbe } from "./projects";

// M3a, Task 4 boundary — composes re-resolved durable selectors, bounded artifacts and the
// current wrapper observation under ONE inspection budget. Every source settles independently:
// a hung filesystem/manifest/process source never erases another row's known facts.

export type InspectionSelector = {
  destination: string;
  runId: string;
  startEventId: number | null;
};

export type InspectionCurrent = {
  outcome: ExecutionOutcome;
  terminal: boolean;
  finishEventId: number | null;
  finishedAt: string | null;
  contractStatus: ContractStatus | null;
  durationMs: number | null;
};

export type InspectionRow = {
  destination: string;
  runId: string;
  branch: string | null;
  branchSource: "manifest" | null;
  report: ArtifactLink | null;
  tests: ArtifactLink | null;
  observation: WorkerObservation;
  refreshNeeded: boolean;
  current: InspectionCurrent | null;
  warnings: string[];
};

export type InspectionDeps = {
  now?: () => number;
  budget?: InspectionBudget;
  reposRoot?: string;
  gitProbe?: GitProbe;
  discoveryMs?: number;
  fsMs?: number;
  gitMs?: number;
  tmux?: () => Promise<string>;
  proc?: ProcReader;
  realpath?: (path: string) => Promise<string>;
  perReadMs?: number;
  lastGood?: Map<string, { at: number; observedAt: string }>;
};

function emptyFacts(destination: string, runId: string): ExecutionArtifactFacts {
  return { destination, runId, branch: null, branchSource: null, report: null, tests: null, warnings: ["root-unavailable"] };
}

export async function inspectExecutionRows(
  db: Database.Database,
  selectors: readonly InspectionSelector[],
  deps: InspectionDeps = {},
): Promise<InspectionRow[]> {
  const now = deps.now ?? Date.now;
  const budget = deps.budget ?? new InspectionBudget(4, 2_000, now);
  const bounded = selectors.slice(0, 20);
  const states = getCurrentRunStates(db, bounded);

  const resolved = states.filter((s) => s.row !== null).map((s) => ({ destination: s.destination, runId: s.runId }));
  const artifactDeps: ArtifactDeps = {
    reposRoot: deps.reposRoot,
    now,
    budget,
    gitProbe: deps.gitProbe,
    discoveryMs: deps.discoveryMs,
    fsMs: deps.fsMs,
    gitMs: deps.gitMs,
  };
  const artifacts = await inspectArtifactsDetailed(resolved, artifactDeps).catch(() => ({
    facts: resolved.map((r) => emptyFacts(r.destination, r.runId)),
    internal: new Map(resolved.map((r) => [artifactKey(r.destination, r.runId), { manifestPath: null, repo: null }])),
  }));
  const facts = new Map(artifacts.facts.map((f) => [artifactKey(f.destination, f.runId), f]));

  const targets: WorkerTarget[] = [];
  for (const state of states) {
    if (state.row === null || state.row.terminal) continue;
    const internal = artifacts.internal.get(artifactKey(state.destination, state.runId));
    targets.push({
      destination: state.destination,
      runId: state.runId,
      startEventId: state.startEventId,
      startTs: state.attemptStartedAt,
      session: state.attemptSession,
      manifestPath: internal?.manifestPath ?? null,
      repo: internal?.repo ?? null,
    });
  }
  const workerDeps: WorkerDeps = {
    tmux: deps.tmux,
    proc: deps.proc,
    realpath: deps.realpath,
    now,
    budget,
    perReadMs: deps.perReadMs,
    cache: deps.lastGood,
  };
  const observations = await inspectWorkers(targets, workerDeps).catch(() =>
    new Map<string, WorkerObservation>(targets.map((t) => [workerKey(t), {
      status: "unknown", warning: "observation-timeout",
    }])));

  return states.map((state, index) => {
    const selector = bounded[index];
    const fact = facts.get(artifactKey(state.destination, state.runId)) ?? emptyFacts(state.destination, state.runId);
    const live = state.row;
    const refreshNeeded = selector.startEventId !== null && live !== null && state.startEventId !== selector.startEventId;
    const target = targets.find((t) => t.destination === state.destination && t.runId === state.runId);
    const observation = live === null || live.terminal || refreshNeeded
      ? { status: "not-checked" as const }
      : target
        ? observations.get(workerKey(target)) ?? { status: "unknown" as const, warning: "observation-timeout" as const }
        : { status: "not-checked" as const };
    return {
      destination: state.destination,
      runId: state.runId,
      branch: fact.branch,
      branchSource: fact.branchSource,
      report: fact.report,
      tests: fact.tests,
      observation,
      refreshNeeded,
      current: live === null ? null : {
        outcome: live.outcome,
        terminal: live.terminal,
        finishEventId: live.finishEventId,
        finishedAt: live.finishedAt,
        contractStatus: live.contractStatus,
        durationMs: live.durationMs,
      },
      warnings: [...fact.warnings, ...(observation.warning ? [observation.warning] : [])],
    };
  });
}
