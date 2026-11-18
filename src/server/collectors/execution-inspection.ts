import type Database from "better-sqlite3";
import { getCurrentRunStates, type CurrentExecutionState } from "../db/workflows";
import {
  InspectionBudget, artifactKey, inspectArtifactsDetailed,
  type ArtifactDeps, type ArtifactLink, type ExecutionArtifactFacts,
} from "./execution-artifacts";
import { inspectWorkers, workerKey, type ProcReader, type WorkerDeps, type WorkerObservation, type WorkerTarget } from "./execution-worker";
import type { ExecutionOutcome, ExecutionWarning } from "../../lib/executions";
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
  // Bounded existing warning codes only, so a fresh finish's contract/conflict warning reaches
  // the client without exposing raw payload or a new public field.
  warnings: ExecutionWarning[];
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
  workerMs?: number;
  launcherPath?: string;
  lastGood?: Map<string, { at: number; observedAt: string }>;
};

function emptyFacts(destination: string, runId: string): ExecutionArtifactFacts {
  return { destination, runId, branch: null, branchSource: null, report: null, tests: null, warnings: ["root-unavailable"] };
}

function stateKey(destination: string, runId: string): string {
  return `${destination}/${runId}`;
}

function currentOf(state: CurrentExecutionState | null): InspectionCurrent | null {
  const live = state?.row ?? null;
  if (live === null) return null;
  return {
    outcome: live.outcome,
    terminal: live.terminal,
    finishEventId: live.finishEventId,
    finishedAt: live.finishedAt,
    contractStatus: live.contractStatus,
    durationMs: live.durationMs,
    warnings: live.warnings,
  };
}

// Nullable start IDs are compared too: a new start after an orphan/null-start row is a changed
// attempt, not an exemption.
function refreshFor(selector: InspectionSelector, state: CurrentExecutionState | null): boolean {
  const live = state?.row ?? null;
  return live !== null && (state?.startEventId ?? null) !== selector.startEventId;
}

// The bounded DB-only answer for a differing concurrent selector set: durable current facts are
// always available; only native-backed observation/artifact fields are not checked/unavailable.
function databaseOnlyRows(db: Database.Database, selectors: readonly InspectionSelector[]): InspectionRow[] {
  const states = getCurrentRunStates(db, selectors);
  const byRun = new Map(states.map((state) => [stateKey(state.destination, state.runId), state]));
  return selectors.map((selector) => {
    const state = byRun.get(stateKey(selector.destination, selector.runId)) ?? null;
    return {
      destination: selector.destination,
      runId: selector.runId,
      branch: null,
      branchSource: null,
      report: null,
      tests: null,
      observation: { status: "not-checked" },
      refreshNeeded: refreshFor(selector, state),
      current: currentOf(state),
      warnings: ["root-unavailable"],
    };
  });
}

async function runPass(
  db: Database.Database,
  selectors: readonly InspectionSelector[],
  deps: InspectionDeps,
  budget: InspectionBudget,
): Promise<InspectionRow[]> {
  const now = deps.now ?? Date.now;
  const before = getCurrentRunStates(db, selectors);

  const resolved = before.filter((state) => state.row !== null)
    .map((state) => ({ destination: state.destination, runId: state.runId }));
  const artifactDeps: ArtifactDeps = {
    reposRoot: deps.reposRoot,
    now,
    budget,
    gitProbe: deps.gitProbe,
    discoveryMs: deps.discoveryMs,
    fsMs: deps.fsMs,
    gitMs: deps.gitMs,
    realpath: deps.realpath,
  };
  const artifacts = resolved.length === 0
    ? { facts: [] as ExecutionArtifactFacts[], internal: new Map<string, { manifestPath: string | null; repo: string | null }>() }
    : await inspectArtifactsDetailed(resolved, artifactDeps).catch(() => ({
        facts: resolved.map((row) => emptyFacts(row.destination, row.runId)),
        internal: new Map(resolved.map((row) => [artifactKey(row.destination, row.runId), { manifestPath: null, repo: null }])),
      }));
  const facts = new Map(artifacts.facts.map((fact) => [artifactKey(fact.destination, fact.runId), fact]));

  // The process snapshot is taken BEFORE the fresh terminal re-check, exactly as the spec's
  // lifecycle table requires.
  const targets: WorkerTarget[] = [];
  for (const state of before) {
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
    workerMs: deps.workerMs,
    launcherPath: deps.launcherPath,
    cache: deps.lastGood,
  };
  const observations = targets.length === 0
    ? new Map<string, WorkerObservation>()
    : await inspectWorkers(targets, workerDeps).catch(() =>
        new Map<string, WorkerObservation>(targets.map((target) => [workerKey(target), {
          status: "unknown", warning: "observation-timeout",
        }])));

  // Fresh durable re-check AFTER the process snapshot: a finish inserted during the snapshot wins
  // over awaiting/absent/observed, and a newer start supersedes the displayed attempt.
  const after = getCurrentRunStates(db, selectors);
  const liveByRun = new Map(after.map((state) => [stateKey(state.destination, state.runId), state]));

  return selectors.map((selector) => {
    const state = liveByRun.get(stateKey(selector.destination, selector.runId)) ?? null;
    const live = state?.row ?? null;
    const refreshNeeded = refreshFor(selector, state);
    const fact = facts.get(artifactKey(selector.destination, selector.runId))
      ?? emptyFacts(selector.destination, selector.runId);
    const target = targets.find((item) =>
      item.destination === selector.destination && item.runId === selector.runId);
    const observation = live === null || live.terminal || refreshNeeded
      ? { status: "not-checked" as const }
      : target
        ? observations.get(workerKey(target)) ?? { status: "unknown" as const, warning: "observation-timeout" as const }
        : { status: "not-checked" as const };
    return {
      destination: selector.destination,
      runId: selector.runId,
      branch: fact.branch,
      branchSource: fact.branchSource,
      report: fact.report,
      tests: fact.tests,
      observation,
      refreshNeeded,
      current: currentOf(state),
      warnings: [...fact.warnings, ...(observation.warning ? [observation.warning] : [])],
    };
  });
}

type InspectionPass = {
  key: string;
  response: Promise<InspectionRow[]>;
  settled: Promise<void>;
  responded: boolean;
};

// Smallest bounded state: ONE active native pass per DB/collector context. It is cleared only
// after the pass response AND every started native operation have settled, so a hung source can
// never be re-entered by the next poll.
const defaultPasses = new WeakMap<object, InspectionPass>();
const injectedPasses = new WeakMap<object, WeakMap<object, InspectionPass>>();

function passStore(db: object, deps: InspectionDeps): WeakMap<object, InspectionPass> {
  if (Object.keys(deps).length === 0) return defaultPasses;
  let store = injectedPasses.get(deps);
  if (!store) {
    store = new WeakMap<object, InspectionPass>();
    injectedPasses.set(deps, store);
  }
  return store;
}

function selectorKey(selectors: readonly InspectionSelector[]): string {
  return JSON.stringify(selectors.map((selector) => [selector.destination, selector.runId, selector.startEventId]));
}

export async function inspectExecutionRows(
  db: Database.Database,
  selectors: readonly InspectionSelector[],
  deps: InspectionDeps = {},
): Promise<InspectionRow[]> {
  const bounded = selectors.slice(0, 20);
  if (bounded.length === 0) return [];
  const key = selectorKey(bounded);
  const store = passStore(db, deps);
  const active = store.get(db);
  if (active) {
    // Identical selectors share an UNRESOLVED pass; once its public response has settled, a
    // durable re-read is fresh again even while the native cleanup latch is still pending.
    return active.key === key && !active.responded
      ? active.response
      : databaseOnlyRows(db, bounded);
  }

  const budget = deps.budget ?? new InspectionBudget(4, 2_000, deps.now ?? Date.now);
  const response = runPass(db, bounded, deps, budget);
  const pass: InspectionPass = {
    key,
    response,
    responded: false,
    settled: response.then(() => budget.settled(), () => budget.settled()),
  };
  response.then(() => { pass.responded = true; }, () => { pass.responded = true; });
  store.set(db, pass);
  void pass.settled.then(() => {
    if (store.get(db) === pass) store.delete(db);
  });
  return response;
}
