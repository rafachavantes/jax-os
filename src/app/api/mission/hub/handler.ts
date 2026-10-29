import { isAbsolute, join } from "node:path";
import { collectorResponse } from "../../../../server/api";
import { getProjects, type ProjectScan } from "../../../../server/collectors/projects";
import { collectCodexSnapshot, createCodexConnector, isCodexCliPresent, type CodexSnapshot } from "../../../../server/collectors/workflow-codex";
import { listLivePanes } from "../../../../server/collectors/workflow-tmux";
import { readChildLogTail } from "../../../../server/collectors/child-log";
import { relativeWithin } from "../../../../server/collectors/files";
import { getDb } from "../../../../server/db";
import { readGeneralSettings } from "../../../../server/settings";
import { getHubData, type ChildLogTailReader, type HubEnvelopeData, type RelToRepos } from "../../../../server/db/workflows";

export type HubRouteDeps = {
  getDb: typeof getDb;
  getProjects: typeof getProjects;
  listLivePanes: typeof listLivePanes;
  collectCodexSnapshot: () => Promise<CodexSnapshot>;
  isCodexCliPresent: () => boolean;
  getHubData: typeof getHubData;
  // Phase 3 (spec §8/§11): OPTIONAL so every existing HubRouteDeps literal (route.test.ts) keeps
  // compiling unchanged — omitting them means "no enrichment", the same default getHubData itself
  // uses. systemDeps below wires the real, filesystem-backed implementations.
  readChildLogTail?: ChildLogTailReader;
  relToRepos?: RelToRepos;
  codexEnabled?: () => boolean; // MOA-504 D11: omitted = enabled (existing literals still compile)
};

const systemDeps: HubRouteDeps = {
  getDb,
  getProjects,
  listLivePanes,
  collectCodexSnapshot: () => collectCodexSnapshot(createCodexConnector),
  isCodexCliPresent,
  codexEnabled: () => { const s = readGeneralSettings(); return s.ok && s.data.integrations.agents.codex; },
  getHubData,
  readChildLogTail,
};

// Per-request artifact resolver: the project root comes from THIS scan's discovered location, and
// an identity that is missing or ambiguous gets no enrichment. Relative spec/plan targets are
// normalized here (never against the process cwd); absolute report paths keep canonical containment
// via relativeWithin. An injected resolver still wins, so tests/other callers stay explicit.
function relToReposFromScan(scan: ProjectScan): RelToRepos {
  const counts = new Map<string, number>();
  for (const project of scan.projects) {
    counts.set(project.dir, (counts.get(project.dir) ?? 0) + 1);
  }
  return (project, target) => {
    if (!scan.reposRoot || counts.get(project) !== 1) return null;
    const found = scan.projects.find((p) => p.dir === project);
    if (!found) return null;
    const projectRoot = join(scan.reposRoot, found.path ?? found.dir);
    const absolute = isAbsolute(target) ? target : join(projectRoot, target);
    return relativeWithin(projectRoot, absolute);
  };
}

// mission/hub takes no request parameters: it reads the DB and calls listLivePanes() and the
// Codex App Server collector once per request — the same subprocess cadence mission/agents runs
// (§7.1c). The two live sources are collected INDEPENDENTLY in BOTH directions (C4): a genuine
// tmux failure is a source warning with an empty snapshot, never a whole-hub failure that hides
// native Codex state; a Codex failure must never erase tmux data. The collector's own no-server
// case stays a healthy empty snapshot (tmuxSourceOk true).
export async function handleHubGet(deps: HubRouteDeps = systemDeps) {
  return collectorResponse<HubEnvelopeData>(async () => {
    const scan = deps.getProjects();
    const { projects } = scan;
    const relToRepos = deps.relToRepos ?? relToReposFromScan(scan);
    const statusMtimes: Record<string, string> = {};
    for (const p of projects) statusMtimes[p.dir] = p.statusMtime;
    let live: Awaited<ReturnType<typeof listLivePanes>>;
    let tmuxSourceOk = true;
    try {
      live = await deps.listLivePanes();
    } catch {
      live = { paneIds: new Set(), incarnation: null, paneCommands: new Map() };
      tmuxSourceOk = false; // explicit warning, never a quiet healthy zero-agents hub
    }
    let codex: CodexSnapshot | null = null;
    const codexAbsent = !(deps.codexEnabled?.() ?? true) || !deps.isCodexCliPresent();
    if (!codexAbsent) {
      try {
        codex = await deps.collectCodexSnapshot();
      } catch {
        codex = null; // source warning, never a whole-hub failure
      }
    }
    return deps.getHubData(
      deps.getDb(), live, statusMtimes, codex, Date.now(), tmuxSourceOk,
      deps.readChildLogTail ?? null, relToRepos, codexAbsent,
    );
  });
}
