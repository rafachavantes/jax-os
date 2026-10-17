import { NextResponse } from "next/server";
import { collectorResponse } from "../../../../server/api";
import { getPrsForProjects, type PrResult } from "../../../../server/collectors/github";
import { getProjects, REPOS_ROOT } from "../../../../server/collectors/projects";
import { readGeneralSettings } from "../../../../server/settings";

export type PrsRouteDeps = {
  getProjects: typeof getProjects;
  getPrsForProjects: typeof getPrsForProjects;
  readGeneralSettings: typeof readGeneralSettings;
};

const systemDeps: PrsRouteDeps = { getProjects, getPrsForProjects, readGeneralSettings };

// Only carded projects (§5) — the same set mission/projects returns.
export async function handlePrsGet(deps: PrsRouteDeps = systemDeps) {
  const settings = deps.readGeneralSettings();
  if (!(settings.ok && settings.data.integrations.github)) {
    // Quiet empty (per-integration table) — the SAME shape a real "no PRs anywhere" response
    // already has. mission.ts's own pr.status derivation treats a missing map entry as "unknown"
    // (loading), not "hidden" — this route's contract is only "never call getPrsForProjects"; the
    // UI-side gate in ProjectCard.tsx is what actually hides the section.
    return NextResponse.json({ ok: true, data: {} });
  }
  return collectorResponse<Record<string, PrResult>>(async () => {
    const { projects } = deps.getProjects();
    return deps.getPrsForProjects(projects.map((p) => p.dir), REPOS_ROOT);
  });
}
