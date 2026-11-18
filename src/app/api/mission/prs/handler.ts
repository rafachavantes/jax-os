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

// Only carded projects (§5) — the same set mission/projects returns. Scope defaults to Mine for
// callers without a query (spec: Mine is the initial scope); the route validates the query.
export async function handlePrsGet(deps: PrsRouteDeps = systemDeps, scope: "mine" | "all" = "mine") {
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
    // The collector resolves a repository from its own path; nested projects therefore pass
    // their discovered location and the returned keys are remapped to the basename identity.
    const locations = projects.map((p) => p.path ?? p.dir);
    const fetched = await deps.getPrsForProjects(locations, REPOS_ROOT, undefined, scope);
    const out: Record<string, PrResult> = Object.create(null);
    projects.forEach((p, i) => { out[p.dir] = fetched[locations[i]]; });
    return out;
  });
}
