import { describe, expect, it, vi } from "vitest";
import type { SettingsResult } from "../../../../server/settings";
import { handlePrsGet, type PrsRouteDeps } from "./handler";

const githubEnabled: SettingsResult = {
  ok: true,
  data: { integrations: { github: true } },
} as SettingsResult;
const githubDisabled: SettingsResult = {
  ok: true,
  data: { integrations: { github: false } },
} as SettingsResult;

describe("GET /api/mission/prs (§9 test 15 — gh failure isolation)", () => {
  it("one project's {ok:false} does not blank the others", async () => {
    const deps: PrsRouteDeps = {
      readGeneralSettings: () => githubEnabled,
      getProjects: () =>
        ({
          projects: [{ dir: "good-a" }, { dir: "bad" }, { dir: "good-b" }],
          skipped: 0,
        }) as unknown as ReturnType<PrsRouteDeps["getProjects"]>,
      getPrsForProjects: async (dirs) => {
        const out: Record<string, { ok: true; prs: []; truncated: boolean } | { ok: false; error: string }> = {};
        for (const dir of dirs) {
          out[dir] = dir === "bad" ? { ok: false, error: "gh: command failed" } : { ok: true, prs: [], truncated: false };
        }
        return out;
      },
    };
    const res = await handlePrsGet(deps);
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(body.data["good-a"]).toEqual({ ok: true, prs: [], truncated: false });
    expect(body.data["bad"]).toEqual({ ok: false, error: "gh: command failed" });
    expect(body.data["good-b"]).toEqual({ ok: true, prs: [], truncated: false });
  });

  it("refuses with a quiet empty map when github is disabled, never calls getPrsForProjects", async () => {
    const getPrsForProjects = vi.fn();
    const deps: PrsRouteDeps = {
      readGeneralSettings: () => githubDisabled,
      getProjects: () =>
        ({ projects: [{ dir: "good-a" }], skipped: 0 }) as unknown as ReturnType<PrsRouteDeps["getProjects"]>,
      getPrsForProjects: getPrsForProjects as unknown as PrsRouteDeps["getPrsForProjects"],
    };
    const json = await (await handlePrsGet(deps)).json();
    expect(json).toEqual({ ok: true, data: {} });
    expect(getPrsForProjects).not.toHaveBeenCalled();
  });
});
