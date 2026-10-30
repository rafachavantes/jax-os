import { describe, expect, it, vi } from "vitest";
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getPrsForProjects } from "../../../../server/collectors/github";
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

  it("passes discovered locations to the collector and remaps result keys to basenames", async () => {
    const getPrsForProjects = vi.fn(async (locations: string[], reposRoot: string) => {
      const out: Record<string, { ok: true; prs: []; truncated: boolean }> = {};
      for (const location of locations) {
        out[location] = { ok: true, prs: [], truncated: false };
        expect(reposRoot).toBeTruthy();
      }
      return out;
    });
    const deps: PrsRouteDeps = {
      readGeneralSettings: () => githubEnabled,
      getProjects: () =>
        ({
          projects: [
            { dir: "good-a" },
            { dir: "nested", path: "company/nested" },
          ],
          skipped: 0,
        }) as unknown as ReturnType<PrsRouteDeps["getProjects"]>,
      getPrsForProjects: getPrsForProjects as unknown as PrsRouteDeps["getPrsForProjects"],
    };
    const body = await (await handlePrsGet(deps)).json();
    expect(getPrsForProjects).toHaveBeenCalledWith(["good-a", "company/nested"], expect.any(String));
    expect(Object.keys(body.data).sort()).toEqual(["good-a", "nested"]);
    expect(body.data["nested"]).toEqual({ ok: true, prs: [], truncated: false });
  });

  it("resolves a nested Git remote through the real collector and remaps its PRs", async () => {
    const root = mkdtempSync(join(tmpdir(), "jax-nested-prs-"));
    try {
      const repo = join(root, "company", "nested");
      mkdirSync(repo, { recursive: true });
      execFileSync("git", ["init", "-q", repo]);
      execFileSync("git", ["-C", repo, "remote", "add", "origin", "https://github.com/acme/remote-repo.git"]);
      const run = vi.fn(async (file: string, args: string[]) => {
        if (file === "git") return execFileSync(file, args, { encoding: "utf8" });
        expect(file).toBe("gh");
        expect(args).toEqual(expect.arrayContaining(["-R", "acme/remote-repo"]));
        return JSON.stringify([{ number: 27, title: "Nested PR", isDraft: false,
          url: "https://github.com/acme/remote-repo/pull/27", statusCheckRollup: [],
          updatedAt: "2026-10-29T00:00:00Z" }]);
      });
      const body = await (await handlePrsGet({
        readGeneralSettings: () => githubEnabled,
        getProjects: () => ({ projects: [{ dir: "nested", path: "company/nested" }],
          skipped: 0, reposRoot: root }) as unknown as ReturnType<PrsRouteDeps["getProjects"]>,
        getPrsForProjects: (locations) => getPrsForProjects(locations, root, run),
      })).json();
      expect(run).toHaveBeenCalledWith("git", ["-C", repo, "remote", "get-url", "origin"]);
      expect(body.ok).toBe(true);
      expect(Object.keys(body.data)).toEqual(["nested"]);
      expect(body.data.nested).toMatchObject({ ok: true, prs: [{ number: 27, title: "Nested PR" }] });
    } finally {
      rmSync(root, { recursive: true, force: true });
    }
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
