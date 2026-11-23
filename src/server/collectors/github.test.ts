import { describe, expect, it } from "vitest";
import {
  fetchProjectPrs, getPrsForProjects, parseGitHubRemote, parsePrListJson, summarizePrState, type Pr,
} from "./github";

describe("parseGitHubRemote", () => {
  it("parses an SSH remote", () => {
    expect(parseGitHubRemote("git@github.com:octo-owner/jax-os.git")).toEqual({
      owner: "octo-owner", repo: "jax-os",
    });
  });

  it("parses an HTTPS remote with and without .git", () => {
    expect(parseGitHubRemote("https://github.com/octo-owner/jax-os.git")).toEqual({
      owner: "octo-owner", repo: "jax-os",
    });
    expect(parseGitHubRemote("https://github.com/octo-owner/jax-os")).toEqual({
      owner: "octo-owner", repo: "jax-os",
    });
  });

  it("returns null for a non-GitHub remote", () => {
    expect(parseGitHubRemote("git@gitlab.com:alice/other.git")).toBeNull();
  });

  it("returns null for a host that merely CONTAINS github.com as a substring, never treating it as GitHub (cold review Finding 6)", () => {
    // A naive `/github\.com/` match accepts "evilgithub.com", which contains "github.com" as a
    // substring — that would report GitHub repo owner/name for a remote hosted somewhere else
    // entirely, returning another repository's PR data. The host must be EXACTLY github.com.
    expect(parseGitHubRemote("https://evilgithub.com/a/b.git")).toBeNull();
    expect(parseGitHubRemote("git@evilgithub.com:a/b.git")).toBeNull();
  });
});

describe("parsePrListJson — merge-ready rule (§5, test 3)", () => {
  it("open non-draft with all checks concluded successfully is merge-ready", () => {
    const json = JSON.stringify([{
      number: 1, title: "green", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
      statusCheckRollup: [{ __typename: "CheckRun", conclusion: "SUCCESS" }],
    }]);
    const { prs } = parsePrListJson(json);
    expect(prs[0]).toMatchObject({ ciState: "green", mergeReady: true });
  });

  it("open non-draft with an empty rollup (no CI configured) is merge-ready", () => {
    // Real evidence, 2026-08-27: gh pr list -R octo-owner/jax-os --json ... returns
    // statusCheckRollup: [] for a repo with no CI. This must count as merge-ready, not hidden.
    const json = JSON.stringify([{
      number: 1, title: "no ci", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z", statusCheckRollup: [],
    }]);
    const { prs } = parsePrListJson(json);
    expect(prs[0]).toMatchObject({ ciState: "none", mergeReady: true });
  });

  it("a draft PR is never merge-ready, even with a clean rollup", () => {
    const json = JSON.stringify([{
      number: 1, title: "draft", isDraft: true, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
      statusCheckRollup: [{ __typename: "CheckRun", conclusion: "SUCCESS" }],
    }]);
    const { prs } = parsePrListJson(json);
    expect(prs[0].mergeReady).toBe(false);
  });

  it("any failing conclusion makes the PR not merge-ready", () => {
    const json = JSON.stringify([{
      number: 1, title: "red", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
      statusCheckRollup: [
        { __typename: "CheckRun", conclusion: "SUCCESS" },
        { __typename: "CheckRun", conclusion: "FAILURE" },
      ],
    }]);
    const { prs } = parsePrListJson(json);
    expect(prs[0]).toMatchObject({ ciState: "failing", mergeReady: false });
  });

  it("a check with no conclusion yet (pending) makes the PR not merge-ready", () => {
    const json = JSON.stringify([{
      number: 1, title: "pending", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
      statusCheckRollup: [{ __typename: "CheckRun", conclusion: null }],
    }]);
    const { prs } = parsePrListJson(json);
    expect(prs[0]).toMatchObject({ ciState: "pending", mergeReady: false });
  });

  it("a CheckRun with `conclusion` entirely ABSENT (not just null) still counts as pending, never failing (cold review Finding 3)", () => {
    // gh's JSON can omit the key altogether for a running check, not just send it as null. The
    // spec (§5.1) pins conclusion null/absent to pending; a shape check keyed on `"conclusion" in
    // member` would treat this row as matching neither known shape and fail closed to "failing".
    const json = JSON.stringify([{
      number: 1, title: "running", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
      statusCheckRollup: [{ __typename: "CheckRun" }],
    }]);
    const { prs } = parsePrListJson(json);
    expect(prs[0]).toMatchObject({ ciState: "pending", mergeReady: false });
  });

  it("a MISSING statusCheckRollup throws — never silently treated as no CI (cold review Finding 13)", () => {
    // Real evidence, 2026-08-27: gh always includes the field when it is requested; a genuine
    // no-CI repo comes back as statusCheckRollup: [], never a missing key. A missing/non-array
    // field means the response shape changed under us — treating it as [] would report a
    // malformed PR as a green, merge-ready no-CI PR, the one direction spec §5/§5.1 forbids.
    const json = JSON.stringify([{ number: 1, title: "malformed", isDraft: false, url: "u1" }]);
    expect(() => parsePrListJson(json)).toThrow(/statusCheckRollup/);
  });

  it("a NON-ARRAY statusCheckRollup throws (cold review Finding 13)", () => {
    const json = JSON.stringify([{
      number: 1, title: "malformed", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z", statusCheckRollup: "oops",
    }]);
    expect(() => parsePrListJson(json)).toThrow(/statusCheckRollup/);
  });

  it("a row with an empty rollup but no number/title/isDraft/url throws — never a silent mergeReady PR with undefined fields (cold review round 3, Finding 6)", () => {
    // Round 2's fix validated only that statusCheckRollup is an array. {"statusCheckRollup":[]}
    // alone still has no number/title/isDraft/url — Boolean(undefined) was false, ciState was
    // "none", and the row became {mergeReady:true} with every display/link field undefined. An
    // empty rollup is the valid no-CI case (previous two tests), but only once the REST of the row
    // is a real PR row.
    const json = JSON.stringify([{ statusCheckRollup: [] }]);
    expect(() => parsePrListJson(json)).toThrow(/number|title|isDraft|url/);
  });
});

describe("parsePrListJson — StatusContext rollup members (§5.1, test 18)", () => {
  it("state: SUCCESS counts as green, not pending", () => {
    const json = JSON.stringify([{
      number: 1, title: "legacy green", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
      statusCheckRollup: [{ __typename: "StatusContext", state: "SUCCESS" }],
    }]);
    expect(parsePrListJson(json).prs[0].ciState).toBe("green");
  });

  it("state: FAILURE and state: ERROR count as failing", () => {
    for (const state of ["FAILURE", "ERROR"]) {
      const json = JSON.stringify([{
        number: 1, title: "legacy red", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
        statusCheckRollup: [{ __typename: "StatusContext", state }],
      }]);
      expect(parsePrListJson(json).prs[0].ciState).toBe("failing");
    }
  });

  it("state: PENDING and state: EXPECTED count as pending", () => {
    for (const state of ["PENDING", "EXPECTED"]) {
      const json = JSON.stringify([{
        number: 1, title: "legacy pending", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
        statusCheckRollup: [{ __typename: "StatusContext", state }],
      }]);
      expect(parsePrListJson(json).prs[0].ciState).toBe("pending");
    }
  });

  it("a rollup member matching neither CheckRun nor StatusContext shape counts as failing (fail closed)", () => {
    const json = JSON.stringify([{
      number: 1, title: "mystery", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
      statusCheckRollup: [{ __typename: "SomethingElse", foo: "bar" }],
    }]);
    expect(parsePrListJson(json).prs[0].ciState).toBe("failing");
  });

  it("an explicit unrecognised __typename fails closed even when a conclusion property looks green (branch review Finding 4)", () => {
    // The property-presence fallback below the two known __typename branches must never run for a
    // member that DOES carry a __typename, just not one we recognise — otherwise
    // {__typename:"Unknown", conclusion:"SUCCESS"} reads as green via the legacy fallback,
    // violating §5.1's fail-closed rule for unknown shapes.
    const json = JSON.stringify([{
      number: 1, title: "spoofed", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z",
      statusCheckRollup: [{ __typename: "Unknown", conclusion: "SUCCESS" }],
    }]);
    expect(parsePrListJson(json).prs[0].ciState).toBe("failing");
  });
});

describe("M2 factual provider fields (plan F3 / Task 4)", () => {
  const base = { number: 1, title: "t", isDraft: false, url: "u", statusCheckRollup: [], updatedAt: "2026-11-18T00:00:00Z" };

  it("keeps known author/review/merge states exactly as GitHub reports them", () => {
    const { prs } = parsePrListJson(JSON.stringify([{
      ...base, author: { login: "alice" }, reviewDecision: "APPROVED", mergeStateStatus: "CLEAN",
    }]));
    expect(prs[0]).toMatchObject({ author: "alice", reviewDecision: "APPROVED", mergeStateStatus: "CLEAN" });
  });

  it("covers every known review and merge state as factual", () => {
    for (const review of ["APPROVED", "CHANGES_REQUESTED", "REVIEW_REQUIRED"]) {
      expect(parsePrListJson(JSON.stringify([{ ...base, reviewDecision: review }])).prs[0].reviewDecision).toBe(review);
    }
    for (const state of ["BEHIND", "BLOCKED", "CLEAN", "DIRTY", "DRAFT", "HAS_HOOKS", "UNKNOWN", "UNSTABLE"]) {
      expect(parsePrListJson(JSON.stringify([{ ...base, mergeStateStatus: state }])).prs[0].mergeStateStatus).toBe(state);
    }
  });

  it("missing, empty and unrecognised optional values become the explicit unknown labels", () => {
    const { prs } = parsePrListJson(JSON.stringify([{ ...base, author: null, reviewDecision: "", mergeStateStatus: "SOMETHING_NEW" }]));
    expect(prs[0]).toMatchObject({ author: "unknown author", reviewDecision: "unknown/not reported", mergeStateStatus: "unknown" });
    const empty = parsePrListJson(JSON.stringify([base])).prs[0];
    expect(empty).toMatchObject({ author: "unknown author", reviewDecision: "unknown/not reported", mergeStateStatus: "unknown" });
    expect(parsePrListJson(JSON.stringify([{ ...base, author: { login: "   " } }])).prs[0].author).toBe("unknown author");
    expect(parsePrListJson(JSON.stringify([{ ...base, author: "not a login\u0000" }])).prs[0].author).toBe("unknown author");
  });

  it("an empty rollup and CLEAN merge state stay factual, never an implied permission to merge", () => {
    const { prs } = parsePrListJson(JSON.stringify([{ ...base, mergeStateStatus: "CLEAN", reviewDecision: "APPROVED" }]));
    expect(prs[0]).toMatchObject({ ciState: "none", mergeStateStatus: "CLEAN", reviewDecision: "APPROVED" });
    // mergeReady remains the old informational flag; no consumer treats it as an owner action.
    expect(prs[0].mergeReady).toBe(true);
  });
});

describe("M2 Mine/All provider scope (plan Task 4, spec AC8/AC9)", () => {
  it("Mine resolves the viewer once and applies --author before --limit", async () => {
    const calls: { file: string; args: string[] }[] = [];
    const run = async (file: string, args: string[]): Promise<string> => {
      calls.push({ file, args });
      if (file === "git") return "git@github.com:octo-owner/jax-os.git\n";
      if (args[0] === "api") return "alice\n";
      return JSON.stringify([{ number: 3, title: "mine", isDraft: false, url: "u", statusCheckRollup: [], updatedAt: "2026-11-18T00:00:00Z", author: { login: "alice" }, reviewDecision: "APPROVED", mergeStateStatus: "CLEAN" }]);
    };
    const out = await getPrsForProjects(["a", "b"], "/home/user/repos", run, "mine");
    expect(out["a"]).toMatchObject({ ok: true, prs: [{ number: 3, author: "alice" }] });
    expect(out["b"]).toMatchObject({ ok: true });
    // Viewer resolved exactly once for the whole collection (two projects).
    expect(calls.filter((c) => c.file === "gh" && c.args[0] === "api")).toHaveLength(1);
    const listCalls = calls.filter((c) => c.file === "gh" && c.args[0] === "pr");
    expect(listCalls).toHaveLength(2);
    for (const call of listCalls) {
      const authorIdx = call.args.indexOf("--author");
      expect(call.args[authorIdx + 1]).toBe("alice");
      expect(authorIdx).toBeLessThan(call.args.indexOf("--limit"));
    }
  });

  it("more than 50 other-author PRs never matter: the provider-filtered response is what Mine returns", async () => {
    const run = async (file: string, args: string[]): Promise<string> => {
      if (file === "git") return "git@github.com:octo-owner/jax-os.git\n";
      if (args[0] === "api") return "alice\n";
      // A Mine-filtered provider answers with ONLY the viewer's PRs (the --author flag is the
      // filter that makes this true upstream); assert we do not locally re-filter it away.
      return JSON.stringify([{ number: 7, title: "my pr", isDraft: false, url: "u", statusCheckRollup: [], updatedAt: "2026-11-18T00:00:00Z", author: { login: "alice" } }]);
    };
    const out = await getPrsForProjects(["jax-os"], "/home/user/repos", run, "mine");
    expect(out["jax-os"]).toMatchObject({ ok: true, prs: [{ number: 7 }] });
  });

  it("a failing viewer lookup marks only GitHub-backed projects unavailable; local/non-GitHub stay quiet empty (diff review F3)", async () => {
    const calls: { file: string; args: string[] }[] = [];
    const run = async (file: string, args: string[]): Promise<string> => {
      calls.push({ file, args });
      if (file === "git") {
        const dir = args[1].split("/").pop();
        if (dir === "local") throw Object.assign(new Error("fatal: No such remote 'origin'"), { code: 2 });
        if (dir === "other") return "git@gitlab.com:someone/other.git\n";
        return `git@github.com:octo-owner/${dir}.git\n`;
      }
      if (args[0] === "api") throw new Error("gh: To use GitHub CLI, set GH_TOKEN");
      return JSON.stringify([]);
    };
    const out = await getPrsForProjects(["local", "other", "gh-a", "gh-b"], "/home/user/repos", run, "mine");
    // No remote / non-GitHub are resolved, successful, quiet-empty outcomes — not identity failures.
    expect(out["local"]).toEqual({ ok: true, prs: [], truncated: false });
    expect(out["other"]).toEqual({ ok: true, prs: [], truncated: false });
    // Only the GitHub-backed projects cannot answer Mine without the viewer identity.
    expect(out["gh-a"]).toEqual({ ok: false, error: "github identity unavailable" });
    expect(out["gh-b"]).toEqual({ ok: false, error: "github identity unavailable" });
    // The viewer is resolved lazily, after a validated GitHub remote exists, and memoized once.
    expect(calls.filter((c) => c.file === "gh" && c.args[0] === "api")).toHaveLength(1);
  });

  it("no GitHub projects means zero viewer queries in Mine", async () => {
    const calls: { file: string; args: string[] }[] = [];
    const run = async (file: string, args: string[]): Promise<string> => {
      calls.push({ file, args });
      if (file === "git") {
        const dir = args[1].split("/").pop();
        if (dir === "local") throw Object.assign(new Error("fatal: No such remote 'origin'"), { code: 2 });
        return "git@gitlab.com:someone/other.git\n";
      }
      return JSON.stringify([]);
    };
    const out = await getPrsForProjects(["local", "other"], "/home/user/repos", run, "mine");
    expect(out["local"]).toEqual({ ok: true, prs: [], truncated: false });
    expect(out["other"]).toEqual({ ok: true, prs: [], truncated: false });
    expect(calls.some((c) => c.args[0] === "api")).toBe(false);
  });

  it("All omits --author and never asks for the viewer", async () => {
    const calls: { file: string; args: string[] }[] = [];
    const run = async (file: string, args: string[]): Promise<string> => {
      calls.push({ file, args });
      if (file === "git") return "git@github.com:octo-owner/jax-os.git\n";
      return JSON.stringify([]);
    };
    await getPrsForProjects(["jax-os"], "/home/user/repos", run, "all");
    expect(calls.some((c) => c.args[0] === "api")).toBe(false);
    expect(calls.some((c) => c.args.includes("--author"))).toBe(false);
  });

  it("a root viewer auth failure is source-unavailable for every project, never All and never a fake zero", async () => {
    const run = async (file: string, args: string[]): Promise<string> => {
      if (file === "git") return "git@github.com:octo-owner/jax-os.git\n";
      if (args[0] === "api") throw new Error("gh: To use GitHub CLI, set GH_TOKEN (secret-value)");
      return JSON.stringify([]);
    };
    const out = await getPrsForProjects(["a", "b"], "/home/user/repos", run, "mine");
    expect(out["a"]).toEqual({ ok: false, error: "github identity unavailable" });
    expect(out["b"]).toEqual({ ok: false, error: "github identity unavailable" });
  });

  it("an invalid viewer login is a source failure, never used as a filter", async () => {
    const run = async (file: string, args: string[]): Promise<string> => {
      if (file === "git") return "git@github.com:octo-owner/jax-os.git\n";
      if (args[0] === "api") return "not a login\n";
      return JSON.stringify([]);
    };
    expect(await getPrsForProjects(["a"], "/home/user/repos", run, "mine")).toMatchObject({
      a: { ok: false, error: "github identity unavailable" },
    });
  });

  it("disabled integration is the route's quiet empty path; no-origin/non-GitHub stays a successful empty per repo", async () => {
    const noOrigin = async (): Promise<string> => {
      throw Object.assign(new Error("fatal: No such remote 'origin'"), { code: 2 });
    };
    expect(await fetchProjectPrs("x", "/home/user/repos", noOrigin, { scope: "all" })).toEqual({ ok: true, prs: [], truncated: false });
  });
});

describe("summarizePrState — worst state wins (§5, test 11)", () => {
  const pr = (ciState: Pr["ciState"]): Pr => ({
    number: 1, title: "x", isDraft: false, url: "u", ciState, mergeReady: ciState !== "failing" && ciState !== "pending", author: "a", reviewDecision: "unknown/not reported" as const, mergeStateStatus: "unknown" as const,
    updatedAt: "2026-09-19T00:00:00Z",
  });

  it("green + pending + failing summarises as failing", () => {
    expect(summarizePrState([pr("green"), pr("pending"), pr("failing")])).toBe("failing");
  });

  it("a set where every PR has an empty rollup (ciState 'none') summarises as no CI", () => {
    expect(summarizePrState([pr("none"), pr("none")])).toBe("none");
  });

  it("green + pending summarises as pending", () => {
    expect(summarizePrState([pr("green"), pr("pending")])).toBe("pending");
  });
});

describe("parsePrListJson — truncation at the 50-row limit (§5.1, test 19)", () => {
  it("a response at exactly 50 rows reports truncated: true", () => {
    const rows = Array.from({ length: 50 }, (_, i) => ({
      number: i + 1, title: `pr ${i}`, isDraft: false, url: `u${i}`, statusCheckRollup: [], updatedAt: "2026-09-19T00:00:00Z",
    }));
    expect(parsePrListJson(JSON.stringify(rows)).truncated).toBe(true);
  });

  it("a response under the limit reports truncated: false", () => {
    const rows = [{ number: 1, title: "pr", isDraft: false, url: "u1", updatedAt: "2026-09-19T00:00:00Z", statusCheckRollup: [] }];
    expect(parsePrListJson(JSON.stringify(rows)).truncated).toBe(false);
  });
});

describe("fetchProjectPrs / getPrsForProjects — invocation and failure isolation", () => {
  it("resolves the remote with git -C <path>, never the process cwd, and calls gh -R explicitly", async () => {
    const calls: { file: string; args: string[] }[] = [];
    const run = async (file: string, args: string[]): Promise<string> => {
      calls.push({ file, args });
      if (file === "git") return "git@github.com:octo-owner/jax-os.git\n";
      return JSON.stringify([{ number: 1, title: "t", isDraft: false, url: "u", statusCheckRollup: [], updatedAt: "2026-09-19T00:00:00Z" }]);
    };
    const result = await fetchProjectPrs("jax-os", "/home/user/repos", run);
    expect(calls[0]).toEqual({ file: "git", args: ["-C", "/home/user/repos/jax-os", "remote", "get-url", "origin"] });
    expect(calls[1]).toEqual({
      file: "gh",
      args: [
        "pr", "list", "-R", "octo-owner/jax-os", "--state", "open", "--limit", "50",
        "--json", "number,title,isDraft,url,statusCheckRollup,updatedAt,author,reviewDecision,mergeStateStatus",
      ],
    });
    expect(result).toMatchObject({ ok: true, truncated: false });
  });

  it("a project with no git remote resolves to an explicit empty result, never null (cold review round 4, Finding 4)", async () => {
    const run = async (): Promise<string> => {
      const err = new Error("Command failed: git -C /x remote get-url origin\nfatal: No such remote 'origin'") as Error & { code?: number };
      err.code = 2; // git's real exit code for a missing remote
      throw err;
    };
    expect(await fetchProjectPrs("no-remote", "/home/user/repos", run)).toEqual({ ok: true, prs: [], truncated: false });
  });

  it("a git failure that is NOT a missing remote (git unavailable, bad path, ...) reports {ok:false}, not a quiet empty result (cold review Finding 1)", async () => {
    // Every git failure used to become {ok:true, prs:[]} — indistinguishable from a project that
    // genuinely has no GitHub remote. A spawn failure (git missing, ENOENT) or any other non-zero
    // exit that isn't "No such remote" must surface as a source error instead.
    const run = async (): Promise<string> => {
      const err = new Error("spawn git ENOENT") as Error & { code?: string };
      err.code = "ENOENT";
      throw err;
    };
    const result = await fetchProjectPrs("broken", "/home/user/repos", run);
    expect(result).toMatchObject({ ok: false });
  });

  it("an exit-code-2 git failure that does NOT name origin as the missing remote reports {ok:false} (branch review Finding 3)", async () => {
    // Repo corruption can also exit git with code 2 — code alone is not proof the remote is
    // missing. Only a diagnostic that actually names `origin` as the missing remote is the
    // verified quiet-empty case; everything else must surface as a source failure.
    const run = async (): Promise<string> => {
      const err = new Error("fatal: repository metadata is corrupt") as Error & { code?: number };
      err.code = 2;
      throw err;
    };
    const result = await fetchProjectPrs("corrupt", "/home/user/repos", run);
    expect(result).toMatchObject({ ok: false });
  });

  it("a project dir named __proto__ appears in the result, never swallowed into the prototype (branch review Finding 6)", async () => {
    const run = async (file: string): Promise<string> => {
      if (file === "git") return "git@github.com:octo-owner/x.git\n";
      return JSON.stringify([]);
    };
    const out = await getPrsForProjects(["__proto__"], "/home/user/repos", run);
    expect(Object.prototype.hasOwnProperty.call(out, "__proto__")).toBe(true);
    expect(out["__proto__"]).toEqual({ ok: true, prs: [], truncated: false });
  });

  it("a gh failure for one project does not affect the others (failure isolation)", async () => {
    const run = async (file: string, args: string[]): Promise<string> => {
      if (file === "git") return `git@github.com:octo-owner/${args[1].split("/").pop()}.git\n`;
      if (args.includes("bad")) throw new Error("gh: authentication failed");
      return JSON.stringify([]);
    };
    const out = await getPrsForProjects(["good-a", "bad", "good-b"], "/home/user/repos", async (file, args) => {
      if (file === "git") {
        const dir = args[1].split("/").pop();
        return `git@github.com:octo-owner/${dir}.git\n`;
      }
      if (args.includes("octo-owner/bad")) throw new Error("gh: authentication failed");
      return JSON.stringify([]);
    });
    expect(out["good-a"]).toEqual({ ok: true, prs: [], truncated: false });
    expect(out["bad"]).toEqual({ ok: false, error: "gh: authentication failed" });
    expect(out["good-b"]).toEqual({ ok: true, prs: [], truncated: false });
  });

  it("a malformed statusCheckRollup from gh becomes {ok:false}, never a green mergeReady PR (Finding 13)", async () => {
    const run = async (file: string): Promise<string> => {
      if (file === "git") return "git@github.com:octo-owner/jax-os.git\n";
      return JSON.stringify([{ number: 9, title: "t", isDraft: false, url: "u" }]); // no statusCheckRollup key
    };
    const result = await fetchProjectPrs("jax-os", "/home/user/repos", run);
    expect(result).toMatchObject({ ok: false });
  });

  it("treats git timeout as source failure, not a quiet empty remote", async () => {
    const run = async () => {
      throw Object.assign(new Error("timeout"), { code: "ETIMEDOUT", stdout: "git@github.com:a/b.git\n" });
    };
    expect(await fetchProjectPrs("jax-os", "/home/user/repos", run)).toMatchObject({ ok: false });
  });

  it("treats gh overflow as source failure, not partial parsed PRs", async () => {
    const leaked = JSON.stringify([{
      number: 1, title: "leaked", isDraft: false, url: "u", statusCheckRollup: [],
    }]);
    const run = async (file: string) => {
      if (file === "git") return "git@github.com:octo-owner/jax-os.git\n";
      throw Object.assign(new Error("overflow"), {
        code: "ERR_CHILD_PROCESS_STDIO_MAXBUFFER",
        stdout: leaked,
      });
    };
    const result = await fetchProjectPrs("jax-os", "/home/user/repos", run);
    expect(result).toMatchObject({ ok: false });
    if (result.ok) throw new Error("parsed overflow stdout");
  });
});

describe("Phase 3 — Pr.updatedAt (spec §9, Decision 13, cold review F4)", () => {
  it("requires updatedAt fail-closed, same as every other field", () => {
    const row = { number: 1, title: "x", isDraft: false, url: "https://x", statusCheckRollup: [], updatedAt: "2026-09-19T00:00:00Z" };
    const { prs } = parsePrListJson(JSON.stringify([row]));
    expect(prs[0].updatedAt).toBe("2026-09-19T00:00:00Z");
    const missing = { ...row };
    delete (missing as Record<string, unknown>).updatedAt;
    expect(() => parsePrListJson(JSON.stringify([missing]))).toThrow(/updatedAt/);
  });
});
