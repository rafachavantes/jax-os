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
    expect(parseGitHubRemote("git@gitlab.com:rafa/other.git")).toBeNull();
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

describe("summarizePrState — worst state wins (§5, test 11)", () => {
  const pr = (ciState: Pr["ciState"]): Pr => ({
    number: 1, title: "x", isDraft: false, url: "u", ciState, mergeReady: ciState !== "failing" && ciState !== "pending",
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
    const result = await fetchProjectPrs("jax-os", "/home/rafa/repos", run);
    expect(calls[0]).toEqual({ file: "git", args: ["-C", "/home/rafa/repos/jax-os", "remote", "get-url", "origin"] });
    expect(calls[1]).toEqual({
      file: "gh",
      args: [
        "pr", "list", "-R", "octo-owner/jax-os", "--state", "open", "--limit", "50",
        "--json", "number,title,isDraft,url,statusCheckRollup,updatedAt",
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
    expect(await fetchProjectPrs("no-remote", "/home/rafa/repos", run)).toEqual({ ok: true, prs: [], truncated: false });
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
    const result = await fetchProjectPrs("broken", "/home/rafa/repos", run);
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
    const result = await fetchProjectPrs("corrupt", "/home/rafa/repos", run);
    expect(result).toMatchObject({ ok: false });
  });

  it("a project dir named __proto__ appears in the result, never swallowed into the prototype (branch review Finding 6)", async () => {
    const run = async (file: string): Promise<string> => {
      if (file === "git") return "git@github.com:octo-owner/x.git\n";
      return JSON.stringify([]);
    };
    const out = await getPrsForProjects(["__proto__"], "/home/rafa/repos", run);
    expect(Object.prototype.hasOwnProperty.call(out, "__proto__")).toBe(true);
    expect(out["__proto__"]).toEqual({ ok: true, prs: [], truncated: false });
  });

  it("a gh failure for one project does not affect the others (failure isolation)", async () => {
    const run = async (file: string, args: string[]): Promise<string> => {
      if (file === "git") return `git@github.com:octo-owner/${args[1].split("/").pop()}.git\n`;
      if (args.includes("bad")) throw new Error("gh: authentication failed");
      return JSON.stringify([]);
    };
    const out = await getPrsForProjects(["good-a", "bad", "good-b"], "/home/rafa/repos", async (file, args) => {
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
    const result = await fetchProjectPrs("jax-os", "/home/rafa/repos", run);
    expect(result).toMatchObject({ ok: false });
  });

  it("treats git timeout as source failure, not a quiet empty remote", async () => {
    const run = async () => {
      throw Object.assign(new Error("timeout"), { code: "ETIMEDOUT", stdout: "git@github.com:a/b.git\n" });
    };
    expect(await fetchProjectPrs("jax-os", "/home/rafa/repos", run)).toMatchObject({ ok: false });
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
    const result = await fetchProjectPrs("jax-os", "/home/rafa/repos", run);
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
