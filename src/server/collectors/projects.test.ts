import { afterEach, describe, expect, it } from "vitest";
import { execFileSync } from "node:child_process";
import { closeSync, fstatSync, mkdirSync, mkdtempSync, openSync, readFileSync, readSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import {
  canonicalOwner, extractNow, extractResiduals, getProjects, isLinkedWorktreeGitdir, ownerProjectName,
  parseCommondir, parseGitdirRef, parseStatusMd, readStatusBounded, STATUS_CAP,
  type GitProbe,
} from "./projects";

const MTIME = "2023-11-14T22:13:20.000Z";

const FULL = `---
project: APG Platform
stage: build
gate: awaiting-approval
builder: claude-code
branch: feat/checkout
tmux: apg-build
flag: main client
updated: 2026-07-06T14:30:00-03:00
---

# Status

## Now
First paragraph after Now
continues on a second line.

## Residuals
- first accepted risk
- second accepted risk

## Next
Ignored.
`;

describe("parseStatusMd", () => {
  it("parses a full frontmatter block", () => {
    expect(parseStatusMd("apg", FULL, MTIME)).toEqual({
      dir: "apg",
      name: "APG Platform",
      stage: "build",
      gate: "awaiting-approval",
      legacyGateField: false,
      builder: "claude-code",
      branch: "feat/checkout",
      tmux: "apg-build",
      flag: "main client",
      archived: false,
      updated: "2026-07-06T14:30:00-03:00",
      now: "First paragraph after Now continues on a second line.",
      residuals: ["first accepted risk", "second accepted risk"],
      statusMtime: MTIME,
    });
  });

  it("keeps ISO timestamp colons intact (splits on first ': ' only)", () => {
    const p = parseStatusMd("x", FULL, MTIME);
    expect(p?.updated).toBe("2026-07-06T14:30:00-03:00");
  });

  it("falls back to dir name and empty optionals, with no gate", () => {
    const raw = "---\nstage: spec\n---\n";
    expect(parseStatusMd("my-repo", raw, MTIME)).toEqual({
      dir: "my-repo",
      name: "my-repo",
      stage: "spec",
      gate: null,
      legacyGateField: false,
      builder: "",
      branch: "",
      tmux: undefined,
      flag: undefined,
      archived: false,
      updated: "",
      now: "",
      residuals: [],
      statusMtime: MTIME,
    });
  });

  it("returns null without frontmatter", () => {
    expect(parseStatusMd("x", "# Status\n\njust markdown", MTIME)).toBeNull();
  });

  it("still rejects a missing or invalid stage — gate no longer participates in rejection (§9 test 10)", () => {
    expect(parseStatusMd("x", "---\ngate: blocked\n---\n", MTIME)).toBeNull(); // stage absent
    expect(parseStatusMd("x", "---\nstage: nope\n---\n", MTIME)).toBeNull(); // stage invalid
  });

  describe("gate parsing and the legacy `state` alias (§9 test 2)", () => {
    it("gate: awaiting-approval and gate: blocked parse as gates", () => {
      expect(parseStatusMd("x", "---\nstage: build\ngate: awaiting-approval\n---\n", MTIME))
        .toMatchObject({ gate: "awaiting-approval", legacyGateField: false });
      expect(parseStatusMd("x", "---\nstage: build\ngate: blocked\n---\n", MTIME))
        .toMatchObject({ gate: "blocked", legacyGateField: false });
    });

    it("gate absent parses as no gate", () => {
      expect(parseStatusMd("x", "---\nstage: build\n---\n", MTIME))
        .toMatchObject({ gate: null, legacyGateField: false });
    });

    it("an unknown gate value parses as no gate, not a failure (§9 test 10)", () => {
      expect(parseStatusMd("x", "---\nstage: build\ngate: nonsense\n---\n", MTIME))
        .toMatchObject({ gate: null, legacyGateField: false });
    });

    it("state: idle and state: building — today's on-disk values — parse as no gate and set the legacy hint", () => {
      expect(parseStatusMd("x", "---\nstage: build\nstate: idle\n---\n", MTIME))
        .toMatchObject({ gate: null, legacyGateField: true });
      expect(parseStatusMd("x", "---\nstage: build\nstate: building\n---\n", MTIME))
        .toMatchObject({ gate: null, legacyGateField: true });
    });

    it("state: blocked carries its meaning across the alias", () => {
      expect(parseStatusMd("x", "---\nstage: build\nstate: blocked\n---\n", MTIME))
        .toMatchObject({ gate: "blocked", legacyGateField: true });
    });

    it("both keys present resolves to gate and ignores state — the file is not legacy", () => {
      expect(parseStatusMd("x", "---\nstage: build\ngate: awaiting-approval\nstate: blocked\n---\n", MTIME))
        .toMatchObject({ gate: "awaiting-approval", legacyGateField: false });
      // gate wins even when its own value is unrecognised — the file is still not legacy
      expect(parseStatusMd("x", "---\nstage: build\ngate: nonsense\nstate: blocked\n---\n", MTIME))
        .toMatchObject({ gate: null, legacyGateField: false });
    });
  });
});

describe("extractNow", () => {
  it("returns empty string when there is no ## Now", () => {
    expect(extractNow("## Next\nstuff")).toBe("");
  });

  it("stops at blank line or next heading", () => {
    expect(extractNow("## Now\n\nPara one.\nStill one.\n\nPara two.")).toBe(
      "Para one. Still one.",
    );
    expect(extractNow("## Now\nOnly this.\n## Next\nNot this.")).toBe("Only this.");
  });
});

describe("extractResiduals (§9 test 1)", () => {
  it("returns the bullet lines under ## Residuals", () => {
    expect(extractResiduals("## Residuals\n- risk one\n- risk two\n")).toEqual([
      "risk one",
      "risk two",
    ]);
  });

  it("returns an empty array when there is no ## Residuals section", () => {
    expect(extractResiduals("## Now\nsomething")).toEqual([]);
  });

  it("returns an empty array when the section is present but empty", () => {
    expect(extractResiduals("## Residuals\n\n## Now\nsomething")).toEqual([]);
  });

  it("stops at the next heading when ## Residuals precedes ## Now", () => {
    expect(
      extractResiduals("## Residuals\n- only this risk\n## Now\nnot a residual"),
    ).toEqual(["only this risk"]);
  });
});

function writeStatus(root: string, dir: string, raw: string) {
  const folder = join(root, dir, ".jax-os");
  mkdirSync(folder, { recursive: true });
  const path = join(folder, "status.md");
  writeFileSync(path, raw);
  return path;
}

function exactCapStatus(): string {
  const head = "---\nstage: build\nproject: Cap\n---\n\n## Now\nHi\n";
  return head + "x".repeat(STATUS_CAP - Buffer.byteLength(head));
}

describe("bounded status reads", () => {
  let root: string;
  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  it("accepts an exact-cap regular file and uses the descriptor mtime", () => {
    root = mkdtempSync(join(tmpdir(), "jax-status-"));
    writeStatus(root, "ok", exactCapStatus());
    const scan = getProjects(root);
    expect(scan.skipped).toBe(0);
    expect(scan.projects).toHaveLength(1);
    expect(scan.projects[0].name).toBe("Cap");
    expect(scan.projects[0].statusMtime).toMatch(/^\d{4}-/);
  });

  it("skips a cap+1 scan entry and does not treat truncation as content", () => {
    root = mkdtempSync(join(tmpdir(), "jax-status-"));
    writeStatus(root, "big", exactCapStatus() + "y");
    expect(getProjects(root)).toEqual({ projects: [], skipped: 1, reposRoot: root });
  });

  it("skips malformed frontmatter and does not skip a missing status file", () => {
    root = mkdtempSync(join(tmpdir(), "jax-status-"));
    writeStatus(root, "bad", "not frontmatter\n");
    mkdirSync(join(root, "empty"));
    const scan = getProjects(root);
    expect(scan.projects).toEqual([]);
    expect(scan.skipped).toBe(1);
  });

  it("rejects stat-below-cap then growth without returning truncated text", () => {
    root = mkdtempSync(join(tmpdir(), "jax-status-"));
    const path = writeStatus(root, "grow", "---\nstage: build\n---\n");
    let checks = 0;
    expect(() => readStatusBounded(path, {
      openSync,
      fstatSync: (fd) => {
        const stat = fstatSync(fd);
        checks += 1;
        if (checks > 1) return Object.assign(Object.create(stat), { size: STATUS_CAP + 1 });
        return stat;
      },
      readSync,
      closeSync,
    })).toThrow();
  });

  it("closes the descriptor when open succeeds and fstat fails", () => {
    root = mkdtempSync(join(tmpdir(), "jax-status-"));
    const path = writeStatus(root, "err", "---\nstage: build\n---\n");
    const closed: number[] = [];
    expect(() => readStatusBounded(path, {
      openSync,
      fstatSync: () => { throw new Error("fstat failed"); },
      readSync,
      closeSync: (fd) => { closed.push(fd); closeSync(fd); },
    })).toThrow("fstat failed");
    expect(closed).toHaveLength(1);
  });
});

// ---- MOA-467 Task 4: linked worktrees are filtered out before their status is parsed,
// via git's own administrative metadata (gitdir/commondir shape), never folder names ----

function runGit(cwd: string, ...args: string[]): void {
  execFileSync("git", args, { cwd, stdio: "pipe" });
}

function initRepo(dir: string): void {
  mkdirSync(dir, { recursive: true });
  runGit(dir, "init", "-b", "main");
  runGit(dir, "config", "user.email", "t@t.test");
  runGit(dir, "config", "user.name", "t");
  writeFileSync(join(dir, "README"), "x\n");
  writeFileSync(join(dir, ".gitignore"), ".jax-os/\n");
  runGit(dir, "add", ".");
  runGit(dir, "commit", "-m", "init");
}

function addWorktree(root: string, control: string, branch: string, name: string): string {
  const path = join(root, name);
  runGit(control, "worktree", "add", "-b", branch, path, "main");
  return path;
}

describe("linked worktree metadata parsing (MOA-467 Task 4)", () => {
  it("parses the gitdir ref line out of a linked worktree's .git file", () => {
    expect(parseGitdirRef("gitdir: /home/rafa/repos/demo/.git/worktrees/w1\n"))
      .toBe("/home/rafa/repos/demo/.git/worktrees/w1");
    expect(parseGitdirRef("gitdir: /path with space/.git/worktrees/w2\r\n"))
      .toBe("/path with space/.git/worktrees/w2");
  });

  it("returns null for content without a gitdir ref line", () => {
    expect(parseGitdirRef("")).toBeNull();
    expect(parseGitdirRef("# comment\n")).toBeNull();
    expect(parseGitdirRef("gitdir:\n")).toBeNull();
  });

  it("parses a commondir path (relative or absolute), ignoring blank lines", () => {
    expect(parseCommondir("../..\n")).toBe("../..");
    expect(parseCommondir("/home/rafa/repos/demo/.git\n")).toBe("/home/rafa/repos/demo/.git");
    expect(parseCommondir("../../..\r\n")).toBe("../../..");
    expect(parseCommondir("")).toBeNull();
    expect(parseCommondir("\n\n")).toBeNull();
  });

  it("classifies a gitdir as linked only when its commondir resolves to the admin parent", () => {
    // git's own shapes: the admin dir lives at <common-dir>/worktrees/<name>, and its
    // commondir file (relative to the gitdir itself) resolves back to <common-dir> --
    // for an ordinary control repo AND for one using --separate-git-dir.
    expect(isLinkedWorktreeGitdir("/home/rafa/repos/demo/.git/worktrees/w1", "../..")).toBe(true);
    expect(isLinkedWorktreeGitdir("/home/rafa/.cache/git/sep/worktrees/w1", "../..")).toBe(true);
    expect(isLinkedWorktreeGitdir("/home/rafa/repos/demo/.git/worktrees/w1",
      "/home/rafa/repos/demo/.git")).toBe(true);
    // a lookalike path with no commondir, or one pointing elsewhere, is not linked
    expect(isLinkedWorktreeGitdir("/home/rafa/repos/demo/.git/worktrees/w1", null)).toBe(false);
    expect(isLinkedWorktreeGitdir("/home/rafa/repos/demo/.git/worktrees/w1", "../../..")).toBe(false);
    // a gitdir that is not inside a `worktrees` admin dir is never linked
    expect(isLinkedWorktreeGitdir("/home/rafa/repos/demo/.git", "../..")).toBe(false);
    expect(isLinkedWorktreeGitdir("/home/rafa/.cache/git/separate", "../..")).toBe(false);
  });
});

describe("getProjects excludes linked worktrees (MOA-467 Task 4)", () => {
  let root: string;
  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  it("yields only the main repo card when two real linked worktrees carry valid status files", () => {
    root = mkdtempSync(join(tmpdir(), "jax-wt-"));
    const control = join(root, "demo");
    initRepo(control);
    addWorktree(root, control, "feat/w1", "demo-w1");
    addWorktree(root, control, "feat/w2", "demo-w2");
    writeStatus(root, "demo", FULL);
    writeStatus(root, "demo-w1", FULL);
    writeStatus(root, "demo-w2", FULL);
    const scan = getProjects(root);
    expect(scan.skipped).toBe(0);
    expect(scan.projects).toHaveLength(1);
    expect(scan.projects[0].dir).toBe("demo");
  });

  it("excludes a real linked worktree whose .git gitdir ref is relative, run from a different cwd", () => {
    root = mkdtempSync(join(tmpdir(), "jax-wt-"));
    const control = join(root, "demo");
    initRepo(control);
    const wt = addWorktree(root, control, "feat/w1", "demo-w1");
    // git writes an absolute gitdir line; a VALID relative one must resolve against the
    // checkout dir containing the .git file, never the collector's own cwd.
    const admin = parseGitdirRef(readFileSync(join(wt, ".git"), "utf8"));
    expect(admin).not.toBeNull();
    writeFileSync(join(wt, ".git"), `gitdir: ${relative(wt, admin as string)}\n`);
    writeStatus(root, "demo", FULL);
    writeStatus(root, "demo-w1", FULL);
    const elsewhere = mkdtempSync(join(tmpdir(), "jax-cwd-"));
    const before = process.cwd();
    try {
      process.chdir(elsewhere);
      const scan = getProjects(root);
      expect(scan.skipped).toBe(0);
      expect(scan.projects).toHaveLength(1);
      expect(scan.projects[0].dir).toBe("demo");
    } finally {
      process.chdir(before);
      rmSync(elsewhere, { recursive: true, force: true });
    }
  });

  it("never promotes a linked worktree whose control repo has no valid status", () => {
    root = mkdtempSync(join(tmpdir(), "jax-wt-"));
    const control = join(root, "demo");
    initRepo(control);
    addWorktree(root, control, "feat/w1", "demo-w1");
    writeStatus(root, "demo-w1", FULL); // valid worktree card, control repo has none
    expect(getProjects(root)).toEqual({ projects: [], skipped: 0, reposRoot: root });
  });

  it("excludes linked worktrees even when the control repo uses --separate-git-dir", () => {
    // The admin dir is not `<control>/.git/worktrees/<name>` here -- it is
    // `<separate-gitdir>/worktrees/<name>` -- so a `.git`-name shape check would miss
    // it. The commondir relationship is what proves the linkage.
    root = mkdtempSync(join(tmpdir(), "jax-wt-"));
    const control = join(root, "sep");
    mkdirSync(control, { recursive: true });
    const gitdir = join(root, "sep-gitdir");
    runGit(control, "init", "-b", "main", `--separate-git-dir=${gitdir}`);
    runGit(control, "config", "user.email", "t@t.test");
    runGit(control, "config", "user.name", "t");
    writeFileSync(join(control, "README"), "x\n");
    runGit(control, "add", "README");
    runGit(control, "commit", "-m", "init");
    runGit(control, "worktree", "add", "-b", "feat/w1", join(root, "sep-w1"), "main");
    writeStatus(root, "sep", FULL);
    writeStatus(root, "sep-w1", FULL);
    const scan = getProjects(root);
    expect(scan.skipped).toBe(0);
    expect(scan.projects).toHaveLength(1);
    expect(scan.projects[0].dir).toBe("sep");
  });

  it("treats a lookalike .git/worktrees metadata path without a linked relationship as independent", () => {
    // The referenced gitdir is shaped like a worktree admin dir but carries no
    // commondir -- a path, not a relationship. The entry keeps its own card.
    root = mkdtempSync(join(tmpdir(), "jax-wt-"));
    mkdirSync(join(root, "fakearea", ".git", "worktrees", "w1"), { recursive: true });
    mkdirSync(join(root, "odd"), { recursive: true });
    writeFileSync(
      join(root, "odd", ".git"),
      `gitdir: ${join(root, "fakearea", ".git", "worktrees", "w1")}\n`,
    );
    writeStatus(root, "odd", FULL);
    const scan = getProjects(root);
    expect(scan.skipped).toBe(0);
    expect(scan.projects).toHaveLength(1);
    expect(scan.projects[0].dir).toBe("odd");
  });

  it("keeps a separate-git-dir repo independent (a .git file is not always a worktree)", () => {
    root = mkdtempSync(join(tmpdir(), "jax-wt-"));
    const repo = join(root, "sep");
    mkdirSync(repo, { recursive: true });
    // `--separate-git-dir` also writes a `.git` FILE, but its gitdir is not a
    // `.git/worktrees/<name>` path -- the repo must still get its own card.
    const gitdir = join(root, "sep-gitdir");
    runGit(repo, "init", "-b", "main", `--separate-git-dir=${gitdir}`);
    writeFileSync(join(repo, "README"), "x\n");
    runGit(repo, "config", "user.email", "t@t.test");
    runGit(repo, "config", "user.name", "t");
    runGit(repo, "add", ".");
    runGit(repo, "commit", "-m", "init");
    writeStatus(root, "sep", "---\nstage: spec\n---\n");
    const scan = getProjects(root);
    expect(scan.skipped).toBe(0);
    expect(scan.projects).toHaveLength(1);
    expect(scan.projects[0].dir).toBe("sep");
  });

  it("treats a garbage .git text file as an independent repo, not a worktree", () => {
    root = mkdtempSync(join(tmpdir(), "jax-wt-"));
    mkdirSync(join(root, "odd", ".jax-os"), { recursive: true });
    writeFileSync(join(root, "odd", ".git"), "gitdir: not really a gitdir\n");
    writeFileSync(join(root, "odd", ".jax-os", "status.md"), "---\nstage: build\n---\n");
    const scan = getProjects(root);
    expect(scan.skipped).toBe(0);
    expect(scan.projects).toHaveLength(1);
    expect(scan.projects[0].dir).toBe("odd");
  });
});

// ---- MOA-469 C3: canonical owner resolution (mirrors the hook's _git_owner) ----
describe("canonicalOwner / ownerProjectName (MOA-469 C3)", () => {
  let root: string;
  afterEach(() => {
    if (root) rmSync(root, { recursive: true, force: true });
  });

  const probeReturning = (map: Record<string, { ok: boolean; stdout: string } | null>): GitProbe =>
    async (args) => map[args.join(" ")] ?? null;

  it("resolves an ordinary checkout to its toplevel basename", async () => {
    const probe = probeReturning({
      "rev-parse --path-format=absolute --git-dir --git-common-dir": { ok: true, stdout: "/repo/.git\n/repo/.git\n" },
      "rev-parse --show-toplevel": { ok: true, stdout: "/repo\n" },
    });
    expect(await canonicalOwner("/repo/src/x", probe)).toBe("/repo");
    expect(await ownerProjectName("/repo/src/x", probe)).toBe("repo");
  });

  it("resolves a --separate-git-dir checkout to its own toplevel, not the gitdir", async () => {
    const probe = probeReturning({
      "rev-parse --path-format=absolute --git-dir --git-common-dir": { ok: true, stdout: "/admin/repo-gitdir\n/admin/repo-gitdir\n" },
      "rev-parse --show-toplevel": { ok: true, stdout: "/repo\n" },
    });
    expect(await ownerProjectName("/repo", probe)).toBe("repo");
  });

  it("resolves a linked worktree to the MAIN checkout, never the worktree's own name", async () => {
    const probe = probeReturning({
      "rev-parse --path-format=absolute --git-dir --git-common-dir": { ok: true, stdout: "/main/.git/worktrees/wt\n/main/.git\n" },
      "worktree list --porcelain": { ok: true, stdout: "worktree /main\nHEAD abc\nbranch refs/heads/main\n" },
    });
    expect(await canonicalOwner("/main-wt/sub", probe)).toBe("/main");
    expect(await ownerProjectName("/main-wt/sub", probe)).toBe("main");
  });

  it("returns null outside any repository and for empty cwd (no basename guess)", async () => {
    const probe = probeReturning({});
    expect(await canonicalOwner("/not/a/repo", probe)).toBeNull();
    expect(await ownerProjectName("/not/a/repo", probe)).toBeNull();
    expect(await ownerProjectName("", probe)).toBeNull();
  });

  it("a failed or timed-out git probe is an unresolved owner (null), never a basename guess", async () => {
    const failed: GitProbe = async () => null;
    expect(await canonicalOwner("/repo", failed)).toBeNull();
    expect(await ownerProjectName("/repo", failed)).toBeNull();
  });

  it("real git: resolves a subdirectory and a linked worktree to the control repo", async () => {
    root = mkdtempSync(join(tmpdir(), "jax-own-"));
    const control = join(root, "demo");
    initRepo(control);
    const wt = addWorktree(root, control, "feat/w1", "demo-w1");
    mkdirSync(join(control, "src", "deep"), { recursive: true });
    const realProbe: GitProbe = async (args, cwd) => {
      try {
        return { ok: true, stdout: execFileSync("git", args, { cwd, encoding: "utf8" }) };
      } catch {
        return null;
      }
    };
    expect(await ownerProjectName(join(control, "src", "deep"), realProbe)).toBe("demo");
    expect(await ownerProjectName(wt, realProbe)).toBe("demo");
  });
});

describe("Phase 3 — archived (spec §10, same pattern as flag/gate)", () => {
  it("archived: true parses to true; absent or any other value parses to false", () => {
    expect(parseStatusMd("p1", "---\nproject: P\nstage: build\narchived: true\n---\n", "2026-09-19T00:00:00.000Z")?.archived).toBe(true);
    expect(parseStatusMd("p1", "---\nproject: P\nstage: build\n---\n", "2026-09-19T00:00:00.000Z")?.archived).toBe(false);
    expect(parseStatusMd("p1", "---\nproject: P\nstage: build\narchived: yes\n---\n", "2026-09-19T00:00:00.000Z")?.archived).toBe(false);
  });
});
