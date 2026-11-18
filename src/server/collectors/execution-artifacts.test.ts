import { afterEach, describe, expect, it } from "vitest";
import { mkdirSync, mkdtempSync, readdirSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { BudgetError, InspectionBudget, inspectArtifacts, type ArtifactDeps } from "./execution-artifacts";
import type { GitProbe } from "./projects";

const RUN = "0123456789ab";
const roots: string[] = [];

function tempRoot(): string {
  const root = mkdtempSync(join(tmpdir(), "jaxos-artifacts-"));
  roots.push(root);
  return root;
}

afterEach(() => {
  while (roots.length) rmSync(roots.pop()!, { recursive: true, force: true });
});

function project(root: string, rel: string, o: { status?: boolean; manifest?: unknown; report?: string; tests?: string } = {}): string {
  const abs = join(root, rel);
  mkdirSync(join(abs, ".git"), { recursive: true });
  if (o.status) writeFileSync(join(abs, ".jax-os", "status.md"), "---\nproject: X\nstage: build\n---\n\n## Now\nworking\n");
  mkdirSync(join(abs, ".local", "runs", RUN), { recursive: true });
  if (o.manifest !== undefined) {
    writeFileSync(join(abs, ".local", "runs", RUN, "manifest.json"), typeof o.manifest === "string" ? o.manifest : JSON.stringify(o.manifest));
  }
  mkdirSync(join(abs, ".local", "reports"), { recursive: true });
  if (o.report !== undefined) writeFileSync(join(abs, ".local", "reports", `${RUN}.md`), o.report);
  if (o.tests !== undefined) writeFileSync(join(abs, ".local", "reports", `${RUN}.tests.txt`), o.tests);
  return abs;
}

function manifestFor(abs: string, extra: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    run_id: RUN, project: abs.split("/").at(-1), repo: abs, kind: "build", role: "builder",
    branch: "feat/x", ...extra,
  };
}

function probeReturning(owner: string): GitProbe {
  return async (args) => {
    if (args[0] === "rev-parse") return { ok: true, stdout: "/admin/wt\n/control/.git\n" };
    if (args[0] === "worktree") return { ok: true, stdout: `worktree ${owner}\nHEAD abc\n` };
    return { ok: true, stdout: "" };
  };
}

function deps(root: string, extra: Partial<ArtifactDeps> = {}): ArtifactDeps {
  return { reposRoot: root, now: () => Date.parse("2026-11-18T12:00:00Z"), ...extra };
}

describe("inspectArtifacts — roots and links", () => {
  it("resolves retained control reports for a top-level destination", async () => {
    const root = tempRoot();
    const abs = project(root, "alpha", {
      manifest: manifestFor(join(root, "alpha")), report: "# report", tests: "ok",
    });
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts).toMatchObject({
      destination: "alpha", runId: RUN, branch: "feat/x", branchSource: "manifest",
      report: { rel: `alpha/.local/reports/${RUN}.md` },
      tests: { rel: `alpha/.local/reports/${RUN}.tests.txt` },
      warnings: [],
    });
    expect(abs).toBe(join(root, "alpha"));
  });

  it("uses the discovered nested path, not the bare destination name", async () => {
    const root = tempRoot();
    const abs = project(root, "group/child", {
      manifest: manifestFor(join(root, "group/child")), report: "r",
    });
    const [facts] = await inspectArtifacts([{ destination: "child", runId: RUN }], deps(root));
    expect(facts.report?.rel).toBe(`group/child/.local/reports/${RUN}.md`);
    expect(abs).toBe(join(root, "group/child"));
  });

  it("treats a same-name collision as root-unavailable", async () => {
    const root = tempRoot();
    project(root, "dup", { report: "r" });
    project(root, "other/dup", { report: "r" });
    const [facts] = await inspectArtifacts([{ destination: "dup", runId: RUN }], deps(root));
    expect(facts.report).toBeNull();
    expect(facts.warnings).toEqual(["root-unavailable"]);
  });

  it("finds a status-only project and rejects an invalid status", async () => {
    const root = tempRoot();
    mkdirSync(join(root, "status-only", ".jax-os"), { recursive: true });
    writeFileSync(join(root, "status-only", ".jax-os", "status.md"), "---\nproject: S\nstage: build\n---\n## Now\nx\n");
    mkdirSync(join(root, "broken-status", ".jax-os"), { recursive: true });
    writeFileSync(join(root, "broken-status", ".jax-os", "status.md"), "not frontmatter");
    const facts = await inspectArtifacts([
      { destination: "status-only", runId: RUN },
      { destination: "broken-status", runId: RUN },
    ], deps(root));
    expect(facts[0].warnings).toContain("manifest-missing");
    expect(facts[0].warnings).not.toContain("root-unavailable");
    expect(facts[1].warnings).toEqual(["root-unavailable"]);
  });

  it("skips a linked worktree root", async () => {
    const root = tempRoot();
    const admin = join(root, "control", ".git", "worktrees", "wt");
    mkdirSync(admin, { recursive: true });
    writeFileSync(join(admin, "commondir"), "../..\n");
    const wt = join(root, "wt");
    mkdirSync(join(wt, ".local", "reports"), { recursive: true });
    writeFileSync(join(wt, ".git"), `gitdir: ${admin}\n`);
    writeFileSync(join(wt, ".local", "reports", "x"), "");
    const [facts] = await inspectArtifacts([{ destination: "wt", runId: RUN }], deps(root));
    expect(facts.warnings).toEqual(["root-unavailable"]);
  });

  it("reports missing artifacts without losing a valid branch", async () => {
    const root = tempRoot();
    project(root, "alpha", { manifest: manifestFor(join(root, "alpha")) });
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts.branch).toBe("feat/x");
    expect(facts.report).toBeNull();
    expect(facts.tests).toBeNull();
    expect(facts.warnings).toEqual(expect.arrayContaining(["report-missing", "tests-missing"]));
  });

  it("rejects an oversized or malformed manifest and never fabricates a branch", async () => {
    const root = tempRoot();
    project(root, "big", { manifest: JSON.stringify({ run_id: RUN, project: "big", repo: join(root, "big"), kind: "build", role: "builder", branch: "x".repeat(70 * 1024) }) });
    writeFileSync(join(root, "big", ".local", "runs", RUN, "manifest.json"), `{"run_id":"${RUN}","pad":"${"x".repeat(70 * 1024)}"}`);
    project(root, "bad", { manifest: "{not-json" });
    const facts = await inspectArtifacts([
      { destination: "big", runId: RUN }, { destination: "bad", runId: RUN },
    ], deps(root));
    expect(facts[0].branch).toBeNull();
    expect(facts[0].warnings).toContain("manifest-invalid");
    expect(facts[1].branch).toBeNull();
    expect(facts[1].warnings).toContain("manifest-invalid");
  });

  it("rejects a manifest bound to another run, destination or repository", async () => {
    const root = tempRoot();
    project(root, "alpha", { manifest: manifestFor(join(root, "alpha"), { run_id: "ffffffffffff" }) });
    project(root, "beta", { manifest: manifestFor(join(root, "beta"), { project: "alpha" }) });
    project(root, "gamma", { manifest: manifestFor(join(root, "alpha")) });
    const facts = await inspectArtifacts([
      { destination: "alpha", runId: RUN }, { destination: "beta", runId: RUN }, { destination: "gamma", runId: RUN },
    ], deps(root));
    expect(facts[0].warnings).toContain("manifest-invalid");
    expect(facts[1].warnings).toContain("manifest-invalid");
    expect(facts[2].warnings).toContain("manifest-invalid");
  });

  it("keeps branch provenance: target is a branch only for native build/diff manifests", async () => {
    const root = tempRoot();
    const specs = ".local/docs/specs/x.md";
    const cases: { name: string; manifest: Record<string, unknown> }[] = [
      { name: "build", manifest: manifestFor(join(root, "build"), { branch: undefined, target: "feat/branch-target" }) },
      { name: "diff", manifest: manifestFor(join(root, "diff"), { kind: "diff", role: "reviewer", branch: undefined, target: "feat/branch-target" }) },
      { name: "spec", manifest: manifestFor(join(root, "spec"), { kind: "spec", role: "reviewer", branch: undefined, target: specs }) },
      { name: "plan", manifest: manifestFor(join(root, "plan"), { kind: "plan", role: "reviewer", branch: undefined, target: specs }) },
    ];
    for (const item of cases) project(root, item.name, { manifest: item.manifest });
    const facts = await inspectArtifacts(cases.map((item) => ({ destination: item.name, runId: RUN })), deps(root));
    expect(facts[0]).toMatchObject({ branch: "feat/branch-target", branchSource: "manifest" });
    expect(facts[1]).toMatchObject({ branch: "feat/branch-target", branchSource: "manifest" });
    expect(facts[2]).toMatchObject({ branch: null, branchSource: null });
    expect(facts[3]).toMatchObject({ branch: null, branchSource: null });
  });

  it("prefers the recorded branch over a target fallback", async () => {
    const root = tempRoot();
    project(root, "alpha", { manifest: manifestFor(join(root, "alpha"), { branch: "feat/recorded", target: "feat/fallback" }) });
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts.branch).toBe("feat/recorded");
    expect(facts.branchSource).toBe("manifest");
  });

  it("never fabricates a branch from a missing or deleted manifest", async () => {
    const root = tempRoot();
    const abs = project(root, "alpha", { manifest: manifestFor(join(root, "alpha")) });
    rmSync(join(abs, ".local", "runs", RUN, "manifest.json"));
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts.branch).toBeNull();
    expect(facts.branchSource).toBeNull();
    expect(facts.warnings).toContain("manifest-missing");
  });

  it("rejects report symlinks, including an escaping target", async () => {
    const root = tempRoot();
    const abs = project(root, "alpha", { manifest: manifestFor(join(root, "alpha")) });
    const outside = join(root, "outside.md");
    writeFileSync(outside, "secret");
    symlinkSync(outside, join(abs, ".local", "reports", `${RUN}.md`));
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts.report).toBeNull();
    expect(facts.warnings).toContain("report-missing");
  });

  it("allows only an identity-bound worktree fallback for a build", async () => {
    const root = tempRoot();
    const abs = project(root, "alpha", { manifest: manifestFor(join(root, "alpha"), { worktree: join(root, "alpha-wt") }) });
    const wt = join(root, "alpha-wt");
    mkdirSync(join(wt, ".local", "reports"), { recursive: true });
    writeFileSync(join(wt, ".local", "reports", `${RUN}.md`), "fallback");
    const matched = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root, { gitProbe: probeReturning(abs) }));
    expect(matched[0].report?.rel).toBe(`alpha-wt/.local/reports/${RUN}.md`);

    const mismatched = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root, { gitProbe: probeReturning("/elsewhere") }));
    expect(mismatched[0].report).toBeNull();
    expect(mismatched[0].warnings).toContain("worktree-unavailable");
  });
});

describe("inspectArtifacts — bounds", () => {
  it("marks every row root-unavailable when the candidate budget is exhausted", async () => {
    const root = tempRoot();
    project(root, "a", { report: "x" });
    project(root, "b", { report: "x" });
    const facts = await inspectArtifacts([{ destination: "a", runId: RUN }], deps(root, { scanLimits: { maxCandidates: 1 } }));
    expect(facts[0].warnings).toEqual(["root-unavailable"]);
  });

  it("never reads an artifact through a path outside the canonical root", async () => {
    const root = tempRoot();
    const abs = project(root, "alpha", { manifest: manifestFor(join(root, "alpha")) });
    // `.local/reports` itself is a symlink to an outside directory: realpath containment refuses
    rmSync(join(abs, ".local", "reports"), { recursive: true, force: true });
    const elsewhere = join(root, "elsewhere");
    mkdirSync(elsewhere);
    writeFileSync(join(elsewhere, `${RUN}.md`), "outside");
    symlinkSync(elsewhere, join(abs, ".local", "reports"));
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts.report).toBeNull();
  });

  it("refuses a report-directory symlink that stays inside the same repository", async () => {
    const root = tempRoot();
    const abs = project(root, "alpha", { manifest: manifestFor(join(root, "alpha")) });
    rmSync(join(abs, ".local", "reports"), { recursive: true, force: true });
    mkdirSync(join(abs, "elsewhere"));
    writeFileSync(join(abs, "elsewhere", `${RUN}.md`), "same-repo");
    symlinkSync(join(abs, "elsewhere"), join(abs, ".local", "reports"));
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts.report).toBeNull();
    expect(facts.warnings).toContain("report-missing");
  });

  it("refuses a file symlink whose target is another file of the same repository", async () => {
    const root = tempRoot();
    const abs = project(root, "alpha", { manifest: manifestFor(join(root, "alpha")) });
    writeFileSync(join(abs, ".local", "other.md"), "same-repo target");
    symlinkSync(join(abs, ".local", "other.md"), join(abs, ".local", "reports", `${RUN}.md`));
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts.report).toBeNull();
  });

  it("treats an incomplete nested scan as unavailable and never caches it", async () => {
    const root = tempRoot();
    project(root, "group/alpha", { report: "r" });
    const listing = async (dir: string) => readdirSync(dir, { withFileTypes: true }).map((entry) => ({
      name: entry.name, isDirectory: entry.isDirectory(), isSymbolicLink: entry.isSymbolicLink(),
    }));
    const incomplete = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root, {
      listDir: async (dir) => {
        if (dir === join(root, "group")) throw new Error("injected filesystem failure");
        return listing(dir);
      },
    }));
    expect(incomplete[0].warnings).toEqual(["root-unavailable"]);

    const complete = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root, { listDir: listing }));
    expect(complete[0].warnings).not.toContain("root-unavailable");
    expect(complete[0].report?.rel).toBe(`group/alpha/.local/reports/${RUN}.md`);
  });

  it("resolves one owned destination despite an unowned nested folder of the same basename", async () => {
    const root = tempRoot();
    project(root, "alpha", { report: "r" });
    mkdirSync(join(root, "group", "alpha"), { recursive: true });
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root));
    expect(facts.warnings).not.toContain("root-unavailable");
    expect(facts.report?.rel).toBe(`alpha/.local/reports/${RUN}.md`);
  });

  it("stops new native reads at the earliest applicable deadline", async () => {
    const root = tempRoot();
    project(root, "alpha", { report: "x" });
    let now = 0;
    let calls = 0;
    const [facts] = await inspectArtifacts([{ destination: "alpha", runId: RUN }], deps(root, {
      budget: new InspectionBudget(4, 100, () => now),
      realpath: async (path) => {
        calls++;
        if (path === root) now = 200;
        return path;
      },
    }));
    expect(facts.warnings).toEqual(["root-unavailable"]);
    expect(calls).toBe(1);
  });
});

describe("InspectionBudget", () => {
  function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
    let resolve!: (value: T) => void;
    const promise = new Promise<T>((res) => { resolve = res; });
    return { promise, resolve };
  }

  it("never starts queued work that wakes after the deadline", async () => {
    let now = 0;
    const budget = new InspectionBudget(1, 50, () => now);
    let starts = 0;
    const gate = deferred<void>();
    const first = budget.run(async () => { starts++; await gate.promise; return 1; }, 10_000);
    await Promise.resolve();
    const queued = budget.run(async () => { starts++; return 2; }, 10_000);
    now = 100;
    gate.resolve();
    await expect(first).resolves.toBe(1);
    await expect(queued).rejects.toBeInstanceOf(BudgetError);
    expect(starts).toBe(1);
  });

  it("does not start scoped work after its deadline while queued on the parent", async () => {
    let now = 0;
    const budget = new InspectionBudget(1, 1_000, () => now);
    const gate = deferred<void>();
    let starts = 0;
    const first = budget.run(async () => { starts++; await gate.promise; }, 1_000);
    await Promise.resolve();
    const queued = budget.scope(100).run(async () => { starts++; return "late"; }, 1_000);
    now = 101;
    gate.resolve();
    await expect(first).resolves.toBeUndefined();
    await expect(queued).rejects.toBeInstanceOf(BudgetError);
    expect(starts).toBe(1);
  });

  it("keeps a timed-out hung native operation's slot until it actually settles", async () => {
    let now = 0;
    const budget = new InspectionBudget(1, 50, () => now);
    let starts = 0;
    const gate = deferred<void>();
    const first = await budget.run(async () => { starts++; await gate.promise; return "first"; }, 10).catch((e) => e);
    expect(first).toBeInstanceOf(BudgetError);
    const second = await budget.run(async () => { starts++; return "second"; }, 10).catch((e) => e);
    expect(second).toBeInstanceOf(BudgetError);
    expect(starts).toBe(1);
    gate.resolve();
    await budget.settled();
    await expect(budget.run(async () => { starts++; return "third"; }, 10)).resolves.toBe("third");
    expect(starts).toBe(2);
  });

  it("runs up to four operations and releases slots on settlement", async () => {
    const budget = new InspectionBudget(4, 10_000, () => 0);
    let starts = 0;
    const gates = Array.from({ length: 4 }, () => deferred<void>());
    const runs = gates.map((gate) => budget.run(async () => { starts++; await gate.promise; }, 10_000));
    await Promise.resolve();
    const fifthGate = deferred<void>();
    const fifth = budget.run(async () => { starts++; await fifthGate.promise; }, 10_000);
    await Promise.resolve();
    expect(starts).toBe(4);
    gates[0].resolve();
    await runs[0];
    await Promise.resolve();
    expect(starts).toBe(5);
    gates.slice(1).forEach((gate) => gate.resolve());
    fifthGate.resolve();
    await Promise.all([...runs, fifth]);
    await expect(budget.settled()).resolves.toBeUndefined();
  });

  it("does not leak a slot when the native function throws synchronously", async () => {
    const budget = new InspectionBudget(1, 10_000, () => 0);
    await expect(budget.run((() => { throw new Error("sync"); }) as unknown as () => Promise<void>, 10))
      .rejects.toThrow("sync");
    await expect(budget.run(async () => "ok", 10)).resolves.toBe("ok");
  });
});
