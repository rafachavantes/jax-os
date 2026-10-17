import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type Database from "better-sqlite3";

let testDb: Database.Database;
vi.mock("../../../../server/db", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../../../../server/db")>();
  return { ...actual, getDb: () => testDb };
});

import { openDb } from "../../../../server/db";
import type { Project, ProjectScan } from "../../../../server/collectors/projects";
import { REPOS_ROOT } from "../../../../server/collectors/projects";
import { JAXFLOW_PROGRAM, JAXFLOW_SCRIPT, type ChildResult, type ChildRunner } from "../../../../server/collectors/workflow-spawn";
import { buildDispatchArgv, handleDispatchPost, type DispatchRouteDeps } from "./handler";

const P1: Project = {
  dir: "p1", name: "P1", stage: "build", gate: null, legacyGateField: false, builder: "codex", branch: "feat/x",
  updated: "2026-09-18T00:00:00-03:00", now: "", residuals: [], statusMtime: "2026-09-18T03:00:00.000Z", archived: false,
};
const SCAN: ProjectScan = { projects: [P1], skipped: 0, reposRoot: REPOS_ROOT };
const CWD = `${REPOS_ROOT}/p1`;
type Call = { file: string; args: string[]; opts: { cwd?: string; env?: Record<string, string>; timeoutMs: number } };
const ok = (stdout: string): ChildResult => ({ ok: true, code: null, exitCode: 0, stdoutTail: stdout, stderrTail: "", durationMs: 3 });
const refusal = (code: string): ChildResult => ({ ok: false, code, exitCode: 2, stdoutTail: "", stderrTail: `${code}\n`, durationMs: 3 });
const identity = (rel: string) => rel;

function deps(result: ChildResult, overrides: Partial<DispatchRouteDeps> = {}): { deps: DispatchRouteDeps; calls: Call[] } {
  const calls: Call[] = [];
  const run: ChildRunner = async (file, args, opts) => { calls.push({ file, args, opts }); return result; };
  // The default resolver fake joins like the real one but never touches the disk: project → CWD, document → CWD/<rel>.
  return { deps: { run, getProjects: () => SCAN, resolveWithin: (base, rel) => `${base}/${rel}`, ...overrides }, calls };
}
function post(body: unknown) {
  return new Request("http://localhost/api/workflow/dispatch", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
}
const rows = () => testDb.prepare("SELECT kind, ok, payload FROM mutations").all() as { kind: string; ok: number | null; payload: string }[];

const REVIEW = { project: "p1", command: "review", kind: "diff", target: "a1b2c3d4e5f6", focus: "check the merge guard" };
const SPEC_REVIEW = { project: "p1", command: "review", kind: "spec", target: ".local/docs/specs/x-spec.md" };
const BUILD = {
  project: "p1", command: "build", plan: ".local/docs/plans/2026-09-18-x-plan.md", phase: "moa-481-x", branch: "feat/x",
  whitelist: "src/,messages/", verify: "pnpm vitest run", build: "pnpm build", profile: "fallback",
};
const BUILD_ARGV = [
  "build", "--plan", `${CWD}/.local/docs/plans/2026-09-18-x-plan.md`, "--phase", "moa-481-x", "--branch", "feat/x",
  "--whitelist", "src/,messages/", "--verify", "pnpm vitest run", "--build", "pnpm build", "--fallback",
  "--from", "jaxos", "--no-callback",
];

describe("buildDispatchArgv (pure)", () => {
  it("review: --spec/--plan/--diff + optional --focus, always --from jaxos --no-callback; a run id is never resolved", () => {
    const neverCalled = () => { throw new Error("resolver must not run for a run id"); };
    expect(buildDispatchArgv(REVIEW, neverCalled)).toEqual(["review", "--diff", "a1b2c3d4e5f6", "--focus", "check the merge guard", "--from", "jaxos", "--no-callback"]);
    expect(buildDispatchArgv(SPEC_REVIEW, identity)).toEqual(["review", "--spec", ".local/docs/specs/x-spec.md", "--from", "jaxos", "--no-callback"]);
  });
  it("document paths go through the resolver (spec §7.1); a resolver throw is the allowlist refusal", () => {
    expect(buildDispatchArgv({ ...SPEC_REVIEW, kind: "plan" }, (rel) => `/abs/${rel}`)).toEqual(["review", "--plan", "/abs/.local/docs/specs/x-spec.md", "--from", "jaxos", "--no-callback"]);
    expect(buildDispatchArgv(SPEC_REVIEW, () => { throw new Error("outside allowlist"); })).toEqual({ error: "outside allowlist" });
    expect(buildDispatchArgv(BUILD, () => { throw new Error("outside allowlist"); })).toEqual({ error: "outside allowlist" });
  });
  it("build: every flag in the documented order; --build and --fallback only when given", () => {
    expect(buildDispatchArgv(BUILD, (rel) => `${CWD}/${rel}`)).toEqual(BUILD_ARGV);
    expect(buildDispatchArgv({ ...BUILD, build: undefined, profile: "default" }, identity)).toEqual([
      "build", "--plan", ".local/docs/plans/2026-09-18-x-plan.md", "--phase", "moa-481-x", "--branch", "feat/x",
      "--whitelist", "src/,messages/", "--verify", "pnpm vitest run", "--from", "jaxos", "--no-callback",
    ]);
  });
  it("rejects every §7.4 bound server-side, before the resolver runs", () => {
    const neverCalled = () => { throw new Error("resolver must not run for an invalid body"); };
    const bad = [
      { ...REVIEW, focus: "a\nb" }, { ...REVIEW, target: "not-a-run-id" }, { ...REVIEW, kind: "x" }, { ...SPEC_REVIEW, target: "a\nb.md" },
      { ...BUILD, phase: "bad phase" }, { ...BUILD, branch: "feat x" }, { ...BUILD, verify: "pnpm test\npnpm build" },
      { ...BUILD, build: "a\tb" }, { ...BUILD, whitelist: "" }, { ...BUILD, plan: "" }, { ...BUILD, profile: "other" },
      { ...BUILD, verify: "x".repeat(501) }, { project: "p1", command: "merge" },
    ];
    for (const b of bad) expect(buildDispatchArgv(b, neverCalled)).toEqual({ error: "invalid payload" });
  });
});

describe("POST /api/workflow/dispatch", () => {
  beforeEach(() => { testDb = openDb(":memory:"); });
  afterEach(() => { testDb.close(); });

  it("spawns with cwd = resolveWithin(REPOS_ROOT, project), JAXOS_CALLER_SESSION=jaxos, and returns the printed run id", async () => {
    const { deps: d, calls } = deps(ok("some notice\nb7c8d9e0f1a2\n"));
    const res = await handleDispatchPost(post(BUILD), d);
    expect(await res.json()).toEqual({ ok: true, data: { run_id: "b7c8d9e0f1a2" } });
    expect(calls).toHaveLength(1);
    expect(calls[0].file).toBe(JAXFLOW_PROGRAM);
    expect(calls[0].args).toEqual([JAXFLOW_SCRIPT, ...BUILD_ARGV]);
    expect(calls[0].opts).toEqual({ cwd: CWD, env: { JAXOS_CALLER_SESSION: "jaxos" }, timeoutMs: 15 * 60_000 });
    const [row] = rows();
    expect(row.kind).toBe("workflow-dispatch");
    expect(JSON.parse(row.payload)).toMatchObject({ project: "p1", command: "build", argv: BUILD_ARGV, outcome: "done", exitCode: 0 });
  });

  it("refuses an unknown project before any spawn or path resolution, and an unavailable scan is a 200 envelope (round-1 F4), never a 5xx", async () => {
    const resolveWithin = vi.fn((base: string, rel: string) => `${base}/${rel}`);
    const { deps: d, calls } = deps(ok("x"), { resolveWithin });
    const res = await handleDispatchPost(post({ ...REVIEW, project: "nope" }), d);
    expect(await res.json()).toEqual({ ok: false, error: "unknown-project" });
    expect(resolveWithin).not.toHaveBeenCalled();
    expect(calls).toEqual([]);
    expect(rows()).toEqual([]);
    const scanFail = deps(ok("x"), { getProjects: () => { throw new Error("ENOENT: repos root missing"); } });
    const res2 = await handleDispatchPost(post(REVIEW), scanFail.deps);
    expect(res2.status).toBe(200);
    expect(await res2.json()).toEqual({ ok: false, error: "project scan unavailable" });
    expect(scanFail.calls).toEqual([]);
    expect(rows()).toEqual([]);
  });

  it("refuses a project dir or a document outside the allowlist before any spawn", async () => {
    const escaped = deps(ok("x"), { resolveWithin: () => { throw new Error("outside allowlist"); } });
    expect(await (await handleDispatchPost(post(REVIEW), escaped.deps)).json()).toEqual({ ok: false, error: "outside allowlist" });
    const docOnly = deps(ok("x"), { resolveWithin: (base, rel) => { if (rel === "p1") return `${base}/${rel}`; throw new Error("outside allowlist"); } });
    expect(await (await handleDispatchPost(post(SPEC_REVIEW), docOnly.deps)).json()).toEqual({ ok: false, error: "outside allowlist" });
    expect(escaped.calls).toEqual([]);
    expect(docOnly.calls).toEqual([]);
    expect(rows()).toEqual([]);
  });

  it("a review --focus or a build --verify carrying a newline never reaches the spawn fake", async () => {
    const { deps: d, calls } = deps(ok("x"));
    expect(await (await handleDispatchPost(post({ ...REVIEW, focus: "a\nb" }), d)).json()).toEqual({ ok: false, error: "invalid payload" });
    expect(await (await handleDispatchPost(post({ ...BUILD, verify: "a\rb" }), d)).json()).toEqual({ ok: false, error: "invalid payload" });
    expect(calls).toEqual([]);
  });

  it("a jaxflow refusal is mutation-rejected with the code; no run id → unconfirmed; audit failure → never spawns", async () => {
    const r1 = await handleDispatchPost(post(REVIEW), deps(refusal("review-running")).deps);
    expect(await r1.json()).toMatchObject({ ok: false, code: "mutation-rejected", error: "review-running" });
    expect(JSON.parse(rows()[0].payload)).toMatchObject({ outcome: "failed", exitCode: 2, stderrTail: "review-running\n" });

    const r2 = await handleDispatchPost(post(REVIEW), deps(ok("no id printed\n")).deps);
    expect(await r2.json()).toMatchObject({ ok: false, code: "mutation-unconfirmed" });
    expect(JSON.parse(rows()[1].payload)).toMatchObject({ outcome: "abandoned", exitCode: 0 });

    testDb.exec("CREATE TRIGGER fail_ins BEFORE INSERT ON mutations BEGIN SELECT RAISE(FAIL, 'fixture'); END");
    const { deps: d, calls } = deps(ok("x"));
    expect(await (await handleDispatchPost(post(REVIEW), d)).json()).toMatchObject({ ok: false, code: "audit-unavailable" });
    expect(calls).toEqual([]);
  });

  it("round-1 F1: route.ts exports a one-argument POST", async () => {
    const { POST } = await import("./route");
    expect(POST.length).toBe(1);
  });
});
