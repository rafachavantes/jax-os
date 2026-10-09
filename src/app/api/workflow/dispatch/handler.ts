import { NextResponse } from "next/server";
import {
  LIMITS, RUN_ID_RE, validOptionalFocus, validPhaseToken, validRefName, validSingleLineCommand,
} from "../../../../lib/workflow";
import { MutationRejected } from "../../../../lib/mutationOutcome";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { validBasename, validRel } from "../../../../server/inputLimits";
import { runMutation } from "../../../../server/mutations";
import { getProjects, REPOS_ROOT, type ProjectScan } from "../../../../server/collectors/projects";
import { resolveWithin } from "../../../../server/collectors/files";
import { redactSecrets } from "../../../../server/collectors/redact";
import {
  ACTION_TIMEOUT_MS, childDetails, jaxflow, lastLine, runChild, type ChildResult, type ChildRunner,
} from "../../../../server/collectors/workflow-spawn";

export type DispatchRouteDeps = {
  run: ChildRunner;
  getProjects: () => ProjectScan;
  resolveWithin: (base: string, rel: string) => string;
};
const systemDeps: DispatchRouteDeps = { run: runChild, getProjects: () => getProjects(), resolveWithin };

const REVIEW_KINDS = ["spec", "plan", "diff"] as const;
const INVALID = { error: "invalid payload" } as const;
const OUTSIDE = { error: "outside allowlist" } as const;

// A repo-relative document path handed to the resolver: non-empty, ≤4096 bytes, no NUL, single line.
function validDocPath(v: unknown): v is string {
  return validRel(v) && !/[\n\r\t]/.test(v);
}

// Spec §7.1 / §7.3 / §7.4: the exact argv list for `review` or `build`, always ending in
// `--from jaxos --no-callback`. `resolveDoc` canonicalizes a document path inside the project
// (the handler passes `resolveWithin(cwd, rel)`; a throw is the allowlist refusal) — a diff
// review's run id never goes through it. Pure and exported so the argv shape is unit-tested alone.
export function buildDispatchArgv(body: unknown, resolveDoc: (rel: string) => string): string[] | { error: string } {
  if (!body || typeof body !== "object" || Array.isArray(body)) return INVALID;
  const b = body as Record<string, unknown>;
  if (b.command === "review") {
    if (!(REVIEW_KINDS as readonly unknown[]).includes(b.kind)) return INVALID;
    const kind = b.kind as (typeof REVIEW_KINDS)[number];
    if (!validOptionalFocus(b.focus)) return INVALID;
    let target: string;
    if (kind === "diff") {
      if (typeof b.target !== "string" || !RUN_ID_RE.test(b.target)) return INVALID;
      target = b.target;
    } else {
      if (!validDocPath(b.target)) return INVALID;
      try { target = resolveDoc(b.target); } catch { return OUTSIDE; }
    }
    return ["review", `--${kind}`, target, ...(b.focus !== undefined ? ["--focus", b.focus as string] : []), "--from", "jaxos", "--no-callback"];
  }
  if (b.command === "build") {
    if (!validDocPath(b.plan) || !validPhaseToken(b.phase) || !validRefName(b.branch)) return INVALID;
    if (!validSingleLineCommand(b.whitelist, LIMITS.whitelist) || !validSingleLineCommand(b.verify, LIMITS.verify)) return INVALID;
    if (b.build !== undefined && !validSingleLineCommand(b.build, LIMITS.build)) return INVALID;
    const profile = b.profile === undefined ? "default" : b.profile;
    if (profile !== "default" && profile !== "fallback") return INVALID;
    let plan: string;
    try { plan = resolveDoc(b.plan); } catch { return OUTSIDE; }
    return [
      "build", "--plan", plan, "--phase", b.phase, "--branch", b.branch, "--whitelist", b.whitelist, "--verify", b.verify,
      ...(b.build !== undefined ? ["--build", b.build as string] : []),
      ...(profile === "fallback" ? ["--fallback"] : []),
      "--from", "jaxos", "--no-callback",
    ];
  }
  return INVALID;
}

export async function handleDispatchPost(req: Request, deps: DispatchRouteDeps = systemDeps) {
  const guard = requireSameOrigin(req);
  if (guard) return guard;
  const read = await readJsonCapped(req, LIMITS.requestBytes);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value;
  if (!body || typeof body !== "object" || Array.isArray(body)) return NextResponse.json({ ok: false, error: "invalid payload" });
  const { project, command } = body as Record<string, unknown>;
  // Round-1 F4: a scan failure must return the agreed 200 {ok:false} envelope, never a 5xx.
  let scan: ProjectScan;
  try {
    scan = deps.getProjects();
  } catch {
    return NextResponse.json({ ok: false, error: "project scan unavailable" });
  }
  // Spec §7.1 (round-3 F1): `project` is the card's directory name under ~/repos, validated
  // against the CURRENT scan before any path is resolved or any other field is read.
  if (!validBasename(project) || !scan.projects.some((p) => p.dir === project)) {
    return NextResponse.json({ ok: false, error: "unknown-project" });
  }
  let cwd: string;
  try {
    cwd = deps.resolveWithin(REPOS_ROOT, project);
  } catch {
    return NextResponse.json({ ok: false, error: "outside allowlist" });
  }
  const argv = buildDispatchArgv(body, (rel) => deps.resolveWithin(cwd, rel));
  if (!Array.isArray(argv)) return NextResponse.json({ ok: false, ...argv });

  let child: ChildResult | null = null;
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "workflow-dispatch", project, command, argv: argv.map(redactSecrets) },
    async () => {
      // JAXOS_CALLER_SESSION is the ONLY env var set on the child (spec §4, Decision 6) — merged
      // over process.env by the helper so PATH/HOME still reach python3 and git.
      child = await jaxflow(argv, { cwd, env: { JAXOS_CALLER_SESSION: "jaxos" }, timeoutMs: ACTION_TIMEOUT_MS }, deps.run);
      if (!child.ok) {
        if (child.code) throw new MutationRejected(child.code);
        throw new Error("jaxflow dispatch did not exit 0");
      }
      const runId = lastLine(child.stdoutTail);
      if (!RUN_ID_RE.test(runId)) throw new Error("jaxflow printed no run id");
      return runId;
    },
    () => (child ? childDetails(child) : {}),
    () => (child ? childDetails(child) : {}),
  );
  if (result.ok) return NextResponse.json({ ok: true, data: { run_id: result.value } });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
