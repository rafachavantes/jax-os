import { NextResponse } from "next/server";
import { LIMITS, RUN_ID_RE } from "../../../../lib/workflow";
import { MutationRejected } from "../../../../lib/mutationOutcome";
import { readJsonCapped, requireSameOrigin } from "../../../../server/api";
import { runMutation } from "../../../../server/mutations";
import {
  CANCEL_TIMEOUT_MS, childDetails, jaxflow, lastLine, runChild, type ChildResult, type ChildRunner,
} from "../../../../server/collectors/workflow-spawn";

export type CancelRouteDeps = { run: ChildRunner };
const systemDeps: CancelRouteDeps = { run: runChild };

// Spec Decision 5: exactly `jaxflow cancel <run_id>` — no cwd (the verb is repo-independent), no
// caller identity, a 25s child timeout above cmd_cancel's own 15s wait.
export async function handleCancelPost(req: Request, deps: CancelRouteDeps = systemDeps) {
  const guard = requireSameOrigin(req);
  if (guard) return guard;
  const read = await readJsonCapped(req, LIMITS.requestBytes);
  if (!read.ok) return NextResponse.json({ ok: false, error: read.error });
  const body = read.value;
  if (!body || typeof body !== "object" || Array.isArray(body)) return NextResponse.json({ ok: false, error: "invalid payload" });
  const { run_id } = body as Record<string, unknown>;
  if (typeof run_id !== "string" || !RUN_ID_RE.test(run_id)) return NextResponse.json({ ok: false, error: "invalid payload" });

  const argv = ["cancel", run_id];
  let child: ChildResult | null = null;
  const result = await runMutation(
    { ts: new Date().toISOString(), kind: "workflow-cancel", argv },
    async () => {
      child = await jaxflow(argv, { timeoutMs: CANCEL_TIMEOUT_MS }, deps.run);
      if (child.ok) return child;
      if (child.code) throw new MutationRejected(child.code);
      throw new Error("jaxflow cancel did not exit 0");
    },
    (c) => childDetails(c),
    () => (child ? childDetails(child) : {}),
  );
  if (result.ok) return NextResponse.json({ ok: true, data: { run_id, outcome: lastLine(result.value.stdoutTail) } });
  const { value: _value, status: _status, ...failure } = result;
  return NextResponse.json(failure);
}
