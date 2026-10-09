import { describe, expect, it } from "vitest";
import { LIMITS } from "../../lib/workflow";
import { JAXFLOW_PROGRAM, JAXFLOW_SCRIPT, childDetails, jaxflow, lastLine, runChild, type ChildRunner } from "./workflow-spawn";

const node = process.execPath;
const GHP = "ghp_" + "abcdefghijklmnopqrstuvwxyz0123456789";

describe("runChild — never throws, one result shape for every outcome", () => {
  it("exit 0: ok, exitCode 0, stdout captured", async () => {
    const r = await runChild(node, ["-e", "process.stdout.write('a1b2c3d4e5f6\\n')"], { timeoutMs: 10_000 });
    expect(r).toMatchObject({ ok: true, code: null, exitCode: 0, stdoutTail: "a1b2c3d4e5f6\n", stderrTail: "" });
    expect(r.durationMs).toBeGreaterThanOrEqual(0);
  });

  it("exit 2: the first non-empty stderr line is the refusal code", async () => {
    const r = await runChild(node, ["-e", "process.stderr.write('already-finished\\nhint: x\\n'); process.exit(2)"], { timeoutMs: 10_000 });
    expect(r).toMatchObject({ ok: false, code: "already-finished", exitCode: 2, stderrTail: "already-finished\nhint: x\n" });
  });

  it("other non-zero exits carry no refusal code", async () => {
    const r = await runChild(node, ["-e", "process.exit(1)"], { timeoutMs: 10_000 });
    expect(r).toMatchObject({ ok: false, code: null, exitCode: 1 });
  });

  it("timeout: resolved (not thrown) with exitCode null", async () => {
    const r = await runChild(node, ["-e", "setTimeout(() => {}, 5000)"], { timeoutMs: 150 });
    expect(r).toMatchObject({ ok: false, code: null, exitCode: null });
  });

  it("spawn error (missing binary): resolved with exitCode null", async () => {
    const r = await runChild("jaxos-no-such-binary-4f2c", ["x"], { timeoutMs: 1000 });
    expect(r).toMatchObject({ ok: false, code: null, exitCode: null });
  });

  it("round-2 F1: a synchronous spawn error (a NUL byte in argv) resolves, never rejects", async () => {
    await expect(runChild(node, ["a\0b"], { timeoutMs: 1000 })).resolves.toMatchObject({
      ok: false, code: null, exitCode: null, stdoutTail: "",
    });
  });

  it("stdout/stderr tails are redacted and capped at LIMITS.childTail", async () => {
    const r = await runChild(node, ["-e", `process.stdout.write('token=${GHP}'); process.stderr.write('x'.repeat(10000))`], { timeoutMs: 10_000 });
    expect(r.stdoutTail).toBe("token=[REDACTED]");
    expect(r.stderrTail).toHaveLength(LIMITS.childTail);
  });

  it("passes cwd and merges env over process.env", async () => {
    const r = await runChild(node, ["-e", "process.stdout.write(process.cwd() + '|' + process.env.JAXOS_CALLER_SESSION + '|' + (process.env.PATH ? 'path' : 'nopath'))"],
      { cwd: "/tmp", env: { JAXOS_CALLER_SESSION: "jaxos" }, timeoutMs: 10_000 });
    expect(r.stdoutTail).toBe("/tmp|jaxos|path");
  });
});

describe("jaxflow() / childDetails / lastLine", () => {
  it("jaxflow prepends python3 + the repo's scripts/jaxflow.py and forwards opts", async () => {
    const calls: { file: string; args: string[]; opts: unknown }[] = [];
    const fake: ChildRunner = async (file, args, opts) => {
      calls.push({ file, args, opts });
      return { ok: true, code: null, exitCode: 0, stdoutTail: "ok\n", stderrTail: "", durationMs: 1 };
    };
    await jaxflow(["cancel", "a1b2c3d4e5f6"], { timeoutMs: 25_000 }, fake);
    expect(JAXFLOW_PROGRAM).toBe("python3");
    expect(JAXFLOW_SCRIPT.endsWith("/scripts/jaxflow.py")).toBe(true);
    expect(calls).toEqual([{ file: "python3", args: [JAXFLOW_SCRIPT, "cancel", "a1b2c3d4e5f6"], opts: { timeoutMs: 25_000 } }]);
  });

  it("childDetails keeps exactly the audit fields; lastLine returns the last non-empty line", () => {
    const r = { ok: false, code: "x", exitCode: 2, stdoutTail: "a\nb\n\n", stderrTail: "x\n", durationMs: 7 };
    expect(childDetails(r)).toEqual({ exitCode: 2, stdoutTail: "a\nb\n\n", stderrTail: "x\n", durationMs: 7 });
    expect(lastLine("a\nb\n\n")).toBe("b");
    expect(lastLine("")).toBe("");
  });
});
