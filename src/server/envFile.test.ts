import { mkdtempSync, mkdirSync, rmSync, symlinkSync, writeFileSync, chmodSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { envFileStatus, loadJaxosEnv } from "./envFile";

let dir: string;
beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "jax-envfile-"));
  process.env.JAXOS_HOME = dir;
});
afterEach(() => {
  delete process.env.JAXOS_HOME;
  rmSync(dir, { recursive: true, force: true });
});

function write(text: string, mode = 0o600) {
  const p = join(dir, ".env");
  writeFileSync(p, text, { mode });
  chmodSync(p, mode);
  return p;
}

describe("envFileStatus", () => {
  it("missing when the file does not exist", () => {
    expect(envFileStatus()).toEqual({ state: "missing" });
  });
  it("not-regular for a directory", () => {
    mkdirSync(join(dir, ".env"));
    expect(envFileStatus()).toEqual({ state: "not-regular" });
  });
  it("not-regular for a symlink, even to a valid 0600 file", () => {
    const real = join(dir, "real.env");
    writeFileSync(real, "X=1\n", { mode: 0o600 });
    symlinkSync(real, join(dir, ".env"));
    expect(envFileStatus()).toEqual({ state: "not-regular" });
  });
  it("bad-mode for 0644", () => {
    write("X=1\n", 0o644);
    expect(envFileStatus()).toEqual({ state: "bad-mode" });
  });
  it("ok for a regular 0600 file", () => {
    write("X=1\n");
    expect(envFileStatus()).toEqual({ state: "ok" });
  });
  it("wrong-owner when the file's owner differs from the running uid", () => {
    // No test here runs as root, so a real different-owner file can't be created
    // deterministically (cold review 0c968393813f F4): mock process.getuid() instead of the
    // file's real owner, which this test process cannot change anyway.
    write("X=1\n");
    const realUid = process.getuid?.() ?? 0;
    const realGetuid = process.getuid;
    process.getuid = (() => realUid + 1) as typeof process.getuid;
    try {
      expect(envFileStatus()).toEqual({ state: "wrong-owner" });
    } finally {
      process.getuid = realGetuid;
    }
  });
});

describe("loadJaxosEnv", () => {
  it("fills only the requested keys, skipping ones already set (process env wins)", () => {
    write("A=from-file\nB=also-file\n");
    const target: Record<string, string | undefined> = { A: "from-process-env" };
    loadJaxosEnv(["A", "B", "C"], target);
    expect(target).toEqual({ A: "from-process-env", B: "also-file" });
  });
  it("strips matching quotes", () => {
    write('A="quoted value"\n');
    const target: Record<string, string | undefined> = {};
    loadJaxosEnv(["A"], target);
    expect(target.A).toBe("quoted value");
  });
  it("does nothing when the file fails the secure check (bad-mode)", () => {
    write("A=x\n", 0o644);
    const target: Record<string, string | undefined> = {};
    loadJaxosEnv(["A"], target);
    expect(target).toEqual({});
  });
  it("does nothing when the file is missing", () => {
    const target: Record<string, string | undefined> = {};
    loadJaxosEnv(["A"], target);
    expect(target).toEqual({});
  });
});
