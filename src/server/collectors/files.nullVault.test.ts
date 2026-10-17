import { describe, expect, it, vi } from "vitest";
vi.hoisted(() => {
  const { mkdtempSync, mkdirSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const dir = mkdtempSync(join(tmpdir(), "jaxos-null-vault-test-"));
  // Portable repos fixture (no vaultPath key on purpose: the null-vault behavior this file
  // tests must stay intact). Without a reposRoot, ROOTS.repos falls back to `$HOME/repos`,
  // which does not exist on a fresh runner.
  const reposDir = mkdtempSync(join(tmpdir(), "jaxos-null-vault-test-repos-"));
  mkdirSync(join(reposDir, "jax-os"));
  writeFileSync(join(reposDir, "jax-os", "package.json"), '{"name":"jax-os"}', "utf8");
  writeFileSync(join(dir, "settings.json"), JSON.stringify({ reposRoot: reposDir }), "utf8");
  process.env.JAXOS_HOME = dir;
});
import { resolveExisting, searchByName, buildIndexUncached, ROOTS } from "./files";
import { searchContent, type SpawnImpl } from "./fileContentSearch";

describe("null vault (spec: Null vault — root=vault/scope=vault → {ok:false,error:disabled}, scope=all → reposOnly)", () => {
  it("a direct root=vault operation throws 'disabled'", () => {
    expect(() => resolveExisting("vault", "x")).toThrow("disabled");
  });
  it("scope=vault search rejects 'disabled'", async () => {
    // searchByName is synchronous: the null-vault guard throws before any promise exists (build a76a86c7b78f finding)
    expect(() => searchByName("q", { kind: "vault" })).toThrow("disabled");
  });
  it("scope=all narrows to repos-only, no refusal, no vault filesystem call", async () => {
    const result = await buildIndexUncached({ kind: "all" });
    expect(result.data.every((h) => h.root === "repos")).toBe(true);
  });
});

// Records which `cwd` each spawn was invoked with, closes with "no matches" (code 0) — enough
// to prove which root(s) a call actually reached without a real rg binary.
function noMatchSpawn(calls: string[]): SpawnImpl {
  return (_file, _args, opts) => {
    calls.push(opts.cwd);
    const handlers: Record<string, ((arg: unknown) => void)[]> = { error: [], close: [] };
    queueMicrotask(() => handlers.close.forEach((fn) => fn(0)));
    return { stdout: { on: () => {} }, on: (ev, fn) => { handlers[ev]?.push(fn); }, kill: () => true };
  };
}

describe("content search null vault (spec: Null vault)", () => {
  it("scope=vault rejects 'disabled' before ever spawning rg", async () => {
    const calls: string[] = [];
    await expect(searchContent({ kind: "vault" }, "q", undefined, noMatchSpawn(calls))).rejects.toThrow("disabled");
    expect(calls).toEqual([]);
  });
  it("scope=all narrows to repos-only — no refusal, no vault rg spawned", async () => {
    const calls: string[] = [];
    await searchContent({ kind: "all" }, "q", undefined, noMatchSpawn(calls));
    expect(calls).toEqual([ROOTS.repos]);
  });
});
