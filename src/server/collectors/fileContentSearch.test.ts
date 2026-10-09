import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";
vi.hoisted(() => {
  const { mkdtempSync, mkdirSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const dir = mkdtempSync(join(tmpdir(), "jaxos-fcs-test-"));
  const vaultDir = mkdtempSync(join(tmpdir(), "jaxos-fcs-test-vault-"));
  // Below, every "daily" hit comes from a FAKE spawnImpl — content is never read from disk —
  // but relativeWithin's realpathSync containment check still needs the file to actually
  // exist. A portable temp fixture, never Rafa's real vault (CI has neither).
  writeFileSync(join(vaultDir, "daily"), "", "utf8");
  // Portable repos fixture: the repo-scope tests use fake rg output, but relativeWithin still
  // needs `package.json` and `.git/config` (the EXCLUDES burst) to exist under the repo cwd.
  // Without a reposRoot, ROOTS.repos fell back to `$HOME/repos` — nonexistent on a fresh runner.
  const reposDir = mkdtempSync(join(tmpdir(), "jaxos-fcs-test-repos-"));
  mkdirSync(join(reposDir, "jax-os", ".git"), { recursive: true });
  writeFileSync(join(reposDir, "jax-os", "package.json"), '{"name":"jax-os"}', "utf8");
  writeFileSync(join(reposDir, "jax-os", ".git", "config"), "", "utf8");
  writeFileSync(join(dir, "settings.json"), JSON.stringify({ vaultPath: vaultDir, reposRoot: reposDir }), "utf8");
  process.env.JAXOS_HOME = dir;
});
import { EXCLUDES, READ_CAP, ROOTS } from "./files";
import {
  buildRgArgs, CONTENT_SEARCH_CAP, CONTENT_SEARCH_TIMEOUT_MS, isExcludedRel, parseRgLine,
  PER_FILE_CAP, runRg, searchContent, type SpawnImpl,
} from "./fileContentSearch";

const RG_AVAILABLE = existsSync("/usr/bin/rg");

// `script` runs on a microtask so on()/stdout.on() register first; `onKill` proves the cross-kill
// wiring (never 200 apiece) by tracking whether THIS fake process was killed by our own code.
function fakeSpawn(
  script: (emit: { data: (s: string) => void; error: (e: Error) => void; close: (code: number) => void }) => void,
  onKill?: () => void,
): SpawnImpl {
  return () => {
    const dataHandlers: ((c: Buffer) => void)[] = [];
    const handlers: Record<string, ((arg: unknown) => void)[]> = { error: [], close: [] };
    const emit = {
      data: (s: string) => dataHandlers.forEach((fn) => fn(Buffer.from(s))),
      error: (e: Error) => handlers.error.forEach((fn) => fn(e)),
      close: (code: number) => handlers.close.forEach((fn) => fn(code)),
    };
    queueMicrotask(() => script(emit));
    return {
      stdout: { on: (_ev: "data", fn: (c: Buffer) => void) => { dataHandlers.push(fn); } },
      on: (ev: string, fn: (arg: unknown) => void) => { handlers[ev]?.push(fn); },
      kill: () => { onKill?.(); return true; },
    };
  };
}
const matchLine = (path: string, line: number, snippet: string) =>
  JSON.stringify({ type: "match", data: { path: { text: path }, line_number: line, lines: { text: `${snippet}\n` } } });

describe("buildRgArgs", () => {
  it("builds the §8 argv: flags, one --glob '!<name>' per EXCLUDES entry, then -- <query>", () => {
    const args = buildRgArgs("needle");
    expect(args.slice(0, 7)).toEqual(["--json", "-i", "--fixed-strings", "--max-filesize", String(READ_CAP), "-m", String(PER_FILE_CAP)]);
    expect(args.slice(7, -2)).toEqual([...EXCLUDES].flatMap((name) => ["--glob", `!${name}`]));
    expect(args.slice(-2)).toEqual(["--", "needle"]);
  });
  it("guards a query starting with '-' behind --", () => expect(buildRgArgs("-x").slice(-2)).toEqual(["--", "-x"]));
});

describe("parseRgLine", () => {
  it("extracts path/line/snippet, trimming the trailing newline", () => {
    expect(parseRgLine(matchLine("src/a.ts", 12, "const x = 1;"))).toEqual({ path: "src/a.ts", line: 12, snippet: "const x = 1;" });
  });
  it("ignores non-match types and malformed JSON", () => {
    expect(parseRgLine(JSON.stringify({ type: "begin", data: {} }))).toBeNull();
    expect(parseRgLine("not json")).toBeNull();
    expect(parseRgLine("")).toBeNull();
  });
});

describe("isExcludedRel (secret/EXCLUDES filtering — mandatory, not a hint, spec §8)", () => {
  it("drops a rel with an EXCLUDES or secret-named segment anywhere in the path", () => {
    for (const bad of ["node_modules/x.ts", "a/.git/config", ".env", "sub/id_rsa", "sub/credentials.json"]) {
      expect(isExcludedRel(bad)).toBe(true);
    }
  });
  it("keeps a clean rel", () => expect(isExcludedRel("src/components/a.tsx")).toBe(false));
});

describe("searchContent — spawn failure classes (injected spawnImpl, no real rg)", () => {
  it("ENOENT names rg specifically", async () => {
    const enoent = fakeSpawn((emit) => emit.error(Object.assign(new Error("spawn rg ENOENT"), { code: "ENOENT" })));
    await expect(searchContent({ kind: "vault" }, "x", undefined, enoent)).rejects.toThrow("rg is required and was not found");
  });
  it("a real error exit (code 2) keeps the generic message", async () => {
    const exit2 = fakeSpawn((emit) => emit.close(2));
    await expect(searchContent({ kind: "vault" }, "x", undefined, exit2)).rejects.toThrow("content search unavailable");
  });
  it("no-matches (code 1) resolves ok with zero hits, not an error", async () => {
    const spawnImpl = fakeSpawn((emit) => emit.close(1));
    await expect(searchContent({ kind: "vault" }, "x", undefined, spawnImpl)).resolves.toEqual({ hits: [], truncated: false });
  });
  it("a hard timeout kills the process and resolves with hits so far, truncated:true — never rejects", async () => {
    vi.useFakeTimers();
    const p = searchContent({ kind: "vault" }, "x", undefined, fakeSpawn(() => { /* never closes */ }));
    await vi.advanceTimersByTimeAsync(CONTENT_SEARCH_TIMEOUT_MS);
    await expect(p).resolves.toEqual({ hits: [], truncated: true });
    vi.useRealTimers();
  });
  it("an aborted signal rejects with the abort reason — nothing left to report (round-1 F2)", async () => {
    const controller = new AbortController();
    const p = searchContent({ kind: "vault" }, "x", controller.signal, fakeSpawn(() => { /* never closes */ }));
    controller.abort(new Error("client gone"));
    await expect(p).rejects.toThrow("client gone");
  });
});

describe("searchContent — scope-to-cwd wiring (fake rg output, REAL path conversion)", () => {
  it("repo:<name> resolves cwd via resolveWithin and prefixes rel with '<name>/' for a real hit; a hit path that doesn't exist under cwd is dropped (relativeWithin's null, belt-and-suspenders)", async () => {
    const real = fakeSpawn((emit) => { emit.data(matchLine("package.json", 1, "hit") + "\n"); emit.close(0); });
    expect(await searchContent({ kind: "repo", name: "jax-os" }, "hit", undefined, real))
      .toEqual({ hits: [{ root: "repos", rel: "jax-os/package.json", line: 1, snippet: "hit" }], truncated: false });
    const missing = fakeSpawn((emit) => { emit.data(matchLine("does-not-exist-xyz.ts", 1, "hit") + "\n"); emit.close(0); });
    expect(await searchContent({ kind: "repo", name: "jax-os" }, "hit", undefined, missing)).toEqual({ hits: [], truncated: false });
  });
  it("vault resolves cwd to ROOTS.vault directly, rel unprefixed", async () => {
    const spawnImpl = fakeSpawn((emit) => { emit.data(matchLine("daily", 1, "hit") + "\n"); emit.close(0); }); // "daily" exists in the real vault
    expect(await searchContent({ kind: "vault" }, "hit", undefined, spawnImpl))
      .toEqual({ hits: [{ root: "vault", rel: "daily", line: 1, snippet: "hit" }], truncated: false });
  });
});

describe("searchContent — reposOnly scope (spec decision 16)", () => {
  it("spawns exactly one rg over ROOTS.repos, never a vault sibling", async () => {
    const cwds: string[] = [];
    const spawnImpl: SpawnImpl = (file, args, opts) => {
      cwds.push(opts.cwd);
      return fakeSpawn((emit) => { emit.close(1); })(file, args, opts); // 1 = no matches
    };
    expect(await searchContent({ kind: "reposOnly" }, "x", undefined, spawnImpl)).toEqual({ hits: [], truncated: false });
    expect(cwds).toEqual([ROOTS.repos]);
  });
});

describe("searchContent — 'all' shares ONE 200 cap and kills the still-running sibling (spec §8, round-1 F2)", () => {
  it("never 200 apiece: repos alone hitting the cap ends the call AND kills the vault sibling that never closes on its own; a normal run merges repos-then-vault", async () => {
    let vaultKilled = false;
    // "all"'s repos-side cwd is ROOTS.repos itself (the parent of every checkout), so a real hit's
    // path already carries its repo's own name — "package.json" alone would resolve to a
    // non-existent /home/rafa/repos/package.json and get silently dropped by the F2 fix below.
    const lines = Array.from({ length: CONTENT_SEARCH_CAP }, (_, i) => matchLine("jax-os/package.json", i + 1, "x")).join("\n") + "\n";
    const reposSpawn = fakeSpawn((emit) => emit.data(lines)); // never closes — mimics a slow rg process
    const vaultSpawn = fakeSpawn(() => {}, () => { vaultKilled = true; }); // never emits/closes — would hang the test if not killed
    let call = 0;
    const capped: SpawnImpl = (file, args, opts) => (call++ === 0 ? reposSpawn : vaultSpawn)(file, args, opts);
    const capResult = await searchContent({ kind: "all" }, "x", undefined, capped);
    expect(capResult.hits).toHaveLength(CONTENT_SEARCH_CAP);
    expect(capResult.hits.every((h) => h.root === "repos")).toBe(true);
    expect(capResult.truncated).toBe(true);
    expect(vaultKilled).toBe(true);

    const reposSpawn2 = fakeSpawn((emit) => { emit.data(matchLine("jax-os/package.json", 1, "r") + "\n"); emit.close(0); });
    const vaultSpawn2 = fakeSpawn((emit) => { emit.data(matchLine("daily", 1, "v") + "\n"); emit.close(0); });
    let call2 = 0;
    const ordered: SpawnImpl = (file, args, opts) => (call2++ === 0 ? reposSpawn2 : vaultSpawn2)(file, args, opts);
    expect((await searchContent({ kind: "all" }, "x", undefined, ordered)).hits).toEqual([
      { root: "repos", rel: "jax-os/package.json", line: 1, snippet: "r" },
      { root: "vault", rel: "daily", line: 1, snippet: "v" },
    ]);
  });
});

describe("searchContent — 'all' kills the sibling when either stream errors, not just on a cap hit (round-2 F5)", () => {
  it("a spawn error on one side rejects the whole call AND kills the sibling that never closes on its own", async () => {
    let vaultKilled = false;
    const reposSpawn = fakeSpawn((emit) => emit.error(Object.assign(new Error("spawn rg ENOENT"), { code: "ENOENT" })));
    const vaultSpawn = fakeSpawn(() => { /* never emits/closes — would hang the test if not killed */ }, () => { vaultKilled = true; });
    let call = 0;
    const errored: SpawnImpl = (file, args, opts) => (call++ === 0 ? reposSpawn : vaultSpawn)(file, args, opts);
    await expect(searchContent({ kind: "all" }, "x", undefined, errored)).rejects.toThrow("rg is required and was not found");
    expect(vaultKilled).toBe(true);
  });
});

describe("searchContent — excluded/missing hits never consume the cap (round-2 F2)", () => {
  it("a run of EXCLUDES-filtered hits ahead of a few valid ones still returns every valid hit uncapped — filtering, not just an rg --glob hint, runs before the cap increments", async () => {
    // ".git/config" exists on disk (any git checkout has one) so it reaches isExcludedRel rather
    // than being dropped earlier by relativeWithin's existence check — this isolates the EXCLUDES
    // filter itself as the thing proven not to cost cap budget.
    // repo:jax-os's cwd IS the checkout, so raw hit paths are plain ("package.json", ".git/config"
    // — not "jax-os/..." — matching the existing repo-scope test above; conversion prefixes them.
    const excludedBurst = Array.from({ length: CONTENT_SEARCH_CAP }, () => matchLine(".git/config", 1, "excluded")).join("\n") + "\n";
    const validLines = [matchLine("package.json", 1, "v1"), matchLine("package.json", 2, "v2"), matchLine("package.json", 3, "v3")].join("\n") + "\n";
    const spawnImpl = fakeSpawn((emit) => { emit.data(excludedBurst); emit.data(validLines); emit.close(0); });
    expect(await searchContent({ kind: "repo", name: "jax-os" }, "x", undefined, spawnImpl)).toEqual({
      hits: [
        { root: "repos", rel: "jax-os/package.json", line: 1, snippet: "v1" },
        { root: "repos", rel: "jax-os/package.json", line: 2, snippet: "v2" },
        { root: "repos", rel: "jax-os/package.json", line: 3, snippet: "v3" },
      ],
      truncated: false,
    });
  });
});

describe.skipIf(!RG_AVAILABLE)("runRg against real rg (temp dir fixture, no ROOTS coupling)", () => {
  let dir: string;
  beforeAll(() => {
    dir = mkdtempSync(join(tmpdir(), "jax-content-search-"));
    writeFileSync(join(dir, "a.ts"), Array.from({ length: 20 }, (_, i) => `needle ${i}`).join("\n") + "\n");
    writeFileSync(join(dir, "b.ts"), "no match here\n");
  });
  afterAll(() => rmSync(dir, { recursive: true, force: true }));

  it("parses real NDJSON output, caps a single file's matches at PER_FILE_CAP (rg's own -m 5)", async () => {
    const { hits, truncated } = await runRg(dir, "needle", undefined);
    expect(hits.filter((h) => h.path === "a.ts")).toHaveLength(PER_FILE_CAP);
    expect(hits.every((h) => h.path !== "b.ts")).toBe(true);
    expect(truncated).toBe(false);
  });
});
