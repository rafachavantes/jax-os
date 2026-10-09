import { describe, expect, it, vi } from "vitest";
vi.hoisted(() => {
  const { mkdtempSync, writeFileSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const home = mkdtempSync(join(tmpdir(), "jaxos-missionproj-home-"));
  const reposRoot = mkdtempSync(join(tmpdir(), "jaxos-missionproj-repos-"));
  // Pin REPOS_ROOT (fixed when ../collectors/projects is first imported, transitively by
  // ./route) to a test-owned temp dir. Left ambient it is `$HOME/repos`, which does not exist
  // on a fresh runner and must never be created on the owner's machine. mkdtempSync itself
  // creates the dir, so readdirSync(REPOS_ROOT) succeeds with zero entries.
  writeFileSync(join(home, "settings.json"), JSON.stringify({ reposRoot }), "utf8");
  process.env.JAXOS_HOME = home;
});
import { GET } from "./route";

describe("GET /api/mission/projects (§7.3, §9 test 20 — envelope regression)", () => {
  it("keeps the nested {projects, skipped} shape under data, not flattened", async () => {
    const res = await GET();
    const body = await res.json();
    expect(body.ok).toBe(true);
    expect(Array.isArray(body.data.projects)).toBe(true);
    expect(typeof body.data.skipped).toBe("number");
  });
});
