import { describe, expect, it, afterEach } from "vitest";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { homedir } from "node:os";
import { join } from "node:path";
import { reposRoot, vaultPath } from "./reposRoot";
import { apiBaseUrl } from "./env";

const dirs: string[] = [];
function tempHome(settingsJson?: string): string {
  const dir = mkdtempSync(join(tmpdir(), "jaxos-reposroot-"));
  dirs.push(dir);
  if (settingsJson !== undefined) writeFileSync(join(dir, "settings.json"), settingsJson, "utf8");
  return dir;
}
afterEach(() => { process.env.JAXOS_HOME = undefined; while (dirs.length) rmSync(dirs.pop()!, { recursive: true, force: true }); });

describe("reposRoot()/vaultPath() (spec: reposRoot/vaultPath consumers)", () => {
  it("defaults to homedir()/repos and null with no settings.json", () => {
    process.env.JAXOS_HOME = tempHome();
    expect(reposRoot()).toBe(join(homedir(), "repos"));
    expect(vaultPath()).toBeNull();
  });

  it("reads a configured reposRoot/vaultPath", () => {
    process.env.JAXOS_HOME = tempHome(JSON.stringify({ reposRoot: "/tmp/x/repos", vaultPath: "/tmp/x/vault" }));
    expect(reposRoot()).toBe("/tmp/x/repos");
    expect(vaultPath()).toBe("/tmp/x/vault");
  });

  it("falls back to the default on malformed settings (never throws)", () => {
    process.env.JAXOS_HOME = tempHome("not json");
    expect(reposRoot()).toBe(join(homedir(), "repos"));
    expect(vaultPath()).toBeNull();
  });

  it("falls back to the default on unreadable settings (never throws)", () => {
    const dir = tempHome();
    writeFileSync(join(dir, "settings.json"), "{}", "utf8");
    // owner-read-only elsewhere in this repo already uses chmod 0 to simulate "unreadable" —
    // same technique, no new pattern.
    require("node:fs").chmodSync(join(dir, "settings.json"), 0o000);
    process.env.JAXOS_HOME = dir;
    expect(reposRoot()).toBe(join(homedir(), "repos"));
    expect(vaultPath()).toBeNull();
    require("node:fs").chmodSync(join(dir, "settings.json"), 0o644); // afterEach can rmSync it
  });

  // Acceptance criterion 5: JAXOS_HOME, PORT and reposRoot set together in ONE run resolve
  // every consumer this build added (path -> reposRoot(), URL -> apiBaseUrl()).
  it("resolves every consumer end-to-end under JAXOS_HOME, PORT and reposRoot set together", () => {
    process.env.JAXOS_HOME = tempHome(JSON.stringify({ reposRoot: "/tmp/e2e-custom-repos" }));
    const prevPort = process.env.PORT;
    process.env.PORT = "3199";
    try {
      expect(reposRoot()).toBe("/tmp/e2e-custom-repos");
      expect(apiBaseUrl()).toBe("http://127.0.0.1:3199");
    } finally {
      if (prevPort === undefined) delete process.env.PORT; else process.env.PORT = prevPort;
    }
  });
});
