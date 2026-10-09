import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { isCliPresent } from "./cli-presence";

const dirs: string[] = [];
function tmp(): string {
  const d = mkdtempSync(join(tmpdir(), "jax-cli-"));
  dirs.push(d);
  return d;
}
function bin(dir: string, name: string, mode = 0o755): void {
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, name), "#!/bin/sh\n");
  chmodSync(join(dir, name), mode);
}
const env = (PATH: string | undefined, HOME?: string) => ({ PATH, HOME } as unknown as NodeJS.ProcessEnv);
afterEach(() => {
  for (const d of dirs) rmSync(d, { recursive: true, force: true });
  dirs.length = 0;
});

describe("isCliPresent (MOA-504 D4)", () => {
  it("is true for an executable regular file on PATH, false for absent / non-executable / directory / unset PATH", () => {
    const root = tmp();
    bin(join(root, "a"), "claude");
    bin(join(root, "b"), "codex", 0o644);
    mkdirSync(join(root, "c", "opencode"), { recursive: true });
    expect(isCliPresent("claude", env(join(root, "a")))).toBe(true);
    expect(isCliPresent("codex", env(join(root, "b")))).toBe(false);
    expect(isCliPresent("opencode", env(join(root, "c"), root))).toBe(false);
    expect(isCliPresent("claude", env(join(root, "b")))).toBe(false);
    expect(isCliPresent("claude", env(undefined))).toBe(false);
  });

  it("opencode only: a PATH miss falls back to $HOME/.opencode/bin/opencode (the installer location)", () => {
    const root = tmp();
    const home = join(root, "home");
    bin(join(home, ".opencode", "bin"), "opencode");
    expect(isCliPresent("opencode", env(join(root, "empty"), home))).toBe(true);
  });

  it("opencode: neither PATH nor the installer fallback -> absent; the fallback is NOT used for other CLIs", () => {
    const root = tmp();
    const home = join(root, "home");
    expect(isCliPresent("opencode", env(join(root, "empty"), home))).toBe(false);
    bin(join(home, ".opencode", "bin"), "claude");
    expect(isCliPresent("claude", env(join(root, "empty"), home))).toBe(false);
  });
});
