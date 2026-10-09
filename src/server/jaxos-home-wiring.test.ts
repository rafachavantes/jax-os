import { afterEach, describe, expect, it, vi } from "vitest";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";

afterEach(() => {
  delete process.env.JAXOS_HOME;
  vi.resetModules();
});

describe("JAXOS_HOME wiring — TS consumers (spec consumer table)", () => {
  it("db/index.ts's jaxos db path (src/server/db/index.ts:241)", async () => {
    process.env.JAXOS_HOME = "/tmp/moa497x";
    vi.resetModules();
    const { DB_PATH } = await import("./db/index");
    expect(DB_PATH).toBe(join("/tmp/moa497x", "jaxos.db"));
  });

  it("workflow-callbacks.ts's CALLBACKS_ROOT (src/server/collectors/workflow-callbacks.ts:10)", async () => {
    process.env.JAXOS_HOME = "/tmp/moa497x";
    vi.resetModules();
    const { CALLBACKS_ROOT } = await import("./collectors/workflow-callbacks");
    expect(CALLBACKS_ROOT).toBe(join("/tmp/moa497x", "callbacks"));
  });

  it("inventory.ts's roots.jax_os (src/server/collectors/inventory.ts:111)", async () => {
    process.env.JAXOS_HOME = "/tmp/moa497x";
    vi.resetModules();
    const { systemInventoryDeps } = await import("./collectors/inventory");
    expect(systemInventoryDeps().roots.jax_os).toBe("/tmp/moa497x");
  });

  it("inventory.ts's defaultVenvPython (src/server/collectors/inventory.ts:117)", async () => {
    const dir = mkdtempSync(join(tmpdir(), "moa497-venv-"));
    mkdirSync(join(dir, "inventory-venv", "bin"), { recursive: true });
    writeFileSync(join(dir, "inventory-venv", "bin", "python"), "");
    process.env.JAXOS_HOME = dir;
    vi.resetModules();
    const { defaultVenvPython } = await import("./collectors/inventory");
    expect(defaultVenvPython()).toBe(join(dir, "inventory-venv", "bin", "python"));
  });
});

describe("default-equals-today — TS consumers, JAXOS_HOME unset (spec Acceptance 4)", () => {
  it("db/index.ts", async () => {
    vi.resetModules();
    const { DB_PATH } = await import("./db/index");
    expect(DB_PATH).toBe(join(homedir(), ".jax-os", "jaxos.db"));
  });
  it("workflow-callbacks.ts", async () => {
    vi.resetModules();
    const { CALLBACKS_ROOT } = await import("./collectors/workflow-callbacks");
    expect(CALLBACKS_ROOT).toBe(join(homedir(), ".jax-os", "callbacks"));
  });
  it("inventory.ts's roots.jax_os", async () => {
    vi.resetModules();
    const { systemInventoryDeps } = await import("./collectors/inventory");
    expect(systemInventoryDeps().roots.jax_os).toBe(join(homedir(), ".jax-os"));
  });
});
