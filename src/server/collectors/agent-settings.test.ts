import { Buffer } from "node:buffer";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import * as agentSettingsModule from "./agent-settings";
import {
  applyAgentSettings,
  previewAgentSettings,
  readCredential,
  runSettingsHelper,
  snapshotAgentSettings,
  HelperRefusal,
  type HelperRunner,
  type SpawnImpl,
} from "./agent-settings";

const SNAP = {
  editor_revision: {
    settings: "absent",
    sources: { "opencode.json": "absent", "opencode.jsonc": "absent" },
  },
  settings: null,
};

const INTENT = { reviewers: {}, builders: {} };
const EXPECTED = SNAP.editor_revision;
const OP = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

describe("public helper wrappers", () => {
  it("snapshot omits native_metadata and never paths", async () => {
    const run = vi.fn<HelperRunner>(async () => SNAP);
    await snapshotAgentSettings({ models: [] }, run);
    expect(run).toHaveBeenCalledWith("snapshot", {});
    expect(run.mock.calls[0][1]).not.toHaveProperty("native_metadata");
    expect(run.mock.calls[0][1]).not.toHaveProperty("paths");
  });

  it("preview and apply omit paths", async () => {
    const run = vi.fn<HelperRunner>(async (op) => (
      op === "apply" ? { effect: "published", editor_revision: EXPECTED, settings: {} } : { affected: ["opencode.jsonc"], aliases: {} }
    ));
    await previewAgentSettings(INTENT, EXPECTED, { models: [] }, run);
    await applyAgentSettings(INTENT, EXPECTED, OP, { models: [] }, run);
    expect(run.mock.calls[0][0]).toBe("preview");
    expect(run.mock.calls[1][0]).toBe("apply");
    for (const [, body] of run.mock.calls) {
      expect(body).not.toHaveProperty("paths");
    }
    expect(run.mock.calls[0][1]).toEqual({ intent: INTENT, expected: EXPECTED });
    expect(run.mock.calls[1][1]).toEqual({ intent: INTENT, expected: EXPECTED, operation_id: OP });
    for (const [, body] of run.mock.calls) {
      expect(body).not.toHaveProperty("native_metadata");
    }
  });

  it("maps helper refusals by code without wrapping extra text", async () => {
    const run: HelperRunner = async () => {
      throw new HelperRefusal("agent-settings-source-changed");
    };
    await expect(previewAgentSettings(INTENT, EXPECTED, {}, run)).rejects.toMatchObject({
      code: "agent-settings-source-changed",
      message: "agent-settings-source-changed",
    });
  });

  it("readCredential is a private op and is not mixed into snapshot", async () => {
    const run = vi.fn<HelperRunner>(async (op) => (op === "read-credential" ? "s3cret" : SNAP));
    const snap = await snapshotAgentSettings({}, run);
    expect(JSON.stringify(snap)).not.toContain("s3cret");
    expect(await readCredential({ kind: "env", env: "JAX_PROVIDER_FIXTURE_API_KEY" }, run)).toBe("s3cret");
    expect(run).toHaveBeenCalledWith("read-credential", {
      binding: { kind: "env", env: "JAX_PROVIDER_FIXTURE_API_KEY" },
    });
  });

  it("sync-status is no longer a recognized op (MOA-498 D5)", async () => {
    await expect(runSettingsHelper("sync-status", {})).rejects.toThrow();
  });

  it("syncStatus is not exported", () => {
    expect((agentSettingsModule as Record<string, unknown>).syncStatus).toBeUndefined();
  });
});

describe("bws.ts is gone (MOA-498 D4)", () => {
  it("the module file no longer exists", () => {
    expect(existsSync(new URL("bws.ts", import.meta.url))).toBe(false);
  });
});

describe("credential binding format (MOA-498 D4)", () => {
  let home: string;
  let savedHome: string | undefined;

  beforeEach(() => {
    home = mkdtempSync(join(tmpdir(), "jax-cred-binding-"));
    writeFileSync(join(home, ".env"), "JAX_PROVIDER_FIXTURE_API_KEY=sk-fixture\n", { mode: 0o600 });
    savedHome = process.env.JAXOS_HOME;
    process.env.JAXOS_HOME = home;
  });

  afterEach(() => {
    if (savedHome === undefined) delete process.env.JAXOS_HOME;
    else process.env.JAXOS_HOME = savedHome;
    rmSync(home, { recursive: true, force: true });
  });

  it("readCredential resolves the new env-kind shape from $JAXOS_HOME/.env", async () => {
    await expect(readCredential({ kind: "env", env: "JAX_PROVIDER_FIXTURE_API_KEY" })).resolves.toBe("sk-fixture");
  });

  it("readCredential rejects the retired bws-env shape (MOA-498 D4 — a deliberate breaking change)", async () => {
    await expect(readCredential({ kind: "bws-env", env: "JAX_PROVIDER_FIXTURE_API_KEY", secret_id: "x" } as never))
      .rejects.toMatchObject({ name: "HelperRefusal", code: "agent-settings-malformed" });
  });
});

describe("runSettingsHelper", () => {
  it("invokes python3 with the helper script and op, JSON stdin, no paths", async () => {
    const recorded = { file: "", args: [] as string[], stdin: "" };
    const spawnImpl = ((file: string, args: string[]) => {
      recorded.file = file;
      recorded.args = args;
      const data: Array<(c: Buffer) => void> = [];
      const close: Array<(code: number) => void> = [];
      queueMicrotask(() => {
        for (const fn of data) fn(Buffer.from(JSON.stringify({ ok: true, data: SNAP })));
        for (const fn of close) fn(0);
      });
      return {
        stdin: {
          write() { return true; },
          end(chunk?: Buffer | string) {
            recorded.stdin += Buffer.isBuffer(chunk) ? chunk.toString("utf8") : String(chunk ?? "");
          },
        },
        stdout: { on(ev: string, fn: (c: Buffer) => void) { if (ev === "data") data.push(fn); } },
        stderr: { on() {}, resume() {} },
        on(ev: string, fn: (arg?: number | Error) => void) {
          if (ev === "close") close.push(fn as (code: number) => void);
        },
        kill() {},
      };
    }) as unknown as SpawnImpl;
    const result = await runSettingsHelper("snapshot", { native_metadata: { k: 1 }, paths: { settings: "/etc/passwd" } }, spawnImpl);
    expect(recorded.file).toBe("python3");
    expect(recorded.args[0]).toMatch(/jaxflow_settings_io\.py$/);
    expect(recorded.args[1]).toBe("snapshot");
    const body = JSON.parse(recorded.stdin) as Record<string, unknown>;
    expect(body.paths).toBeUndefined();
    expect(body.native_metadata).toEqual({ k: 1 });
    expect(result).toEqual(SNAP);
  });

  it("maps helper {ok:false} to HelperRefusal with the code only", async () => {
    const spawnImpl = ((file: string, args: string[]) => {
      const data: Array<(c: Buffer) => void> = [];
      const close: Array<(code: number) => void> = [];
      queueMicrotask(() => {
        for (const fn of data) fn(Buffer.from(JSON.stringify({ ok: false, error: "agent-settings-source-changed", data: "s3cret" })));
        for (const fn of close) fn(0);
      });
      return {
        stdin: { write() { return true; }, end() {} },
        stdout: { on(ev: string, fn: (c: Buffer) => void) { if (ev === "data") data.push(fn); } },
        stderr: { on() {}, resume() {} },
        on(ev: string, fn: (arg?: number | Error) => void) {
          if (ev === "close") close.push(fn as (code: number) => void);
        },
        kill() {},
      };
    }) as unknown as SpawnImpl;
    await expect(runSettingsHelper("preview", { intent: {}, expected: {} }, spawnImpl)).rejects.toMatchObject({
      name: "HelperRefusal",
      code: "agent-settings-source-changed",
      message: "agent-settings-source-changed",
    });
  });
});
