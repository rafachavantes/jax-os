import { spawn } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const HELPER = join(dirname(fileURLToPath(import.meta.url)), "../../../scripts/jaxflow_settings_io.py");
const IO_CAP = 16 * 1024 * 1024;
const TIMEOUT_MS = 10_000;
const OPS = new Set(["snapshot", "preview", "apply", "read-credential"]);

export class HelperRefusal extends Error {
  readonly code: string;
  constructor(code: string) {
    super(code);
    this.name = "HelperRefusal";
    this.code = code;
  }
}

export type HelperRunner = (op: string, body: Record<string, unknown>) => Promise<unknown>;

export type BindingRef = {
  provider_id: string;
  env_name: string;
};

export type ReadCredentialBinding =
  | { kind: "env"; env: string }
  | { kind: "native"; connection: string };

export type SpawnImpl = (
  file: string,
  args: string[],
  options: { stdio: ["pipe", "pipe", "pipe"] },
) => {
  stdin: { write: (d: Buffer | string) => boolean; end: (d?: Buffer | string) => void };
  stdout: { on: (ev: "data", fn: (c: Buffer) => void) => void };
  stderr: { on: (ev: string, fn?: (c: Buffer) => void) => void; resume?: () => void };
  on: (ev: "error" | "close", fn: (arg?: number | Error) => void) => void;
  kill: (signal?: NodeJS.Signals) => boolean | void;
};

function refusalCode(value: unknown): string {
  return typeof value === "string" && /^[a-z0-9-]{1,64}$/.test(value) ? value : "agent-settings-malformed";
}

export function runSettingsHelper(
  op: string,
  body: Record<string, unknown>,
  spawnImpl: SpawnImpl = spawn as SpawnImpl,
): Promise<unknown> {
  if (!OPS.has(op)) return Promise.reject(new HelperRefusal("agent-settings-malformed"));
  const { paths: _ignored, ...rest } = body;
  let encoded: Buffer;
  try {
    encoded = Buffer.from(JSON.stringify(rest), "utf8");
  } catch {
    return Promise.reject(new HelperRefusal("agent-settings-malformed"));
  }
  if (encoded.length > IO_CAP) return Promise.reject(new HelperRefusal("agent-settings-too-large"));
  return new Promise((resolve, reject) => {
    const child = spawnImpl("python3", [HELPER, op], { stdio: ["pipe", "pipe", "pipe"] });
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;
    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      finish(new HelperRefusal("native-config-malformed"));
    }, TIMEOUT_MS);
    const finish = (err: Error | null, value?: unknown) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (err) reject(err);
      else resolve(value);
    };
    child.stdout.on("data", (chunk) => {
      size += chunk.length;
      if (size > IO_CAP) {
        child.kill("SIGKILL");
        finish(new HelperRefusal("agent-settings-too-large"));
        return;
      }
      chunks.push(chunk);
    });
    child.stderr.resume?.();
    child.on("error", () => finish(new HelperRefusal("agent-settings-permissions")));
    child.on("close", () => {
      try {
        const parsed = JSON.parse(Buffer.concat(chunks).toString("utf8")) as {
          ok?: boolean;
          error?: unknown;
          data?: unknown;
        };
        if (parsed && parsed.ok === true) {
          finish(null, parsed.data);
          return;
        }
        finish(new HelperRefusal(refusalCode(parsed?.error)));
      } catch {
        finish(new HelperRefusal("agent-settings-malformed"));
      }
    });
    child.stdin.end(encoded);
  });
}

const defaultRun: HelperRunner = (op, body) => runSettingsHelper(op, body);

export function snapshotAgentSettings(
  _nativeMetadata?: unknown,
  run: HelperRunner = defaultRun,
  bindings?: BindingRef[],
) {
  return run("snapshot", bindings ? { bindings } : {});
}

export function previewAgentSettings(
  intent: unknown,
  expected: unknown,
  _nativeMetadata?: unknown,
  run: HelperRunner = defaultRun,
  bindings?: BindingRef[],
) {
  return run("preview", { intent, expected, ...(bindings ? { bindings } : {}) });
}

export function applyAgentSettings(
  intent: unknown,
  expected: unknown,
  operationId: string,
  _nativeMetadata?: unknown,
  run: HelperRunner = defaultRun,
  bindings?: BindingRef[],
) {
  return run("apply", {
    intent,
    expected,
    operation_id: operationId,
    ...(bindings ? { bindings } : {}),
  });
}

export async function readCredential(
  binding: ReadCredentialBinding,
  run: HelperRunner = defaultRun,
): Promise<string> {
  const data = await run("read-credential", { binding });
  if (typeof data !== "string" || data === "") throw new HelperRefusal("agent-settings-malformed");
  return data;
}
