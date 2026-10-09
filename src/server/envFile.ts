import { lstatSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { jaxosHome } from "./env";

export type EnvFileState = "ok" | "missing" | "not-regular" | "wrong-owner" | "bad-mode";

function envPath(): string {
  return join(jaxosHome(), ".env");
}

// MOA-498 D1: the TS mirror of scripts/jev_client.py's env_file_status()/_check_secure —
// same lstat-based check (regular file, owned by the running uid, mode 0600), never a
// second, stricter contract. Never returns a value.
export function envFileStatus(): { state: EnvFileState } {
  let info;
  try {
    info = lstatSync(envPath());
  } catch {
    return { state: "missing" };
  }
  if (!info.isFile()) return { state: "not-regular" };
  if (info.uid !== process.getuid?.()) return { state: "wrong-owner" };
  if ((info.mode & 0o777) !== 0o600) return { state: "bad-mode" };
  return { state: "ok" };
}

function parseEnvValue(text: string, key: string): string | undefined {
  const m = text.match(new RegExp(`^${key}=(.*)$`, "m"));
  return m ? m[1].trim().replace(/^"(.*)"$/, "$1") : undefined;
}

// Fills `target[key]` for every key in `keys` found in $JAXOS_HOME/.env, skipping any key
// already truthy on `target` (process env / .env.local already won — Next loads those into
// process.env before next.config.ts runs). No-op when the file fails the secure check.
export function loadJaxosEnv(keys: string[], target: Record<string, string | undefined> = process.env): void {
  if (envFileStatus().state !== "ok") return;
  let text: string;
  try {
    text = readFileSync(envPath(), "utf8");
  } catch {
    return;
  }
  for (const key of keys) {
    if (target[key]) continue;
    const v = parseEnvValue(text, key);
    if (v !== undefined) target[key] = v;
  }
}

// Reads one key straight from $JAXOS_HOME/.env on every call, no process.env write and no
// caching — for values (like the notification webhook target) that must pick up an owner
// edit to the file without a process restart. undefined when the file fails the secure check.
export function readJaxosEnvValue(key: string): string | undefined {
  if (envFileStatus().state !== "ok") return undefined;
  try {
    return parseEnvValue(readFileSync(envPath(), "utf8"), key);
  } catch {
    return undefined;
  }
}
