// MOA-497 Build A: JAXOS_HOME / API base URL resolution -- process env, read once per
// language, NOT file-backed config (settings.ts is that). See scripts/jaxflow_env.py for
// the Python twin; the two MUST agree on every default.
import { homedir } from "node:os";
import { join } from "node:path";

export function jaxosHome(): string {
  return process.env.JAXOS_HOME || join(homedir(), ".jax-os");
}

export function apiBaseUrl(): string {
  return `http://127.0.0.1:${process.env.PORT ?? "3100"}`;
}
