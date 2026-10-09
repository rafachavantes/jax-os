// MOA-504 D4: PATH-presence probe for an agent CLI, moved out of workflow-codex.ts and generalized.
// Collectors boundary: all system access lives here. Requires an executable REGULAR file (a bare
// accessSync(X_OK) also accepts an executable directory named like the CLI).
import { accessSync, constants as fsConstants, statSync } from "node:fs";
import { homedir } from "node:os";
import { delimiter, join } from "node:path";

function executableFile(candidate: string): boolean {
  try {
    if (!statSync(candidate).isFile()) return false;
    accessSync(candidate, fsConstants.X_OK);
    return true;
  } catch {
    return false;
  }
}

export function isCliPresent(name: string, env: NodeJS.ProcessEnv = process.env): boolean {
  const onPath = (env.PATH ?? "").split(delimiter).some((dir) => dir !== "" && executableFile(join(dir, name)));
  if (onPath || name !== "opencode") return onPath;
  // The official installer puts opencode in ~/.opencode/bin, which jaxos.service's PATH lacks;
  // mirrors scripts/jaxflow_settings_io.py's own fallback.
  return executableFile(join(env.HOME || homedir(), ".opencode", "bin", "opencode"));
}
