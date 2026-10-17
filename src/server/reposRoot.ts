// reposRoot()/vaultPath(): the ONE place `reposRoot`/`vaultPath` (settings.json fields, per
// the foundation spec's schema) are resolved to a usable value. Called fresh at each use site
// (no cache — Global Constraints); `readGeneralSettings()`'s own default-filling already
// covers "file absent" and "field absent", so the only branch this module adds is the
// {ok:false} (malformed/unreadable) fallback, per the machine-sweep spec's ONE rule: "the
// documented default is used whenever settings are absent, malformed, or unreadable."
import { homedir } from "node:os";
import { join } from "node:path";
import { readGeneralSettings } from "./settings";

export function reposRoot(): string {
  const result = readGeneralSettings();
  return result.ok ? result.data.reposRoot : join(homedir(), "repos");
}

export function vaultPath(): string | null {
  const result = readGeneralSettings();
  return result.ok ? result.data.vaultPath : null;
}
