import { useQuery } from "@tanstack/react-query";
import type { GeneralSettings } from "../server/settings";
import type { EnvFileState } from "../server/envFile"; // type-only — erased at build, no runtime fs import in client code

export type { GeneralSettings };

export const SETTINGS_QUERY_KEY = ["settings"];

// The route's `ok:true` response carries sibling fields the plain `Envelope<T>` shape doesn't
// have — `webhookConfigured` (Task 2), `credentialsConfigured` (F3) and `envFileState` (MOA-498):
// read-only configured/missing status for NOTIFICATION_WEBHOOK_URL/NOTIFICATION_WEBHOOK_SECRET,
// for the native Claude/Codex/OpenCode GO credentials, and for $JAXOS_HOME/.env itself, never
// part of GeneralSettings (not persisted, not editable). `ok:false` stays the plain `Envelope`
// error shape.
export type GeneralSettingsResult =
  | { ok: true; data: GeneralSettings; webhookConfigured: boolean; credentialsConfigured: { claude: boolean; codex: boolean; opencodeGo: boolean }; envFileState: EnvFileState }
  | { ok: false; error: string };

export async function fetchGeneralSettings(): Promise<GeneralSettingsResult> {
  const res = await fetch("/api/settings");
  return res.json();
}

// Mirrors postArchiveAfterDays (MissionSettingsSection.tsx) — exported so a test can assert the
// exact PUT args without a simulated click (this codebase's static-markup convention).
export async function postGeneralSettings(
  patch: Partial<GeneralSettings>, fetchImpl: typeof fetch = fetch,
): Promise<GeneralSettingsResult> {
  return fetchImpl("/api/settings", {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(patch),
  }).then((r) => r.json()).catch((e) => ({ ok: false as const, error: String(e) }));
}

// F2 (cold review e52d9e6dc555): the ONE vault-usability check every UI consumer shares — a flag
// on with no configured path is not usable, same rule the server's gateVaultRoot/gateVaultScope
// already enforce per-request.
export function vaultUsable(settings: GeneralSettingsResult | undefined): boolean {
  return settings?.ok === true && settings.data.integrations.vault === true && settings.data.vaultPath !== null;
}

// One shared hook for every consumer (Geral itself and Part 2's per-integration gates): React
// Query dedupes the underlying GET by queryKey however many components call it.
export function useGeneralSettings() {
  return useQuery<GeneralSettingsResult>({
    queryKey: SETTINGS_QUERY_KEY, queryFn: fetchGeneralSettings, refetchInterval: 10_000,
  });
}

export type Agents = GeneralSettings["integrations"]["agents"];
const ALL_AGENTS_ON: Agents = { claude: true, codex: true, opencode: true };

// MOA-504 D10: every client gate fails OPEN. While the settings query is loading, not ok, or the
// payload is not a settings payload at all, nothing is hidden: never hide configuration on unknown.
export function agentsOf(result: GeneralSettingsResult | undefined): Agents {
  return (result?.ok === true ? result.data?.integrations?.agents : undefined) ?? ALL_AGENTS_ON;
}
