import type { GeneralSettings, SettingsResult } from "../server/settings";

// The ONE app-wide banner's label derivation (spec: "ONE app-wide warning banner"):
// settings-malformed|settings-unreadable -> a translation key; valid/absent -> no banner.
export function bannerLabelKey(result: SettingsResult): "settings.malformed" | "settings.unreadable" | null {
  if (result.ok) return null;
  return result.error === "settings-malformed" ? "settings.malformed" : "settings.unreadable";
}

// MOA-504 D7: "none" = no agent on; "no-reviewer" = only OpenCode on (jaxflow review needs claude or
// codex; builds still work). Unknown / not ok / no agents -> null (the existing banner covers errors).
export function agentsBannerKind(
  result: { ok: boolean; data?: { integrations?: { agents?: GeneralSettings["integrations"]["agents"] } } } | undefined,
): "none" | "no-reviewer" | null {
  const a = result?.ok ? result.data?.integrations?.agents : undefined;
  if (!a) return null;
  if (!a.claude && !a.codex && !a.opencode) return "none";
  if (!a.claude && !a.codex) return "no-reviewer";
  return null;
}
