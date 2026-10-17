import { readGeneralSettings } from "./settings";

// Settings-derived route gates. They live here (not in settings.ts / collectors/files.ts) because
// route tests replace those modules with partial mocks; this module just reads whatever
// readGeneralSettings the test supplies.
export function vaultUsableNow(): boolean {
  const s = readGeneralSettings();
  return s.ok && s.data.integrations.vault && s.data.vaultPath !== null;
}

export function linearEnabled(): boolean {
  const s = readGeneralSettings();
  return s.ok && s.data.integrations.linear;
}
