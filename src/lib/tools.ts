// Pure view-model derivation for /tools/inventory (spec §7.1). No Node
// built-ins, no VALUE import from src/server/ — imported directly by the
// "use client" section AND by vitest, same convention as src/lib/mission.ts.

import type { InventoryData, InventoryExecutor, InventoryItem, InventoryKind } from "../server/collectors/inventory";

export type AppKey = "claude" | "codex" | "opencode";
export const APP_ORDER: AppKey[] = ["claude", "codex", "opencode"];

// Typed machine enums for display (spec §7 / F9): helper-native English stage
// and effect strings are NEVER rendered verbatim — they map to locale keys.
export const KNOWN_STAGES = [
  "pending",
  "unconfirmed",
  "activation-pending",
  "audit-pending",
  "conflict",
  "applied",
  "not-applied",
] as const;

export function stageKey(stage: string): string {
  return (KNOWN_STAGES as readonly string[]).includes(stage) ? stage : "unknown";
}

export const KNOWN_KINDS = ["plugin", "skill", "agent"] as const;
export const KNOWN_ORIGINS = [
  "settings",
  "cache",
  "independent",
  "config-path",
  "permission-map",
  "skill-override",
  "registration-link",
  "config",
  "config-array",
  "local-file",
  "local-agent",
] as const;

export function labelKey(prefix: "kindLabel" | "originLabel", value: string): string {
  const known = prefix === "kindLabel" ? KNOWN_KINDS : KNOWN_ORIGINS;
  return (known as readonly string[]).includes(value) ? `${prefix}.${value}` : value;
}

export type InventoryFilter = { query: string; executor: "all" | InventoryExecutor; kind: "all" | InventoryKind };

export type InventoryView = {
  rows: InventoryItem[];
  known: number; // rows known before filtering
  filtered: number; // rows after filtering
  partial: boolean; // at least one executor source failed
  anyError: boolean;
  empty: boolean;
};

function inventoryMatches(item: InventoryItem, q: string): boolean {
  if (q === "") return true;
  return (
    item.name.toLowerCase().includes(q) ||
    item.description.toLowerCase().includes(q) ||
    item.origin.toLowerCase().includes(q)
  );
}

// Preserves every executor/source identity: same display name from different
// executors or origins stays a separate row (stable id is the key).
export function deriveInventoryView(data: InventoryData | null, filter: InventoryFilter): InventoryView {
  const q = filter.query.trim().toLowerCase();
  if (!data) return { rows: [], known: 0, filtered: 0, partial: true, anyError: true, empty: q !== "" };
  const all: InventoryItem[] = [];
  let anyError = false;
  for (const executor of APP_ORDER) {
    const bucket = data.executors[executor];
    if (!bucket) continue;
    if (!bucket.ok) {
      anyError = true;
      continue;
    }
    all.push(...bucket.items);
  }
  const rows = all.filter(
    (item) =>
      (filter.executor === "all" || item.executor === filter.executor) &&
      (filter.kind === "all" || item.kind === filter.kind) &&
      inventoryMatches(item, q),
  );
  return { rows, known: all.length, filtered: rows.length, partial: anyError, anyError, empty: q !== "" && rows.length === 0 };
}
