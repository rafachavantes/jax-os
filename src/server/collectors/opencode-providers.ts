import { isIP } from "node:net";
import { applyAgentSettings, previewAgentSettings, snapshotAgentSettings, type BindingRef, type HelperRunner } from "./agent-settings";
import { CONNECTION_ADAPTERS, type ConnectionChoice, type ModelChoice } from "../../lib/agent-settings";

const CONN_KEYS = [
  "id",
  "label",
  "adapter",
  "base_url",
  "auth",
  "health",
  "credential",
  "editable",
  "reason",
  "used_by",
  "models",
] as const;
const MODEL_KEYS = ["id", "label", "origin", "efforts", "no_effort", "compatible", "reason", "context", "effort_template"] as const;
const CONNECTION_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const ADAPTERS = new Set<string>(CONNECTION_ADAPTERS);

export type ProviderConnectionDraft = { id: string; adapter: string; label?: string; base_url?: string };

function isObj(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === "object" && !Array.isArray(value);
}

function keysOk(obj: Record<string, unknown>, required: string[], optional: string[] = []): boolean {
  const allow = new Set([...required, ...optional]);
  for (const key of Object.keys(obj)) if (!allow.has(key)) return false;
  return required.every((key) => Object.hasOwn(obj, key));
}

function blockedHost(host: string): boolean {
  const name = host.toLowerCase().replace(/\.$/, "");
  if (!name || name === "localhost" || name.endsWith(".localhost")) return true;
  const v = isIP(name);
  if (v === 4) {
    const [a, b] = name.split(".").map(Number);
    return a === 0 || a === 10 || a === 127 || a >= 224
      || (a === 169 && b === 254)
      || (a === 192 && b === 168)
      || (a === 172 && b >= 16 && b <= 31);
  }
  if (v === 6) {
    const n = name.toLowerCase();
    return n === "::" || n === "::1" || n.startsWith("fe80:") || n.startsWith("fc") || n.startsWith("fd") || n.startsWith("ff");
  }
  return false;
}

function requireHttpsUrl(url: unknown): url is string {
  if (typeof url !== "string" || /[\u0000-\u001f]/.test(url)) return false;
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return false;
  }
  if (parsed.protocol !== "https:") return false;
  if (parsed.username || parsed.password) return false;
  if (parsed.search || parsed.hash) return false;
  return !blockedHost(parsed.hostname);
}

// The single connection-definition parser shared by the provider and credential
// routes; node:net and these URL/address guards stay server-only.
export function parseProviderConnection(value: unknown): ProviderConnectionDraft | null {
  if (!isObj(value) || !keysOk(value, ["id", "adapter"], ["label", "base_url"])) return null;
  if (typeof value.id !== "string" || !CONNECTION_RE.test(value.id)) return null;
  if (typeof value.adapter !== "string" || !ADAPTERS.has(value.adapter)) return null;
  if (Object.hasOwn(value, "label") && typeof value.label !== "string") return null;
  if (Object.hasOwn(value, "base_url") && value.base_url != null && !requireHttpsUrl(value.base_url)) return null;
  return value as ProviderConnectionDraft;
}

function pick<T extends string>(row: Record<string, unknown>, keys: readonly T[]): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of keys) {
    if (key in row) out[key] = row[key];
  }
  return out;
}

// Decision for the pre-registration credential read: an existing connection's
// real auth/safety controls always win, and only a genuinely new, valid draft
// may read before it is registered.
export function eligibleSecretReadAllowed(
  connection: { editable: boolean; auth: string } | undefined,
  providerId: string,
  draft: ProviderConnectionDraft | null,
): boolean {
  if (connection) return connection.editable && connection.auth !== "oauth" && connection.auth !== "unknown";
  return draft != null && draft.id === providerId;
}

export function sanitizeConnections(raw: unknown): ConnectionChoice[] {
  const list = Array.isArray(raw)
    ? raw
    : raw && typeof raw === "object" && Array.isArray((raw as { connections?: unknown }).connections)
      ? (raw as { connections: unknown[] }).connections
      : [];
  const connections: ConnectionChoice[] = [];
  for (const item of list) {
    if (!item || typeof item !== "object") continue;
    const row = item as Record<string, unknown>;
    if (typeof row.id !== "string") continue;
    const models: ModelChoice[] = [];
    if (Array.isArray(row.models)) {
      for (const model of row.models) {
        if (!model || typeof model !== "object") continue;
        const entry = model as Record<string, unknown>;
        if (typeof entry.id !== "string" || entry.id.startsWith("jaxflow-builder-")) continue;
        models.push(pick(entry, MODEL_KEYS) as ModelChoice);
      }
    }
    connections.push({ ...(pick(row, CONN_KEYS) as Omit<ConnectionChoice, "models">), models });
  }
  return connections;
}

export async function snapshotProviders(
  _nativeMetadata?: unknown,
  run?: HelperRunner,
  bindings?: BindingRef[],
) {
  const data = await snapshotAgentSettings(_nativeMetadata, run, bindings);
  const revision = data && typeof data === "object" ? (data as { editor_revision?: unknown }).editor_revision : undefined;
  return { editor_revision: revision, connections: sanitizeConnections(data) };
}

export function previewProviders(
  intent: unknown,
  expected: unknown,
  _nativeMetadata?: unknown,
  run?: HelperRunner,
  bindings?: BindingRef[],
) {
  return previewAgentSettings(intent, expected, _nativeMetadata, run, bindings);
}

export function applyProviders(
  intent: unknown,
  expected: unknown,
  operationId: string,
  _nativeMetadata?: unknown,
  run?: HelperRunner,
  bindings?: BindingRef[],
) {
  return applyAgentSettings(intent, expected, operationId, _nativeMetadata, run, bindings);
}
