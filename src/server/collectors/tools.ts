import { realpathSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join, sep } from "node:path";
import type { InventoryData, InventoryExecutor, InventoryItem } from "./inventory";

// Tools Viewer projection (spec C.1 / §7.1). The native inventory read
// (`getInventoryData`) is the SINGLE production discovery: those items are
// projected here into the existing per-executor plugin/skill/agent arrays,
// alongside the executors/recovery contract the Inventory UI consumes. No
// second filesystem walk or config parser runs here.

export type PluginInfo = {
  name: string;
  marketplace: string | null;
  version: string | null;
  enabled: boolean;
  group: "yours" | "system";
  skills: string[];
};
export type SkillInfo = { name: string; source: "dir" | "registry" | "broken" };
export type AgentInfo = { name: string; description: string | null };
export type AppTools = { plugins: PluginInfo[]; skills: SkillInfo[]; agents: AgentInfo[]; loadingSources?: string[] };
export type AppError = { error: string };
export type ToolsData = { claude: AppTools | AppError; codex: AppTools | AppError; opencode: AppTools | AppError };

// ---- pure metadata rules (still exercised by the aggregation) ----

export type NormalizedSpecifier = { name: string; version: string | null };

// OpenCode specifier normalization (spec §3): absolute paths and git specs have
// no version; split at the LAST @ so a scoped package's leading @ survives.
export function normalizeSpecifier(spec: string): NormalizedSpecifier {
  if (spec.startsWith("/")) {
    const name = spec.split("/").filter(Boolean).pop() ?? spec;
    return { name, version: null };
  }
  const gitAt = spec.indexOf("@git+");
  if (gitAt > 0) return { name: spec.slice(0, gitAt), version: null };
  const at = spec.lastIndexOf("@");
  if (at > 0) return { name: spec.slice(0, at), version: spec.slice(at + 1) };
  return { name: spec, version: null };
}

// Local plugin files (~/.config/opencode/plugin/*.{js,ts,tsx}) are named by
// basename without extension (spec §3).
export function localPluginName(filename: string): string {
  return filename.replace(/\.(js|ts|tsx)$/, "");
}

// D2 (spec decision table): group = "system" iff the marketplace suffix
// starts with "openai-"; everything else is "yours".
export function codexGroup(marketplace: string | null): "yours" | "system" {
  return marketplace !== null && marketplace.startsWith("openai-") ? "system" : "yours";
}

// Split a `name@marketplace` registration; a plain registration has no
// marketplace. Falls back to the whole string when the @ is leading.
function splitRegistration(registration: string): { name: string; marketplace: string | null } {
  const at = registration.lastIndexOf("@");
  if (at > 0) return { name: registration.slice(0, at), marketplace: registration.slice(at + 1) };
  return { name: registration, marketplace: null };
}

// The central skill registry (spec §3 Registry), resolved per call. Missing
// registry falls back to the unresolved path — nothing matches it.
function registryRootReal(): string {
  const raw = join(homedir(), ".agents", "skills");
  try {
    return realpathSync(raw);
  } catch {
    return raw;
  }
}

function isUnder(root: string, target: string): boolean {
  return target === root || target.startsWith(root + sep);
}

function resolveLinkTarget(item: InventoryItem): string | null {
  if (!item.canonicalTarget) return null;
  const raw = item.canonicalTarget.startsWith("/") ? item.canonicalTarget : join(dirname(item.path), item.canonicalTarget);
  try {
    return realpathSync(raw);
  } catch {
    return null;
  }
}

function projectSkill(item: InventoryItem, registry: string): SkillInfo {
  if (item.state === "broken") return { name: item.name, source: "broken" };
  if (item.origin === "registration-link") {
    const resolved = resolveLinkTarget(item);
    if (resolved && isUnder(registry, resolved)) return { name: item.name, source: "registry" };
  }
  return { name: item.name, source: "dir" };
}

function projectPlugin(item: InventoryItem, bundled: Map<string, string[]>): PluginInfo {
  const enabled = item.state === "enabled";
  const skills = bundled.get(item.registration) ?? [];
  if (item.executor === "claude") {
    // Configured and cache-only plugins share the `plugin@marketplace` identity.
    const { name, marketplace } = splitRegistration(item.registration);
    return { name, marketplace, version: item.installedVersion ?? null, enabled, group: "yours", skills };
  }
  if (item.executor === "codex") {
    const { name, marketplace } = splitRegistration(item.registration);
    return { name, marketplace, version: null, enabled, group: codexGroup(marketplace), skills };
  }
  // OpenCode: a configured array entry keeps its spec; a local plugin file is
  // named by basename.
  if (item.origin === "local-file") {
    return { name: localPluginName(item.name), marketplace: null, version: null, enabled, group: "yours", skills };
  }
  const spec = normalizeSpecifier(item.registration);
  return { name: spec.name, marketplace: null, version: spec.version, enabled, group: "yours", skills };
}

function projectExecutor(executor: InventoryExecutor, data: InventoryData, registry: string): AppTools | AppError {
  const bucket = data.executors[executor];
  if (!bucket) return { error: "inventory-missing" };
  if (!bucket.ok) return { error: bucket.error };
  const items = bucket.items as InventoryItem[];
  const bundled = new Map<string, string[]>();
  for (const item of items) {
    if (item.kind === "skill" && item.parent) {
      const list = bundled.get(item.parent) ?? [];
      list.push(item.name);
      bundled.set(item.parent, list);
    }
  }
  const plugins: PluginInfo[] = [];
  const skills: SkillInfo[] = [];
  const agents: AgentInfo[] = [];
  for (const item of items) {
    if (item.kind === "plugin") plugins.push(projectPlugin(item, bundled));
    else if (item.kind === "skill") skills.push(projectSkill(item, registry));
    else if (item.kind === "agent") agents.push({ name: item.name, description: item.description || null });
  }
  const projected: AppTools = { plugins, skills, agents };
  const loadingSources = (bucket as { loadingSources?: string[] }).loadingSources;
  if (loadingSources) projected.loadingSources = loadingSources;
  return projected;
}

// Per-app failure isolation (spec §4): a failed native source bucket yields that
// app's AppError; the other two still render. This never throws. The registry
// root is injectable for tests; production resolves the real one.
export function getToolsData(data: InventoryData, registryRoot: string = registryRootReal()): ToolsData {
  return {
    claude: projectExecutor("claude", data, registryRoot),
    codex: projectExecutor("codex", data, registryRoot),
    opencode: projectExecutor("opencode", data, registryRoot),
  };
}

export type { InventoryData };
