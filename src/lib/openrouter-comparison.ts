import type { ConnectionChoice, Percentile, PercentileKey, Routing } from "@/lib/agent-settings";
import { configuredModels } from "@/lib/agent-settings";
import type { EndpointRow, EndpointWindow } from "@/server/collectors/openrouter";

export type Match = "yes" | "no" | "unknown";
export type EndpointEvaluation = { eligible: Match; goals: Match | "none" };
export type ComparisonRequest = { connection: string; model: string };

const KEYS: PercentileKey[] = ["p50", "p75", "p90", "p99"];

export function combine(checks: Match[]): Match {
  return checks.includes("no") ? "no" : checks.includes("unknown") ? "unknown" : "yes";
}

export function anyOf(checks: Match[]): Match {
  return checks.includes("yes") ? "yes" : checks.includes("unknown") ? "unknown" : "no";
}

function invert(value: Match): Match {
  return value === "yes" ? "no" : value === "no" ? "yes" : "unknown";
}

export function ceiling(price: number | null, max: number | undefined): Match {
  return max === undefined ? "yes" : price === null ? "unknown" : price <= max ? "yes" : "no";
}

const SERVICE_TIERS = new Set(["fast", "flex"]);

export function slugMatches(selector: string, tag: string | null): Match {
  if (tag === null) return "unknown";
  if (selector.includes("/")) return selector === tag ? "yes" : "no";
  if (selector === tag) return "yes";
  if (tag.startsWith(`${selector}/`)) {
    const suffix = tag.slice(selector.length + 1).split("/")[0];
    return SERVICE_TIERS.has(suffix) ? "no" : "yes";
  }
  return "no";
}

function goalCheck(
  goal: Percentile | undefined,
  observation: EndpointWindow | null,
  kind: "min" | "max",
): Match[] {
  if (!goal) return [];
  const checks: Match[] = [];
  for (const key of KEYS) {
    const required = goal[key];
    if (typeof required !== "number") continue;
    const observed = observation ? observation[key] : undefined;
    if (typeof observed !== "number" || !Number.isFinite(observed)) {
      checks.push("unknown");
      continue;
    }
    checks.push(kind === "min" ? (observed >= required ? "yes" : "no") : observed <= required ? "yes" : "no");
  }
  return checks;
}

function quantizationMatch(row: EndpointRow, quantizations: string[] | undefined): Match {
  if (!quantizations || !quantizations.length) return "yes";
  if (row.quantization === null) return "unknown";
  return quantizations.includes(row.quantization) ? "yes" : "no";
}

export function evaluateEndpoint(row: EndpointRow, routing: Routing): EndpointEvaluation {
  const only = routing.only ?? [];
  const ignore = routing.ignore ?? [];
  const eligibleChecks: Match[] = [
    only.length ? anyOf(only.map((selector) => slugMatches(selector, row.tag))) : "yes",
    ignore.length ? combine(ignore.map((selector) => invert(slugMatches(selector, row.tag)))) : "yes",
    ceiling(row.prices.prompt, routing.max_price?.prompt),
    ceiling(row.prices.completion, routing.max_price?.completion),
    quantizationMatch(row, routing.quantizations),
  ];
  const goalChecks: Match[] = [
    ...goalCheck(routing.preferred_min_throughput, row.throughput, "min"),
    ...goalCheck(routing.preferred_max_latency, row.latency, "max"),
  ];
  return {
    eligible: combine(eligibleChecks),
    goals: goalChecks.length ? combine(goalChecks) : "none",
  };
}

export function singlePercentileKey(goal: Percentile | undefined | null): PercentileKey | null {
  if (!goal) return null;
  const keys = KEYS.filter((key) => typeof goal[key] === "number");
  return keys.length === 1 ? keys[0] : null;
}

export function comparisonRequestMatches(
  request: ComparisonRequest,
  current: ComparisonRequest,
): boolean {
  return request.connection === current.connection && request.model === current.model;
}

export function canCompare(connection: ConnectionChoice | undefined, model: string): boolean {
  if (!connection || connection.adapter !== "openrouter" || !model) return false;
  return configuredModels(connection).some((row) => row.id === model);
}

export function comparisonUrl(connection: string, model: string, refresh: boolean): string {
  return (
    "/api/opencode-providers/endpoints?connection=" +
    encodeURIComponent(connection) +
    "&model=" +
    encodeURIComponent(model) +
    (refresh ? "&refresh=1" : "")
  );
}
