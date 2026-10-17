import { createHash } from "node:crypto";
import type { Percentile } from "../../lib/agent-settings";
import { readCredential, type ReadCredentialBinding } from "./agent-settings";
import { snapshotProviders } from "./opencode-providers";

const ORIGIN = "https://openrouter.ai";
const CAP = 2 * 1024 * 1024;
const FRESH_MS = 60_000;
const TIMEOUT_MS = 10_000;
const CONNECTION_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/;
const PERCENTILES = ["p50", "p75", "p90", "p99"] as const;

export type OpenRouterCredential = { token: string; identity: string; revision: string };
export type EndpointPrices = { prompt: number | null; completion: number | null };
export type EndpointWindow = Percentile & { window: "30m" };
export type EndpointRow = {
  tag: string | null;
  provider: string | null;
  prices: EndpointPrices;
  latency: EndpointWindow | null;
  throughput: EndpointWindow | null;
  supported_parameters: string[] | null;
  partial: boolean;
  eligible: boolean;
  quantization: string | null;
};
export type EndpointComparison = {
  connection: string;
  model: string;
  fetched_at: string;
  stale: boolean;
  error: string | null;
  endpoints: EndpointRow[];
};
export type CompareEndpointsInput = {
  connection: string;
  model: string;
  models: readonly string[];
  credential: OpenRouterCredential;
  refresh?: boolean;
};
export type CompareDeps = { fetch?: typeof fetch; now?: () => number };

export type CompareSnapshotConnection = {
  id: string;
  adapter: string | null;
  auth: "api-key" | "oauth" | "none" | "unknown";
  credential: { kind: "native" } | { kind: "env"; env: string } | null;
  models: Array<{ id: string }>;
};
export type CompareComposition = {
  snapshot: () => Promise<{ connections: CompareSnapshotConnection[]; editor_revision?: unknown }>;
  read: (binding: ReadCredentialBinding) => Promise<string>;
  compare: (input: CompareEndpointsInput) => Promise<unknown>;
};

function sourceRevisionOf(snap: { editor_revision?: unknown }): string {
  if (!snap.editor_revision || typeof snap.editor_revision !== "object") return "absent";
  const sources = (snap.editor_revision as { sources?: unknown }).sources;
  if (!sources || typeof sources !== "object") return "absent";
  const opencode = (sources as Record<string, unknown>)["opencode.json"];
  return typeof opencode === "string" && opencode !== "absent" ? opencode : "absent";
}

export async function composeCompare(
  c: CompareComposition,
  connection: string,
  model: string,
  refresh: boolean,
): Promise<unknown> {
  const snap = await c.snapshot();
  const conn = snap.connections.find((row) => row.id === connection);
  if (!conn || conn.adapter !== "openrouter" || conn.auth === "oauth") throw unavailable();
  if (!conn.models.some((row) => row.id === model)) throw unavailable();
  const models = conn.models.map((row) => row.id);
  const cred = conn.credential;
  let token: string;
  let identity: string;
  let revision: string;
  if (cred?.kind === "env") {
    token = await c.read(cred);
    identity = cred.env;
    revision = createHash("sha256").update(token).digest("hex");
  } else if (cred?.kind === "native") {
    token = await c.read({ kind: "native", connection });
    identity = "native";
    revision = sourceRevisionOf(snap);
  } else {
    throw unavailable();
  }
  return c.compare({
    connection,
    model,
    models,
    credential: { token, identity, revision },
    refresh,
  });
}

export function productionCompareComposition(): CompareComposition {
  return {
    snapshot: () => snapshotProviders({}),
    read: (binding) => readCredential(binding),
    compare: (input) => compareEndpoints(input),
  };
}

type Fresh = { at: number; data: EndpointComparison };

const fresh = new Map<string, Fresh>();
const lastGood = new Map<string, EndpointComparison>();
const inflight = new Map<string, Promise<EndpointComparison>>();

function unavailable(): Error {
  return new Error("unavailable");
}

function cacheKey(input: CompareEndpointsInput): string {
  return `${input.connection}\n${input.model}\n${input.credential.identity}\n${input.credential.revision}`;
}

export function invalidateOpenRouterCache(connection?: string): void {
  if (!connection) {
    fresh.clear();
    lastGood.clear();
    inflight.clear();
    return;
  }
  const prefix = `${connection}\n`;
  for (const map of [fresh, lastGood, inflight]) {
    for (const key of map.keys()) if (key.startsWith(prefix)) map.delete(key);
  }
}

function splitModel(model: string): { author: string; slug: string } {
  const i = model.indexOf("/");
  if (i <= 0 || i === model.length - 1) throw unavailable();
  const author = model.slice(0, i);
  const slug = model.slice(i + 1);
  const parts = [author, ...slug.split("/")];
  if (parts.some((p) => !p || p === "." || p === "..")) throw unavailable();
  return { author, slug };
}

function million(raw: unknown): { value: number | null; invalid: boolean } {
  if (raw == null || raw === "") return { value: null, invalid: false };
  const n = typeof raw === "number" ? raw : typeof raw === "string" ? Number(raw) : NaN;
  if (!Number.isFinite(n) || n < 0) return { value: null, invalid: true };
  return { value: n * 1_000_000, invalid: false };
}

function percentiles(raw: unknown, scale = 1): { value: EndpointWindow | null; invalid: boolean } {
  if (raw == null) return { value: null, invalid: false };
  if (typeof raw !== "object" || Array.isArray(raw)) return { value: null, invalid: true };
  const src = raw as Record<string, unknown>;
  const out: Percentile = {};
  for (const key of PERCENTILES) {
    const v = src[key];
    if (typeof v === "number" && Number.isFinite(v)) out[key] = v / scale;
  }
  if (Object.keys(out).length) return { value: { ...out, window: "30m" }, invalid: false };
  return { value: null, invalid: PERCENTILES.some((key) => key in src) };
}

function params(raw: unknown): { value: string[] | null; invalid: boolean } {
  if (raw == null) return { value: null, invalid: false };
  if (!Array.isArray(raw) || raw.some((item) => typeof item !== "string")) return { value: null, invalid: true };
  return { value: [...raw], invalid: false };
}

function tagOf(raw: unknown): string | null {
  if (typeof raw !== "string") return null;
  const tag = raw.trim();
  return tag ? tag : null;
}

function projectRow(raw: unknown): EndpointRow | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const o = raw as Record<string, unknown>;
  const pricing = o.pricing && typeof o.pricing === "object" && !Array.isArray(o.pricing)
    ? o.pricing as Record<string, unknown>
    : null;
  const prompt = million(pricing?.prompt);
  const completion = million(pricing?.completion);
  // OpenRouter reports latency in milliseconds; routing goals and the UI use seconds.
  const latency = percentiles(o.latency_last_30m, 1000);
  const throughput = percentiles(o.throughput_last_30m);
  const supported = params(o.supported_parameters);
  const tag = tagOf(o.tag);
  const provider = typeof o.provider_name === "string" && o.provider_name ? o.provider_name : null;
  const quantization = tagOf(o.quantization);
  return {
    tag,
    provider,
    prices: { prompt: prompt.value, completion: completion.value },
    latency: latency.value,
    throughput: throughput.value,
    supported_parameters: supported.value,
    partial: prompt.invalid || completion.invalid || latency.invalid || throughput.invalid || supported.invalid,
    eligible: tag != null,
    quantization,
  };
}

function project(json: unknown, connection: string, model: string, at: number): EndpointComparison {
  if (!json || typeof json !== "object") throw unavailable();
  const data = (json as { data?: unknown }).data;
  if (!data || typeof data !== "object") throw unavailable();
  const endpoints = (data as { endpoints?: unknown }).endpoints;
  if (!Array.isArray(endpoints)) throw unavailable();
  const seen = new Set<string>();
  const rows: EndpointRow[] = [];
  for (const item of endpoints) {
    const row = projectRow(item);
    if (!row) continue;
    if (row.tag) {
      if (seen.has(row.tag)) continue;
      seen.add(row.tag);
    }
    rows.push(row);
  }
  return {
    connection,
    model,
    fetched_at: new Date(at).toISOString(),
    stale: false,
    error: null,
    endpoints: rows,
  };
}

async function readCapped(res: Response): Promise<string> {
  const declared = Number(res.headers.get("content-length"));
  if (Number.isFinite(declared) && declared > CAP) throw unavailable();
  const reader = res.body?.getReader();
  if (!reader) return "";
  const chunks: Uint8Array[] = [];
  let seen = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      seen += value.byteLength;
      if (seen > CAP) {
        await reader.cancel().catch(() => {});
        throw unavailable();
      }
      chunks.push(value);
    }
  } catch (e) {
    if (e instanceof Error && e.message === "unavailable") throw e;
    throw unavailable();
  } finally {
    reader.releaseLock();
  }
  const bytes = new Uint8Array(seen);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    throw unavailable();
  }
}

async function fetchComparison(
  input: CompareEndpointsInput,
  at: number,
  fetchFn: typeof fetch,
): Promise<EndpointComparison> {
  const { author, slug } = splitModel(input.model);
  const url = `${ORIGIN}/api/v1/models/${encodeURIComponent(author)}/${encodeURIComponent(slug)}/endpoints`;
  let res: Response;
  try {
    res = await fetchFn(url, {
      headers: { Authorization: `Bearer ${input.credential.token}`, Accept: "application/json" },
      redirect: "error",
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
  } catch {
    throw unavailable();
  }
  if (!res.ok) throw unavailable();
  let parsed: unknown;
  try {
    parsed = JSON.parse(await readCapped(res));
  } catch (e) {
    if (e instanceof Error && e.message === "unavailable") throw e;
    throw unavailable();
  }
  return project(parsed, input.connection, input.model, at);
}

export async function compareEndpoints(
  input: CompareEndpointsInput,
  deps: CompareDeps = {},
): Promise<EndpointComparison> {
  if (!CONNECTION_RE.test(input.connection) || !input.credential.token) throw unavailable();
  splitModel(input.model);
  if (!input.models.includes(input.model)) throw unavailable();
  const key = cacheKey(input);
  const now = deps.now?.() ?? Date.now();
  if (!input.refresh) {
    const hit = fresh.get(key);
    if (hit && now - hit.at < FRESH_MS) return hit.data;
  }
  const pending = inflight.get(key);
  if (pending) return pending;
  const fetchFn = deps.fetch ?? fetch;
  const run = (async () => {
    try {
      const data = await fetchComparison(input, now, fetchFn);
      fresh.set(key, { at: now, data });
      lastGood.set(key, data);
      return data;
    } catch {
      const good = lastGood.get(key);
      if (good) return { ...good, stale: true, error: "unavailable" };
      throw unavailable();
    } finally {
      inflight.delete(key);
    }
  })();
  inflight.set(key, run);
  return run;
}
